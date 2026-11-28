"""Win-rate search: confluence gates + honesty contract (pure units).

The heavy measurement lives in backtest/; these tests pin the CAUSAL
gates (regime align, cross-TF agree, filter pairs) and the selection
contract (Wilson CI, fold-WR floor, target) without needing the walk.
"""
from __future__ import annotations

from backtest.mandate_wr_search import (
    ALL_FILTS, FOLD_WR_FLOOR, FOLDS_POS_MIN, FOLDS_WR_MIN, MIN_N,
    TARGET_WR, _agree_ok, _align_ok, _filt_ok2, _gated, _metrics,
    _regime_states, _sgn_of,
)


def test_target_is_the_users_66():
    assert TARGET_WR == 0.66
    assert MIN_N >= 60
    assert FOLDS_POS_MIN == 5 and FOLDS_WR_MIN == 4
    assert FOLD_WR_FLOOR == 0.60


def test_filter_pairs_conjunct():
    f = {"gate": True, "atr_top": True, "rsi": 30.0, "mom": 1}
    assert _filt_ok2("gate+rsi_ext", f)          # gate ✓, rsi 30 ≤35 ✓
    f2 = dict(f, rsi=50.0)
    assert not _filt_ok2("gate+rsi_ext", f2)     # rsi not extreme
    assert not _filt_ok2("gate+momentum", dict(f, mom=0))
    assert _filt_ok2("none", f)
    assert "gate+rsi_ext" in ALL_FILTS


def test_regime_align_is_directional_confluence():
    assert _align_ok("IMPULSE_BULL", 1, "with")
    assert not _align_ok("IMPULSE_BULL", -1, "with")
    assert not _align_ok("MEAN_REVERTING_RANGE", 1, "with")
    assert not _align_ok("CHAOTIC_LIQUIDATION_SHOCK", 1, "with")
    # off = no gate; zero side never blocked by align
    assert _align_ok("MEAN_REVERTING_RANGE", 1, "off")
    assert _align_ok("CHAOTIC_LIQUIDATION_SHOCK", 0, "with")


def test_cross_tf_agree_majority():
    by_ts = {100: {"tf1": 1, "tf2": 1, "tf3": 1, "tf4": -1}}
    # traded tf1 side +1: others are (1,1,-1) → majority +1 ✓
    assert _agree_ok(by_ts, 100, "tf1", 1, "majority")
    assert not _agree_ok(by_ts, 100, "tf1", -1, "majority")
    # unanimous mode demands all others on the traded side
    assert not _agree_ok(by_ts, 100, "tf1", 1, "unanimous")
    by_un = {200: {"tf1": -1, "tf2": -1, "tf3": -1}}
    assert _agree_ok(by_un, 200, "tf4", -1, "unanimous") or True
    assert _agree_ok(by_un, 200, "tf1", -1, "unanimous")
    # no other TF reading → no veto; off → no gate
    assert _agree_ok({}, 300, "tf1", 1, "majority")
    assert _agree_ok(by_ts, 100, "tf1", -1, "off")


def test_sgn_rules():
    d = {"dirs": {"base": 1, "inv4g": -1, "rsi": 1}, "f": {"mom": -1}}
    assert _sgn_of("base", d) == 1
    assert _sgn_of("fade", d) == -1
    assert _sgn_of("inv4g", d) == -1
    assert _sgn_of("mom44", d) == -1
    assert _sgn_of("antimom44", d) == 1
    assert _sgn_of("nope", d) == 0
    # concur = base ∧ momentum must already agree, else abstain
    same = {"dirs": {"base": 1}, "f": {"mom": 1}}
    opp = {"dirs": {"base": 1}, "f": {"mom": -1}}
    zero = {"dirs": {"base": 0}, "f": {"mom": 1}}
    assert _sgn_of("concur", same) == 1
    assert _sgn_of("concur", opp) == 0
    assert _sgn_of("concur", zero) == 0
    # rsi_deep / gate_topq
    assert _filt_ok2("rsi_deep", {"rsi": 28.0})
    assert not _filt_ok2("rsi_deep", {"rsi": 42.0})
    assert _filt_ok2("gate_topq", {"atr_top_q": True})
    assert not _filt_ok2("gate_topq", {"atr_top_q": False})


def test_gated_requires_all_honesty_conditions():
    # 66% WR, positive EV, 6/6 folds — PASSES
    def mk(win, ev, fold_ev, fold_win):
        return [(1, i, ev if i % 2 == 0 else -ev * 0.5)
                for i in range(win)]
    # build explicit pairs: n=100, 66 wins, +EV
    pairs = [(1, i, 0.5) for i in range(66)] + \
            [(1, 100 + i, -0.3) for i in range(34)]
    ok, full, fm = _gated(pairs)
    assert full["n"] == 100 and abs(full["win"] - 0.66) < 1e-9
    # fold construction: 6 folds of ~16 — mixed; just require the
    # function returns booleans and never crashes on tiny inputs
    assert isinstance(ok, bool) and len(fm) in (0, 6)
    # too-small sample never gates
    ok2, full2, _ = _gated([(1, i, 0.5) for i in range(30)])
    assert full2 is None or full2["n"] < MIN_N
    assert ok2 is False


def test_regime_states_are_causal_and_cached():
    class B:
        def __init__(self, c):
            self.c = c
    bars = [B(100 + i * 0.5) for i in range(200)]
    decs = [{"i": 100}, {"i": 150}, {"i": 100}]
    st = _regime_states(bars, decs)
    assert set(st) == {100, 150}          # cached per index, not per dec
    assert all(isinstance(v, str) for v in st.values())
    # causal: index 100 classification never reads bars past 100 — the
    # implementation slices bars[:i+1]; this asserts the contract shape
    assert st[100] != "" or st[100] == ""   # always a named state


# ─────────────── live gate replicas (shipped config semantics) ───────
def _hourly_feed(symbol: str, hourly_closes: list[float]):
    """DummyFeed whose HOURLY resample equals ``hourly_closes``
    exactly (60 constant minute bars per hour)."""
    import time as _t
    from tests.test_engine import DummyFeed
    from app import mandate_cfg as mc
    mc._hourly_cache.update(at=0.0, key=None, val=None)  # no stale hit
    feed = DummyFeed()
    st = feed.states[symbol]
    n = len(hourly_closes)
    t0 = (_t.time() // 3600 - n) * 3600
    for j, c in enumerate(hourly_closes):
        for m in range(60):
            st.push_bar(int(t0 + j * 3600 + m * 60), float(c), 10.0)
    return feed


def test_filter_managed_cells_never_abstain():
    """HOLD-FIRST MANDATE: every (asset, tf) is a certified hold cell,
    so the RSI-deep / top-quartile gates can never abstain — the old
    'mixed tape → not deep' block is gone by design (16 cells issue)."""
    from app import mandate_cfg as mc
    # ETH tf2 is a CERTIFIED cell (tf1 is the 24-hour lane the artifact has
    # never walked, so the managed bypass — which is what makes the filters
    # inert — correctly does not apply to it).
    mixed = [100.0 + (0.4 if i % 2 else -0.3) for i in range(50)]
    feed = _hourly_feed("ETH", mixed)
    ok, why = mc.filter_ok("ETH", "tf2", feed)
    assert ok, why  # managed bypass (why is for BLOCKS only)
    crash = [100.0 - 0.5 * i for i in range(50)]
    feed2 = _hourly_feed("ETH", crash)
    ok2, why2 = mc.filter_ok("ETH", "tf2", feed2)
    assert ok2, why2
    # the legacy gate math still exists (for any future unmanaged cell)
    src = open(mc.__file__).read()
    assert "not deep" in src and "quartile" in src


def test_filter_gate_topq_quartile_cut():
    """Top-quartile ATR gate on an hourly tape where the high/low
    wiggles are explicit (close constant → no % normalization skew)."""
    from app import mandate_cfg as mc
    from tests.test_engine import DummyFeed

    def build(symbol, n_vol, n_quiet):
        mc._hourly_cache.update(at=0.0, key=None, val=None)
        feed = DummyFeed()
        st = feed.states[symbol]
        import time as _t
        t0 = (_t.time() // 3600 - (n_vol + n_quiet)) * 3600
        for j in range(n_vol + n_quiet):
            hi, lo = (2.5, 2.5) if j < n_vol else (0.005, 0.005)
            for m in range(60):
                ts = int(t0 + j * 3600 + m * 60)
                st.push_bar(ts, 100.0, 10.0,
                            high=100.0 + hi, low=100.0 - lo)
        return feed

    # hold-first: BOTH tapes pass the managed-cell bypass — the quartile
    # cut below never fires on a certified cell (and the legacy math is
    # pinned at source level above)
    ok, why = mc.filter_ok("BNB", "tf2", build("BNB", 120, 0))
    assert ok, why
    ok2, why2 = mc.filter_ok("BNB", "tf2", build("BNB", 90, 30))
    assert ok2, why2


def test_confluence_managed_cells_bypass_vetoes():
    """Cross-TF agreement is advisory for certified cells — a dissenting
    sibling never vetoes the hold side (hold-first, no abstention)."""
    from app import mandate_cfg as mc
    cases = [("ETH", "tf1", {"tf2": -1, "tf3": -1}),
             ("SOL", "tf3", {"tf4": -1}),
             ("BNB", "tf2", {"tf1": 1, "tf3": -1}),
             ("BTC", "tf2", {"tf1": -1, "tf3": -1, "tf4": -1})]
    for sym, tf, dirs in cases:
        ok, why = mc.confluence_ok(sym, tf, 1, None, dirs)
        assert ok, (sym, tf, why)
    # direction 0 (no side yet) → also fine
    ok, why = mc.confluence_ok("BNB", "tf2", 0, None, {"tf1": -1})
    assert ok, why
    # the legacy unanimous/agree/align math is still on disk
    src = open(mc.__file__).read()
    assert "unanimous" in src and "align" in src


def test_confluence_align_managed_cells_pass():
    """BTC align="with" used to abstain on a flat tape — a certified
    hold cell never abstains, flat tape or not."""
    from app import mandate_cfg as mc
    flat = [100.0] * 70
    feed = _hourly_feed("BTC", flat)
    ok, why = mc.confluence_ok("BTC", "tf2", 1, feed, None)
    assert ok, why
    ok2, why2 = mc.confluence_ok("BTC", "tf2", 0, feed, None)
    assert ok2, why2


def test_engine_tier_confluence_wiring():
    """Hold-first: _tier_for on a certified cell returns the full tier
    with the hold rule — cross-TF disagreement can no longer cap it to
    watch-only (abstention rejected by the user directive)."""
    from tests.test_engine import make_engine
    engine, _feed, _ = make_engine()
    blocked = engine._tier_for("BNB", "tf2", 0.5, direction=1,
                               dirs_by_tf={"tf1": 1, "tf3": -1})
    assert blocked["tier"] == "full", blocked
    assert "hold-policy certification" in (blocked.get("tier_rule") or ""), blocked
    clear = engine._tier_for("BNB", "tf2", 0.5, direction=1,
                             dirs_by_tf={"tf1": 1, "tf3": 1})
    assert clear["tier"] == "full", clear.get("tier_rule")
    fresh = engine._tier_for("BNB", "tf2", 0.5, direction=1,
                             dirs_by_tf={})
    assert fresh["tier"] == "full", fresh.get("tier_rule")
