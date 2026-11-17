"""The edge gate and the live sleeve runtime must never flatter a result.

These tests exist because the failure mode of "statistically certified
edge" work is not a crash — it is a plausible-looking number produced by
a fill assumption, a fold boundary, or a resampling bug. Each test below
pins one of those honest-structure guarantees.
"""
import json
import math
import os
import random
import time

import pytest

from app import sleeves as sleeve_mod
from app.edge_gate import (
    STATUS_CLOSED,
    STATUS_OPEN,
    STATUS_SHADOW,
    Trade,
    fold_stats,
    profit_factor,
    sleeve_gate,
    summarise,
    t_stat_block_bootstrap,
    t_stat_iid,
)


# --------------------------------------------------------------- helpers
def _trades(pnls, fold=0, start=1_600_000_000, step=3600):
    return [
        Trade(entry_t=start + i * step, exit_t=start + i * step + 3600,
              fold=fold, pnl_bps=p)
        for i, p in enumerate(pnls)
    ]


def _good_trades(n_per_fold=40, folds=5, mean=40.0, sd=10.0, seed=1):
    rng = random.Random(seed)
    out = []
    for f in range(folds):
        for _ in range(n_per_fold):
            out.append(
                Trade(
                    entry_t=1_600_000_000 + f * 10_000_000,
                    exit_t=1_600_000_000 + f * 10_000_000 + 3600,
                    fold=f,
                    pnl_bps=rng.gauss(mean, sd),
                )
            )
    return out


# --------------------------------------------------------------- statistics
def test_profit_factor_and_iid_t_are_textbook():
    p = [10.0, -5.0, 10.0, -5.0]
    assert profit_factor(p) == pytest.approx(2.0)
    assert t_stat_iid([1.0, 1.0, 1.0]) == 0.0        # zero variance -> no claim
    assert t_stat_iid([1.0]) == 0.0                   # too few samples
    assert profit_factor([1.0, 2.0]) == float("inf")  # no losses


def test_block_bootstrap_never_flatters_autocorrelated_trades():
    """With positive autocorrelation the block t must be the conservative one.

    An iid t on clustered trades overstates significance; if the bootstrap
    ever agrees with the optimistic number, it is not doing its job.
    """
    rng = random.Random(7)
    w = 0.0
    pnls = []
    for _ in range(400):
        w = 0.8 * w + rng.gauss(0, 20)
        pnls.append(20 + w)
    t_iid = t_stat_iid(pnls)
    t_bb = t_stat_block_bootstrap(pnls, block=20, iters=400, seed=3)
    assert t_bb < t_iid


def test_block_bootstrap_needs_enough_trades():
    assert t_stat_block_bootstrap([1.0, 2.0, 3.0], block=20) == 0.0


def test_fold_stats_group_by_fold():
    trades = _trades([10.0, -10.0], fold=0) + _trades([5.0, 5.0], fold=1)
    fs = fold_stats(trades)
    assert [f.fold for f in fs] == [0, 1]
    assert fs[0].net_ev_bps == pytest.approx(0.0)
    assert fs[1].net_ev_bps == pytest.approx(5.0)


# --------------------------------------------------------------- gate
def test_gate_promotes_only_when_every_bar_is_cleared():
    holdout = _trades([30.0] * 60, fold=99)
    r = sleeve_gate("s", _good_trades(), asset="ETH", family="f",
                    holdout_trades=holdout, block=20, bootstrap_iters=300)
    assert r.status == STATUS_OPEN
    assert r.n > 0 and r.profit_factor > 1.25
    assert r.positive_folds >= 3


def test_gate_refuses_negative_expectancy():
    bad = _trades([-20.0] * 200, fold=0)
    r = sleeve_gate("s", bad, holdout_trades=_trades([10.0] * 60), block=20, bootstrap_iters=200)
    assert r.status == STATUS_CLOSED
    assert any("Net EV" in x for x in r.reasons)


def test_gate_refuses_a_negative_locked_holdout():
    """Good folds cannot buy a bad holdout — that is the anti-cherry-pick rule."""
    r = sleeve_gate(
        "s", _good_trades(), asset="ETH", family="f",
        holdout_trades=_trades([-5.0] * 60, fold=99),
        block=20, bootstrap_iters=300,
    )
    assert r.status == STATUS_SHADOW
    assert any("holdout" in x for x in r.reasons)


def test_gate_refuses_a_holdout_that_is_too_small_to_mean_anything():
    r = sleeve_gate(
        "s", _good_trades(), asset="ETH", family="f",
        holdout_trades=_trades([50.0] * 5, fold=99),
        block=20, bootstrap_iters=300,
    )
    assert r.status == STATUS_SHADOW
    assert any("holdout n" in x for x in r.reasons)


def test_gate_refuses_minority_of_positive_folds():
    # one huge winning fold cannot carry four losing folds
    pnls = ([900.0] * 10) + ([-30.0] * 10 * 4)
    trades = []
    for i, p in enumerate(pnls):
        trades.append(Trade(entry_t=1_600_000_000 + i, exit_t=1_600_000_001,
                            fold=i // 10, pnl_bps=p))
    r = sleeve_gate("s", trades, holdout_trades=_trades([10.0] * 60),
                    block=20, bootstrap_iters=200)
    assert r.status == STATUS_SHADOW
    assert any("positive folds" in x for x in r.reasons)


def test_gate_refuses_a_thin_sample_even_when_it_looks_great():
    r = sleeve_gate("s", _trades([50.0] * 20), holdout_trades=_trades([50.0] * 60),
                    block=5, bootstrap_iters=200)
    assert r.status == STATUS_SHADOW
    assert any("n 20" in x for x in r.reasons)


def test_gate_never_opens_on_zero_trades():
    r = sleeve_gate("s", [], holdout_trades=[])
    assert r.status == STATUS_CLOSED
    assert r.reasons == ["no trades"]


def test_gate_reports_both_t_statistics_and_uses_the_conservative_one():
    r = sleeve_gate("s", _good_trades(), holdout_trades=_trades([30.0] * 60),
                    block=20, bootstrap_iters=400)
    assert r.t_iid != 0.0
    assert r.t_effective == min((r.t_iid, r.t_block_bootstrap), key=abs)


def test_summarise_counts_and_coverage():
    a = sleeve_gate("a", _good_trades(), asset="ETH", family="f",
                    holdout_trades=_trades([30.0] * 60), block=20, bootstrap_iters=200)
    b = sleeve_gate("b", _trades([-10.0] * 200), asset="BTC", family="g",
                    holdout_trades=_trades([10.0] * 60), block=20, bootstrap_iters=200)
    s = summarise([a, b])
    assert s["total"] == 2
    assert s["open"] + s["shadow"] + s["closed"] == 2
    assert "ETH" in s["covered_assets"]


# --------------------------------------------------------------- runtime
class _St:
    def __init__(self, times, closes):
        self.times = times
        self.closes = closes


class _Feed:
    def __init__(self, mapping):
        self._m = mapping

    def state(self, symbol):
        return self._m[symbol]


def _grid(n15=900, seed=5):
    """15m closes, upsampled to 1m bars the way the live feed delivers them."""
    rng = random.Random(seed)
    now = int(time.time())
    closes = [100.0]
    for _ in range(n15 - 1):
        closes.append(closes[-1] * math.exp(rng.gauss(0, 0.004)))
    times15 = [now - (n15 - i) * 900 for i in range(n15)]
    t1, c1 = [], []
    for t, c in zip(times15, closes):
        for m in range(15):
            t1.append(t + m * 60)
            c1.append(c)
    return t1, c1


def _engine():
    """Minimal stand-in for PredictionEngine: just the feed + sleeve cache.

    A full PredictionEngine is deliberately NOT constructed here — it
    touches shared brain/flywire caches on disk and would leak state into
    unrelated tests that run later in the same session.
    """
    from types import SimpleNamespace

    from app.engine import PredictionEngine

    t, c = _grid()
    stub = SimpleNamespace(
        feed=_Feed({"BTC": _St(t, c), "ETH": _St(t, c)}),
        _sleeve_cache=(0.0, None),
    )
    stub.sleeve_block = PredictionEngine.sleeve_block.__get__(stub)
    return stub


def _registry(status="open"):
    return {
        "generated_at": 1,
        "summary": {},
        "sleeves": [
            {"sleeve_id": "catchup_subhour_spot_15m::ETH", "asset": "ETH",
             "status": status, "net_ev_bps": 88.2, "profit_factor": 1.48,
             "t_effective": 2.47, "n": 297, "reasons": []},
            {"sleeve_id": "catchup_subhour_spot_30m::ETH", "asset": "BNB",
             "status": "shadow", "net_ev_bps": 63.5, "profit_factor": 1.33,
             "t_effective": 1.86, "n": 332, "reasons": ["|t| 1.86 < 2.0"]},
        ],
    }


def test_resample_drops_the_partial_trailing_bar():
    t = [0, 60, 900, 960, 1800]          # last bucket 1800 is incomplete
    c = [1.0, 2.0, 3.0, 4.0, 5.0]
    bt, bc = sleeve_mod.resample_bars(t, c, 900)
    assert bt == [0, 900]
    assert bc == [2.0, 4.0]


def test_impulse_z_is_none_while_warming_up():
    t, c = _grid(n15=50)
    assert sleeve_mod.compute_impulse_z(t, c) is None


def test_impulse_z_matches_the_certified_formula():
    t, c = _grid(n15=900, seed=11)
    z = sleeve_mod.compute_impulse_z(t, c)
    assert z is not None
    # recompute by hand on the same 15m grid
    bt, bc = sleeve_mod.resample_bars(t, c, 900)
    bt, bc = bt[:-1], bc[:-1]
    rets = [0.0] + [math.log(bc[i] / bc[i - 1]) for i in range(1, len(bc))]
    win = rets[-500:]
    m = sum(win) / len(win)
    sd = math.sqrt(sum((r - m) ** 2 for r in win) / len(win))
    expect = sum(rets[-48:]) / (sd * math.sqrt(48))
    assert z == pytest.approx(expect)


def test_only_open_sleeves_may_emit_a_state():
    t, c = _grid()
    feed = _Feed({"BTC": _St(t, c), "ETH": _St(t, c), "BNB": _St(t, c)})
    out = sleeve_mod.evaluate_sleeves(feed, registry=_registry(), now=t[-1])
    assert out["certified"] == ["ETH"]
    assert [s["asset"] for s in out["signals"]] == ["ETH"]
    assert out["assets"]["BNB"]["status"] == "shadow"
    assert out["assets"]["BNB"]["state"] == "unproven"


def test_a_shadow_sleeve_emits_no_signal_even_when_its_unproven():
    t, c = _grid()
    feed = _Feed({"BTC": _St(t, c), "ETH": _St(t, c), "BNB": _St(t, c)})
    out = sleeve_mod.evaluate_sleeves(feed, registry=_registry(status="shadow"), now=t[-1])
    assert out["signals"] == []
    assert out["certified"] == []


def test_missing_registry_is_silent_not_fatal(tmp_path):
    reg = sleeve_mod.load_registry(str(tmp_path / "nope.json"))
    assert reg["sleeves"] == []
    t, c = _grid()
    feed = _Feed({"BTC": _St(t, c), "ETH": _St(t, c)})
    out = sleeve_mod.evaluate_sleeves(feed, registry=reg, now=t[-1])
    assert out["signals"] == []


def test_engine_sleeve_block_covers_every_ui_asset():
    """All four UI assets must render a sleeve row, even with nothing certified."""
    from app.config import SYMBOL_ORDER

    eng = _engine()
    block = eng.sleeve_block(now=1_700_000_000.0)
    assert set(SYMBOL_ORDER).issubset(set(block["assets"]))
    for symbol in SYMBOL_ORDER:
        assert block["assets"][symbol]["status"] in ("open", "shadow", "none")
    # nothing certified for an asset must not be dressed up as a signal
    for symbol, row in block["assets"].items():
        if block["assets"][symbol]["status"] == "none":
            assert row["state"] == "none"


def test_engine_sleeve_block_is_cached_between_snapshots():
    eng = _engine()
    a = eng.sleeve_block(now=1_700_000_000.0)
    b = eng.sleeve_block(now=1_700_000_001.0)
    assert a is b          # same object: the 15m grid cannot change in 1s


def test_committed_registry_is_consistent_with_its_own_summary():
    reg = sleeve_mod.load_registry()
    if not reg.get("sleeves"):
        pytest.skip("registry not generated in this checkout")
    open_ids = [s["sleeve_id"] for s in reg["sleeves"] if s["status"] == "open"]
    assert reg["summary"]["open_sleeves"] == open_ids
    for s in reg["sleeves"]:
        if s["status"] == "open":
            assert s["net_ev_bps"] > 0
            assert s["profit_factor"] > 1.25
            assert abs(s["t_effective"]) >= 2.0
