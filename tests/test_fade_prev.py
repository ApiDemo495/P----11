"""fade_prev — the continuous-cycle direction rule (continuous WR 52–53%).

−sign(move of the previous COMPLETED epoch-aligned grid window
[s−W, s)): both reference closes are strictly older than the current
window's start, so the side is computable at lock time and cannot look
ahead.  Measured at the live horizons (backtest/continuous.py, 6 years
of real bars) it beats the old momentum rules (mom44 ≈ 49%) on every
cell where it was switched in (BTC/ETH/BNB); PAXG keeps mom44 (measured
tie at its only gradeable cell).
"""
from __future__ import annotations

import time

from app import mandate_cfg as mc
from tests.test_engine import DummyFeed


def _feed_with(symbol: str, prev_move: float, cur_bars: int = 10):
    """Minute tape: one completed window with `prev_move` drift, then
    `cur_bars` minutes of the CURRENT window. Returns (feed, s)."""
    W = 44 * 60                       # tf2 — BTC/ETH/BNB cells
    now = int(time.time())
    s = now // W * W                  # current window start
    feed = DummyFeed()
    st = feed.states[symbol]
    # bar strictly before the previous window: base close 100.0
    base = s - W - 60
    st.push_bar(base, 100.0, 1.0)
    # previous window [s-W, s): drift prev_move across the window
    n_prev = 40
    for j in range(n_prev):
        t = s - W + 60 * (j + 1)      # last one lands at s-60 < s
        c = 100.0 + prev_move * (j + 1) / n_prev
        st.push_bar(t, c, 1.0)
    # current window so far (after s) — must NOT influence the side
    for j in range(cur_bars):
        st.push_bar(s + 60 * j, 500.0 if j % 2 else 1.0, 1.0)
    return feed, s


def test_fade_prev_sign_follows_previous_window():
    from app import mandate_cfg as _mc
    up_feed, _ = _feed_with("BTC", +1.0)     # prev window rose → fade → -1
    assert _mc._fade_prev_side(up_feed, "BTC", "tf2") == -1
    dn_feed, _ = _feed_with("BTC", -1.0)     # prev window fell → fade → +1
    assert _mc._fade_prev_side(dn_feed, "BTC", "tf2") == 1


def test_fade_prev_ignores_current_window_bars():
    """Causality: bars inside the current window never move the side —
    only data strictly before s = floor(now/W)*W is read."""
    a, sa = _feed_with("BTC", +1.0, cur_bars=2)
    b, sb = _feed_with("BTC", +1.0, cur_bars=30)
    assert sa == sb
    assert mc._fade_prev_side(a, "BTC", "tf2") == \
        mc._fade_prev_side(b, "BTC", "tf2") == -1


def test_fade_prev_thin_tape_falls_back_to_zero():
    feed = DummyFeed()
    st = feed.states["BTC"]
    st.push_bar(int(time.time()) - 60, 100.0, 1.0)   # a single bar
    assert mc._fade_prev_side(feed, "BTC", "tf2") == 0


def test_apply_direction_fade_prev_wired_for_live_cells():
    """HOLD POLICY supersedes fade_prev for direction (PR #13): the
    TRADE side comes from backtest/results/hold_wr.json — exact
    certified side per (asset, tf) — while the legacy mandate config
    stays as the fallback when the hold record is missing."""
    # legacy config object unchanged (fallback path)
    assert mc.cfg("BTC")["direction"] == "fade_prev"
    # SOL replaced PAXG on 2026-11-27 with NO measured mandate (PAXG's row
    # did not survive the certified walk), so its fallback direction is the
    # undeclared "base" rather than a rule borrowed from a deleted asset.
    assert mc.cfg("SOL")["direction"] == "base"
    # direction_rule surfaces the CERTIFIED rule so the engine prints
    # and routes exactly what was measured
    assert mc.direction_rule("BTC", "tf2") == "hold:mom525600"
    # v8 §17 sides (the v6-era constant-"down" cell is gone — every
    # certified side on the 9-year tape is a mom* family).
    #
    # These are the CERTIFIED lanes. ETH tf1 and SOL tf4 used to be listed
    # here; both are now tf1-shaped (24 h) or belong to an asset with no
    # walk, so they correctly fall through to the mandate default instead.
    assert mc.direction_rule("ETH", "tf2") == "hold:mom129600"
    assert mc.direction_rule("BNB", "tf4") == "hold:mom43200"
    # tf1 is a 24-HOUR lane as of 2026-11-27 and the hold artifact is keyed
    # by duration, so it is not managed: the rule falls through to the
    # mandate default ("base"), NOT to a 22-minute certification.
    assert mc.direction_rule("SOL", "tf1") == "base"
    # certified side BEATS the raw vote: on a flat tape the hold rule
    # answers +1 for both raw directions (never a skip, never stale)
    bnb_feed, _ = _feed_with("BNB", +1.0)
    assert mc.apply_direction("BNB", "tf2", +1, 0, bnb_feed) == 1
    assert mc.apply_direction("BNB", "tf2", -1, 0, bnb_feed) == 1
    # thin tape on a long-momentum cell → certified warmup fallback +1
    # (hold.py's full-coverage rule — never a skip, never a stale raw)
    btc_feed, _ = _feed_with("BTC", +1.0)
    assert mc.apply_direction("BTC", "tf2", -1, 0, btc_feed) == 1
    # Every CERTIFIED lane is managed by the hold policy. tf1 is deliberately
    # NOT: it is a 24-hour window the duration-keyed artifact has never
    # walked. All 16 cells still exist, still compute and still issue — the
    # difference is only whether a certified side exists to quote.
    from app import hold_policy as hp
    for sym, tf in hp.known_cells():
        if sym == "SOL":
            # no walk for SOL yet: NO lane of it is certified
            assert not hp.managed(sym, tf), (sym, tf)
        elif tf == "tf1":
            assert not hp.managed(sym, tf), (sym, tf)
        else:
            assert hp.managed(sym, tf), (sym, tf)
