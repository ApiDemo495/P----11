"""Contracts for the sequential portfolio engine.

The engine is the referee of the whole alpha programme: it decides whether a
claimed edge exists.  These pins are therefore about *accounting*, not
accuracy — next-bar entry, stop-wins-ties, one position at a time, taker
fees out of every trade, overlap-aware statistics, and no peeking at the
decision bar's own close.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.portfolio import run_sleeve, trade_stats


class _Bars:
    """Minimal bars object: the engine only reads o/h/l/c."""

    def __init__(self, o, h, l, c):
        self.o = np.asarray(o, float)
        self.h = np.asarray(h, float)
        self.l = np.asarray(l, float)
        self.c = np.asarray(c, float)
        self.t = np.arange(len(self.c), dtype=float)


def _flat(n=40, px=100.0):
    return _Bars(np.full(n, px), np.full(n, px), np.full(n, px),
                 np.full(n, px))


def test_entry_fills_at_next_bar_open_not_the_decision_close():
    # decision at bar 5 (close 100), bar 6 opens at 110 -> entry must be 110
    o = np.full(40, 100.0)
    o[6] = 110.0
    b = _Bars(o, o, o, np.full(40, 100.0))
    score = np.zeros(40)
    score[5] = 1.0
    # barriers far away so only the vertical (time) exit can trigger
    sigma = np.full(40, 0.5)
    trades = run_sleeve(b, score, pt=2.0, sl=1.0, hz=5, sigma=sigma,
                        fees_bps=0.0)
    assert len(trades) == 1
    assert trades[0]["i"] == 6            # filled on bar 6
    # flat market after entry -> time stop at close 100 from entry 110
    assert trades[0]["gross"] == pytest.approx(100.0 / 110.0 - 1.0)


def test_stop_wins_ties_when_a_bar_touches_both_barriers():
    # width 0.01 -> TP 102, SL 98 on entry 100; one bar spans both
    b = _Bars([100.0] * 6, [100.0, 100.0, 103.0, 100.0, 100.0, 100.0],
              [100.0, 100.0, 97.0, 100.0, 100.0, 100.0], [100.0] * 6)
    score = np.zeros(6)
    score[0] = 1.0
    trades = run_sleeve(b, score, pt=2.0, sl=1.0, hz=3,
                        sigma=np.full(6, 0.01), fees_bps=0.0)
    assert trades[0]["exit"] == "sl"
    assert trades[0]["gross"] == pytest.approx(-0.01)


def test_one_position_at_a_time_no_overlapping_trades():
    rng = np.random.default_rng(7)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 600)))
    b = _Bars(c, c * 1.001, c * 0.999, c)
    score = np.sign(rng.normal(size=600))
    trades = run_sleeve(b, score, pt=2.0, sl=1.0, hz=20,
                        sigma=np.full(600, 0.004), fees_bps=12.0)
    assert len(trades) > 5
    for prev, nxt in zip(trades, trades[1:]):
        assert nxt["i"] > prev["i"] + prev["bars"]   # no overlap


def test_mask_blocks_entries():
    b = _flat()
    score = np.ones(40)
    mask = np.zeros(40, bool)
    assert run_sleeve(b, score, mask, pt=2.0, sl=1.0, hz=5,
                      sigma=np.full(40, 0.01)) == []
    mask[10] = True
    trades = run_sleeve(b, score, mask, pt=2.0, sl=1.0, hz=5,
                        sigma=np.full(40, 0.01))
    assert len(trades) == 1 and trades[0]["i"] == 11


def test_taker_fee_is_subtracted_from_every_trade():
    b = _flat()
    score = np.zeros(40)
    score[3] = 1.0
    sigma = np.full(40, 0.01)
    free = run_sleeve(b, score, pt=2.0, sl=1.0, hz=4, sigma=sigma, fees_bps=0.0)
    paid = run_sleeve(b, score, pt=2.0, sl=1.0, hz=4, sigma=sigma, fees_bps=12.0)
    assert free[0]["gross"] == pytest.approx(paid[0]["gross"])
    assert paid[0]["net"] == pytest.approx(paid[0]["gross"] - 12e-4)


def test_future_score_cannot_change_a_past_fill():
    # only bars before the fill decide the entry side
    b = _flat(30)
    s1 = np.zeros(30)
    s1[4] = 1.0
    s2 = s1.copy()
    s2[9:] = -1.0                       # different future
    a = run_sleeve(b, s1, pt=2.0, sl=1.0, hz=3, sigma=np.full(30, 0.01))
    c = run_sleeve(b, s2, pt=2.0, sl=1.0, hz=3, sigma=np.full(30, 0.01))
    assert a[0]["i"] == c[0]["i"] == 5
    assert a[0]["side"] == c[0]["side"] == 1.0


def test_time_stop_exits_at_the_close_of_the_horizon_bar():
    b = _flat()
    score = np.zeros(40)
    score[2] = -1.0
    trades = run_sleeve(b, score, pt=5.0, sl=5.0, hz=10,
                        sigma=np.full(40, 0.01))
    assert trades[0]["exit"] == "time"
    assert trades[0]["bars"] == 10
    assert trades[0]["gross"] == pytest.approx(0.0)


def test_stats_are_empty_safe():
    st = trade_stats([])
    assert st["n"] == 0 and math.isnan(st["ev_bp"])


def test_stats_overlap_correction_shrinks_the_t_stat():
    """Clustered trades on one path must not look like independent evidence."""
    trades = [{"i": 10 * k, "bars": 100, "gross": 0.01, "net": 0.0088,
               "exit": "tp", "r": 1.0, "sigma": 0.01} for k in range(60)]
    st = trade_stats(trades, fees_bps=12.0)
    assert st["n"] == 60
    # holding 100 bars with entries 10 bars apart -> ~10x overlap
    assert st["neff"] < 10
    assert st["t_stat"] < 10.0


def test_stats_profit_factor_and_win_rate_are_from_net():
    trades = ([{"i": 10 * k, "bars": 5, "gross": 0.02, "net": 0.0188,
                "exit": "tp", "r": 2.0, "sigma": 0.01} for k in range(3)] +
              [{"i": 10 * k + 5, "bars": 5, "gross": -0.01, "net": -0.0112,
                "exit": "sl", "r": -1.0, "sigma": 0.01} for k in range(2)])
    st = trade_stats(trades, fees_bps=12.0)
    assert st["win_rate"] == pytest.approx(0.6)
    assert st["profit_factor"] == pytest.approx((3 * 0.0188) / (2 * 0.0112), rel=1e-3)


def test_trailing_stop_never_moves_against_the_position():
    # price ramps up then collapses: the trail must lock in a gain
    c = np.array([100, 100.5, 101, 102, 103, 104, 105, 100, 99, 98], float)
    b = _Bars(c, c, c, c)
    score = np.zeros(10)
    score[0] = 1.0
    trades = run_sleeve(b, score, pt=50.0, sl=50.0, hz=9, trail=1.0,
                        sigma=np.full(10, 0.01), fees_bps=0.0)
    assert trades[0]["exit"] == "sl"
    assert trades[0]["gross"] > 0.03        # exited above entry via the trail


# ─────────────────── maker execution and the fill-model bias ───────────────────


def _dip_bars():
    """Bar 1 dips 2 % below its open, then recovers; bar 2 opens at the dip."""
    o = np.array([100.0, 100.0, 98.0, 98.0, 98.0, 98.0], float)
    h = np.array([100.0, 100.0, 98.5, 98.0, 98.0, 98.0], float)
    l = np.array([100.0, 96.0, 97.5, 98.0, 98.0, 98.0], float)
    c = np.array([100.0, 97.0, 98.2, 98.0, 98.0, 98.0], float)
    return _Bars(o, h, l, c)


def test_maker_fill_next_bar_needs_the_level_to_hold():
    b = _dip_bars()
    score = np.zeros(6)
    score[0] = 1.0                      # long, limit 0.5σ under the open
    sig = np.full(6, 0.04)              # 0.5σ = 2 % -> limit 98
    tr = run_sleeve(b, score, pt=6.0, sl=5.0, hz=3, sigma=sig,
                    entry="maker_pullback", pullback=0.5,
                    maker_fee_bps=2.0, taker_fee_bps=5.0, fill_mode="next_bar")
    assert tr, "the order should fill: bar 2 opens at the limit"
    assert tr[0]["i"] == 2              # filled one bar after the touch
    assert tr[0]["fee_bp"] == pytest.approx(8.0)


def test_maker_fill_is_dropped_when_price_bounces_away():
    b = _dip_bars()
    b.o[2] = 99.5                       # next open is above the limit
    score = np.zeros(6)
    score[0] = 1.0
    sig = np.full(6, 0.04)
    diag = {}
    tr = run_sleeve(b, score, pt=6.0, sl=5.0, hz=3, sigma=sig,
                    entry="maker_pullback", pullback=0.5, fill_mode="next_bar",
                    diag=diag)
    assert tr == []
    assert diag["attempts"] == 1 and diag["fills"] == 0


def test_wick_fill_mode_is_recorded_as_optimistic():
    """The wick mode fills at the extreme; the guard test below proves why the
    default is `next_bar`."""
    b = _dip_bars()
    score = np.zeros(6)
    score[0] = 1.0
    sig = np.full(6, 0.04)
    tr = run_sleeve(b, score, pt=6.0, sl=5.0, hz=3, sigma=sig,
                    entry="maker_pullback", pullback=0.5, fill_mode="wick")
    assert tr and tr[0]["i"] == 1


def test_random_walk_coin_flip_earns_nothing_under_the_default_fill_model():
    """A coin flip on a random walk must be ~flat gross.

    The naive "fill at the limit inside the touched bar" model pays a coin
    flip ~+15 bp per trade — free money that does not exist.  This is the
    regression guard for that artefact: the default mode must stay unbiased.
    """
    rng = np.random.default_rng(11)
    n = 4000
    r = rng.normal(0, 0.006, n)
    c = 100 * np.exp(np.cumsum(r))
    o = c * np.exp(rng.normal(0, 0.0024, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, n)))
    b = _Bars(o, h, lo, c)
    sig = np.full(n, 0.006)
    side = np.sign(rng.normal(size=n))
    mask = np.ones(n, bool)
    out = {}
    for mode in ("wick", "next_bar"):
        tr = run_sleeve(b, side, mask, pt=6.0, sl=1.0, hz=48, trail=3.0,
                        runner=True, sigma=sig, entry="maker_pullback",
                        pullback=0.5, maker_fee_bps=2.0, taker_fee_bps=5.0,
                        maker_penalty_bps=1.0, min_penetration=1e-4,
                        fill_mode=mode)
        out[mode] = trade_stats(tr, fees_bps=8.0)["gross_ev_bp"]
    assert out["wick"] > 8.0, ("the wick model should show its bias; if it "
                               "no longer does, revisit the default")
    assert abs(out["next_bar"]) < 6.0, (
        f"next_bar must be unbiased on a random walk, got {out['next_bar']}")


def test_block_bootstrap_widens_the_error_for_dependent_trades():
    """Clustered trades must not masquerade as independent evidence."""
    from app.portfolio import block_bootstrap
    rng = np.random.default_rng(5)
    indep = rng.normal(0.001, 0.01, 400)
    walk = np.cumsum(rng.normal(0, 0.002, 400))
    walk = walk - walk.mean()                      # persistent, not iid
    a = block_bootstrap(indep, block=20)
    b = block_bootstrap(walk, block=20)
    assert a["se_bp"] > 0
    # the plain t-stat would be mean/(sd/sqrt(400)); the bootstrap must not
    # be fooled by the persistence
    assert b["se_bp"] > 3 * a["se_bp"]
    assert b["lo_bp"] < b["hi_bp"]
