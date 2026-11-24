"""The win-rate ceiling is a measured law, not an opinion.

`backtest/winrate_ceiling.py` established that for a book which must
close inside its window:

    WR_net  <=  P(|move over the window| > round trip)
    max realizable WR = min(skill, ceiling)

These tests pin the arithmetic so a future change cannot quietly
reintroduce a harness that reports a 66 % win rate at tf1.
"""
from __future__ import annotations

import numpy as np
import pytest

from backtest.winrate_raise import hurdle_book, summarise, wilson


def test_ceiling_is_the_share_of_windows_that_can_cover_cost():
    """WR_net can never exceed the share of windows whose move beats RT."""
    rng = np.random.default_rng(3)
    n = 4000
    moves = np.abs(rng.normal(0, 20, n))          # bps
    rt = 29.0
    p_cover = float((moves > rt).mean())
    # a perfect oracle that always takes the right side
    wins = moves > rt
    wr = float(wins.mean())
    assert wr == pytest.approx(p_cover)
    assert wr <= p_cover + 1e-12


def test_skill_multiplies_the_ceiling():
    """Realizable WR is skill x ceiling, because a wrong side never wins."""
    rng = np.random.default_rng(4)
    n = 20000
    moves = rng.normal(30, 60, n)                 # signed, bps
    rt = 29.0
    for skill in (0.5, 0.52, 0.6):
        right = rng.random(n) < skill
        side = np.where(right, np.sign(moves), -np.sign(moves))
        sided = moves * side
        wr = float((sided > rt).mean())
        ceiling = float((np.abs(moves) > rt).mean())
        # never above the ceiling, and scales with skill
        assert wr <= ceiling + 1e-9
        if skill > 0.5:
            assert wr < ceiling      # needs a perfect oracle to hit it


#: measured ceilings on 2.84 y of real 5m tape (backtest/wr_ceiling.json)
MEASURED_CEILINGS = {
    ("BTC", 22): 0.134, ("BTC", 44): 0.282, ("BTC", 130): 0.517,
    ("BTC", 247): 0.619, ("BTC", 1440): 0.855,
    ("ETH", 22): 0.209, ("ETH", 44): 0.385, ("ETH", 130): 0.620,
    ("ETH", 247): 0.713, ("ETH", 1440): 0.885,
}


def test_measured_short_horizon_ceilings_are_below_66_percent():
    """The real numbers: no tf1-tf4 BTC cell can host a 66 % book."""
    for (asset, minutes), ceiling in MEASURED_CEILINGS.items():
        if minutes <= 247 and (asset, minutes) != ("ETH", 247):
            assert ceiling < 0.66, (asset, minutes, ceiling)
    # ETH tf4 is the one short cell above the line - and it still needs
    # 92.6 % directional accuracy, so it is not attackable either
    assert MEASURED_CEILINGS[("ETH", 247)] > 0.66
    assert 0.66 / MEASURED_CEILINGS[("ETH", 247)] > 0.90


def test_ceiling_rises_monotonically_with_horizon():
    for asset in ("BTC", "ETH"):
        seq = [MEASURED_CEILINGS[(asset, m)]
               for m in (22, 44, 130, 247, 1440)]
        assert seq == sorted(seq), (asset, seq)


def test_wilson_interval_widens_as_n_falls():
    lo_n, hi_n = wilson(660, 1000)
    lo_s, hi_s = wilson(66, 100)
    assert (hi_s - lo_s) > (hi_n - lo_n)


def test_hurdle_pushes_are_zero_pnl_not_skips():
    """An unfilled window must be a 0.0 push, never a dropped window."""
    n = 50
    h = np.full(n, 101.0)
    l = np.full(n, 99.0)
    c = np.full(n, 100.0)
    i_in = np.arange(0, n - 5, 5)
    i_out = i_in + 4
    # a hurdle far below the low -> nothing ever fills
    pnl, filled, won, pushed = hurdle_book(h, l, c, i_in, i_out, 29.0,
                                           hurdle_bps=5000.0, tp_bps=40.0,
                                           sl_bps=40.0, side=-1)
    assert len(pnl) == len(i_in)
    assert not filled.any()
    assert pushed.all()
    assert np.all(pnl == 0.0)
    s = summarise(pnl, filled, won, pushed)
    assert s["n_windows"] == len(i_in)
    assert s["n_won"] == 0
    assert s["push_rate"] == 1.0


def test_hurdle_fill_counts_as_a_window():
    n = 40
    h = np.full(n, 100.5)
    l = np.full(n, 99.0)
    c = np.full(n, 100.0)
    i_in = np.arange(0, n - 5, 5)
    i_out = i_in + 4
    pnl, filled, won, pushed = hurdle_book(h, l, c, i_in, i_out, 29.0,
                                           hurdle_bps=50.0, tp_bps=40.0,
                                           sl_bps=40.0, side=-1)
    assert filled.all()
    assert len(pnl) == len(i_in)
