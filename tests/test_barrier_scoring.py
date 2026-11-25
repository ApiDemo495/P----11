"""Barrier books must score a window that touches NOTHING at the close.

Reported symptom that led here: geometry_validate reported 94.2 % win rates on
1d/7d cells with `tie_rate = 0.000`, row after row, for different assets. The
books were scoring "neither barrier touched" as a TIE, and the default tie rule
is the stop — so a window that drifted +12 bps and closed in profit was booked
as a full -80 bps loss. `fu == fd` is true BOTH when both barriers fall in one
bar and when neither is ever reached (both sentinels == k).

These tests pin the distinction, because an off-by-one here silently rewrites
every backtest conclusion in the repo.
"""
from __future__ import annotations

import numpy as np

from backtest.geometry_scan import barrier_book
from backtest.geometry_validate import short_book


def _tape(path):
    px = np.array(path, float)
    return px.reshape(1, -1), px.reshape(1, -1), px.reshape(1, -1)


def test_flat_window_settles_at_the_close_not_as_a_stop():
    # rises 12 bps, never near tp=20 / sl=80
    C, H, L = _tape([100, 100.05, 100.10, 100.12])
    pnl, won, tie = barrier_book(C, H, L, 20, 80, 0.0)
    assert tie[0] is np.False_ or tie[0] == False        # not a tie
    assert abs(pnl[0] - 12.0) < 1e-6
    assert bool(won[0]) is True

    # the mirror: same tape, short book, must lose the same 12 bps
    pnl_s, won_s, tie_s = short_book(C, H, L, 20, 80, 0.0)
    assert abs(pnl_s[0] + 12.0) < 1e-6
    assert bool(won_s[0]) is False


def test_target_touch_still_wins():
    C, H, L = _tape([100, 100.15, 100.25, 100.30])
    pnl, won, tie = barrier_book(C, H, L, 20, 80, 0.0)
    assert abs(pnl[0] - 20.0) < 1e-6 and bool(won[0]) is True and not tie[0]


def test_stop_touch_still_loses():
    C, H, L = _tape([100, 99.7, 99.15, 99.10])
    pnl, won, tie = barrier_book(C, H, L, 20, 80, 0.0)
    assert abs(pnl[0] + 80.0) < 1e-6 and bool(won[0]) is False and not tie[0]


def test_both_barriers_in_one_bar_is_the_only_tie():
    # bar 2 spikes to +25 bps and drops to -85 bps: order unknown -> tie
    c = np.array([[100.0, 99.9], [100.0, 100.1]])
    h = np.array([[100.0, 100.25], [100.0, 100.1]])
    l = np.array([[100.0, 99.15], [100.0, 100.1]])
    pnl_stop, _, tie = barrier_book(c, h, l, 20, 80, 0.0, tie="stop")
    assert bool(tie[0]) is True and abs(pnl_stop[0] + 80.0) < 1e-6
    pnl_tp, _, _ = barrier_book(c, h, l, 20, 80, 0.0, tie="tp")
    assert abs(pnl_tp[0] - 20.0) < 1e-6


def test_tie_rate_cannot_be_zero_everywhere_in_a_long_window_book():
    """A 7d book over 147 windows once reported tie_rate exactly 0.000 for
    every asset and both sides — the fingerprint of the sentinel bug."""
    # construct a book where a good share of windows touch nothing
    k = 6
    n = 40
    base = 100.0
    c = np.full((n, k), base)
    c[:, -1] = base * 1.0005          # every window closes +5 bps (inside band)
    h = c.copy()
    l = c.copy()
    pnl, won, tie = barrier_book(c, h, l, 200, 800, 0.0)
    assert tie.sum() == 0, "no window touched a barrier, so none is a tie"
    assert won.all(), "all 40 closed in profit and must all count as wins"
    assert np.allclose(pnl, 5.0)
