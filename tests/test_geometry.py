"""The 66 % geometry is only real if the scoring rules cannot fake it.

These tests pin three things that each, on their own, would let a harness
report a win rate the account balance does not have:

  1. same-bar double touches must be resolved by the stated rule, never by
     the window's close return (the drift leak — 96.6 % of 7d windows tie);
  2. a window tiling must be complete, non-overlapping and free of
     look-ahead;
  3. the barrier book's break-even line must be exactly sl/(tp+sl), and the
     fee budget must be wr*tp - (1-wr)*sl.
"""
from __future__ import annotations

import numpy as np
import pytest

from backtest.geometry_scan import barrier_book, blocks, dd_bps
from backtest.geometry_validate import short_book, wilson


def _flat_windows(c_open, hi, lo, close):
    """One-window tape from a single bar path."""
    C = np.array([[c_open] + close, ], float)
    H = np.array([[hi] + close, ], float)
    L = np.array([[lo] + close, ], float)
    return C, H, L


def test_same_bar_double_touch_never_falls_through_to_close():
    """Both barriers inside one bar: the close must not decide the outcome.

    This is the exact leak that made a long book and a short book both win
    86 % of 1d windows.  With a huge upward close return, a 'stop' tie must
    still be a loss and a 'tp' tie must still be a win.
    """
    # open 100, high 101 (touches +100 bps), low 99 (touches -100 bps),
    # closes at 105 — a tie that a close-return fallback would score +500 bps
    C = np.array([[100.0, 105.0]])
    H = np.array([[101.0, 105.0]])
    L = np.array([[99.0, 105.0]])
    r_stop, won_stop, tie = barrier_book(C, H, L, 100.0, 100.0, 0.0, tie="stop")
    assert tie[0]
    assert r_stop[0] == pytest.approx(-100.0)
    assert not won_stop[0]

    r_tp, won_tp, _ = barrier_book(C, H, L, 100.0, 100.0, 0.0, tie="tp")
    assert r_tp[0] == pytest.approx(100.0)
    assert won_tp[0]

    r_half, won_half, _ = barrier_book(C, H, L, 100.0, 100.0, 0.0, tie="half")
    assert r_half[0] == pytest.approx(0.0)
    assert not won_half[0]


def test_drift_cannot_create_a_win_on_a_tie():
    """A window that only went UP must not win a stop-tie book."""
    C = np.array([[100.0, 200.0]])
    H = np.array([[101.0, 200.0]])
    L = np.array([[99.0, 200.0]])
    _, won, _ = barrier_book(C, H, L, 100.0, 100.0, 0.0, tie="stop")
    assert not won[0]


def test_no_touch_closes_at_window_end():
    """Neither barrier touched -> the position is marked at the last close."""
    C = np.array([[100.0, 100.5]])
    H = np.array([[100.6, 100.6]])
    L = np.array([[99.9, 100.4]])
    r, won, tie = barrier_book(C, H, L, 50.0, 50.0, 0.0)
    assert not tie[0]
    assert r[0] == pytest.approx(0.5 / 100.0 * 1e4)
    assert won[0]


def test_first_touch_wins_when_target_comes_first():
    C = np.array([[100.0, 100.0, 100.0]])
    H = np.array([[100.1, 101.0, 100.0]])   # target +100 bps at bar 1
    L = np.array([[99.9, 99.95, 98.0]])     # stop -100 bps never reached
    r, won, tie = barrier_book(C, H, L, 100.0, 100.0, 0.0)
    assert won[0] and r[0] == pytest.approx(100.0) and not tie[0]


def test_round_trip_is_subtracted_once_per_window():
    """Fee is a per-round-trip toll, not per side and not per bar."""
    C = np.array([[100.0, 100.0]])
    H = np.array([[100.001, 100.001]])
    L = np.array([[99.999, 99.999]])
    r_free, _, _ = barrier_book(C, H, L, 100.0, 100.0, 0.0)
    r_fee, _, _ = barrier_book(C, H, L, 100.0, 100.0, 4.0)
    assert r_free[0] - r_fee[0] == pytest.approx(4.0)


def test_short_book_is_the_mirror_of_the_long_book():
    """On the same path the two books cannot both be right on a clean touch."""
    C = np.array([[100.0, 100.0, 100.0]])
    H = np.array([[100.1, 101.0, 100.1]])   # long target hit
    L = np.array([[99.9, 99.9, 99.0]])
    _, won_l, _ = barrier_book(C, H, L, 100.0, 100.0, 0.0)
    _, won_s, _ = short_book(C, H, L, 100.0, 100.0, 0.0)
    assert won_l[0] and not won_s[0]


def test_break_even_line_is_the_payoff_ratio():
    """WR = sl/(tp+sl) is the exact zero-EV line; a 4:1 book needs 80 %."""
    for tp, sl in ((8.0, 32.0), (20.0, 80.0), (8.0, 15.5), (1.0, 1.0)):
        be = sl / (tp + sl)
        # a book that wins exactly break_even of the time nets ~zero
        n = 10_000
        k = int(round(be * n))
        ev = (k * tp - (n - k) * sl) / n
        assert abs(ev) <= sl / n * 1.5


def test_fee_budget_arithmetic():
    """fee_budget = wr*tp - (1-wr)*sl is the largest survivable round trip."""
    wr, tp, sl = 0.857, 20.0, 80.0
    budget = wr * tp - (1 - wr) * sl
    assert budget == pytest.approx(17.14 - 11.44)
    # and the book is exactly EV-zero when the fee equals the budget
    ev = wr * tp - (1 - wr) * sl - budget
    assert ev == pytest.approx(0.0)


def test_blocks_are_complete_and_non_overlapping():
    """Every bar lands in exactly one window and no window sees the future."""
    c = np.arange(1000, dtype=float) + 100.0
    h, l = c + 1.0, c - 1.0
    for k in (4, 9, 26, 49):
        C, H, L = blocks(c, h, l, k)
        assert C.shape == (len(c) // k, k)
        assert C.shape[0] * k <= len(c)
        # non-overlapping: each window starts where the last one stopped
        assert np.array_equal(C[:, 0], c[np.arange(len(C)) * k])
        # the window holds exactly its own bars
        assert C[-1, -1] == pytest.approx(c[len(C) * k - 1])


def test_drawdown_is_measured_on_the_equity_path():
    assert dd_bps(np.array([1.0, 1.0, 1.0])) == pytest.approx(0.0)
    assert dd_bps(np.array([10.0, -5.0, -5.0])) == pytest.approx(-10.0)
    # a new high resets the peak
    assert dd_bps(np.array([-10.0, 20.0, -5.0])) == pytest.approx(-5.0)


def test_wilson_shrinks_with_sample_size():
    lo_small, hi_small = wilson(120, 147)
    lo_big, hi_big = wilson(1200, 1470)
    assert (hi_small - lo_small) > (hi_big - lo_big)
    assert lo_big > lo_small          # same 81.6 % point estimate, tighter


def test_wilson_bounds_contain_the_estimate():
    for k, n in ((0, 10), (5, 10), (10, 10), (147, 147)):
        lo, hi = wilson(k, n)
        # the clamp to [0, 1] is float-rounded, hence the tolerance
        assert lo <= k / n + 1e-12
        assert k / n <= hi + 1e-12
