"""Pins the 66 %-at-2:1 measurement so it cannot quietly be re-narrated.

These are arithmetic and integrity checks on backtest/target_66.py — the
heavy tape walk lives in `python -m backtest.target_66` and needs the
Binance mirror, so it is not part of the unit suite.
"""
from __future__ import annotations

import math

import pytest

from backtest.target_66 import (p_hit_first, required_drift, up_probability,
                               SIGNALS)


def test_two_to_one_geometric_win_rate_is_one_third():
    assert 1 / (1 + 2.0) == pytest.approx(1 / 3)


def test_66_at_two_to_one_is_double_the_geometric_rate():
    assert 0.66 / (1 / 3) == pytest.approx(1.98, abs=0.01)


def test_66_at_two_to_one_needs_near_unit_drift():
    """lambda = 0.982 per window — i.e. ~69 % directional accuracy."""
    lam = required_drift(2.0, 1.0, 0.66)
    assert lam == pytest.approx(0.982, abs=0.01)
    assert up_probability(lam / 2) == pytest.approx(0.688, abs=0.005)


def test_a_signal_cannot_move_the_near_barrier():
    """With SL nearer than TP, the stop is reached on noise alone: even a
    large drift leaves 2:1 well under 66 %."""
    for lam in (0.0, 0.2, 0.4, 0.6, 0.8):
        assert p_hit_first(2.0, 1.0, lam) < 0.66


def test_the_frontier_is_one_over_one_plus_ratio():
    """Every measured cell sat on the geometric curve. This is the law the
    frontier table demonstrates, checked in the drift-free limit."""
    for r in (0.25, 0.33, 0.5, 1.0, 2.0):
        assert p_hit_first(r, 1.0, 0.0) == pytest.approx(1 / (1 + r))


def test_no_abstention_signal_set():
    """The search must include both a signal and its FADE, so 'the other
    side' cannot be presented as a discovery later."""
    assert "trend-72h" in SIGNALS and "trend-72h-fade" in SIGNALS
    assert "revert-z96" in SIGNALS and "revert-z96-fade" in SIGNALS
