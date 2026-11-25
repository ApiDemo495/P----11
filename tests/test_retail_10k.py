"""Pins §36: retail cost arithmetic, the win-rate/payoff identity, and that
overlapping trades are not independent (the bug that faked a t of 9.49)."""
from __future__ import annotations

import numpy as np
import pytest

from backtest.retail_10k import SCENARIOS, RetailCosts
from backtest.target_66 import p_hit_first, required_drift


def test_retail_cost_components_add_up():
    c = RetailCosts(fee_bps=20.0, slippage_bps=5.0, spread_bps=5.0)
    assert c.round_trip_bps == 30.0
    assert c.round_trip == pytest.approx(0.003)


def test_scenarios_are_ordered():
    vals = [s.round_trip_bps for s in SCENARIOS.values()]
    assert vals == sorted(vals), "cost scenarios must be ordered cheapest first"


def test_driftless_win_rate_is_one_over_one_plus_r():
    """The identity behind §36.1: at R:1 the free win rate is 1/(1+R)."""
    for r in (0.5, 1.0, 1.5, 2.0, 3.0):
        assert p_hit_first(r, 1.0, 0.0) == pytest.approx(1.0 / (1.0 + r))


def test_66_percent_at_2to1_needs_more_than_66_percent_directional_accuracy():
    need = required_drift(2.0, 1.0, 0.66)
    assert need > 0.9, "a 66%/2:1 target implies a drift of ~1 sigma per window"


def test_breakeven_win_rate_formula():
    """(1+c)/(1+R+c). Costs can only raise the bar."""
    def be(r, c):
        return (1 + c) / (1 + r + c)
    assert be(2.0, 0.0) == pytest.approx(1 / 3)
    assert be(2.0, 0.1) > be(2.0, 0.0)
    assert be(1.5, 0.0) == pytest.approx(0.4)


def test_overlapping_trades_inflate_apparent_significance():
    """A worked example of the §36.4 artifact: N overlapping windows of the
    same series look ~sqrt(k) times more significant than k independent ones."""
    rng = np.random.default_rng(0)
    base = rng.normal(0.001, 1.0, 400)          # modest real drift
    indep_t = base.mean() / (base.std() / np.sqrt(len(base)))
    # overlapping: same underlying path counted 8 times over
    overlapped = np.repeat(base, 8)
    over_t = overlapped.mean() / (overlapped.std() / np.sqrt(len(overlapped)))
    # compare magnitudes: overlapping the same information sqrt(k) times must
    # inflate |t| by about sqrt(k) even though no new data was added
    assert abs(over_t) > abs(indep_t) * 2.0, "overlap must visibly inflate the t-stat"
    assert abs(over_t / indep_t) == pytest.approx(np.sqrt(8), rel=0.01)
