"""Sizing must not be able to lie about ruin.

The numbers in §29 are only worth acting on if the risk arithmetic is exact:
Kelly must be a real maximum of realised log growth, the drawdown definition
must match what a trader loses, and the safe-size rule must be monotone in
the constraint it is given.
"""
from __future__ import annotations

import numpy as np
import pytest

from backtest.path_risk import (_curve_stats, bootstrap_paths, kelly,
                                risk_of, safe_leverage)


def test_kelly_is_a_maximum_of_log_growth():
    """No leverage on the grid beats the returned one on mean log growth."""
    rng = np.random.default_rng(11)
    daily = rng.normal(0.0006, 0.012, 900)
    L, g = kelly(daily, lo=0.5, hi=30, steps=300)
    for probe in np.linspace(0.5, 30, 60):
        assert np.mean(np.log(1.0 + probe * daily)) <= g + 1e-9
    assert L > 0


def test_kelly_scales_inversely_with_variance():
    """Same mean, double the volatility -> roughly half the optimal size."""
    rng = np.random.default_rng(12)
    a = rng.normal(0.001, 0.01, 4000)
    b = a * 0.5 + 0.0005          # half the spread, same mean
    La, _ = kelly(a, lo=0.5, hi=60, steps=400)
    Lb, _ = kelly(b, lo=0.5, hi=60, steps=400)
    assert Lb > 1.5 * La          # lower vol supports more leverage


def test_curve_stats_drawdown_is_peak_to_trough():
    daily = np.array([0.10, -0.20, 0.05])
    st = _curve_stats(daily, 1.0)
    eq = np.cumprod(1 + daily)
    peak = np.maximum.accumulate(eq)
    assert st["max_dd"] == pytest.approx(float((eq / peak - 1).min()))


def test_loss_of_100_percent_is_flagged_as_ruin():
    """Leverage that can wipe the account must be reported, not smoothed over."""
    daily = np.array([0.5, -0.34, 0.02])
    st = _curve_stats(daily, 2.0)      # -34 % x 2 = -68 %, survives
    assert not st["ruined"]
    st2 = _curve_stats(np.array([-0.5, 0.01]), 2.0)   # -50 % x 2 = -100 %
    assert st2["ruined"] and st2["cagr"] == -1.0


def test_risk_of_measures_a_real_drawdown_not_zero_equity():
    """The old 'ruin' test could never fire; this one must."""
    rng = np.random.default_rng(13)
    daily = rng.normal(-0.004, 0.02, 600)      # a losing book
    paths = bootstrap_paths(daily, n_paths=300, seed=3)
    low = risk_of(daily, 1.0, paths)
    high = risk_of(daily, 10.0, paths)
    assert high["p_dd50"] >= low["p_dd50"]
    assert high["p_dd80"] >= low["p_dd80"]
    assert low["median_dd"] <= 0.0


def test_safe_leverage_is_monotone_in_the_constraint():
    """A looser drawdown budget can never require a smaller position."""
    rng = np.random.default_rng(14)
    daily = rng.normal(0.0005, 0.011, 1200)
    paths = bootstrap_paths(daily, n_paths=400, seed=5)
    tight = safe_leverage(daily, paths, limit=0.01)
    loose = safe_leverage(daily, paths, limit=0.20)
    assert loose >= tight
    # and the chosen size must actually satisfy its own constraint
    assert risk_of(daily, max(tight, 0.5), paths)["p_dd50"] <= 0.01 + 1e-9


def test_safe_leverage_never_exceeds_the_grid():
    rng = np.random.default_rng(15)
    daily = rng.normal(0.001, 0.005, 800)
    paths = bootstrap_paths(daily, n_paths=200, seed=9)
    L = safe_leverage(daily, paths, limit=0.05)
    assert 0.0 <= L <= 40.0


def test_bootstrap_paths_keep_losing_streaks():
    """Block resampling must preserve runs, otherwise ruin is understated.

    The invariant is that every block is a *contiguous slice of the original
    series* — that is what keeps a 20-day losing streak in one piece.  (A
    naive 'consecutive values differ' check is wrong: two adjacent blocks can
    legitimately start and end on the same value.)
    """
    daily = np.where(np.arange(1000) % 2 == 0, -0.01, 0.012)
    paths = bootstrap_paths(daily, n_paths=20, block=20, seed=2)
    assert paths.shape == (20, 1000)
    origin = {tuple(daily[i:i + 20]): i for i in range(len(daily) - 20)}
    for row in paths:
        for b in range(0, 1000, 20):
            block = tuple(row[b:b + 20])
            assert block in origin, "block is not a contiguous slice of the tape"
    # and a run of losses survives inside a block
    streak = np.array([-0.01] * 20 + [0.01] * 20)
    p = bootstrap_paths(streak, n_paths=50, block=20, seed=1)
    assert (np.prod(1 + p, axis=1) < 0.90).any()
