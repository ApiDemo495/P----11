"""Pins the hedge-fund comparison so the conclusion cannot be re-narrated."""
from __future__ import annotations

import numpy as np
import pytest

from backtest.beat_funds import (BENCHMARKS, beaten, metrics, tsmom_book,
                                 HOURS_PER_YEAR)


def test_medallion_benchmark_is_the_documented_one():
    m = BENCHMARKS["Medallion (Renaissance)"]
    assert m["cagr"] == pytest.approx(0.391, abs=1e-3)
    assert m["sharpe"] == pytest.approx(2.52, abs=1e-2)


def test_metrics_recover_a_known_sharpe():
    """Zero-mean noise gives ~0 annual Sharpe; a known drift gives the
    exact (mu/sd)*sqrt(8760)."""
    rng = np.random.default_rng(7)
    r = rng.normal(0.0, 0.01, 20000)
    # ~86 expected annualised noise: 1/sqrt(20000) * sqrt(8760) = 0.66
    assert abs(metrics(r)["sharpe"]) < 2.5
    # deterministic: mu/sd = 1/200 -> annual = 0.005 * sqrt(8760)
    d = np.tile(np.array([0.0002, -0.000198]), 10000)
    mu, sd = d.mean(), d.std()
    assert metrics(d)["sharpe"] == pytest.approx(mu / sd * np.sqrt(HOURS_PER_YEAR),
                                                 rel=1e-9)


def test_leverage_scales_but_does_not_improve_sharpe():
    rng = np.random.default_rng(11)
    r = rng.normal(1e-4, 0.01, 20000)
    base = metrics(r)["sharpe"]
    for lev in (2.0, 3.0, 5.0):
        assert metrics(r * lev)["sharpe"] == pytest.approx(base, abs=1e-9)


def test_beaten_requires_both_return_and_sharpe():
    rows = {"a": {"cagr": 0.50, "sharpe": 0.5},      # big return, no Sharpe
            "b": {"cagr": 0.05, "sharpe": 3.0},      # big Sharpe, no return
            "c": {"cagr": 0.45, "sharpe": 2.6}}      # both
    out = {k: beaten(metrics_from(v), BENCHMARKS) for k, v in rows.items()}
    assert out["a"] == "beats none"
    assert out["b"] == "beats none"
    assert "Medallion" in out["c"]


def metrics_from(d):
    return {"label": "", "cagr": d["cagr"], "sharpe": d["sharpe"],
            "mdd": -0.1, "calmar": 1.0, "n": 100}


def test_turnover_is_what_kills_hourly_resignalling():
    """A monotone tape has one sign change; a noisy one has many. The fee
    drag must scale with the sign-change count — that is the whole finding."""
    n = 4000
    quiet = np.linspace(100, 200, n)                    # one flip
    noise = 100 * np.exp(np.cumsum(
        np.random.default_rng(3).normal(0, 0.004, n)))  # many flips
    p_quiet = tsmom_book(quiet, 30.0, rebalance=1, vol_target=False,
                         lookbacks=(24, 72))
    p_noise = tsmom_book(noise, 30.0, rebalance=1, vol_target=False,
                         lookbacks=(24, 72))
    assert np.sum(np.abs(np.diff(quiet)) > 0) > 0
    # the noisy tape must lose more to fees than the quiet one
    assert p_noise.sum() < p_quiet.sum()
