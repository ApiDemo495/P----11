"""Contracts for the derivatives loader and the event-driven setups.

The whole point of the derivatives programme is that the features are *real*
and *causal*: funding is a settled number that exists at its stamp, a bar's
aggregate only exists once the bar closes, and a setup may never see the
future.  These pins enforce all three, plus the shape of each trigger.
"""
from __future__ import annotations

import os
import types

import numpy as np
import pytest

from app import derivs, event_alpha as ea


def _bars(c, t=None, v=None):
    c = np.asarray(c, float)
    n = c.size
    return types.SimpleNamespace(
        o=c.copy(), h=c * 1.001, l=c * 0.999, c=c,
        t=np.arange(n) * 3600.0 if t is None else np.asarray(t, float),
        v=np.ones(n) if v is None else np.asarray(v, float))


# ─────────────────────────────── derivs ───────────────────────────────


def test_align_is_causal_for_every_series():
    """A spike stamped after `t` must never appear at `t`."""
    t = np.arange(0, 10) * 3600.0
    fund = {"t": np.array([0.0, 8 * 3600.0]), "funding_rate": np.array([0.0001, 0.5])}
    basis = {"t": np.array([0.0, 9 * 3600.0]), "c": np.array([0.0, 0.9])}
    oi = {"t": np.array([0.0, 9 * 3600.0]), "oi_value": np.array([1.0, 100.0])}
    d = {"funding": fund, "basis_1h": basis, "oi_1h": oi}
    f = derivs.align(d, t, bar_seconds=3600.0)
    # the 0.5 funding settlement happens at t = 8h and is visible from then
    assert f["funding_rate"][8] == pytest.approx(0.5)
    assert f["funding_rate"][7] == pytest.approx(0.0001)
    # bar-aggregated series lag one bar, so nothing shows at its own stamp
    assert f["basis_bp"][9] == 0.0
    # open interest is lagged as well
    assert f["oi_chg_24h"][9] == 0.0


def test_align_without_files_is_zero_and_flags_no_derivs():
    t = np.arange(0, 12) * 3600.0
    f = derivs.align({}, t)
    assert np.all(f["funding_rate"] == 0.0)
    assert np.all(f["basis_bp"] == 0.0)
    assert np.all(f["has_derivs"] == 0.0)


def test_load_returns_empty_for_unknown_symbol(tmp_path):
    assert derivs.load("NOPE", cache=str(tmp_path)) == {}
    assert derivs.available("NOPE") is False


def test_rolling_z_is_bounded_and_causal_for_constant_prefix():
    x = np.zeros(50)
    z = derivs._roll_z(x, 10)
    assert np.all(np.nan_to_num(z) == 0.0)


# ─────────────────────────────── setups ───────────────────────────────


def test_leadlag_fires_only_when_the_laggard_has_not_moved():
    src = np.array([100, 100, 100, 100, 108, 108, 108], float)   # BTC jump
    tgt = np.array([100, 100, 100, 100, 100.5, 100.5, 100.5])    # ETH lags
    sig = ea.leadlag(_bars(tgt), _bars(src), k=2, impulse_sigma=0.5,
                     lag_max=0.35)
    fires = np.nonzero(sig["mask"])[0]
    assert fires.size >= 1
    assert np.all(sig["score"][fires] > 0)      # long the laggard
    # once the laggard has priced the move, the trigger stops firing
    tgt2 = np.array([100, 100, 100, 100, 108, 108, 108], float)
    sig2 = ea.leadlag(_bars(tgt2), _bars(src), k=2, impulse_sigma=0.5,
                      lag_max=0.35)
    assert int(sig2["mask"].sum()) == 0


def test_leadlag_is_short_when_btc_impulses_down():
    src = np.array([100, 100, 100, 100, 92, 92, 92], float)
    tgt = np.array([100, 100, 100, 100, 99.8, 99.8, 99.8], float)
    sig = ea.leadlag(_bars(tgt), _bars(src), k=2, impulse_sigma=0.5,
                     lag_max=0.5)
    fires = np.nonzero(sig["mask"])[0]
    assert fires.size >= 1
    assert np.all(sig["score"][fires] < 0)


def test_squeeze_needs_compression_and_a_breakout():
    # quiet drift, then a range break on a volume surge
    c = np.r_[np.full(400, 100.0), [101.5, 102.0]]   # flat, then a break
    v = np.r_[np.full(400, 1.0), [9.0, 9.0]]
    sig = ea.squeeze(_bars(c, v=v), None, compression=0.6, surge=1.5,
                     lookback=20)
    fires = np.nonzero(sig["mask"])[0]
    assert fires.size >= 1
    assert np.all(sig["score"][fires] > 0)


def test_basis_fade_fades_crowded_longs_on_a_rejection_wick():
    n = 60
    o = np.full(n, 100.0)
    c = np.full(n, 100.0)
    h = np.full(n, 100.0)
    lo = np.full(n, 100.0)
    c[-1], o[-1] = 99.0, 100.0          # closes down
    h[-1] = 112.0                        # long upper wick
    lo[-1] = 98.8
    v = np.full(n, 10.0)
    v[-1] = 60.0                         # volume climax on the rejection
    b = types.SimpleNamespace(o=o, h=h, l=lo, c=c, t=np.arange(n) * 3600.0,
                              v=v)
    f = {"funding_z": np.zeros(n), "basis_z": np.zeros(n),
         "funding_seen": np.ones(n)}
    f["funding_z"][-1] = 3.0
    f["basis_z"][-1] = 2.5
    sig = ea.basis_fade(b, f, funding_z=1.5, basis_z=1.0, climax=2.0, wick=0.4)
    assert sig["mask"][-1]
    assert sig["score"][-1] == -1.0      # short the crowded long side


def test_basis_fade_is_silent_without_derivatives():
    b = _bars(np.linspace(100, 101, 50))
    sig = ea.basis_fade(b, {}, funding_z=1.0, basis_z=1.0, climax=1.0,
                        wick=0.1)
    assert int(sig["mask"].sum()) == 0


def test_plan_is_frozen_and_carries_the_adverse_fill_model():
    for name in ("squeeze", "basis_fade", "leadlag"):
        p = ea.plan(name)
        assert p["runner"] is True, "convex exits are the point"
        assert p["sl"] <= 1.2, "a structural stop must stay tight"
        assert p["trail"] > p["sl"], "the runner must be wider than the stop"
        assert p["fill_mode"] == "next_bar", (
            "wick fills pay a coin flip ~15 bp on a random walk")
    assert ea.CATCHUP == {"k": 3, "impulse_sigma": 0.5, "lag_max": 0.30}
