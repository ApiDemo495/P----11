"""Pins the §34.2 lesson: rolling windows must be causal, and the bug that
produced a fake t-stat of 47 must not be able to come back."""
from __future__ import annotations

import numpy as np
import pytest


def causal_sum(x: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate(([0.0], np.cumsum(x)))
    out = np.full(len(x), np.nan)
    out[w - 1:] = c[w:] - c[:-w]
    return out


def test_causal_sum_uses_only_past_bars():
    x = np.arange(10.0)
    out = causal_sum(x, 4)
    assert out[4] == pytest.approx(x[1:5].sum())
    assert out[9] == pytest.approx(x[6:10].sum())
    assert np.isnan(out[:3]).all()


def test_causal_sum_rejects_future_injection():
    """Changing a LATER bar must never change an EARLIER output."""
    x = np.arange(20.0)
    a = causal_sum(x, 5)
    y = x.copy()
    y[12:] += 1000.0          # perturb only the future
    b = causal_sum(y, 5)
    # out[i] averages x[i-w+1..i], so out[11] still ends at bar 11 (safe);
    # out[12] is the first window that legitimately contains bar 12.
    assert np.allclose(a[:12], b[:12], equal_nan=True)
    assert not np.isclose(a[12], b[12])


def test_centered_convolve_would_leak():
    """Documents the exact bug: 'same' is centered, so it is not causal.
    This test exists to make the trap explicit, not to endorse it."""
    x = np.arange(10.0)
    centered = np.convolve(x, np.ones(4), "same")
    causal = causal_sum(x, 4)
    i = 4
    assert not np.isclose(centered[i], causal[i])
    assert centered[i] > causal[i]        # it has already seen bars 5,6


def test_no_lookahead_in_forward_return_alignment():
    """Forward return at i must be c[i+f]/c[i], never the reverse."""
    c = np.array([100.0, 110.0, 121.0])
    f = 1
    fr = np.full(len(c), np.nan)
    fr[:-f] = np.log(c[f:] / c[:-f])
    assert fr[0] == pytest.approx(np.log(1.10))
    assert np.isnan(fr[-1])
