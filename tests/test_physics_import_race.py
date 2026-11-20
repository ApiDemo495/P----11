"""P18 must not depend on sklearn inspecting another thread's imports."""
import importlib

import numpy as np

from app.physics import betti_numbers


def test_betti_uses_equivalent_distances_without_sklearn_pairwise(monkeypatch):
    ripser_module = importlib.import_module("ripser.ripser")
    prices = [100 + i * 0.07 + np.sin(i * 0.7) for i in range(40)]
    returns = np.diff(prices) / np.asarray(prices[:-1])
    points = np.column_stack((returns[1:], returns[:-1]))
    eps = 2 * np.std(returns)
    diagrams = ripser_module.ripser(points, thresh=eps, maxdim=1)["dgms"]
    expected = tuple(int(np.sum((d[:, 0] <= eps) &
                                (np.isinf(d[:, 1]) | (d[:, 1] > eps))))
                     for d in diagrams)

    def fail(*args, **kwargs):
        raise AssertionError("sklearn pairwise path was used")

    monkeypatch.setattr(ripser_module, "pairwise_distances", fail)
    assert betti_numbers(prices) == expected
