"""Bgn synchronization — the coupling constants are pinned by the spec."""
import math

from app.bgn import BASE_COUPLING, base_coupling
from app.bgn import BgnSynchronizer
from app.history import HistoryStore


def test_base_coupling_matches_spec_values():
    # exact values quoted in the specification (beta = 1.2 -> 2*beta^2 = 2.88)
    # tolerance covers the spec's own rounded intermediates (e.g. s23)
    assert math.isclose(BASE_COUPLING[("tf1", "tf2")], 0.846, abs_tol=0.0015)
    assert math.isclose(BASE_COUPLING[("tf1", "tf3")], 0.334, abs_tol=0.0015)
    assert math.isclose(BASE_COUPLING[("tf1", "tf4")], 0.131, abs_tol=0.0015)
    assert math.isclose(BASE_COUPLING[("tf2", "tf3")], 0.664, abs_tol=0.0025)
    assert math.isclose(BASE_COUPLING[("tf2", "tf4")], 0.354, abs_tol=0.0025)
    assert math.isclose(BASE_COUPLING[("tf3", "tf4")], 0.867, abs_tol=0.0015)


def test_diagonal_always_one():
    for i in ["tf1", "tf2", "tf3", "tf4"]:
        for j in ["tf1", "tf2", "tf3", "tf4"]:
            if i == j:
                assert BASE_COUPLING[(i, j)] == 1.0


def test_adjust_weighted_average_bootstrap():
    # constant vector adjusts to itself
    hist = HistoryStore()
    bgn = BgnSynchronizer(hist)
    matrix = bgn.matrix("BTC")
    raw = {"tf1": 1.0, "tf2": 0.0, "tf3": 0.0, "tf4": 0.0}
    adj = BgnSynchronizer.adjust(matrix, raw)
    row = matrix["tf1"]
    expected = row["tf1"] * 1.0 / sum(row.values())
    assert math.isclose(adj["tf1"], expected, rel_tol=1e-9)

    const = {t: 0.7 for t in ["tf1", "tf2", "tf3", "tf4"]}
    adj_const = BgnSynchronizer.adjust(matrix, const)
    assert all(math.isclose(v, 0.7, rel_tol=1e-9) for v in adj_const.values())

    # self-coupling dominates every row
    for ti in ["tf1", "tf2", "tf3", "tf4"]:
        row = matrix[ti]
        assert row[ti] == max(row.values())

    # distant timeframes are insulated: tf2's raw signal influences tf1 far
    # more than it influences tf4 (s12 = 0.846 vs s42 = 0.354, normalized)
    den1 = sum(matrix["tf1"].values())
    den4 = sum(matrix["tf4"].values())
    assert matrix["tf1"]["tf2"] / den1 > matrix["tf4"]["tf2"] / den4


def test_base_coupling_symmetry():
    assert math.isclose(base_coupling(22, 44), base_coupling(44, 22))
    assert base_coupling(22, 22) == 1.0
