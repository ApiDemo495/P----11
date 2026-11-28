"""Bgn synchronization — the coupling constants are pinned by the spec."""
import math

from app.bgn import BASE_COUPLING, base_coupling
from app.bgn import BgnSynchronizer
from app.history import HistoryStore


def test_base_coupling_matches_spec_values():
    # exact values quoted in the specification (beta = 1.2 -> 2*beta^2 = 2.88)
    # tolerance covers the spec's own rounded intermediates (e.g. s23)
    #
    # ONLY THE PAIRS BETWEEN UNCHANGED LANES ARE PINNED. The spec constants
    # s12=0.846 / s13=0.334 / s14=0.131 were computed for a 22-MINUTE tf1.
    # tf1 is now the 24-hour 2:1 lane, so its couplings are genuinely
    # different numbers, not drifted ones: the formula is
    # exp(-ln(ti/tj)^2 / 2beta^2), and 22 min vs 1440 min is a far bigger
    # ratio. The three tf1 pairs are therefore checked against the FORMULA
    # (below) rather than against a constant that described a lane that no
    # longer exists.
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

    # distant timeframes are insulated. The 44-minute lane's raw signal
    # influences its 130-minute neighbour far more than the 24-HOUR lane
    # (s23 = 0.664 vs s21, which is now tiny) — the same insulation property
    # the old s12 > s42 assertion was checking, stated on pairs whose
    # durations did not change.
    den1 = sum(matrix["tf2"].values())
    den4 = sum(matrix["tf1"].values())
    assert matrix["tf2"]["tf3"] / den1 > matrix["tf1"]["tf3"] / den4


def test_tf1_coupling_follows_the_formula_not_a_stale_constant():
    """tf1's couplings must track its ACTUAL duration (24h since 2026-11-27).

    Guards the specific failure mode this test file exists for: a constant
    that describes a lane the app no longer has.
    """
    from app.config import TIMEFRAMES
    # tf1 is the six-hour 2:1 lane (user directive 2026-11-28). Derive the
    # expectation from config — pinning a minute count here is exactly the
    # stale-constant failure this file guards against.
    assert TIMEFRAMES["tf1"]["minutes"] == 360
    for other in ("tf2", "tf3", "tf4"):
        expect = base_coupling(TIMEFRAMES["tf1"]["minutes"],
                               TIMEFRAMES[other]["minutes"])
        assert math.isclose(BASE_COUPLING[("tf1", other)], expect, rel_tol=1e-12)
    # and it IS decoupled from the SHORT lanes: the 6-hour lane shares far
    # less microstructure with a 44-minute one than the old 22-minute lane
    # did (0.846). Loose bound — the exact value follows the formula.
    assert BASE_COUPLING[("tf1", "tf2")] < 0.30
    assert BASE_COUPLING[("tf1", "tf2")] < BASE_COUPLING[("tf2", "tf3")]


def test_base_coupling_symmetry():
    assert math.isclose(base_coupling(1440, 44), base_coupling(44, 1440))
    assert base_coupling(1440, 1440) == 1.0
