"""Tests for the diagnosis-driven extensions (F36-F44, E2, P26-P27).

These lock in the four things the 3-year grading forced:

 1. the new formulas are bounded, finite and DIRECTIONALLY SANE on clean
    tapes (an uptrend must not score negative);
 2. a perfectly linear tape must not read as "no signal" — zero diff
    variance used to divide to zero exactly when the trend was strongest;
 3. the discipline overlay may only ever REDUCE position size;
 4. a barrier whose target sits inside the round trip is refused, because
    that is the specific geometry that made 86 % of windows "win" while
    the book bled.
"""
from __future__ import annotations

import math
import random

import pytest

from app.emotion import (complacency_drift, discipline_overlay, fomo_pressure,
                         panic_urgency, revenge_bias)
from app.formulas.f36_f44 import (f36_tvf, f37_rpg, f38_vrp, f39_ctx, f40_vlq,
                                  f41_fnd, f42_mdc, f43_crc, f44_hdg)
from app.hold_policy import (MIN_REWARD_RISK, MIN_TP_OVER_RT,
                             barrier_payability)
from app.history import HistoryStore
from app.models import FormulaContext
from app.physics import p26_barrier_friction, p27_travel_time

FNS = [("TVF", f36_tvf), ("RPG", f37_rpg), ("VRP", f38_vrp),
       ("CTX", f39_ctx), ("VLQ", f40_vlq), ("FND", f41_fnd),
       ("MDC", f42_mdc), ("CRC", f43_crc), ("HDG", f44_hdg)]


def ctx(px, vol=None, w=None):
    n = len(px)
    return FormulaContext(
        symbol="BTC", tf="tf1", W=n - 1 if w is None else w, P=list(px),
        V=vol or [100.0] * n, T=list(range(n)), hist=HistoryStore(),
        O=list(px), H=list(px), L=list(px))


def up(n=120):
    return [100.0 + i * 0.1 for i in range(n)]


def down(n=120):
    return [100.0 - i * 0.1 for i in range(n)]


# ─────────────────────── registration / contract ────────────────────
def test_new_formulas_are_registered():
    from app.formulas import FORMULA_FUNCTIONS, FORMULA_NAMES
    for fid in range(36, 45):
        assert fid in FORMULA_FUNCTIONS, f"F{fid} not registered"
        assert fid in FORMULA_NAMES, f"F{fid} has no name"


def test_new_formulas_have_declared_identities_and_guards():
    from app.formulas.trace import DOUBLE_CHECK, PRIMARY_GUARD
    for fid in range(36, 45):
        assert fid in DOUBLE_CHECK, f"F{fid} missing identity"
        assert PRIMARY_GUARD.get(fid, 0) > 0, f"F{fid} has no bar guard"


# ───────────────────────── bounded and finite ───────────────────────
@pytest.mark.parametrize("name,fn", FNS)
def test_formula_is_bounded_and_finite(name, fn):
    rng = random.Random(7)
    for _ in range(120):
        px = [100.0]
        for _ in range(rng.randint(30, 160)):
            px.append(max(0.01, px[-1] * (1 + rng.gauss(0, 0.003))))
        v = fn(ctx(px))
        assert isinstance(v, float), f"{name} returned {type(v)}"
        assert math.isfinite(v), f"{name} returned {v}"
        assert -1.0 <= v <= 1.0, f"{name} out of bounds: {v}"


@pytest.mark.parametrize("name,fn", FNS)
def test_formula_survives_a_constant_tape(name, fn):
    """A dead-flat tape must not raise or NaN — it is a real market state."""
    v = fn(ctx([100.0] * 60))
    assert math.isfinite(v) and -1 <= v <= 1


def test_linear_trend_is_not_read_as_no_signal():
    """Regression: zero diff-variance divided to 0.000 on a perfect trend.

    A perfectly linear tape has stdev(diffs) == 0, so the naive
    vol-normalised move was 0/0 -> 0, reporting "no signal" precisely when
    the trend was strongest.  TVF, VRP and CTX must all see it.
    """
    assert f36_tvf(ctx(up())) > 0.5
    assert f38_vrp(ctx(up())) > 0.5
    assert f39_ctx(ctx(up())) > 0.5


def test_linear_downtrend_is_negative():
    assert f36_tvf(ctx(down())) < -0.5
    assert f38_vrp(ctx(down())) < -0.5
    assert f39_ctx(ctx(down())) < -0.5


def test_cost_to_edge_never_inverts_sign():
    """Regression: CTX used to score an uptrend NEGATIVE.

    The edge on offer is the realised move; using *expected* vol inverted
    the sign on clean trends.  Sign must follow the move.
    """
    for k in (0.02, 0.05, 0.1, 0.3):
        a = f39_ctx(ctx([100.0 * (1 + k) ** i for i in range(80)]))
        b = f39_ctx(ctx([100.0 * (1 - k) ** i for i in range(80)]))
        assert a > 0, f"uptrend scored {a} at step {k}"
        assert b < 0, f"downtrend scored {b} at step {k}"


def test_hold_duration_separates_trend_from_chop():
    trend = f44_hdg(ctx(up()))
    chop = f44_hdg(ctx([100.0 + (0.5 if i % 2 else -0.5) for i in range(120)]))
    assert trend > 0.5
    assert chop < -0.3
    assert trend > chop


def test_funding_is_zero_and_honest_without_data():
    assert f41_fnd(ctx(up())) == 0.0        # no series attached


def test_cross_asset_is_zero_without_others():
    assert f43_crc(ctx(up())) == 0.0


def test_cross_asset_reads_agreement():
    c = ctx(up())
    c.others = {"ETH": (list(range(120)), up())}
    assert f43_crc(c) > 0.5                 # both up -> confirmed
    c.others = {"ETH": (list(range(120)), down())}
    assert f43_crc(c) < -0.5                # disagree -> stand down


# ────────────────────── emotion / discipline ────────────────────────
def test_panic_urgency_high_when_move_is_below_cost():
    d = panic_urgency([1e-5], 29.0)         # 1 bp move vs 29 bp cost
    assert d["value"] > 0.9
    d2 = panic_urgency([0.01], 29.0)        # 100 bp move vs 29 bp cost
    assert d2["value"] < 0.3


def test_fomo_flags_a_spent_move():
    spent = [100.0] * 15 + [100.0 + i * 2.0 for i in range(1, 8)]
    assert fomo_pressure(spent, 5.0)["value"] > 0.5


def test_revenge_detects_sizing_up_after_losses():
    # a mixed ledger: losses followed by big size, wins followed by small
    d = revenge_bias([-1, 1, -1, 1, -1, 1],
                     [1.0, 3.0, 1.0, 3.0, 1.0, 3.0])
    assert d["value"] > 0.3
    assert d["ratio"] > 1.0


def test_revenge_admits_when_it_cannot_measure():
    """An all-loss ledger has no win-side baseline: say so, do not invent one."""
    d = revenge_bias([-1, -1, -1, -1], [1.0, 3.0, 3.0, 3.0])
    assert d["value"] == 0.0
    assert "unmeasurable" in d["reason"]


def test_complacency_rises_with_win_streak():
    assert complacency_drift(0, 1.0)["value"] < 0.1
    assert complacency_drift(12, 2.0)["value"] > 0.6


def test_discipline_overlay_can_only_reduce_size():
    """No emotional reading may ever increase position size."""
    calm = discipline_overlay(
        fomo={"value": 0.0}, panic={"value": 0.0},
        revenge={"value": 0.0}, complacent={"value": 0.0})
    assert calm["size_mult"] == 1.0
    hot = discipline_overlay(
        fomo={"value": 0.9}, panic={"value": 0.9},
        revenge={"value": 0.9}, complacent={"value": 0.9})
    assert hot["size_mult"] < calm["size_mult"]
    assert hot["block_new"] is True


def test_discipline_mult_stays_in_bounds():
    hot = discipline_overlay(
        fomo={"value": 1.0}, panic={"value": 1.0},
        revenge={"value": 1.0}, complacent={"value": 1.0})
    assert 0.0 < hot["size_mult"] <= 1.0


# ───────────────────── physics: the cost gate ───────────────────────
def test_p26_refuses_a_target_inside_the_round_trip():
    v, d = p26_barrier_friction(None, tp_bps=23, sl_bps=150, rt_bps=29)
    assert v < 0 and d["verdict"] == "unpayable"


def test_p26_accepts_a_payable_geometry():
    v, d = p26_barrier_friction(None, tp_bps=600, sl_bps=1500, rt_bps=29)
    assert v > 0.25 and d["verdict"] == "payable"


def test_p26_penalises_lottery_shape():
    """A 1:40 geometry is not efficient just because its target clears cost."""
    good, _ = p26_barrier_friction(None, 600, 1500, 29)
    lottery, _ = p26_barrier_friction(None, 600, 24000, 29)
    assert lottery < good


def test_p27_says_when_a_target_is_inside_one_window():
    class S:
        W = 22

    rng = random.Random(0)
    s = S()
    px = [1000.0]
    for _ in range(300):
        px.append(px[-1] * (1 + rng.gauss(0, 0.0008)))
    s.P = px
    near, dn = p27_travel_time(s, 20.0)
    far, df = p27_travel_time(s, 500.0)
    assert near > far
    assert "one window" in df["verdict"] or df["ratio"] > 1


def test_barrier_payability_refuses_the_certified_geometry():
    """The app's own certified target sat inside the round trip."""
    d = barrier_payability(23.0, 150.0, 29.0)
    assert d["payable"] is False
    assert d["below_cost"] is True


def test_barrier_payability_accepts_a_sane_geometry():
    d = barrier_payability(600.0, 1500.0, 29.0)
    assert d["payable"] is True
    assert d["reward_risk"] > MIN_REWARD_RISK
    assert d["tp_bps"] > MIN_TP_OVER_RT * d["rt_bps"]


def test_barrier_payability_rejects_lottery_geometry():
    d = barrier_payability(600.0, 24000.0, 29.0)
    assert d["payable"] is False
    assert d["lottery_shape"] is True


def test_barrier_payability_handles_degenerate_input():
    assert barrier_payability(0.0, 100.0, 29.0)["payable"] is False
    assert barrier_payability(100.0, 0.0, 29.0)["payable"] is False
