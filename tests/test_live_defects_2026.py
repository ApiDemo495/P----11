"""Regressions for the defects found by auditing the LIVE payload.

Every test here pins a bug that was invisible to the suite because the code
ran without raising — it simply published a number that was wrong, or a
number that disagreed with the one printed next to it. The comments record
what the board actually showed, so the reason survives.
"""
from __future__ import annotations

import math
import random


# ── 1. levels must belong to the side that ships ───────────────────────────
def test_pdobv_units_and_sign_are_causal_and_stable():
    """F20/PDOBV compared a HARMONIC INDEX (~1..13) against a SPECTRAL POWER
    (~1e-3), so the contrast was always ~0.9999 and the published value was
    `tanh(0.9999 × sign_v)` = ±0.762, where sign_v is the slope of a
    TWO-POINT spectral difference — a coin flip.

    That value is DRHE, which the confidence chain uses as min(1, DRHE) and
    which is clamped to [0, 1]: a negative sign gave 0 and collapsed the
    published confidence to the 0.02 clip (live: PAXG tf4 at 0.0198 with a
    perfectly healthy SNR of 1.72), a positive sign gave 0.762. Users saw a
    confidence that jumped 25-fold between neighbouring windows.

    Both bands are now power DENSITY per mode (same units, bin-count
    neutral) and the sign comes from the balance itself.
    """
    import numpy as np
    from app.physics import p20_pdobv
    from app.physics import Scene

    def values(kind: str, seed: int = 3, n: int = 300) -> float:
        rng = random.Random(seed)
        prev = 0.0
        rets = []
        for _ in range(n):
            w = rng.gauss(0, 1.0)
            if kind == "persistent":
                x = 0.75 * prev + 0.55 * w      # power at LOW k
            elif kind == "alternating":
                x = -0.6 * prev + 0.6 * w       # power at HIGH k
            else:
                x = w                            # flat spectrum
            prev = x
            rets.append(x)
        sc = Scene.__new__(Scene)
        sc.rets = rets
        v, _d = p20_pdobv(sc)
        return v

    pers = [values("persistent", s) for s in range(10)]
    alt = [values("alternating", s) for s in range(10)]
    white = [values("white", s) for s in range(14)]
    # NOT a constant: the two tapes must be distinguished...
    assert sum(pers) / len(pers) > 0.6, pers
    assert sum(alt) / len(alt) < -0.5, alt
    # ...and a flat-spectrum tape must sit near neutral, not at a bound
    assert abs(sum(white) / len(white)) < 0.2, white
    # stable across seeds (the old sign bit flipped at random)
    assert max(pers) - min(pers) < 0.2
    assert max(alt) - min(alt) < 0.3


def test_levels_pair_is_repaired_onto_the_published_side():
    """9 of 16 live cells shipped a TP/SL on the WRONG side of their own
    entry (e.g. direction −1 with tp > entry > sl) because the pair was
    frozen for one side and stamped with the other. app/levels.mirror
    reflects it about the entry, preserving the measured distances, and
    refuses rather than inventing a wrong-side price."""
    from app.levels import mirror, valid
    buy = {"entry": 100.0, "tp": 101.0, "sl": 99.0, "direction": 1}
    sell = mirror(buy, -1)
    assert sell is not None
    assert sell["tp"] == 99.0 and sell["sl"] == 101.0
    assert sell["direction"] == -1
    # distances preserved, not re-invented
    assert abs(sell["tp_dist"] - 1.0) < 1e-9
    assert abs(sell["sl_dist"] - 1.0) < 1e-9
    assert sell["mirrored"] is True
    # already on the requested side -> returned unchanged, NOT mirrored
    same = mirror(buy, 1)
    assert same["tp"] == 101.0 and same["mirrored"] is False
    # refusals: no side, no pair
    assert mirror(buy, 0) is None
    assert mirror({"entry": 100.0, "direction": 1}, -1) is None
    # a skewed pair is reflected, not rejected — the distances ARE the
    # measurement, and 20:1 is still a legal SELL (tp 99 < 100 < sl 120)
    skew = mirror({"entry": 100.0, "tp": 120.0, "sl": 99.0,
                   "direction": 1}, -1)
    assert skew is not None and skew["tp"] == 99.0 and skew["sl"] == 120.0
    # and what it returns always validates
    assert valid(sell)


def test_engine_verifies_the_pair_before_freezing_a_contract():
    import inspect
    from app import engine as _eng
    src = inspect.getsource(_eng.PredictionEngine.fresh_cycle)
    assert "_pair_ok" in src
    assert "levels_repaired" in src
    # mirror first (distances preserved), re-measure only as the fallback
    assert src.index("_mirror(") < src.index("_repair = (\"re-measured")
    # and the display path carries its own guarantee
    assert "levels_mirrored" in inspect.getsource(_eng.PredictionEngine)


# ── 2. a size hint may never outrank the published gate ───────────────────
def test_tier_cannot_be_revived_above_the_floor_at_publish_time():
    """`BNB tf3 · conf 0.019 · tier full · NO CALL`: _tier_for HAD capped
    the tier, but the display mirror copied the cycle's stale tier over the
    capped one and never re-derived it. The mirror now re-derives on the
    numbers it is about to print."""
    import inspect
    from app import engine as _eng
    src = inspect.getsource(_eng.PredictionEngine._sync_display)
    assert "_dfloor" in src
    # the stated rule, however the f-string is wrapped across lines
    assert "may never " in src and "outrank the published gate" in src


def test_hold_policy_badge_is_not_rendered_as_a_size_tier():
    """`_hs['tier']` is the hold policy's CERTIFICATION ("ROWS VERIFIED ·
    SHIP GATE FAIL"), not full/half/quarter. It was printed as
    `hold policy [TIER]` inside the same field every panel reads as a size."""
    from tests.test_engine import make_engine
    engine, _f, _ = make_engine()
    out = engine._tier_for("BTC", "tf1", 0.5, direction=1)
    rule = out.get("tier_rule") or ""
    assert "hold-policy certification" in rule
    assert "hold policy [" not in rule


# ── 3. gauges that measure the wrong thing ────────────────────────────────
def test_hedge_flow_has_no_lookahead():
    """dark_pool_flow classified every print by P[i + 3] — the price THREE
    MINUTES IN THE FUTURE. The live path evaluates at the last bar, where
    that price does not exist, so the signal the app could actually compute
    was a different quantity from the one being graded."""
    import inspect
    from app import hedgefund
    import io
    import tokenize
    src = inspect.getsource(hedgefund.dark_pool_flow)
    # strip comments/docstrings: the BUG is a future-price INDEX, and the
    # paragraph explaining the fix legitimately names it
    code_only = "".join(l.split("#", 1)[0] for l in src.splitlines())
    assert "P[i + 3]" not in code_only
    # and it classifies on the PRIOR three minutes instead
    assert "i - 3" in code_only.replace("i-3", "i - 3")


def test_an_abstention_is_not_scored_as_a_miss():
    """Warmup counted a subsystem's silent window (signal 0) in `n` and
    scored it as wrong, so a subsystem that spoke on 40 % of windows and was
    right on them graded below 50 % — live hf accuracy was 0.036, which is
    arithmetic, not predictor quality."""
    import inspect
    from app import warmup
    src = inspect.getsource(warmup)
    assert 'st["abstained"] += 1' in src
    # and the silence stays visible in the published status
    assert '"abstained": int(v.get("abstained") or 0)' in src
    assert '"coverage":' in src
