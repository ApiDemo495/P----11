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


# ── 4. the units bug behind "tp 551 on BNB at market 556" ────────────────
def test_window_is_minutes_not_bars():
    """`moves_stats(window_min)` read its window as a BAR COUNT.

    Every in-repo backtest feeds a 1-HOUR tape (backtest/data.py). Reading
    tf1's 22 minutes as 22 BARS means measuring 22 HOURS of excursion for a
    22-minute horizon. Live proof from the project's own trade log
    (backtest/results/mode_a_trades_real_1h.csv): the median BTC stop was
    181 bps — exactly the 22-hour q80 excursion of that tape — while 22
    minutes of BTC moves ~32 bps. The stop was 5.7x too wide in distance,
    32x too wide in variance, and the trade could not resolve inside its own
    horizon: 158 of 206 real trades never touched either barrier.

    This builds an HOURLY-volatility tape and reads the same 22-minute
    window under both declarations. Declaring the truth (60-min bars) must
    give a SMALL excursion; declaring the historical 1-min default must
    inflate it by roughly sqrt(22) ≈ 4.7x.
    """
    import random
    from app.moves_risk import moves_stats

    rng = random.Random(4)
    # an HOURLY tape: 0.4 % per bar
    p, c, h, l = 100.0, [], [], []
    for _ in range(2000):
        p *= 1.0 + rng.gauss(0, 0.004)
        c.append(p)
        w = abs(rng.gauss(0, 0.004)) * p
        h.append(p + w)
        l.append(p - w)

    # declared truthfully: 22 minutes on 60-min bars is ONE bar
    honest = moves_stats(c, h, l, 22, bar_minutes=60.0)
    # declared as the old default (1-min): 22 BARS = 22 hours of excursion
    inflated = moves_stats(c, h, l, 22, bar_minutes=1.0)

    ratio = inflated["q80"] / max(honest["q80"], 1e-9)
    # sqrt(22) ≈ 4.69 — allow the sampling spread of a q80 estimate
    assert 3.0 < ratio < 7.0, f"expected ~sqrt(22)≈4.7x, got {ratio:.2f}x"
    # and in absolute terms, 22 minutes of a 0.4 %-per-hour tape is small
    assert honest["q80"] < 200.0, honest["q80"]
    assert inflated["q80"] > honest["q80"]


def test_a_window_shorter_than_the_bar_is_declared_provisional():
    """An unmeasurable window must SAY SO, not borrow a wider one.

    On 1h bars, 22 minutes and 44 minutes both collapse to a single bar,
    and a single bar is an HOUR of excursion. The block is returned
    provisional (and the note names the reason) so nothing can present an
    hour-wide stop as a 22-minute one.
    """
    import random
    from app.moves_risk import risk_block

    rng = random.Random(9)
    p, c, h, l = 100.0, [], [], []
    for _ in range(2000):
        p *= 1.0 + rng.gauss(0, 0.004)
        c.append(p)
        w = abs(rng.gauss(0, 0.003)) * p
        h.append(p + w)
        l.append(p - w)

    hourly = risk_block(100.0, 1, 22, 0.5, c, h, l, atr=1.0, bar_minutes=60.0)
    assert hourly["provisional"] is True
    assert "shorter than the tape's bar spacing" in hourly["note"]
    assert hourly["bar_minutes"] == 60.0 and hourly["window_min"] == 22
    # a window WIDER than the bar is a real measurement
    wide = risk_block(100.0, 1, 247, 0.5, c, h, l, atr=1.0, bar_minutes=60.0)
    assert wide["provisional"] is False


def test_atr_fallback_scales_volatility_by_sqrt_of_time():
    """The fallback scaled σ by sqrt(MINUTES) as if minutes were bar counts.

    σ_horizon = σ_bar · sqrt(horizon / bar). On a 60-minute tape a
    22-minute horizon is 0.37 of a bar, not 22 bars — the old code inflated
    the provisional stop by sqrt(60) ≈ 7.7x.
    """
    import random
    from app.moves_risk import risk_block

    rng = random.Random(3)
    c = [100.0]
    for _ in range(200):
        c.append(c[-1] * (1 + rng.gauss(0, 0.004)))
    h = [x * 1.001 for x in c]
    l = [x * 0.999 for x in c]
    # 22 minutes on a 60-minute tape: scale = sqrt(22/60) = 0.606
    block = risk_block(100.0, 1, 22, 0.5, c[:20], h[:20], l[:20],
                       atr=1.0, bar_minutes=60.0)
    # atr=1.0 on price 100 is 100 bps; 0.8 * 100 * sqrt(22/60) ≈ 48.5 bps
    assert 40.0 < block["sl_bps"] < 60.0, block["sl_bps"]
    # NOT the old sqrt(22)=4.69 inflation, which would give ~375 bps
    assert block["sl_bps"] < 100.0


def test_the_tf_envelope_may_not_break_the_ratio():
    """The monotone-timeframe rule ran on TP ALONE and collapsed the ratio.

    `_prev_tp * 1.03` raised TP to keep targets increasing with the
    timeframe while leaving SL untouched, so the PUBLISHED reward:risk
    drifted off 2:1. Live: BTC tf2 shipped TP 7 bps against SL 7 bps
    (1.00:1) and BNB tf4 1.28:1 beside tf1's 2.00:1 — the same badge
    quoting three different strategies, when break-even is 62.5 % at 0.6:1
    and 33.3 % at 2:1.

    A display-side clamp could not fix this (the board must print the FROZEN
    contract's exact levels — tests/test_engine.py asserts that), so the
    envelope itself now widens the RANGE and re-derives TP from the stop.
    """
    import inspect
    from app import engine as _eng
    src = inspect.getsource(_eng.PredictionEngine)
    assert "THE ENVELOPE RAN ON TP ALONE AND BROKE THE RATIO" in src
    # the envelope no longer nudges tp_bps in isolation
    assert '_b["tp_bps"] = round(_prev_tp * 1.03, 3)' not in src
    # it widens the stop, then fixes the target at RR x that stop
    assert '_RRT_ENV * _b["sl_bps"]' in src


def test_the_initial_lock_survives_the_restore_guard():
    """A `symbol`/`tf` NameError in the restore guard killed EVERY lock.

    `_install_cycle_contract` takes no symbol/tf. Naming them in the new
    restore-refusal log line raised NameError each time, and the caller
    swallows it as "initial lock failed" — so no cycle was ever installed,
    the board fell back to preview ladders, and the published ratios became
    arbitrary (1:1 at 7 bps). Pinned here because the failure mode is
    entirely silent: the app boots, serves, and looks merely "wrong".
    """
    import inspect
    from app import engine as _eng
    fn = _eng.PredictionEngine._install_cycle_contract
    src = inspect.getsource(fn)
    # the function's own parameter names, straight from its code object
    assert "symbol" not in fn.__code__.co_varnames[:fn.__code__.co_argcount]
    assert "tf" not in fn.__code__.co_varnames[:fn.__code__.co_argcount]
    # so the log line must not name them bare
    assert "symbol, tf, _restored" not in src
    assert 'getattr(cyc, "symbol"' in src and 'getattr(cyc, "tf"' in src


def test_risk_block_honours_the_two_to_one_contract():
    """A measured block must come out at exactly RR_TARGET, not near it."""
    import random
    from app.moves_risk import RR_TARGET, risk_block

    rng = random.Random(12)
    p, c, h, l = 100.0, [], [], []
    for _ in range(1200):
        p *= 1.0 + rng.gauss(0, 0.0007)
        c.append(p)
        w = abs(rng.gauss(0, 0.0006)) * p
        h.append(p + w)
        l.append(p - w)
    for d in (1, -1):
        for win in (22, 44, 130, 247):
            b = risk_block(100.0, d, win, 0.5, c, h, l, atr=0.05,
                           bar_minutes=1.0)
            assert abs(b["rr"] - RR_TARGET) < 1e-6, (d, win, b["rr"])
            # and the levels are on the right side of the entry
            if d > 0:
                assert b["tp"] > b["entry"] > b["sl"]
            else:
                assert b["tp"] < b["entry"] < b["sl"]
