"""Pins the four silent-wrong-answer bugs found by the 2026-11-26 audit.

The common thread: none of these raised, logged, or looked broken. Each
produced a number or a state that was confidently WRONG, which is the only
class of bug that can actually cost money.
"""
from __future__ import annotations

import math
import tempfile

import pytest

from app.cci import CCIEngine, calibrate_confidence, finalize_confidence
from app.ledger import SettlementLedger


# ── 1. the disagreement penalty must never invert the signal ─────────
def _fuse(vals, symbol="AUDIT"):
    eng = CCIEngine()
    w = {i: 1.0 for i in range(1, 7)}
    adj = {i: v for i, v in zip(range(1, 7), vals)}
    return eng.compute(symbol, "tf1", adj, drhe=1.0, raw_weights=w)


def test_disagreement_penalty_never_flips_the_sign():
    """Was: cci*(1 - 0.5*|rev|/|cci|) went NEGATIVE past |rev| > 2|cci|, so a
    net-bearish vote produced `direction: +1` (a BUY)."""
    for vals in ([0.10, 0.10, 0.10, 0.10, 0.10, -0.62],
                 [0.50, 0.10, 0.10, 0.10, 0.10, -0.95],
                 [0.30, 0.10, 0.10, 0.10, 0.10, -0.90],
                 [0.05, 0.05, 0.05, 0.05, 0.05, -0.40]):
        r = _fuse(vals)
        if abs(r["cci"]) < 1e-9:
            continue
        assert (r["cci"] > 0) == (r["cci_adjusted"] > 0), \
            f"penalty inverted the sign for {vals}: {r['cci']} -> {r['cci_adjusted']}"
        assert r["direction"] == (1 if r["cci_adjusted"] > 0 else -1 if r["cci_adjusted"] < 0 else 0)


def test_disagreement_penalty_only_reduces_magnitude():
    """A penalty that GROWS the number is not a penalty. Measured before the
    fix: -0.050 -> +0.400 (8x larger)."""
    for vals in ([0.50, 0.10, 0.10, 0.10, 0.10, -0.95],
                 [0.20, 0.20, 0.20, 0.20, 0.20, -0.90]):
        r = _fuse(vals)
        assert abs(r["cci_adjusted"]) <= abs(r["cci"]) + 1e-12


def test_heavy_disagreement_collapses_to_no_call_not_a_reversal():
    r = _fuse([0.50, 0.10, 0.10, 0.10, 0.10, -0.95])
    assert r["direction"] == 0
    assert r["issued"] is False


# ── 2. a failed computation must not report maximum confidence ───────
def test_nan_cci_yields_no_confidence_not_maximum():
    """Was: math.tanh(inf)=1.0 and max(0.02, min(0.98, nan)) returns 0.98
    because min/max discard NaN - so a broken calc printed 0.98 conviction."""
    for bad in (float("nan"), float("inf"), -float("inf")):
        r = calibrate_confidence({"cci_adjusted": bad, "drhe": 1.0})
        assert r["confidence"] == 0.0, f"{bad} produced {r['confidence']}"
        assert r.get("invalid") is True


def test_normal_confidence_still_monotone_and_bounded():
    prev = -1.0
    for c in (0.0, 0.01, 0.05, 0.2, 0.8, 3.0):
        v = calibrate_confidence({"cci_adjusted": c, "drhe": 1.0})["confidence"]
        assert v >= prev - 1e-12
        assert 0.0 <= v <= 1.0
        prev = v


def test_finalize_confidence_survives_missing_and_nan():
    """Was: pack["confidence"] raised KeyError on a raw CCI dict."""
    for pack in ({"confidence": float("nan")}, {"cci_adjusted": float("nan")},
                 {}, {"confidence": 0.5}):
        r = finalize_confidence(pack=pack, experience_quality=0.5,
                                p_dominant=0.5, state_aligned=True,
                                rtp=0.5, pcv=0.1)
        assert math.isfinite(r["confidence"])
        assert 0.0 <= r["confidence"] <= 1.0


# ── 3. a window with no barriers must not be graded ─────────────────
def _ledger():
    return SettlementLedger(path=tempfile.mkdtemp() + "/l.json")


def test_unavailable_barriers_are_not_counted_as_a_win():
    """Was: a settlement with barriers_available=False still stored
    result "win"/"loss", and stats() grades every win/loss row - so an
    ungradable window moved the published win rate."""
    led = _ledger()
    led.settle("BTC", "tf1", 100.0, 130.0, 1,
               first_touch="unavailable", barriers_available=False)
    st = led.stats("BTC", "tf1")
    assert st["n"] == 0 and st["wins"] == 0
    assert led.rows["BTC|tf1"][0]["result"] == "ungraded"


def test_ungraded_rows_do_not_dilute_real_grades():
    led = _ledger()
    led.settle("BTC", "tf1", 100.0, 110.0, 1)     # real win
    led.settle("BTC", "tf1", 100.0, 95.0, 1)      # real loss
    for _ in range(5):                            # five ungradable windows
        led.settle("BTC", "tf1", 100.0, 200.0, 1,
                   first_touch="unavailable", barriers_available=False)
    st = led.stats("BTC", "tf1")
    assert st["n"] == 2 and st["wins"] == 1 and st["rate"] == 0.5


def test_directional_grading_without_barrier_argument_still_works():
    """barriers_available omitted (None) is plain directional grading and
    must keep working - the caller is not in the barrier world at all."""
    led = _ledger()
    led.settle("BTC", "tf1", 100.0, 101.0, +1)
    led.settle("BTC", "tf1", 100.0, 99.0, +1)
    led.settle("BTC", "tf1", 100.0, 100.0, +1)
    st = led.stats("BTC", "tf1")
    assert st["n"] == 3 and st["wins"] == 1 and st["losses"] == 2


# ── 4. the UI formatters must never print NaN / invent zeros ─────────
def test_ui_formatters_share_one_unknown_rule():
    """Was: bp(NaN) rendered the literal string "NaN bp" into the panel and
    pct(NaN)/pct(null) rendered "0.00%" - a missing number displayed as a
    measured zero."""
    from pathlib import Path
    src = Path("static/app.js").read_text()
    assert "const _unknown = v => v == null || !isFinite(Number(v))" in src
    # the old, unguarded definitions must be gone
    assert "const bp = v => (v == null ?" not in src
    assert "const pct = v => `${(Number(v || 0)" not in src
