"""The NEW-mode target fix (2y real-data mandate).

The old formula — factor × σ (a MEAN-scaled expected move) — was
measured against the q80-MAE stop on 2y of real 1h bars: the ratio is
structurally 0.65–0.80, so EVERY new-mode window fell below the user's
1:1 floor and the model could never issue a trade (backtest/mandate.py
smoke: 34/34 new rows ratio-floor-skipped before the fix).

The fix: the target is the stop's own tail unit tilted by signal
strength — factor × NEW_TARGET_RR × sl_bps — so a saturated signal
reaches rr = NEW_TARGET_RR ≥ 1 while a weak signal stays gated below
the floor exactly as the user's rule intends.
"""
from __future__ import annotations

import math


def test_new_target_rr_is_tradable_at_full_strength():
    from app.engine import NEW_TARGET_RR

    factor_full = 0.35 + 0.65 * min(1.0, 0.10 / 0.08)   # saturated CCI
    assert math.isclose(factor_full, 1.0)
    assert factor_full * NEW_TARGET_RR >= 1.0, \
        "a full-strength signal must clear the 1:1 floor"


def test_new_target_still_gates_weak_signals():
    from app.engine import NEW_TARGET_RR

    factor_weak = 0.35 + 0.65 * min(1.0, 0.01 / 0.08)   # |cci| = 0.01
    assert factor_weak * NEW_TARGET_RR < 1.0, \
        "weak signals must stay watch-only under the user's floor"


def test_old_sigma_mean_formula_is_gone():
    """The mean-vs-tail comparison that made NEW unissuable."""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent
           / "app" / "engine.py").read_text()
    # the structural fix: magnitude comes from the stop, not from σ
    # (rr comes from the per-asset mandate config when set, else the
    # global NEW_TARGET_RR — the structure is the same)
    assert "_mag = _factor * _rr_n * _sl_bps" in src
    assert "_rr_n = _mcfg_c.new_rr(symbol) or NEW_TARGET_RR" in src
    assert "0.65 * min(1.0, abs(float(" in src   # factor kept (tilt)


def test_mandate_tiers_follow_measured_cells():
    """CERTIFIED HOLD POLICY owns all 16 (asset × tf) cells
    (backtest/results/hold_wr.json, PR #13): 7 cells are HOLD-OUT
    VERIFIED, BNB's 4 are IN-SAMPLE SHADOW (single-year), PAXG tf4 is
    IN-SAMPLE SHADOW · HOLDOUT MISS, PAXG tf1–tf3 are UNGRADEABLE /
    WARMUP (1 h history) — and every one of them may ISSUE at a huge
    score: the hold system has NO abstention gates (filters and
    confluence are skipped for managed cells; the legacy mandate tf
    cap only applies when hold_wr.json is missing).  The tier_rule
    carries the hold badge + measured win rate so the UI explains
    WHY each cell is live."""
    import os

    from tests.test_engine import make_engine
    from app import hold_policy as hp
    engine, _feed, _ = make_engine()
    for sym, tf in hp.known_cells():
        out = engine._tier_for(sym, tf, 0.5)     # a huge score
        assert out["tier"] == "full", (sym, tf, out.get("tier_rule"))
        assert "hold policy" in out["tier_rule"], (sym, tf)
        assert f"[{hp.tier(sym, tf)}]" in out["tier_rule"], (sym, tf)
    # the certified tier distribution is exact — and it is now gated by
    # the artifact's own ship verdict.  The shipped hold_wr.json FAILS
    # backtest/gates18.py gate (ii) (it records no REAL post-cost win
    # rate for any cell, so it cannot beat the v8 reference), therefore
    # the 7 cells whose rows held up out-of-fold are reported as
    # ROWS VERIFIED · SHIP GATE FAIL — never as HOLD-OUT VERIFIED.
    counts = hp.tier_counts()
    assert hp.artifact_ship_verdict()["gate_ii_pass"] is False
    assert counts.get("HOLD-OUT VERIFIED", 0) == 0
    assert counts[hp.TIER_SHIPGATE] == 7
    assert counts.get("HOLDOUT ROWS · ERA n<3", 0) == 0
    assert counts.get("IN-SAMPLE SHADOW", 0) == 0
    assert counts["IN-SAMPLE SHADOW · HOLDOUT MISS"] == 5
    assert counts["UNGRADEABLE / WARMUP"] == 4
    # legacy fallback still caps when the hold record is absent
    import app.hold_policy as _hp
    old_path = os.environ.get("HOLD_WR_PATH")
    os.environ["HOLD_WR_PATH"] = "/nonexistent/hold_wr.json"
    try:
        _hp.reload("/nonexistent/hold_wr.json")
        out = engine._tier_for("BTC", "tf1", 0.5)
        assert out["tier"] == "watch-only", out
        assert "mandate" in out["tier_rule"]
    finally:
        if old_path is None:
            os.environ.pop("HOLD_WR_PATH", None)
        else:
            os.environ["HOLD_WR_PATH"] = old_path
        _hp.reload(old_path)
