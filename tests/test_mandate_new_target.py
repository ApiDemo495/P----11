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
    assert "_mag = _factor * NEW_TARGET_RR * _sl_bps" in src
    assert "0.65 * min(1.0, abs(float(" in src   # factor kept (tilt)


def test_paxg_tier_capped_by_two_sample_mandate():
    """PAXG: negative holdout EV in both modes × all TFs × both splits
    (backtest/mandate.py 2y walk) AND the independent prior window
    (REAL_RESULTS per-TF <50%) — two samples agree: cap the tier."""
    from tests.test_engine import make_engine
    engine, _feed, _ = make_engine()
    out = engine._tier_for("PAXG", "tf4", 0.5)   # a huge score
    assert out["tier"] == "watch-only"
    assert "mandate" in out["tier_rule"]
    # BTC keeps its score-band tier untouched
    btc = engine._tier_for("BTC", "tf4", 0.5)
    assert btc["tier"] == "full"
