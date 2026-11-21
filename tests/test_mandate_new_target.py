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
    """The win-rate directive search (43k–107k configs/asset, n≥60,
    EV>0 after friction, ≥5/6 folds EV>0, ≥4/6 folds WR≥0.60, full
    WR≥0.66 — see backtest/mandate_wr_search.py) decides which cells
    issue — app/mandate_cfg.py. BTC issues tf2/tf4 (mom44+regime+majority
    agree, WR 69.7%), ETH tf1/tf2 (inv4g+deep RSI, 70.5%), BNB tf2/tf3
    (mom44+top-quartile gate+unanimous agree, 66.2%), PAXG tf3/tf4
    (mom44+RSI band+majority agree, 70.5%); every non-measured cell
    stays capped watch-only."""
    from tests.test_engine import make_engine
    engine, _feed, _ = make_engine()
    for sym, tf in (("BTC", "tf2"), ("BTC", "tf4"), ("ETH", "tf1"),
                    ("ETH", "tf2"), ("BNB", "tf2"), ("BNB", "tf3"),
                    ("PAXG", "tf3"), ("PAXG", "tf4")):
        out = engine._tier_for(sym, tf, 0.5)     # a huge score
        assert out["tier"] == "full", (sym, tf, out.get("tier_rule"))
    for sym, tf in (("BTC", "tf1"), ("BTC", "tf3"), ("ETH", "tf3"),
                    ("ETH", "tf4"), ("BNB", "tf1"), ("BNB", "tf4"),
                    ("PAXG", "tf1"), ("PAXG", "tf2")):
        out = engine._tier_for(sym, tf, 0.5)
        assert out["tier"] == "watch-only", (sym, tf)
        assert "mandate" in out["tier_rule"]
