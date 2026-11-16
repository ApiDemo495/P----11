"""Contracts for the alpha-research upgrade.

These are *behavioural* contracts, not accuracy claims: the accuracy work
is measured by `backtest/alpha_lab.py` on real bars and recorded in
`docs/ALPHA_RESEARCH.md`. What is pinned here is the machinery the live
pipeline depends on — signs, causality, degradation, calibration hygiene.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from app import alpha_lab as lab
from app.barriers import FRICTION_MULT, friction_floor_bps, levels, vol_scaled
from app.calib import Calibrator, isotonic_apply, isotonic_fit
from app.spillover import score as spill_score


# ───────────────────────── alpha lab ─────────────────────────


def _synthetic_bars(n: int = 900, seed: int = 3):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.002, n)
    closes = 100 * np.exp(np.cumsum(steps))
    highs = closes * (1 + np.abs(rng.normal(0, 0.001, n)))
    lows = closes * (1 - np.abs(rng.normal(0, 0.001, n)))
    vol = rng.lognormal(6, 0.4, n)
    return lab.Bars(
        t=np.arange(n) * 3600, o=closes, h=highs, l=lows, c=closes,
        v=vol, quote_v=vol * closes, trades=rng.lognormal(5, 0.3, n),
        taker_buy_v=vol * rng.uniform(0.3, 0.7, n),
        taker_buy_qv=vol * closes * 0.5)


def test_triple_barrier_first_touch_and_sl_tie():
    b = lab.Bars(t=np.arange(6) * 3600, o=np.full(6, 100.0),
                 h=np.array([100, 101, 103, 103, 103, 103.0]),
                 l=np.array([100, 99, 97, 97, 97, 97.0]),
                 c=np.full(6, 100.0), v=np.ones(6))
    # both barriers touched in the same bar → SL must win the tie
    tb = lab.triple_barrier(b, 0, horizon=5, pt_frac=0.02, sl_frac=0.02)
    assert tb["label"] == -1 and tb["barrier"] == "sl"


def test_triple_barrier_target_when_only_target_touched():
    b = lab.Bars(t=np.arange(6) * 3600, o=np.full(6, 100.0),
                 h=np.array([100, 100.5, 103, 103, 103, 103.0]),
                 l=np.array([100, 99.9, 99.8, 99.8, 99.8, 99.8]),
                 c=np.full(6, 100.0), v=np.ones(6))
    tb = lab.triple_barrier(b, 0, horizon=5, pt_frac=0.02, sl_frac=0.02)
    assert tb["label"] == 1 and tb["barrier"] == "tp"


def test_purged_folds_never_overlap_and_respect_time_order():
    folds = lab.purged_folds(500, folds=4, embargo=10)
    assert folds
    for tr, te in folds:
        assert tr.max() < te.min()          # forward only
        assert te.min() - tr.max() >= 10    # embargo honoured
        assert len(set(tr) & set(te)) == 0


def test_features_are_causal():
    """Truncating the future must not change a past feature row."""
    b = _synthetic_bars(600)
    X_full, idx_full, _ = lab.build_features(b)
    cut = 400
    truncated = lab.Bars(t=b.t[:cut], o=b.o[:cut], h=b.h[:cut], l=b.l[:cut],
                         c=b.c[:cut], v=b.v[:cut], quote_v=b.quote_v[:cut],
                         trades=b.trades[:cut],
                         taker_buy_v=b.taker_buy_v[:cut],
                         taker_buy_qv=b.taker_buy_qv[:cut])
    X_cut, idx_cut, _ = lab.build_features(truncated)
    keep = min(len(idx_cut), len(idx_full))
    assert keep > 50
    assert np.allclose(X_cut[:keep], X_full[:keep], atol=1e-9)


def test_isotonic_is_monotone_and_bracketed():
    x = np.linspace(0, 1, 200)
    y = (x + np.random.default_rng(1).normal(0, 0.05, x.size) > 0.5).astype(float)
    kx, ky = isotonic_fit(x, y)
    assert len(kx) == len(ky) and len(kx) > 0
    assert all(ky[i] <= ky[i + 1] + 1e-12 for i in range(len(ky) - 1))
    assert all(0.0 <= v <= 1.0 for v in ky)
    outside = isotonic_apply(kx, ky, -5.0)
    assert outside == ky[0]


def test_evaluate_symbol_runs_and_reports_both_sides():
    b = _synthetic_bars(1200)
    r = lab.evaluate_symbol(b, horizon=2, friction_bps=12.0, market=b)
    assert "error" not in r, r
    for key in ("before", "after", "after_gated", "calib", "barriers",
                "sign_stability"):
        assert key in r
    assert 0.0 <= r["after"]["sign_acc"] <= 1.0
    assert r["barriers"]["k_pt"] > 0 and r["barriers"]["k_sl"] > 0


def test_sign_prior_matches_the_documented_orientation():
    # every flow/technical feature must carry an explicit, non-zero prior
    for name in lab.FEATURE_NAMES:
        assert lab.SIGN_PRIOR.get(name, 0.0) != 0.0, name
    # momentum is documented as contrarian — the core finding
    assert lab.SIGN_PRIOR["mom_z"] == -1.0
    assert lab.SIGN_PRIOR["xasset"] == -1.0


# ───────────────────────── spillover ─────────────────────────


def test_spillover_requires_minimum_tape():
    out = spill_score([100.0] * 5, market_closes=[100.0] * 500)
    assert out["available"] is False and out["direction"] == 0


def test_spillover_flips_sign_with_the_tape():
    up = [100 * (1.002 ** i) for i in range(200)]
    down = [100 * (0.998 ** i) for i in range(200)]
    a = spill_score(up, highs=up, lows=up, market_closes=down,
                    bar_minutes=60, market_bar_minutes=60)
    b = spill_score(down, highs=down, lows=down, market_closes=up,
                    bar_minutes=60, market_bar_minutes=60)
    assert a["available"] and b["available"]
    # a rising tape → contrarian short; a falling tape → contrarian long
    assert a["direction"] < 0 and b["direction"] > 0
    assert a["score"] < 0 < b["score"]


def test_spillover_horizons_scale_with_bar_length():
    c = [100 * (1.001 ** i) for i in range(500)]
    fast = spill_score(c, market_closes=c, bar_minutes=1.0,
                       market_bar_minutes=1.0)
    slow = spill_score(c, market_closes=c, bar_minutes=60.0,
                       market_bar_minutes=60.0)
    assert fast["xasset"]["bars"] == 360      # 6 h of 1-minute bars
    assert slow["xasset"]["bars"] == 6        # 6 h of hourly bars


# ───────────────────────── calibration ─────────────────────────


def test_calibrator_is_identity_below_min_samples(tmp_path):
    cal = Calibrator(path=str(tmp_path / "c.json"))
    for _ in range(5):
        cal.observe("BTC", "TF1", 0.6, True)
    out = cal.apply("BTC", "TF1", 0.6)
    assert out["p"] is None and out["source"] == "insufficient"


def test_calibrator_corrects_overconfidence():
    cal = Calibrator(path=None)
    cal._lanes.clear()
    cal._fits.clear()
    # stated 0.7 wins only ~40% of the time
    for i in range(400):
        conf = 0.55 + 0.4 * ((i % 10) / 10.0)
        won = (i % 100) < 40
        cal.observe("BTC", "TF2", conf, won)
    hi = cal.apply("BTC", "TF2", 0.90)
    lo = cal.apply("BTC", "TF2", 0.55)
    assert hi["p"] is not None and hi["source"] == "lane"
    assert hi["p"] < 0.70          # the map pulls the false confidence down
    assert hi["p"] >= lo["p"]      # but stays monotone
    tel = cal.telemetry("BTC", "TF2")
    assert tel["n"] >= 400 and tel["brier_iso"] <= tel["brier_raw"] + 1e-9


def test_calibrator_ignores_non_finite_confidences():
    cal = Calibrator(path=None)
    cal._lanes.clear()
    for bad in (None, float("nan"), float("inf"), -0.2, 1.7, "x"):
        cal.observe("BTC", "TF1", bad, True)
    assert cal._lanes == {}


# ───────────────────────── barriers ─────────────────────────


def test_barrier_friction_floor_marks_thin_targets_watch_only():
    thin = vol_scaled(2.0, tf="TF1", friction_bps=12.0)
    assert thin["ok"] is False and "friction floor" in thin["reason"]
    wide = vol_scaled(40.0, tf="TF4", friction_bps=12.0)
    assert wide["ok"] is True
    assert wide["tp_bps"] >= friction_floor_bps(12.0)


def test_barrier_levels_point_the_right_way():
    long = levels(100.0, 1, 40.0, tf="TF4", friction_bps=12.0)
    short = levels(100.0, -1, 40.0, tf="TF4", friction_bps=12.0)
    assert long["tp"] > 100 > long["sl"]
    assert short["tp"] < 100 < short["sl"]


def test_barrier_needs_finite_sigma():
    out = vol_scaled(None, tf="TF2")
    assert out["ok"] is False and out["tp_bps"] is None
    assert FRICTION_MULT == 3.0


# ───────────────────────── end-to-end wiring ─────────────────────────


def test_engine_exposes_the_new_telemetry():
    """The systems payload must carry the upgrade's telemetry keys."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "app",
                            "engine.py")).read()
    for key in ('"p_correct"', '"calib_iso"', '"spillover"',
                '"vol_barriers"', '"spillover": int('):
        assert key in src, key
    js = open(os.path.join(os.path.dirname(__file__), "..", "static",
                           "app.js")).read()
    assert "_pWinRow" in js and "P(win)" in js
