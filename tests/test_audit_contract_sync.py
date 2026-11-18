"""Regression guards for the principal architectural audit.

Three invariants, each one a live screenshot bug:

  §1.2  one canonical ladder — ENTRY < TP1 < TP2 < TP3 < TF-TP (BUY),
        mirrored for a SELL, monotone and inside the span, on every panel
  §2.1  calibrated confidence — an SNR→tanh base with BOUNDED modulation,
        so a genuine setup can clear the 0.35 floor and no stack of
        dampers can multiply a real signal into 0.0002
  §1.3  window freeze — an active lock cannot be re-anchored every 10 s
"""
from __future__ import annotations

import pathlib

import pytest

from app.cci import NOISE_FLOOR, calibrate_confidence
from app.contracts import (build_contract, build_ladder, divergences,
                           ladder_valid, rung_fractions, validate)


# ── §1.2 the ladder ────────────────────────────────────────────────────
def test_rung_fractions_strictly_increase_and_end_at_one():
    fr = rung_fractions([5, 10, 15, 22], 22)
    assert len(fr) == 4
    assert fr == sorted(fr) and len(set(fr)) == 4
    assert fr[-1] == 1.0
    assert 0.0 < fr[0] < 0.5          # a 5 m rung is not a rounding error


@pytest.mark.parametrize("direction", [1, -1])
def test_ladder_is_monotone_inside_the_span(direction):
    entry, tp = 65536.0, 66099.29
    if direction < 0:
        tp = 64972.71
    rows = build_ladder(entry, direction, tp, [5, 10, 15, 22])
    assert [r["minutes"] for r in rows] == [5, 10, 15, 22]
    assert ladder_valid(entry, direction, tp, rows)
    levels = [r["level"] for r in rows]
    if direction > 0:
        assert all(a < b for a, b in zip(levels, levels[1:]))
        assert entry < levels[0] < levels[-1] == pytest.approx(tp, rel=1e-9)
    else:
        assert all(a > b for a, b in zip(levels, levels[1:]))
        assert entry > levels[0] > levels[-1] == pytest.approx(tp, rel=1e-9)
    # no rung may be a token distance from the entry (the "$4 TP1" bug)
    assert min(abs(lv - entry) for lv in levels) > 0.5


def test_ladder_rejects_levels_on_the_wrong_side():
    assert build_ladder(100.0, 1, 99.0, [5, 10]) == []
    assert build_ladder(100.0, -1, 101.0, [5, 10]) == []


def test_canonical_contract_round_trip_and_tamper():
    c = build_contract(symbol="BTC", timeframe="tf1", direction=1,
                       entry=65536.10, horizon_tp=66099.29, sl=65069.90,
                       minutes=[5, 10, 15, 22], atr=22.9291,
                       base_confidence=0.45, final_confidence=0.39,
                       cci_adjusted=0.031, cycle_id="cyc-1", locked_at=1e9)
    assert c and validate(c)
    assert c["tp1"] and c["tp2"] and c["tp3"]           # no "— OPEN"
    assert c["tp1"] < c["tp2"] < c["tp3"] < c["horizon_tp"]
    assert c["sl"] < c["entry"]
    bad = dict(c)
    bad["tp2"] = c["tp2"] + 1.0
    assert not validate(bad)                             # digest catches it


def test_canonical_contract_refuses_impossible_inputs():
    assert build_contract(symbol="BTC", timeframe="tf1", direction=1,
                          entry=100.0, horizon_tp=101.0, sl=101.5,
                          minutes=[1, 2]) == {}
    assert build_contract(symbol="BTC", timeframe="tf1", direction=1,
                          entry=0.0, horizon_tp=101.0, sl=99.0,
                          minutes=[1, 2]) == {}


def test_divergences_names_the_panels_that_disagree():
    c = build_contract(symbol="BTC", timeframe="tf1", direction=1,
                       entry=65536.10, horizon_tp=66099.29, sl=65069.90,
                       minutes=[5, 10, 15, 22])
    panels = {
        "analysing": {"entry": 65550.0, "tp": 66113.0, "sl": 65084.0},
        "dash_card": {"entry": 65536.10, "tp": 66099.29, "sl": 65069.90},
    }
    out = divergences(panels, c)
    assert "entry" in out["analysing"] and "tp" in out["analysing"]
    assert out["dash_card"] == []


# ── §2 calibrated confidence ───────────────────────────────────────────
def _pack(cci: float, drhe: float = 1.0) -> dict:
    return {"cci_adjusted": cci, "drhe": drhe, "confidence": abs(cci)}


def test_a_real_setup_can_clear_the_floor():
    """The exact screenshot chain: |CCI| 0.0218 × DRHE 1.00, EQ 0.57,
    RTP 0.44, PCV −0.881, WFCP 1.00, DRPT 0.64, GSPI 0.03.

    The old product was 0.0002 against a 0.35 floor — permanently
    NO TRADE. The calibrated value must be able to issue.
    """
    out = calibrate_confidence(
        _pack(0.0218), experience_quality=0.57, p_dominant=0.3,
        state_aligned=True, rtp=0.44, pcv=-0.881,
        wfcp=1.0, drpt=0.64, gspi=0.03)
    assert out["confidence"] > 0.35
    assert out["confidence"] > 100 * 0.0002


def test_noise_floor_signal_stays_watch_only():
    weak = calibrate_confidence(_pack(NOISE_FLOOR * 0.1),
                                experience_quality=0.5, rtp=0.0, pcv=0.0)
    assert weak["confidence"] < 0.35
    assert weak["issued"] is False


def test_confidence_is_monotone_in_the_signal():
    confs = [calibrate_confidence(_pack(c),
                                  experience_quality=0.5, rtp=0.2,
                                  pcv=-0.1)["confidence"]
             for c in (0.002, 0.01, 0.02, 0.04, 0.08)]
    assert confs == sorted(confs)


def test_worst_case_dampers_cannot_zero_a_signal():
    """Every penalty at its maximum: the modulator is bounded, so the
    result keeps a usable residue instead of collapsing to 0.0002."""
    out = calibrate_confidence(_pack(0.5), experience_quality=0.0,
                               p_dominant=0.0, state_aligned=False,
                               rtp=5.0, pcv=-5.0, wfcp=0.0, drpt=5.0,
                               gspi=5.0)
    assert out["modulator"] >= 0.72
    assert out["confidence"] >= 0.35        # a 25-noise-unit signal still issues
    assert out["confidence"] <= 0.98


def test_chain_steps_are_published_for_the_ui():
    out = calibrate_confidence(_pack(0.03), experience_quality=0.6,
                               rtp=0.7, pcv=-0.2, wfcp=0.9, drpt=0.1)
    steps = out["steps"]
    assert [s["step"] for s in steps] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert steps[-1]["value"] == pytest.approx(out["confidence"], abs=1e-4)
    assert out["base_confidence"] > 0.4


# ── §1.3 window freeze + wiring guards ─────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_engine_uses_the_calibrated_confidence():
    src = (ROOT / "app" / "engine.py").read_text()
    assert "calibrate_confidence(" in src
    assert 'fin["confidence"] = _cal["confidence"]' in src
    # the percentile stays as a diagnostic, never as the displayed number
    assert 'fin["conf_percentile"]' in src
    assert 'fin["confidence"] = round(_pct, 4)' not in src


def test_engine_freezes_a_locked_window():
    src = (ROOT / "app" / "engine.py").read_text()
    assert "RELOCK_MIN_MOVE_ATR = 3.0" in src
    assert "RELOCK_MIN_AGE_S = 15 * 60.0" in src
    assert '"frozen": True' in src


def test_engine_builds_the_canonical_contract():
    src = (ROOT / "app" / "engine.py").read_text()
    assert "build_contract as _build_contract" in src
    assert "cyc.canonical = _build_contract(" in src
    models = (ROOT / "app" / "models.py").read_text()
    assert '"canonical": self.canonical' in models
