"""Contracts for the cost gate.

The gate is the platform's only tradeability claim, so the pins are:
the verdict follows the measured out-of-fold registry (never a live fit),
insufficient evidence can never be called tradable, and the break-even
arithmetic matches the barrier algebra in the docs.
"""
from __future__ import annotations

import pytest

from app import edge_gate
from app.edge_gate import (MIN_MARGIN_BP, MEASURED, breakeven_cost_bps,
                           bucket, gate, measured, required_accuracy,
                           required_gross_bp)


def test_horizon_buckets_are_contiguous_and_ordered():
    assert bucket(0) == "intraday"
    assert bucket(60) == "intraday"
    assert bucket(89.9) == "intraday"
    assert bucket(90) == "session"
    assert bucket(719) == "session"
    assert bucket(720) == "swing"
    assert bucket(2879) == "swing"
    assert bucket(2880) == "position"
    assert bucket(1e9) == "position"


def test_unmeasured_lane_is_zero_gross_and_blocked():
    m = measured("DOGE", "intraday")
    assert m["gross_bp"] == 0.0 and m["n"] == 0
    g = gate("DOGE", "intraday", 12.0)
    assert g["state"] == "blocked"
    assert g["evidence"] == "insufficient"
    assert "no oof measurement" in g["note"].lower()


def test_gate_state_follows_measured_net_expectancy(monkeypatch):
    monkeypatch.setitem(MEASURED, ("TEST", "intraday"),
                        {"gross_bp": 20.0, "n": 500, "t_stat": 4.0,
                         "source": "unit"})
    assert gate("TEST", "intraday", 12.0)["state"] == "open"
    # positive but inside the margin -> thin, not open
    monkeypatch.setitem(MEASURED, ("TEST", "session"),
                        {"gross_bp": 13.0, "n": 500, "t_stat": 4.0,
                         "source": "unit"})
    assert gate("TEST", "session", 12.0)["state"] == "thin"
    # fee above the edge -> blocked, and the shortfall is reported
    monkeypatch.setitem(MEASURED, ("TEST", "swing"),
                        {"gross_bp": 5.0, "n": 500, "t_stat": 4.0,
                         "source": "unit"})
    g = gate("TEST", "swing", 12.0)
    assert g["state"] == "blocked"
    assert g["expected_net_bp"] == pytest.approx(-7.0)
    assert g["required_gross_bp"] == pytest.approx(12.0 + MIN_MARGIN_BP)


def test_insufficient_evidence_is_never_tradable():
    """A big nominal gross with |t| < 2 must not be sold as tradable."""
    fake = {"gross_bp": 40.0, "n": 60, "t_stat": 0.4, "source": "unit"}
    edge_gate.MEASURED[("WEAK", "intraday")] = fake
    try:
        g = gate("WEAK", "intraday", 12.0)
        assert g["evidence"] == "insufficient"
        assert "insufficient" in g["note"]
    finally:
        edge_gate.MEASURED.pop(("WEAK", "intraday"), None)


def test_every_registry_lane_reports_its_evidence():
    for (asset, tf), m in MEASURED.items():
        g = gate(asset, tf, 12.0)
        assert g["asset"] == asset and g["tf"] == tf
        assert g["gross_bp"] == pytest.approx(m["gross_bp"])
        assert g["state"] in ("open", "thin", "blocked")
        assert g["note"], "a verdict without a reason is not a verdict"
        if m["n"] and abs(m["t_stat"]) < 2.0:
            assert g["evidence"] == "insufficient"


def test_no_live_lane_is_called_tradable_at_current_costs():
    """The 2026-11 falsification left no lane with gross > friction."""
    live = [("BTC", "intraday", 12.0), ("ETH", "intraday", 12.0),
            ("BNB", "intraday", 12.0), ("PAXG", "intraday", 18.0),
            ("BTC", "session", 12.0), ("ETH", "session", 12.0),
            ("BNB", "session", 12.0), ("PAXG", "session", 18.0),
            ("BTC", "swing", 12.0), ("ETH", "swing", 12.0),
            ("BNB", "swing", 12.0)]
    for asset, tf, fee in live:
        g = gate(asset, tf, fee)
        assert g["state"] == "blocked", (asset, tf, g)


def test_breakeven_cost_is_the_measured_gross():
    assert breakeven_cost_bps(7.7) == pytest.approx(7.7)
    assert breakeven_cost_bps(-2.4) == pytest.approx(-2.4)


def test_required_accuracy_matches_barrier_algebra():
    # ±2σ barriers, 12 bp friction, σ = 100 bp -> 0.5 + 12/(2*2*100)
    assert required_accuracy(12.0, 100.0, 2.0) == pytest.approx(0.53)
    assert required_accuracy(12.0, 440.0, 2.0) == pytest.approx(0.5 + 12 / 1760)
    # wider barriers and cheaper fees both lower the bar
    assert required_accuracy(6.0, 100.0, 2.0) < required_accuracy(12.0, 100.0, 2.0)
    assert required_accuracy(12.0, 200.0, 2.0) < required_accuracy(12.0, 100.0, 2.0)
    assert required_accuracy(12.0, 0.0, 2.0) == 1.0


def test_required_gross_defaults_to_friction_plus_margin():
    assert required_gross_bp(12.0) == pytest.approx(12.0 + MIN_MARGIN_BP)
    assert required_gross_bp(18.0, 5.0) == pytest.approx(23.0)
