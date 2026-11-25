"""Pins the four "always-on warning" bugs reported from the live app.

Each of these was a state the UI painted on essentially every open, which by
definition carries no information. A caution flag must be rare enough that
seeing it means something.
"""
from __future__ import annotations

import random

import pytest

from app.cost_gate import gate
from app.regime import classify_tape


def _random_walk(n=240, vol=0.002, drift=0.0, seed=0):
    r = random.Random(seed)
    px = 50000.0
    out = []
    for _ in range(n):
        px *= 1 + drift + r.gauss(0, vol)
        out.append(px)
    return out


# ── 1. the regime fallback must not hardcode the caution state ───────
def test_unremarkable_tape_is_not_labelled_mean_reverting():
    """The final fallback used to return MEAN_REVERTING_RANGE unconditionally,
    which the UI renders as the yellow 'SIDEWAYS' chip."""
    r = classify_tape(_random_walk(240, 0.002, 0.0, seed=1))
    assert r["state"] != "MEAN_REVERTING_RANGE" or r.get("confidence", 0) > 0.5


def test_unremarkable_tape_is_flagged_neutral_and_low_confidence():
    r = classify_tape(_random_walk(240, 0.002, 0.0, seed=7))
    assert r.get("neutral") is True
    assert r["confidence"] <= 0.15


def test_caution_states_are_rare_on_pure_noise():
    """A warning that fires on most tapes is noise. On 200 random walks the
    caution state must be uncommon (it was ~100% before the fix)."""
    states = [classify_tape(_random_walk(240, 0.002, 0.0, seed=s))["state"]
              for s in range(200)]
    caution = sum(1 for s in states if s in ("MEAN_REVERTING_RANGE",
                                            "CHAOTIC_LIQUIDATION_SHOCK"))
    assert caution / len(states) < 0.25, f"caution fired on {caution}/200"


def test_a_real_range_still_gets_labelled_a_range():
    """The fix must not break genuine detection: an oscillating tape is a
    range and must still be named one."""
    px, osc = 50000.0, []
    for i in range(240):
        px = 50000.0 * (1 + 0.004 * (1 if i % 2 else -1))
        osc.append(px)
    r = classify_tape(osc)
    assert r["state"] in ("MEAN_REVERTING_RANGE", "CHAOTIC_LIQUIDATION_SHOCK")


# ── 2. cost gate: no measurement is not a failed measurement ─────────
def test_cold_start_cost_gate_is_unmeasured_not_blocked():
    r = gate("BTC", "tf1", 28.9)
    assert r["state"] == "unmeasured"
    assert r["expected_net_bp"] is None
    # must not ASSERT coverage either way - it has to say it cannot tell
    note = (r.get("note") or "").lower()
    assert "cannot say" in note
    assert "not covered" not in note


def test_unmeasured_note_does_not_claim_fee_not_covered():
    r = gate("ETH", "tf2", 30.0)
    assert "not covered" not in (r.get("note") or "").lower()
    assert r["evidence"] == "none"


def test_measured_lane_still_gets_a_measured_verdict():
    """The new early-return must not swallow real measurements."""
    from app.cost_gate import measured
    found = None
    for asset in ("BTC", "ETH", "BNB", "SOL", "PAXG"):
        for tf in ("tf1", "tf2", "tf3", "tf4"):
            if measured(asset, tf)["n"] > 0:
                found = (asset, tf)
                break
        if found:
            break
    if not found:
        pytest.skip("no registered measurement to exercise")
    r = gate(found[0], found[1], 8.0)
    assert r["state"] in ("open", "thin", "blocked", "unproven")


# ── 3. BGN must not claim perfect coherence with no evidence ─────────
def test_bgn_bootstrap_is_not_perfect_coherence():
    from app.bgn import BgnSynchronizer
    from app.history import HistoryStore

    hist = HistoryStore()
    sync = BgnSynchronizer(hist)
    c = sync._coherence("BTC", "tf1", "tf2")
    assert c != 1.0, "bootstrap must not assert perfect cross-TF agreement"
    assert c == pytest.approx(0.5)


def test_bgn_reports_unmeasured_pairs():
    from app.bgn import BgnSynchronizer
    from app.history import HistoryStore

    sync = BgnSynchronizer(HistoryStore())
    assert sync.status()["measured"] is True     # nothing evaluated yet
    sync._coherence("BTC", "tf1", "tf2")
    st = sync.status()
    assert st["unmeasured_pairs"] >= 1
    assert st["measured"] is False
    assert "unmeasured" in st["note"]
