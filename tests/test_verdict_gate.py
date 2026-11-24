"""The badge must never contradict the number printed next to it, and the
trend-interaction gate must only ever be able to WITHHOLD a call.

Reported defect this file pins (from the live board): a BTC tf4 card read
``confidence 0.009`` · ``score_floor_ratio 2.132`` · ``tier half`` ·
``verdict "ISSUE"`` · ``issued true`` at the same time. Three separate
derivations of one badge, and a cycle that never re-derived it, produced
an "ISSUED" card at 2.3 % of the asset's issue floor.
"""
from __future__ import annotations

import inspect

import pytest

from app import config
from app import engine as eng_mod
from app import models
from app import trend_gate as tg


@pytest.fixture(autouse=True)
def _clean_gate(monkeypatch):
    monkeypatch.delenv("TREND_GATE", raising=False)
    tg.reset()
    yield
    tg.reset()


# ---------------------------------------------------------------- the badge

def test_badge_is_derived_from_the_number_beside_it():
    floor = config.asset_issue_floor("BTC")
    ok, verdict, note = eng_mod.public_verdict("BTC", "tf4", 1, 0.009, "half")
    assert ok is False and verdict == "NO CALL"
    assert f"{floor:.3f}" in note

    ok, verdict, _ = eng_mod.public_verdict("BTC", "tf4", 1, floor + 0.001, "half")
    assert ok is True and verdict == "ISSUE"


def test_every_asset_floor_is_respected_not_the_flat_one():
    # BTC .385 / ETH .4042 / BNB .3657 / PAXG .4042 in this build; a confidence
    # between the flat floor and the asset floor must NOT be an ISSUE.
    for sym in ("BTC", "ETH", "BNB", "PAXG"):
        floor = config.asset_issue_floor(sym)
        assert floor > config.CONFIDENCE_FLOOR
        mid = (config.CONFIDENCE_FLOOR + floor) / 2
        ok, verdict, _ = eng_mod.public_verdict(sym, "tf1", 1, mid, "full")
        assert ok is False and verdict == "NO CALL", (sym, mid, floor)


def test_watch_only_never_issues():
    floor = config.asset_issue_floor("ETH")
    ok, verdict, _ = eng_mod.public_verdict("ETH", "tf2", 1, floor + 0.02,
                                            "watch-only")
    assert (ok, verdict) == (True, "WATCH")


def test_flat_side_is_never_issued():
    for conf in (0.0, 0.9, 5.0):
        ok, verdict, _ = eng_mod.public_verdict("BNB", "tf3", 0, conf, "full")
        assert ok is False and verdict == "NO CALL"


def test_verdict_has_a_single_derivation_site():
    """The rule lives in exactly one place and every path goes through it."""
    rule = inspect.getsource(__import__("app.verdict", fromlist=["x"]))
    assert rule.count('verdict = "ISSUE"') == 1
    eng = inspect.getsource(eng_mod)
    # the engine must not re-derive the badge itself any more
    assert 'verdict = "ISSUE"' not in eng
    assert eng.count("public_verdict(") >= 3
    # and the published card is reconciled against its published number
    assert "from .verdict import reconcile" in eng


def test_reconcile_only_ever_downgrades():
    from app import verdict as vd
    floor = config.asset_issue_floor("BTC")
    # stored ISSUE whose confidence has since been cut below the floor
    ok, verdict, note = vd.reconcile("BTC", "tf4", 1, 0.009, "half",
                                     True, "ISSUE", "cleared the 0.385 floor")
    assert (ok, verdict) == (False, "NO CALL") and "0.385" in note
    # stored NO CALL with a recovering confidence is NOT upgraded here
    ok, verdict, _ = vd.reconcile("BTC", "tf1", 1, floor + 0.05, "full",
                                  False, "NO CALL", "")
    assert (ok, verdict) == (False, "NO CALL")
    # a consistent card passes through untouched
    ok, verdict, note = vd.reconcile("BTC", "tf1", 1, floor + 0.05, "full",
                                     True, "ISSUE", "cleared the 0.385 floor")
    assert (ok, verdict, note) == (True, "ISSUE", "cleared the 0.385 floor")


def test_preview_card_uses_the_same_rule_as_a_locked_card():
    src = inspect.getsource(eng_mod.PredictionEngine._provisional_cycle_dict)
    assert "from .verdict import describe as vd_describe" in src
    assert '"WATCH" if res.get("gate_ok") else "NO CALL"' not in src


# ------------------------------------------------------- refresh consistency

def test_refresh_rederives_badge_from_the_refreshed_confidence():
    """A cycle refreshed at GRC must not keep the badge of its first call."""
    src = inspect.getsource(eng_mod.PredictionEngine._apply_grc_lock_rules)
    assert "public_verdict(" in src
    assert src.index("public_verdict(") > src.index('cyc.confidence = res["confidence"]')
    assert "_tg.observe" in src


# ------------------------------------------------------------- trend gate

def test_gate_needs_evidence_before_it_blocks():
    assert tg.check("BTC", "tf1", 1)["blocked"] is False
    tg.observe("BTC", "tf1", 100.0)
    tg.observe("BTC", "tf1", 101.0)
    r = tg.check("BTC", "tf1", 1)
    assert r["blocked"] is False and "no trend reading" in r["why"]


def test_gate_withholds_the_chase_and_keeps_the_fade():
    for c in (100.0, 101.0, 102.0):
        tg.observe("BTC", "tf1", c)
    assert tg.trend("BTC", "tf1") == 1
    chase = tg.check("BTC", "tf1", 1)
    fade = tg.check("BTC", "tf1", -1)
    assert chase["blocked"] is True and fade["blocked"] is False
    # both branches carry their measurement and sample size
    for r in (chase, fade):
        assert r["measured"]["opposes_n"] > 0 and r["measured"]["agrees_n"] > 0
        assert 0.4 < r["measured"]["agrees_wr"] < r["measured"]["opposes_wr"] < 0.7


def test_gate_never_flips_a_side_and_never_invents_a_reading():
    for c in (102.0, 101.0, 100.0):
        tg.observe("ETH", "tf3", c)
    for side in (-1, 0, 1):
        before = side
        tg.check("ETH", "tf3", side)
        assert side == before
    tg.reset()
    for _ in range(3):
        tg.observe("ETH", "tf3", 100.0)      # perfectly flat tape
    assert tg.trend("ETH", "tf3") == 0
    assert tg.check("ETH", "tf3", 1)["blocked"] is False


def test_gate_is_off_by_switch(monkeypatch):
    for c in (100.0, 101.0, 102.0):
        tg.observe("BTC", "tf1", c)
    assert tg.check("BTC", "tf1", 1)["blocked"] is True
    monkeypatch.setenv("TREND_GATE", "off")
    r = tg.check("BTC", "tf1", 1)
    assert r["blocked"] is False and r["enabled"] is False


def test_gate_ignores_nonpositive_and_garbage_prices():
    tg.observe("BTC", "tf1", 0.0)
    tg.observe("BTC", "tf1", -5.0)
    tg.observe("BTC", "tf1", "nope")           # type: ignore[arg-type]
    assert tg.stats().get("BTC|tf1", 0) == 0


def test_gate_is_bounded_memory():
    for i in range(500):
        tg.observe("PAXG", "tf4", 1000.0 + i)
    assert tg.stats()["PAXG|tf4"] == tg.TREND_LAG_ROWS + 1


def test_blocked_gate_forces_no_call_through_the_one_rule():
    floor = config.asset_issue_floor("BTC")
    for c in (100.0, 101.0, 102.0):
        tg.observe("BTC", "tf4", c)
    ok, verdict, note = eng_mod.public_verdict("BTC", "tf4", 1, floor + 0.05,
                                               "full")
    assert ok is False and verdict == "NO CALL"
    assert "chases" in note
    # the opposite side at the same moment still stands
    ok2, verdict2, _ = eng_mod.public_verdict("BTC", "tf4", -1, floor + 0.05,
                                              "full")
    assert verdict2 == "ISSUE"


# ------------------------------------------------------------ payload shape

def test_cycle_publishes_gate_evidence_and_the_note():
    cycle = models.TFCycle(symbol="BTC", tf="tf1", W=22, direction=1,
                           confidence=0.5, cci=1.0, drhe=1.0, issued=True,
                           verdict="ISSUE", gate_ok=True,
                           issue_note="cleared the 0.385 floor",
                           trend_gate={"blocked": False, "trend": 1})
    d = cycle.to_dict(None)
    assert d["issue_note"] == "cleared the 0.385 floor"
    assert d["trend_gate"]["trend"] == 1
    assert d["verdict"] == "ISSUE" and d["gate_ok"] is True


def test_ui_reads_the_gate_evidence_not_a_bare_blocked_flag():
    src = open("static/app.js", encoding="utf-8").read()
    assert "_trendGateRow" in src
    assert "trend_gate" in src
    # the panel must show both measured branches and their n
    assert "agrees_wr" in src and "opposes_n" in src


# ------------------------------------------------------- audit reproducibility

def test_audit_reproduces_the_numbers_the_gate_ships():
    """The constants in app/trend_gate.py must match the real record."""
    from backtest import trend_gate_audit as aud
    import os
    if not os.path.exists(aud.CSV_PATH):        # pragma: no cover - offline
        pytest.skip("real per-cell record not checked out")
    rep = aud.build()
    p = rep["pooled"]
    assert p["opp_n"] == tg.MEASURED_N_OPPOSES
    assert p["agree_n"] == tg.MEASURED_N_AGREES
    assert abs(p["opp_wr"] - tg.MEASURED_WR_OPPOSES) < 0.001
    assert abs(p["agree_wr"] - tg.MEASURED_WR_AGREES) < 0.001
    assert abs(rep["baseline_wr"] - tg.MEASURED_WR_ALL) < 0.001
    # the lever is only real if it points the right way with disjoint CIs
    assert p["opp_ci"][0] > p["agree_ci"][1]
    # and the honest caveat is still honest: holdout is not proof
    assert rep["split"]["hold_wr"] > rep["baseline_wr"]


def test_note_states_every_reason_a_call_was_withheld():
    from app import verdict as vd
    for c in (100.0, 101.0, 102.0):
        tg.observe("BTC", "tf4", c)
    ok, verdict, note = vd.public_verdict("BTC", "tf4", 1, 0.10, "full")
    assert verdict == "NO CALL"
    assert "chases" in note and "below floor" in note
    # a call that clears the floor but chases is withheld for ONE reason
    ok, verdict, note = vd.public_verdict("BTC", "tf4", 1, 0.90, "full")
    assert verdict == "NO CALL" and "chases" in note and "floor" not in note


def test_rejected_confidence_filter_is_documented_not_shipped():
    """Record the negative result so the failing variant is not re-invented."""
    import os
    from backtest import trend_gate_audit as aud
    if not os.path.exists(aud.CSV_PATH):        # pragma: no cover - offline
        pytest.skip("real per-cell record not checked out")
    import csv
    from collections import defaultdict
    rows = list(csv.DictReader(open(aud.CSV_PATH, encoding="utf-8")))
    by = defaultdict(list)
    for r in rows:
        by[(r["asset"], r["tf"])].append(r)
    recs = []
    for seq in by.values():
        seq.sort(key=lambda r: float(r["t"]))
        for k in range(aud.LAG, len(seq)):
            mv = float(seq[k]["entry"]) - float(seq[k - aud.LAG]["entry"])
            if mv == 0:
                continue
            trend = 1 if mv > 0 else -1
            recs.append((trend == int(seq[k]["direction"]),
                         float(seq[k]["conf"]), int(seq[k]["win"]),
                         float(seq[k]["t"])))
    # TEMPORAL order: a holdout must be later data, never a re-sorted sample
    recs.sort(key=lambda r: r[3])
    # the variant: withhold only high-confidence chases
    keep = [r for r in recs if not (r[0] and r[1] >= 0.60)]
    cut = int(len(keep) * 0.65)
    hold = keep[cut:]
    wr = sum(r[2] for r in hold) / len(hold)
    # it must NOT look like a proven edge: inside noise of the baseline
    assert wr < 0.56
    # while the shipped rule keeps its gap
    from backtest import trend_gate_audit as a2
    rep = a2.build()
    assert rep["pooled"]["opp_wr"] - rep["pooled"]["agree_wr"] > 0.10
