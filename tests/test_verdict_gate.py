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
    for sym in ("BTC", "ETH", "BNB", "SOL"):
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
    assert (ok, verdict) == (False, "NO CALL") and "floor" in note
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


# ------------------------------------------------------------- trend gate

def test_rule_needs_evidence_before_it_acts():
    assert tg.check("BTC", "tf1", 1)["flipped"] is False
    assert tg.check("BTC", "tf1", 1)["blocked"] is False
    tg.observe("BTC", "tf1", 100.0)
    tg.observe("BTC", "tf1", 101.0)
    r = tg.check("BTC", "tf1", 1)
    assert r["flipped"] is False and r["blocked"] is False
    assert "no trend reading" in r["why"]


def test_rule_flips_the_chase_and_keeps_the_fade():
    """WIN RATE WITHOUT ABSTENTION (2026-11-26): a chase is no longer
    withheld — the side is REVERSED, so the window still ships a call."""
    for c in (100.0, 101.0, 102.0):
        tg.observe("BTC", "tf1", c)
    assert tg.trend("BTC", "tf1") == 1
    chase = tg.check("BTC", "tf1", 1)
    fade = tg.check("BTC", "tf1", -1)
    assert chase["flipped"] is True and chase["side_out"] == -1
    assert chase["blocked"] is False            # never an abstention
    assert fade["flipped"] is False and fade["side_out"] == -1
    assert fade["blocked"] is False
    # both branches carry their measurement and sample size
    for r in (chase, fade):
        assert r["measured"]["opposes_n"] > 0 and r["measured"]["agrees_n"] > 0
        assert 0.4 < r["measured"]["agrees_wr"] < r["measured"]["opposes_wr"] < 0.7
        # the flipped policy's own measured record travels with the decision
        assert r["measured"]["flip_n"] > 0
        assert r["measured"]["flip_wr"] > r["measured"]["baseline_wr"]


def test_rule_never_mutates_the_callers_side_and_never_invents_a_reading():
    for c in (102.0, 101.0, 100.0):
        tg.observe("ETH", "tf3", c)
    for side in (-1, 0, 1):
        before = side
        out = tg.check("ETH", "tf3", side)
        assert side == before               # caller's int is untouched
        assert out["side_out"] in (-1, 1) or side == 0
    tg.reset()
    for _ in range(3):
        tg.observe("ETH", "tf3", 100.0)      # perfectly flat tape
    assert tg.trend("ETH", "tf3") == 0
    out = tg.check("ETH", "tf3", 1)
    assert out["flipped"] is False and out["blocked"] is False


def test_noise_filter_is_disarmed_because_it_was_measured_and_failed():
    """The tempting "only fade moves bigger than tick noise" filter was
    audited and REJECTED: every level of it lowers both coverage and win
    rate (0.5σ → 55.9 % on 74 % of windows vs 56.8 % on 100 %). It must
    therefore be off by default, and a big σ must not silently disarm the
    rule — that would be the abstention the user forbade, wearing a filter's
    name. The knob survives only for a future A/B."""
    assert tg.MIN_TREND_FRACTION == 0.0
    for c in (100.0, 100.02, 100.04):       # +4 bps over two decisions
        tg.observe("BTC", "tf4", c)
    tg.set_window_sigma("BTC", "tf4", 60.0)     # 60 bps σ: would be "noise"
    out = tg.check("BTC", "tf4", 1)
    assert out["trend"] == 1
    # disarmed → it acts anyway, and says the filter did not decide this
    assert out["flipped"] is True and out["side_out"] == -1
    assert out.get("below_noise") is not True


def test_noise_filter_knob_still_works_when_armed(monkeypatch):
    """Armed by hand, the filter must abstain from ACTING (never from
    publishing) — so it can be A/B'd on fresh live tape without a code edit."""
    import importlib
    monkeypatch.setenv("TREND_GATE_MIN_FRACTION", "0.5")
    try:
        g = importlib.reload(tg)
        for c in (100.0, 100.02, 100.04):
            g.observe("BTC", "tf4", c)
        g.set_window_sigma("BTC", "tf4", 60.0)
        out = g.check("BTC", "tf4", 1)
        assert out["flipped"] is False and out["blocked"] is False
        assert out["below_noise"] is True and "inside the noise" in out["why"]
        # the call still ships a side: no abstention, ever
        assert out["side_out"] == 1
        # on a quiet cell the same move IS acted on
        g.set_window_sigma("BTC", "tf4", 2.0)
        assert g.check("BTC", "tf4", 1)["flipped"] is True
    finally:
        monkeypatch.delenv("TREND_GATE_MIN_FRACTION", raising=False)
        importlib.reload(tg)


def test_abstain_mode_is_still_available_for_ab():
    import os
    for c in (100.0, 101.0, 102.0):
        tg.observe("SOL", "tf1", c)
    os.environ["TREND_GATE"] = "abstain"
    try:
        out = tg.check("SOL", "tf1", 1)
        assert out["blocked"] is True and out["flipped"] is False
        assert out["mode"] == "abstain"
        # the same call under the default mode flips instead
        os.environ.pop("TREND_GATE")
        assert tg.check("SOL", "tf1", 1)["flipped"] is True
    finally:
        os.environ.pop("TREND_GATE", None)


def test_rule_is_off_by_switch(monkeypatch):
    for c in (100.0, 101.0, 102.0):
        tg.observe("BTC", "tf1", c)
    assert tg.check("BTC", "tf1", 1)["flipped"] is True
    monkeypatch.setenv("TREND_GATE", "off")
    r = tg.check("BTC", "tf1", 1)
    assert r["flipped"] is False and r["blocked"] is False
    assert r["enabled"] is False and r["side_out"] == 1


def test_gate_ignores_nonpositive_and_garbage_prices():
    tg.observe("BTC", "tf1", 0.0)
    tg.observe("BTC", "tf1", -5.0)
    tg.observe("BTC", "tf1", "nope")           # type: ignore[arg-type]
    assert tg.stats().get("BTC|tf1", 0) == 0


def test_gate_is_bounded_memory():
    for i in range(500):
        tg.observe("SOL", "tf4", 1000.0 + i)
    assert tg.stats()["SOL|tf4"] == tg.TREND_LAG_ROWS + 1


def test_flip_reaches_the_badge_and_the_side():
    """One rule, one flip: the badge rule now also REPORTS the resolved
    side, so a card cannot print a side its own note just reversed."""
    from app.verdict import resolve
    floor = config.asset_issue_floor("BTC")
    for c in (100.0, 101.0, 102.0):
        tg.observe("BTC", "tf4", c)
    ok, verdict, note, side = resolve("BTC", "tf4", 1, floor + 0.05, "full")
    assert side == -1                     # the flip
    assert verdict == "ISSUE" and ok is True
    assert "FLIPPED" in note
    # the opposite side at the same moment is already correct: unchanged
    ok2, verdict2, _, side2 = resolve("BTC", "tf4", -1, floor + 0.05, "full")
    assert verdict2 == "ISSUE" and side2 == -1
    # and the legacy three-tuple is unchanged for every existing caller
    assert eng_mod.public_verdict("BTC", "tf4", 1, floor + 0.05,
                                  "full") == (ok, verdict, note)


def test_the_flip_is_fully_documented_in_the_payload():
    for c in (100.0, 101.0, 102.0):
        tg.observe("BTC", "tf2", c)
    r = tg.check("BTC", "tf2", 1)
    m = r["measured"]
    for key in ("flip_n", "flip_wr", "flip_ci", "flip_folds", "flip_net_bps",
                "baseline_wr", "agrees_wr", "opposes_wr"):
        assert key in m, key
    # the stated folds are the measured ones, including the losing fold
    assert min(m["flip_folds"]) < 0.45
    assert sum(1 for f in m["flip_folds"] if f > 0.5) == 4


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


def test_note_states_every_reason_a_call_was_held_back():
    from app import verdict as vd
    for c in (100.0, 101.0, 102.0):
        tg.observe("BTC", "tf4", c)
    # below the floor AND flipped: say both
    ok, verdict, note = vd.public_verdict("BTC", "tf4", 1, 0.10, "full")
    assert verdict == "NO CALL"
    assert "chased" in note and "below floor" in note
    # clears the floor and is flipped: the flip is the whole story
    ok, verdict, note = vd.public_verdict("BTC", "tf4", 1, 0.90, "full")
    assert verdict == "ISSUE" and "FLIPPED" in note and "floor" not in note


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
    # and the module documents the rejection so it is not re-invented
    src = open("app/trend_gate.py", encoding="utf-8").read()
    assert "tested and REJECTED" in src


def test_note_prints_the_same_number_the_card_prints():
    """The note and the card must round confidence identically."""
    from app import verdict as vd
    for conf in (0.2805, 0.0164, 0.3849, 0.3851, 0.9):
        ok, _, note = vd.public_verdict("BTC", "tf1", 1, conf, "full")
        # a CLEARED card prints the floor it cleared; a withheld card prints
        # its own confidence. Both are 2-dp, matching the card's confidence.
        expected = (f"{config.asset_issue_floor('BTC'):.3f}" if ok
                    else f"{conf:.3f}")
        assert expected in note, (conf, note)
        # and the verdict must follow the printed numbers' ordering
        assert ok == (conf > config.asset_issue_floor("BTC"))


# ------------------------------------------------- calibration de-duplication

def test_calibration_sample_is_booked_once_per_window():
    """Two grading paths settle the same window; it must count once."""
    from app.engine import calib_should_record
    done: dict = {}
    key = ("BTC", "tf1")
    assert calib_should_record(done, key, 0.001) is True
    # the same settled window arriving from the other path
    assert calib_should_record(done, key, 0.001) is False
    assert calib_should_record(done, key, 0.001) is False
    # a genuinely different window is a new sample
    assert calib_should_record(done, key, -0.004) is True
    # and cells are independent
    assert calib_should_record(done, ("ETH", "tf1"), 0.001) is True


def test_both_grading_paths_use_the_dedup_rule():
    eng = inspect.getsource(eng_mod)
    assert eng.count("calib_should_record(") >= 3   # def + two call sites
    # the paired rings and the lock-learning path must both be guarded
    paired = inspect.getsource(eng_mod.PredictionEngine._grade_settlement) \
        if hasattr(eng_mod.PredictionEngine, "_grade_settlement") else eng
    assert "calib_p" in paired


def test_every_confidence_mutation_re_derives_the_badge():
    """A confidence changed after the badge was written must not print a
    note naming the old number (the live ETH tf2 defect: conf 0.135 beside
    'below floor (0.08 ...)')."""
    from app import verdict as vd
    ok, verdict, note = vd.reconcile("ETH", "tf2", 1, 0.1347, "watch-only",
                                     False, "NO CALL",
                                     "below floor (0.08 <= 0.40)")
    assert "0.135" in note and "0.08" not in note
    # but an engine-declined call keeps its own explanation
    ok, verdict, note = vd.reconcile("ETH", "tf2", 1, 0.1347, "watch-only",
                                     False, "NO CALL",
                                     "double-check conflict: 3 formulas "
                                     "disagree")
    assert note.startswith("double-check conflict")
    # the two post-hoc mutation sites must re-derive, not inherit
    for name in ("_on_target_hit", "_on_sl_hit"):
        fn = inspect.getsource(getattr(eng_mod.PredictionEngine, name))
        assert "public_verdict(" in fn, name


# --------------------------------------------------- P is a confidence proxy

def test_printed_p_is_labelled_as_a_proxy_with_the_measured_p_beside_it():
    """`p_side` is the logistic of the score; the calibration panel says the
    realised hit-rate is ~0.50 in every bin. A card must not print "P 1.00"
    as if it were a win probability."""
    js = open("static/app.js", encoding="utf-8").read()
    assert "function _pTag(" in js
    assert "score-units" in js
    assert "p_correct" in js            # the measured P travels with it
    # and no bare P readout survives
    assert "` · P ${fmt(cyc.p_side, 2)}`" not in js


def test_calibration_panel_is_honest_about_the_flat_hit_rate():
    """/api/history/calibration must be able to say 'this is a coin flip'."""
    src = open("app/main.py", encoding="utf-8").read()
    seg = src[src.index('@app.get("/api/history/calibration")'):]
    seg = seg[:seg.index("@app.get", 10)]
    assert "brier" in seg and "hit_rate" in seg
    # the note must not imply the numbers rise when they do not
    assert "should RISE" in seg


def test_deployed_stack_is_reproducible_and_keeps_every_window():
    """The number the user experiences must be regenerable — and it must be
    the FLIP, not the retired withhold-only policy: 100 % coverage, no
    window silenced. (This test used to assert coverage < 0.40, i.e. it
    pinned the abstention the user explicitly forbade.)"""
    from backtest import trend_gate_audit as aud
    import os
    if not os.path.exists(aud.CSV_PATH):        # pragma: no cover - offline
        pytest.skip("real per-cell record not checked out")
    rep = aud.build()
    dep = rep["deployed"]
    assert dep["mode"] == "flip"
    # every window that has a reading gets a side — nothing is withheld
    assert dep["coverage"] > 0.95
    # the rule is better than everything-shipped on win rate...
    assert dep["wr"] > rep["baseline_wr"] + 0.03
    assert dep["ci"][0] > rep["baseline_ci"][0]
    # ...and the audit is honest that it is not a profitability gain
    assert dep["net_bps"] < rep["baseline"]["net_bps"]
    # per-asset disagreement is reported rather than hidden
    assert dep["per_asset"]["ETH"]["net_bps"] < 0
    doc = open("app/trend_gate.py", encoding="utf-8").read()
    assert "Window filter (deliberately disarmed)" in doc
    assert "net bps/row FALLS" in doc


def test_grc_refresh_does_not_repeat_the_same_price_into_the_gate():
    """A GRC re-analyses the OPEN window at the SAME price. Feeding that to the
    gate filled its 3-sample ring with [p, p, p], so every cell read 'trend 0'
    within minutes and the gate stopped withholding anything."""
    src = inspect.getsource(eng_mod.PredictionEngine._apply_grc_lock_rules)
    assert "_tg.observe" not in src and "trend_gate.observe" not in src
    # the settled-window path is the only live writer
    settle = inspect.getsource(eng_mod.PredictionEngine._settle_and_lock)
    assert "_tg.observe" in settle


def test_observe_ignores_a_repeated_price():
    """The same close arriving twice must not consume a ring slot: that is how
    the GRC path erased every reading on the live board."""
    for c in (100.0, 103.0, 106.0):
        tg.observe("BTC", "tf2", c)
    assert tg.trend("BTC", "tf2") == 1
    assert tg.stats()["BTC|tf2"] == 3
    tg.observe("BTC", "tf2", 106.0)           # same price again
    assert tg.stats()["BTC|tf2"] == 3
    assert tg.trend("BTC", "tf2") == 1        # reading survives


def test_stale_trend_note_is_regenerated_once_the_reading_is_gone():
    """Live symptom: ETH tf1 printed "chases the 2-decision trend (up) —
    withheld" while its own trend field read 0. A note that describes a rule
    decision must be re-derived like the floor note is."""
    from app import verdict as vd
    stale = ("chases the 2-decision trend (up) — measured 46.0 % on 478 "
             "windows vs 60.3 % on 383 against; call withheld")
    ok, verdict, note = vd.reconcile("ETH", "tf1", 1, 0.30, "half",
                                     False, "NO CALL", stale)
    assert "chases" not in note
    assert note == "below floor (0.300 <= 0.404)"
    # and a genuinely live reading still produces its own note
    from app import trend_gate as g
    for c in (100.0, 101.0, 102.0):
        g.observe("ETH", "tf1", c)
    ok, verdict, note = vd.public_verdict("ETH", "tf1", 1, 0.30, "half")
    assert "chased" in note and ok is False


def test_card_note_and_evidence_come_from_one_computation():
    """A card must never print 'chases the trend — withheld' beside its own
    trend_gate block saying trend 0 / not blocked (live symptom)."""
    from app import verdict as vd
    card = {
        "direction": 1, "confidence": 0.30, "tier": "half",
        "gate_ok": True, "verdict": "ISSUE", "execution_eligible": True,
        "issue_note": ("chases the 2-decision trend (up) — measured 46.0 % "
                       "… call withheld"),
        "trend_gate": {"trend": 0, "blocked": False},
    }
    out = vd.reconcile_card("BTC", "tf1", card)
    assert "chases" not in out["issue_note"]
    assert out["trend_gate"]["blocked"] is False
    assert out["verdict"] == "NO CALL" and out["gate_ok"] is False
    assert out["execution_eligible"] is False
    # and the published evidence must agree with the published note
    assert ("chases" in out["issue_note"]) == bool(
        out["trend_gate"].get("blocked"))
    src = open("app/engine.py", encoding="utf-8").read()
    assert "reconcile_card" in src


# ------------------------------------------------- rungs stay on the locked side

def test_rung_predicate_rejects_a_target_above_a_short_entry():
    from app.levels import rung_on_side
    # the exact live defect: BNB|tf1 short at 439.473416, TP1 at 439.64265
    assert rung_on_side(439.473416, -1, 439.64265) is False
    assert rung_on_side(439.473416, -1, 435.5069) is True
    assert rung_on_side(100.0, 1, 110.0) is True
    assert rung_on_side(100.0, 1, 90.0) is False
    assert rung_on_side(100.0, 0, 110.0) is False      # no side -> no target


def test_every_ladder_rung_is_between_entry_and_target():
    from app.contracts import build_ladder
    from app.levels import rung_on_side
    for entry, direction, tp, mins in (
            (439.473416, -1, 433.811284, [5, 10, 15, 22]),
            (443.26, 1, 456.5, [11, 22, 33, 44]),
            (68251.34, 1, 69334.97, [5, 10, 15, 22])):
        rows = build_ladder(entry, direction, tp, mins)
        assert rows, (entry, direction)
        for r in rows:
            assert rung_on_side(entry, direction, r["level"]), r
            assert abs(r["level"] - entry) <= abs(tp - entry) + 1e-6, r


def test_grc_never_reprojects_a_rung_on_the_live_lean():
    """A rung may move in PRICE, never in SIDE. The GRC used to project rung 1
    on res['direction'] — the live lean — which rebuilt a short's target above
    its own entry."""
    src = inspect.getsource(eng_mod.PredictionEngine._apply_grc_lock_rules)
    # strip comments so the explanatory note about the old bug cannot match
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert 'res["direction"] or t1.direction' not in code
    assert "t1.direction = res[" not in code
    assert "_side1 = int(cyc.direction or t1.direction or 0)" in code


def test_intermediate_rungs_are_priced_on_the_canonical_contract():
    """TP2/TP3 are level-less on the CYCLE by design (a rung is only tracked
    once the one before it is hit), so their price must live on the canonical
    contract — otherwise a panel prints OPEN for a level the lock knows."""
    from app.contracts import build_ladder
    entry, direction, tp, mins = 443.261675, -1, 438.992799, [11, 22, 33, 44]
    rungs = build_ladder(entry, direction, tp, mins)
    assert len(rungs) == len(mins)
    from app.levels import rung_on_side
    for r in rungs:
        assert r["level"] is not None
        assert rung_on_side(entry, direction, r["level"]), r
    # and the cycle keeps its documented shape
    src = inspect.getsource(eng_mod.PredictionEngine)
    seg = src[src.index("_rungs = _build_ladder("):]
    seg = seg[:seg.index("cyc = TFCycle(")]
    assert '_level, _status = None, "pending"' in seg
