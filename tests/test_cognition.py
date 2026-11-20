"""Cognition + metacognition: the trace must be honest, reproducible, and
able to change the answer.

The user asked for "cognitive thinking and meta cognition with thought
process". The failure mode to guard against is a thought process that looks
impressive and is not load-bearing: six bullet points, three invented numbers
and a confidence the app has never once earned. These tests make each of the
four claims the module makes falsifiable.
"""
from __future__ import annotations

import json

import pytest

from app.cognition import (CONFLICT_DIR, MIN_GRADED_CONFIDENT, STEPS,
                           brier_decompose, competence, deterministic_narration,
                           guard_narration, narration_prompt, self_check,
                           should_abstain, think, thought_signature)


def _res(**kw):
    base = {"direction": 1, "confidence": 0.42, "cci_adjusted": 0.02,
            "mass": {"for": 0.12, "against": 0.08, "conflicting": False,
                     "supporters": [{"name": "a", "contribution": 0.06}],
                     "opponents": [{"name": "b", "contribution": -0.04}],
                     "subsystem_votes": {"brain": 0.003, "news": 0.002,
                                         "physics": -0.001}},
            "brain": {"ready": True, "vote": 0.1},
            "emotion": {"narrative": "feels greed"},
            "issue": True, "issued": True}
    base.update(kw)
    return base


def _store(**kw):
    base = {"physics": {"tf1": {"ufp": 0.01}}, "news": {"niv": 0.1},
            "hedge": {"tf1": {"combined": 0.2}},
            "feed": {"simulated": False, "last_delivery_age_s": 2.0}}
    base.update(kw)
    return base


# ── 1. the trace is complete and ordered ───────────────────────────────
def test_trace_has_every_step_in_order_and_never_an_empty_one():
    out = think(symbol="BTC", tf="tf1", res=_res(), store=_store())
    assert [s["step"] for s in out["steps"]] == list(STEPS)
    for s in out["steps"]:
        assert s["detail"].strip(), f"{s['step']} concluded nothing"
        assert s["status"] in ("done", "partial", "warn", "conflict")
        assert isinstance(s["cites"], list)
    assert out["n_steps"] == 6


def test_the_trace_passes_its_own_self_check():
    out = think(symbol="BTC", tf="tf1", res=_res(), store=_store())
    assert out["consistency"]["ok"], out["consistency"]["problems"]


def test_self_check_actually_catches_a_broken_trace():
    """Remove a step and the checker must notice - otherwise it is decoration."""
    out = think(symbol="BTC", tf="tf1", res=_res(), store=_store())
    broken = dict(out)
    broken["steps"] = out["steps"][:-1]
    assert self_check(broken)["ok"] is False
    blanked = dict(out)
    blanked["steps"] = [dict(s, detail="") for s in out["steps"]]
    assert self_check(blanked)["ok"] is False


# ── 2. thought is reproducible ─────────────────────────────────────────
def test_same_evidence_gives_the_same_signature():
    a = think(symbol="BTC", tf="tf1", res=_res(), store=_store())
    b = think(symbol="BTC", tf="tf1", res=_res(), store=_store())
    assert a["signature"] == b["signature"]
    # ...and the signature must not be a function of the clock
    assert a["generated_at"] != 0 and b["generated_at"] != 0


def test_different_evidence_gives_a_different_signature():
    a = think(symbol="BTC", tf="tf1", res=_res(), store=_store())
    b = think(symbol="BTC", tf="tf1", res=_res(confidence=0.61),
              store=_store())
    assert a["signature"] != b["signature"]


def test_signature_ignores_timestamps_but_not_reasoning():
    steps = [{"step": "observe", "status": "done", "detail": "a",
              "cites": [], "ts": 1}]
    same = [dict(steps[0], ts=999)]
    assert thought_signature(steps) == thought_signature(same)
    other = [dict(steps[0], detail="b")]
    assert thought_signature(steps) != thought_signature(other)


# ── 3. working memory is a filter, not a list ──────────────────────────
def test_withheld_inputs_are_named_with_a_reason():
    out = think(symbol="BTC", tf="tf1", res=_res(brain={"ready": False}),
                store={"feed": {"simulated": True}})
    wm = out["working_memory"]
    assert wm["withheld"], "nothing reported as withheld on a bare payload"
    for w in wm["withheld"]:
        assert w["why"].strip(), f"{w['label']} withheld with no reason"
    srcs = [w["source"] for w in wm["withheld"]]
    assert "brain" in srcs
    # and what DID arrive is reported too
    assert wm["present"]
    assert 0.0 <= wm["coverage"] <= 1.0


def test_channel_abstentions_from_the_emotion_model_are_carried_through():
    em = {"zero_reasons": {"2": "fear prototype is far from this state"},
          "model": {"channels": {"2": {"name": "Fear"}}}}
    out = think(symbol="BTC", tf="tf1", res=_res(emotion=em),
                store=_store(emotions={"tf1": em}))
    labels = [w["label"] for w in out["working_memory"]["withheld"]]
    assert "Fear" in labels


# ── 4. deliberation exposes the rejected alternative ───────────────────
def test_deliberation_names_the_outcome_it_rejected():
    out = think(symbol="BTC", tf="tf1", res=_res(), store=_store())
    alt = out["deliberation"]["alternative"]
    assert alt and alt["considered"] == "SELL", alt
    assert "rejected" in alt["why_not"] or alt["why_not"].strip()


def test_route_agreement_is_about_sign_not_magnitude():
    """Two routes that agree on BUY must not be called a conflict just because
    one is a big number and the other a small one."""
    out = think(symbol="BTC", tf="tf1", res=_res(), store=_store())
    d = out["deliberation"]
    assert d["routes_agree"] is True
    assert d["split"] is False
    assert d["magnitude_gap"] is not None      # the size difference is kept


def test_a_real_split_is_reported_as_a_conflict_step():
    res = _res(direction=1,
               mass={"for": 0.02, "against": 0.30, "conflicting": True,
                     "supporters": [], "opponents": [],
                     "subsystem_votes": {"a": 0.1, "b": 0.1, "c": 0.1}})
    out = think(symbol="BTC", tf="tf1", res=res, store=_store())
    d = out["deliberation"]
    assert d["routes_agree"] is False
    assert d["split"] is True
    assert any(s["status"] == "conflict" for s in out["steps"])
    assert out["consistency"]["ok"], out["consistency"]["problems"]


def test_missing_mass_says_so_instead_of_inventing_a_route():
    out = think(symbol="BTC", tf="tf1", res={"direction": 0,
                                             "confidence": 0.0},
                store=_store())
    d = out["deliberation"]
    assert d["route_evidence"]["value"] is None
    assert "nothing to deliberate over" in d["summary"]


# ── 5. assumptions are claims with a verdict ───────────────────────────
def test_a_simulated_tape_is_flagged_as_a_broken_assumption():
    out = think(symbol="BTC", tf="tf1", res=_res(),
                store=_store(feed={"simulated": True,
                                   "last_delivery_age_s": 1.0}))
    flags = {a["claim"]: a["holds"] for a in out["assumptions"]}
    hit = [c for c in flags if "real market prints" in c]
    assert hit and flags[hit[0]] is False


def test_an_unknown_premise_is_unverifiable_not_assumed_true():
    out = think(symbol="BTC", tf="tf1", res=_res(), store=_store())
    unknown = [a for a in out["assumptions"] if a["holds"] is None]
    assert unknown, "with no graded outcomes the confidence premise is unknown"
    assert all(a["evidence"].strip() for a in unknown)


def test_a_partial_fear_greed_is_named_as_an_assumption():
    em = {"fear_greed": {"partial": True, "coverage": 0.5}}
    out = think(symbol="BTC", tf="tf1", res=_res(emotion=em),
                store=_store(emotions={"tf1": em}))
    claims = [a["claim"] for a in out["assumptions"]]
    assert any("Fear & Greed" in c for c in claims), claims


# ── 6. metacognition: competence and the dangling certainty ────────────
def _graded(n, win_rate, conf=0.6, wins=None):
    wins = int(round(n * win_rate)) if wins is None else wins
    return [{"result": "win" if i < wins else "loss",
             "confidence": conf} for i in range(n)]


def test_competence_is_provisional_until_enough_outcomes():
    c = competence(0.9, _graded(5, 0.6))
    assert c["state"] == "provisional"
    assert c["trust"] == "provisional"
    assert c["n"] == 5


def test_overconfidence_is_detected_and_forces_abstention():
    comp = competence(0.95, _graded(MIN_GRADED_CONFIDENT + 10, 0.45))
    assert comp["trust"] == "overconfident"
    assert comp["dangling"] > 0.05
    out = should_abstain(0.95, comp, floor=0.21)
    assert out["abstain"] is True
    assert "unearned" in out["reason"]


def test_calibrated_confidence_is_allowed_through():
    comp = competence(0.58, _graded(MIN_GRADED_CONFIDENT + 10, 0.56))
    assert comp["trust"] == "calibrated"
    assert should_abstain(0.58, comp, floor=0.21)["abstain"] is False


def test_no_evidence_never_counts_as_evidence_against():
    """A small sample must not veto: 'I cannot tell' is not 'no'."""
    comp = competence(0.9, _graded(3, 0.0))
    assert should_abstain(0.9, comp, floor=0.21)["abstain"] is False


def test_below_the_floor_always_abstains():
    out = should_abstain(0.10, competence(0.10, []), floor=0.35)
    assert out["abstain"] is True
    assert "floor" in out["reason"]


def test_brier_decomposition_separates_the_two_failure_modes():
    # perfectly calibrated, zero resolution: always predicts the base rate
    flat = [(0.5, 1), (0.5, 0), (0.5, 1), (0.5, 0)] * 5
    bd = brier_decompose(flat)
    assert bd["state"] == "measured"
    assert bd["reliability"] is not None and bd["reliability"] < 0.01
    assert bd["resolution"] is not None and bd["resolution"] < 0.01
    # decisive and right: high resolution
    sharp = [(0.95, 1)] * 10 + [(0.05, 0)] * 10
    sd = brier_decompose(sharp)
    assert sd["resolution"] > bd["resolution"]
    assert sd["brier"] < 0.05


def test_brier_refuses_small_samples():
    bd = brier_decompose([(0.6, 1), (0.6, 0)])
    assert bd["state"] == "insufficient" and bd["brier"] is None


# ── 7. the narrator is not allowed to be an analyst ────────────────────
def test_narration_guard_rejects_an_invented_figure():
    out = think(symbol="BTC", tf="tf1", res=_res(), store=_store())
    ok = deterministic_narration(out)
    assert guard_narration(ok, out)["ok"] is True
    invented = ok + "\nAlso, funding is 8.31 % and open interest 41271."
    g = guard_narration(invented, out)
    assert g["ok"] is False
    assert "8.31" in g["stray_numbers"] or "41271" in g["stray_numbers"]


def test_narration_guard_rejects_a_hidden_abstention():
    res = _res(confidence=0.05)
    out = think(symbol="BTC", tf="tf1", res=res, store=_store())
    assert out["abstain"]["abstain"] is True
    g = guard_narration("The app is going long with conviction.", out)
    assert g["missing_abstain"] is True and g["ok"] is False


def test_narration_prompt_carries_the_trace_not_the_market():
    out = think(symbol="BTC", tf="tf1", res=_res(), store=_store())
    p = narration_prompt(out)
    assert "narrator" in p.lower()
    # the prompt is JSON, so the arrow arrives escaped - compare decoded
    assert json.loads(p.split("TRACE:", 1)[1])["counterfactual"] == \
        out["counterfactual"]
    trace = json.loads(p.split("TRACE:", 1)[1])
    for s, sent in zip(out["steps"], trace["steps"]):
        assert sent["detail"] == s["detail"]
        assert sent["status"] == s["status"]


# ── 8. nothing is invented when the payload is bare ────────────────────
def test_a_bare_payload_produces_an_honest_trace_not_a_pretty_one():
    out = think(symbol="BTC", tf="tf1", res={}, store={})
    assert out["ok"] is True
    assert [s["step"] for s in out["steps"]] == list(STEPS)
    # it must not claim competence it does not have
    assert out["competence"]["state"] == "unknown"
    assert out["abstain"]["abstain"] is True
    # and every step still says something true (often: "nothing to go on")
    blob = json.dumps(out).lower()
    for lie in ('"hit_rate": 0.5', "nan", "infinity"):
        assert lie not in blob
    # with nothing to reason from, it must abstain - and the trace stays
    # internally consistent BECAUSE it abstains
    assert out["working_memory"]["present"] == []
    assert out["consistency"]["ok"], out["consistency"]["problems"]


# ── 9. the engine actually records and grades its thinking ─────────────
def test_analysis_records_a_thought_and_its_signature():
    """The trace must be produced BY the pipeline, not only on request."""
    from tests.test_engine import make_engine, run, _boot
    from conftest import run_sync

    eng, _f, _e = make_engine()
    run(_boot(eng))

    async def _emit(_m):
        pass

    eng.emit = _emit

    async def go():
        assert eng.start_analyse("ETH", "tf2")["ok"]
        await eng._analyse_task

    run_sync(go())
    job = eng.analyse_job
    assert job["status"] == "done", job.get("error")
    thinking = job["result"].get("thinking")
    assert thinking, "the analysis produced no thinking summary"
    assert thinking["signature"], "no reproducible signature"
    assert thinking["n_steps"] == len(STEPS)
    assert (thinking["consistency"] or {}).get("ok") is True
    assert thinking["counterfactual"].strip()
    # ...and it landed in the ledger
    key = "ETH|tf2"
    assert eng._thoughts.get(key), "nothing recorded in the thought ledger"
    rec = eng._thoughts[key][-1]
    assert rec["signature"] == thinking["signature"]
    assert rec["outcome"] is None          # not settled yet


def test_a_settled_window_grades_the_thought_that_made_it():
    """Without this the app could reason beautifully and never learn."""
    from tests.test_engine import make_engine, run, _boot

    eng, _f, _e = make_engine()
    run(_boot(eng))
    key = "BTC|tf1"
    eng._thoughts[key] = [{"ts": 1.0, "signature": "abc", "confidence": 0.5,
                           "signal": "BUY", "abstain": False,
                           "consistency": True, "split": False,
                           "outcome": None}]
    eng._grade_last_thought("BTC", "tf1", True)
    assert eng._thoughts[key][-1]["outcome"] == "win"
    assert eng._thoughts[key][-1]["graded_at"] > 0
    # grading the same settlement twice must not rewrite history
    eng._grade_last_thought("BTC", "tf1", False)
    assert eng._thoughts[key][-1]["outcome"] == "win"


def test_thought_stats_measures_reasoning_not_bookkeeping():
    from tests.test_engine import make_engine, run, _boot

    eng, _f, _e = make_engine()
    run(_boot(eng))
    eng._thoughts = {}
    for i in range(40):
        eng._thoughts.setdefault("BTC|tf1", []).append(
            {"confidence": 0.8, "outcome": "win" if i < 14 else "loss"})
    st = eng.thought_stats()
    assert st["n"] == 40
    assert st["hit_rate"] == pytest.approx(0.35, abs=0.01)
    # a stated 0.80 against a 35 % record is a dangling certainty
    assert st["dangling"] > 0.05
    assert st["trust"] == "overconfident"
