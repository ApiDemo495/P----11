"""Cognitive layer: a thinking process + metacognition for every call.

Why this module exists
----------------------
The engine already computes a lot and *reports* it, but reporting is not
thinking. Every panel said WHAT the numbers were; none said HOW the app got
from evidence to a call, what it considered and rejected, what it assumed, or
how much it should trust itself. Those are the four things a person means by
"thought process":

1. **Working memory** - what actually entered this window's reasoning, and
   what was deliberately withheld from it (with the reason).
2. **Deliberation** - two independent routes to a side (the weighted evidence
   sum, and the vote head-count), their agreement, and the outcome that was
   considered and rejected.
3. **Assumptions** - every premise the call depends on, as a testable claim
   with a measured validity flag.
4. **Metacognition** - the app's estimate of its own competence, and the gap
   between that and what it can prove. That gap forces abstention.

Design rules (each pinned by a test)
------------------------------------
* **Nothing here invents a number**: every field is read from the payload or
  arithmetic over those readings. A missing input is None and named.
* **Thought must be reproducible**: same payload in, same trace out; the
  signature excludes timestamps so two runs can be diffed.
* **Calibration is measured, never assumed** - provisional until enough
  settled outcomes exist, and the UI must say so too.
* **Brier decomposition** reports reliability and resolution separately: a
  model can be well calibrated and still useless.

The language model, when loaded, only NARRATES this trace. It never supplies
a step, a number or a conclusion.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from typing import Any, Iterable

# -- thresholds, in one place so the UI can cite them --------------------
MIN_GRADED_CONFIDENT = 30   # graded outcomes before a confident claim
MIN_GRADED_CALIB = 15       # before Brier is attempted at all
DANGLING_TOL = 0.05         # |conf - hit_rate| that counts as unearned
CONFLICT_DIR = 0.60         # route disagreement above this is a real split
CIRCULAR_RHO = 0.55         # source correlation that voids independence

STEPS = ("observe", "recall", "deliberate", "check", "decide", "reflect")

SOURCE_LABELS = {
    "formulas": "path formulas", "brain": "fly connectome",
    "physics": "book physics", "emotion": "affect channels",
    "news": "news pipe", "hedge": "hedge-flow",
    "spillover": "cross-asset spillover", "micro": "microstructure",
    "agent": "cloud/local agent", "meta": "meta-combiner",
}


def _num(v: Any) -> float | None:
    """A finite float or None. Never coerces a string into a fake number."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# ======================================================================
# Metacognition - the model's estimate of its own competence
# ======================================================================
def brier_decompose(pairs: Iterable[tuple[float, int]], bins: int = 5) -> dict:
    """Murphy (1973) decomposition: Brier = reliability - resolution + unc.

    Reported separately because they fail differently: a model can be
    perfectly reliable (stated 60 %, won 60 %) and still useless (resolution
    ~ 0, it never separates the outcomes at all).
    """
    ps = [(float(p), int(y)) for p, y in pairs
          if p is not None and y in (0, 1) and 0.0 <= float(p) <= 1.0]
    n = len(ps)
    if n < MIN_GRADED_CALIB:
        return {"n": n, "brier": None, "reliability": None,
                "resolution": None, "uncertainty": None,
                "state": "insufficient",
                "why": f"needs {MIN_GRADED_CALIB} graded outcomes, has {n}"}
    base = sum(y for _, y in ps) / n
    brier = sum((p - y) ** 2 for p, y in ps) / n
    binsz = max(1, math.ceil(n / bins))
    rel = res = 0.0
    for i in range(0, n, binsz):
        chunk = ps[i:i + binsz]
        mp = sum(p for p, _ in chunk) / len(chunk)
        my = sum(y for _, y in chunk) / len(chunk)
        rel += len(chunk) / n * (mp - my) ** 2
        res += len(chunk) / n * (my - base) ** 2
    return {"n": n, "brier": round(brier, 5),
            "reliability": round(rel, 5), "resolution": round(res, 5),
            "uncertainty": round(base * (1.0 - base), 5),
            "base_rate": round(base, 4), "state": "measured",
            "why": ("reliability = how far stated probabilities sit from "
                    "observed frequencies; resolution = how much they "
                    "separate. Brier = rel - res + unc")}


def competence(conf: float | None, graded: list[dict],
               window: int = 60) -> dict:
    """How good is this model at this call, and does it know?

    ``graded`` are settled rows carrying ``confidence`` and ``result``. The
    estimate is the hit-rate over recent settled outcomes - the only evidence
    the app has that its own confidences mean anything.
    """
    rows = [r for r in graded
            if r.get("result") in ("win", "loss")
            and _num(r.get("confidence")) is not None][-window:]
    n = len(rows)
    p_correct = _num(conf)
    if n == 0:
        return {"n": 0, "hit_rate": None, "wilson_lo": None,
                "brier": None, "state": "unknown",
                "how_sure": "no graded outcomes yet - the app cannot know",
                "dangling": None, "trust": None}
    wins = sum(1 for r in rows if r.get("result") == "win")
    rate = wins / n
    # Wilson 95 % lower bound: 60 % on 5 rows is not 60 % competence
    z = 1.96
    denom = 1.0 + z * z / n
    centre = rate + z * z / (2 * n)
    spread = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
    lo = max(0.0, (centre - spread) / denom)
    bd = brier_decompose([(_num(r.get("confidence")),
                           1 if r["result"] == "win" else 0) for r in rows])
    dangling = None if p_correct is None else round(p_correct - rate, 4)
    if n < MIN_GRADED_CONFIDENT:
        trust = "provisional"
    elif p_correct is None:
        trust = "unscored"
    elif dangling is not None and dangling > DANGLING_TOL:
        trust = "overconfident"
    elif dangling is not None and dangling < -DANGLING_TOL:
        trust = "underconfident"
    else:
        trust = "calibrated"
    return {
        "n": n, "hit_rate": round(rate, 4), "wilson_lo": round(lo, 4),
        "brier": bd["brier"], "brier_parts": bd,
        "state": "measured" if n >= MIN_GRADED_CONFIDENT else "provisional",
        "how_sure": (f"{wins}/{n} settled calls won ({rate:.0%}); the 95 % "
                     f"lower bound is {lo:.0%}, so a claim above that is not "
                     f"yet earned" if n else
                     "no graded outcomes yet - the app cannot know"),
        "dangling": dangling, "trust": trust,
    }


def should_abstain(conf: float | None, comp: dict, floor: float = 0.21) -> dict:
    """Knowing you do not know must be able to change the answer.

    Fires when confidence is below the issue floor, or above the measured
    hit-rate by more than the tolerance on a sample big enough to judge.
    Never fires on a small sample: "I have no evidence" is not evidence
    against.
    """
    c = _num(conf)
    if c is None:
        return {"abstain": True, "reason": "no confidence was produced"}
    if c < floor:
        return {"abstain": True,
                "reason": f"confidence {c:.3f} below the issue floor {floor:.3f}"}
    n = int(comp.get("n") or 0)
    hit = _num(comp.get("hit_rate"))
    if n >= MIN_GRADED_CONFIDENT and hit is not None and c - hit > DANGLING_TOL:
        return {"abstain": True,
                "reason": (f"confidence {c:.3f} exceeds the measured hit-rate "
                           f"{hit:.3f} by more than {DANGLING_TOL:.2f} over "
                           f"{n} graded calls - the certainty is unearned")}
    return {"abstain": False,
            "reason": "confidence is within measured competence"}


# ======================================================================
# The trace
# ======================================================================
def _step(name: str, title: str, detail: str, *, status: str = "done",
          cites: Iterable[str] = (), data: dict | None = None) -> dict:
    return {"step": name, "title": title, "detail": detail,
            "status": status, "cites": [c for c in cites if c],
            "data": data or {}}


def _withheld_reason(src: str, res: dict) -> str:
    if src == "brain":
        b = res.get("brain") or {}
        return (b.get("reason") or "connectome not ready - the vote is "
                "withheld rather than guessed")
    return {
        "news": "no scored headline this window - the pipe abstains",
        "spillover": "peer cross-section too thin to measure",
        "micro": "order book not deep enough on this tape",
        "emotion": "affect model produced nothing for this window",
        "physics": "book physics had no usable input",
        "hedge": "no hedge-flow measurement for this timeframe",
        "formulas": "no formula readings in the payload",
    }.get(src, "not present in the payload the engine built")


def _working_memory(store: dict, res: dict) -> dict:
    """What entered the reasoning this window, and what was withheld."""
    present, withheld = [], []
    em = (store.get("emotions") or {}).get(res.get("_tf") or "") or \
        (res.get("emotion") or {})
    checks = {
        "formulas": bool(res.get("breakdown")
                         or res.get("cci_adjusted") is not None),
        "brain": bool((res.get("brain") or {}).get("ready")
                      or (res.get("brain") or {}).get("vote") is not None),
        "physics": bool((store.get("physics") or {}).get(res.get("_tf") or "")),
        "emotion": bool(em),
        "news": bool(store.get("news")),
        "hedge": bool((store.get("hedge") or {}).get(res.get("_tf") or "")),
        "spillover": bool(res.get("spillover")),
        "micro": bool(res.get("micro")),
    }
    for src, ok in checks.items():
        item = {"source": src, "label": SOURCE_LABELS.get(src, src)}
        if ok:
            present.append(item)
        else:
            item["why"] = _withheld_reason(src, res)
            withheld.append(item)
    # explicit abstentions the subsystems themselves declared
    for i, why in sorted((em.get("zero_reasons") or {}).items(),
                         key=lambda kv: str(kv[0])):
        nm = ((em.get("model") or {}).get("channels") or {}).get(str(i), {})
        withheld.append({"source": f"channel:{i}",
                         "label": nm.get("name") or f"channel {i}",
                         "why": str(why)[:160]})
    n_declared = sum(1 for w in withheld
                     if not w["source"].startswith("channel:"))
    total = len(present) + n_declared
    return {
        "present": present, "withheld": withheld,
        "unknown": [] if res.get("issued") is not None
        else ["whether this window was ISSUED"],
        "coverage": round(len(present) / total, 3) if total else 0.0,
        "summary": (f"{len(present)}/{total} independent sources reached this "
                    f"window's reasoning"
                    + (f"; {len(withheld)} reading(s) withheld, each with its "
                       f"reason" if withheld else "")),
    }


def _reject_reason(direction: int, f: float | None, a: float | None,
                   bd: dict) -> str:
    parts = []
    if f is not None and a is not None:
        net = f - a
        same = (net > 0) == (direction > 0)
        parts.append(f"the fused mass is {f:.3f} for vs {a:.3f} against "
                     f"(net {net:+.3f}, {'supporting' if same else 'opposing'} "
                     f"the chosen side)")
    if bd.get("conflicting"):
        parts.append("the double-check flagged the voters as conflicting")
    pen = _num(bd.get("penalty"))
    if pen:
        parts.append(f"a {pen:.3f} disagreement penalty was already applied")
    parts.append("no opposing supporter outweighs the leading cluster")
    return "; ".join(parts)


def _deliberation(res: dict) -> dict:
    """Two independent routes to a side, and the alternative that lost.

    Route A is the engine's own fused mass (supporters minus opponents, the
    quantity the confidence is built from). Route B is a head-count over the
    subsystem votes - a genuinely different statistic, because it ignores how
    strongly each one voted. Agreement between them is evidence the side does
    not depend on one arithmetic choice.
    """
    direction = int(_num(res.get("direction")) or 0)
    mass = res.get("mass") or {}
    bd = res.get("breakdown") or {}
    f, a = _num(mass.get("for")), _num(mass.get("against"))
    voters = mass.get("subsystem_votes") or {}
    supporters = mass.get("supporters") or []
    opponents = mass.get("opponents") or []

    net = None if (f is None or a is None) else round(f - a, 5)
    pos = sum(1 for v in voters.values() if (_num(v) or 0) > 0)
    neg = sum(1 for v in voters.values() if (_num(v) or 0) < 0)
    flat = len(voters) - pos - neg
    headcount = round((pos - neg) / len(voters), 4) if voters else None
    agree = None
    if net is not None and headcount is not None:
        agree = ((net > 0) == (headcount > 0)
                 and (net == 0) == (headcount == 0))
    # Conflict is DISAGREEMENT, not distance. Measuring |net − headcount|
    # marked a call as split whenever the two routes agreed but differed in
    # size (net +0.043 vs head-count +0.83 scored as 0.79 conflict) - which
    # then tripped the self-check and painted a warning over a call the app
    # was in fact agreed on. Sign disagreement is the split; a size gap is
    # reported separately as `magnitude_gap` because the two routes are on
    # different scales by construction.
    conflict = 0.0
    magnitude_gap = None
    if net is not None and headcount is not None:
        magnitude_gap = round(abs(net) - abs(headcount), 4)
        if agree is False:
            conflict = round(min(1.0, abs(net - headcount) or 1.0), 4)

    alt = None
    if direction != 0:
        alt = {"considered": "SELL" if direction > 0 else "BUY",
               "why_not": _reject_reason(direction, f, a, mass)}

    if net is None:
        summary = ("the fusion carried no mass reading, so there is nothing "
                   "to deliberate over - the side rests on the CCI alone")
    else:
        summary = (f"fused mass {f:+.3f} for vs {a:+.3f} against "
                   f"(net {net:+.3f}) over {len(supporters)} supporter(s); "
                   f"vote route {pos}\u25b2/{neg}\u25bc/{flat}\u00b7 = "
                   f"{(headcount or 0):+.2f}")
        if agree is True:
            summary += (" - the two routes agree, so the side is not an "
                        "artefact of one statistic")
        elif agree is False:
            summary += (" - THE ROUTES DISAGREE: the side rests on the size of "
                        "the contributions, not the head-count, and that is "
                        "stated rather than hidden")
        if mass.get("conflicting"):
            summary += "; the double-check flagged conflicting voters"
        if alt:
            summary += f"; the {alt['considered']} case was tested and rejected"
    return {
        "route_evidence": {"value": net, "for": f, "against": a,
                           "components_used": len(supporters) + len(opponents),
                           "rule": "fused mass: supporters\u2212opponents "
                                   "\u00d7 their weights"},
        "route_votes": {"value": headcount, "pos": pos, "neg": neg,
                        "flat": flat, "n": len(voters),
                        "rule": "(agreeing \u2212 disagreeing) / voters, "
                                "ignoring how strongly each voted"},
        "routes_agree": agree, "conflict": conflict,
        "magnitude_gap": magnitude_gap,
        "split": bool(agree is False and conflict >= CONFLICT_DIR),
        "conflicting_flag": bool(mass.get("conflicting")),
        "alternative": alt, "summary": summary,
    }


def _assumptions(res: dict, store: dict, comp: dict) -> list[dict]:
    """Every premise the call depends on, as a testable claim."""
    out: list[dict] = []

    def add(claim: str, holds: bool | None, evidence: str):
        out.append({"claim": claim,
                    "holds": None if holds is None else bool(holds),
                    "evidence": evidence})

    feed = store.get("feed") or res.get("feed") or {}
    age = _num(feed.get("last_delivery_age_s"))
    add("the tape is current enough to act on",
        None if age is None else age < 30.0,
        f"last delivery {age:.0f}s ago" if age is not None
        else "no delivery age reported")

    sim = feed.get("simulated")
    add("the prices are real market prints, not a simulation",
        None if sim is None else (not sim),
        ("tape is SIMULATED - every price-derived claim rehearses the method, "
         "it is not a claim about the market") if sim
        else "live exchange tape")

    add("the confidence means what it says",
        None if int(comp.get("n") or 0) < MIN_GRADED_CONFIDENT
        else (comp.get("trust") == "calibrated"),
        comp.get("how_sure") or "no graded evidence")

    add("no two sources are secretly the same source", True,
        "sources are counted independently here, but correlation is not "
        f"re-estimated: above \u03c1\u2248{CIRCULAR_RHO} two 'independent' "
        "readings can be one")

    em = (store.get("emotions") or {}).get(res.get("_tf") or "") or \
        (res.get("emotion") or {})
    fg = em.get("fear_greed") or {}
    if fg:
        add("the Fear & Greed composite is fully measured",
            not bool(fg.get("partial")),
            (f"only {round((fg.get('coverage') or 0) * 100)}% of the published "
             "weighting has an input; the printed index is renormalised over "
             "what is measured") if fg.get("partial")
            else "all components measured")
    return out


def _confidence_logic(res: dict) -> dict:
    """State the confidence arithmetic as the engine applied it."""
    parts = []
    conf = _num(res.get("confidence"))
    cci = _num(res.get("cci_adjusted"))
    pc = _num(res.get("p_correct"))
    if cci is not None:
        parts.append(f"|cci_adj| {abs(cci):.4f} through the graded tanh map")
    pen = _num(res.get("penalty"))
    if pen is not None:
        parts.append(f"a {pen:.3f} disagreement penalty")
    if pc is not None:
        cs = res.get("calib_iso") or {}
        parts.append(f"the isotonic P(win) {pc:.3f} (n={cs.get('n', 0)}, "
                     f"{cs.get('source', 'map')})")
    if not parts:
        return {"detail": "the payload carried no confidence arithmetic",
                "ceiling": None, "conf": conf, "note": ""}
    ceiling = None if pc is None else round(_clamp(pc, 0.02, 0.98), 4)
    over = (conf is not None and ceiling is not None
            and conf > ceiling + 0.05)
    return {
        "detail": "; ".join(parts), "ceiling": ceiling, "conf": conf,
        "note": ("a confidence above the isotonic P(win) is a claim the "
                 "settled record does not support" if over
                 else "confidence sits inside the calibrated ceiling"),
    }


def _counterfactual(res: dict, delib: dict) -> str:
    """What would have changed the answer - the falsifier."""
    direction = int(_num(res.get("direction")) or 0)
    hits = []
    for key, label in (("spillover", "cross-asset spillover"),
                       ("edge", "cost gate"), ("horizon", "horizon state")):
        node = res.get(key) or {}
        st = node.get("state") if isinstance(node, dict) else None
        if st and st not in ("pass", "tradable", "ok", "measured"):
            hits.append(f"{label} is {st}")
    conflict = delib.get("conflict") or 0.0
    if conflict >= CONFLICT_DIR:
        hits.append(f"route conflict is {conflict:.2f}, above the "
                    f"{CONFLICT_DIR:.2f} line")
    if not hits:
        hits.append("no single falsifier is active: the call is limited by "
                    "magnitude, not by a failing check")
    head = "\u2192 FLAT" if direction == 0 else "\u2192 reversed"
    return head + " if " + "; ".join(hits)


def thought_signature(steps: list[dict]) -> str:
    """A stable hash of the reasoning, so two runs can be diffed.

    Excludes every timestamp: identical evidence must give an identical
    signature, otherwise "reproducible thinking" is untestable.
    """
    blob = json.dumps(
        [{"step": s["step"], "status": s["status"], "detail": s["detail"],
          "cites": s["cites"]} for s in steps],
        sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def self_check(payload: dict) -> dict:
    """Does the trace hold together on its own terms?

    A thinking process that can print an inconsistent trace is worse than no
    trace, because it is persuasive. These are the contradictions worth
    catching mechanically.
    """
    problems = []
    steps = payload.get("steps") or []
    order = [s.get("step") for s in steps]
    if order != list(STEPS):
        problems.append(f"step order is {order}, expected {list(STEPS)}")
    for s in steps:
        if not str(s.get("detail") or "").strip():
            problems.append(f"step '{s.get('step')}' concluded nothing")
    # "no sources" is a DATA GAP, not a contradiction - but only if the app
    # also abstained. Proceeding to a call with nothing in working memory is
    # the contradiction worth flagging.
    if not (payload.get("working_memory") or {}).get("present") and \
            not (payload.get("abstain") or {}).get("abstain"):
        problems.append("proceeded to a call with no source in working memory")
    if (payload.get("deliberation") or {}).get("split") and \
            not any(s.get("status") == "conflict" for s in steps):
        problems.append("the routes split but no step reports the conflict")
    if (payload.get("abstain") or {}).get("abstain") and \
            not any(s.get("status") == "warn" for s in steps):
        problems.append("the app abstained without saying so in a step")
    comp = payload.get("competence") or {}
    if comp.get("state") == "measured" and comp.get("hit_rate") is None:
        problems.append("competence claims to be measured with no hit-rate")
    return {"ok": not problems, "problems": problems,
            "checked": ["order", "non-empty", "sources", "conflict",
                        "abstain", "competence"]}


def think(*, symbol: str, tf: str, res: dict, store: dict,
          graded: list[dict] | None = None, floor: float = 0.21,
          system_mode: str = "old") -> dict:
    """Produce the full trace. Pure: reads ``res``/``store`` only."""
    res = dict(res or {})
    res["_tf"] = tf
    store = dict(store or {})
    graded = list(graded or [])
    conf = _num(res.get("confidence"))
    comp = competence(conf, graded)
    abst = should_abstain(conf, comp, floor=floor)
    wm = _working_memory(store, res)
    delib = _deliberation(res)
    assum = _assumptions(res, store, comp)
    clogic = _confidence_logic(res)
    cfact = _counterfactual(res, delib)

    direction = int(_num(res.get("direction")) or 0)
    signal = "BUY" if direction > 0 else "SELL" if direction < 0 else "FLAT"
    note = ((res.get("emotion") or {}).get("narrative") or "").strip()
    failing = sum(1 for a in assum if a["holds"] is False)
    conf_txt = f"{conf:.3f}" if conf is not None else "\u2014"

    steps = [
        _step("observe", "What I am looking at",
              (f"{symbol} \u00b7 {tf} \u00b7 {system_mode.upper()} system \u00b7 "
               f"confidence {conf_txt} \u00b7 call {signal}"
               + (f" \u00b7 {note[:180]}" if note else "")),
              cites=["engine.compute_asset", "emotion.compute"],
              data={"signal": signal, "confidence": conf,
                    "system_mode": system_mode}),
        _step("recall", "What already reached this decision",
              wm["summary"], status="done" if wm["present"] else "partial",
              cites=["learning.formula_accuracy", "ledger.stats"], data=wm),
        _step("deliberate", "How I weighed it", delib["summary"],
              status="conflict" if delib["split"] else "done",
              cites=["cci.compute", "cci.finalize_confidence"], data=delib),
        _step("check", "What I assume, and what I set aside",
              (f"{failing} assumption(s) do not hold; "
               f"{len(wm['withheld'])} input(s) withheld"
               + (f"; ABSTAINING - {abst['reason']}" if abst["abstain"] else "")),
              status="warn" if (abst["abstain"] or failing) else "done",
              cites=["emotion.compute", "feed.stats"],
              data={"assumptions": assum}),
        _step("decide", "What I concluded, and why that ceiling",
              f"{signal} at confidence {conf_txt} \u00b7 {clogic['detail']}",
              cites=["engine._analyse_body"],
              data={"confidence_logic": clogic, "counterfactual": cfact}),
        _step("reflect", "How much to trust myself",
              comp["how_sure"] + (f" \u00b7 {comp['trust']}"
                                  if comp.get("trust") else ""),
              status={"calibrated": "done", "provisional": "partial",
                      "overconfident": "warn",
                      "underconfident": "warn"}.get(comp.get("trust"),
                                                    "partial"),
              cites=["learning", "ledger"], data=comp),
    ]
    payload = {
        "ok": True, "symbol": symbol, "tf": tf, "system_mode": system_mode,
        "steps": steps, "working_memory": wm, "deliberation": delib,
        "assumptions": assum, "confidence_logic": clogic,
        "counterfactual": cfact, "competence": comp, "abstain": abst,
        "thinking": True, "n_steps": len(steps),
        "signature": thought_signature(steps),
        "generated_at": time.time(),
    }
    payload["consistency"] = self_check(payload)
    return payload


# ======================================================================
# Narration - the language model may only re-tell the trace
# ======================================================================
NARRATE_SYSTEM = (
    "You are the narrator of a trading app's thought process. You are given a "
    "JSON thought trace the app already produced. Re-tell it in plain "
    "language, step by step, in at most 220 words. Rules: use ONLY figures "
    "present in the JSON; never add a reason, number or conclusion; if the "
    "trace says the app abstained, say so plainly; if an assumption does not "
    "hold, name it. You are a narrator, not an analyst - you cannot change "
    "the decision.")


def narration_prompt(payload: dict) -> str:
    slim = {
        "symbol": payload.get("symbol"), "tf": payload.get("tf"),
        "steps": [{"step": s["step"], "title": s["title"],
                   "detail": s["detail"], "status": s["status"]}
                  for s in payload.get("steps") or []],
        "assumptions_failing": [a["claim"] for a in
                                payload.get("assumptions") or []
                                if a.get("holds") is False],
        "abstain": payload.get("abstain"),
        "competence": {k: (payload.get("competence") or {}).get(k)
                       for k in ("hit_rate", "n", "trust", "how_sure")},
        "counterfactual": payload.get("counterfactual"),
    }
    return (NARRATE_SYSTEM + "\n\nTRACE:\n"
            + json.dumps(slim, indent=1, default=str))


def deterministic_narration(payload: dict) -> str:
    """The fallback narrative - no model required, every sentence sourced."""
    lines = [f"{s['step'].upper()} - {s['title']}: {s['detail']}"
             for s in payload.get("steps") or []]
    bad = [a["claim"] for a in payload.get("assumptions") or []
           if a.get("holds") is False]
    if bad:
        lines.append("ASSUMPTIONS THAT DO NOT HOLD: " + "; ".join(bad))
    abst = payload.get("abstain") or {}
    if abst.get("abstain"):
        lines.append("ABSTAINING - " + str(abst.get("reason")))
    lines.append("FALSIFIER - " + str(payload.get("counterfactual")))
    return "\n".join(lines)


def guard_narration(text: str, payload: dict) -> dict:
    """Reject a narration that disagrees with the structured trace.

    Deterministic, not another model: every number the narrator prints must
    appear in the trace, and an abstention must be mentioned. A narrator that
    invents a figure is worse than no narrator.
    """
    allowed: set[str] = set()
    for s in payload.get("steps") or []:
        allowed.update(re.findall(r"-?\d+\.?\d*", str(s.get("detail") or "")))
        allowed.update(re.findall(r"-?\d+\.?\d*", str(s.get("title") or "")))
    for a in payload.get("assumptions") or []:
        allowed.update(re.findall(r"-?\d+\.?\d*", str(a.get("evidence") or "")))
    allowed.update(re.findall(r"-?\d+\.?\d*",
                              str(payload.get("counterfactual") or "")))
    used = re.findall(r"-?\d+\.?\d*", text or "")
    stray = sorted({u for u in used if u not in allowed})
    abst = (payload.get("abstain") or {}).get("abstain")
    missing = bool(abst) and "abstain" not in (text or "").lower()
    ok = not stray and not missing
    return {"ok": ok, "stray_numbers": stray[:8], "missing_abstain": missing,
            "why": ("narration matches the trace" if ok else
                    "narration invents figures or hides the abstention - the "
                    "structured trace is authoritative")}
