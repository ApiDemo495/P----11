"""One derivation for the card badge — and the rule that keeps it honest.

Reported defect (live board, BTC tf4)::

    confidence 0.009 · tier half · verdict "ISSUE" · issued true

…on a 0.385 issue floor, i.e. an "ISSUED" card at 2.3 % of the floor it was
supposed to clear, while its siblings read NO CALL. The badge had been
derived in three places (fresh cycle, GRC refresh, ANALYSE final) and then
FROZEN: ``_on_sl_hit`` and the target-lock path lower ``cyc.confidence``
without touching the badge, so the text and the number next to it drifted
apart with every penalty.

This module is now the only place that decides what a badge says:

* :func:`public_verdict` — the rule. Fresh cycles, GRC refreshes and ANALYSE
  finals all call it, so a call that cleared the floor is described the same
  way wherever it is printed.
* :func:`reconcile` — the invariant applied at *serialisation* time. It can
  only ever DOWNGRADE (ISSUE → WATCH → NO CALL), never upgrade, so it cannot
  invent a call the engine did not make; it exists so that a confidence
  changed after the badge was written can never be published next to a badge
  that contradicts it. Aborting a print is not "changing a side": the side,
  the levels and the lock are untouched and stay on the record for grading
  (docs11-doublecheck §3).

Semantics, unchanged: the side is always issued for grading; ``ISSUE`` means
the side cleared the asset's own issue floor (``config.asset_issue_floor``)
and the tier is not watch-only. ``WATCH`` is the same side at watch-only size.

The trend-interaction gate (:mod:`app.trend_gate`) is consulted here, and it
too can only withhold.
"""
from __future__ import annotations

from typing import Any

_ORDER = {"NO CALL": 0, "WATCH": 1, "ISSUE": 2}


def _floor(symbol: str) -> float:
    from .config import CONFIDENCE_FLOOR, asset_issue_floor
    try:
        return float(asset_issue_floor(symbol))
    except Exception:
        return float(CONFIDENCE_FLOOR)


def _num(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


#: the exact shapes this module generates; anything else is an engine
#: explanation and is never rewritten
_GENERATED_PREFIXES = ("below floor (", "cleared the ", "chases the ",
                       "opposes the ", "trend-interaction gate withheld")


def _is_generated(note: str | None) -> bool:
    text = str(note or "")
    return any(text.startswith(p) for p in _GENERATED_PREFIXES)


def public_verdict(symbol: str, tf: str, direction: int, confidence: float,
                   tier: str | None, issue_note: str | None = None,
                   ) -> tuple[bool, str, str]:
    """The badge rule: (gate_ok, verdict, issue_note).

    ``gate_ok`` is what execution reads; ``verdict`` is what the user reads;
    ``issue_note`` says why, in the same units (the floor), so the badge is
    always auditable against the confidence printed beside it.

    THE FLIP LIVES HERE (2026-11-26). This is the one function every card,
    lock and execution path already calls for its badge, so it is also the
    only place the trend interaction has to be applied for the whole app to
    agree. It returns a THIRD element as before (the note) and the caller can
    read the resolved side from :func:`resolve_side` — see
    :func:`resolve` for the tuple that carries the flipped side.
    """
    ok, verdict, note, _side = resolve(symbol, tf, direction, confidence,
                                       tier, issue_note)
    return ok, verdict, note


def resolve(symbol: str, tf: str, direction: int, confidence: float,
            tier: str | None, issue_note: str | None = None,
            ) -> tuple[bool, str, str, int]:
    """Same rule, plus the SIDE the app must actually publish.

    The trend interaction (``app/trend_gate.py``) can now FLIP a chasing call
    instead of withholding it, so the side is part of the verdict: a card
    that prints "BUY" while its own sealed contract is a short would be a new
    contradiction, and this rule exists to prevent exactly that class of bug.

    ``side_out`` == ``direction`` whenever nothing acted (rule off, no
    reading, thin trend, or the call already opposes the trend), so callers
    that ignore it keep the old behaviour exactly.
    """
    floor = _floor(symbol)
    conf = _num(confidence)
    want = int(_num(direction))
    side_out = want
    gate_ok = bool(want != 0 and conf > floor)
    tier_s = str(tier or "watch-only")

    trend: dict[str, Any] = {}
    try:
        from . import trend_gate
        trend = trend_gate.check(symbol, tf, want)
    except Exception:                      # never break a card on the rule
        trend = {}
    if trend.get("blocked"):
        # ── NO WINDOW MAY PRINT "NO CALL" FOR A SIDE IT HAS (2026-11-26) ──
        # `blocked` only exists in the legacy TREND_GATE=abstain mode, which
        # is kept for A/B. But the standing mandate is that every window
        # carries a tradeable side and no panel withholds one, and live cards
        # were printing `NO CALL` with a direction attached because this
        # branch forced gate_ok False. The evidence behind the rule is that
        # the trend-following side wins MORE often, so the honest resolution
        # is the same one the default `flip` mode uses: INVERT the side
        # rather than silence it. The abstain mode keeps its vocabulary
        # (`why` still says withheld) and its measurement's direction; it no
        # longer removes the call.
        _flipped = int(trend.get("side_out") or 0)
        if _flipped in (-1, 1):
            side_out = _flipped
        elif want in (-1, 1):
            side_out = -want
        gate_ok = bool(side_out != 0 and conf > floor)
    if trend.get("flipped"):
        # a flipped side is a NEW side: keep it only if it is a real side.
        _flipped = int(trend.get("side_out") or 0)
        if _flipped in (-1, 1) and _flipped != want:
            side_out = _flipped

    if gate_ok and tier_s != "watch-only":
        verdict = "ISSUE"
    elif gate_ok:
        verdict = "WATCH"
    else:
        verdict = "NO CALL"

    # THREE decimals — exactly how the card prints confidence
    # (`fx(res.confidence, 3)`). Any other precision reads as a different
    # number beside it: "conf 0.235" next to "below floor (0.24 …)".
    floor_note = (f"cleared the {floor:.3f} floor" if conf > floor
                  else f"below floor ({conf:.3f} <= {floor:.3f})")
    if trend.get("blocked"):
        note = str(trend.get("why") or "trend-interaction rule withheld")
        if conf <= floor:
            # say BOTH reasons — a card that is withheld for one reason while
            # it also fails the floor must not read as if only one applied
            note += f"; also {floor_note}"
    elif trend.get("flipped"):
        # the flip IS the headline: it changed the call, so it must be said
        note = str(trend.get("why") or "trend rule flipped this side")
        if conf <= floor:
            note += f"; also {floor_note}"
    elif issue_note and not _is_generated(issue_note):
        # an ENGINE explanation (double-check conflict, degraded evidence…)
        # is kept — it says something this rule cannot know
        note = str(issue_note)
    else:
        # ours is always recomputed: a note formatted beside an older
        # confidence would otherwise contradict the number printed with it
        note = floor_note
    return gate_ok, verdict, note, side_out



def reconcile(symbol: str, tf: str, direction: int, confidence: float,
              tier: str | None, gate_ok: bool, verdict: str,
              issue_note: str) -> tuple[bool, str, str]:
    """Downgrade-only consistency check, applied before anything is printed.

    The published fields are returned together, so ``gate_ok``/``verdict``/
    ``issue_note`` can never describe a different number than the
    ``confidence`` they ship with. A stored badge that is MORE optimistic
    than the rule allows is replaced; a more cautious one is left alone (the
    engine may have declined a call for reasons the rule cannot see, e.g. a
    double-check conflict, and that must never be overridden here).
    """
    ok_rule, verdict_rule, note_rule = public_verdict(
        symbol, tf, direction, confidence, tier, issue_note)
    cur = str(verdict or "NO CALL")
    cur_ok = bool(gate_ok) and not (cur == "NO CALL")
    # a stored ISSUE/WATCH whose own confidence no longer clears the floor
    if cur_ok and not ok_rule:
        return ok_rule, verdict_rule, note_rule
    if _ORDER.get(cur, 0) > _ORDER.get(verdict_rule, 0):
        # stored verdict claims more than the rule allows for this confidence
        return False, verdict_rule, note_rule
    # The stored note was formatted next to an OLDER confidence. Even when the
    # badge is still right, a note naming a different number is a printed
    # contradiction ("conf 0.135" beside "below floor (0.08 …)" — a live
    # defect), so any note in one of this module's own generated formats is
    # regenerated from the numbers shipping now.
    if cur_ok or _is_generated(issue_note):
        return cur_ok, cur, note_rule
    # else the engine declined this call for a reason the rule cannot see (a
    # double-check conflict, a degraded-evidence cap) — keep its own words
    return cur_ok, cur, str(issue_note or note_rule)


def describe(symbol: str, tf: str, direction: int, confidence: float,
             tier: str | None) -> dict[str, Any]:
    """Same rule, evidence-shaped (for panels that show the whole decision)."""
    ok, verdict, note, side = resolve(symbol, tf, direction, confidence, tier)
    ev: dict[str, Any] = {}
    try:
        from . import trend_gate
        ev = trend_gate.check(symbol, tf, int(_num(direction)))
    except Exception:
        ev = {}
    return {"gate_ok": ok, "verdict": verdict, "issue_note": note,
            "side": side, "flipped": bool(side != int(_num(direction))
                                          and side in (-1, 1)),
            "floor": _floor(symbol), "confidence": _num(confidence),
            "trend_gate": ev}


def reconcile_card(symbol: str, tf: str, card: dict) -> dict:
    """Apply the whole consistency rule to one published card, in place.

    The badge, its note AND the evidence block are derived in one call, so a
    card can never ship a note that describes a gate decision its own
    ``trend_gate`` block denies (the live symptom: "chases the 2-decision
    trend — withheld" beside ``trend: 0``, because the evidence was a stored
    copy from an earlier refresh while the note was regenerated at print).
    """
    _dir_in = int(card.get("direction") or 0)
    ok, verdict, note = reconcile(
        symbol, tf, _dir_in,
        _num(card.get("confidence")), card.get("tier"),
        bool(card.get("gate_ok")), str(card.get("verdict") or ""),
        str(card.get("issue_note") or ""))
    _desc = describe(symbol, tf, _dir_in,
                     _num(card.get("confidence")), card.get("tier"))
    side = int(_desc.get("side") or _dir_in)
    if side != _dir_in and side in (-1, 1):
        # PUBLISH THE SIDE THE RULE ACTUALLY TOOK. A flip that leaves the
        # card printing the original side while every note says "flipped"
        # would be a brand-new contradiction, and the levels below belong to
        # the ORIGINAL side — so they are dropped rather than mirrored here.
        # The mirroring happens once, at the level writer, from the frozen
        # contract; a display layer must never invent a price.
        card["direction_before_flip"] = _dir_in
        card["direction"] = side
        card["signal"] = ("BUY" if side > 0 else "SELL")
        card["flip_pending_levels"] = True
        card["trend_flip"] = {
            "from": _dir_in, "to": side,
            "why": str((_desc.get("trend_gate") or {}).get("why") or ""),
            "measured": (_desc.get("trend_gate") or {}).get("measured"),
        }
    card["gate_ok"], card["verdict"] = ok, verdict
    card["issue_note"] = note
    card["trend_gate"] = _desc["trend_gate"]
    card["execution_eligible"] = bool(card.get("execution_eligible")) and ok
    return card
