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
    """
    floor = _floor(symbol)
    conf = _num(confidence)
    want = int(_num(direction))
    gate_ok = bool(want != 0 and conf > floor)
    tier_s = str(tier or "watch-only")

    trend: dict[str, Any] = {}
    try:
        from . import trend_gate
        trend = trend_gate.check(symbol, tf, want)
    except Exception:                      # never break a card on the gate
        trend = {}
    if trend.get("blocked"):
        gate_ok = False

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
        note = str(trend.get("why") or "trend-interaction gate withheld")
        if conf <= floor:
            # say BOTH reasons — a card that is withheld for one reason while
            # it also fails the floor must not read as if only one applied
            note += f"; also {floor_note}"
    elif issue_note and not _is_generated(issue_note):
        # an ENGINE explanation (double-check conflict, degraded evidence…)
        # is kept — it says something this rule cannot know
        note = str(issue_note)
    else:
        # ours is always recomputed: a note formatted beside an older
        # confidence would otherwise contradict the number printed with it
        note = floor_note
    return gate_ok, verdict, note


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
    ok, verdict, note = public_verdict(symbol, tf, direction, confidence, tier)
    ev: dict[str, Any] = {}
    try:
        from . import trend_gate
        ev = trend_gate.check(symbol, tf, int(_num(direction)))
    except Exception:
        ev = {}
    return {"gate_ok": ok, "verdict": verdict, "issue_note": note,
            "floor": _floor(symbol), "confidence": _num(confidence),
            "trend_gate": ev}
