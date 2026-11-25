"""First-touch resolution of a cycle's barriers — the honest pass grader.

Why this module exists
----------------------
`Engine._record_learning` used to grade a pass with two independent `any()`
scans over the cycle's event list:

    target_hit = any(e["type"] == "target_hit" for e in evs)
    sl_hit     = any(e["type"] == "sl_hit"     for e in evs)

Three separate things were wrong with that, and every one of them only ever
moved the record in the optimistic direction:

1. ``"sl_hit"`` is not an event type this engine ever appends. The stop-loss
   path emits ``"sl_crossed"`` (engine._on_sl_hit), so ``sl_hit`` was False
   on every pass, forever. Graded history booked 0 stops in 80 snapshots.
2. ``any()`` is order-blind. The SL "SAFE, BE CAUTIOUS" branch keeps the
   cycle alive, so a lane that stopped out and later printed TP1 was booked
   as a target hit — SL-then-TP scored as a win. Only the first touch can
   resolve a position.
3. ``"target_hit"`` also fires for the intermediate rungs TP1..TP3. Reaching
   rung 1 is not a win; only the contract target (``cyc.tp``) is.

This module answers one question, in one place, the way the rest of the repo
already grades tape (``app/hold_era.py``, ``app/predlock.py``): which barrier
was touched FIRST, and did it end the trade in profit or loss.
"""
from __future__ import annotations

from typing import Any, Iterable

#: event types that terminate a position
SL_TYPES = ("sl_hit", "sl_crossed")


def _level_matches(level: Any, target: Any) -> bool:
    """Is this rung the contract target? (float-safe, parity with engine)"""
    try:
        a, b = float(level), float(target)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= max(1e-7, abs(b) * 1e-10)


def resolve(events: Iterable[dict], *, tp: Any = None) -> dict:
    """Return the first terminal barrier among ``events``.

    ``events`` is a cycle's ordered event list. Returns::

        {"resolved": bool, "win": bool, "loss": bool, "kind": str,
         "ts": float | None}

    ``kind`` is ``"tp"``, ``"sl"`` or ``""`` (still open). Ties on the same
    timestamp resolve to the stop, matching ``hold_era`` ("SL wins same-bar
    ties") — the pessimistic reading, which is the only defensible one.
    """
    seen: list[tuple[float, str]] = []
    for e in events or ():
        et = e.get("type")
        if et in SL_TYPES:
            seen.append((float(e.get("ts") or 0.0), "sl"))
        elif et == "target_hit":
            # a rung only counts if it IS the contract target; when no
            # contract target is known, fall back to treating the rung as
            # terminal (the old behaviour) rather than silently dropping it
            if tp is None or _level_matches(e.get("level"), tp):
                seen.append((float(e.get("ts") or 0.0), "tp"))
    if not seen:
        return {"resolved": False, "win": False, "loss": False,
                "kind": "", "ts": None}
    seen.sort(key=lambda p: (p[0], 0 if p[1] == "sl" else 1))
    ts, kind = seen[0]
    return {"resolved": True, "win": kind == "tp", "loss": kind == "sl",
            "kind": kind, "ts": ts}
