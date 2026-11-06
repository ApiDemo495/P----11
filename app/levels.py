"""levels.py — the ONE source of truth for TP/SL levels (rewrite).

Rule, with no exceptions anywhere in the app:

    a level belongs to the prediction that created it, anchored at that
    prediction's LOCKED entry price, and frozen. BUY:  TP > entry > SL.
    SELL: TP < entry < SL.  Always. At issue, at every snapshot, after
    any re-lock, in every panel — hero, calls table, quad cards, chart.

Nothing in this system ever derives a displayed level from the CURRENT
price: a stop recomputed as "spot minus ATR" after the price has risen
can land ABOVE a BUY entry (the exact live bug this replaces: entry
75,460 showed SL 75,606). The cycle's frozen risk block (measured moves,
frozen at the lock in fresh_cycle) is the only input; this module turns
it into display levels and GUARDS every emission so a wrong-side or
missing level can never reach the screen — it is repaired against the
locked entry or withheld, never invented from spot.
"""

from __future__ import annotations


def freeze(entry: float, direction: int, block: dict | None) -> dict | None:
    """Display levels from a frozen measured-moves block + locked entry.

    Returns dict(tp, sl, tp_dist, sl_dist, ratio, entry, direction) or None
    when there is no usable block/entry — callers must then show "levels
    pending", never fabricate from the live price.
    """
    if not block or not entry or not direction:
        return None
    tp_bps = float(block.get("tp_bps") or 0.0)
    sl_bps = float(block.get("sl_bps") or 0.0)
    if tp_bps <= 0 or sl_bps <= 0:
        return None
    d = 1 if direction > 0 else -1
    tp = entry * (1.0 + d * tp_bps / 1e4)
    sl = entry * (1.0 - d * sl_bps / 1e4)
    out = {
        "tp": round(tp, 8), "sl": round(sl, 8),
        "tp_dist": round(abs(tp - entry), 8), "sl_dist": round(abs(entry - sl), 8),
        "ratio": round(tp_bps / sl_bps, 3) if sl_bps else None,
        "entry": float(entry), "direction": d,
    }
    return out if valid(out) else repair(out, block)


def valid(levels: dict | None) -> bool:
    """The invariant: TP and SL on their correct sides of the locked entry."""
    if not levels:
        return False
    e, d = levels.get("entry"), levels.get("direction") or 0
    tp, sl = levels.get("tp"), levels.get("sl")
    if not e or not d or tp is None or sl is None:
        return False
    return (tp > e > sl) if d > 0 else (tp < e < sl)


def repair(levels: dict, block: dict | None) -> dict | None:
    """Rebuild a wrong-side/missing level set from the block bps at the
    LOCKED entry. Never touches the current price."""
    if not levels or not levels.get("entry") or not levels.get("direction"):
        return None
    return freeze(float(levels["entry"]), int(levels["direction"]), block)


def guard(levels: dict | None, block: dict | None) -> dict | None:
    """Snapshot-time gate: pass valid levels through untouched; repair
    invalid ones from the frozen block; withhold (None) what cannot be
    repaired. A wrong-side level must never reach the UI."""
    if valid(levels):
        return levels
    if levels and levels.get("entry") and levels.get("direction"):
        return repair(levels, block)
    return None


def from_cycle(cyc) -> dict | None:
    """Frozen levels for a TFCycle: anchor = the risk block's lock entry
    (falling back to the cycle's targets anchor), direction = the locked
    side. This is the ONLY way display code obtains TP/SL."""
    risk = getattr(cyc, "risk", None) or {}
    entry = risk.get("entry") or getattr(cyc, "_levels_entry", None)
    if not entry:
        t = cyc.target(1) if hasattr(cyc, "target") else None
        entry = getattr(t, "anchor", None) or getattr(cyc, "entry", None)
    direction = int(getattr(cyc, "direction", 0) or 0)
    return guard(
        {"tp": getattr(cyc, "tp", None), "sl": getattr(cyc, "sl", None),
         "tp_dist": getattr(cyc, "tp_dist", None),
         "sl_dist": getattr(cyc, "sl_dist", None),
         "ratio": getattr(cyc, "ratio", None),
         "entry": entry, "direction": direction},
        risk or None,
    )
