"""Canonical window contract — ONE authoritative level/confidence object.

Why this module exists (principal audit §1): the DASH tab showed four
different entry prices, four different TP ladders and two confidences for
the *same* BTC · TF1 window, because every panel recomputed levels from
its own formula:

  * ``engine._project_level``   entry + dir·|conf|·ATR·√min·0.5
    → with conf ≈ 0.02 that is an $8 "target" (the TP1 = entry + $4 bug)
  * ``moves_risk.risk_block``   measured q80-MAE stop / q50..q95 target
  * ``levels.freeze``           a third bps block
  * the UI cards                whatever the last GRC happened to write

``build_contract`` freezes one immutable object per (asset, timeframe,
cycle) carrying direction, entry, the TF-horizon target, the TP1..TP4
rungs, SL, ATR, both confidences, a content digest and the lock time.

Every rung is a strict, monotone fraction of the entry→TF-TP span, so the
invariant holds for BOTH sides by construction::

    BUY :  ENTRY < TP1 < TP2 < TP3 < TF-TP      and  SL < ENTRY
    SELL:  ENTRY > TP1 > TP2 > TP3 > TF-TP      and  SL > ENTRY

The rung fractions blend diffusion time (√t, the honest scaling of a
random walk's displacement) with linear time, so TP1 sits meaningfully
away from the entry while the last rung is exactly the TF target.  A
sub-timeframe never gets its own competing TP ladder any more.

Nothing downstream may recompute a level — display, grading, sizing and
execution all read this object.
"""
from __future__ import annotations

import hashlib
import json
import math

CONTRACT_VERSION = 2
#: smallest rung-to-rung step, as a fraction of the entry→TF-TP span
MIN_RUNG_STEP = 0.02
#: smallest rung offset from the entry, as a fraction of the same span
MIN_FIRST_RUNG = 0.04


def clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def _num(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def rung_fractions(minutes, horizon_minutes) -> list[float]:
    """Strictly increasing rung weights in (0, 1]; the last one is 1.0.

    ``√t`` (diffusion) plus ``t`` (trend persistence): a 5-minute rung of
    a 22-minute window lands at ~0.36 of the span, not at 0.05, and the
    ladder is monotone in price by construction.
    """
    mins = [float(m) for m in minutes if m is not None and float(m) > 0]
    if not mins:
        return []
    horizon = max(1.0, float(horizon_minutes or mins[-1] or 1.0))
    out: list[float] = []
    for m in mins:
        t = clamp(m / horizon, 0.0, 1.0)
        out.append(clamp(0.55 * math.sqrt(t) + 0.45 * t,
                         MIN_FIRST_RUNG, 1.0))
    # strictly increasing …
    for i in range(1, len(out)):
        floor = out[i - 1] + MIN_RUNG_STEP
        if out[i] < floor:
            out[i] = floor
    # … ending exactly at the horizon (the TF target itself)
    out[-1] = 1.0
    # … and still increasing after that clamp, walking backwards
    for i in range(len(out) - 2, -1, -1):
        ceiling = out[i + 1] - MIN_RUNG_STEP
        if out[i] > ceiling:
            out[i] = max(MIN_FIRST_RUNG * 0.5, ceiling)
    return [round(v, 6) for v in out]


def build_ladder(entry: float, direction: int, horizon_tp: float,
                 minutes, horizon_minutes=None) -> list[dict]:
    """TP1..TPn rungs strictly between the entry and the TF target."""
    e = _num(entry)
    tp = _num(horizon_tp)
    try:
        d = 1 if int(direction) >= 0 else -1
    except (TypeError, ValueError):
        return []
    if e is None or e <= 0 or tp is None or tp <= 0:
        return []
    span = tp - e
    if abs(span) <= 0 or (span > 0) != (d > 0):
        return []
    mins = [int(m) for m in minutes if m]
    if not mins:
        return []
    fr = rung_fractions(mins, horizon_minutes if horizon_minutes is not None
                        else mins[-1])
    if len(fr) != len(mins):
        return []
    rows: list[dict] = []
    for k, (m, f) in enumerate(zip(mins, fr), start=1):
        level = e + span * f if k < len(mins) else tp
        rows.append({
            "k": k,
            "minutes": int(m),
            "level": round(float(level), 8),
            "fraction": round(float(f), 6),
            "dist_bps": round(abs(level - e) / e * 1e4, 2),
            "is_horizon": k == len(mins),
        })
    return rows if ladder_valid(e, d, tp, rows) else []


def ladder_valid(entry: float, direction: int, horizon_tp: float,
                 rows: list[dict]) -> bool:
    """Monotone, strictly between the entry and the TF target."""
    if not rows:
        return False
    d = 1 if direction > 0 else -1
    prev = float(entry)
    for row in rows:
        lvl = _num(row.get("level"))
        if lvl is None:
            return False
        if d > 0 and not (lvl > prev and lvl <= horizon_tp + 1e-9):
            return False
        if d < 0 and not (lvl < prev and lvl >= horizon_tp - 1e-9):
            return False
        prev = lvl
    last = _num(rows[-1].get("level"))
    return last is not None and abs(last - horizon_tp) <= max(1e-8, abs(horizon_tp) * 1e-10)


def _digest(payload: dict) -> str:
    wire = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()[:24]


def build_contract(*, symbol: str, timeframe: str, direction: int,
                   entry: float, horizon_tp: float, sl: float,
                   minutes, atr: float = 0.0,
                   base_confidence: float = 0.0,
                   final_confidence: float = 0.0,
                   cci_adjusted: float = 0.0,
                   cycle_id: str | None = None,
                   locked_at: float | None = None,
                   source: str = "measured-moves") -> dict:
    """Freeze the one canonical object for this window.

    Returns ``{}`` when the inputs cannot form a valid contract — callers
    must treat that as "no contract" and never invent levels.
    """
    e = _num(entry)
    tp = _num(horizon_tp)
    stop = _num(sl)
    try:
        d = 1 if int(direction) >= 0 else -1
    except (TypeError, ValueError):
        return {}
    if e is None or e <= 0 or tp is None or stop is None or tp <= 0 or stop <= 0:
        return {}
    # the SL must sit on the opposite side of the entry
    if (d > 0 and not stop < e) or (d < 0 and not stop > e):
        return {}
    rows = build_ladder(e, d, tp, minutes)
    if not rows:
        return {}
    ladder = {f"tp{row['k']}": row["level"] for row in rows[:-1]}
    payload = {
        "version": CONTRACT_VERSION,
        "symbol": str(symbol or ""),
        "timeframe": str(timeframe or ""),
        "cycle_id": str(cycle_id or ""),
        "direction": d,
        "entry": round(e, 8),
        "horizon_tp": round(tp, 8),
        "sl": round(stop, 8),
        "atr": round(float(_num(atr) or 0.0), 6),
        "rungs": rows,
        "tp1": ladder.get("tp1"),
        "tp2": ladder.get("tp2"),
        "tp3": ladder.get("tp3"),
        "sl_dist_bps": round(abs(e - stop) / e * 1e4, 2),
        "tp_dist_bps": round(abs(tp - e) / e * 1e4, 2),
        "rr": round(abs(tp - e) / abs(e - stop), 4) if abs(e - stop) > 0 else None,
        "base_confidence": round(float(_num(base_confidence) or 0.0), 4),
        "final_confidence": round(float(_num(final_confidence) or 0.0), 4),
        "cci_adjusted": round(float(_num(cci_adjusted) or 0.0), 6),
        "locked_at": float(locked_at) if locked_at else None,
        "source": str(source),
    }
    payload["id"] = "cwc-" + _digest(payload)
    return payload


def validate(contract: dict | None) -> bool:
    """Re-derive the ladder and digest — a tampered contract is not valid."""
    if not isinstance(contract, dict) or not contract:
        return False
    if contract.get("version") != CONTRACT_VERSION or not contract.get("id"):
        return False
    e = _num(contract.get("entry"))
    tp = _num(contract.get("horizon_tp"))
    stop = _num(contract.get("sl"))
    if e is None or tp is None or stop is None:
        return False
    try:
        d = int(contract.get("direction"))
    except (TypeError, ValueError):
        return False
    if d not in (-1, 1):
        return False
    rows = contract.get("rungs") or []
    if not ladder_valid(e, d, tp, rows):
        return False
    if (d > 0 and not stop < e) or (d < 0 and not stop > e):
        return False
    stored = contract.get("tp1")
    if stored is not None and abs(float(stored) - float(rows[0]["level"])) > 1e-8:
        return False
    payload = {k: v for k, v in contract.items() if k != "id"}
    try:
        return contract["id"] == "cwc-" + _digest(payload)
    except (TypeError, ValueError):
        return False

# [removed 2026-11-23] levels_for_display — dead code: zero references anywhere


def divergences(panels: dict, contract: dict | None,
                tol_bps: float = 1.0) -> dict:
    """Which panels disagree with the canonical contract (audit helper).

    ``panels`` maps a panel name to a dict that may carry ``entry`` /
    ``tp`` / ``sl``.  A panel is 'in sync' when every level it shows is
    within ``tol_bps`` of the canonical value, so the DASH tab can prove
    the desync is gone instead of asserting it.
    """
    if not isinstance(contract, dict) or not contract:
        return {}
    ref = {"entry": contract.get("entry"), "tp": contract.get("horizon_tp"),
           "sl": contract.get("sl")}
    out: dict[str, list[str]] = {}
    for name, panel in (panels or {}).items():
        if not isinstance(panel, dict):
            continue
        bad = []
        for key, want in ref.items():
            got = _num(panel.get(key))
            if got is None or want is None or not want:
                continue
            if abs(got - want) / abs(want) * 1e4 > tol_bps:
                bad.append(key)
        out[name] = bad
    return out


def rung_level(entry: float, direction: int, horizon_tp: float,
               minutes, horizon_minutes=None, k: int = 1,
               sl: float | None = None) -> float | None:
    """The canonical price of rung ``k`` — and never a price past the stop.

    The GRC refresh used to re-project rung 1 with
    ``price + dir·|conf|·ATR·√minutes·SCALE`` from the CURRENT price, with no
    reference to the contract at all. That is unbounded drift: on the live
    lock file 11 of 16 lanes carried a rung 1 on the WRONG SIDE of their own
    entry (BNB tf4: short at 443.06, rung 1 at 493.50, stop at 466.46), and
    the drifted level was then written back onto the rung's direction — so
    the lane's displayed target argued with the lane's own contract.

    Rungs belong to the frozen entry → horizon-TP span (``build_ladder``);
    a refresh may only move them along that span. If ``sl`` is supplied the
    level is additionally clamped inside (entry, sl) so a rung can never sit
    beyond the barrier that ends the trade.
    """
    e = _num(entry)
    tp = _num(horizon_tp)
    if e is None or e <= 0 or tp is None or tp <= 0:
        return None
    d = 1 if int(direction or 0) >= 0 else -1
    mins = [int(m) for m in (minutes or []) if m]
    if not mins:
        return None
    rows = build_ladder(e, d, tp, mins, horizon_minutes)
    if not rows:
        return None
    row = rows[min(max(int(k), 1), len(rows)) - 1]
    level = _num(row.get("level"))
    if level is None:
        return None
    stop = _num(sl)
    if stop is not None and stop > 0:
        # A rung is unreachable if the STOP is already inside it: for a long
        # the stop sits BELOW the entry, for a short ABOVE it. Keep a hair on
        # the profitable side of the stop, and never place a rung on the wrong
        # side of its own entry.
        head = abs(stop - e) * 0.02
        if d > 0 and level <= stop:                 # long: rung behind the stop
            level = max(e, stop + head)
        elif d < 0 and level >= stop:               # short: rung behind the stop
            level = min(e, stop - head)
    return float(level)


def clamp_rung(level: float | None, entry: float, direction: int,
               sl: float | None = None, fallback: float | None = None,
               fallback_sl: float | None = None) -> float | None:
    """Keep a projected rung inside (entry, stop] on the trade's own side.

    Rung 1 refreshes at every GRC by design (it is the lowest lock
    strength), so its price is allowed to move — but it must stay on the
    profitable side of the entry and reachable before the stop. Without this
    the refresh wrote rung 1 to the WRONG side of the trade: 11 of 16 lanes
    in data/locks.json carried a target that argued with their own contract
    (BNB tf4: short at 443.06, rung 1 at 493.50, stop at 466.46).
    """
    lv = _num(level)
    e = _num(entry)
    if lv is None or e is None or e <= 0:
        return None
    d = 1 if int(direction or 0) >= 0 else -1
    stop = _num(sl)
    # The stop always sits on the LOSING side of the entry (below it for a
    # long, above it for a short), so a rung is unreachable when it is not
    # beyond the entry AND beyond the stop in the profitable direction.
    bad = (lv <= e) if d > 0 else (lv >= e)
    if not bad and stop is not None and stop > 0:
        bad = (lv <= stop) if d > 0 else (lv >= stop)
    if not bad:
        return lv
    # Prefer the canonical ladder point for this rung. Never return the entry
    # itself: a rung AT the entry is an instant zero-profit "hit", which the
    # lock path then treats as a target that was reached.
    fb = _num(fallback)
    if fb is not None and ((fb > e) if d > 0 else (fb < e)):
        fstop = _num(fallback_sl if fallback_sl is not None else sl)
        if (fstop is None or fstop <= 0
                or ((fb > fstop) if d > 0 else (fb < fstop))):
            return fb
    if stop is not None and stop > 0:
        return e + d * abs(stop - e) * 0.5
    return e + d * abs(e) * 1e-4 if e else None
