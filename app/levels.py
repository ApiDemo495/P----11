"""Locked-entry TP/SL levels and tamper-evident cycle contracts.

A prediction's entry, side, TP and SL are frozen together at lock time.
Display, grading and execution consume that same contract; no caller may
silently repair it from the latest spot price.  The older ``freeze`` /
``guard`` helpers remain for measured previews and legacy cycles.
"""
from __future__ import annotations

import hashlib
import json
import math

CONTRACT_VERSION = 1
MAX_RR = 4.0


def _number(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _digest(payload: dict) -> str:
    wire = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       allow_nan=False).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()[:24]


def make_lock_id(lock: dict | None) -> str | None:
    """Stable identity for one entry/window lock (not its mutable signal)."""
    if not isinstance(lock, dict):
        return None
    symbol, tf = str(lock.get("symbol") or ""), str(lock.get("tf") or "")
    entry = _number(lock.get("price", lock.get("entry")))
    locked_at = _number(lock.get("locked_at"))
    if not symbol or not tf or entry is None or entry <= 0 \
            or locked_at is None or locked_at <= 0:
        return None
    start = _number(lock.get("window_start"))
    end = _number(lock.get("window_end"))
    relocks = lock.get("relocks", 0)
    try:
        relocks = int(relocks)
    except (TypeError, ValueError, OverflowError):
        return None
    if relocks < 0:
        return None
    payload = {"symbol": symbol, "tf": tf, "entry": round(entry, 8),
               "locked_at": round(locked_at, 6),
               "window_start": round(start if start is not None else locked_at, 6),
               "window_end": round(end if end is not None else locked_at, 6),
               "relocks": relocks}
    return "lock-" + _digest(payload)


def lock_id_valid(lock: dict | None) -> bool:
    return bool(isinstance(lock, dict) and lock.get("lock_id")
                and lock.get("lock_id") == make_lock_id(lock))


def freeze(entry: float, direction: int, block: dict | None) -> dict | None:
    """Create levels from a measured bps block at the locked entry."""
    entry = _number(entry)
    if not block or entry is None or entry <= 0 or not direction:
        return None
    tp_bps = _number(block.get("tp_bps"))
    sl_bps = _number(block.get("sl_bps"))
    if tp_bps is None or sl_bps is None or tp_bps <= 0 or sl_bps <= 0:
        return None
    d = 1 if direction > 0 else -1
    tp = entry * (1.0 + d * tp_bps / 1e4)
    sl = entry * (1.0 - d * sl_bps / 1e4)
    out = {"tp": round(tp, 8), "sl": round(sl, 8),
           "tp_dist": round(abs(tp - entry), 8),
           "sl_dist": round(abs(entry - sl), 8),
           "ratio": round(tp_bps / sl_bps, 6),
           "entry": entry, "direction": d}
    return out if valid(out) else None


def valid(levels: dict | None) -> bool:
    """The invariant: TP and SL on their correct sides of the entry."""
    if not isinstance(levels, dict):
        return False
    e = _number(levels.get("entry"))
    tp, sl = _number(levels.get("tp")), _number(levels.get("sl"))
    try:
        d = int(levels.get("direction") or 0)
    except (TypeError, ValueError, OverflowError):
        return False
    if e is None or e <= 0 or tp is None or sl is None or d not in (-1, 1):
        return False
    return (tp > e > sl) if d > 0 else (tp < e < sl)


def repair(levels: dict, block: dict | None) -> dict | None:
    """Rebuild from measured bps at the original entry, never current spot."""
    if not isinstance(levels, dict):
        return None
    entry = _number(levels.get("entry"))
    try:
        direction = int(levels.get("direction") or 0)
    except (TypeError, ValueError, OverflowError):
        return None
    if entry is None or entry <= 0 or direction not in (-1, 1):
        return None
    return freeze(entry, direction, block)


def guard(levels: dict | None, block: dict | None) -> dict | None:
    """Legacy display gate. Locked contracts use ``contract_valid`` instead."""
    if valid(levels):
        return levels
    if levels and levels.get("entry") and levels.get("direction"):
        return repair(levels, block)
    return None


def make_contract(levels: dict | None, *, symbol: str, timeframe: str,
                  cycle: str, side: int, revision: int, lock_id: str,
                  source: str = "measured-moves") -> dict | None:
    """Freeze one price pair and bind it to cycle, side, revision and lock."""
    if not valid(levels) or not symbol or not timeframe or not cycle or not lock_id:
        return None
    try:
        side, revision = int(side), int(revision)
    except (TypeError, ValueError, OverflowError):
        return None
    if side not in (-1, 1) or revision < 1 or side != int(levels["direction"]):
        return None
    entry, tp, sl = (float(levels[k]) for k in ("entry", "tp", "sl"))
    tp_dist, sl_dist = abs(tp - entry), abs(entry - sl)
    if sl_dist <= 0:
        return None
    ratio = tp_dist / sl_dist
    if not math.isfinite(ratio) or ratio <= 0 or ratio > MAX_RR + 1e-6:
        return None
    contract = {
        "version": CONTRACT_VERSION, "valid": True,
        "symbol": str(symbol), "timeframe": str(timeframe),
        "cycle_id": str(cycle), "side": side, "revision": revision,
        "lock_id": str(lock_id), "entry": round(entry, 8),
        "tp": round(tp, 8), "sl": round(sl, 8),
        "tp_dist": round(tp_dist, 8), "sl_dist": round(sl_dist, 8),
        "ratio": round(ratio, 6), "below_floor": ratio < 1.0 - 1e-6,
        "source": str(source),
    }
    contract["id"] = "contract-" + _digest(contract)
    return contract


def contract_valid(contract: dict | None, *, symbol: str | None = None,
                   timeframe: str | None = None, cycle: str | None = None,
                   side: int | None = None, revision: int | None = None,
                   lock_id: str | None = None, entry: float | None = None) -> bool:
    if not isinstance(contract, dict) or contract.get("version") != CONTRACT_VERSION \
            or contract.get("valid") is not True:
        return False
    try:
        actual_side, actual_revision = int(contract.get("side")), int(contract.get("revision"))
    except (TypeError, ValueError, OverflowError):
        return False
    lv = {"entry": contract.get("entry"), "tp": contract.get("tp"),
          "sl": contract.get("sl"), "direction": actual_side}
    if not valid(lv) or actual_revision < 1:
        return False
    try:
        ratio = abs(float(contract["tp"]) - float(contract["entry"])) \
            / abs(float(contract["entry"]) - float(contract["sl"]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    stored_ratio = _number(contract.get("ratio"))
    stored_tp_dist = _number(contract.get("tp_dist"))
    stored_sl_dist = _number(contract.get("sl_dist"))
    if any(value is None for value in (stored_ratio, stored_tp_dist, stored_sl_dist)):
        return False
    if (not math.isfinite(ratio) or ratio > MAX_RR + 1e-6
            or abs(ratio - stored_ratio) > 1e-5):
        return False
    tp_dist = abs(float(contract["tp"]) - float(contract["entry"]))
    sl_dist = abs(float(contract["entry"]) - float(contract["sl"]))
    if (abs(tp_dist - float(contract.get("tp_dist", -1))) > max(1e-7, tp_dist * 1e-9)
            or abs(sl_dist - float(contract.get("sl_dist", -1))) > max(1e-7, sl_dist * 1e-9)):
        return False
    if contract.get("below_floor") is not (ratio < 1.0 - 1e-6):
        return False
    checks = ((symbol, contract.get("symbol")),
              (timeframe, contract.get("timeframe")),
              (cycle, contract.get("cycle_id")),
              (side, actual_side), (revision, actual_revision),
              (lock_id, contract.get("lock_id")))
    if any(expected is not None and expected != actual
           for expected, actual in checks):
        return False
    if entry is not None:
        e = _number(entry)
        if e is None or abs(e - float(contract["entry"])) > max(1e-7, abs(e) * 1e-10):
            return False
    payload = {k: v for k, v in contract.items() if k != "id"}
    try:
        expected_id = "contract-" + _digest(payload)
    except (TypeError, ValueError, OverflowError):
        return False
    return contract.get("id") == expected_id


def validate_cycle_contract(cycle_state, contract_lock: dict | None = None,
                            contract: dict | None = None) -> dict | None:
    """Validate the cycle's exact frozen pair against its lock fingerprint."""
    contract = contract if contract is not None else getattr(cycle_state, "level_contract", None)
    contract_lock = (contract_lock if contract_lock is not None
                     else getattr(cycle_state, "contract_lock", None))
    if not lock_id_valid(contract_lock):
        return None
    if contract_lock.get("symbol") != getattr(cycle_state, "symbol", None) \
            or contract_lock.get("tf") != getattr(cycle_state, "tf", None):
        return None
    if not contract_valid(
            contract, symbol=getattr(cycle_state, "symbol", None),
            timeframe=getattr(cycle_state, "tf", None),
            cycle=getattr(cycle_state, "cycle_id", None),
            side=getattr(cycle_state, "direction", None),
            revision=getattr(cycle_state, "level_revision", None),
            lock_id=contract_lock.get("lock_id"),
            entry=contract_lock.get("price")):
        return None
    return contract


def unavailable_contract(*, symbol: str, timeframe: str, cycle: str | None,
                         reason: str, lock_id: str | None = None,
                         side: int | None = None, revision: int | None = None) -> dict:
    return {"version": CONTRACT_VERSION, "valid": False,
            "symbol": symbol, "timeframe": timeframe, "cycle_id": cycle,
            "lock_id": lock_id, "side": side, "revision": revision,
            "reason": reason}


def from_cycle(cyc) -> dict | None:
    """Return the validated frozen contract, or legacy frozen levels."""
    contract = getattr(cyc, "level_contract", None)
    if contract is not None:
        checked = validate_cycle_contract(cyc)
        return ({**checked, "direction": checked["side"]}
                if checked is not None else None)
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
         "entry": entry, "direction": direction}, risk or None)
