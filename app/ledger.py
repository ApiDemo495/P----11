"""Durable settlement ledger with retry-safe first-touch grading.

Window direction and barrier order are separate outcomes. Only verified
resolved TP/SL touches contribute to first-touch percentages; ambiguous,
unavailable and neither-touch windows stay visible but are not denominator
cases. Seed rows use stable IDs so replay can be retried safely.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time

LEDGER_FILE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "ledger.json")
MIN_N_RATE = 20
TOUCH_STATES = {"target", "stop", "neither", "unavailable", "ambiguous"}
TOUCH_LABELS = {"target": "TP first", "stop": "SL first",
                "neither": "neither", "unavailable": "unavailable",
                "ambiguous": "ambiguous"}
LEGACY_TOUCH = {"TP first": "target", "TP": "target", "target": "target",
                "SL first": "stop", "SL": "stop", "stop": "stop",
                "neither": "neither", "unavailable": "unavailable",
                "ambiguous": "ambiguous"}


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _positive(value) -> float | None:
    number = _finite(value)
    return number if number is not None and number > 0 else None


class SettlementLedger:
    def __init__(self, path: str | None = None) -> None:
        if path is None and os.environ.get("PYTEST_CURRENT_TEST"):
            # Isolate test runs from the production ledger and from stale
            # rows left by an earlier pytest invocation.
            path = os.path.join(tempfile.gettempdir(),
                                f"predict4_test_ledger_{os.getpid()}.json")
        self.path = path or LEDGER_FILE
        self.rows: dict[str, list[dict]] = {}
        self.settle_calls = 0
        self.last_settle_ts = 0.0
        self.last_settle: dict = {}
        self.totals = {"win": 0, "loss": 0, "skip": 0}
        self._lock = threading.RLock()
        self.load()

    @staticmethod
    def _key(symbol: str, tf: str) -> str:
        return f"{symbol}|{tf}"

    def load(self) -> None:
        with self._lock:
            try:
                with open(self.path, encoding="utf-8") as fh:
                    data = json.load(fh)
                raw = data.get("rows") if isinstance(data, dict) else None
                if not isinstance(raw, dict):
                    raw = {}
                self.rows = {}
                for key, values in (raw or {}).items():
                    if not isinstance(values, list):
                        continue
                    valid_rows = []
                    for value in values:
                        if not isinstance(value, dict):
                            continue
                        row = dict(value)
                        if row.get("result") not in ("win", "loss", "skip"):
                            row["result"] = "skip"
                        valid_rows.append(row)
                    self.rows[str(key)] = valid_rows
            except (OSError, ValueError, TypeError):
                self.rows = {}
            self.settle_calls = 0
            self.last_settle_ts = 0.0
            self.last_settle = {}
            self.totals = {"win": 0, "loss": 0, "skip": 0}
            last_row, last_key = {}, ""
            for key, values in self.rows.items():
                for row in values:
                    result = row.get("result", "skip")
                    self.totals[result] = self.totals.get(result, 0) + 1
                    self.settle_calls += 1
                    stamp = _finite(row.get("ts")) or 0.0
                    if stamp >= self.last_settle_ts:
                        self.last_settle_ts = stamp
                        last_row, last_key = row, key
            if last_row:
                symbol, tf = (last_key.split("|") + [""])[:2]
                self.last_settle = {"symbol": symbol, "tf": tf,
                                    "result": last_row.get("result"),
                                    "side": last_row.get("side"),
                                    "entry": last_row.get("entry")}

    def save(self) -> bool:
        tmp = None
        with self._lock:
            try:
                directory = os.path.dirname(self.path) or "."
                os.makedirs(directory, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump({"rows": self.rows}, fh, allow_nan=False)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, self.path)
                return True
            except (OSError, TypeError, ValueError):
                if tmp:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                return False

    def settle(self, symbol: str, tf: str, entry: float, settle_price: float,
               side: int, window_end: float | None = None,
               level_outcome: str | None = None, seeded: bool = False,
               conf: float | None = None, record_id: str | None = None,
               first_touch: str | None = None,
               barriers_available: bool | None = None,
               ambiguous: bool = False) -> dict:
        """Record one settlement; ``record_id`` makes inserts idempotent."""
        entry_value, settle_value = _positive(entry), _positive(settle_price)
        try:
            side_value = int(side)
        except (TypeError, ValueError, OverflowError):
            side_value = 0
        side_value = 1 if side_value > 0 else (-1 if side_value < 0 else 0)
        if side_value == 0 or entry_value is None or settle_value is None:
            result = "skip"
        else:
            move = settle_value - entry_value
            result = ("loss" if move == 0.0 else
                      "win" if (move > 0) == (side_value > 0) else "loss")

        if (not isinstance(first_touch, str) or first_touch not in TOUCH_STATES):
            first_touch = (LEGACY_TOUCH.get(level_outcome)
                           if isinstance(level_outcome, str) else None)
        if barriers_available is None:
            barriers_available = first_touch in {
                "target", "stop", "neither", "ambiguous"}
        barriers_available = bool(barriers_available)
        if not barriers_available:
            first_touch, ambiguous = "unavailable", False
        elif ambiguous or first_touch == "ambiguous":
            first_touch, ambiguous = "ambiguous", True
        elif first_touch is None:
            first_touch = "unavailable"
        else:
            ambiguous = False
        level_outcome = TOUCH_LABELS[first_touch]
        end_value = _finite(window_end)
        try:
            confidence = float(conf) if conf is not None else None
            if confidence is not None and not math.isfinite(confidence):
                confidence = None
        except (TypeError, ValueError, OverflowError):
            confidence = None
        rid = str(record_id)[:240] if record_id else None
        row = {
            "ts": time.time(), "window_end": end_value,
            "entry": round(entry_value, 6) if entry_value is not None else None,
            "settle": round(settle_value, 6) if settle_value is not None else None,
            "side": side_value, "result": result, "seeded": bool(seeded),
            "conf": round(confidence, 4) if confidence is not None else None,
            "level_outcome": str(level_outcome)[:40] if level_outcome else None,
            "first_touch": first_touch,
            "barriers_available": bool(barriers_available),
            "ambiguous": bool(ambiguous),
        }
        if rid:
            row["record_id"] = rid
        key = self._key(str(symbol), str(tf))
        with self._lock:
            old_rows = list(self.rows.get(key, []))
            if rid:
                existing = next((saved for saved in old_rows
                                 if saved.get("record_id") == rid), None)
                if existing is not None:
                    return {**existing, "_duplicate": True, "_persisted": True}
            bucket = self.rows.setdefault(key, [])
            bucket.append(row)
            if len(bucket) > 500:
                del bucket[:-500]
            if not self.save():
                self.rows[key] = old_rows
                if not old_rows:
                    self.rows.pop(key, None)
                return {**row, "_persisted": False}
            self.settle_calls += 1
            self.last_settle_ts = row["ts"]
            self.last_settle = {"symbol": str(symbol), "tf": str(tf),
                                "result": result, "side": side_value,
                                "entry": row["entry"]}
            self.totals[result] = self.totals.get(result, 0) + 1
            return {**row, "_persisted": True}

    def stats(self, symbol: str, tf: str) -> dict:
        rows = self.rows.get(self._key(symbol, tf), [])
        graded = [r for r in rows if r.get("result") in ("win", "loss")]
        wins = sum(r.get("result") == "win" for r in graded)
        losses = len(graded) - wins
        skipped = sum(r.get("result") == "skip" for r in rows)
        resolved = []
        touch_counts = {state: 0 for state in TOUCH_STATES}
        for row in rows:
            touch = row.get("first_touch")
            if not isinstance(touch, str) or touch not in TOUCH_STATES:
                legacy = row.get("level_outcome")
                touch = LEGACY_TOUCH.get(legacy) if isinstance(legacy, str) else None
            if touch in TOUCH_STATES:
                touch_counts[touch] += 1
            if touch in ("target", "stop"):
                resolved.append(touch)
        streak = best = cur = 0
        for row in reversed(graded):
            if row.get("result") == "win":
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        n = len(graded)
        bps = []
        for row in graded:
            entry_value, settle_value = _positive(row.get("entry")), _positive(row.get("settle"))
            try:
                side_value = int(row.get("side") or 0)
            except (TypeError, ValueError, OverflowError):
                continue
            if entry_value and settle_value and side_value:
                bps.append(side_value * (settle_value / entry_value - 1) * 1e4)
        wins_bps = [value for value in bps if value > 0]
        loss_bps = [-value for value in bps if value < 0]
        target_first = resolved.count("target")
        return {
            "exp_bps": round(sum(bps) / len(bps), 2) if bps else None,
            "payoff": (round((sum(wins_bps) / len(wins_bps)) /
                             (sum(loss_bps) / len(loss_bps)), 3)
                       if wins_bps and loss_bps else None),
            "tp_first_n": len(resolved),
            "tp_first_pct": round(target_first / len(resolved), 4)
                             if resolved else None,
            "first_touch": touch_counts,
            "n": n, "wins": wins, "losses": losses,
            "ties": 0, "seeded": sum(bool(r.get("seeded")) for r in rows),
            "skipped": skipped,
            "rate": round(wins / n, 4) if n else None,
            "enough": n >= MIN_N_RATE,
            "streak": cur, "best_streak": best,
            "last20": [r.get("result") for r in graded[-20:]],
        }

    def pooled(self, tf: str) -> dict:
        wins = losses = skipped = 0
        for key, rows in self.rows.items():
            if not key.endswith(f"|{tf}"):
                continue
            for row in rows:
                if row.get("result") == "win":
                    wins += 1
                elif row.get("result") == "loss":
                    losses += 1
                else:
                    skipped += 1
        n = wins + losses
        return {"n": n, "wins": wins, "losses": losses, "skipped": skipped,
                "rate": round(wins / n, 4) if n else None,
                "enough": n >= MIN_N_RATE}

    def api_payload(self, symbols: list[str], tfs: list[str]) -> dict:
        per = {f"{symbol}|{tf}": self.stats(symbol, tf)
               for symbol in symbols for tf in tfs}
        pooled = {tf: self.pooled(tf) for tf in tfs}
        return {"per": per, "pooled": pooled, "min_n": MIN_N_RATE,
                "rule": "strict sign · tie = loss · skips never count as wins",
                "settle_calls": self.settle_calls,
                "last_settle_ago_s": (round(time.time() - self.last_settle_ts, 1)
                                      if self.last_settle_ts else None),
                "last_settle": self.last_settle,
                "totals": dict(self.totals),
                "rows_total": sum(len(values) for values in self.rows.values())}
