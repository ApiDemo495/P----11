"""Forward expectancy tables (advice §2.7): per formula × per horizon
(the timeframe IS the horizon here), the empirical P(up | reading bucket),
settled strict-sign on the real tape.

This is how a formula built for one horizon speaks at another: the fused
score uses forward P(up) − ½ (once n ≥ 30 per bucket) instead of the raw
value — no rescaling of the formula body, re-learned per horizon.

Rows settle from the lock's frozen formula values (fv) against the window's
strict-sign outcome. Persisted to data/forward.json (atomic).
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from typing import Any

BUCKETS = (-1.0, -0.5, 0.0, 0.5, 1.0)
MIN_N_USE = 30          # advice: n ≥ 30 per bucket before the table speaks


def _bucket(value: float) -> float:
    v = max(-1.0, min(1.0, float(value)))
    return min(BUCKETS, key=lambda b: abs(b - v))


class ForwardTables:
    """rows[(fid, symbol, tf, bucket)] = {"up": int, "n": int}"""

    def __init__(self, path: str = "data/forward.json") -> None:
        self.path = path
        self.rows: dict[tuple[int, str, str, float], dict[str, int]] = {}
        self._lock = threading.RLock()
        self._record_ids: set[str] = set()
        self._record_id_order: list[str] = []
        self._load()

    # ── persistence ────────────────────────────────────────────────────
    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
            for key, cell in (raw.get("rows") or {}).items():
                fid, sym, tf, bucket = key.rsplit("|", 3)
                self.rows[(int(fid), sym, tf, float(bucket))] = {
                    "up": int(cell.get("up", 0)), "n": int(cell.get("n", 0))}
            ids = raw.get("record_ids") or []
            self._record_id_order = [str(v) for v in ids if v][-10000:]
            self._record_ids = set(self._record_id_order)
        except (OSError, ValueError, KeyError, TypeError):
            self.rows = {}
            self._record_ids, self._record_id_order = set(), []

    def save(self) -> None:
        with self._lock:
            out: dict[str, Any] = {"rows": {},
                                   "record_ids": self._record_id_order[-10000:]}
            for (fid, sym, tf, b), cell in self.rows.items():
                out["rows"][f"{fid}|{sym}|{tf}|{b}"] = dict(cell)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path) or ".",
                                   suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(out, fh, allow_nan=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    # ── recording (called at settlement) ───────────────────────────────
    def record(self, symbol: str, tf: str, fv: dict[int, float],
               entry: float, settle: float,
               record_id: str | None = None) -> int:
        """One settled window: bucket frozen readings, credit P(up).

        ``record_id`` makes live settlement retries and verified-history
        seeding idempotent across both this process and restarts.
        """
        if not fv or not entry or not settle:
            return 0
        up = 1 if settle > entry else 0        # strict: tie is NOT up
        rid = str(record_id)[:240] if record_id else None
        with self._lock:
            if rid and rid in self._record_ids:
                return 0
            old_cells: dict[tuple[int, str, str, float], dict | None] = {}
            n = 0
            for fid, value in fv.items():
                if not isinstance(fid, (int, str)):
                    continue
                fid_i = int(fid)
                if fid_i == 20:                  # DRHE is a confidence gauge
                    continue
                key = (fid_i, symbol, tf, _bucket(float(value)))
                if key not in old_cells:
                    old_cells[key] = (dict(self.rows[key]) if key in self.rows
                                      else None)
                cell = self.rows.setdefault(key, {"up": 0, "n": 0})
                cell["n"] += 1
                cell["up"] += up
                n += 1
            if not n:
                return 0
            if rid:
                self._record_ids.add(rid)
                self._record_id_order.append(rid)
                while len(self._record_id_order) > 10000:
                    expired = self._record_id_order.pop(0)
                    self._record_ids.discard(expired)
            try:
                self.save()
            except Exception:
                for key, old in old_cells.items():
                    if old is None:
                        self.rows.pop(key, None)
                    else:
                        self.rows[key] = old
                if rid:
                    self._record_ids.discard(rid)
                    try:
                        self._record_id_order.remove(rid)
                    except ValueError:
                        pass
                raise
            return n

    # ── reading (fusion + UI) ──────────────────────────────────────────
    def p_up(self, fid: int, symbol: str, tf: str, value: float) -> tuple[float | None, int]:
        """(P(up), n) for a reading at its bucket — None until n ≥ MIN_N_USE."""
        cell = self.rows.get((fid, symbol, tf, _bucket(value)))
        if not cell or cell["n"] < MIN_N_USE:
            return None, 0 if not cell else cell["n"]
        return cell["up"] / cell["n"], cell["n"]

    def payload(self, symbol: str | None = None) -> dict:
        out: dict[str, Any] = {"min_n": MIN_N_USE, "buckets": {}, "tables": {}}
        for (fid, sym, tf, b), cell in self.rows.items():
            if symbol and sym != symbol:
                continue
            out["buckets"].setdefault(f"F{fid} {sym} {tf}", []).append({
                "bucket": b, "n": cell["n"],
                "p_up": round(cell["up"] / cell["n"], 4) if cell["n"] else None,
                "settled": cell["n"] >= MIN_N_USE,
            })
        return out
