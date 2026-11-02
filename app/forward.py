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
        except (OSError, ValueError, KeyError):
            self.rows = {}

    def save(self) -> None:
        out: dict[str, Any] = {"rows": {}}
        for (fid, sym, tf, b), cell in self.rows.items():
            out["rows"][f"{fid}|{sym}|{tf}|{b}"] = cell
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path) or ".",
                                   suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(out, fh)
        os.replace(tmp, self.path)

    # ── recording (called at settlement) ───────────────────────────────
    def record(self, symbol: str, tf: str, fv: dict[int, float],
               entry: float, settle: float) -> int:
        """One settled window: bucket every frozen reading, credit P(up)."""
        if not fv or not entry or not settle:
            return 0
        up = 1 if settle > entry else 0        # strict: tie is NOT up
        n = 0
        for fid, value in fv.items():
            if not isinstance(fid, (int, str)):
                continue
            fid_i = int(fid)
            if fid_i == 20:                    # DRHE is a confidence gauge
                continue
            key = (fid_i, symbol, tf, _bucket(float(value)))
            cell = self.rows.setdefault(key, {"up": 0, "n": 0})
            cell["n"] += 1
            cell["up"] += up
            n += 1
        if n:
            self.save()
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
