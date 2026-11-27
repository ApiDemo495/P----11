"""Section 13 — persistent, queryable GRC history.

One JSON file per day (history_YYYY-MM-DD.json) under HISTORY_DIR; files
older than HISTORY_RETENTION_DAYS are compressed into weekly summaries that
keep only daily accuracy stats + formula-weight snapshots.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone

from .config import HISTORY_DIR, HISTORY_RETENTION_DAYS


class HistoryDB:
    def __init__(self, root: str = HISTORY_DIR) -> None:
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    # ---------------------------------------------------------------- write
    def _path(self, day: str) -> str:
        return os.path.join(self.root, f"history_{day}.json")

    def append(self, record: dict) -> None:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self._path(day)
        rows: list[dict] = []
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    rows = json.load(fh)
            except (OSError, ValueError):
                rows = []
        rows.append(record)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as fh:
                json.dump(rows, fh)
            os.replace(tmp, path)
        except OSError:
            pass

    # ----------------------------------------------------------- retention
    def compress_old(self) -> int:
        """Weekly-summary compression for files past the retention window."""
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(days=HISTORY_RETENTION_DAYS)).strftime("%Y-%m-%d")
        compressed = 0
        for name in sorted(os.listdir(self.root)):
            if not (name.startswith("history_") and name.endswith(".json")):
                continue
            day = name[len("history_"):-len(".json")]
            if day >= cutoff or len(day) != 10:
                continue
            try:
                with open(os.path.join(self.root, name)) as fh:
                    rows = json.load(fh)
            except (OSError, ValueError):
                continue
            stats = _day_stats(day, rows)
            summary_path = os.path.join(self.root, "weekly_summaries.json")
            summaries: list[dict] = []
            if os.path.exists(summary_path):
                try:
                    with open(summary_path) as fh:
                        summaries = json.load(fh)
                except (OSError, ValueError):
                    summaries = []
            summaries.append(stats)
            try:
                with open(summary_path, "w") as fh:
                    json.dump(summaries, fh)
                os.remove(os.path.join(self.root, name))
                compressed += 1
            except OSError:
                pass
        return compressed

    # --------------------------------------------------------------- reads
    def days(self) -> list[dict]:
        out = []
        for name in sorted(os.listdir(self.root), reverse=True):
            if not (name.startswith("history_") and name.endswith(".json")):
                continue
            day = name[len("history_"):-len(".json")]
            if len(day) != 10:
                continue
            try:
                with open(os.path.join(self.root, name)) as fh:
                    rows = json.load(fh)
            except (OSError, ValueError):
                continue
            out.append(_day_stats(day, rows))
        # include compressed summaries at the end (older)
        sp = os.path.join(self.root, "weekly_summaries.json")
        if os.path.exists(sp):
            try:
                with open(sp) as fh:
                    out.extend(json.load(fh))
            except (OSError, ValueError):
                pass
        return out

    def day(self, day: str) -> list[dict]:
        try:
            with open(self._path(day)) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return []


def _day_stats(day: str, rows: list[dict]) -> dict:
    """Day P&L and hit counts, from RESOLVED windows only.

    Two things this used to get wrong: ``accuracy`` divided hits by
    (hits + sls), so an unresolved window simply left the denominator — the
    headline was computed over a hand-picked subset of finished trades; and
    ``pnl_pct`` was summed nowhere, so the realised number the user asked
    about was never published. Coverage is now reported beside accuracy.
    """
    hits = sls = preds = resolved = 0
    pnl = 0.0
    for r in rows:
        for tf_row in (r.get("outcomes") or {}).values():
            preds += 1
            _win = bool(tf_row.get("target_hit"))
            _loss = bool(tf_row.get("sl_hit"))
            if _win:
                hits += 1
            if _loss:
                sls += 1
            if _win or _loss or tf_row.get("resolved"):
                resolved += 1
                try:
                    pnl += float(tf_row.get("pnl_pct") or 0.0)
                except (TypeError, ValueError):
                    pass
    acc = hits / resolved if resolved else None
    return {"date": day, "grcs": len(rows), "predictions": preds,
            "resolved": resolved,
            "targets_hit": hits, "sls_hit": sls, "accuracy": acc,
            "coverage": (resolved / preds) if preds else None,
            "pnl_pct": round(pnl, 4)}


def empty_record(now: float | None = None) -> dict:
    ts = now if now is not None else time.time()
    return {
        "timestamp": datetime.fromtimestamp(ts, timezone.utc).isoformat(),
        "prices": {},
        "predictions": {},
        "outcomes": {},
        "formula_outputs": {},
        "brain_states": {},
        "emotion_composite": 0.0,
        "physics_composite": 0.0,
        "news_impact": 0.0,
        "agent_signals": {"gemini": {"direction": 0, "confidence": 0.0},
                          "local": {"direction": 0, "confidence": 0.0}},
        "weights": {},
    }
