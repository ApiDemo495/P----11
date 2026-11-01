"""Settlement ledger — THE single source of accuracy truth (advice G-1).

Strict sign: win = sign(settle - entry) == side; tie -> loss. Per
(asset, timeframe): W/L/T, strict-sign rate with n, streaks, last 20.
Every accuracy claim in the app reads from this object; nothing else
writes a rate. No-trade windows are counted separately as "skipped" —
never as wins.
"""
from __future__ import annotations

import json
import os
import tempfile
import time

LEDGER_FILE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "ledger.json")
MIN_N_RATE = 20           # below this the UI must print "not enough windows"


class SettlementLedger:
    def __init__(self, path: str | None = None) -> None:
        self.path = path or LEDGER_FILE
        self.rows: dict[str, list[dict]] = {}   # "SYM|tf" -> settled rows
        self.load()

    # ---------------- persistence ----------------
    def load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.rows = {k: list(v) for k, v in (data.get("rows") or {}).items()}
        except (OSError, ValueError):
            self.rows = {}

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path) or ".",
                                       suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                json.dump({"rows": self.rows}, fh)
            os.replace(tmp, self.path)
        except OSError:
            pass

    # ---------------- settlement ----------------
    @staticmethod
    def _key(symbol: str, tf: str) -> str:
        return f"{symbol}|{tf}"

    def settle(self, symbol: str, tf: str, entry: float, settle_price: float,
               side: int, window_end: float | None = None) -> dict:
        """Grade one completed window. side==0 -> skipped (never a win)."""
        if side == 0 or not entry or not settle_price:
            result = "skip"
        else:
            move = settle_price - entry
            if move == 0.0:
                result = "loss"                # tie = loss (advice 0.4)
            else:
                result = "win" if (move > 0) == (side > 0) else "loss"
        row = {"ts": time.time(), "window_end": window_end or time.time(),
               "entry": round(entry, 6), "settle": round(settle_price, 6),
               "side": int(side), "result": result}
        key = self._key(symbol, tf)
        self.rows.setdefault(key, []).append(row)
        del self.rows[key][:-500]              # keep last 500 per pair
        self.save()
        return row

    # ---------------- reads ----------------
    def stats(self, symbol: str, tf: str) -> dict:
        rows = self.rows.get(self._key(symbol, tf), [])
        graded = [r for r in rows if r["result"] in ("win", "loss")]
        wins = sum(1 for r in graded if r["result"] == "win")
        losses = len(graded) - wins
        skipped = sum(1 for r in rows if r["result"] == "skip")
        streak, best, cur = 0, 0, 0
        for r in reversed(graded):
            if r["result"] == "win":
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        streak = cur
        n = len(graded)
        return {
            "n": n, "wins": wins, "losses": losses,
            "ties": 0,                     # ties are graded as losses
            "skipped": skipped,
            "rate": round(wins / n, 4) if n else None,
            "enough": n >= MIN_N_RATE,
            "streak": streak, "best_streak": best,
            "last20": [r["result"] for r in graded[-20:]],
        }

    def pooled(self, tf: str) -> dict:
        """Asset-agnostic pooled stats for one horizon (advice 0.5: pool
        until per-asset n exists)."""
        wins = losses = skipped = 0
        for key, rows in self.rows.items():
            if not key.endswith(f"|{tf}"):
                continue
            for r in rows:
                if r["result"] == "win":
                    wins += 1
                elif r["result"] == "loss":
                    losses += 1
                else:
                    skipped += 1
        n = wins + losses
        return {"n": n, "wins": wins, "losses": losses, "skipped": skipped,
                "rate": round(wins / n, 4) if n else None,
                "enough": n >= MIN_N_RATE}

    def api_payload(self, symbols: list[str], tfs: list[str]) -> dict:
        per = {f"{s}|{tf}": self.stats(s, tf) for s in symbols for tf in tfs}
        pooled = {tf: self.pooled(tf) for tf in tfs}
        return {"per": per, "pooled": pooled, "min_n": MIN_N_RATE,
                "rule": "strict sign · tie = loss · skips never count as wins"}
