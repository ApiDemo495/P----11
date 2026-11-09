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
        # tests must never write rows into the production ledger file
        # (a settled test row would poison real accuracy stats)
        if path is None and os.environ.get("PYTEST_CURRENT_TEST"):
            import tempfile
            path = os.path.join(tempfile.gettempdir(),
                                "predict4_test_ledger.json")
        self.path = path or LEDGER_FILE
        self.rows: dict[str, list[dict]] = {}   # "SYM|tf" -> settled rows
        # settlement visibility: is the ledger ALIVE? (user: 'history is
        # all zero' — the panel now says when the last settle happened)
        self.settle_calls = 0
        self.last_settle_ts = 0.0
        self.last_settle: dict = {}
        self.totals = {"win": 0, "loss": 0, "skip": 0}
        self.load()

    # ---------------- persistence ----------------
    def load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.rows = {k: list(v) for k, v in (data.get("rows") or {}).items()}
        except (OSError, ValueError):
            self.rows = {}
        # derive the liveness counters from PERSISTED rows too — otherwise
        # a fresh process prints "0 settles" while24 rows sit in the file
        last_ts, last_row, last_key = 0.0, {}, ""
        for k, v in self.rows.items():
            for r in v:
                res = r.get("result", "skip")
                self.totals[res] = self.totals.get(res, 0) + 1
                self.settle_calls += 1
                if float(r.get("ts") or 0) >= last_ts:
                    last_ts = float(r.get("ts") or 0)
                    last_row, last_key = r, k
        if last_row:
            self.last_settle_ts = last_ts
            sym, tf = (last_key.split("|") + [""])[:2]
            self.last_settle = {"symbol": sym, "tf": tf,
                                "result": last_row.get("result"),
                                "side": last_row.get("side"),
                                "entry": last_row.get("entry")}

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
               side: int, window_end: float | None = None,
               level_outcome: str | None = None,
               seeded: bool = False, conf: float | None = None) -> dict:
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
               "side": int(side), "result": result,
               "seeded": bool(seeded),
               "conf": (round(conf, 4) if conf is not None else None),
               # docs11-charts B-3: window outcome AND level outcome are
               # two separate columns (TP first / SL first / neither)
               "level_outcome": level_outcome}
        key = self._key(symbol, tf)
        self.rows.setdefault(key, []).append(row)
        del self.rows[key][:-500]              # keep last 500 per pair
        self.settle_calls += 1
        self.last_settle_ts = row["ts"]
        self.last_settle = {"symbol": symbol, "tf": tf,
                            "result": result, "side": int(side),
                            "entry": row["entry"]}
        self.totals[result] = self.totals.get(result, 0) + 1
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
        # realised TP-first % (docs11-charts B-3 hourly-review input):
        # compare against branch_stats' predicted P(target before stop)
        _lv = [r.get("level_outcome") for r in graded
               if r.get("level_outcome")]
        _tp_first = _lv.count("TP first")
        # expectancy: mean signed bps per graded window + payoff ratio
        # (avg win bps / avg loss bps) — the numbers that decide whether
        # a direction is worth trading, beyond the raw hit-rate
        _bps = [(int(r["side"]) * (float(r["settle"]) / float(r["entry"]) - 1)
                 * 1e4) for r in graded
                if r.get("entry") and r.get("settle") and r.get("side")]
        _wb = [x for x in _bps if x > 0]
        _lb = [-x for x in _bps if x < 0]
        return {
            "exp_bps": round(sum(_bps) / len(_bps), 2) if _bps else None,
            "payoff": (round((sum(_wb) / len(_wb)) / (sum(_lb) / len(_lb)), 3)
                       if _wb and _lb else None),
            "tp_first_n": len(_lv),
            "tp_first_pct": (round(_tp_first / len(_lv), 4)
                             if _lv else None),
            "n": n, "wins": wins, "losses": losses,
            "ties": 0,                     # ties are graded as losses
            "seeded": sum(1 for r in rows if r.get("seeded")),
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
        import time as _t
        return {"per": per, "pooled": pooled, "min_n": MIN_N_RATE,
                "rule": "strict sign · tie = loss · skips never count as wins",
                # liveness: proves settlement is running (or how long it
                # has been quiet) instead of a wall of honest zeros
                "settle_calls": self.settle_calls,
                "last_settle_ago_s": (round(_t.time() - self.last_settle_ts, 1)
                                      if self.last_settle_ts else None),
                "last_settle": self.last_settle,
                "totals": dict(self.totals),
                "rows_total": sum(len(v) for v in self.rows.values())}
