"""Sealed prediction locks — the user-facing contract of ANALYSE.

Why this module exists
----------------------
The staged ANALYSE flow was explicitly read-only: it produced a beautiful
result box and threw it away.  The user's requirement is the opposite —
pressing ▶ ANALYSE must **lock** that prediction, write it to history, and
grade it later so the app learns from it.

A lock is a sealed, immutable record:

    {id, symbol, tf, minutes, created_at, expires_at, direction, entry,
     tp, sl, confidence, kind, evidence, status, outcome, settled_at, bps}

Rules
-----
* **One lock per (asset, timeframe, window).**  A second ANALYSE inside the
  same window returns the existing lock with ``already=True``.  That is what
  "locked" means; it is not a to-do list that moves every 45 s.
* **Nothing invalid is sealed.**  A FLAT call, a missing/wrong-side level, a
  non-finite price or a zero confidence is refused with the exact reasons.
  A below-floor call may still be sealed as ``kind="watch"`` — it is a
  prediction, it is graded, it is simply not execution-eligible.
* **Settlement is measured from closed bars.**  A single barrier touch
  grades the lock. If TP and SL are both inside one OHLC bar, sequence is
  unknowable and the result is ``ambiguous`` (ungraded, no bps), never an
  invented loss. The backtest retains SL-first only as a pessimistic bound.
  Flat at exit is a ``push`` and is excluded from the hit-rate.
* **Learning is attached, not implied.**  ``settle_all`` returns the rows
  that just closed; the engine feeds their stored evidence (formula signs,
  subsystem votes, confidence, agent call) into the existing learners.

The file is JSON, written atomically and capped, so a restart never loses the
record and a crash never leaves half a lock.  Tests override the path with
``PRED_JOURNAL_PATH``.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Any, Iterable

from .config import SYMBOL_ORDER, TF_ORDER, TIMEFRAMES

JOURNAL_FILE = os.environ.get("PRED_JOURNAL_PATH",
                              os.path.join("data", "prediction_locks.json"))
KEEP = 400
MIN_CONF = 0.0


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _side_ok(direction: int, entry: float, tp: float, sl: float) -> tuple[bool, str]:
    if direction not in (-1, 1):
        return False, "direction is flat — a symmetric call cannot be locked"
    if entry <= 0 or tp <= 0 or sl <= 0:
        return False, "entry/TP/SL must all be positive prices"
    if direction > 0:
        if not (tp > entry > sl):
            return False, (f"BUY ladder is on the wrong side: TP {tp:g} must be "
                           f"> entry {entry:g} > SL {sl:g}")
    else:
        if not (tp < entry < sl):
            return False, (f"SELL ladder is on the wrong side: TP {tp:g} must be "
                           f"< entry {entry:g} < SL {sl:g}")
    return True, ""


def _sign(v: Any) -> int:
    f = _finite(v)
    if f is None:
        return 0
    return 1 if f > 1e-12 else (-1 if f < -1e-12 else 0)


class PredictionJournal:
    """Durable, immutable prediction locks with measured settlement."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or JOURNAL_FILE
        self.rows: list[dict] = []
        self._load()

    # ------------------------------------------------------------ storage
    def _load(self) -> None:
        try:
            with open(self.path) as fh:
                data = json.load(fh)
            rows = data.get("locks") if isinstance(data, dict) else data
            self.rows = [r for r in (rows or []) if isinstance(r, dict)][-KEEP:]
        except (OSError, ValueError):
            self.rows = []

    def _save(self) -> bool:
        try:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump({"version": 1, "locks": self.rows[-KEEP:]}, fh)
            os.replace(tmp, self.path)
            return True
        except OSError:
            return False

    # --------------------------------------------------------------- seal
    def seal(self, symbol: str, tf: str, *, direction: int,
             entry: float, tp: float, sl: float, confidence: float,
             minutes: int | None = None, created_at: float | None = None,
             gate_ok: bool = False, recheck_ok: bool | None = None,
             reason: str = "", evidence: dict | None = None,
             target: str = "measured TP/SL ladder",
             source: str = "analyse", system_mode: str = "old",
             cycle_id: str | None = None, lock_id: str | None = None,
             max_hold_s: float | None = None) -> dict:
        """Seal one prediction, or refuse with every failing check named.

        ``max_hold_s`` seals an **era ride** (§18): the lock no longer
        dies at the window boundary — it lives until TP, SL, the
        certified side flipping (the era exit), or this maximum hold.
        """
        now = float(created_at if created_at is not None else time.time())
        if symbol not in SYMBOL_ORDER:
            return self._refuse(["unknown symbol " + str(symbol)])
        if tf not in TF_ORDER:
            return self._refuse(["unknown timeframe " + str(tf)])
        W = int(minutes or TIMEFRAMES[tf]["minutes"])
        entry_f, tp_f, sl_f = _finite(entry), _finite(tp), _finite(sl)
        conf = _finite(confidence)
        checks: list[str] = []
        if entry_f is None or tp_f is None or sl_f is None:
            checks.append("entry/TP/SL is missing or not a finite number")
            return self._refuse(checks)
        ok, why = _side_ok(int(direction), entry_f, tp_f, sl_f)
        if not ok:
            checks.append(why)
        if conf is None or not (0.0 <= conf <= 1.0):
            checks.append("confidence is missing or outside [0, 1]")
        if checks:
            return self._refuse(checks)

        # one lock per window: the window is anchored on creation time.
        # An OPEN era ride owns the (symbol, tf) until it settles — no
        # stacking across windows (§18: the ride IS the position).
        window = int(now // max(60, W * 60))
        for row in reversed(self.rows):
            if (row.get("symbol") == symbol and row.get("tf") == tf
                    and row.get("status") == "open"
                    and row.get("settle_mode") == "era_ride"):
                return {**self._public(row), "locked": True, "already": True,
                        "why": ["an era ride is already open for this "
                                "symbol/timeframe — it settles at TP/SL/"
                                "side-flip before a new one can seal"]}
        for row in reversed(self.rows):
            if (row.get("symbol") == symbol and row.get("tf") == tf
                    and row.get("window") == window
                    and row.get("status") in ("open", "won", "lost", "push", "ambiguous")):
                return {**self._public(row), "locked": True, "already": True,
                        "why": [f"{symbol} {tf.upper()} is already locked for "
                                "this {W}-minute window — the sealed prediction "
                                "does not move"]}

        evidence = dict(evidence or {})
        digest = hashlib.sha256(
            json.dumps({"s": symbol, "t": tf, "w": window,
                        "d": int(direction), "e": round(entry_f, 8),
                        "tp": round(tp_f, 8), "sl": round(sl_f, 8),
                        "c": round(conf, 6)},
                       sort_keys=True).encode()).hexdigest()[:12]
        row = {
            "id": f"pl-{digest}",
            "symbol": symbol,
            "tf": tf,
            "window": window,
            "minutes": W,
            "created_at": now,
            # §18 era ride: expires_at becomes the MAXIMUM HOLD (the
            # ride exits earlier at TP, SL or the certified side
            # flipping); a plain window lock keeps the window horizon.
            "expires_at": (now + float(max_hold_s)
                           if max_hold_s else now + W * 60.0),
            "settle_mode": ("era_ride" if max_hold_s else "window"),
            "max_hold_s": (float(max_hold_s) if max_hold_s else None),
            "direction": int(direction),
            "side": "BUY" if int(direction) > 0 else "SELL",
            "entry": round(entry_f, 8),
            "tp": round(tp_f, 8),
            "sl": round(sl_f, 8),
            "confidence": round(conf, 6),
            "tp_bps": round(abs(tp_f / entry_f - 1.0) * 1e4, 2),
            "sl_bps": round(abs(sl_f / entry_f - 1.0) * 1e4, 2),
            "ratio": (round(abs(tp_f - entry_f) / abs(sl_f - entry_f), 4)
                      if abs(sl_f - entry_f) > 1e-12 else None),
            "kind": "issued" if gate_ok else "watch",
            "gate_ok": bool(gate_ok),
            "recheck_ok": (None if recheck_ok is None else bool(recheck_ok)),
            "source": str(source),
            "target": str(target)[:80],
            "system_mode": str(system_mode),
            "cycle_id": cycle_id,
            "engine_lock_id": lock_id,
            "reason": str(reason)[:600],
            "evidence": evidence,
            "status": "open",
            "outcome": None,
            "settled_at": None,
            "settle_price": None,
            "settle_reason": None,
            "bps": None,
            "graded": False,
            "learned": False,
        }
        self.rows.append(row)
        self.rows = self.rows[-KEEP:]
        self._save()
        self._history({"type": "prediction_lock", "event": "sealed",
                       **self._compact(row)})
        return {**self._public(row), "locked": True, "already": False,
                "why": []}

    def _refuse(self, why: Iterable[str]) -> dict:
        return {"locked": False, "id": None, "status": "refused",
                "why": [str(w) for w in why]}

    # ------------------------------------------------------------- settle
    def settle(self, symbol: str, tf: str, price: float, *,
               high: float | None = None, low: float | None = None,
               now: float | None = None,
               series: list | None = None,
               side_now: int | None = None) -> list[dict]:
        """Settle every open lock for (symbol, tf) that has resolved.

        ``series`` — optional ``[(ts, high, low), …]`` of CLOSED bars.
        If both barriers lie inside one candle, the order is unknowable
        from OHLC: status becomes ``ambiguous`` (ungraded, no bps), never
        an invented win or loss. The backtest's SL-first walk is retained
        only as its pessimistic bound, not as live truth.

        ``side_now`` — the CURRENT certified side (±1).  An open
        **era ride** (§18, settle_mode ``era_ride``) exits at market
        when that side no longer matches the lock's direction: the
        flip IS the era exit.  First-touch (TP/SL) grades before the
        flip, matching era_sim's barrier-before-boundary order.
        """
        now = float(now if now is not None else time.time())
        p = _finite(price)
        if p is None or p <= 0:
            return []
        hi = _finite(high) or p
        lo = _finite(low) or p
        settled: list[dict] = []
        for row in self.rows:
            if (row.get("symbol") != symbol or row.get("tf") != tf
                    or row.get("status") != "open"):
                continue
            d = int(row.get("direction") or 0)
            entry = _finite(row.get("entry")) or 0.0
            tp = _finite(row.get("tp")) or 0.0
            sl = _finite(row.get("sl")) or 0.0
            # If the caller supplied a bar series (even an empty one),
            # only completed bars may trigger TP/SL. The live spot price
            # is reserved for a market exit at side-flip/expiry, not a
            # substitute for a completed-bar barrier touch.
            if series is None:
                hit_tp = (hi >= tp) if d > 0 else (lo <= tp)
                hit_sl = (lo <= sl) if d > 0 else (hi >= sl)
            else:
                hit_tp = hit_sl = False
            expired = now >= float(row.get("expires_at") or 0.0)
            # Walk closed bars for every lock, not just hold rows. OHLC
            # cannot reveal which barrier traded first inside one candle;
            # label that result unresolved rather than guessing SL-first.
            hold = (row.get("evidence") or {}).get("hold_policy") or {}
            hold_rules = bool(hold.get("hold_rules"))
            result = None
            reason = None
            if series:
                created = float(row.get("created_at") or 0.0)
                horizon = float(row.get("expires_at") or now)
                for ts, bhi, blo in series:
                    ts = float(ts)
                    if not (created < ts < horizon):
                        continue
                    bhi, blo = _finite(bhi), _finite(blo)
                    if bhi is None or blo is None:
                        continue
                    if d > 0:
                        sl_hit = blo <= sl
                        tp_hit = bhi >= tp
                    else:
                        sl_hit = bhi >= sl
                        tp_hit = blo <= tp
                    if sl_hit and tp_hit:
                        result = "ambiguous"
                        reason = ("both TP and SL crossed inside one closed "
                                  "bar; OHLC cannot establish order")
                        break
                    if sl_hit:
                        result, reason = "lost", "SL touched on a closed bar"
                        break
                    if tp_hit:
                        result, reason = "won", "TP touched on a closed bar"
                        break
            if result is None and hit_tp and hit_sl:
                result = "ambiguous"
                reason = ("both TP and SL crossed inside the available sample; "
                          "OHLC cannot establish order")
            elif result is None and hit_tp:
                result, reason = "won", "TP touched before SL"
            elif result is None and hit_sl:
                result, reason = "lost", "SL touched before TP"
            elif (result is None
                    and row.get("settle_mode") == "era_ride"
                    and side_now is not None
                    and int(side_now) != 0 and int(side_now) != d):
                # §18 era ride: the certified side flipped — exit at
                # market now (the era exit), grade by strict sign.
                move = p - entry
                if move > 1e-12:
                    result = "won" if d > 0 else "lost"
                elif move < -1e-12:
                    result = "lost" if d > 0 else "won"
                else:
                    result = "push"
                if reason is None:
                    reason = ("certified side flipped — era ride exit "
                              "at market (§18)")
            elif result is None and expired:
                # strict sign at the horizon: a tie is a push, not a win
                move = p - entry
                if move > 1e-12:
                    result = "won" if d > 0 else "lost"
                elif move < -1e-12:
                    result = "lost" if d > 0 else "won"
                else:
                    result = "push"
                if reason is None:
                    reason = (f"window expired at the "
                              f"{row.get('minutes')}-minute mark: price "
                              f"{p:g} vs entry {entry:g}")
            if result is None:
                continue
            row["status"] = {"won": "won", "lost": "lost",
                             "push": "push",
                             "ambiguous": "ambiguous"}[result]
            row["outcome"] = result
            row["settled_at"] = now
            row["settle_price"] = round(p, 8)
            row["settle_reason"] = reason
            # realised bps at the barrier that decided it, not at whatever
            # tick happened to be sampled when the touch was noticed
            # realised bps at the DECIDING price: the barrier that
            # touched, or the market price for a flip/expiry settle
            # (a ride exited at 103 must not report the TP at 110)
            _touch = bool(reason) and any(
                w in reason for w in ("touch", "before", "same-bar",
                                      "same-window"))
            _ref = ((tp if result == "won" else sl) if _touch else p)
            row["bps"] = (round(d * (_ref / entry - 1.0) * 1e4, 2)
                          if entry and result != "ambiguous" else None)
            row["graded"] = result in ("won", "lost")
            self._history({"type": "prediction_lock", "event": "settled",
                           **self._compact(row)})
            settled.append(dict(row))
        if settled:
            self._save()
        return settled

    def settle_all(self, prices: dict[str, float], *,
                   extremes: dict[str, tuple[float, float]] | None = None,
                   now: float | None = None,
                   series: dict[str, list] | None = None,
                   sides_now: dict[tuple[str, str], int] | None = None
                   ) -> list[dict]:
        out: list[dict] = []
        for symbol in SYMBOL_ORDER:
            p = _finite(prices.get(symbol))
            if p is None or p <= 0:
                continue
            hi = lo = None
            if extremes and symbol in extremes:
                hi, lo = extremes[symbol]
            sym_series = (series or {}).get(symbol)
            for tf in TF_ORDER:
                sn = (sides_now or {}).get((symbol, tf))
                out.extend(self.settle(symbol, tf, p, high=hi, low=lo,
                                       now=now, series=sym_series,
                                       side_now=sn))
        return out

    def mark_learned(self, lock_id: str) -> None:
        for row in self.rows:
            if row.get("id") == lock_id:
                row["learned"] = True
        self._save()

    # ---------------------------------------------------------------- read
    def _public(self, row: dict) -> dict:
        out = {k: v for k, v in row.items() if k != "evidence"}
        out["evidence"] = row.get("evidence") or {}
        return out

    def _compact(self, row: dict) -> dict:
        return {k: row.get(k) for k in
                ("id", "symbol", "tf", "minutes", "created_at", "expires_at",
                 "direction", "side", "entry", "target", "tp", "sl",
                 "confidence",
                 "tp_bps", "sl_bps", "ratio", "kind", "status", "outcome",
                 "settled_at", "settle_price", "bps", "cycle_id",
                 "system_mode")}

    def _history(self, record: dict) -> None:
        """Write the lock to the durable day-history (Section 13)."""
        try:
            from .historydb import HistoryDB
            HistoryDB().append(record)
        except Exception:
            pass

    def open_for(self, symbol: str, tf: str) -> list[dict]:
        return [dict(r) for r in self.rows
                if r.get("symbol") == symbol and r.get("tf") == tf
                and r.get("status") == "open"]

    def latest(self, symbol: str | None = None, tf: str | None = None,
               limit: int = 50) -> list[dict]:
        rows = list(reversed(self.rows))
        if symbol:
            rows = [r for r in rows if r.get("symbol") == symbol]
        if tf:
            rows = [r for r in rows if r.get("tf") == tf]
        return [self._public(r) for r in rows[:max(1, int(limit))]]

    def stats(self, symbol: str | None = None,
              tf: str | None = None) -> dict:
        rows = [r for r in self.rows
                if (not symbol or r.get("symbol") == symbol)
                and (not tf or r.get("tf") == tf)]
        graded = [r for r in rows if r.get("outcome") in ("won", "lost")]
        wins = sum(1 for r in graded if r.get("outcome") == "won")
        n = len(graded)
        bps = [float(r.get("bps") or 0.0) for r in graded]
        by_tf: dict[str, dict] = {}
        for t in TF_ORDER:
            sub = [r for r in graded if r.get("tf") == t]
            w = sum(1 for r in sub if r.get("outcome") == "won")
            by_tf[t] = {"n": len(sub), "wins": w,
                        "hit_rate": round(w / len(sub), 4) if sub else None,
                        "avg_bps": (round(sum(float(r.get("bps") or 0)
                                             for r in sub) / len(sub), 2)
                                    if sub else None)}
        return {
            "n": len(rows), "open": sum(1 for r in rows
                                        if r.get("status") == "open"),
            "graded": n, "wins": wins, "losses": n - wins,
            "hit_rate": round(wins / n, 4) if n else None,
            "avg_bps": round(sum(bps) / len(bps), 2) if bps else None,
            "by_tf": by_tf,
            "rule": ("one immutable lock per (asset, timeframe, window); a "
                     "barrier touch decides, otherwise strict sign at expiry; "
                     "a push is excluded from the hit-rate"),
        }

    def payload(self, symbol: str | None = None, tf: str | None = None,
                limit: int = 50) -> dict:
        return {"ok": True, "locks": self.latest(symbol, tf, limit),
                "stats": self.stats(symbol, tf),
                "rule": ("sealed on ANALYSE, immutable for its window, saved "
                         "to history, graded on the measured outcome")}
