"""Mode B — blind bar-by-bar replay (app + your discretion).

Picks a random window, hides asset + date, steps candle by candle
showing the app's live prediction at t. [T]ake / [S]kip / [A]djust
SL-TP, then fast-forwards to first touch and logs the outcome to
SQLite (and the trade carries source='discretionary' so the report
can compare your filter against the raw signal stream).
"""
from __future__ import annotations

import os
import random
import sqlite3

from .data import Bar
from .predictor import HORIZON, AppPredictor
from .runner import Trade, settle
from .slice import DataSlice

DB_PATH = os.path.join(os.path.dirname(__file__), "replay.db")


def _db(path: str = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE IF NOT EXISTS replay_trades(
        asset TEXT, ts INTEGER, direction TEXT, entry REAL, sl REAL,
        tp REAL, exit_ts INTEGER, exit_reason TEXT, r_gross REAL,
        r_net REAL, pnl REAL, tier TEXT, source TEXT, taken INTEGER)""")
    return con


def log_trade(con: sqlite3.Connection, tr: Trade, taken: bool) -> None:
    con.execute("INSERT INTO replay_trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (tr.asset, tr.ts, tr.direction, tr.entry, tr.sl, tr.tp,
                 tr.exit_ts, tr.exit_reason, tr.r_gross, tr.r_net,
                 tr.pnl, tr.tier, tr.source, 1 if taken else 0))
    con.commit()


def load_filtered(path: str = DB_PATH) -> list[Trade]:
    if not os.path.exists(path):
        return []
    con = _db(path)
    rows = con.execute("SELECT asset,ts,direction,entry,sl,tp,exit_ts,"
                       "exit_reason,r_gross,r_net,pnl,tier,source "
                       "FROM replay_trades WHERE taken=1").fetchall()
    return [Trade(*r) for r in rows]


def replay(series: dict[str, list[Bar]], predictor: AppPredictor,
           asset: str | None = None, start: int | None = None,
           warmup: int = 80, actions: list[str] | None = None,
           seed: int | None = None, db_path: str = DB_PATH) -> Trade | None:
    """One blind replay session. `actions` scripts the keys for
    non-interactive use ('t' take, 's' skip, 'a' adjust +10%/-10%)."""
    rng = random.Random(seed)
    asset = asset or rng.choice(list(series))
    bars = series[asset]
    t = start if start is not None else rng.randrange(warmup, len(bars) - HORIZON - 1)
    ds = DataSlice(bars)
    con = _db(db_path)
    scripted = iter(actions or [])
    print(f"── BLIND REPLAY ── asset ??? · date ??? · bars {warmup}..")
    while t + HORIZON < len(bars):
        view = ds.at(t)
        sig = predictor.predict(view, asset)
        last = view.last
        print(f"[{t}] close {last.c:.2f} (H {last.h:.2f} L {last.l:.2f})",
              "· app: " + ("FLAT" if sig is None else
              f"{'LONG' if sig.direction > 0 else 'SHORT'} "
              f"TP {sig.tp:.2f} SL {sig.sl:.2f} conf {sig.confidence:.3f}"))
        if sig is None:
            t += 1
            continue
        act = next(scripted, None)
        if act is None:
            act = input("[t]ake [s]kip [a]djust [q]uit > ").strip().lower()
        if act == "q":
            return None
        if act == "a":
            risk = abs(sig.entry - sig.sl)
            sig.sl = sig.entry - sig.direction * risk * 0.9
            sig.tp = sig.entry + sig.direction * risk * 1.8
            print(f"   adjusted: TP {sig.tp:.2f} SL {sig.sl:.2f} (rr 2.0 kept)")
            act = next(scripted, None) or (input("[t]ake [s]kip > ").strip().lower()
                                              if actions is None else "t")
        taken = act == "t"
        tr = settle(asset, sig, bars[t], ds.future(t, t + HORIZON + 1))
        tr.source = "discretionary"
        log_trade(con, tr, taken)
        print(f"   REVEAL {asset} @{tr.ts}: {'TAKEN' if taken else 'skipped'} → "
              f"{tr.exit_reason} net {tr.r_net:+.2f}R")
        return tr
    return None
