"""Printable win-rate dial report — "raise my win rate without abstention".

Prints, for every measured cell, the highest net win rate a *lockable*
geometry reaches (reward:risk >= 0.25, target >= 1.5x round trip), the
win rate the same windows show at a 1:1 barrier, the break-even win rate
implied by the geometry, and the measured price per trade.

The break-even column is the point of the report: a geometry wins
``k x stop`` when it hits and loses one stop when it does not, so it
needs ``1 / (1 + k)`` wins to break even. Dialling the win rate up also
dials that requirement up — the gap column is what actually decides
whether a book makes money, and it is identical to net EV.

Usage:
    PYTHONPATH=. python3 backtest/winrate_dial_report.py
    PYTHONPATH=. python3 backtest/winrate_dial_report.py --cell BTC|tf4
"""
from __future__ import annotations

import argparse
import json
import os

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _load(name: str) -> dict:
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def rows_for(cell: dict, min_k: float) -> list[dict]:
    out = []
    for side, sd in (cell.get("sides") or {}).items():
        for r in sd.get("frontier") or []:
            if r["k"] < min_k or r.get("wr") is None:
                continue
            be = 1.0 / (1.0 + r["k"])
            out.append({**r, "side": side, "breakeven_wr": be,
                        "gap_pp": (r["wr"] - be) * 100.0})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="")
    ap.add_argument("--min-k", type=float, default=0.25,
                    help="lockable reward:risk floor (hold_policy MIN_REWARD_RISK)")
    a = ap.parse_args(argv)
    doc = _load("winrate_frontier.json")
    cells = doc.get("cells") or {}
    keys = [a.cell] if a.cell else sorted(cells)
    print("net win rate at lockable geometry — every window traded, no skips")
    print(f"{'cell':8s} {'side':9s} {'k':>5s} {'m':>5s} {'WR':>7s} "
          f"{'need':>7s} {'gap':>7s} {'EV bps':>8s} {'n':>6s}")
    print("-" * 74)
    summary = {}
    for key in keys:
        cell = cells.get(key)
        if not cell:
            print(f"{key}: not measured")
            continue
        rows = rows_for(cell, a.min_k)
        if not rows:
            continue
        best = max(rows, key=lambda r: r["wr"])
        bestgap = max(rows, key=lambda r: r["gap_pp"])
        at1 = [r for r in rows if r["k"] == 1.0 and r["m"] == 1.0]
        for label, r in (("best WR", best), ("best gap", bestgap)):
            print(f"{key:8s} {r['side']:9s} {r['k']:5.2f} {r['m']:5.2f} "
                  f"{r['wr']*100:6.1f}% {r['breakeven_wr']*100:6.1f}% "
                  f"{r['gap_pp']:+6.1f} {r['ev_bps']:+8.1f} {r['n']:6d}"
                  + ("   <- highest win rate" if label == "best WR"
                     else "   <- least bad economics"))
        summary[key] = {"best_wr": best["wr"], "best_wr_k": best["k"],
                        "best_wr_side": best["side"],
                        "best_wr_ev_bps": best["ev_bps"],
                        "best_gap_pp": bestgap["gap_pp"],
                        "best_gap_k": bestgap["k"],
                        "best_gap_side": bestgap["side"],
                        "at_k1_wr": at1[0]["wr"] if at1 else None,
                        "at_k1_ev_bps": at1[0]["ev_bps"] if at1 else None}
    print("\nDIAL EFFECT (same windows): k=1.0 -> tightest lockable k")
    print(f"{'cell':8s} {'WR@1:1':>8s} {'WR dialled':>11s} {'delta':>8s} "
          f"{'EV@1:1':>8s} {'EV dialled':>11s}")
    print("-" * 62)
    for key, s in summary.items():
        if s["at_k1_wr"] is None:
            continue
        print(f"{key:8s} {s['at_k1_wr']*100:7.1f}% {s['best_wr']*100:10.1f}% "
              f"{(s['best_wr']-s['at_k1_wr'])*100:+7.1f} "
              f"{s['at_k1_ev_bps']:+8.1f} {s['best_wr_ev_bps']:+11.1f}")
    out = os.path.join(RESULTS, "winrate_dial_report.json")
    with open(out, "w") as f:
        json.dump({"min_k": a.min_k, "cells": summary}, f, indent=1)
    print(f"\n-> {os.path.relpath(out, os.getcwd())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
