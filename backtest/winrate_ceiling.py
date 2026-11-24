"""THE WIN-RATE CEILING — why 66 % is impossible at short horizons.

The constraint nobody stated
----------------------------
Every previous harness searched for a better *signal*.  None of them
asked the prior question: **is a 66 % net win rate even arithmetically
reachable at this horizon?**

For a book that must close its position inside the window (which is what
"one trade ends when the next starts" means), a window can only be a WIN
if the absolute move it captures exceeds the round trip.  Therefore

    WR_net  <=  P(|move over the window| > RT)        ... an upper bound

and the skill actually required to hit a target WR is

    required_accuracy = target_WR / P(|move| > RT)

which must be <= 1.0 for the target to be reachable at all.

This module measures both curves on real tape for every horizon.  The
result is not a strategy — it is the feasibility map that says which
cells can be attacked and which are hopeless by a factor of five.

Usage
    .venv/bin/python -m backtest.winrate_ceiling --assets BTC ETH
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data import FRICTION_BPS, load_cache          # noqa: E402

RESULTS = os.path.join("backtest", "results")
OUT = os.path.join(RESULTS, "winrate_ceiling.json")
TARGET = 0.66
#: the app's own timeframes, plus the long horizons where the math changes
HORIZONS = (("tf1", 22), ("tf2", 44), ("tf3", 130), ("tf4", 247),
            ("1d", 1440), ("2d", 2880), ("7d", 10080), ("14d", 20160),
            ("30d", 43200))


def windows(t, step, w_min):
    W = int(w_min * 60)
    starts = np.arange(int(np.ceil(t[0] / W) * W),
                       int(t[-1]) + step - W + 1, W, dtype=np.int64)
    i_in = np.searchsorted(t, starts, side="left")
    i_out = np.searchsorted(t, starts + W - step, side="right") - 1
    ok = (i_in >= 0) & (i_out > i_in) & (i_out < len(t))
    return i_in[ok], i_out[ok]


def run(asset):
    bars = load_cache(asset, "5m")
    if not bars:
        return {"asset": asset, "status": "no_tape"}
    t = np.array([b.t for b in bars])
    c = np.array([b.c for b in bars], float)
    h = np.array([b.h for b in bars], float)
    lo = np.array([b.l for b in bars], float)
    step = int(t[1] - t[0])
    rt = 2.0 * FRICTION_BPS[asset]
    rows = []
    for name, w_min in HORIZONS:
        i_in, i_out = windows(t, step, w_min)
        n = len(i_in)
        if n < 20:
            rows.append({"tf": name, "minutes": w_min, "n": int(n),
                         "status": "too_few"})
            continue
        mv = np.abs(c[i_out] / c[i_in] - 1.0) * 1e4
        p_cover = float((mv > rt).mean())
        # the ceiling, and how much skill a 66 % target needs
        ceil = p_cover
        req_acc = (TARGET / ceil) if ceil > 0 else float("inf")
        rows.append({
            "tf": name, "minutes": w_min, "n": int(n),
            "median_abs_move_bps": round(float(np.median(mv)), 2),
            "p_move_exceeds_rt": round(p_cover, 4),
            "max_possible_wr": round(ceil, 4),
            "target_wr": TARGET,
            "reachable": bool(ceil >= TARGET),
            "accuracy_required": (round(req_acc, 4)
                                  if np.isfinite(req_acc) else None),
            "verdict": ("reachable" if ceil >= TARGET
                        else f"IMPOSSIBLE — ceiling is {ceil*100:.1f} %, "
                             f"{TARGET/ceil:.1f}x short" if ceil > 0
                        else "no window can pay"),
            "hedge_ratio": round(ceil / TARGET, 3),
        })
    return {"asset": asset, "status": "ok", "rt_bps": round(rt, 2),
            "bars": len(c), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="*", default=["BTC", "ETH"])
    args = ap.parse_args()
    out = []
    for asset in args.assets:
        r = run(asset)
        out.append(r)
        if r.get("status") != "ok":
            print(f"{asset}: {r['status']}")
            continue
        print(f"\n=== {asset}  round trip {r['rt_bps']} bps  "
              f"{r['bars']:,} bars")
        print(f"  {'tf':>4s} {'mins':>6s} {'n':>7s} {'median move':>12s} "
              f"{'ceiling':>9s} {'acc needed':>11s}  verdict")
        for row in r["rows"]:
            if row.get("status") == "too_few":
                print(f"  {row['tf']:>4s} {row['minutes']:6d} {row['n']:7d}"
                      f"   too few windows")
                continue
            acc = row["accuracy_required"]
            acc_s = f"{acc*100:.1f}%" if acc is not None else "n/a"
            print(f"  {row['tf']:>4s} {row['minutes']:6d} {row['n']:7d} "
                  f"{row['median_abs_move_bps']:11.1f}b "
                  f"{row['max_possible_wr']*100:8.1f}% {acc_s:>11s}  "
                  f"{row['verdict']}")
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)

    print("\n── which horizons can even host a 66 % book? ──")
    for r in out:
        if r.get("status") != "ok":
            continue
        ok = [x for x in r["rows"] if x.get("reachable")]
        names = ", ".join(x["tf"] for x in ok) or "NONE"
        print(f"{r['asset']}: reachable at {names}")
        if ok:
            best = min(ok, key=lambda x: x["accuracy_required"])
            print(f"{'':6s} easiest cell {best['tf']}: needs "
                  f"{best['accuracy_required']*100:.1f}% directional "
                  f"accuracy to reach 66 % (ceiling "
                  f"{best['max_possible_wr']*100:.1f}%)")
        short = [x for x in r["rows"]
                 if x.get("max_possible_wr", 0) < TARGET and x.get("n", 0) >= 20]
        if short:
            worst = min(short, key=lambda x: x["max_possible_wr"])
            print(f"{'':6s} worst cell {worst['tf']}: ceiling "
                  f"{worst['max_possible_wr']*100:.1f}%, i.e. "
                  f"{worst['hedge_ratio']:.1f}x short of the target")
    print(f"\nNOTE: measured directional skill on this tape is ~52 %. "
          f"A cell is only attackable when 'accuracy_required' <= skill.")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
