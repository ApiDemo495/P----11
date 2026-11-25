"""IS THE 66 % BOOK REAL, OR IS IT AN ARTEFACT OF THE SCORING RULES?

``geometry_scan`` finds 81-88 % net win rates at tf3-tf7d on a zero-fee venue.
Four things can manufacture a number like that and every one must be killed
before it is worth a dollar:

  1. **Drift.**  A book that wins on both sides is a statement about the tape.
     Test: run the mirror (short) book; if it earns too, the cell is drift.
  2. **In-sample fitting.**  Test: fit on the first 65 %, report only the
     holdout.
  3. **Ties.**  When both barriers are touched inside one 5-min bar the
     intra-bar order is unknown.  Test: score them as stop, as target, and as
     half — a real edge must survive ALL THREE.
  4. **Too few samples.**  Test: Wilson lower bound must clear break-even.

A cell is a SURVIVOR only when the holdout clears all four.  This is the
module that decides whether anything in this repo is tradeable.

Usage
    .venv/bin/python -m backtest.geometry_validate --assets BTC ETH
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data import load_cache                              # noqa: E402
from backtest.geometry_scan import RTS, barrier_book, blocks, dd_bps  # noqa: E402

RESULTS = os.path.join("backtest", "results")
OUT = os.path.join(RESULTS, "geometry_validate.json")
TRAIN_FRAC = 0.65
TIES = ("stop", "half", "tp")
#: every scanned cell that showed positive EV on a cheap venue
CANDIDATES = (("tf1", 4, 8.0, 32.0), ("tf2", 9, 8.0, 32.0),
              ("tf3", 26, 8.0, 32.0), ("tf4", 49, 8.0, 32.0),
              ("1d", 288, 8.0, 32.0), ("1d", 288, 20.0, 80.0),
              ("7d", 2016, 8.0, 32.0), ("7d", 2016, 20.0, 80.0),
              ("7d", 2016, 8.0, 15.5))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def short_book(C, H, L, tp_bps, sl_bps, rt_bps, tie="stop"):
    """The mirror image: short at the open, stop above, target below."""
    n, k = C.shape
    e = C[:, 0]
    tp = e * (1.0 - tp_bps * 1e-4)
    sl = e * (1.0 + sl_bps * 1e-4)
    dn = L <= tp[:, None]
    up = H >= sl[:, None]
    fd = np.where(dn.any(axis=1), dn.argmax(axis=1), k)
    fu = np.where(up.any(axis=1), up.argmax(axis=1), k)
    close_bps = (1.0 - C[:, -1] / e) * 1e4
    # see geometry_scan.barrier_book: a tie is BOTH barriers in one bar, never
    # "neither barrier was reached" (that must settle at the close)
    tie_mask = (fd == fu) & (fd < k)
    tie_val = {"stop": -sl_bps, "tp": float(tp_bps),
               "half": 0.5 * tp_bps - 0.5 * sl_bps}[tie]
    r = np.where(fd < fu, float(tp_bps),
                 np.where(fu < fd, -float(sl_bps),
                          np.where(tie_mask, tie_val, close_bps)))
    r = r - rt_bps
    return r, r > 0, tie_mask


def run(asset):
    bars = load_cache(asset, "5m")
    if not bars:
        return {"asset": asset, "status": "no_tape"}
    c = np.array([b.c for b in bars], float)
    h = np.array([b.h for b in bars], float)
    l = np.array([b.l for b in bars], float)
    rows = []
    for tf, k, tp, sl in CANDIDATES:
        C, H, L = blocks(c, h, l, k)
        n = len(C)
        cut = int(n * TRAIN_FRAC)
        be = sl / (tp + sl)
        for vname, rt in RTS:
            for side, fn in (("long", barrier_book), ("short", short_book)):
                # every tie rule, so robustness is visible per row
                per_tie = {}
                for rule in TIES:
                    r, won, ties = fn(C, H, L, tp, sl, rt, tie=rule)
                    k_ho = int(won[cut:].sum())
                    n_ho = n - cut
                    lo_ho, _ = wilson(k_ho, n_ho)
                    per_tie[rule] = {
                        "wr": round(float(won.mean()), 4),
                        "wr_holdout": round(k_ho / n_ho, 4) if n_ho else None,
                        "ci_lo_holdout": round(lo_ho, 4),
                        "ev_holdout_bps": round(float(r[cut:].mean()), 3),
                        "dd_bps": round(dd_bps(r), 1),
                    }
                # strictest rule (stop) is the headline
                head = per_tie["stop"]
                wr_ho = head["wr_holdout"]
                passes = all(
                    per_tie[x]["wr_holdout"] is not None
                    and per_tie[x]["wr_holdout"] > be
                    and per_tie[x]["ci_lo_holdout"] > be
                    and per_tie[x]["ev_holdout_bps"] > 0
                    for x in TIES)
                rows.append({
                    "asset": asset, "tf": tf, "minutes": k * 5,
                    "venue": vname, "rt_bps": rt, "side": side,
                    "tp_bps": tp, "sl_bps": sl,
                    "break_even_wr": round(be, 4),
                    "n": n, "n_holdout": n - cut,
                    "tie_rate": round(float(ties.mean()), 4),
                    "wr_all": head["wr"],
                    "wr_holdout": wr_ho,
                    "ci_lo_holdout": head["ci_lo_holdout"],
                    "ev_holdout_bps": head["ev_holdout_bps"],
                    "dd_bps": head["dd_bps"],
                    "by_tie": per_tie,
                    "survives_all_tie_rules": bool(passes),
                })
    return {"asset": asset, "status": "ok", "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="*", default=["BTC", "ETH"])
    args = ap.parse_args()
    res = [run(a) for a in args.assets]
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)

    for blk in res:
        if blk.get("status") != "ok":
            continue
        print(f"\n{'='*100}\n{blk['asset']}: holdout ({1-TRAIN_FRAC:.0%} of tape), "
              f"tie rule = stop\n{'='*100}")
        for vname, rt in RTS:
            for side in ("long", "short"):
                rs = [x for x in blk["rows"]
                      if x["venue"] == vname and x["side"] == side]
                print(f"\n  {vname} {rt:g} bps  {side}")
                print(f"    {'cell':>10s} {'tp/sl':>11s} {'n':>6s} "
                      f"{'tie%':>5s} {'WR all':>7s} {'WR hold':>8s} "
                      f"{'CI lo':>6s} {'BE':>6s} {'EV hold':>8s} "
                      f"{'DD':>7s}  tie-robust")
                for x in rs:
                    if not x["n_holdout"]:
                        continue
                    print(f"    {x['tf']+'/'+str(x['minutes']):>10s} "
                          f"{x['tp_bps']:5.1f}/{x['sl_bps']:5.1f} "
                          f"{x['n']:6d} {x['tie_rate']*100:4.1f}% "
                          f"{x['wr_all']*100:6.1f}% {x['wr_holdout']*100:7.1f}% "
                          f"{x['ci_lo_holdout']*100:5.1f}% "
                          f"{x['break_even_wr']*100:5.1f}% "
                          f"{x['ev_holdout_bps']:+7.2f} {x['dd_bps']:+7.0f}  "
                          f"{'YES' if x['survives_all_tie_rules'] else 'no'}")

    print(f"\n{'='*100}\nSURVIVORS — holdout beats break-even under ALL THREE tie "
          f"rules\n{'='*100}")
    n_s = 0
    for blk in res:
        for x in blk.get("rows", []):
            if x["survives_all_tie_rules"]:
                n_s += 1
                print(f"  {x['asset']} {x['venue']} {x['tf']}/{x['minutes']}m "
                      f"{x['side']} tp{x['tp_bps']:g}/sl{x['sl_bps']:g}: "
                      f"WR {x['wr_holdout']*100:.1f} % (CI lo "
                      f"{x['ci_lo_holdout']*100:.1f} %, BE "
                      f"{x['break_even_wr']*100:.1f} %) "
                      f"EV {x['ev_holdout_bps']:+.2f} bps n={x['n_holdout']}")
    if not n_s:
        print("  NONE — nothing here is tradeable yet.")
    print(f"\n  {n_s} survivor cells of {sum(len(b.get('rows', [])) for b in res)}")
    print(f"\nNOTE: 'tie-robust' means the win rate is not an artefact of how "
          f"same-bar double touches are scored.  Compare the long and short "
          f"blocks: if both win, it is drift and the book is not a strategy.")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
