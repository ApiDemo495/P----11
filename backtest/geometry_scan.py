"""THE FEE THAT BROKE THE 66 % GEOMETRY — barriers re-run at real venue fees.

Why this exists
---------------
There is an old, correct piece of arithmetic in the repo and it was never
joined to the venue question:

    a book has positive EV  iff  WR > sl / (tp + sl)
    so  WR = 66 % is break-even at  tp / sl = 0.515

i.e. a 66 % win rate *is* a legitimate geometry — "risk 2 to make 1" —
and it needs no skill at all to sit at break-even.  What it does need is a
take-profit large enough to clear the round trip.  At the legacy 28.9 bps
toll the profit target cannot be small, so the stop must be very large,
the fill/first-touch path gets wild, and the measured result was 7 %
(§25/§26).  On a 0 bps venue the toll disappears from that inequality
entirely.

So this module runs the same first-touch barrier book at four round trips
and reports, for each barrier geometry, the honest quadrifecta:

    win rate / EV per window / worst single window / max drawdown

A high win rate is only ever printed next to those last three numbers.

Usage
    .venv/bin/python -m backtest.geometry_scan --assets BTC ETH
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data import load_cache                        # noqa: E402

RESULTS = os.path.join("backtest", "results")
OUT = os.path.join(RESULTS, "geometry_scan.json")

#: blocks of exactly this many 5-min bars, so windows tile the tape cleanly
HORIZONS = (("tf1", 4), ("tf2", 9), ("tf3", 26), ("tf4", 49),
            ("1d", 288), ("7d", 2016))
#: take-profit in bps and the tp/sl payoff ratio
TPS = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0, 80.0)
RATIOS = (0.25, 0.4, 0.515, 0.66, 1.0, 1.5)
#: venue round trips, bps
RTS = (("lighter", 0.0), ("mexc_fut", 4.0), ("binance", 10.0),
       ("legacy", 28.872))
TARGET = 0.66


def blocks(c, h, l, k):
    """Tile the tape into non-overlapping k-bar windows (no look-ahead)."""
    n = len(c) // k
    sl = slice(0, n * k)
    C = c[sl].reshape(n, k)
    H = h[sl].reshape(n, k)
    L = l[sl].reshape(n, k)
    return C, H, L


def barrier_book(C, H, L, tp_bps, sl_bps, rt_bps, tie="stop"):
    """Long at each window open; first touch of tp/sl; else close at the end.

    A window where BOTH barriers are touched inside the SAME 5-minute bar has
    an unknown intra-bar order.  ``tie`` decides how those are scored:

        "stop"  -> the stop (pessimistic; the default)
        "tp"    -> the target (optimistic)
        "half"  -> 0.5*tp - 0.5*sl (the honest expectation for a coin flip)

    An earlier version of this function fell through to the window's CLOSE
    return on ties, which looked innocent and was not: on 1d/7d windows 85-96 %
    of them tie, so the close return (i.e. the drift of the tape) decided the
    result and both the long and the short book "won" 86 % of the time.  Ties
    are now always resolved explicitly and are counted.

    Returns (pnl_bps, won_mask, tie_mask).
    """
    n, k = C.shape
    e = C[:, 0]
    tp = e * (1.0 + tp_bps * 1e-4)
    sl = e * (1.0 - sl_bps * 1e-4)
    up = H >= tp[:, None]
    dn = L <= sl[:, None]
    fu = np.where(up.any(axis=1), up.argmax(axis=1), k)
    fd = np.where(dn.any(axis=1), dn.argmax(axis=1), k)
    close_bps = (C[:, -1] / e - 1.0) * 1e4
    # A "tie" means BOTH barriers were touched in the SAME bar (unknown
    # intra-bar order).  When NEITHER is touched the window simply ran out —
    # it must settle at the close, NOT be scored as a stop.  That distinction
    # was missing: fu == fd is true both when both barriers fall in one bar
    # AND when neither is ever touched (both sentinels == k), so a flat
    # window that closed +12 bps was booked as a full -80 bps loss.  It also
    # made "tie_rate" meaningless on 1d/7d windows.
    tie_mask = (fu == fd) & (fu < k)
    if tie == "stop":
        tie_val = np.full(n, -sl_bps)
    elif tie == "tp":
        tie_val = np.full(n, float(tp_bps))
    elif tie == "half":
        tie_val = np.full(n, 0.5 * tp_bps - 0.5 * sl_bps)
    else:
        raise ValueError(tie)
    r = np.where(fu < fd, float(tp_bps),
                 np.where(fd < fu, -float(sl_bps),
                          np.where(tie_mask, tie_val, close_bps)))
    r = r - rt_bps
    return r, r > 0, tie_mask


def dd_bps(r):
    """Max peak-to-trough of the cumulative P&L path, in bps."""
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    return float((eq - peak).min()) if len(eq) else 0.0


def run(asset):
    bars = load_cache(asset, "5m")
    if not bars:
        return {"asset": asset, "status": "no_tape"}
    c = np.array([b.c for b in bars], float)
    h = np.array([b.h for b in bars], float)
    l = np.array([b.l for b in bars], float)

    out = {"asset": asset, "status": "ok", "target": TARGET, "rts": [],
           "cells": []}
    for vname, rt in RTS:
        out["rts"].append({"venue": vname, "rt_bps": rt})

    for tf, k in HORIZONS:
        C, H, L = blocks(c, h, l, k)
        n = len(C)
        if n < 60:
            continue
        for vname, rt in RTS:
            for tp in TPS:
                for ratio in RATIOS:
                    sl = tp / ratio
                    r, won, ties = barrier_book(C, H, L, tp, sl, rt)
                    wr = float(won.mean())
                    ev = float(r.mean())
                    out["cells"].append({
                        "asset": asset, "tf": tf, "bars": k,
                        "minutes": k * 5, "n": n, "venue": vname,
                        "rt_bps": rt, "tp_bps": tp,
                        "sl_bps": round(sl, 1), "ratio": ratio,
                        "break_even_wr": round(sl / (tp + sl), 4),
                        "win_rate": round(wr, 4),
                        "ev_bps": round(ev, 3),
                        "worst_bps": round(float(r.min()), 1),
                        "max_dd_bps": round(dd_bps(r), 1),
                        "tie_rate": round(float(ties.mean()), 4),
                        "edge_vs_break_even": round(wr - sl / (tp + sl), 4),
                        "total_bps": round(float(r.sum()), 0),
                    })
    return out


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
        print(f"\n{'='*84}\n{blk['asset']}: first-touch barrier book, "
              f"non-overlapping windows\n{'='*84}")
        print(f"  {'venue':9s} {'RT':>6s} {'tf':>4s} {'tp':>5s} {'sl':>6s} "
              f"{'WR':>7s} {'BE WR':>6s} {'EV':>8s} {'worst':>8s} "
              f"{'maxDD':>8s} {'n':>6s}")
        # per venue, the best WR cell that is also EV-positive
        for vname, rt in RTS:
            cells = [x for x in blk["cells"]
                     if x["venue"] == vname and x["ev_bps"] > 0]
            if not cells:
                print(f"  {vname:9s} {rt:5.1f}b   — no EV-positive "
                      f"geometry at any horizon —")
                continue
            cells.sort(key=lambda x: -x["win_rate"])
            for x in cells[:4]:
                print(f"  {vname:9s} {rt:5.1f}b {x['tf']:>4s} "
                      f"{x['tp_bps']:5.1f} {x['sl_bps']:6.1f} "
                      f"{x['win_rate']*100:6.1f}% {x['break_even_wr']*100:5.1f}% "
                      f"{x['ev_bps']:+8.2f} {x['worst_bps']:+8.1f} "
                      f"{x['max_dd_bps']:+8.0f} {x['n']:6d}")

    print(f"\n{'='*84}\nTHE 66 % CELL — highest WR with positive EV, per venue"
          f"\n{'='*84}")
    for blk in res:
        if blk.get("status") != "ok":
            continue
        for vname, rt in RTS:
            cells = [x for x in blk["cells"]
                     if x["venue"] == vname and x["ev_bps"] > 0]
            if not cells:
                print(f"  {blk['asset']} {vname:9s}: none")
                continue
            b = max(cells, key=lambda x: x["win_rate"])
            verdict = "REACHES 66 %" if b["win_rate"] >= TARGET else \
                f"{TARGET/b['win_rate'] - 1:.0%} short"
            print(f"  {blk['asset']} {vname:9s} {b['venue']:9s} "
                  f"{b['tf']}/{b['minutes']}m tp{b['tp_bps']:g}/sl{b['sl_bps']:g}: "
                  f"WR {b['win_rate']*100:.1f} %  EV {b['ev_bps']:+.2f} bps  "
                  f"worst {b['worst_bps']:+.0f}  DD {b['max_dd_bps']:+.0f}  "
                  f"→ {verdict}")
    print(f"\nNOTE: ties are scored as the stop; windows are non-overlapping so "
          f"n is the true sample.  A high WR with a large sl is a low-payoff "
          f"book and its worst-window / drawdown columns are the price.")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
