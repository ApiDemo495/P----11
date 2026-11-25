"""The no-abstention win-rate FRONTIER on real tape.

Every window trades. The only free parameters are the side (always ±1)
and the barrier geometry (TP = k*SL, SL = m*scale). The cost floor is
applied by widening BOTH barriers, so a geometry can never buy win rate
by aiming the target inside the round trip — the win rate reported here
is one that actually settles as a net win.

Sweeping k traces the whole win-rate range: tight k → high win rate,
wide k → low win rate. The point of the table is the exchange rate
between the two, and the ceiling: the highest net win rate any payable
geometry reaches on that cell.

Data: 5m tape (Jan–Aug 2026) for 22/44/130/247 min cells (BTC, ETH);
1h tape (Jan 2024–Aug 2026) for 1d/2d/3d cells (BTC, ETH, BNB, PAXG).

Usage: PYTHONPATH=. python3 backtest/winrate_frontier.py
"""
from __future__ import annotations
import json, os
import numpy as np

from backtest.winrate_noabstain import (load_5m, wilson, scale_at as scale5,
                                       hourly_closes, rsi_series)
from backtest.winrate_noabstain_wf import (load_1h, windows, scale_at, sides,
                                           FRICTION)

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
K_GRID = [0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00, 1.50, 2.00, 3.00]
M_GRID = [0.55, 1.00, 2.00]
MIN_TP_OVER_RT = 1.5


def settle(side_s, entry, exit_, sc, h, l, i_entry, i_exit, rt_bps, k, m):
    n = len(side_s)
    pnl = np.zeros(n)
    rt = rt_bps * 1e-4
    floor = MIN_TP_OVER_RT * rt_bps
    for i in range(n):
        sd = int(side_s[i]); e = float(entry[i])
        sl_d = float(sc[i]) * m
        tp_d = float(sc[i]) * m * k
        tp_bps = tp_d / e * 1e4
        if 0 < tp_bps < floor:
            f = floor / tp_bps
            tp_d *= f; sl_d *= f
        tp_px = e + tp_d * sd
        sl_px = e - sl_d * sd
        out = 0
        for j in range(int(i_entry[i]) + 1, int(i_exit[i]) + 1):
            hi, lo = float(h[j]), float(l[j])
            if sd > 0:
                sh, th = lo <= sl_px, hi >= tp_px
            else:
                sh, th = hi >= sl_px, lo <= tp_px
            if sh:
                out = -1; break
            if th:
                out = 1; break
        if out < 0:
            pnl[i] = -sl_d / e - rt
        elif out > 0:
            pnl[i] = tp_d / e - rt
        else:
            pnl[i] = sd * (float(exit_[i]) / e - 1.0) - rt
    return pnl


def stat(pnl):
    n = len(pnl)
    if n < 10:
        return None
    w = int(np.sum(pnl > 0))
    lo, hi = wilson(w, n)
    gp = float(np.sum(pnl[pnl > 0])); gl = float(-np.sum(pnl[pnl < 0]))
    return {"n": n, "wr": round(w / n, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "ev_bps": round(float(np.mean(pnl) * 1e4), 2),
            "pf": round(gp / gl, 3) if gl > 0 else None}


def grid_5m(asset, tf, w_min):
    t, o, h, l, c, v = load_5m(asset)
    W = w_min * 60
    t0 = int(np.ceil(int(t[0]) / W) * W)
    starts = np.arange(t0, int(t[-1]) + 300 - W + 1, W, dtype=np.int64)
    i_entry = np.searchsorted(t, starts - 300, side="right") - 1
    i_exit = np.searchsorted(t, starts + W - 300, side="right") - 1
    ok = (i_entry >= 0) & (i_exit > i_entry) & (i_exit < len(t))
    starts, i_entry, i_exit = starts[ok], i_entry[ok], i_exit[ok]
    entry, exit_ = c[i_entry], c[i_exit]
    sc = scale5(t, c, starts, w_min)
    h_ts, h_c = hourly_closes(t, c)
    hb = (t[i_entry] // 3600) * 3600
    j = np.searchsorted(h_ts, hb, side="right") - 1
    n = len(starts)
    up = np.ones(n, dtype=int)
    mom = np.ones(n, dtype=int)
    d = np.full(n, np.nan)
    ok = j >= 44
    d[ok] = c[i_entry[ok]] - h_c[j[ok] - 44]
    mom[ok] = np.where(d[ok] > 0, 1, -1)
    base = np.searchsorted(t, starts - W, side="right") - 1
    rev = np.ones(n, dtype=int)
    for i in range(n):
        ref = int(base[i]) if base[i] >= 0 else int(i_entry[i]) - 1
        if ref < 0:
            continue
        mv = c[i_entry[i]] - c[ref]
        rev[i] = -1 if mv > 0 else 1
    rr = rsi_series(h_c, 14)
    rf = np.ones(n, dtype=int)
    for i in range(n):
        jj = int(j[i])
        rf[i] = -1 if (0 <= jj < len(rr) and rr[jj] > 50) else 1
    return {"up": up, "mom44h": mom, "revW": rev, "rsi_fade": rf}, \
        entry, exit_, sc, h, l, i_entry, i_exit


def main():
    rt_map = dict(FRICTION)
    out = {"k_grid": K_GRID, "m_grid": M_GRID,
           "min_tp_over_rt": MIN_TP_OVER_RT,
           "note": ("net win = settled pnl > 0 including the full round trip; "
                    "cost floor widens BOTH barriers"),
           "cells": {}}
    print("cell            side      m     |  best WR (k)          EV bps   |  WR at k=1.0")
    for asset in ["BTC", "ETH"]:
        for tf, w_min in [("tf1", 22), ("tf2", 44), ("tf3", 130), ("tf4", 247)]:
            ss, entry, exit_, sc, h, l, i_entry, i_exit = grid_5m(asset, tf, w_min)
            rt = 2.0 * rt_map[asset]
            n = len(entry); cut = int(n * 0.60)
            cell = {"asset": asset, "tf": tf, "w_min": w_min, "rt_bps": rt,
                    "n": n, "source": "5m", "sides": {}}
            for sname, s in ss.items():
                rows = []
                for m in M_GRID:
                    for k in K_GRID:
                        st = stat(settle(s, entry, exit_, sc, h, l,
                                         i_entry, i_exit, rt, k, m)[cut:])
                        if st:
                            rows.append({"k": k, "m": m, **st})
                if not rows:
                    continue
                best = max(rows, key=lambda r: r["wr"])
                at1 = [r for r in rows if r["k"] == 1.0 and r["m"] == 1.0]
                cell["sides"][sname] = {"frontier": rows, "best_wr": best,
                                        "at_k1": at1[0] if at1 else None}
                print(f"{asset}|{tf:4s} {sname:9s} m={best['m']:.2f} | "
                      f"{best['wr']*100:5.1f}% (k={best['k']:<4}) "
                      f"EV {best['ev_bps']:+7.1f} n={best['n']:<5} | "
                      f"{(at1[0]['wr']*100 if at1 else float('nan')):5.1f}%")
            out["cells"][f"{asset}|{tf}"] = cell

    for asset in ["BTC", "ETH", "BNB", "PAXG"]:
        t, o, h, l, c = load_1h(asset)
        rt = 2.0 * rt_map[asset]
        for hz, w_min in [("1d", 1440), ("2d", 2880), ("3d", 4320)]:
            starts, i_entry, i_exit = windows(t, w_min)
            if len(starts) < 60:
                continue
            entry, exit_ = c[i_entry], c[i_exit]
            sc = scale_at(t, c, starts, w_min)
            ss = sides(t, c, starts, i_entry, w_min)
            n = len(entry); cut = int(n * 0.60)
            cell = {"asset": asset, "tf": hz, "w_min": w_min, "rt_bps": rt,
                    "n": n, "source": "1h", "sides": {}}
            for sname, s in ss.items():
                rows = []
                for m in M_GRID:
                    for k in K_GRID:
                        st = stat(settle(s, entry, exit_, sc, h, l,
                                         i_entry, i_exit, rt, k, m)[cut:])
                        if st:
                            rows.append({"k": k, "m": m, **st})
                if not rows:
                    continue
                best = max(rows, key=lambda r: r["wr"])
                at1 = [r for r in rows if r["k"] == 1.0 and r["m"] == 1.0]
                cell["sides"][sname] = {"frontier": rows, "best_wr": best,
                                        "at_k1": at1[0] if at1 else None}
                print(f"{asset}|{hz:4s} {sname:9s} m={best['m']:.2f} | "
                      f"{best['wr']*100:5.1f}% (k={best['k']:<4}) "
                      f"EV {best['ev_bps']:+7.1f} n={best['n']:<5} | "
                      f"{(at1[0]['wr']*100 if at1 else float('nan')):5.1f}%")
            out["cells"][f"{asset}|{hz}"] = cell

    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "winrate_frontier.json"), "w"), indent=1)
    print("\n-> backtest/results/winrate_frontier.json")


if __name__ == "__main__":
    main()
