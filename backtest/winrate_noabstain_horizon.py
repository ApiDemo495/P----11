"""No-abstention horizon sweep: can a FULL-COVERAGE book reach a higher
net win rate by moving the horizon out, keeping a payable geometry?

Every epoch-aligned window is graded (no skips, no filters). Sides are
full-coverage causal rules. Geometry alternatives are all cost-payable
(TP >= 1.5 x round trip, reward:risk >= 0.25).

Usage: python3 backtest/winrate_noabstain_horizon.py
"""
from __future__ import annotations
import json, os
import numpy as np
from backtest.winrate_noabstain import (load_5m, wilson, scale_at,
                                       hourly_closes, rsi_series, FRICTION)

RESULTS = os.path.join(os.path.dirname(__file__), "results")
HORIZONS = [("1d", 1440), ("2d", 2880), ("5d", 7200), ("10d", 14400)]


def windows(t, w_min):
    W = w_min * 60
    t0 = int(np.ceil(int(t[0]) / W) * W)
    starts = np.arange(t0, int(t[-1]) + 300 - W + 1, W, dtype=np.int64)
    i_entry = np.searchsorted(t, starts - 300, side="right") - 1
    i_exit = np.searchsorted(t, starts + W - 300, side="right") - 1
    ok = (i_entry >= 0) & (i_exit > i_entry) & (i_exit < len(t))
    return starts[ok], i_entry[ok], i_exit[ok]


def sides(t, c, starts, i_entry, w_min):
    n = len(starts)
    up = np.ones(n, dtype=int)
    h_ts, h_c = hourly_closes(t, c)
    hb = (t[i_entry] // 3600) * 3600
    j = np.searchsorted(h_ts, hb, side="right") - 1
    mom = np.ones(n, dtype=int)
    d = np.full(n, np.nan)
    ok = j >= 44
    d[ok] = c[i_entry[ok]] - h_c[j[ok] - 44]
    mom[ok] = np.where(d[ok] > 0, 1, np.where(d[ok] < 0, -1, 1))
    W = w_min * 60
    base = np.searchsorted(t, starts - W, side="right") - 1
    rev = np.ones(n, dtype=int)
    for i in range(n):
        b = int(base[i])
        ref = b if b >= 0 else int(i_entry[i]) - 1
        if ref < 0:
            continue
        mv = c[i_entry[i]] - c[ref]
        rev[i] = -1 if mv > 0 else (1 if mv < 0 else 1)
    rsi_h = rsi_series(h_c, 14)
    rf = np.ones(n, dtype=int)
    for i in range(n):
        jj = int(j[i])
        if 0 <= jj < len(rsi_h):
            rf[i] = -1 if rsi_h[jj] > 50 else 1
    return {"up": up, "mom44h": mom, "revW": rev, "rsi_fade": rf}


def settle(side_s, entry, exit_, scale, h, l, i_entry, i_exit, rt_bps, k, m):
    n = len(side_s)
    pnl = np.zeros(n)
    rt = rt_bps * 1e-4
    for i in range(n):
        side = int(side_s[i]); e = float(entry[i])
        sl_d = float(scale[i]) * m
        tp_d = float(scale[i]) * m * k
        tp_bps = tp_d / e * 1e4
        floor = 1.5 * rt_bps
        if 0 < tp_bps < floor:            # cost floor: widen BOTH barriers
            f = floor / tp_bps
            tp_d *= f; sl_d *= f
        tp_px = e + tp_d * side
        sl_px = e - sl_d * side
        a, b = int(i_entry[i]) + 1, int(i_exit[i])
        out = 0
        for jj in range(a, b + 1):
            hi, lo = float(h[jj]), float(l[jj])
            if side > 0:
                s_h, t_h = lo <= sl_px, hi >= tp_px
            else:
                s_h, t_h = hi >= sl_px, lo <= tp_px
            if s_h:
                out = -1; break
            if t_h:
                out = 1; break
        if out < 0:
            pnl[i] = -sl_d / e - rt
        elif out > 0:
            pnl[i] = tp_d / e - rt
        else:
            pnl[i] = side * (float(exit_[i]) / e - 1.0) - rt
    return pnl


def stats(pnl):
    n = len(pnl); w = int(np.sum(pnl > 0))
    lo, hi = wilson(w, n)
    gp = float(np.sum(pnl[pnl > 0])); gl = float(-np.sum(pnl[pnl < 0]))
    return {"n": n, "wr": round(w / n, 4) if n else 0.0,
            "ci95": [round(lo, 4), round(hi, 4)],
            "ev_bps": round(float(np.mean(pnl) * 1e4), 2) if n else 0.0,
            "pf": round(gp / gl, 3) if gl > 0 else None}


def main():
    out = {}
    for asset in ["BTC", "ETH"]:
        t, o, h, l, c, v = load_5m(asset)
        rt = 2.0 * FRICTION[asset]
        for hname, w_min in HORIZONS:
            starts, i_entry, i_exit = windows(t, w_min)
            if len(starts) < 40:
                continue
            entry, exit_ = c[i_entry], c[i_exit]
            sc = scale_at(t, c, starts, w_min)
            ss = sides(t, c, starts, i_entry, w_min)
            n = len(starts); cut = int(n * 0.65)
            print(f"\n=== {asset} {hname} ({w_min}m) n={n} RT={rt:.1f}bps ===")
            cell = {}
            for sname, s in ss.items():
                cell[sname] = {}
                for gname, (k, m) in {"prop_k0.5_m1.0": (0.5, 1.0),
                                      "prop_k1.0_m1.0": (1.0, 1.0),
                                      "prop_k1.0_m2.0": (1.0, 2.0),
                                      "prop_k2.0_m1.0": (2.0, 1.0)}.items():
                    pnl = settle(s, entry, exit_, sc, h, l, i_entry, i_exit, rt, k, m)
                    tr, ho = stats(pnl[:cut]), stats(pnl[cut:])
                    cell[sname][gname] = {"train": tr, "holdout": ho}
                    print(f"  {sname:9s} {gname:14s} train {tr['wr']*100:5.1f}% EV {tr['ev_bps']:+7.1f}"
                          f" | hold {ho['wr']*100:5.1f}% EV {ho['ev_bps']:+7.1f} PF {ho['pf']}")
            out[f"{asset}|{hname}"] = {"asset": asset, "horizon": hname, "minutes": w_min,
                                       "rt_bps": rt, "n": n, "cells": cell}
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "winrate_noabstain_horizon.json"), "w"), indent=1)
    print("\n-> backtest/results/winrate_noabstain_horizon.json")


if __name__ == "__main__":
    main()
