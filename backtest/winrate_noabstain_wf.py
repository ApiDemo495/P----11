"""No-abstention walk-forward: pick the highest net win rate on TRAIN,
freeze (side, horizon, geometry), then grade 4 sequential OOS blocks.

Constraint honoured throughout: **every window trades** — no filter, no
gate, no no-call. The only things a policy may choose are the side
(always ±1), the horizon and the barrier geometry (kept cost-payable:
TP >= 1.5 x round trip and reward:risk >= 0.25, via BOTH-side widening
so the ratio can never be gamed by squeezing the target).

Usage: PYTHONPATH=. python3 backtest/winrate_noabstain_wf.py
"""
from __future__ import annotations
import csv, json, math, os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
RESULTS = os.path.join(ROOT, "results")

FRICTION = {"BTC": 14.436, "ETH": 14.904, "BNB": 15.482, "PAXG": 18.700}
HORIZONS = {"1d": 1440, "2d": 2880, "3d": 4320, "5d": 7200}
GEOMS = {  # (k, m) — TP = k * SL, SL = m * scale
    "RR0.5/m1.0": (0.5, 1.0),
    "RR1.0/m1.0": (1.0, 1.0),
    "RR1.0/m2.0": (1.0, 2.0),
    "RR2.0/m1.0": (2.0, 1.0),
    "RR0.25/m1.0": (0.25, 1.0),
}
MIN_RR = 0.25
MIN_TP_OVER_RT = 1.5
TRAIN_FRAC = 0.60
N_BLOCKS = 4


def load_1h(sym):
    path = os.path.join(CACHE, f"{sym}_1h.csv")
    rows = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 4))
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    return rows[:, 0].astype(np.int64), rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def windows(t, w_min):
    W = int(w_min) * 60
    t0 = int(np.ceil(int(t[0]) / W) * W)
    starts = np.arange(t0, int(t[-1]) + 3600 - W + 1, W, dtype=np.int64)
    i_entry = np.searchsorted(t, starts - 3600, side="right") - 1
    i_exit = np.searchsorted(t, starts + W - 3600, side="right") - 1
    ok = (i_entry >= 0) & (i_exit > i_entry) & (i_exit < len(t))
    return starts[ok], i_entry[ok], i_exit[ok]


def scale_at(t, c, starts, w_min, samples=60, q=0.80):
    lb = max(1, int(round(w_min / 60)))   # bars per window
    idx = np.searchsorted(t, starts, side="right") - 1
    out = np.full(len(starts), np.nan)
    for j, i in enumerate(idx):
        if i < lb * 2:
            continue
        lo = max(lb, i - samples * lb)
        js = np.arange(lo, i + 1, lb)
        if len(js) < 8:
            continue
        out[j] = float(np.quantile(np.abs(c[js] - c[js - lb]), q))
    med = np.nanmedian(out)
    return np.where(np.isfinite(out) & (out > 0), out, med)


def rsi(closes, n=14):
    out = np.full(len(closes), 50.0)
    for i in range(n, len(closes)):
        g = l = 0.0
        for k in range(i - n + 1, i + 1):
            d = closes[k] - closes[k - 1]
            if d >= 0:
                g += d
            else:
                l -= d
        out[i] = 100.0 if l <= 0 else 100 - 100 / (1 + g / l)
    return out


def sides(t, c, starts, i_entry, w_min):
    n = len(starts)
    up = np.ones(n, dtype=int)
    mom = np.ones(n, dtype=int)
    d = np.full(n, np.nan)
    ok = i_entry >= 44
    d[ok] = c[i_entry[ok]] - c[i_entry[ok] - 44]
    mom[ok] = np.where(d[ok] > 0, 1, np.where(d[ok] < 0, -1, 1))
    base = np.searchsorted(t, starts - w_min * 60, side="right") - 1
    rev = np.ones(n, dtype=int)
    for i in range(n):
        b = int(base[i])
        ref = b if b >= 0 else int(i_entry[i]) - 1
        if ref < 0:
            continue
        mv = c[i_entry[i]] - c[ref]
        rev[i] = -1 if mv > 0 else 1
    r = rsi(c, 14)
    rf = np.ones(n, dtype=int)
    for i in range(n):
        j = int(i_entry[i])
        rf[i] = -1 if r[j] > 50 else 1
    return {"up": up, "mom44h": mom, "revW": rev, "rsi_fade": rf}


def settle(side_s, entry, exit_, sc, h, l, i_entry, i_exit, rt_bps, k, m):
    """Full-coverage barrier settle; cost floor widens BOTH barriers so
    the reward:risk the geometry promises is the one that trades."""
    n = len(side_s)
    pnl = np.zeros(n)
    rt = rt_bps * 1e-4
    floor = MIN_TP_OVER_RT * rt_bps
    for i in range(n):
        side = int(side_s[i])
        e = float(entry[i])
        sl_d = float(sc[i]) * m
        tp_d = float(sc[i]) * m * k
        tp_bps = tp_d / e * 1e4
        if 0 < tp_bps < floor:
            f = floor / tp_bps
            tp_d *= f
            sl_d *= f
        tp_px = e + tp_d * side
        sl_px = e - sl_d * side
        out = 0
        for j in range(int(i_entry[i]) + 1, int(i_exit[i]) + 1):
            hi, lo = float(h[j]), float(l[j])
            if side > 0:
                sh, th = lo <= sl_px, hi >= tp_px
            else:
                sh, th = hi >= sl_px, lo <= tp_px
            if sh:
                out = -1
                break
            if th:
                out = 1
                break
        if out < 0:
            pnl[i] = -sl_d / e - rt
        elif out > 0:
            pnl[i] = tp_d / e - rt
        else:
            pnl[i] = side * (float(exit_[i]) / e - 1.0) - rt
    return pnl


def stats(pnl):
    n = len(pnl)
    if n == 0:
        return {"n": 0, "wr": None, "ev_bps": None, "pf": None, "ci95": None}
    w = int(np.sum(pnl > 0))
    lo, hi = wilson(w, n)
    gp = float(np.sum(pnl[pnl > 0]))
    gl = float(-np.sum(pnl[pnl < 0]))
    return {"n": n, "wr": round(w / n, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "ev_bps": round(float(np.mean(pnl) * 1e4), 2),
            "pf": round(gp / gl, 3) if gl > 0 else None,
            "worst_bps": round(float(np.min(pnl) * 1e4), 1)}


def main():
    out = {"design": {"train_frac": TRAIN_FRAC, "blocks": N_BLOCKS,
                      "min_reward_risk": MIN_RR,
                      "min_tp_over_rt": MIN_TP_OVER_RT,
                      "grid": {"horizons": list(HORIZONS),
                               "geoms": list(GEOMS),
                               "sides": ["up", "mom44h", "revW", "rsi_fade"]}},
           "cells": {}}
    all_oos = []
    for asset in ["BTC", "ETH", "BNB", "PAXG"]:
        t, o, h, l, c = load_1h(asset)
        rt = 2.0 * FRICTION[asset]
        cands = []
        for hz, w_min in HORIZONS.items():
            starts, i_entry, i_exit = windows(t, w_min)
            if len(starts) < 60:
                continue
            entry, exit_ = c[i_entry], c[i_exit]
            sc = scale_at(t, c, starts, w_min)
            ss = sides(t, c, starts, i_entry, w_min)
            n = len(starts)
            cut = int(n * TRAIN_FRAC)
            for sname, s in ss.items():
                for gname, (k, m) in GEOMS.items():
                    if k / m < MIN_RR:
                        continue
                    pnl = settle(s, entry, exit_, sc, h, l, i_entry, i_exit, rt, k, m)
                    train, oos = stats(pnl[:cut]), stats(pnl[cut:])
                    # eligible = train EV positive (a win rate that loses
                    # money is not an improvement) AND train n >= 40
                    eligible = bool(train["ev_bps"] and train["ev_bps"] > 0
                                    and train["n"] >= 40)
                    cands.append({"horizon": hz, "w_min": w_min, "side": sname,
                                  "geom": gname, "k": k, "m": m,
                                  "train": train, "oos": oos, "eligible": eligible,
                                  "pnl_oos": pnl[cut:].tolist()})
        if not cands:
            continue
        elig = [x for x in cands if x["eligible"]]
        pool = elig or cands
        best = max(pool, key=lambda x: (x["train"]["wr"], x["train"]["ev_bps"] or -1e9))
        # per-block OOS grading of the frozen pick
        p = np.array(best["pnl_oos"])
        blocks = np.array_split(p, N_BLOCKS)
        blk = [stats(b) for b in blocks]
        # baseline: current live book — mom44h at the 4h TF, live tp_only k=0.15/m0.55
        bl = None
        starts, i_entry, i_exit = windows(t, 247)
        entry, exit_ = c[i_entry], c[i_exit]
        sc = scale_at(t, c, starts, 247)
        ss = sides(t, c, starts, i_entry, 247)
        n = len(starts)
        cutb = int(n * TRAIN_FRAC)
        pnl_b = settle(ss["mom44h"], entry, exit_, sc, h, l, i_entry, i_exit,
                       rt, 0.15, 0.55)
        bl = {"train": stats(pnl_b[:cutb]), "oos": stats(pnl_b[cutb:]),
              "blocks": [stats(b) for b in np.array_split(pnl_b[cutb:], N_BLOCKS)]}
        # best achievable win rate on OOS across the whole grid (cheating
        # upper bound, reported only to bound what geometry can buy)
        ceil_wr = max(x["oos"]["wr"] for x in cands if x["oos"]["n"] >= 20)
        ceil_pos = [x for x in cands
                    if x["oos"]["wr"] == ceil_wr and x["oos"]["n"] >= 20][0]
        cell = {"asset": asset, "rt_bps": rt, "n_candidates": len(cands),
                "n_eligible": len(elig),
                "selected": {kk: vv for kk, vv in best.items() if kk != "pnl_oos"},
                "oos_blocks": blk,
                "oos_pooled": stats(p),
                "live_baseline_247_mom44h": bl,
                "oos_ceiling": {"wr": ceil_wr, "horizon": ceil_pos["horizon"],
                                "side": ceil_pos["side"], "geom": ceil_pos["geom"],
                                "ev_bps": ceil_pos["oos"]["ev_bps"]}}
        out["cells"][asset] = cell
        s = best
        print(f"\n=== {asset} (RT {rt:.1f}bps, {cell['n_candidates']} candidates, "
              f"{cell['n_eligible']} train-EV>0) ===")
        print(f"  live book 247m mom44h tp_only : train {bl['train']['wr']*100:5.1f}% "
              f"EV {bl['train']['ev_bps']:+7.1f} | OOS {bl['oos']['wr']*100:5.1f}% "
              f"EV {bl['oos']['ev_bps']:+7.1f} (n={bl['oos']['n']})")
        print(f"  SELECTED {s['side']}@{s['horizon']} {s['geom']} (k={s['k']} m={s['m']}): "
              f"train {s['train']['wr']*100:5.1f}% EV {s['train']['ev_bps']:+7.1f} "
              f"(n={s['train']['n']})")
        print(f"           OOS pooled {s['oos']['wr']*100:5.1f}% "
              f"EV {s['oos']['ev_bps']:+7.1f} PF {s['oos']['pf']} "
              f"CI {s['oos']['ci95']} (n={s['oos']['n']})")
        for i, b in enumerate(blk, 1):
            print(f"           block{i}: WR {b['wr']*100:5.1f}% EV {b['ev_bps']:+7.1f} "
                  f"n={b['n']}")
        all_oos.append((asset, s, cell))

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "winrate_noabstain_wf.json"), "w") as f:
        json.dump(out, f, indent=1)

    print("\n================ POOLED OOS ================")
    tot_n = tot_w = 0
    tot_pnl = []
    for asset, s, cell in all_oos:
        o = cell["oos_pooled"]
        tot_n += o["n"]; tot_w += int(round(o["wr"] * o["n"]))
        tot_pnl.extend(np.array(s["pnl_oos"]).tolist())
        print(f"  {asset:5s} {s['side']:9s}@{s['horizon']:3s} {s['geom']:12s} "
              f"WR {o['wr']*100:5.1f}% EV {o['ev_bps']:+7.1f} PF {o['pf']} n={o['n']}")
    lo, hi = wilson(tot_w, tot_n)
    print(f"  ALL   WR {tot_w/tot_n*100:5.1f}% CI [{lo*100:.1f},{hi*100:.1f}] "
          f"EV {np.mean(tot_pnl)*1e4:+.1f}bps n={tot_n}")
    print("-> backtest/results/winrate_noabstain_wf.json")


if __name__ == "__main__":
    main()
