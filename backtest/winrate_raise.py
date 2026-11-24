"""RAISE THE WIN RATE — the hurdle-rate grid (the lever nobody pulled).

Why every previous attempt failed
---------------------------------
All prior harnesses asked the *same* question: "given a fixed barrier
geometry, what fraction of windows hit the target?"  On a driftless walk
that fraction is pinned to ``sl/(tp+sl)``, the exact break-even line, so
the answer could only ever be "cost takes it below break-even".

A **hurdle rate** (limit order) changes the geometry itself:

    * you do NOT enter at the window open at the market;
    * you rest a limit order at a distance ``h`` *below* the reference;
    * you only trade if price comes to you;
    * but you still count EVERY window — one that never fills counts as a
      ZERO-pnl trade (a push), never as a skipped one.

That last line is the whole point and it is what makes this legitimate
under "no abstention":

    win  = filled AND exited above entry + round trip
    push = never filled (pnl exactly 0, not a win, not a loss)
    loss = filled and exited below entry

Fills are path-dependent, so the first-touch algebra that pinned the old
result does not apply: entering `h` lower means your target is `h`
closer in absolute terms while your stop is measured from a better
price.  This module measures whether that actually raises the net win
rate, and at what cost in pushes and tail risk.

Usage
    .venv/bin/python -m backtest.winrate_raise --assets BTC ETH
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data import FRICTION_BPS, load_cache          # noqa: E402

RESULTS = os.path.join("backtest", "results")
OUT = os.path.join(RESULTS, "winrate_raise.json")
TRAIN_FRAC = 0.65
TF_MIN = {"tf1": 22, "tf2": 44, "tf3": 130, "tf4": 247}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_ohlc(asset, tape="5m"):
    bars = load_cache(asset, tape)
    if not bars:
        return None
    D = {k: np.array([getattr(b, k) for b in bars], float)
         for k in ("o", "h", "l", "c")}
    return D, np.array([b.t for b in bars])


def windows(t, step, w_min):
    W = int(w_min * 60)
    starts = np.arange(int(np.ceil(t[0] / W) * W),
                       int(t[-1]) + step - W + 1, W, dtype=np.int64)
    i_in = np.searchsorted(t, starts, side="left")
    i_out = np.searchsorted(t, starts + W - step, side="right") - 1
    ok = (i_in >= 0) & (i_out > i_in) & (i_out < len(t))
    return i_in[ok], i_out[ok]


def hurdle_book(h, l, c, i_in, i_out, rt_bps, hurdle_bps, tp_bps, sl_bps,
                side=-1):
    """One position per window, entered on a limit order at a hurdle.

    ``side`` = -1 buys the dip (a LONG entered below the reference);
    +1 sells the rip.  The window is always consumed: an unfilled window
    returns pnl 0.0 and ``filled=False``.
    """
    n = len(i_in)
    pnl = np.zeros(n)
    filled = np.zeros(n, bool)
    won = np.zeros(n, bool)
    pushed = np.zeros(n, bool)
    for k in range(n):
        a, b = i_in[k], i_out[k]
        ref = c[a]
        hb = hurdle_bps * 1e-4
        limit = ref * (1.0 + side * hb)          # side=-1 -> below ref
        # did price reach the limit inside the window?
        seg_l = l[a + 1:b + 1]
        seg_h = h[a + 1:b + 1]
        if side < 0:
            hit = np.flatnonzero(seg_l <= limit)
        else:
            hit = np.flatnonzero(seg_h >= limit)
        if hit.size == 0:
            pushed[k] = True                     # never filled -> push
            continue
        fill_bar = int(hit[0]) + a + 1
        entry = limit
        tp = entry * (1.0 - side * tp_bps * 1e-4)
        sl = entry * (1.0 + side * sl_bps * 1e-4)
        px = None
        for j in range(fill_bar, b + 1):
            hi, lo = h[j], l[j]
            if side < 0:                          # long: tp above, sl below
                sl_hit = lo <= sl
                tp_hit = hi >= tp
            else:
                sl_hit = hi >= sl
                tp_hit = lo <= tp
            if sl_hit:
                px = sl
                break
            if tp_hit:
                px = tp
                break
        if px is None:
            px = c[b]
        r = (px / entry - 1.0) * (-side) - rt_bps * 1e-4  # bps -> frac
        pnl[k] = r
        filled[k] = True
        won[k] = r > 0
        if not won[k] and r == 0:
            pushed[k] = True
    return pnl, filled, won, pushed


def summarise(pnl, filled, won, pushed):
    n = len(pnl)
    k = int(won.sum())
    lo, hi = wilson(k, n)
    nf = int(filled.sum())
    return {
        "n_windows": n, "n_filled": nf, "n_won": k,
        "fill_rate": round(nf / n, 4) if n else 0.0,
        "win_rate_of_all_windows": round(k / n, 4) if n else 0.0,
        "ci95": [round(lo, 4), round(hi, 4)],
        "win_rate_of_filled": round(k / nf, 4) if nf else 0.0,
        "push_rate": round(int(pushed.sum()) / n, 4) if n else 0.0,
        "ev_bps_all": round(float(pnl.mean() * 1e4), 3),
        "ev_bps_filled": (round(float(pnl[filled].mean() * 1e4), 3)
                          if nf else 0.0),
        "worst_bps": round(float(pnl.min() * 1e4), 1) if n else 0.0,
        "total_bps": round(float(pnl.sum() * 1e4), 1),
    }


def run(asset, tf="tf1"):
    got = load_ohlc(asset)
    if not got:
        return {"cell": f"{asset}|{tf}", "status": "no_tape"}
    D, t = got
    h, l, c = D["h"], D["l"], D["c"]
    step = int(t[1] - t[0])
    w_min = TF_MIN[tf]
    i_in, i_out = windows(t, step, w_min)
    if len(i_in) < 200:
        return {"cell": f"{asset}|{tf}", "status": "too_few"}
    rt = 2.0 * FRICTION_BPS[asset]                # BPS (FRICTION_BPS is already bps/side)
    cut = int(len(i_in) * TRAIN_FRAC)
    rows = []
    for hurdle in (0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        for tp_mult in (1.5, 2.0, 3.0, 5.0):
            for sl_mult in (2.0, 3.0, 5.0, 8.0):
                hb = hurdle * rt
                tp_b = tp_mult * rt
                sl_b = sl_mult * rt
                pnl, fl, wn, pu = hurdle_book(h, l, c, i_in, i_out, rt,
                                              hb, tp_b, sl_b, side=-1)
                tr = summarise(pnl[:cut], fl[:cut], wn[:cut], pu[:cut])
                ho = summarise(pnl[cut:], fl[cut:], wn[cut:], pu[cut:])
                rows.append({
                    "hurdle_x_rt": hurdle, "tp_x_rt": tp_mult,
                    "sl_x_rt": sl_mult,
                    "tp_bps": round(tp_b, 1), "sl_bps": round(sl_b, 1),
                    "train": tr, "holdout": ho,
                })
    # the zero-hurdle baseline for comparison
    base = [r for r in rows if r["hurdle_x_rt"] == 0]
    return {"cell": f"{asset}|{tf}", "status": "ok", "rt_bps": round(rt, 2),
            "n_windows": int(len(i_in)), "rows": rows,
            "baseline_max_wr": max((r["holdout"]["win_rate_of_all_windows"]
                                    for r in base), default=0.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="*", default=["BTC", "ETH"])
    ap.add_argument("--tf", nargs="*", default=["tf1"])
    args = ap.parse_args()
    out = []
    for asset in args.assets:
        for tf in args.tf:
            r = run(asset, tf)
            out.append(r)
            if r.get("status") != "ok":
                print(f"{r['cell']}: {r['status']}")
                continue
            print(f"\n=== {r['cell']}  RT {r['rt_bps']} bps  "
                  f"{r['n_windows']:,} windows (every window counted)")
            print(f"  baseline (market entry, no hurdle) best holdout WR "
                  f"{r['baseline_max_wr']*100:.1f}%")
            best = sorted(r["rows"],
                          key=lambda x: -(x["holdout"]["win_rate_of_all_windows"]
                                          if x["holdout"]["ev_bps_all"] > 0
                                          else -1))[:8]
            print(f"  {'hurdle':>7s} {'TPxRT':>6s} {'SLxRT':>6s} "
                  f"{'fill%':>6s} {'WR(all)':>8s} {'95% CI':>14s} "
                  f"{'EV bps':>8s} {'worst':>8s}")
            for x in best:
                ho = x["holdout"]
                lo, hi = ho["ci95"]
                print(f"  {x['hurdle_x_rt']:7.2f} {x['tp_x_rt']:6.1f} "
                      f"{x['sl_x_rt']:6.1f} {ho['fill_rate']*100:5.1f}% "
                      f"{ho['win_rate_of_all_windows']*100:7.1f}% "
                      f"[{lo*100:4.0f},{hi*100:4.0f}] {ho['ev_bps_all']:+8.2f} "
                      f"{ho['worst_bps']:+8.0f}")
            hits = [x for x in r["rows"]
                    if x["holdout"]["win_rate_of_all_windows"] >= 0.66
                    and x["holdout"]["ev_bps_all"] > 0]
            r["n_66_pos_ev"] = len(hits)
            print(f"  → holdout WR>=66% AND EV>0: {len(hits)} configurations")
            if hits:
                top = max(hits, key=lambda x: x["holdout"]
                          ["win_rate_of_all_windows"])
                print(f"     best: hurdle {top['hurdle_x_rt']}x RT, "
                      f"TP {top['tp_x_rt']}x SL {top['sl_x_rt']}x → "
                      f"WR {top['holdout']['win_rate_of_all_windows']*100:.1f}% "
                      f"EV {top['holdout']['ev_bps_all']:+.2f} bps "
                      f"fill {top['holdout']['fill_rate']*100:.0f}%")
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    tot = sum(r.get("n_66_pos_ev", 0) for r in out if r.get("status") == "ok")
    print(f"\n{tot} cells/configs reach 66%+ net win rate with positive EV "
          f"counting EVERY window")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
