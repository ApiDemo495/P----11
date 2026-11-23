"""HIGH WIN RATE WITH POSITIVE EV — the small-target / wide-stop / no-timeout book.

The mechanism, stated before any number is quoted
------------------------------------------------
A win is defined as *closing positive after the full round trip*.  The
win **rate** and the win **size** are separate dials:

* a target just above the round trip is touched very often → high win rate;
* a stop far away is touched very rarely → the loss, when it comes, is huge.

So a high post-cost win rate is not impossible at all — it is the normal
result of a wide stop with a small target, and the cost is tail risk, not
win rate.  The earlier harnesses could not see this because they capped
the holding period at one window, which forced most trades to time out
near zero and drag the win rate down.

This module therefore drops the time cap (a trade runs until its target
or its stop is touched, or until the tape ends) and simulates the
**continuous, non-overlapping, always-in-the-market** book the user asked
for: the next entry opens on the bar after the previous exit, forever.

What is reported
----------------
For every geometry: win rate (net), EV in bps and in units of risk, the
*rare* loss size, worst single trade, and the compounded result.  A high
win rate is only a good answer if the tail is survivable, so the tail is
printed next to the win rate and never behind it.

Usage
    .venv/bin/python -m backtest.wr_tail --assets BTC ETH
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
OUT = os.path.join(RESULTS, "wr_tail.json")


def sequential_book(h, l, c, rt, tp_bps, sl_bps, side, max_bars):
    """One position at a time, re-entered the bar after every exit.

    Returns per-trade net returns (fraction) and the bar index of entry.
    The stop wins ties, and a trade that neither hits its target nor its
    stop before ``max_bars`` exits at the market — so nothing is dropped.
    """
    tp = tp_bps * 1e-4
    sl = sl_bps * 1e-4
    n = len(c)
    out_r = []
    out_i = []
    i = 1
    while i < n - 1:
        entry = c[i]
        end = min(n - 1, i + max_bars)
        if side > 0:
            tp_px = entry * (1.0 + tp)
            sl_px = entry * (1.0 - sl)
            seg_h = h[i + 1:end + 1]
            seg_l = l[i + 1:end + 1]
            tp_i = np.flatnonzero(seg_h >= tp_px)
            sl_i = np.flatnonzero(seg_l <= sl_px)
            it = int(tp_i[0]) if tp_i.size else 10 ** 9
            isl = int(sl_i[0]) if sl_i.size else 10 ** 9
        else:
            tp_px = entry * (1.0 - tp)
            sl_px = entry * (1.0 + sl)
            seg_h = h[i + 1:end + 1]
            seg_l = l[i + 1:end + 1]
            tp_i = np.flatnonzero(seg_l <= tp_px)
            sl_i = np.flatnonzero(seg_h >= sl_px)
            it = int(tp_i[0]) if tp_i.size else 10 ** 9
            isl = int(sl_i[0]) if sl_i.size else 10 ** 9
        if isl <= it and isl < 10 ** 9:
            out_r.append(-sl - rt)          # stop (ties go to the stop)
            exit_i = i + 1 + isl
        elif it < 10 ** 9:
            out_r.append(tp - rt)           # target
            exit_i = i + 1 + it
        else:
            out_r.append(side * (c[end] / entry - 1.0) - rt)  # market exit
            exit_i = end
        out_i.append(i)
        i = max(exit_i + 1, i + 1)
    return np.asarray(out_r), np.asarray(out_i)


def summarise(r, risk_bps):
    if len(r) == 0:
        return {"n": 0}
    wins = r > 0
    n = len(r)
    k = int(wins.sum())
    # Wilson interval — a win rate without an interval is a rumour
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    comp = float(np.prod(1.0 + r))
    return {
        "n": n, "wins": k, "win_rate": round(p, 4),
        "ci95": [round(max(0.0, cen - half), 4), round(min(1.0, cen + half), 4)],
        "ev_bps": round(float(r.mean() * 1e4), 2),
        "median_bps": round(float(np.median(r) * 1e4), 2),
        "worst_trade_bps": round(float(r.min() * 1e4), 1),
        "best_trade_bps": round(float(r.max() * 1e4), 1),
        "compounded_x": round(comp, 3),
        "compounded_pct": round((comp - 1) * 100, 1),
        "sharpe_like": round(float(r.mean() / max(r.std(), 1e-12) * np.sqrt(n)), 2),
        "risk_bps": round(risk_bps, 1),
    }


def run(asset, geometries=None):
    bars = load_cache(asset, "5m")
    if not bars:
        return {"asset": asset, "status": "no_tape"}
    h = np.array([b.h for b in bars], dtype=float)
    l = np.array([b.l for b in bars], dtype=float)
    c = np.array([b.c for b in bars], dtype=float)
    rt = 2.0 * FRICTION_BPS[asset] * 1e-4
    rt_bps = rt * 1e4
    max_bars = len(c) - 2

    if geometries is None:
        # target: just above the round trip, up to 4x it.
        # stop: from 5x to 300x the round trip — increasingly far away.
        geometries = []
        for tp_mult in (1.25, 1.5, 2.0, 3.0, 5.0, 10.0):
            for sl_mult in (10, 25, 50, 100, 200, 400):
                geometries.append((tp_mult * rt_bps, sl_mult * rt_bps))

    rows = []
    for tp_bps, sl_bps in geometries:
        if sl_bps <= tp_bps:
            continue
        r, idx = sequential_book(h, l, c, rt, tp_bps, sl_bps, 1, max_bars)
        s = summarise(r, sl_bps)
        s.update({"tp_bps": round(tp_bps, 1), "sl_bps": round(sl_bps, 1),
                  "tp_over_rt": round(tp_bps / rt_bps, 2),
                  "sl_over_rt": round(sl_bps / rt_bps, 1),
                  "rt_bps": round(rt_bps, 2)})
        rows.append(s)
    return {"asset": asset, "status": "ok", "rt_bps": round(rt_bps, 2),
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
        print(f"\n=== {asset}  RT {r['rt_bps']} bps  {r['bars']:,} bars "
              f"— always in market, no abstention")
        print(f"  {'TPxRT':>6s} {'SLxRT':>6s} {'n':>5s} {'WR':>7s} "
              f"{'95% CI':>15s} {'EV bps':>8s} {'worst':>8s} {'comp':>9s}")
        for row in r["rows"]:
            lo, hi = row["ci95"]
            print(f"  {row['tp_over_rt']:6.2f} {row['sl_over_rt']:6.0f} "
                  f"{row['n']:5d} {row['win_rate']*100:6.1f}% "
                  f"[{lo*100:4.0f},{hi*100:4.0f}] {row['ev_bps']:+8.1f} "
                  f"{row['worst_trade_bps']:+8.0f} {row['compounded_pct']:+8.1f}%")
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)

    print("\n── does a 66 % net win rate with EV > 0 exist? ──")
    found = []
    for r in out:
        if r.get("status") != "ok":
            continue
        hits = [row for row in r["rows"]
                if row.get("win_rate", 0) >= 0.66 and row.get("ev_bps", -1) > 0]
        found.extend((r["asset"], h) for h in hits)
        best = max(r["rows"], key=lambda x: (x["win_rate"] if x.get("ev_bps", -1) > 0 else -1))
        if hits:
            print(f"{r['asset']}: {len(hits)} geometries at >=66% WR AND EV>0")
            top = max(hits, key=lambda x: x["win_rate"])
            print(f"{'':6s} best: TP {top['tp_bps']}b SL {top['sl_bps']}b "
                  f"WR {top['win_rate']*100:.1f}% EV {top['ev_bps']:+.1f}bp "
                  f"worst {top['worst_trade_bps']:+.0f}bp "
                  f"comp {top['compounded_pct']:+.1f}%")
        else:
            print(f"{r['asset']}: none")
    print(f"\n{len(found)} (asset, geometry) pairs clear both bars")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
