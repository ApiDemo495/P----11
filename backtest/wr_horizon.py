"""Where does the win rate actually cross 66 %? A horizon study.

The finding this module formalises
---------------------------------
A short-horizon barrier trade cannot pay a 29 bps round trip: measured
time-exit win rates are 6 % at 22 min and 24 % at 130 min, with EV pinned
at −round trip.  But the same measurement at *long* horizons rises with
the horizon, because a 3-year crypto tape has real positive drift
(BTC +127 %, ETH +36 % over 2023-11 → 2026-08).

Win rate therefore is a function of holding period, not a fixed property
of the market.  This module maps that curve, and — crucially — does not
stop at the point estimate.  A 66 % win rate computed on 33 non-overlapping
monthly windows is not a result; it is a coin landing 22 times.  Every row
carries a Wilson interval so the claim can be judged, and the walk-forward
split is applied to the horizon choice itself (pick on the first 65 %,
report the last 35 %).

What "continuous, no abstention" means here
-------------------------------------------
The book is ALWAYS long, and the next window opens the moment the last
one closes — no gaps, no skipped windows, no cash.  That is the user's
stated cycle.  Because a long-only book never abstains, the win rate is
simply the share of windows that closed higher than they opened, net of
the round trip.

Usage
    .venv/bin/python -m backtest.wr_horizon --assets BTC ETH
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
OUT = os.path.join(RESULTS, "wr_horizon.json")
TRAIN_FRAC = 0.65

#: horizons in days — geometric, so each row is a similar ratio away
HORIZONS_D = (0.125, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval — honest about small n.

    A normal approximation returns a zero-width interval at p=1 and is
    badly wrong at p=0.66 with n=33; Wilson is the right tool and it is
    what makes the small-n rows in this report readable.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def windows_hold(t, step, hold_s):
    """Non-overlapping windows: next entry = previous exit. No gaps."""
    W = int(hold_s)
    starts = np.arange(int(np.ceil(t[0] / W) * W),
                       int(t[-1]) + step - W + 1, W, dtype=np.int64)
    i_in = np.searchsorted(t, starts, side="left")
    i_out = np.searchsorted(t, starts + W - step, side="right") - 1
    ok = (i_in >= 0) & (i_out > i_in) & (i_out < len(t))
    return i_in[ok], i_out[ok]


def run(asset, horizons=HORIZONS_D):
    bars = load_cache(asset, "5m")
    if not bars:
        return {"asset": asset, "status": "no_tape"}
    t = np.array([b.t for b in bars])
    c = np.array([b.c for b in bars], dtype=float)
    step = int(t[1] - t[0])
    rt = 2.0 * FRICTION_BPS[asset] * 1e-4
    rows = []
    for d in horizons:
        i_in, i_out = windows_hold(t, step, d * 86400)
        n = len(i_in)
        if n < 8:
            rows.append({"hold_days": d, "n": n, "status": "too_few"})
            continue
        pnl = (c[i_out] / c[i_in] - 1.0) - rt          # always long, net
        wins = pnl > 0
        k, n = int(wins.sum()), len(pnl)
        lo, hi = wilson(k, n)
        cut = max(1, int(n * TRAIN_FRAC))
        rows.append({
            "hold_days": d, "n": n, "wins": k,
            "win_rate": round(k / n, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "ev_bps": round(float(pnl.mean() * 1e4), 2),
            "median_bps": round(float(np.median(pnl) * 1e4), 2),
            "ev_R_proxy": round(float(pnl.mean() / max(pnl.std(), 1e-12)), 4),
            "train_wr": round(float(wins[:cut].mean()), 4),
            "holdout_wr": round(float(wins[cut:].mean()), 4),
            "holdout_n": int(n - cut),
            "holdout_ev_bps": round(float(pnl[cut:].mean() * 1e4), 2),
            "lower_bound_below_66": lo < 0.66,
            "time_in_market": "100 % (contiguous, no abstention)",
        })
    return {"asset": asset, "status": "ok", "rt_bps": round(rt * 1e4, 2),
            "span": [int(t[0]), int(t[-1])], "bars": len(bars),
            "total_return_pct": round(float((c[-1] / c[0] - 1) * 100), 1),
            "rows": rows}


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
        print(f"\n=== {asset}  {r['bars']:,} bars  total return "
              f"{r['total_return_pct']:+.1f}%  RT {r['rt_bps']} bps")
        print(f"  {'hold':>8s} {'n':>6s} {'WR':>7s} {'95% CI':>16s} "
              f"{'EV bps':>9s} {'holdout WR':>11s} {'holdout n':>10s}")
        for row in r["rows"]:
            if row.get("status") == "too_few":
                print(f"  {row['hold_days']:7.3f}d {row['n']:6d}   too few")
                continue
            lo, hi = row["ci95"]
            flag = "" if lo >= 0.66 else ("  ← CI includes 66%"
                                          if hi >= 0.66 else "")
            print(f"  {row['hold_days']:7.3f}d {row['n']:6d} "
                  f"{row['win_rate']*100:6.1f}% [{lo*100:5.1f},{hi*100:5.1f}] "
                  f"{row['ev_bps']:+9.1f} {row['holdout_wr']*100:10.1f}% "
                  f"{row['holdout_n']:10d}{flag}")
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)

    print("\n── the 66 % question, stated properly ──")
    for r in out:
        if r.get("status") != "ok":
            continue
        solid = [row for row in r["rows"]
                 if row.get("win_rate", 0) >= 0.66 and row["ci95"][0] >= 0.66]
        point = [row for row in r["rows"]
                 if row.get("win_rate", 0) >= 0.66]
        print(f"{r['asset']}: point estimate >=66% at "
              f"{[row['hold_days'] for row in point] or 'no horizon'}")
        print(f"{'':6s} CI lower bound >=66% at "
              f"{[row['hold_days'] for row in solid] or 'NO HORIZON'}")
        pos = [row for row in r["rows"] if row.get("ev_bps", -1) > 0]
        print(f"{'':6s} EV > 0 at "
              f"{[row['hold_days'] for row in pos] or 'no horizon'} days")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
