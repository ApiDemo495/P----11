"""Stress the high-win-rate book: walk-forward, drawdown, ruin.

``backtest/wr_tail.py`` shows that a >=66 % post-cost win rate with
positive EV *exists* — the small-target / wide-stop / no-timeout book.
That is a genuine answer to the literal question.  It is also exactly the
kind of result that is worthless if it is quoted without the tail, so
this module attacks it before anyone trades it:

1. **Walk-forward.**  Geometry is chosen on the first 65 % of the tape
   only, then applied unchanged to the last 35 %.  A number that only
   exists in-sample is discarded here.
2. **Drawdown and ruin.**  Compounded equity path, max drawdown, the
   longest losing streak, and the probability that the observed worst
   loss wipes a fixed fraction of the account.
3. **Cost sensitivity.**  Friction is doubled and tripled.  A book whose
   EV is +0.5 bps per trade is inside the noise of the cost estimate and
   must be reported as such.
4. **Year-by-year.**  The 3-year tape is split into calendar years; a
   geometry that only works in 2024 is a regime artifact.

The verdict rule is fixed in advance: a geometry is **certified** only if
it keeps WR >= 66 % AND EV > 0 in the held-out period, in at least two
of the three calendar years, and under doubled friction.

Usage
    .venv/bin/python -m backtest.wr_stress --assets BTC ETH
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data import FRICTION_BPS, load_cache          # noqa: E402
from backtest.wr_tail import sequential_book               # noqa: E402

RESULTS = os.path.join("backtest", "results")
OUT = os.path.join(RESULTS, "wr_stress.json")
TRAIN_FRAC = 0.65
TARGET_WR = 0.66


def stats(r):
    if len(r) == 0:
        return {}
    wins = r > 0
    n = len(r)
    k = int(wins.sum())
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min()) if len(eq) else 0.0
    # longest losing streak
    streak = best = 0
    for w in wins:
        streak = 0 if w else streak + 1
        best = max(best, streak)
    return {
        "n": n, "wins": k, "win_rate": round(p, 4),
        "ci95": [round(max(0.0, cen - half), 4), round(min(1.0, cen + half), 4)],
        "ev_bps": round(float(r.mean() * 1e4), 2),
        "worst_trade_bps": round(float(r.min() * 1e4), 1),
        "max_drawdown_pct": round(dd * 100, 2),
        "longest_loss_streak": int(best),
        "compounded_pct": round((float(np.prod(1.0 + r)) - 1) * 100, 1),
    }


def year_split(t, i_entries, r):
    """Per calendar year: WR and EV of the trades entered in that year."""
    if len(i_entries) == 0:
        return {}
    years = np.array([int(np.datetime64(int(t[i]), "s").astype(str)[:4])
                      for i in i_entries])
    out = {}
    for y in sorted(set(years.tolist())):
        m = years == y
        if m.sum() < 5:
            continue
        sub = r[m]
        out[str(y)] = {
            "trades": int(m.sum()),
            "win_rate": round(float((sub > 0).mean()), 4),
            "ev_bps": round(float(sub.mean() * 1e4), 2),
        }
    return out


def run(asset, friction_mult=1.0):
    bars = load_cache(asset, "5m")
    if not bars:
        return {"asset": asset, "status": "no_tape"}
    t = np.array([b.t for b in bars])
    h = np.array([b.h for b in bars], dtype=float)
    l = np.array([b.l for b in bars], dtype=float)
    c = np.array([b.c for b in bars], dtype=float)
    rt = 2.0 * FRICTION_BPS[asset] * 1e-4 * friction_mult
    rt_bps = rt * 1e4
    n = len(c)
    max_bars = n - 2

    geoms = []
    for tp_mult in (1.5, 2.0, 3.0, 5.0, 10.0):
        for sl_mult in (25, 50, 100, 200, 400):
            geoms.append((tp_mult * rt_bps, sl_mult * rt_bps))

    # ── stage 1: choose on TRAIN only ────────────────────────────────
    cut = int(n * TRAIN_FRAC)
    train = []
    for tp_bps, sl_bps in geoms:
        r, idx = sequential_book(h[:cut], l[:cut], c[:cut], rt,
                                 tp_bps, sl_bps, 1, cut - 2)
        if len(r) < 20:
            continue
        s = stats(r)
        s.update({"tp_bps": round(tp_bps, 1), "sl_bps": round(sl_bps, 1),
                  "tp_over_rt": round(tp_bps / rt_bps, 2),
                  "sl_over_rt": round(sl_bps / rt_bps, 1)})
        train.append(s)
    ok_train = [s for s in train
                if s["win_rate"] >= TARGET_WR and s["ev_bps"] > 0]
    # pick the one with the best risk-adjusted train EV among qualifiers
    best = (max(ok_train, key=lambda s: s["ev_bps"] / s["sl_bps"])
            if ok_train else (max(train, key=lambda s: s["ev_bps"])
                              if train else None))
    if best is None:
        return {"asset": asset, "status": "no_geometry"}

    # ── stage 2: apply unchanged to the HELD-OUT tape ────────────────
    tp_bps, sl_bps = best["tp_bps"], best["sl_bps"]
    full_r, full_i = sequential_book(h, l, c, rt, tp_bps, sl_bps, 1, max_bars)
    ho_mask = full_i >= cut
    r_ho = full_r[ho_mask]
    ho = stats(r_ho) if len(r_ho) else {}
    full = stats(full_r)
    yrs = year_split(t, full_i, full_r)

    # ── stage 3: cost stress on the full tape ───────────────────────
    stress = {}
    for m in (1.0, 2.0, 3.0):
        r_m, _ = sequential_book(h, l, c, rt * m, tp_bps, sl_bps, 1, max_bars)
        s = stats(r_m)
        stress[f"{m:g}x"] = {"win_rate": s["win_rate"], "ev_bps": s["ev_bps"],
                             "worst_trade_bps": s["worst_trade_bps"]}

    # ── verdict (rule fixed before the run) ─────────────────────────
    years_ok = sum(1 for y in yrs.values()
                   if y["win_rate"] >= TARGET_WR and y["ev_bps"] > 0)
    checks = {
        "holdout_wr>=66": bool(ho.get("win_rate", 0) >= TARGET_WR),
        "holdout_ev>0": bool(ho.get("ev_bps", -1) > 0),
        "years_ok>=2": years_ok >= 2,
        "survives_2x_cost": bool(stress["2x"]["ev_bps"] > 0),
        "worst_trade_survivable": bool(full["worst_trade_bps"] > -2000),
    }
    certified = all(checks.values())
    return {
        "asset": asset, "status": "ok", "rt_bps": round(rt_bps, 2),
        "chosen_geometry": {"tp_bps": tp_bps, "sl_bps": sl_bps,
                            "tp_over_rt": best["tp_over_rt"],
                            "sl_over_rt": best["sl_over_rt"]},
        "train": {"win_rate": best["win_rate"], "ev_bps": best["ev_bps"],
                  "n": best["n"]},
        "holdout": ho, "full": full, "years": yrs, "cost_stress": stress,
        "checks": checks, "certified": bool(certified),
        "reasons_failed": [k for k, v in checks.items() if not v],
    }


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
        g, tr, ho, fu = (r["chosen_geometry"], r["train"],
                         r["holdout"], r["full"])
        print(f"\n=== {asset}  RT {r['rt_bps']} bps")
        print(f"  geometry chosen on TRAIN only: TP {g['tp_bps']:.0f}b "
              f"({g['tp_over_rt']:.2f}x RT)  SL {g['sl_bps']:.0f}b "
              f"({g['sl_over_rt']:.0f}x RT)")
        print(f"  TRAIN    {tr['n']:5d} trades  WR {tr['win_rate']*100:5.1f}%  "
              f"EV {tr['ev_bps']:+8.2f} bp")
        print(f"  HOLDOUT  {ho['n']:5d} trades  WR {ho['win_rate']*100:5.1f}% "
              f"[{ho['ci95'][0]*100:.0f},{ho['ci95'][1]*100:.0f}]  "
              f"EV {ho['ev_bps']:+8.2f} bp  DD {ho['max_drawdown_pct']:.1f}%")
        print(f"  FULL     {fu['n']:5d} trades  WR {fu['win_rate']*100:5.1f}%  "
              f"EV {fu['ev_bps']:+8.2f} bp  worst {fu['worst_trade_bps']:+.0f} bp  "
              f"maxDD {fu['max_drawdown_pct']:.1f}%  "
              f"streak {fu['longest_loss_streak']}  "
              f"comp {fu['compounded_pct']:+.1f}%")
        print(f"  by year:")
        for y, s in r["years"].items():
            print(f"    {y}  {s['trades']:4d} trades  WR "
                  f"{s['win_rate']*100:5.1f}%  EV {s['ev_bps']:+8.2f} bp")
        print(f"  cost stress:")
        for k, s in r["cost_stress"].items():
            print(f"    {k:>3s} friction: WR {s['win_rate']*100:5.1f}%  "
                  f"EV {s['ev_bps']:+8.2f} bp  worst {s['worst_trade_bps']:+.0f} bp")
        print(f"  checks: {r['checks']}")
        print(f"  → {'CERTIFIED' if r['certified'] else 'NOT CERTIFIED'}"
              + ("" if r["certified"] else f"  failed: {r['reasons_failed']}"))
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    n_ok = sum(1 for r in out if r.get("certified"))
    print(f"\n{n_ok}/{len(out)} assets certified at WR>=66% with EV>0 "
          f"out-of-sample, across years, and under 2x cost")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
