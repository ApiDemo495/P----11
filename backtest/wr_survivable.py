"""Find the geometry that maximises net win rate SUBJECT TO survival.

``backtest/wr_stress.py`` proves the important, uncomfortable thing: a
66 % net win rate is easy — 93.8 % is available on BTC.  It is bought by
pushing the stop so far away that it is almost never touched, which
converts a stream of small wins into a rare catastrophic loss (−29 % in
one trade).  Win rate alone is therefore not a goal; it is a dial that
trades against tail risk.

This module searches the *survivable* region instead.  A geometry is
admissible only if all of these hold, and they are checked on data the
geometry was not chosen on:

    holdout win rate  >= 66 %
    holdout EV        > 0        (net of the full round trip)
    worst single loss > -TAIL_CAP_BPS
    max drawdown       > -DD_CAP_PCT
    EV stays > 0 under doubled friction

Among admissible geometries it reports the one with the highest holdout
win rate, plus the whole frontier (win rate vs worst loss) so the
trade-off is visible rather than hidden behind a single best number.

New lever tested here that the earlier harnesses lacked: a **max-hold
cap**.  A trade that has neither hit its target nor its stop exits at
the market after ``hold_bars``.  That bounds how long capital is exposed
and stops a wide stop from silently becoming "hold through a 30 % crash".

Usage
    .venv/bin/python -m backtest.wr_survivable --assets BTC ETH
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
OUT = os.path.join(RESULTS, "wr_survivable.json")
TRAIN_FRAC = 0.65
TARGET_WR = 0.66
TAIL_CAP_BPS = -1500.0        # a single trade may not lose more than 15 %
DD_CAP_PCT = -35.0


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
    dd = float(((eq - peak) / peak).min())
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
        "compounded_pct": round((np.prod(1.0 + r) - 1) * 100, 1),
    }


def run(asset):
    bars = load_cache(asset, "5m")
    if not bars:
        return {"asset": asset, "status": "no_tape"}
    t = np.array([b.t for b in bars])
    h = np.array([b.h for b in bars], dtype=float)
    l = np.array([b.l for b in bars], dtype=float)
    c = np.array([b.c for b in bars], dtype=float)
    rt = 2.0 * FRICTION_BPS[asset] * 1e-4
    rt_bps = rt * 1e4
    n = len(c)
    cut = int(n * TRAIN_FRAC)

    tp_grid = [1.5, 2.0, 3.0, 5.0, 10.0, 20.0]
    sl_grid = [10, 25, 50, 100, 200]
    hold_grid = [288, 2016, 6048, 24192, n - 2]   # 1d, 7d, 21d, 84d, unbounded

    frontier = []
    for tp_m in tp_grid:
        for sl_m in sl_grid:
            for hb in hold_grid:
                tp_bps, sl_bps = tp_m * rt_bps, sl_m * rt_bps
                if sl_bps <= tp_bps:
                    continue
                hb = int(min(hb, n - 2))
                r_all, i_all = sequential_book(h, l, c, rt, tp_bps, sl_bps,
                                               1, hb)
                if len(r_all) < 30:
                    continue
                m_ho = i_all >= cut
                if m_ho.sum() < 15:
                    continue
                ho = stats(r_all[m_ho])
                fu = stats(r_all)
                r2, _ = sequential_book(h, l, c, rt * 2, tp_bps, sl_bps, 1, hb)
                s2 = stats(r2)
                admissible = (
                    ho["win_rate"] >= TARGET_WR
                    and ho["ev_bps"] > 0
                    and fu["worst_trade_bps"] > TAIL_CAP_BPS
                    and fu["max_drawdown_pct"] > DD_CAP_PCT
                    and s2["ev_bps"] > 0
                )
                frontier.append({
                    "tp_bps": round(tp_bps, 1), "sl_bps": round(sl_bps, 1),
                    "tp_over_rt": tp_m, "sl_over_rt": sl_m,
                    "hold_bars": hb,
                    "hold_days": round(hb * 300 / 86400, 1),
                    "holdout": ho, "full": fu,
                    "ev_2x_friction_bps": s2["ev_bps"],
                    "admissible": bool(admissible),
                })
    if not frontier:
        return {"asset": asset, "status": "no_geometry"}
    adm = [f for f in frontier if f["admissible"]]
    best = (max(adm, key=lambda f: f["holdout"]["win_rate"]) if adm
            else max(frontier, key=lambda f: f["holdout"]["win_rate"]))
    # the frontier: best win rate at each worst-loss band
    bands = []
    for cap in (-500, -800, -1200, -1500, -2500, -5000, -100000):
        sub = [f for f in frontier
               if f["full"]["worst_trade_bps"] > cap
               and f["holdout"]["ev_bps"] > 0]
        if sub:
            top = max(sub, key=lambda f: f["holdout"]["win_rate"])
            bands.append({"tail_cap_bps": cap,
                          "best_wr": top["holdout"]["win_rate"],
                          "ev": top["holdout"]["ev_bps"],
                          "geom": f"TP {top['tp_over_rt']}xRT / "
                                  f"SL {top['sl_over_rt']}xRT / "
                                  f"{top['hold_days']}d"})
    return {
        "asset": asset, "status": "ok", "rt_bps": round(rt_bps, 2),
        "n_configs": len(frontier), "n_admissible": len(adm),
        "best": best, "frontier": frontier, "bands": bands,
        "caps": {"tail_bps": TAIL_CAP_BPS, "max_dd_pct": DD_CAP_PCT},
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
        b = r["best"]
        ho, fu = b["holdout"], b["full"]
        print(f"\n=== {asset}  RT {r['rt_bps']} bps  "
              f"({r['n_admissible']}/{r['n_configs']} admissible)")
        print(f"  best admissible geometry: TP {b['tp_bps']:.0f}b "
              f"({b['tp_over_rt']}x RT) · SL {b['sl_bps']:.0f}b "
              f"({b['sl_over_rt']}x RT) · max hold {b['hold_days']:.0f}d")
        print(f"  HOLDOUT {ho['n']:4d} trades  WR {ho['win_rate']*100:5.1f}% "
              f"[{ho['ci95'][0]*100:.0f},{ho['ci95'][1]*100:.0f}]  "
              f"EV {ho['ev_bps']:+7.2f} bp")
        print(f"  FULL    {fu['n']:4d} trades  WR {fu['win_rate']*100:5.1f}%  "
              f"EV {fu['ev_bps']:+7.2f} bp  worst {fu['worst_trade_bps']:+.0f} bp  "
              f"maxDD {fu['max_drawdown_pct']:.1f}%  "
              f"comp {fu['compounded_pct']:+.1f}%")
        print(f"  EV under 2x friction: {b['ev_2x_friction_bps']:+.2f} bp  "
              f"→ {'ADMISSIBLE' if b['admissible'] else 'NOT ADMISSIBLE'}")
        if r["bands"]:
            print(f"  frontier (best holdout WR subject to a tail cap, "
                  f"with holdout EV>0):")
            for band in r["bands"]:
                cap = band["tail_cap_bps"]
                cap_s = "none" if cap < -50000 else f"{cap/100:.0f}%"
                print(f"    worst trade > {cap_s:>5s}: WR "
                      f"{band['best_wr']*100:5.1f}%  EV {band['ev']:+7.2f} bp  "
                      f"[{band['geom']}]")
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    n_adm = sum(1 for r in out if r.get("n_admissible"))
    print(f"\nassets with at least one admissible geometry: {n_adm}/{len(out)}")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
