"""Walk-forward validation of the low-turnover trend book.

``backtest/money_machine.py`` produced the important number in the whole
project: the app's design (flip side every window) makes **149,038 position
changes** over three years and pays **~43,000 % of starting capital** in
costs, which is total ruin.  A simple 10-day/40-day trend filter made
+46 % CAGR on ETH with **31** changes.

That comparison is one parameter set, so it proves nothing yet.  This
module does it properly:

  * a grid of (fast, slow) trend pairs, chosen on the FIRST 65 % only;
  * the chosen pair applied unchanged to the LAST 35 %;
  * the same test on a second tape (ETH) to see if the choice transfers;
  * cost sensitivity at 1x, 2x, 3x friction;
  * a turnover budget, so the winner cannot be a lucky parameter that
    happens to trade 400 times.

The verdict rule is fixed before the run: a book is **validated** only if
it beats buy-and-hold on Calmar in the held-out period AND stays
profitable under doubled friction.

Usage
    .venv/bin/python -m backtest.trend_validate --assets BTC ETH
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data import FRICTION_BPS, load_cache          # noqa: E402
from backtest.money_machine import (BARS_PER_YEAR_5M, load_ohlc,  # noqa: E402
                                    metrics, run_positions, strat_trend,
                                    strat_trend_voltarget)

RESULTS = os.path.join("backtest", "results")
OUT = os.path.join(RESULTS, "trend_validate.json")
TRAIN_FRAC = 0.65
#: 5m bars per day
BPD = 288


def evaluate_window(price, pos, rt, warmup):
    net, eq, cost, turn = run_positions(price, pos, rt, warmup)
    if len(net) < 100:
        return None
    return metrics(eq, net, cost_paid=cost, turnover=turn)


def run(asset):
    got = load_ohlc(asset, "5m")
    if not got:
        return {"asset": asset, "status": "no_tape"}
    D, t = got
    price = D["c"]
    rt = 2.0 * FRICTION_BPS[asset] * 1e-4
    n = len(price)
    cut = int(n * TRAIN_FRAC)

    grid = [(f, s) for f in (5, 10, 20, 40, 50) for s in (20, 40, 100, 200)
            if s > f]
    train = []
    for fast_d, slow_d in grid:
        pos = strat_trend(price, fast_d * BPD, slow_d * BPD)
        m = evaluate_window(price[:cut], pos[:cut], rt, slow_d * BPD)
        if m:
            m["fast_d"], m["slow_d"] = fast_d, slow_d
            train.append(m)
    if not train:
        return {"asset": asset, "status": "no_grid"}
    # choose on train Calmar, with a turnover sanity cap
    elig = [m for m in train if m["turnover_units"] <= 60]
    best = max(elig or train, key=lambda m: m["calmar"])

    # apply unchanged to the holdout
    pos_all = strat_trend(price, best["fast_d"] * BPD, best["slow_d"] * BPD)
    warm = best["slow_d"] * BPD
    net_all, eq_all, cost_all, turn_all = run_positions(price, pos_all, rt, warm)
    # split the SAME position series at the cut (index aligns after warmup)
    ho_start = max(0, cut - warm)
    net_ho = net_all[ho_start:]
    eq_ho = np.cumprod(1.0 + net_ho)
    ho = metrics(eq_ho, net_ho,
                 cost_paid=float(np.abs(np.diff(
                     pos_all[cut:], prepend=pos_all[cut])).sum() * rt),
                 turnover=float(np.abs(np.diff(
                     pos_all[cut:], prepend=pos_all[cut])).sum()))
    bh_pos = np.ones(n)
    _, bh_eq, _, _ = run_positions(price, bh_pos, rt, 0)
    ho_bh_net = np.diff(price[cut:]) / price[cut:-1]
    bh_ho = metrics(np.cumprod(1 + ho_bh_net), ho_bh_net)

    # full-period + cost stress
    stress = {}
    for mult in (1.0, 2.0, 3.0):
        net_m, eq_m, cost_m, turn_m = run_positions(
            price, pos_all, rt * mult, warm)
        m = metrics(eq_m, net_m, cost_paid=cost_m, turnover=turn_m)
        stress[f"{mult:g}x"] = {"cagr_pct": m["cagr_pct"],
                                "max_drawdown_pct": m["max_drawdown_pct"],
                                "calmar": m["calmar"]}

    checks = {
        "holdout_calmar_beats_buyhold": bool(ho["calmar"] > bh_ho["calmar"]),
        "holdout_cagr_positive": bool(ho["cagr_pct"] > 0),
        "survives_2x_cost": bool(stress["2x"]["cagr_pct"] > 0),
        "turnover_sane": bool(best["turnover_units"] <= 60),
    }
    return {
        "asset": asset, "status": "ok", "rt_bps": round(rt * 1e4, 2),
        "chosen": {"fast_d": best["fast_d"], "slow_d": best["slow_d"],
                   "train_calmar": best["calmar"],
                   "train_cagr": best["cagr_pct"],
                   "train_turnover": best["turnover_units"]},
        "holdout": ho, "holdout_buy_hold": bh_ho,
        "full": metrics(eq_all, net_all, cost_paid=cost_all,
                        turnover=turn_all),
        "cost_stress": stress, "checks": checks,
        "validated": all(checks.values()),
        "reasons_failed": [k for k, v in checks.items() if not v],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="*", default=["BTC", "ETH"])
    args = ap.parse_args()
    rows = []
    for asset in args.assets:
        r = run(asset)
        rows.append(r)
        if r.get("status") != "ok":
            print(f"{asset}: {r['status']}")
            continue
        c, ho, bh = r["chosen"], r["holdout"], r["holdout_buy_hold"]
        fu = r["full"]
        print(f"\n=== {asset}  RT {r['rt_bps']} bps")
        print(f"  chosen on train only: {c['fast_d']}d/{c['slow_d']}d  "
              f"(train CAGR {c['train_cagr']:+.1f}%, calmar {c['train_calmar']:.2f}, "
              f"{c['train_turnover']:.0f} changes)")
        print(f"  HOLDOUT  CAGR {ho['cagr_pct']:+7.1f}%  maxDD {ho['max_drawdown_pct']:6.1f}%  "
              f"calmar {ho['calmar']:5.2f}  sharpe {ho['sharpe']:5.2f}")
        print(f"  buy&hold CAGR {bh['cagr_pct']:+7.1f}%  maxDD {bh['max_drawdown_pct']:6.1f}%  "
              f"calmar {bh['calmar']:5.2f}  sharpe {bh['sharpe']:5.2f}")
        print(f"  FULL     CAGR {fu['cagr_pct']:+7.1f}%  maxDD {fu['max_drawdown_pct']:6.1f}%  "
              f"total {fu['total_x']:.2f}x  cost paid {fu['cost_paid_pct_of_start']:.1f}%")
        for k, s in r["cost_stress"].items():
            print(f"    {k:>3s}: CAGR {s['cagr_pct']:+7.1f}%  "
                  f"maxDD {s['max_drawdown_pct']:6.1f}%  calmar {s['calmar']:5.2f}")
        print(f"  checks: {r['checks']}")
        print(f"  → {'VALIDATED' if r['validated'] else 'NOT VALIDATED'}"
              + ("" if r["validated"] else f"  failed {r['reasons_failed']}"))
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(rows, open(OUT, "w"), indent=1)
    n = sum(1 for r in rows if r.get("validated"))
    print(f"\n{n}/{len(rows)} assets VALIDATED out-of-sample, beating buy&hold "
          f"on Calmar and surviving 2x friction")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
