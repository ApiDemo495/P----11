"""THE SURVIVING BOOK, IN CAPITAL — CAGR, drawdown, and per-year honesty.

``geometry_validate`` leaves 19 candidate cells that clear break-even on the
holdout under every tie rule.  A win rate is not an account balance, so this
module runs the two that matter at full size and reports:

    * compounding equity, 1x notional, one position at a time
    * CAGR and maximum drawdown
    * the SAME numbers year by year, so a cell that only worked in the 2024
      melt-up is visible as such
    * the execution budget: how many bps of fee the edge can absorb before
      it turns negative (the whole result hangs on this number)

Usage
    .venv/bin/python -m backtest.venue_book
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data import load_cache                              # noqa: E402
from backtest.geometry_scan import blocks, dd_bps                 # noqa: E402
from backtest.geometry_validate import short_book, wilson         # noqa: E402
from backtest.geometry_scan import barrier_book                   # noqa: E402

RESULTS = os.path.join("backtest", "results")
OUT = os.path.join(RESULTS, "venue_book.json")
TRAIN_FRAC = 0.65
YEAR_S = 365 * 24 * 3600
#: (asset, tf, bars, tp, sl, side) — the cells §geometry_validate kept
BOOKS = (("BTC", "tf3", 26, 8.0, 32.0, "long"),
         ("BTC", "tf4", 49, 8.0, 32.0, "long"),
         ("BTC", "1d", 288, 20.0, 80.0, "short"),
         ("ETH", "tf3", 26, 8.0, 32.0, "long"),
         ("ETH", "1d", 288, 20.0, 80.0, "long"))


def book_metrics(asset, tf, k, tp, sl, side):
    bars = load_cache(asset, "5m")
    if not bars:
        return None
    c = np.array([b.c for b in bars], float)
    h = np.array([b.h for b in bars], float)
    l = np.array([b.l for b in bars], float)
    t = np.array([b.t for b in bars], np.int64)
    C, H, L = blocks(c, h, l, k)
    n = len(C)
    fn = barrier_book if side == "long" else short_book
    # rt=0 is the Lighter case; fee sensitivity is reported separately
    r, won, ties = fn(C, H, L, tp, sl, 0.0)
    # window start times, in the same tiling as blocks()
    t0 = t[np.arange(n) * k]
    cut = int(n * TRAIN_FRAC)

    def stats(seg_r, seg_w):
        if len(seg_r) < 5:
            return None
        eq = np.cumprod(1.0 + seg_r * 1e-4)
        span = (len(seg_r) * k * 300) / YEAR_S
        cagr = eq[-1] ** (1.0 / span) - 1.0 if span > 0 and eq[-1] > 0 else -1.0
        peak = np.maximum.accumulate(eq)
        mdd = float((eq / peak - 1.0).min())
        return {"n": int(len(seg_r)),
                "wr": round(float(seg_w.mean()), 4),
                "ev_bps": round(float(seg_r.mean()), 3),
                "total_ret": round(float(eq[-1] - 1.0), 4),
                "cagr": round(float(cagr), 4),
                "max_dd": round(mdd, 4),
                "worst_bps": round(float(seg_r.min()), 1),
                "sharpe": round(float(seg_r.mean() / seg_r.std(ddof=1)
                                       * np.sqrt(252 * 24 * 60 / (k * 5))), 2)
                if seg_r.std() > 0 else None}

    # how much fee can this edge absorb, in bps of round trip?
    # EV = wr*tp - (1-wr)*sl - fee  ->  fee_budget = wr*tp - (1-wr)*sl
    wr_all = float(won.mean())
    fee_budget = wr_all * tp - (1 - wr_all) * sl

    per_year = []
    for y in sorted({int(x) for x in (t0 // YEAR_S) }):
        pass
    years = {}
    for i, ts in enumerate(t0):
        years.setdefault(__import__("datetime").datetime.utcfromtimestamp(
            int(ts)).year, []).append(i)
    for y in sorted(years):
        idx = np.array(years[y])
        st = stats(r[idx], won[idx])
        if st:
            st["year"] = y
            per_year.append(st)

    return {"asset": asset, "tf": tf, "minutes": k * 5, "side": side,
            "tp_bps": tp, "sl_bps": sl, "venue": "lighter (0 bps)",
            "n": n, "tie_rate": round(float(ties.mean()), 4),
            "fee_budget_bps": round(float(fee_budget), 2),
            "full": stats(r, won),
            "train": stats(r[:cut], won[:cut]),
            "holdout": stats(r[cut:], won[cut:]),
            "per_year": per_year,
            "ci95": [round(x, 4) for x in wilson(int(won.sum()), n)]}


def main():
    res = [b for b in (book_metrics(*x) for x in BOOKS) if b]
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)

    for b in res:
        print(f"\n{'='*84}\n{b['asset']} {b['tf']} ({b['minutes']}m) "
              f"{b['side']} tp{b['tp_bps']:g}/sl{b['sl_bps']:g} @ 0 bps\n{'='*84}")
        f, tr, ho = b["full"], b["train"], b["holdout"]
        print(f"  windows {b['n']}  ties {b['tie_rate']*100:.1f} %  "
              f"WR {f['wr']*100:.1f} % CI [{b['ci95'][0]*100:.1f},"
              f"{b['ci95'][1]*100:.1f}]")
        print(f"  full    : CAGR {f['cagr']*100:+7.1f} %  DD {f['max_dd']*100:+6.1f} %"
              f"  Sharpe {f['sharpe']}  EV {f['ev_bps']:+.2f} bps")
        print(f"  train   : CAGR {tr['cagr']*100:+7.1f} %  DD {tr['max_dd']*100:+6.1f} %"
              f"  WR {tr['wr']*100:.1f} %")
        print(f"  holdout : CAGR {ho['cagr']*100:+7.1f} %  DD {ho['max_dd']*100:+6.1f} %"
              f"  WR {ho['wr']*100:.1f} %  EV {ho['ev_bps']:+.2f} bps")
        print(f"  fee budget: {b['fee_budget_bps']:+.2f} bps round trip — "
              f"a 4 bps venue leaves "
              f"{b['fee_budget_bps']-4:+.2f} bps")
        print(f"  {'year':>6s} {'n':>5s} {'WR':>7s} {'EV bps':>8s} "
              f"{'return':>9s} {'DD':>8s}")
        for y in b["per_year"]:
            print(f"  {y['year']:6d} {y['n']:5d} {y['wr']*100:6.1f}% "
                  f"{y['ev_bps']:+8.2f} {y['total_ret']*100:+8.1f}% "
                  f"{y['max_dd']*100:+7.1f}%")

    print(f"\n{'='*84}\nSUMMARY — the honest money-machine answer\n{'='*84}")
    for b in res:
        ho = b["holdout"]
        print(f"  {b['asset']} {b['tf']:>3s} ({b['minutes']:4d}m) {b['side']:5s} "
              f"WR {ho['wr']*100:5.1f} %  holdout CAGR {ho['cagr']*100:+6.1f} %  "
              f"DD {ho['max_dd']*100:+6.1f} %  fee budget "
              f"{b['fee_budget_bps']:+5.2f} bps")
    print(f"\nNOTE: 1x notional, one position at a time, no leverage, no "
          f"compounding assumptions beyond the equity curve.  The 'fee budget' "
          f"column is the entire risk: these books are profitable ONLY while "
          f"the round trip stays under it.  A taker fee of 0.028 % (Lighter "
          f"Premium) is 5.6 bps and wipes out most of them.")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
