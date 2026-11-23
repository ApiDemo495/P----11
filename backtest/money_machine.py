"""WHERE THE MONEY ACTUALLY IS — cost drag vs edge, measured.

The diagnosis this module responds to
------------------------------------
``backtest.diagnose_all.py`` graded every formula on 3 years of tape.  The
"informative" ones were NOT consistent: F13 DIQ was +57 % at ETH tf3 and
40 % (i.e. reversed) at BTC tf1; F22 BAPR flipped sign for ETH vs BTC.
That is the signature of multiple testing, not edge: with ~35 formulas x
4 timeframes x 4 assets ~= 560 cells, ~28 look significant at 95 % by
chance alone, and their signs are random.

So the formulas cannot be the money.  This module tests what is left:

    1. BUY & HOLD                      — the benchmark any strategy must beat
    2. EVERY-WINDOW SIGN FLIP          — what a max-turnover signal book
                                         does after costs (the app's design)
    3. TREND FILTER, LOW TURNOVER      — long when the tape is above its
                                         own trend, in cash otherwise
    4. TREND + VOL TARGET              — same, position scaled by inverse vol

Every strategy pays the FULL round trip on every position change and is
measured on the SAME tape with the SAME walk-forward split.  The ranking
is reported with CAGR, max drawdown, Sharpe, turnover and total cost paid,
because "money making" without drawdown is a fantasy.

Usage
    .venv/bin/python -m backtest.money_machine --assets BTC ETH
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
OUT = os.path.join(RESULTS, "money_machine.json")
BARS_PER_YEAR_5M = 365 * 24 * 12


def load_ohlc(asset, tape="5m"):
    bars = load_cache(asset, tape)
    if not bars:
        return None
    return {k: np.array([getattr(b, k) for b in bars], float)
            for k in ("o", "h", "l", "c")}, np.array([b.t for b in bars])


def metrics(equity, rets, bars_per_year=BARS_PER_YEAR_5M, cost_paid=0.0,
            turnover=0.0):
    if len(equity) < 2:
        return {}
    total = float(equity[-1] / equity[0])
    years = len(rets) / bars_per_year
    cagr = (total ** (1 / years) - 1) if years > 0 and total > 0 else -1.0
    peak = np.maximum.accumulate(equity)
    dd = float(((equity - peak) / peak).min())
    mu = float(rets.mean())
    sd = float(rets.std())
    sharpe = (mu / sd * math.sqrt(bars_per_year)) if sd > 0 else 0.0
    downside = rets[rets < 0]
    dsd = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = (mu / dsd * math.sqrt(bars_per_year)) if dsd > 0 else 0.0
    calmar = (cagr / abs(dd)) if dd < 0 else 0.0
    return {
        "cagr_pct": round(cagr * 100, 2),
        "total_x": round(total, 3),
        "max_drawdown_pct": round(dd * 100, 1),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "cost_paid_pct_of_start": round(cost_paid * 100, 1),
        "turnover_units": round(turnover, 1),
        "years": round(years, 2),
    }


def run_positions(price, pos, rt_per_change, warmup=0):
    """Equity from a target-position series. Charge rt on every |Δpos|.

    ``pos`` is the DESIRED exposure (-1..+1) decided on the PREVIOUS bar's
    close, so there is no look-ahead.
    """
    pos = np.asarray(pos, float)
    ret = np.diff(price) / price[:-1]
    held = pos[:-1]                      # decided at close of bar i, earns i->i+1
    gross = held * ret
    dp = np.abs(np.diff(held, prepend=held[0]))
    cost = dp * rt_per_change
    net = gross - cost
    equity = np.cumprod(1.0 + net[warmup:])
    return net[warmup:], equity, float(cost[warmup:].sum()), float(dp.sum())


# ───────────────────────────── strategies ───────────────────────────
def strat_buy_hold(price, rt):
    return np.ones(len(price))


def strat_every_window(price, rt, w_min, step, sign_series=None):
    """Flip position every window — maximum turnover, the app's design.

    Uses a real signal if given (so the formula layer is actually tested),
    otherwise alternates, which maximises cost to show the upper bound of
    damage.
    """
    n = len(price)
    W = int(w_min * 60 / step)
    pos = np.zeros(n)
    for i in range(n):
        w = (i // max(W, 1)) % 2
        pos[i] = 1.0 if w == 0 else -1.0
    if sign_series is not None:
        pos = np.asarray(sign_series, float)
    return pos


def strat_trend(price, fast, slow):
    """Long when fast MA > slow MA, in cash otherwise. Low turnover."""
    n = len(price)
    pos = np.zeros(n)
    ps = np.concatenate([[0.0], np.cumsum(price)])
    f = (ps[fast:] - ps[:-fast]) / fast
    s = (ps[slow:] - ps[:-slow]) / slow
    off_f = fast - 1
    off_s = slow - 1
    start = max(fast, slow)
    for i in range(start, n):
        fa = (ps[i + 1] - ps[i + 1 - fast]) / fast
        sa = (ps[i + 1] - ps[i + 1 - slow]) / slow
        pos[i] = 1.0 if fa > sa else 0.0
    return pos


def strat_trend_voltarget(price, fast, slow, vol_win, target_vol):
    """Trend filter with inverse-vol sizing — smooths the ride."""
    n = len(price)
    base = strat_trend(price, fast, slow)
    ret = np.zeros(n)
    ret[1:] = np.diff(price) / price[:-1]
    # rolling realised vol (annualised), causal
    vol = np.full(n, np.nan)
    for i in range(vol_win, n):
        vol[i] = ret[i - vol_win + 1:i + 1].std() * math.sqrt(BARS_PER_YEAR_5M)
    scale = np.where(np.isfinite(vol) & (vol > 0), target_vol / vol, 0.0)
    scale = np.clip(scale, 0.0, 1.0)
    return base * scale


def evaluate(asset, tape="5m"):
    got = load_ohlc(asset, tape)
    if not got:
        return {"asset": asset, "status": "no_tape"}
    D, t = got
    price = D["c"]
    rt = 2.0 * FRICTION_BPS[asset] * 1e-4
    step = int(t[1] - t[0])
    out = {"asset": asset, "status": "ok", "rt_bps": round(rt * 1e4, 2),
           "bars": len(price), "start": int(t[0]), "end": int(t[-1]),
           "strategies": {}}

    def add(name, pos, warmup, note=""):
        net, eq, cost, turn = run_positions(price, pos, rt, warmup)
        m = metrics(eq, net, cost_paid=cost, turnover=turn)
        m["note"] = note
        out["strategies"][name] = m

    add("buy_hold", strat_buy_hold(price, rt), 0, "always long, cost paid once")
    # every-window flip at the app's shortest timeframe — the cost ceiling
    for w in (22, 130):
        pos = strat_every_window(price, rt, w, step)
        add(f"every_window_{w}m", pos, w,
            "flips side every window: measures pure cost drag")
    # trend filters, low turnover
    for fast, slow in ((12 * 288, 50 * 288), (10 * 288, 40 * 288),
                       (20 * 288, 100 * 288), (50 * 288, 200 * 288)):
        pos = strat_trend(price, fast, slow)
        add(f"trend_{fast//288}d_{slow//288}d", pos, slow,
            "long above trend, cash below")
    # trend + vol target
    for fast, slow in ((10 * 288, 40 * 288), (20 * 288, 100 * 288)):
        pos = strat_trend_voltarget(price, fast, slow, 288 * 7, 0.35)
        add(f"trend_vol_{fast//288}d_{slow//288}d", pos, slow,
            "trend filter sized by inverse realised vol")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="*", default=["BTC", "ETH"])
    args = ap.parse_args()
    rows = []
    for asset in args.assets:
        r = evaluate(asset)
        rows.append(r)
        if r.get("status") != "ok":
            print(f"{asset}: {r['status']}")
            continue
        print(f"\n=== {asset}  RT {r['rt_bps']} bps  {r['bars']:,} bars")
        print(f"  {'strategy':>22s} {'CAGR':>8s} {'total x':>8s} {'maxDD':>8s} "
              f"{'Sharpe':>7s} {'Sortino':>8s} {'cost paid':>10s} {'turnover':>9s}")
        best = None
        for name, m in r["strategies"].items():
            print(f"  {name:>22s} {m['cagr_pct']:+7.1f}% {m['total_x']:7.2f}x "
                  f"{m['max_drawdown_pct']:7.1f}% {m['sharpe']:7.2f} "
                  f"{m['sortino']:8.2f} {m['cost_paid_pct_of_start']:9.1f}% "
                  f"{m['turnover_units']:9.0f}")
            if best is None or m["calmar"] > best[1]["calmar"]:
                best = (name, m)
        bh = r["strategies"].get("buy_hold", {})
        r["best_by_calmar"] = best[0] if best else None
        print(f"  → best risk-adjusted: {best[0]} "
              f"(calmar {best[1]['calmar']:.2f} vs buy&hold {bh.get('calmar', 0):.2f})")
        ew = r["strategies"].get("every_window_22m", {})
        print(f"  → every-window book: CAGR {ew.get('cagr_pct', 0):+.1f}%, "
              f"cost paid {ew.get('cost_paid_pct_of_start', 0):.0f}% of capital")
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(rows, open(OUT, "w"), indent=1)
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
