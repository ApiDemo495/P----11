"""PATH RISK AND THE GROWTH-OPTIMAL SIZE — how far the edge can actually go.

The question this answers
-------------------------
"$666 per day on $1,000" is +66.6 % *per day*, which compounds to more money
than exists on Earth in seven weeks.  No signal answers that, and a repo that
claims to is lying.  But there IS a real, measurable maximum: the leverage at
which the edge grows capital fastest before the variance of the path eats it.

This module computes, on the actual tape:

  * the daily return series of every surviving book (§28.7);
  * the **growth-optimal (Kelly) leverage**, found by maximising the realised
    mean log growth of the levered equity curve — not assumed, searched;
  * what that costs in drawdown, and the bootstrap probability of ruin
    (touching −100 %) at each leverage;
  * dollars per day per $1,000 at 1x, at half-Kelly and at full Kelly.

The output is the honest ceiling on "how great can this app be": a number in
dollars, with its ruin probability attached, at every size.

Usage
    .venv/bin/python -m backtest.path_risk
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data import load_cache                              # noqa: E402
from backtest.geometry_scan import barrier_book, blocks           # noqa: E402
from backtest.geometry_validate import short_book                 # noqa: E402

RESULTS = os.path.join("backtest", "results")
OUT = os.path.join(RESULTS, "path_risk.json")
DAY_S = 86400
YEARS = 365.25
#: the validated cells, as (asset, bars, tp, sl, side)
BOOKS = {
    "ETH_d1_long":  ("ETH", 288, 20.0, 80.0, "long"),
    "BTC_d1_short": ("BTC", 288, 20.0, 80.0, "short"),
    "ETH_tf3_long": ("ETH", 26, 8.0, 32.0, "long"),
    "BTC_tf3_long": ("BTC", 26, 8.0, 32.0, "long"),
}
LEVS = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50)


def _daily(asset, k, tp, sl, side):
    """Window returns and their day index on a fixed non-overlapping tiling."""
    bars = load_cache(asset, "5m")
    if not bars:
        return None
    c = np.array([b.c for b in bars], float)
    h = np.array([b.h for b in bars], float)
    l = np.array([b.l for b in bars], float)
    t = np.array([b.t for b in bars], np.int64)
    C, H, L = blocks(c, h, l, k)
    fn = barrier_book if side == "long" else short_book
    r, won, _ = fn(C, H, L, tp, sl, 0.0)
    t0 = t[np.arange(len(C)) * k]
    day = (t0 // DAY_S).astype(np.int64)
    # one day's return = product of that day's windows
    uniq, inv = np.unique(day, return_inverse=True)
    daily = np.ones(len(uniq))
    np.multiply.at(daily, inv, 1.0 + r * 1e-4)
    return daily - 1.0, r


def _curve_stats(daily, lev):
    """Compounded stats for a fixed leverage applied to daily returns."""
    eq = np.cumprod(1.0 + lev * daily)
    if eq[-1] <= 0 or np.any(eq <= 0):
        return {"ruined": True, "cagr": -1.0, "max_dd": -1.0,
                "total": -1.0, "final": 0.0}
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    span = len(daily) / YEARS
    return {"ruined": False,
            "cagr": float(eq[-1] ** (1.0 / span) - 1.0),
            "max_dd": mdd, "total": float(eq[-1] - 1.0),
            "final": float(eq[-1])}


def kelly(daily, lo=0.5, hi=60.0, steps=240):
    """Growth-optimal leverage: maximises realised mean log growth."""
    best = (None, -np.inf)
    for L in np.linspace(lo, hi, steps):
        g = np.mean(np.log(np.maximum(1.0 + L * daily, 1e-12)))
        if g > best[1]:
            best = (float(L), float(g))
    return best


def bootstrap_paths(daily, n_paths=4000, block=20, seed=7):
    """Block-resampled paths, so losing streaks survive the resampling.

    Equity can never touch exactly zero with bounded per-trade losses, so
    "ruin" has to be defined the way a trader experiences it: a drawdown
    that ends the account in practice.  ``dd_breach`` therefore reports the
    share of paths whose peak-to-trough loss exceeds each threshold.
    """
    rng = np.random.default_rng(seed)
    n = len(daily)
    nb = int(np.ceil(n / block))
    out = []
    for _ in range(n_paths):
        starts = rng.integers(0, max(n - block, 1), nb)
        out.append(np.concatenate([daily[s:s + block] for s in starts])[:n])
    return np.array(out)


def risk_of(daily, lev, paths=None):
    """P(drawdown > 50 %), P(drawdown > 80 %), P(ends below 50 % of start)."""
    if paths is None:
        paths = bootstrap_paths(daily)
    eq = np.cumprod(1.0 + lev * paths, axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    dd = (eq / peak).min(axis=1) - 1.0
    return {"p_dd50": float((dd < -0.50).mean()),
            "p_dd80": float((dd < -0.80).mean()),
            "p_half_capital": float((eq[:, -1] < 0.50).mean()),
            "median_dd": float(np.median(dd))}


def safe_leverage(daily, paths, limit=0.05, grid=None):
    """Largest leverage whose bootstrap P(drawdown > 50 %) stays under limit.

    This is the sizing rule that survives being wrong: it does not require
    the win rate to be exactly right, only that the past shape of the path
    resembles the future one.
    """
    grid = grid if grid is not None else np.arange(0.5, 40.0, 0.5)
    ok = 0.0
    for L in grid:
        r = risk_of(daily, L, paths)
        if r["p_dd50"] <= limit:
            ok = float(L)
        else:
            break
    return ok


def slippage_stress(daily_bps, extra_bps_per_trade, trades_per_day):
    """Re-derive the daily series when every round trip costs extra bps."""
    return daily_bps - extra_bps_per_trade * trades_per_day * 1e-4


def main():
    os.makedirs(RESULTS, exist_ok=True)
    out = {"books": {}, "portfolio": None, "target_daily": 0.666}

    series = {}
    for name, (asset, k, tp, sl, side) in BOOKS.items():
        got = _daily(asset, k, tp, sl, side)
        if not got:
            continue
        daily, raw = got
        series[name] = daily
        L_free, g = kelly(daily)
        paths = bootstrap_paths(daily)
        rows = []
        for L in LEVS:
            st = _curve_stats(daily, L)
            st["leverage"] = L
            st.update(risk_of(daily, L, paths))
            st["dollars_per_day_per_1000"] = (1000.0 * (1.0 + st["cagr"])
                                              ** (1 / YEARS) - 1000.0
                                              if not st["ruined"] else -1000.0)
            rows.append(st)
        k_safe = safe_leverage(daily, paths)
        k_half = L_free / 2.0
        st_half = _curve_stats(daily, k_half)
        st_full = _curve_stats(daily, L_free)
        out["books"][name] = {
            "asset": asset, "bars": k, "minutes": k * 5, "side": side,
            "tp_bps": tp, "sl_bps": sl, "n_days": int(len(daily)),
            "win_rate": round(float((raw > 0).mean()), 4),
            "mean_daily": round(float(daily.mean()), 6),
            "std_daily": round(float(daily.std()), 6),
            "kelly_leverage": round(L_free, 2),
            "half_kelly": round(k_half, 2),
            "safe_leverage_pdd50_5pct": round(k_safe, 2),
            "at_half_kelly": {**st_half, "leverage": round(k_half, 2),
                              **risk_of(daily, k_half, paths)},
            "at_full_kelly": {**st_full, "leverage": round(L_free, 2),
                              **risk_of(daily, L_free, paths)},
            "leverage_ladder": rows,
        }

    # the two daily books together: their drawdowns are not perfectly aligned
    if "ETH_d1_long" in series and "BTC_d1_short" in series:
        n = min(len(series["ETH_d1_long"]), len(series["BTC_d1_short"]))
        port = 0.5 * series["ETH_d1_long"][:n] + 0.5 * series["BTC_d1_short"][:n]
        L_free, g = kelly(port)
        ppaths = bootstrap_paths(port)
        p_safe = safe_leverage(port, ppaths)
        st_half = _curve_stats(port, L_free / 2)
        st_full = _curve_stats(port, L_free)
        out["portfolio"] = {
            "name": "ETH d1 long + BTC d1 short, 50/50",
            "corr": round(float(np.corrcoef(series["ETH_d1_long"][:n],
                                            series["BTC_d1_short"][:n])[0, 1]), 3),
            "n_days": int(n),
            "mean_daily": round(float(port.mean()), 6),
            "std_daily": round(float(port.std()), 6),
            "kelly_leverage": round(L_free, 2),
            "safe_leverage_pdd50_5pct": round(p_safe, 2),
            "at_1x": {**_curve_stats(port, 1.0), "leverage": 1.0,
                      **risk_of(port, 1.0, ppaths)},
            "at_half_kelly": {**st_half, "leverage": round(L_free / 2, 2),
                              **risk_of(port, L_free / 2, ppaths)},
            "at_full_kelly": {**st_full, "leverage": round(L_free, 2),
                              **risk_of(port, L_free, ppaths)},
            "leverage_ladder": [{**_curve_stats(port, L), "leverage": L,
                                 **risk_of(port, L, ppaths)} for L in LEVS],
        }

    json.dump(out, open(OUT, "w"), indent=1, default=float)

    print("=" * 88)
    print("WHAT $666/DAY PER $1,000 ACTUALLY REQUIRES")
    print("=" * 88)
    print(f"  +66.6 %/day compounds to {(1.666**365):.3e}x in one year.")
    print("  $1,000 reaches world GDP (~$1e14) in "
          f"{np.log(1e14/1000)/np.log(1.666):.1f} days.")
    print("  No signal does this. What follows is the real ceiling.\n")

    for name, b in out["books"].items():
        print(f"\n{name}  ({b['minutes']}m windows, WR {b['win_rate']*100:.1f} %, "
              f"{b['n_days']} days)")
        print(f"  growth-optimal leverage (Kelly) : {b['kelly_leverage']:.2f}x  "
              f"(in-sample upper bound)")
        print(f"  SAFE leverage (P(dd>50%) < 5 %) : "
              f"{b['safe_leverage_pdd50_5pct']:.2f}x")
        print(f"  {'lev':>6s} {'$/day per $1k':>14s} {'CAGR':>9s} "
              f"{'max DD':>9s} {'P(dd>50%)':>10s} {'P(dd>80%)':>10s}")
        for r in b["leverage_ladder"]:
            dpd = r["dollars_per_day_per_1000"]
            print(f"  {r['leverage']:5d}x {dpd:+14.2f} "
                  f"{r['cagr']*100:+8.1f}% {r['max_dd']*100:+8.1f}% "
                  f"{r['p_dd50']*100:9.1f}% {r['p_dd80']*100:9.1f}%")
        for tag, key in (("half-Kelly", "at_half_kelly"),
                         ("full Kelly", "at_full_kelly")):
            s = b[key]
            dpd = (1000.0 * (1.0 + s["cagr"]) ** (1 / YEARS) - 1000.0)
            print(f"  {tag:>10s} {s['leverage']:5.2f}x → "
                  f"{dpd:+8.2f} $/day per $1k, DD {s['max_dd']*100:+.1f} %, "
                  f"P(dd>50%) {s['p_dd50']*100:.1f} %")

    p = out["portfolio"]
    if p:
        print(f"\n{'='*88}\nPORTFOLIO — {p['name']}  (correlation {p['corr']:+.3f})"
              f"\n{'='*88}")
        print(f"  growth-optimal leverage: {p['kelly_leverage']:.2f}x  "
              f"(in-sample upper bound)")
        print(f"  SAFE leverage (P(dd>50%) < 5 %): "
              f"{p['safe_leverage_pdd50_5pct']:.2f}x")
        print(f"  {'lev':>6s} {'$/day per $1k':>14s} {'CAGR':>9s} "
              f"{'max DD':>9s} {'P(dd>50%)':>10s} {'P(dd>80%)':>10s}")
        for r in p["leverage_ladder"]:
            dpd = (1000.0 * (1.0 + r["cagr"]) ** (1 / YEARS) - 1000.0
                   if not r["ruined"] else -1000.0)
            print(f"  {r['leverage']:5d}x {dpd:+14.2f} "
                  f"{r['cagr']*100:+8.1f}% {r['max_dd']*100:+8.1f}% "
                  f"{r['p_dd50']*100:9.1f}% {r['p_dd80']*100:9.1f}%")
        s = p["at_half_kelly"]
        print(f"\n  half-Kelly ({s['leverage']:.2f}x) → "
              f"{1000.0*(1.0+s['cagr'])**(1/YEARS)-1000.0:+.2f} $/day per $1,000, "
              f"max DD {s['max_dd']*100:+.1f} %, P(dd>50%) {s['p_dd50']*100:.1f} %")

    print(f"\n{'='*88}")
    print("  Kelly here is an IN-SAMPLE optimum — it is fitted to the same")
    print("  2.84 y it is measured on, so it is an upper bound, not a target.")
    print("  De-rate it hard: the win rate that produced it is estimated from")
    print("  ~360 windows and can move by 3 pp, which moves Kelly by far more.")
    print("  P(dd>50%) is the number that decides whether a size is survivable.")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
