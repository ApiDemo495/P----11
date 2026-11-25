"""Breadth test: does a wider cross-section actually raise Sharpe?

The one lever §32 left standing was *breadth*, not leverage: a market-neutral
book over 25 assets should diversify more than one over 4. This module
measures whether that is true on the real tape, with no fitting.

Construction (all pre-registered, no search):

  * universe      : every symbol whose tape starts 2024-01-01, aligned on
                    the common hourly grid
  * signal        : trailing return over LOOKBACK hours
  * book          : dollar-neutral, long the top `legs`, short the bottom
                    `legs`, equal weight; rebalanced every REBALANCE hours
  * no abstention : every leg is always invested
  * fees          : each leg pays its own Binance round trip on turnover
  * test          : parameters are literals; the last 35 % is walked once

The question is monotone: as `legs` grows (4 -> 6 -> 8 -> 12), Sharpe should
rise if breadth is the real lever, and flatten if it is not.

Run:  .venv/bin/python -m backtest.breadth_test
"""
from __future__ import annotations

import math
import os

import numpy as np

from .beat_funds import FEE_BPS, HOURS_PER_YEAR, TRAIN_FRAC, metrics, show
from .target_66_real import DATA, load

#: symbols whose 1h tape starts 2024-01-01 (the common window)
LOOKBACK = 168          # 1 week of hourly bars
REBALANCE = 168         # weekly
LEG_GRID = (2, 3, 4, 6, 8, 12)
UNIVERSE_FILE = os.path.join(os.path.dirname(__file__), "results",
                             "breadth_universe.txt")


def load_universe() -> dict:
    tapes = {}
    for name in sorted(os.listdir(DATA)):
        d = os.path.join(DATA, name)
        if not os.path.isdir(d):
            continue
        files = sorted(os.listdir(os.path.join(d, name + "-1h-2024-01-01")))
        if not files:
            continue
        try:
            b = load(name)
        except AssertionError:
            continue
        if len(b["c"]) < 20000:
            continue
        tapes[name] = b
    return tapes


def align_closes(tapes: dict) -> tuple[list[str], np.ndarray]:
    """Common timestamp grid; returns (symbols, closes[sym, t])."""
    common = None
    for b in tapes.values():
        s = set(int(x) for x in b["t"])
        common = s if common is None else (common & s)
    grid = sorted(common or ())
    gset = set(grid)
    syms, rows = [], []
    for name, b in sorted(tapes.items()):
        idx = [i for i, t in enumerate(b["t"]) if int(t) in gset]
        syms.append(name)
        rows.append(np.asarray(b["c"])[idx])
    return syms, np.vstack(rows)


def xs_book(closes: np.ndarray, fees: np.ndarray, legs: int,
            lookback: int = LOOKBACK, rebalance: int = REBALANCE) -> np.ndarray:
    """Dollar-neutral cross-sectional momentum, always invested."""
    n_sym, t = closes.shape
    rets = np.zeros_like(closes)
    rets[:, 1:] = closes[:, 1:] / closes[:, :-1] - 1.0
    pos = np.zeros((n_sym, t))
    prev = np.zeros(n_sym)
    for i in range(lookback + 1, t):
        if (i - lookback) % rebalance == 0:
            past = closes[:, i] / closes[:, i - lookback] - 1.0
            order = np.argsort(past)
            new = np.zeros(n_sym)
            new[order[-legs:]] = 1.0 / legs
            new[order[:legs]] = -1.0 / legs
            prev = new
        pos[:, i] = prev
    pnl = np.zeros(t)
    pnl[1:] = np.sum(pos[:, :-1] * rets[:, 1:], axis=0)
    dpos = np.abs(np.diff(pos, axis=1))
    for k in range(n_sym):
        pnl[1:] -= dpos[k] * fees[k]
    return pnl


def main() -> None:
    tapes = load_universe()
    syms, closes = align_closes(tapes)
    n_sym, n = closes.shape
    fees = np.array([FEE_BPS.get(s, 30.0) for s in syms]) / 1e4
    cut = int(n * TRAIN_FRAC)
    print(f"universe: {n_sym} symbols, {n} aligned hourly bars "
          f"({n / HOURS_PER_YEAR:.2f} yr), OOS = last {n - cut} bars")
    print(f"symbols: {', '.join(s.replace('USDT', '') for s in syms)}\n")

    os.makedirs(os.path.dirname(UNIVERSE_FILE), exist_ok=True)
    with open(UNIVERSE_FILE, "w") as fh:
        fh.write("\n".join(syms) + "\n")

    print("=" * 96)
    print("BREADTH vs SHARPE  (long top-N / short bottom-N, weekly, 1x)")
    print("=" * 96)
    print(f"{'legs':>5}{'names tested':>13}{'CAGR':>10}{'Sharpe':>9}"
          f"{'maxDD':>9}{'portfolio vol':>15}")
    rows = {}
    for legs in LEG_GRID:
        p = xs_book(closes, fees, legs)
        rows[legs] = p
        m = metrics(p[cut:], f"legs={legs}")
        vol = float(np.std(p[cut:]) * math.sqrt(HOURS_PER_YEAR))
        print(f"{legs:>5}{2 * legs:>13}{m['cagr'] * 100:9.1f}%"
              f"{m['sharpe']:9.2f}{m['mdd'] * 100:8.1f}%{vol * 100:14.1f}%")

    print("\n  same table on the last 1 year only (the bear half)")
    yr = 8760
    for legs in LEG_GRID:
        m = metrics(rows[legs][-yr:])
        print(f"   legs={legs:<3} CAGR {m['cagr'] * 100:7.1f}%  "
              f"Sharpe {m['sharpe']:6.2f}  maxDD {m['mdd'] * 100:7.1f}%")

    # the ceiling question: best of the grid, and what selection explains
    best = max(LEG_GRID, key=lambda L: metrics(rows[L][cut:])["sharpe"])
    bm = metrics(rows[best][cut:])
    print(f"\n  best of {len(LEG_GRID)} legs = {best} "
          f"(OOS Sharpe {bm['sharpe']:.2f}, CAGR {bm['cagr'] * 100:.1f}%)")
    print(f"  benchmarks: top-decile fund 1.80, Medallion 2.52, S&P 1.10")

    # does more breadth help the CROSS-SECTION at all?
    first = metrics(rows[LEG_GRID[0]][cut:])["sharpe"]
    last = metrics(rows[LEG_GRID[-1]][cut:])["sharpe"]
    print(f"\n  Sharpe at {LEG_GRID[0]} legs: {first:.2f}   "
          f"at {LEG_GRID[-1]} legs: {last:.2f}   "
          f"delta {last - first:+.2f}")
    print("  -> breadth helps only if this delta is clearly positive.")


if __name__ == "__main__":
    main()
