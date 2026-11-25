"""Long-only trend + volatility targeting: the one configuration in this
project that beats its own benchmark in both halves of the sample.

Why this and not the long/short books of §32-§34: the published literature
is consistent about two things (see §35 of REAL_RESULTS.md for sources):

  1. Crypto long/short momentum profits "mostly originate from the long leg;
     the short leg is exposed to high jump risks and incurs losses"
     (ACFR working paper, Time-Series and Cross-Sectional Momentum in Crypto).
  2. Volatility targeting / risk management "nearly doubles the Sharpe ratio"
     while keeping turnover comparable (Grobys et al., 'Cryptocurrency
     momentum has (not) its moments').

Both replicate on this tape, and the replication is measurable: the short leg
alone costs -115 % cumulative over 2.75 years. That is the whole story of
§32-§34 - the books were long a real trend premium and short a jump premium.

Honest limits, carried in the code rather than the prose:
  * long-only crypto is BETA. The benchmark here is therefore not the S&P but
    equal-weight buy-and-hold with the same vol target.
  * the horizon is 2.75 years. A Sharpe of 1.4 on that is not a fund.
  * survivorship: the universe is symbols listed today. Removing the top three
    contributors costs -0.68 Sharpe (1.38 -> 0.70). Measured, not assumed.

Run:  .venv/bin/python -m backtest.longonly
"""
from __future__ import annotations

import math

import numpy as np

from .beat_funds import FEE_BPS, HOURS_PER_YEAR, TRAIN_FRAC, metrics
from .breadth_test import align_closes, load_universe

#: pre-registered, not searched
LEGS = 4
REBALANCE = 720          # monthly
LOOKBACK = 336           # 2 weeks
VOL_WINDOW = 720
TARGET_VOL = 0.40
MAX_LEV = 3.0


def vol_scale(raw: np.ndarray, target: float = TARGET_VOL,
              window: int = VOL_WINDOW, cap: float = MAX_LEV) -> np.ndarray:
    """Leverage the series toward `target` annual vol. The leverage applied at
    bar i is computed from returns up to i-1 only (no lookahead)."""
    cs = np.cumsum(np.insert(raw, 0, 0.0))
    c2 = np.cumsum(np.insert(raw ** 2, 0, 0.0))
    s1 = cs[window + 1:] - cs[1:-window]
    s2 = c2[window + 1:] - c2[1:-window]
    rv = np.zeros(len(raw))
    var = np.maximum(s2 / window - (s1 / window) ** 2, 0.0)
    rv[window:] = np.sqrt(var) * math.sqrt(HOURS_PER_YEAR)
    lev = np.where(rv > 0, target / np.maximum(rv, 1e-9), 1.0)
    lev = np.clip(np.nan_to_num(lev, nan=1.0), 0.0, cap)
    out = np.zeros_like(raw)
    out[window:] = raw[window:] * lev[window - 1:-1]
    out[:window] = raw[:window]
    return out


def long_only_book(closes: np.ndarray, fees: np.ndarray,
                   legs: int = LEGS, rebalance: int = REBALANCE,
                   lookback: int = LOOKBACK,
                   drop: tuple[int, ...] = ()) -> np.ndarray:
    """Top-N by trailing return, equal weight, always fully invested.
    No abstention: there is no state in which nothing is held."""
    n_sym, n = closes.shape
    rets = np.zeros_like(closes)
    rets[:, 1:] = closes[:, 1:] / closes[:, :-1] - 1.0
    allowed = np.ones(n_sym, dtype=bool)
    for j in drop:
        allowed[j] = False

    pos = np.zeros((n_sym, n))
    prev = np.zeros(n_sym)
    for i in range(lookback + 1, n):
        if (i - lookback) % rebalance == 0:
            past = closes[:, i] / closes[:, i - lookback] - 1.0
            order = np.argsort(past)
            new = np.zeros(n_sym)
            chosen = [j for j in order[-legs:] if allowed[j]]
            for j in chosen:
                new[j] = 1.0 / legs
            prev = new
        pos[:, i] = prev

    pnl = np.zeros(n)
    pnl[1:] = np.sum(pos[:, :-1] * rets[:, 1:], axis=0)
    dpos = np.abs(np.diff(pos, axis=1))
    for k in range(n_sym):
        pnl[1:] -= dpos[k] * fees[k]
    return vol_scale(pnl)


def newey_west_alpha(y: np.ndarray, x: np.ndarray, lags: int = 48):
    """Alpha and its t-statistic, HAC standard errors."""
    X = np.vstack([np.ones(len(x)), x]).T
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    S = (e[:, None] * X).T @ (e[:, None] * X)
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = (e[lag:, None] * X[lag:]).T @ (e[:-lag, None] * X[:-lag])
        S += w * (G + G.T)
    cov = XtXi @ S @ XtXi
    return float(b[1]), float(b[0] * HOURS_PER_YEAR * 100), float(b[0] / math.sqrt(cov[0, 0]))


def main() -> None:
    tapes = load_universe()
    syms, closes = align_closes(tapes)
    n_sym, n = closes.shape
    fees = np.array([FEE_BPS.get(s, 30.0) for s in syms]) / 1e4
    cut = int(n * TRAIN_FRAC)

    rets = np.zeros_like(closes)
    rets[:, 1:] = closes[:, 1:] / closes[:, :-1] - 1.0
    ew = np.mean(rets, axis=0)
    ew[0] = 0.0
    bench = vol_scale(ew.copy())

    cand = long_only_book(closes, fees)
    print(f"universe {n_sym} symbols, {n} bars, OOS = last {n - cut}\n")
    print(f"{'':>34}{'CAGR':>9}{'Sharpe':>8}{'maxDD':>9}")
    for lab, p in (("equal-weight buy & hold + VT", bench),
                   ("long-only trend, monthly + VT", cand)):
        m = metrics(p)
        print(f"{lab:>34}{m['cagr'] * 100:8.1f}%{m['sharpe']:8.2f}{m['mdd'] * 100:8.1f}%")

    beta, alpha, t = newey_west_alpha(cand, bench)
    print(f"\nalpha vs benchmark: {alpha:+.1f}%/yr  beta {beta:+.2f}  t {t:+.2f}")
    print(f"  per half: train {metrics(cand[:cut])['sharpe']:.2f} "
          f"test {metrics(cand[cut:])['sharpe']:.2f} "
          f"(bench {metrics(bench[:cut])['sharpe']:.2f} / {metrics(bench[cut:])['sharpe']:.2f})")

    contrib = []
    for j in range(n_sym):
        contrib.append((cand.sum() - long_only_book(closes, fees, drop=(j,)).sum(), j))
    contrib.sort(reverse=True)
    print("\nsurvivorship (drop top contributors):")
    for k in (1, 2, 3, 5):
        drop = tuple(j for _, j in contrib[:k])
        m = metrics(long_only_book(closes, fees, drop=drop))
        print(f"  drop top-{k}: Sharpe {m['sharpe']:5.2f}  CAGR {m['cagr'] * 100:5.1f}%"
              f"   ({', '.join(syms[j].replace('USDT', '') for j in drop)})")

    w = 6 * 730
    edges = []
    for st in range(720, n - w, w // 2):
        edges.append(metrics(cand[st:st + w])["sharpe"] - metrics(bench[st:st + w])["sharpe"])
    edges = np.array(edges)
    print(f"\nrolling 6-month windows: edge over benchmark positive in "
          f"{int((edges > 0).sum())}/{len(edges)}, mean {edges.mean():+.2f}")

    m = metrics(cand)
    print(f"\nagainst the fund benchmarks: Sharpe {m['sharpe']:.2f} "
          f"vs S&P 1.10, top-decile 1.80, Medallion 2.52")


if __name__ == "__main__":
    main()
