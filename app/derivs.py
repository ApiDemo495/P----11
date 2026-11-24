"""Derivatives context: funding, basis, open interest, taker flow.

Spot OHLCV cannot see leverage.  A candle tells you the price moved; funding,
basis and open interest tell you *who* was forced to move it — and that is
where post-fee expectancy comes from, because forced flow is not arbitraged
away by the next millisecond.

Data (real archives, nothing synthesised)
=========================================
`backtest/cache/derivs/{SYM}_{kind}.csv`, written by
`scripts/fetch_binance_mirror.py --source derivs`:

    funding   t, funding_rate, mark_price          (8h settlements)
    basis_1h  t,o,h,l,c,... premium-index klines    (mark vs index, 1h)
    perp_1h   t,o,h,l,c,... USD-M perpetual klines  (1h)
    oi_1h     t, oi_base, oi_value                  (1h snapshots)
    taker_1h  t, buy_vol, sell_vol, buy_sell_ratio  (1h)

Causality
=========
Every feature is an **as-of** join on timestamps: the value used at bar `t`
is the newest observation whose timestamp is `<= t` (funding) or `< t`
(bar-aggregated series, because a bar's close is only known at its end).
`align()` does the whole join vectorised, so a lane can be built once and
sliced by any fold without re-reading files.
"""

from __future__ import annotations

import csv
import os

import numpy as np

DAY = 86400.0
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "backtest", "cache", "derivs")

KINDS = ("funding", "basis_1h", "perp_1h", "oi_1h", "taker_1h")


def _read(path: str) -> dict[str, np.ndarray]:
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    cols: dict[str, list[float]] = {k: [] for k in rows[0]}
    for r in rows:
        for k in cols:
            try:
                cols[k].append(float(r[k]))
            except (TypeError, ValueError):
                cols[k].append(float("nan"))
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def load(symbol: str, cache: str | None = None) -> dict[str, dict]:
    """All derivative series available for one symbol (empty when absent)."""
    root = cache or CACHE
    out: dict[str, dict] = {}
    for kind in KINDS:
        d = _read(os.path.join(root, f"{symbol}_{kind}.csv"))
        if d:
            out[kind] = d
    return out


def available(symbol: str) -> bool:
    return "funding" in load(symbol)


def _asof(src_t: np.ndarray, src_v: np.ndarray, dst_t: np.ndarray,
          lag: float = 0.0) -> np.ndarray:
    """Newest source value with `t <= dst_t - lag` (NaN before the first)."""
    if src_t.size == 0:
        return np.full(dst_t.size, np.nan)
    order = np.argsort(src_t)
    st, sv = src_t[order], src_v[order]
    idx = np.searchsorted(st, dst_t - lag, side="right") - 1
    out = np.full(dst_t.size, np.nan)
    ok = idx >= 0
    out[ok] = sv[idx[ok]]
    return out


def _roll_z(x: np.ndarray, win: int) -> np.ndarray:
    """Causal rolling z-score (NaN-safe, min 30 % of the window)."""
    n = x.size
    out = np.full(n, np.nan)
    if n == 0:
        return out
    win = max(2, int(win))
    for i in range(n):
        j = max(0, i - win + 1)
        seg = x[j:i + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size < max(3, int(0.3 * win)):
            continue
        mu = float(seg.mean())
        sd = float(seg.std())
        if sd > 1e-12:
            out[i] = (x[i] - mu) / sd
    return out


def _roll_mean(x: np.ndarray, win: int) -> np.ndarray:
    n = x.size
    out = np.full(n, np.nan)
    win = max(1, int(win))
    for i in range(n):
        j = max(0, i - win + 1)
        seg = x[j:i + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = float(seg.mean())
    return out


def align(d: dict[str, dict], t: np.ndarray,
          bar_seconds: float = 3600.0) -> dict[str, np.ndarray]:
    """Feature block aligned to decision timestamps `t` (seconds, sorted).

    Returns NaN-free arrays (missing data -> 0 for flows, 0 for z-scores)
    plus a `has_derivs` mask so a lane can refuse to trade where the
    archive does not reach.
    """
    t = np.asarray(t, dtype=float)
    lag = float(bar_seconds)          # bar aggregates are known one bar later
    f = {}                            # type: dict[str, np.ndarray]

    fund = d.get("funding")
    if fund is not None:
        ft = fund["t"]
        fr = fund["funding_rate"]
        seen = _asof(ft, np.ones_like(fr), t, lag=0.0)
        f["funding_rate"] = _asof(ft, fr, t, lag=0.0)          # settled value
        # 3-day funding carry (9 settlements) — the carry you would pay to
        # hold the crowded side, evaluated at the decision stamp
        f["funding_carry_3d"] = _asof(ft, _cumsum_time(ft, fr, 3 * DAY), t,
                                      lag=0.0)
        f["funding_seen"] = np.isfinite(seen).astype(float)
        f["funding_z"] = _asof(
            ft, _roll_z(np.nan_to_num(fr, nan=0.0), 30), t, lag=0.0)
        f["funding_abs_z"] = np.abs(np.nan_to_num(f["funding_z"], nan=0.0))
    else:
        f["funding_seen"] = np.zeros(t.size)
        f["funding_rate"] = np.zeros(t.size)
        f["funding_carry_3d"] = np.zeros(t.size)
        f["funding_z"] = np.zeros(t.size)
        f["funding_abs_z"] = np.zeros(t.size)

    basis = d.get("basis_1h")
    if basis is not None and "c" in basis:
        bt, bp = basis["t"], basis["c"]
        raw = _asof(bt, bp, t, lag=lag)       # premium index (fraction)
        f["basis_bp"] = np.nan_to_num(raw, nan=0.0) * 1e4
        f["basis_z"] = np.nan_to_num(_asof(bt, _roll_z(np.nan_to_num(bp), 240),
                                           t, lag=lag), nan=0.0)
    else:
        f["basis_bp"] = np.zeros(t.size)
        f["basis_z"] = np.zeros(t.size)

    oi = d.get("oi_1h")
    if oi is not None and "oi_value" in oi:
        ot, ov = oi["t"], oi["oi_value"]
        cur = _asof(ot, ov, t, lag=lag)
        prev = _asof(ot, ov, t, lag=lag + DAY)          # 24h earlier
        f["oi_chg_24h"] = np.nan_to_num(
            np.where(prev > 0, cur / np.maximum(prev, 1e-9) - 1.0, 0.0), nan=0.0)
        f["oi_z"] = np.nan_to_num(
            _asof(ot, _roll_z(np.nan_to_num(ov), 168), t, lag=lag), nan=0.0)
        f["oi_seen"] = np.isfinite(cur).astype(float)
    else:
        f["oi_chg_24h"] = np.zeros(t.size)
        f["oi_z"] = np.zeros(t.size)
        f["oi_seen"] = np.zeros(t.size)

    taker = d.get("taker_1h")
    if taker is not None and "buy_sell_ratio" in taker:
        tt, br = taker["t"], taker["buy_sell_ratio"]
        f["taker_ratio"] = np.nan_to_num(_asof(tt, br, t, lag=lag), nan=1.0)
        f["taker_z"] = np.nan_to_num(
            _asof(tt, _roll_z(np.nan_to_num(br, nan=1.0), 168), t, lag=lag),
            nan=0.0)
        f["taker_seen"] = np.isfinite(_asof(tt, br, t, lag=lag)).astype(float)
    else:
        f["taker_ratio"] = np.ones(t.size)
        f["taker_z"] = np.zeros(t.size)
        f["taker_seen"] = np.zeros(t.size)

    # a lane may only trade where the archive actually reaches
    f["has_derivs"] = np.clip(np.nan_to_num(f.get("funding_seen", 0.0), nan=0.0)
                              + np.nan_to_num(f.get("oi_seen", 0.0), nan=0.0),
                              0.0, 1.0)
    return f


def _cumsum_time(t: np.ndarray, v: np.ndarray, span: float) -> np.ndarray:
    """Trailing sum over `span` seconds, evaluated at each source stamp."""
    if t.size == 0:
        return np.zeros(0)
    order = np.argsort(t)
    ts, vs = t[order], np.nan_to_num(v[order])
    cs = np.concatenate([[0.0], np.cumsum(vs)])
    out = np.zeros(ts.size)
    lo = np.searchsorted(ts, ts - span, side="left")
    for i in range(ts.size):
        out[i] = cs[i + 1] - cs[lo[i]]
    return out

# [removed 2026-11-23] squeeze_score — dead code: zero references anywhere
