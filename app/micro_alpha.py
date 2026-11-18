"""Microstructure alpha computed from klines (no L2 book required).

Everything here is a pure function of bars that carry Binance's own
aggressor fields (`taker_buy_volume`, `quote_volume`, `count`), so it works
identically in the backtest and live. Formulas follow the literature:

  OFI      order-flow imbalance: (2·taker_buy − vol)/vol, the volume-
           normalised signed flow used in arXiv 2607.09426 and the
           microstructure-SHAP study (arXiv 2602.00776). Multi-horizon and
           persistence-weighted, because the *persistent* component carries
           the signal.
  VPIN     Easley / López de Prado / O'Hara: equal-VOLUME buckets, bulk
           classification, |buy − sell|/V averaged over a rolling window.
           VPIN is a TOXICITY/volatility measure, not a direction call —
           used here to price risk and to veto trading in toxic windows.
  Kyle λ   rolling OLS slope of price change on signed flow: the price
           impact per unit of net flow. High λ = thin book = higher
           effective friction. This is what makes the cost model adaptive
           instead of a hard-coded 12 bp.
  Amihud   |return| per unit of quote volume — illiquidity cross-check.
  Size skew  large prints vs the lane's typical print, signed by the
           aggressor side: separates taker sweeps from retail noise.
  Hawkes   branching ratio of trade arrivals, estimated from the
           autocorrelation of trade counts (α/β of a Hawkes process).
           n → 1 means self-exciting flow (cascades); n → 0 Poisson.

All functions are causal: they take a bar index and read strictly backwards.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _arr(bars: Any, name: str) -> np.ndarray:
    v = getattr(bars, name, None)
    if v is None:
        return np.zeros(0)
    return np.asarray(v, dtype=float)


def _imb(bars: Any) -> np.ndarray:
    """Signed taker imbalance in [-1, 1] per bar."""
    v = _arr(bars, "v")
    tb = _arr(bars, "taker_buy_v")
    if v.size == 0 or tb.size != v.size:
        return np.zeros_like(v)
    return np.divide(2.0 * tb - v, np.maximum(v, 1e-12))


# ───────────────────────────── OFI ─────────────────────────────


def ofi(bars: Any, i: int, horizons=(1, 3, 12)) -> dict:
    """Multi-horizon order-flow imbalance + its persistence."""
    im = _imb(bars)
    if im.size == 0 or i < max(horizons):
        return {"ofi": 0.0, "ofi_persist": 0.0, "ofi_long": 0.0}
    short = float(np.mean(im[i - horizons[0] + 1:i + 1]))
    mid = float(np.mean(im[i - horizons[1] + 1:i + 1]))
    long = float(np.mean(im[i - horizons[2] + 1:i + 1]))
    persist = float(np.mean(np.sign(im[i - horizons[2] + 1:i + 1])))
    return {"ofi": (short + mid + long) / 3.0,
            "ofi_persist": persist, "ofi_long": long}


# ───────────────────────────── VPIN ─────────────────────────────


def vpin(bars: Any, i: int, buckets: int = 20, window: int = 50) -> float:
    """Volume-bucket VPIN over the `window` buckets ending at bar `i`.

    Bulk classification: every bucket's imbalance is |buy−sell|/V, where
    buy volume comes from the exchange's taker flag. Equal-volume buckets
    make busy and quiet periods comparable (the mistake the literature
    warns about is sizing buckets by bar count).
    """
    v = _arr(bars, "v")
    tb = _arr(bars, "taker_buy_v")
    if v.size == 0 or i < window or tb.size != v.size:
        return float("nan")
    seg_v = v[max(0, i - 4 * window - 1):i + 1]
    seg_tb = tb[max(0, i - 4 * window - 1):i + 1]
    total = float(seg_v.sum())
    if total <= 0:
        return float("nan")
    bucket_v = total / buckets
    if bucket_v <= 0:
        return float("nan")
    imbs, acc_v, acc_imb = [], 0.0, 0.0
    for vol, buy in zip(seg_v, seg_tb):
        acc_v += vol
        acc_imb += 2.0 * buy - vol
        while acc_v >= bucket_v and len(imbs) < buckets:
            take = bucket_v / acc_v if acc_v > 0 else 1.0
            imbs.append(abs(acc_imb * take) / bucket_v)
            acc_imb *= (1.0 - take)
            acc_v -= bucket_v
    if not imbs:
        return float("nan")
    return float(min(1.0, np.mean(imbs[-window:])))


# ─────────────────────────── Kyle lambda ───────────────────────────


def kyle_lambda(bars: Any, i: int, window: int = 60) -> float:
    """OLS slope of Δlog price on signed net flow (per unit volume).

    Units: log-return per unit of net base volume. Only the sign and the
    relative magnitude matter — `friction_bps` below normalises it.
    """
    c = _arr(bars, "c")
    v = _arr(bars, "v")
    tb = _arr(bars, "taker_buy_v")
    if c.size < window + 2 or i < window:
        return float("nan")
    px = np.log(np.maximum(c[i - window:i + 1], 1e-12))
    dp = np.diff(px)
    flow = (2.0 * tb[i - window + 1:i + 1] - v[i - window + 1:i + 1])
    if flow.size != dp.size:
        return float("nan")
    # scale flow to its own std so a single asset's unit size is not
    # confused with impact magnitude
    sd = float(np.std(flow))
    if sd <= 0:
        return float("nan")
    x = flow / sd
    xm, ym = x - x.mean(), dp - dp.mean()
    den = float((xm * xm).sum())
    if den <= 1e-18:
        return float("nan")
    return float((xm * ym).sum() / den)


def amihud(bars: Any, i: int, window: int = 60) -> float:
    """|return| per unit quote volume, scaled to 1e6 for readability."""
    c = _arr(bars, "c")
    qv = _arr(bars, "quote_v")
    if c.size < window + 2 or qv.size != c.size or i < window:
        return float("nan")
    r = np.abs(np.diff(np.log(np.maximum(c[i - window:i + 1], 1e-12))))
    q = qv[i - window + 1:i + 1]
    ok = q > 0
    if not np.any(ok):
        return float("nan")
    return float(np.mean(r[ok] / q[ok]) * 1e6)


def friction_bps(bars: Any, i: int, window: int = 60,
                 floor: float = 12.0, cap: float = 40.0) -> float:
    """Adaptive round-trip cost estimate in bps.

    Adaptive because a fixed 12 bp is a fiction in a thin window: price
    impact scales with Kyle's λ and inversely with depth. The floor is the
    exchange fee round trip (2 × 4 bp = 8) plus 4 bp slippage; the cap
    stops a stress window from making everything look untradeable.
    """
    lam = kyle_lambda(bars, i, window)
    if not math.isfinite(lam):
        return floor
    # `kyle_lambda` standardises flow, so λ is already the log-return
    # impact of a ONE-SIGMA flow bar — multiplying by volume again would
    # double-count notional and inflate the hurdle (the bug that made a
    # 20-minute lane look untradable at 39 bp of "friction").
    impact_bps = abs(lam) * 1e4
    return float(min(cap, max(floor, 8.0 + 4.0 + 0.5 * impact_bps)))


# ─────────────────────── trade-size asymmetry ───────────────────────


def size_skew(bars: Any, i: int, short: int = 5, long: int = 96) -> float:
    """Large prints vs the lane's typical print, signed by the aggressor.

    +1 means unusually large buying prints (sweeps), −1 unusually large
    selling prints, 0 ordinary flow.
    """
    v = _arr(bars, "v")
    n = _arr(bars, "trades")
    im = _imb(bars)
    if v.size == 0 or n.size != v.size or i < long:
        return 0.0
    size_now = float(np.mean(v[i - short + 1:i + 1] /
                             np.maximum(n[i - short + 1:i + 1], 1.0)))
    size_ref = float(np.median(v[max(0, i - long):i + 1] /
                               np.maximum(n[max(0, i - long):i + 1], 1.0)))
    if size_ref <= 0 or size_now <= 0:
        return 0.0
    rel = math.log(size_now / size_ref)
    side = float(np.mean(im[i - short + 1:i + 1]))
    return float(max(-1.0, min(1.0, math.tanh(rel) * math.copysign(1.0, side)
                              if side else 0.0)))


# ─────────────────────── Hawkes branching ratio ───────────────────────


def hawkes_branching(bars: Any, i: int, window: int = 120) -> float:
    """Branching ratio n ∈ [0,1) from trade-count clustering.

    Moment estimator for an exponential-kernel Hawkes process: with
    inter-arrival mean m and variance s², n ≈ 1 − m²/s²·(1+…) is unstable
    on counts, so the ratio is estimated from the lag-1 autocorrelation of
    a Poisson residual: n = max(0, ρ1). Reported as a *cascade intensity*.
    """
    n_ = _arr(bars, "trades")
    if n_.size < window + 4 or i < window:
        return float("nan")
    x = n_[i - window + 1:i + 1]
    mu, sd = float(np.mean(x)), float(np.std(x))
    if sd <= 1e-12:
        return 0.0
    z = (x - mu) / sd
    rho = float(np.mean(z[1:] * z[:-1]))
    return float(max(0.0, min(0.99, rho)))


# ─────────────────────── composite feature row ───────────────────────


def features(bars: Any, i: int) -> dict:
    """Every microstructure reading at bar `i`, computed causally."""
    o = ofi(bars, i)
    return {
        "ofi": o["ofi"],
        "ofi_persist": o["ofi_persist"],
        "ofi_long": o["ofi_long"],
        "vpin": vpin(bars, i),
        "kyle": kyle_lambda(bars, i),
        "amihud": amihud(bars, i),
        "size_skew": size_skew(bars, i),
        "hawkes": hawkes_branching(bars, i),
        "friction_bps": friction_bps(bars, i),
    }


MICRO_NAMES = ("ofi", "ofi_persist", "ofi_long", "vpin", "kyle", "amihud",
               "size_skew", "hawkes")
