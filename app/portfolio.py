"""Sequential single-position portfolio engine.

Why this module exists
======================
The per-bar lab (`evaluate_deep`) measures the average outcome of *every*
qualifying bar.  That is a signal study, not a strategy: it stacks
overlapping bets on the same price path, so its trade count is inflated and
its EV is dominated by whichever regime happened to be sampled first.  A
real book can hold one position per symbol, must wait for the exit before
re-entering, and pays friction on each round trip.

This engine settles that book exactly:

* entry at the **next bar open** after the decision bar (taker),
* exits inside the bar on high/low touches, **stop wins ties**,
* optional volatility trailing stop that only ratchets in our favour,
* optional time stop, one position at a time, no pyramiding,
* every statistic reported on the *trade* series with an overlap-aware
  t-statistic so a hundred clustered trades cannot masquerade as evidence.

Nothing here looks ahead: `score[i]` / `mask[i]` are decisions taken at the
close of bar `i`; the fill is at `open[i+1]`.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_FEES_BPS = 12.0


def _block_neff(n: int, block: float) -> float:
    """Independent-sample count for `n` observations that overlap in time."""
    n = float(max(0, n))
    block = float(max(1.0, block))
    return n / block if n else 0.0


def run_sleeve(bars, score, mask=None, *, pt=2.0, sl=1.0, hz=60,
               sigma=None, fees_bps: float = DEFAULT_FEES_BPS,
               trail=None, be_at=None, cooldown=0) -> list[dict]:
    """Settle a single-position book over `bars`.

    bars    Bars object (o/h/l/c/t).
    score   array, one decision per bar; sign is the side, 0 means flat.
    mask    optional bool array of extra tradability conditions.
    pt/sl   barrier multiples of `sigma` (realised vol scaled to `hz`).
    hz      maximum holding bars (time stop).
    trail   optional trailing stop distance in `sigma` units.
    be_at   optional R multiple at which the stop moves to break-even.
    """
    c = np.asarray(bars.c, dtype=float)
    h = np.asarray(bars.h, dtype=float)
    lo = np.asarray(bars.l, dtype=float)
    op = np.asarray(bars.o, dtype=float)
    s = np.nan_to_num(np.asarray(score, dtype=float), nan=0.0, posinf=0.0,
                      neginf=0.0)
    n = c.size
    if sigma is None:
        from .events import sigma_series
        sigma = sigma_series(bars, 240)
    sg = np.asarray(sigma, dtype=float)
    ok = np.ones(n, bool) if mask is None else np.asarray(mask, bool).copy()
    ok &= np.isfinite(sg) & (sg > 0)
    fee = float(fees_bps) / 1e4
    trades: list[dict] = []
    i = 0
    while i < n - 2:
        if not ok[i] or s[i] == 0.0:
            i += 1
            continue
        j = i + 1
        entry = float(op[j])
        if not entry > 0:
            i += 1
            continue
        side = 1.0 if s[i] > 0 else -1.0
        width = float(sg[i])
        tp_px = entry * (1.0 + side * pt * width)
        sl_px = entry * (1.0 - side * sl * width)
        stop = sl_px
        best = entry
        end = min(j + int(hz), n - 1)
        exit_kind, exit_px, exit_bar = "time", float(c[end]), end
        for t in range(j, end + 1):
            # stop first: a bar that touches both is booked as a loss
            if side > 0:
                if trail is not None:
                    best = max(best, float(h[t]))
                if lo[t] <= stop:
                    exit_kind, exit_px, exit_bar = "sl", stop, t
                    break
                if h[t] >= tp_px:
                    exit_kind, exit_px, exit_bar = "tp", tp_px, t
                    break
                if trail is not None:
                    stop = max(stop, best - trail * width)
            else:
                if trail is not None:
                    best = min(best, float(lo[t]))
                if h[t] >= stop:
                    exit_kind, exit_px, exit_bar = "sl", stop, t
                    break
                if lo[t] <= tp_px:
                    exit_kind, exit_px, exit_bar = "tp", tp_px, t
                    break
                if trail is not None:
                    stop = min(stop, best + trail * width)
            if be_at is not None:
                move = (float(c[t]) / entry - 1.0) * side
                if move >= be_at * sl * width:
                    stop = max(stop, entry) if side > 0 else min(stop, entry)
        gross = (exit_px / entry - 1.0) * side
        trades.append({
            "i": int(j), "side": side, "exit": exit_kind,
            "bars": int(exit_bar - j), "gross": float(gross),
            "net": float(gross - fee), "sigma": width,
            "r": float(gross - fee) / max(1e-12, sl * width),
        })
        i = int(exit_bar) + 1 + int(cooldown)
    return trades


def trade_stats(trades: list[dict], *, block: float | None = None,
                fees_bps: float = DEFAULT_FEES_BPS) -> dict:
    """Honest statistics for a trade series (taker fees, no smoothing)."""
    n = len(trades)
    if n == 0:
        return {"n": 0, "ev_bp": float("nan"), "neff": 0.0,
                "t_stat": float("nan")}
    net = np.asarray([t["net"] for t in trades], dtype=float)
    gross = np.asarray([t["gross"] for t in trades], dtype=float)
    bars = np.asarray([max(1, t["bars"]) for t in trades], dtype=float)
    fee = float(fees_bps) / 1e4
    wins = net > 0
    gsum = float(net[net > 0].sum())
    lsum = float(-net[net < 0].sum())
    # overlap: average holding length relative to the spacing of entries
    entry_i = np.asarray([t["i"] for t in trades], dtype=float)
    spacing = float(np.median(np.diff(entry_i))) if n > 1 else float(bars[0])
    blk = block if block is not None else max(1.0, float(bars.mean()) / max(1.0, spacing))
    neff = _block_neff(n, blk)
    sd = float(net.std(ddof=1)) if n > 1 else 0.0
    # a degenerate series (every trade identical) carries no dispersion
    # information; without a floor the t-stat divides by float noise
    sd = sd if sd > 1e-9 else 0.0
    t_stat = (float(net.mean()) / (sd / math.sqrt(neff)) if sd > 0 and neff > 0
              else 0.0)
    downside = net[net < 0]
    dsd = float(downside.std(ddof=1)) if downside.size > 1 else 0.0
    dsd = dsd if dsd > 1e-9 else 0.0
    eq = np.cumsum(net)
    peak = np.maximum.accumulate(eq)
    dd = float(np.max(peak - eq)) if eq.size else 0.0
    return {
        "n": int(n),
        "neff": round(float(neff), 1),
        "ev_bp": round(float(net.mean()) * 1e4, 2),
        "gross_ev_bp": round(float(gross.mean()) * 1e4, 2),
        "fee_bp": round(fee * 1e4, 1),
        "win_rate": round(float(wins.mean()), 4),
        "profit_factor": round(gsum / lsum, 3) if lsum > 0 else float("inf"),
        "sharpe": round(float(net.mean() / sd * math.sqrt(neff)) if sd > 0 else 0.0, 3),
        "sortino": round(float(net.mean() / dsd * math.sqrt(neff)) if dsd > 0 else 0.0, 3),
        "t_stat": round(float(t_stat), 2),
        "max_dd_R": round(float(dd) / max(1e-9, float(np.median([t["r"] for t in trades])) or 1e-9), 3),
        "avg_bars": round(float(bars.mean()), 1),
        "tp_share": round(float(np.mean([t["exit"] == "tp" for t in trades])), 3),
        "sl_share": round(float(np.mean([t["exit"] == "sl" for t in trades])), 3),
    }


def bucket_neff(n: int, per_bucket: int) -> float:
    """Independent trades when `n` fires arrive in `per_bucket` clusters."""
    return _block_neff(n, max(1, per_bucket))
