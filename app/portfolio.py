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
               trail=None, be_at=None, cooldown=0, runner=False,
               entry="taker", pullback=0.25, maker_fee_bps=2.0,
               taker_fee_bps=5.0, maker_penalty_bps=1.0,
               min_penetration=0.0, fill_mode="next_bar",
               diag: dict | None = None) -> list[dict]:
    """Settle a single-position book over `bars`.

    bars    Bars object (o/h/l/c/t).
    score   array, one decision per bar; sign is the side, 0 means flat.
    mask    optional bool array of extra tradability conditions.
    pt/sl   barrier multiples of `sigma` (realised vol scaled to `hz`).
    hz      maximum holding bars (time stop).
    trail   trailing stop distance in `sigma` units (chandelier).
    be_at   R multiple at which the stop moves to break-even.
    runner  convex mode: no take-profit at all, the trail or the clock
            closes the trade.  Convexity comes from an uncapped right tail
            against a tight structural stop, which is the only way a
            sub-50 % hit rate can pay a round trip.
    entry   "taker"  -> market at the next open (pays taker both ways)
            "maker_pullback" -> resting limit `pullback` sigma below/above
            the next open; fills only when the bar trades *through* it, and
            then pays maker on entry, taker on exit, plus an adverse-fill
            penalty.  Unfilled signals are counted in `diag`, never booked.
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
    maker = str(entry) == "maker_pullback"
    fallback_fee = float(fees_bps) / 1e4
    taker_leg = float(taker_fee_bps) / 1e4
    attempts = 0
    trades: list[dict] = []
    i = 0
    while i < n - 2:
        if not ok[i] or s[i] == 0.0:
            i += 1
            continue
        j = i + 1
        side = 1.0 if s[i] > 0 else -1.0
        width = float(sg[i])
        attempts += 1
        if maker:
            ref = float(op[j])
            if not ref > 0:
                i += 1
                continue
            limit = ref * (1.0 - side * pullback * width)
            # A resting order only fills if price trades *through* the level.
            # `fill_mode` decides what price the fill actually gets:
            #   "wick"      fill exactly at the limit inside the bar that
            #               pierced it.  This is the standard backtest
            #               fiction: the fill is always at the best price of
            #               the bar, and a random-walk replay proves it — a
            #               coin-flip side earns ~+15 bp per trade, which no
            #               market pays for liquidity provision.
            #   "next_bar"  (default) the order is filled one bar later, at
            #               that bar's open if it is still at/through the
            #               level, else dropped.  You only get filled while
            #               the move continues against you, which is exactly
            #               the adverse selection the maker literature
            #               measures.
            through = limit * (1.0 - side * float(min_penetration))
            touched = (lo[j] <= through) if side > 0 else (h[j] >= through)
            if not touched:
                i += 1
                continue
            if str(fill_mode) == "wick":
                entry_px = limit
            else:
                k = j + 1
                if k >= n - 1:
                    i += 1
                    continue
                nxt = float(op[k])
                ok_open = (nxt <= limit) if side > 0 else (nxt >= limit)
                if not ok_open:
                    i += 1
                    continue
                entry_px = nxt
                j = k
                if not entry_px > 0:
                    i += 1
                    continue
            fee = (float(maker_fee_bps) + float(maker_penalty_bps)
                   + float(taker_fee_bps)) / 1e4
        else:
            entry_px = float(op[j])
            if not entry_px > 0:
                i += 1
                continue
            fee = fallback_fee
        tp_px = math.inf if runner else entry_px * (1.0 + side * pt * width)
        sl_px = entry_px * (1.0 - side * sl * width)
        stop = sl_px
        best = entry_px
        end = min(j + int(hz), n - 1)
        exit_kind, exit_px, exit_bar = "time", float(c[end]), end
        for t in range(j, end + 1):
            # stop first: a bar that touches both is booked as a loss.
            # The entry bar cannot ratchet the trail: we do not know whether
            # its high came before or after the fill, and assuming "high
            # first" is how trailing-stop backtests invent profits.
            if side > 0:
                if trail is not None and t > j:
                    best = max(best, float(h[t]))
                if lo[t] <= stop:
                    exit_kind, exit_px, exit_bar = "sl", stop, t
                    break
                if not runner and h[t] >= tp_px:
                    exit_kind, exit_px, exit_bar = "tp", tp_px, t
                    break
                if trail is not None:
                    # trail distance is a fraction of price, not an absolute
                    stop = max(stop, best * (1.0 - trail * width))
            else:
                if trail is not None and t > j:
                    best = min(best, float(lo[t]))
                if h[t] >= stop:
                    exit_kind, exit_px, exit_bar = "sl", stop, t
                    break
                if not runner and lo[t] <= tp_px:
                    exit_kind, exit_px, exit_bar = "tp", tp_px, t
                    break
                if trail is not None:
                    stop = min(stop, best * (1.0 + trail * width))
            if be_at is not None:
                move = (float(c[t]) / entry_px - 1.0) * side
                if move >= be_at * sl * width:
                    stop = max(stop, entry_px) if side > 0 else min(stop, entry_px)
        gross = (exit_px / entry_px - 1.0) * side
        risk = max(1e-12, sl * width)
        trades.append({
            "i": int(j), "side": side, "exit": exit_kind,
            "bars": int(exit_bar - j), "gross": float(gross),
            "fee_bp": round(fee * 1e4, 2),
            "net": float(gross - fee), "sigma": width,
            "r": float(gross - fee) / risk,
            "mfe": float((best / entry_px - 1.0) * side),
        })
        i = int(exit_bar) + 1 + int(cooldown)
    if diag is not None:
        diag["attempts"] = attempts
        diag["fills"] = len(trades)
        diag["fill_rate"] = (len(trades) / attempts) if attempts else 0.0
        diag["entry"] = entry
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
    fees = np.asarray([t.get("fee_bp", fees_bps) for t in trades], dtype=float)
    fee = float(np.mean(fees)) / 1e4 if fees.size else float(fees_bps) / 1e4
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
        "fee_bp": round(float(np.mean(fees)) if fees.size else fee * 1e4, 2),
        "fills": int(n),
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


def block_bootstrap(returns, block: int = 20, draws: int = 2000,
                    seed: int = 0) -> dict:
    """Circular block bootstrap of a trade series.

    Trades cluster: a burst of events shares one price path, so the plain
    t-statistic (`mean / sd * sqrt(n)`) overstates significance exactly when
    the strategy is most active.  Resampling *blocks* of consecutive trades
    preserves that autocorrelation and gives an honest standard error.
    """
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return {"n": int(n), "mean_bp": float("nan"), "se_bp": float("nan"),
                "t": float("nan"), "lo_bp": float("nan"), "hi_bp": float("nan")}
    block = int(max(1, min(block, n // 2)))
    rng = np.random.default_rng(seed)
    nblocks = int(math.ceil(n / block))
    starts = rng.integers(0, n, size=(draws, nblocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    samples = x[idx.reshape(draws, -1)[:, :n]]
    means = samples.mean(axis=1)
    se = float(means.std(ddof=1))
    mean = float(x.mean())
    return {"n": int(n), "block": int(block),
            "mean_bp": round(mean * 1e4, 2),
            "se_bp": round(se * 1e4, 2),
            "t": round(mean / se, 2) if se > 0 else float("nan"),
            "lo_bp": round(float(np.percentile(means, 2.5)) * 1e4, 2),
            "hi_bp": round(float(np.percentile(means, 97.5)) * 1e4, 2)}

# [removed 2026-11-23] bucket_neff — dead code: zero references anywhere
