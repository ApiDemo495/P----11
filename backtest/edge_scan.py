"""Edge scan: honest, purged, out-of-fold search for tradable sleeves.

Everything here is measured, nothing is assumed. The scan

  1. loads REAL bars from ``backtest/cache`` (see ``docs/ALPHA_RESEARCH.md``
     for provenance — Binance spot 1m resampled to 15m/30m/1h and Binance
     USD-M perpetual 1h),
  2. enumerates a fixed grammar of trigger families,
  3. runs PURGED WALK-FORWARD selection: parameters for fold *f* are
     chosen using only trades that closed before fold *f* started (minus
     an embargo equal to the holding period), then scored on fold *f*,
  4. hands the pooled out-of-fold trades to ``app.edge_gate.sleeve_gate``,
     which decides ``open`` / ``shadow`` / ``closed``.

Fill model (never negotiable): a decision taken on bar ``t``'s close is
filled at bar ``t+1``'s OPEN and exits at bar ``t+H``'s CLOSE. Friction
(round trip) is subtracted from every trade.

Usage::

    python -m backtest.edge_scan                 # full scan, writes results
    python -m backtest.edge_scan --quick         # smaller grid, same rules
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from app.edge_gate import (
    STATUS_OPEN,
    STATUS_SHADOW,
    Trade,
    sleeve_gate,
    summarise,
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

FRICTION_BPS = {
    "BTC": 12.0, "ETH": 12.0, "BNB": 12.0,
    "SOL": 12.0, "XRP": 12.0, "ADA": 12.0, "AVAX": 12.0, "LINK": 12.0,
    "PAXG": 18.0,
}

# Research window / locked holdout split.
#
# The split is defined on each asset's OWN tape: the first 65% of bars is
# the research window (folds are built only there), the last 35% is LOCKED
# and never informs selection. A sleeve is scored on the holdout exactly
# once, with its parameters frozen to the modal fold pick, and the gate
# requires the holdout to be positive too — so a sleeve cannot be promoted
# on parameter search alone.
#
# A per-asset fraction, rather than one calendar date, is used because the
# assets have different listing histories (PAXG starts 2025-03, the spot
# 1m archive starts 2017). A single calendar cut would leave PAXG with no
# research window at all.
HOLDOUT_FRACTION = 0.35
N_FOLDS = 5


def holdout_start_for(times: list) -> int:
    """First bar of the locked holdout for this asset's own tape."""
    if not times:
        return 0
    idx = int(len(times) * (1.0 - HOLDOUT_FRACTION))
    idx = max(1, min(idx, len(times) - 1))
    return times[idx]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@dataclass
class Series:
    symbol: str
    tf: str
    t: list
    o: list
    h: list
    l: list
    c: list

    def __len__(self) -> int:
        return len(self.t)


def load_series(symbol: str, tf: str) -> Series | None:
    path = os.path.join(CACHE_DIR, f"{symbol}_{tf}.csv")
    if not os.path.exists(path):
        return None
    t, o, h, l, c = [], [], [], [], []
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) < 5:
                continue
            t.append(int(float(row[0])))
            o.append(float(row[1]))
            h.append(float(row[2]))
            l.append(float(row[3]))
            c.append(float(row[4]))
    if len(t) < 500:
        return None
    return Series(symbol, tf, t, o, h, l, c)


def align(a: Series, b: Series) -> tuple[Series, Series]:
    """Intersect two series on bar-open timestamps."""
    da = {x: i for i, x in enumerate(a.t)}
    keep_a, keep_b = [], []
    for i, x in enumerate(b.t):
        j = da.get(x)
        if j is not None:
            keep_a.append(j)
            keep_b.append(i)

    def take(s: Series, idx: list) -> Series:
        return Series(
            s.symbol, s.tf,
            [s.t[i] for i in idx], [s.o[i] for i in idx],
            [s.h[i] for i in idx], [s.l[i] for i in idx],
            [s.c[i] for i in idx],
        )

    return take(a, keep_a), take(b, keep_b)


# ---------------------------------------------------------------------------
# indicators (causal only — every value at i uses data up to and including i)
# ---------------------------------------------------------------------------
def log_returns(series: Series) -> list:
    out = [0.0] * len(series)
    for i in range(1, len(series)):
        prev = series.c[i - 1]
        out[i] = math.log(series.c[i] / prev) if prev > 0 and series.c[i] > 0 else 0.0
    return out


def rolling_std(rets: list, window: int) -> list:
    n = len(rets)
    out = [float("nan")] * n
    acc = 0.0
    acc2 = 0.0
    for i in range(n):
        r = rets[i]
        acc += r
        acc2 += r * r
        if i >= window:
            old = rets[i - window]
            acc -= old
            acc2 -= old * old
        if i + 1 >= window:
            m = acc / window
            var = acc2 / window - m * m
            out[i] = math.sqrt(max(var, 1e-12))
    return out


def impulse(rets: list, sd: list, L: int) -> list:
    """Normalised cumulative move over the last L bars (z-like)."""
    n = len(rets)
    out = [float("nan")] * n
    for i in range(L, n):
        s = sd[i]
        if not (s == s) or s <= 0:
            continue
        acc = 0.0
        for j in range(i - L + 1, i + 1):
            acc += rets[j]
        out[i] = acc / (s * math.sqrt(L))
    return out


# ---------------------------------------------------------------------------
# macro / regime indicators (all causal, all O(n))
# ---------------------------------------------------------------------------
def donchian_breakout_signal(series: Series, lookback: int, hold_h: int,
                             bar_s: int, require_compression: bool = False,
                             vol_window: int = 500) -> list:
    """Close beyond the prior ``lookback``-bar extreme -> trend leg.

    Levels use ONLY bars strictly before ``i`` (``series.c[i-lookback:i]``),
    so the breakout level can never contain the bar that breaks it.
    Optional compression gate: only take breakouts while realised
    volatility sits in the bottom third of its own trailing distribution.
    """
    n = len(series)
    sig = [0] * n
    rets = log_returns(series)
    sd = rolling_std(rets, vol_window)
    # trailing percentile rank of volatility (causal, windowed)
    pct_window = max(vol_window, 500)
    hist: list = []
    for i in range(n):
        if sd[i] == sd[i] and sd[i] > 0:
            hist.append(sd[i])
        if i < lookback + vol_window:
            continue
        if require_compression:
            window = hist[-pct_window:] if len(hist) >= pct_window else None
            if not window:
                continue
            lo = sorted(window)[: max(1, len(window) // 3)]
            if sd[i] > lo[-1]:
                continue
        prior = series.c[i - lookback:i]
        if not prior:
            continue
        hi = max(prior)
        lo_ = min(prior)
        if series.c[i] > hi:
            sig[i] = 1
        elif series.c[i] < lo_:
            sig[i] = -1
    return sig


def ratio_trend_signal(a: Series, b: Series, lookback: int, thr: float) -> list:
    """Relative-strength breakout on the ratio series a/b.

    Persistent ratio trends are the exchange-token / alt-cycle story: when
    A outperforms B for a sustained stretch, the ratio itself trends. The
    breakout level again excludes the evaluating bar.
    """
    n = min(len(a), len(b))
    ratio = [a.c[i] / b.c[i] if b.c[i] > 0 else float("nan") for i in range(n)]
    sig = [0] * n
    for i in range(lookback + 1, n):
        r = ratio[i]
        if not (r == r):
            continue
        prior = ratio[i - lookback:i]
        prior = [x for x in prior if x == x]
        if len(prior) < lookback // 2:
            continue
        m = sum(prior) / len(prior)
        var = sum((x - m) ** 2 for x in prior) / len(prior)
        s = math.sqrt(max(var, 1e-18))
        z = (r - m) / s if s > 0 else 0.0
        if z > thr:
            sig[i] = 1
        elif z < -thr:
            sig[i] = -1
    return sig


def atr(series: Series, window: int = 24) -> list:
    n = len(series)
    out = [float("nan")] * n
    trs: list = []
    for i in range(1, n):
        tr = max(series.h[i] - series.l[i],
                 abs(series.h[i] - series.c[i - 1]),
                 abs(series.l[i] - series.c[i - 1]))
        trs.append(tr)
        if len(trs) > window:
            trs.pop(0)
        if len(trs) >= max(2, window // 2):
            out[i] = sum(trs) / len(trs)
    return out


def channel_breakout_signal(series: Series, lookback: int, atr_mult: float,
                            atr_win: int = 24) -> list:
    """Bollinger-style channel breakout sized in ATR units.

    Used for gold, where a multi-day hold is what makes friction (~18 bp)
    small relative to the move. Channel = SMA +/- atr_mult * ATR.
    """
    n = len(series)
    sig = [0] * n
    a = atr(series, atr_win)
    for i in range(lookback, n):
        if not (a[i] == a[i]):
            continue
        window = series.c[i - lookback:i]
        m = sum(window) / len(window)
        upper = m + atr_mult * a[i]
        lower = m - atr_mult * a[i]
        if series.c[i] > upper:
            sig[i] = 1
        elif series.c[i] < lower:
            sig[i] = -1
    return sig


def riskoff_gate(ref: Series, drawdown_win: int = 168, vol_win: int = 168,
                 dd_thr: float = -0.10, vol_pct: float = 0.75) -> list:
    """True while the crypto reference is in a risk-off stretch.

    Risk-off = BTC is in a material drawdown from its trailing high AND its
    realised volatility is in the top quartile of the trailing window. Both
    legs are causal (they look only at bars up to and including ``i``).
    """
    n = len(ref)
    rets = log_returns(ref)
    sd = rolling_std(rets, vol_win)
    out = [False] * n
    for i in range(drawdown_win, n):
        window = ref.c[i - drawdown_win:i + 1]
        peak = max(window)
        dd = ref.c[i] / peak - 1.0 if peak > 0 else 0.0
        if not (sd[i] == sd[i]):
            continue
        hist = [x for x in sd[i - vol_win:i + 1] if x == x]
        if len(hist) < 10:
            continue
        q = sorted(hist)[int(len(hist) * vol_pct)]
        out[i] = (dd <= dd_thr) and (sd[i] >= q)
    return out


def drawdown_recovery_signal(series: Series, lookback: int, thr: float) -> list:
    """Capitulation -> recovery trigger.

    Long when the asset has fallen ``thr`` from its trailing high and then
    closes back above the prior bar's high (first sign of a bid); short the
    mirror image. This is the volume-climax continuation idea expressed
    with price structure only, because the mirror carries no taker data
    for every asset.
    """
    n = len(series)
    sig = [0] * n
    for i in range(lookback + 1, n):
        peak = max(series.c[i - lookback:i])
        trough = min(series.c[i - lookback:i])
        if peak > 0 and series.c[i] / peak - 1.0 <= -thr and series.c[i] > series.h[i - 1]:
            sig[i] = 1
        elif trough > 0 and series.c[i] / trough - 1.0 >= thr and series.c[i] < series.l[i - 1]:
            sig[i] = -1
    return sig


# ---------------------------------------------------------------------------
# trade generation — unbiased next_bar fill, greedy non-overlapping
# ---------------------------------------------------------------------------
def vol_target_scales(
    series: Series,
    target_vol: float = 0.01,
    vol_window: int = 168,
    min_scale: float = 0.25,
    max_scale: float = 2.0,
) -> list:
    """Per-bar position scale for volatility targeting.

    ``scale = target_vol / realised_vol`` clipped to [min, max]. Computed
    from bars up to and including ``i`` only, so the size used for a trade
    filled at ``i+1`` is knowable at decision time. Sizing therefore cannot
    smuggle in future volatility information.
    """
    rets = log_returns(series)
    sd = rolling_std(rets, vol_window)
    out = []
    for i in range(len(series)):
        s_i = sd[i]
        if not (s_i == s_i) or s_i <= 0:
            out.append(1.0)
            continue
        out.append(min(max_scale, max(min_scale, target_vol / s_i)))
    return out


def make_trades(
    entry_series: Series,
    exit_series: Series,
    sig: list,
    hold: int,
    fold_of_bar: list,
    friction_bps: float,
    asset: str,
    vol_scale: list | None = None,
) -> list[Trade]:
    n = min(len(entry_series), len(exit_series))
    out: list[Trade] = []
    i = 0
    while i < n - hold - 1:
        d = sig[i]
        if d:
            fill = entry_series.o[i + 1]
            exit_px = exit_series.c[min(i + hold, n - 1)]
            if fill > 0 and exit_px > 0:
                # position size: 1.0 unless volatility targeting is on. Both
                # the move AND the friction scale with notional — a bigger
                # position pays a bigger round trip.
                size = vol_scale[i] if vol_scale is not None else 1.0
                pnl = size * (d * (exit_px / fill - 1.0) * 1e4 - friction_bps)
                out.append(
                    Trade(
                        entry_t=entry_series.t[i + 1],
                        exit_t=exit_series.t[min(i + hold, n - 1)],
                        fold=fold_of_bar[i],
                        pnl_bps=pnl,
                        direction=d,
                        asset=asset,
                    )
                )
            i += hold
        else:
            i += 1
    return out


def _basket_trades(legs: list, sig: list, hold: int, fold_of: list,
                   friction_bps: float) -> list[Trade]:
    """Equal-weight basket trade: one observation per impulse event.

    Averaging the legs BEFORE recording the trade is what keeps n honest —
    an ETH+BNB basket is one position, not two, and treating it as two is
    how correlation sneaks back in as extra significance.
    """
    n = min(len(leg) for leg in legs) if legs else 0
    out: list[Trade] = []
    i = 0
    while i < n - hold - 1:
        d = sig[i]
        if d:
            rets = []
            for leg in legs:
                fill = leg.o[i + 1]
                exit_px = leg.c[min(i + hold, n - 1)]
                if fill > 0 and exit_px > 0:
                    rets.append(d * (exit_px / fill - 1.0) * 1e4)
            if rets:
                out.append(Trade(
                    entry_t=legs[0].t[i + 1],
                    exit_t=legs[0].t[min(i + hold, n - 1)],
                    fold=fold_of[i],
                    pnl_bps=sum(rets) / len(rets) - friction_bps,
                    direction=d,
                    asset="BASKET",
                ))
            i += hold
        else:
            i += 1
    return out


def _monthly_trades(series: Series, sig: list, hold: int, fold_of: list,
                    friction_bps: float) -> list[Trade]:
    """Non-overlapping trades on a monthly grid.

    A monthly benchmark has no intra-month path, so the fill is the next
    month's price (equivalent to the next_bar open rule at this resolution)
    and the exit is ``hold`` months later. Nothing finer can be claimed.
    """
    n = len(series)
    out: list[Trade] = []
    i = 0
    while i < n - hold - 1:
        d = sig[i]
        if d:
            fill = series.o[i + 1]
            exit_px = series.c[min(i + hold, n - 1)]
            if fill > 0 and exit_px > 0:
                out.append(Trade(
                    entry_t=series.t[i + 1],
                    exit_t=series.t[min(i + hold, n - 1)],
                    fold=fold_of[i],
                    pnl_bps=d * (exit_px / fill - 1.0) * 1e4 - friction_bps,
                    direction=d,
                    asset="GOLD",
                ))
            i += hold
        else:
            i += 1
    return out


def fold_of_bar_times(times: list, fold_edges: list) -> list:
    out = []
    for x in times:
        f = 0
        for k in range(len(fold_edges) - 1):
            if fold_edges[k] <= x < fold_edges[k + 1]:
                f = k
                break
        else:
            f = len(fold_edges) - 2
        out.append(f)
    return out


def fold_edges_for(times: list, n_folds: int, end: int | None = None) -> list:
    """Fold boundaries spanning [first bar, end), research window only."""
    lo = times[0]
    hi = end if end is not None else times[-1] + 1
    hi = min(hi, times[-1] + 1)
    span = hi - lo
    return [lo + int(span * k / n_folds) for k in range(n_folds)] + [hi]


# ---------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    family: str
    asset: str
    params: dict

    @property
    def key(self) -> str:
        return f"{self.family}|{self.asset}|" + "|".join(
            f"{k}={v}" for k, v in sorted(self.params.items())
        )


def catchup_signals(
    lead_rets: list,
    lead_sd: list,
    n: int,
    L: int,
    thr: float,
    invert: bool,
    extra: list | None = None,
) -> list:
    imp = impulse(lead_rets, lead_sd, L)
    sig = [0] * n
    for i in range(n):
        z = imp[i]
        if not (z == z):
            continue
        if extra is not None and not extra[i]:
            continue
        if z > thr:
            sig[i] = -1 if invert else 1
        elif z < -thr:
            sig[i] = 1 if invert else -1
    return sig


def self_z_signals(rets: list, sd: list, n: int, L: int, thr: float, fade: bool) -> list:
    imp = impulse(rets, sd, L)
    sig = [0] * n
    for i in range(n):
        z = imp[i]
        if not (z == z):
            continue
        if abs(z) <= thr:
            continue
        direction = 1 if z > 0 else -1
        sig[i] = -direction if fade else direction
    return sig


def session_breakout_signals(
    series: Series, rets: list, sd: list, open_hour: int, close_hour: int, thr: float
) -> list:
    """Breakout of the London/NY session range, evaluated at session close.

    Session boundaries are UTC hours, so the trigger fires on the same
    wall-clock hour every day — a property of the instrument's liquidity
    cycle, not of the price path.
    """
    import datetime as _dt

    n = len(series)
    sig = [0] * n
    span = (close_hour - open_hour) % 24 or 24
    for i in range(span, n):
        hour = _dt.datetime.fromtimestamp(series.t[i], _dt.timezone.utc).hour
        if hour != close_hour:
            continue
        s = sd[i]
        if not (s == s) or s <= 0:
            continue
        start = i - span
        move = series.c[i] / series.o[start + 1] - 1.0 if series.o[start + 1] > 0 else 0.0
        z = move / (s * math.sqrt(span))
        if z > thr:
            sig[i] = 1
        elif z < -thr:
            sig[i] = -1
    return sig


def weekend_gap_signals(series: Series, ref: Series, thr: float, fade: bool) -> list:
    """Crypto-vs-gold weekend gap: ref (BTC) moves while PAXG sits still."""
    import datetime as _dt

    n = len(series)
    sig = [0] * n
    for i in range(1, n - 1):
        d = _dt.datetime.fromtimestamp(series.t[i], _dt.timezone.utc)
        if d.weekday() != 0 or d.hour != 0:  # Monday 00:00 UTC bar
            continue
        # find the last bar before the weekend (Friday 23:00 or later)
        j = i - 1
        while j > 0:
            dd = _dt.datetime.fromtimestamp(series.t[j], _dt.timezone.utc)
            if dd.weekday() == 4:
                break
            j -= 1
        if j <= 0:
            continue
        prev = ref.c[j]
        if prev <= 0:
            continue
        move = ref.c[i] / prev - 1.0
        if abs(move) < thr:
            continue
        direction = 1 if move > 0 else -1
        sig[j] = -direction if fade else direction
    return sig


# ---------------------------------------------------------------------------
# purged walk-forward selection
# ---------------------------------------------------------------------------
def purged_select(
    cand_trades: dict,
    fold_edges: list,
    hold_seconds: dict,
    min_train: int = 30,
) -> tuple[list[Trade], dict]:
    """For each fold, pick the candidate with the best TRAIN Net EV.

    Train = every trade that CLOSED strictly before the fold starts, minus
    an embargo of one full holding period (``hold_seconds``) so that no
    training trade can overlap the test window it is used to justify. The
    chosen candidate is then scored ONLY on that fold's trades — those
    out-of-fold trades are the sole evidence the gate ever sees.
    """
    n_folds = len(fold_edges) - 1
    oof: list[Trade] = []
    picks: dict[int, str] = {}
    for f in range(n_folds):
        start = fold_edges[f]
        best_key = None
        best_ev = float("-inf")
        for key, trades in cand_trades.items():
            embargo = hold_seconds.get(key, 3600)
            train = [t for t in trades if t.exit_t + embargo <= start]
            if len(train) < min_train:
                continue
            ev = sum(t.pnl_bps for t in train) / len(train)
            if ev > best_ev:
                best_ev, best_key = ev, key
        if best_key is None:
            picks[f] = "<no-train-window>"
            continue
        picks[f] = best_key
        for t in cand_trades[best_key]:
            if fold_edges[f] <= t.entry_t < fold_edges[f + 1]:
                oof.append(t)
    return oof, picks


def score_holdout(
    cand_trades: dict,
    modal_key: str,
    holdout_start: int,
    hold_seconds: dict,
) -> list[Trade]:
    """Score the frozen (modal) candidate on the locked holdout window.

    The holdout is the LAST fold of the gate's evidence and is never used
    to choose parameters: by the time this runs, ``modal_key`` is already
    fixed from the research window alone.
    """
    out: list[Trade] = []
    for t in cand_trades.get(modal_key, []):
        if t.entry_t >= holdout_start:
            out.append(t)
    return out


def modal_pick(picks: dict) -> str | None:
    counts: dict[str, int] = {}
    for v in picks.values():
        if v.startswith("<"):
            continue
        counts[v] = counts.get(v, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


# ---------------------------------------------------------------------------
# the scan
# ---------------------------------------------------------------------------
def scan(quick: bool = False, verbose: bool = True) -> dict:
    cand_trades: dict[str, list[Trade]] = {}
    hold_bars: dict[str, int] = {}
    meta: dict[str, dict] = {}
    failures: list[str] = []

    def add(cand: Candidate, trades: list[Trade], folds: list, bar_seconds: int):
        if not trades:
            return
        key = cand.key
        cand_trades[key] = trades
        # holding period expressed in SECONDS — the purge embargo must be
        # time-based, never bar-index based, so it is correct for 15m bars
        # and 1h bars alike
        hold_bars[key] = max(bar_seconds, int(cand.params.get("hold_h", 1)) * 3600)
        meta[key] = {
            "family": cand.family,
            "asset": cand.asset,
            "params": dict(cand.params),
            "bar_seconds": bar_seconds,
            "folds": len(folds) - 1,
        }

    # ---- family A: BTC impulse -> alt catch-up (perp 1h, 2024-2026) ----
    lead_perp = load_series("BTC", "perp_1h")
    if lead_perp is None:
        failures.append("missing BTC_perp_1h cache")
        lead_perp = None
    else:
        lead_rets = log_returns(lead_perp)
        lead_sd = rolling_std(lead_rets, 500)
        Ls = [12, 24, 48] if not quick else [24]
        Hs = [12, 24, 48] if not quick else [48]
        thrs = [1.5, 2.0, 2.5] if not quick else [2.0]
        for alt in ["ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "LINK"]:
            alt_series = load_series(alt, "perp_1h")
            if alt_series is None:
                failures.append(f"missing {alt}_perp_1h cache")
                continue
            lead_a, alt_a = align(lead_perp, alt_series)
            cut = holdout_start_for(lead_a.t)
            edges = fold_edges_for(lead_a.t, N_FOLDS, cut)
            fold_of = fold_of_bar_times(lead_a.t, edges)
            # BNB extra conditioning: BNB/BTC relative strength regime
            extra_rules = [(None, "plain")]
            if alt == "BNB":
                rel = [alt_a.c[i] / lead_a.c[i] if lead_a.c[i] > 0 else 0.0 for i in range(len(alt_a))]
                rel_z = [float("nan")] * len(rel)
                for i in range(168, len(rel)):
                    seg = rel[i - 168:i]
                    m = sum(seg) / len(seg)
                    var = sum((x - m) ** 2 for x in seg) / len(seg)
                    s = math.sqrt(max(var, 1e-18))
                    rel_z[i] = (rel[i] - m) / s if s > 0 else float("nan")
                extra_rules.append(([rel_z[i] < 0.0 for i in range(len(rel))], "bnb_weak_vs_btc"))
            for L in Ls:
                for thr in thrs:
                    for H in Hs:
                        for extra, tag in extra_rules:
                            params = {"L_h": L, "thr": thr, "hold_h": H, "cond": tag}
                            sig = catchup_signals(
                                lead_rets, lead_sd, len(lead_a), L, thr, False, extra
                            )
                            trades = make_trades(
                                alt_a, alt_a, sig, H, fold_of,
                                FRICTION_BPS.get(alt, 12.0), alt,
                            )
                            add(Candidate("catchup_btc_impulse", alt, params), trades, edges,
                                lead_a.t[1] - lead_a.t[0] if len(lead_a.t) > 1 else 3600)

    # ---- family B: sub-hour catch-up (BTC 15m/30m impulse -> ETH/BNB) ----
    for tf in (["spot_30m"] if quick else ["spot_15m", "spot_30m"]):
        lead = load_series("BTC", tf)
        if lead is None:
            failures.append(f"missing BTC_{tf} cache")
            continue
        bar_s = lead.t[1] - lead.t[0] if len(lead.t) > 1 else 900
        lead_rets = log_returns(lead)
        lead_sd = rolling_std(lead_rets, 500)
        Ls_h = [6, 12, 24] if not quick else [12]
        Hs_h = [12, 24, 48] if not quick else [48]
        thrs = [1.5, 2.0, 2.5] if not quick else [2.0]
        for alt in ["ETH", "BNB"]:
            alt_series = load_series(alt, tf)
            if alt_series is None:
                failures.append(f"missing {alt}_{tf} cache")
                continue
            lead_a, alt_a = align(lead, alt_series)
            cut = holdout_start_for(lead_a.t)
            edges = fold_edges_for(lead_a.t, N_FOLDS, cut)
            fold_of = fold_of_bar_times(lead_a.t, edges)
            for L_h in Ls_h:
                L = max(1, L_h * 3600 // bar_s)
                for H_h in Hs_h:
                    H = max(1, H_h * 3600 // bar_s)
                    for thr in thrs:
                        params = {"L_h": L_h, "thr": thr, "hold_h": H_h}
                        sig = catchup_signals(lead_rets, lead_sd, len(lead_a), L, thr, False)
                        trades = make_trades(
                            alt_a, alt_a, sig, H, fold_of,
                            FRICTION_BPS.get(alt, 12.0), alt,
                        )
                        key_len = len(lead_a) and len(alt_a)
                        add(Candidate(f"catchup_subhour_{tf}", alt, params), trades, edges,
                            bar_s if key_len else 900)

    # ---- family C: own-asset squeeze / volatility fade (perp 1h) ----
    if lead_perp is not None:
        for sym in ["BTC", "ETH", "BNB"]:
            s = load_series(sym, "perp_1h")
            if s is None:
                continue
            rets = log_returns(s)
            sd = rolling_std(rets, 500)
            cut = holdout_start_for(s.t)
            edges = fold_edges_for(s.t, N_FOLDS, cut)
            fold_of = fold_of_bar_times(s.t, edges)
            for L in ([6, 12, 24] if not quick else [12]):
                for thr in ([2.0, 2.5, 3.0] if not quick else [2.5]):
                    for H in ([6, 12, 24] if not quick else [12]):
                        for fade in ([True, False] if not quick else [True]):
                            params = {"L_h": L, "thr": thr, "hold_h": H,
                                      "mode": "fade" if fade else "squeeze"}
                            sig = self_z_signals(rets, sd, len(s), L, thr, fade)
                            trades = make_trades(
                                s, s, sig, H, fold_of,
                                FRICTION_BPS.get(sym, 12.0), sym,
                            )
                            add(Candidate("vol_z", sym, params), trades, edges,
                                s.t[1] - s.t[0])

    # ---- family D: PAXG macro/session + mean reversion ----
    paxg = load_series("PAXG", "perp_1h")
    btc = load_series("BTC", "perp_1h")
    if paxg is not None:
        paxg_a, btc_a = (align(paxg, btc) if btc is not None else (paxg, paxg))
        rets = log_returns(paxg_a)
        sd = rolling_std(rets, 500)
        cut = holdout_start_for(paxg_a.t)
        edges = fold_edges_for(paxg_a.t, N_FOLDS, cut)
        fold_of = fold_of_bar_times(paxg_a.t, edges)
        for open_h, close_h in ([(7, 10), (13, 16)] if not quick else [(7, 10)]):
            for thr in ([1.0, 1.5, 2.0] if not quick else [1.5]):
                for H in ([4, 8, 12] if not quick else [8]):
                    params = {"session": f"{open_h}-{close_h}", "thr": thr, "hold_h": H}
                    sig = session_breakout_signals(paxg_a, rets, sd, open_h, close_h, thr)
                    trades = make_trades(
                        paxg_a, paxg_a, sig, H, fold_of, FRICTION_BPS["PAXG"], "PAXG",
                    )
                    add(Candidate("paxg_session_breakout", "PAXG", params), trades, edges,
                        paxg_a.t[1] - paxg_a.t[0])
        if btc is not None:
            for thr in ([0.003, 0.01] if not quick else [0.003]):
                for H in ([12, 24] if not quick else [24]):
                    params = {"gap_thr": thr, "hold_h": H, "mode": "fade"}
                    sig = weekend_gap_signals(paxg_a, btc_a, thr, True)
                    trades = make_trades(
                        paxg_a, paxg_a, sig, H, fold_of, FRICTION_BPS["PAXG"], "PAXG",
                    )
                    add(Candidate("paxg_weekend_gap", "PAXG", params), trades, edges,
                        paxg_a.t[1] - paxg_a.t[0])
        for L in ([1, 2, 4] if not quick else [2]):
            for thr in ([2.0, 2.5, 3.0] if not quick else [2.5]):
                for H in ([4, 8, 12] if not quick else [8]):
                    params = {"L_h": L, "thr": thr, "hold_h": H, "mode": "fade"}
                    sig = self_z_signals(rets, sd, len(paxg_a), L, thr, True)
                    trades = make_trades(
                        paxg_a, paxg_a, sig, H, fold_of, FRICTION_BPS["PAXG"], "PAXG",
                    )
                    add(Candidate("paxg_vol_fade", "PAXG", params), trades, edges,
                        paxg_a.t[1] - paxg_a.t[0])

    # ---- family E: BTC macro trend / compression / capitulation ----
    # 9.25 years of spot tape: weekly-scale holds where sigma >> friction.
    btc_spot = load_series("BTC", "spot_1h")
    if btc_spot is not None:
        bar_s = btc_spot.t[1] - btc_spot.t[0] if len(btc_spot) > 1 else 3600
        cut = holdout_start_for(btc_spot.t)
        edges = fold_edges_for(btc_spot.t, N_FOLDS, cut)
        fold_of = fold_of_bar_times(btc_spot.t, edges)
        # Donchian / range breakout, with and without a compression gate
        for lb_h in ([48, 120, 168] if not quick else [120]):
            for hold_h in ([48, 120, 168] if not quick else [120]):
                for comp in ([False, True] if not quick else [False]):
                    lb = max(2, lb_h)
                    params = {"lookback_h": lb_h, "hold_h": hold_h,
                              "compression_gate": comp}
                    sig = donchian_breakout_signal(
                        btc_spot, lb, hold_h, bar_s, require_compression=comp)
                    trades = make_trades(btc_spot, btc_spot, sig, hold_h, fold_of,
                                         FRICTION_BPS["BTC"], "BTC")
                    add(Candidate("btc_donchian", "BTC", params), trades, edges, bar_s)
        # capitulation -> continuation
        for lb_h in ([168, 336] if not quick else [168]):
            for thr in ([0.15, 0.25] if not quick else [0.20]):
                for hold_h in ([48, 120] if not quick else [48]):
                    params = {"lookback_h": lb_h, "drawdown": thr, "hold_h": hold_h}
                    sig = drawdown_recovery_signal(btc_spot, lb_h, thr)
                    trades = make_trades(btc_spot, btc_spot, sig, hold_h, fold_of,
                                         FRICTION_BPS["BTC"], "BTC")
                    add(Candidate("btc_capitulation", "BTC", params), trades, edges, bar_s)

    # ---- family F: BNB relative-strength and range expansion ----
    bnb_spot = load_series("BNB", "spot_1h")
    btc_h = load_series("BTC", "spot_1h")
    eth_h = load_series("ETH", "spot_1h")
    if bnb_spot is not None:
        bar_s = bnb_spot.t[1] - bnb_spot.t[0] if len(bnb_spot) > 1 else 3600
        # ratio trend vs BTC and vs ETH
        for ref_name, ref in (("BTC", btc_h), ("ETH", eth_h)):
            if ref is None:
                continue
            bnb_a, ref_a = align(bnb_spot, ref)
            cut = holdout_start_for(bnb_a.t)
            edges = fold_edges_for(bnb_a.t, N_FOLDS, cut)
            fold_of = fold_of_bar_times(bnb_a.t, edges)
            for lb_h in ([120, 336] if not quick else [168]):
                for thr in ([1.0, 1.5] if not quick else [1.5]):
                    for hold_h in ([48, 120] if not quick else [120]):
                        params = {"ratio_vs": ref_name, "lookback_h": lb_h,
                                  "z": thr, "hold_h": hold_h}
                        sig = ratio_trend_signal(bnb_a, ref_a, lb_h, thr)
                        trades = make_trades(bnb_a, bnb_a, sig, hold_h, fold_of,
                                             FRICTION_BPS["BNB"], "BNB")
                        add(Candidate("bnb_ratio_trend", "BNB", params),
                            trades, edges, bar_s)
        # range expansion on the 4h grid, resolved from the 1h spot tape
        bnb_4h = load_series("BNB", "spot_4h")
        if bnb_4h is not None:
            bar4 = bnb_4h.t[1] - bnb_4h.t[0]
            cut = holdout_start_for(bnb_4h.t)
            edges = fold_edges_for(bnb_4h.t, N_FOLDS, cut)
            fold_of = fold_of_bar_times(bnb_4h.t, edges)
            for lb_h in ([120, 240] if not quick else [168]):
                for hold_h in ([48, 120] if not quick else [48]):
                    params = {"lookback_h": lb_h, "hold_h": hold_h}
                    sig = donchian_breakout_signal(bnb_4h, max(2, lb_h // 4), hold_h, bar4)
                    trades = make_trades(bnb_4h, bnb_4h, sig, hold_h, fold_of,
                                         FRICTION_BPS["BNB"], "BNB")
                    add(Candidate("bnb_range_expansion", "BNB", params),
                        trades, edges, bar4)

    # ---- family G: PAXG gold macro — multi-day trend & risk-off persistence ----
    paxg_spot = load_series("PAXG", "spot_1h")
    if paxg_spot is not None and btc_h is not None:
        paxg_a, btc_a = align(paxg_spot, btc_h)
        bar_s = paxg_a.t[1] - paxg_a.t[0] if len(paxg_a.t) > 1 else 3600
        cut = holdout_start_for(paxg_a.t)
        edges = fold_edges_for(paxg_a.t, N_FOLDS, cut)
        fold_of = fold_of_bar_times(paxg_a.t, edges)
        # ATR channel breakout with 3-10 day holds (friction 18 bp)
        for lb_h in ([72, 168, 240] if not quick else [168]):
            for atr_mult in ([0.5, 1.0] if not quick else [1.0]):
                for hold_h in ([72, 168] if not quick else [168]):
                    params = {"lookback_h": lb_h, "atr_mult": atr_mult, "hold_h": hold_h}
                    sig = channel_breakout_signal(paxg_a, lb_h, atr_mult)
                    trades = make_trades(paxg_a, paxg_a, sig, hold_h, fold_of,
                                         FRICTION_BPS["PAXG"], "PAXG")
                    add(Candidate("paxg_channel_breakout", "PAXG", params),
                        trades, edges, bar_s)
        # safe-haven: long PAXG while crypto is risk-off and PAXG trend is up
        riskoff = riskoff_gate(btc_a)
        paxg_rets = log_returns(paxg_a)
        for lb_h in ([72, 168] if not quick else [168]):
            for hold_h in ([72, 168] if not quick else [168]):
                params = {"lookback_h": lb_h, "hold_h": hold_h, "gate": "btc_riskoff"}
                mom = impulse(paxg_rets, rolling_std(paxg_rets, 500), lb_h)
                sig = [0] * len(paxg_a)
                for i in range(len(paxg_a)):
                    z = mom[i]
                    if not (z == z) or not riskoff[i]:
                        continue
                    if z > 0.5:
                        sig[i] = 1
                    elif z < -0.5:
                        sig[i] = -1
                trades = make_trades(paxg_a, paxg_a, sig, hold_h, fold_of,
                                     FRICTION_BPS["PAXG"], "PAXG")
                add(Candidate("paxg_riskoff_momentum", "PAXG", params),
                    trades, edges, bar_s)
        # long-only gold trend (gold is not shortable in the same way; test
        # the honest long-only variant separately)
        for lb_h in ([168, 336] if not quick else [168]):
            for hold_h in ([72, 168] if not quick else [168]):
                params = {"lookback_h": lb_h, "hold_h": hold_h, "side": "long_only"}
                sig = donchian_breakout_signal(paxg_a, lb_h, hold_h, bar_s)
                sig = [max(0, x) for x in sig]
                trades = make_trades(paxg_a, paxg_a, sig, hold_h, fold_of,
                                     FRICTION_BPS["PAXG"], "PAXG")
                add(Candidate("paxg_gold_trend_long", "PAXG", params), trades, edges, bar_s)

    # ---- family H: gold macro benchmark (monthly, 1833-2026) ----
    # PAXG's own perp history is 18 months; three years of monthly tape
    # cannot certify anything. This uses the long gold benchmark for the
    # SLOW macro physics and sizes holds in months, where 18 bp of friction
    # is irrelevant. It is a distinct sleeve family and is never wired to
    # the PAXG trigger — it is a macro overlay with its own name.
    gold = load_series("GOLD", "monthly")
    if gold is not None:
        # monthly grid: only the last ~40 years are relevant to modern gold
        start = int(datetime(1985, 1, 1, tzinfo=timezone.utc).timestamp())
        gold = Series("GOLD", "monthly",
                      *[[f[i] for i in range(len(gold)) if gold.t[i] >= start]
                        for f in (gold.t, gold.o, gold.h, gold.l, gold.c)])
        bar_s = 30 * 24 * 3600  # nominal month
        cut = holdout_start_for(gold.t)
        edges = fold_edges_for(gold.t, N_FOLDS, cut)
        fold_of = fold_of_bar_times(gold.t, edges)
        for lb_m in ([6, 12] if not quick else [12]):
            for atr_mult in ([0.5, 1.0] if not quick else [1.0]):
                for hold_m in ([3, 6] if not quick else [6]):
                    params = {"lookback_m": lb_m, "atr_mult": atr_mult, "hold_m": hold_m}
                    sig = channel_breakout_signal(gold, lb_m, atr_mult, atr_win=12)
                    trades = _monthly_trades(gold, sig, hold_m, fold_of, 18.0)
                    add(Candidate("gold_macro_channel", "GOLD", params), trades, edges, bar_s)
        for lb_m in ([12, 24] if not quick else [12]):
            for hold_m in ([3, 6] if not quick else [6]):
                params = {"lookback_m": lb_m, "hold_m": hold_m, "side": "long_only"}
                sig = donchian_breakout_signal(gold, lb_m, hold_m, bar_s)
                sig = [max(0, x) for x in sig]
                trades = _monthly_trades(gold, sig, hold_m, fold_of, 18.0)
                add(Candidate("gold_macro_trend_long", "GOLD", params), trades, edges, bar_s)

    # ---- family I: time-series momentum (TSMOM), 2-4 week lookback ----
    # The one anomaly with a large literature behind it (Moskowitz-Ooi-
    # Pedersen): each asset's own trailing return predicts its own future.
    # Tested on the 9-year spot tape, sized by volatility targeting, and
    # judged by the same gate as everything else. No threshold tuning: the
    # signal is the SIGN of the trailing return, so there is no free
    # parameter to fit beyond the lookback/hold pair.
    for sym in ["BTC", "ETH", "BNB"]:
        s_ts = load_series(sym, "spot_1h")
        if s_ts is None:
            continue
        bar_s = s_ts.t[1] - s_ts.t[0] if len(s_ts) > 1 else 3600
        cut = holdout_start_for(s_ts.t)
        edges = fold_edges_for(s_ts.t, N_FOLDS, cut)
        fold_of = fold_of_bar_times(s_ts.t, edges)
        rets = log_returns(s_ts)
        sd = rolling_std(rets, 500)
        vs = vol_target_scales(s_ts)
        for L_h in ([336, 720] if not quick else [720]):
            for hold_h in ([72, 168] if not quick else [168]):
                imp = impulse(rets, sd, L_h)
                sig = [0 if not (z == z) else (1 if z > 0 else -1) for z in imp]
                for sizing in ([False, True] if not quick else [True]):
                    params = {"lookback_h": L_h, "hold_h": hold_h,
                              "sizing": "vol_target" if sizing else "unit"}
                    trades = make_trades(
                        s_ts, s_ts, sig, hold_h, fold_of, FRICTION_BPS[sym], sym,
                        vol_scale=vs if sizing else None,
                    )
                    add(Candidate("tsmom", sym, params), trades, edges, bar_s)

    # ---- family J: cross-asset spillover matrix (15m / 30m) ----
    # The certified ETH sleeve is one cell of a whole matrix: a 12h impulse
    # in one large-cap tends to be partly repaid by the others over ~48h.
    # Every cell is measured separately so nobody can claim the effect is
    # specific to the pair that happened to pass.
    for tf in (["spot_15m"] if quick else ["spot_15m", "spot_30m"]):
        grid = {a: load_series(a, tf) for a in ("BTC", "ETH", "BNB")}
        if any(v is None for v in grid.values()):
            continue
        for lead_name in ("BTC", "ETH", "BNB"):
            for fol in ("BTC", "ETH", "BNB"):
                if fol == lead_name:
                    continue
                a, b = align(grid[lead_name], grid[fol])
                bar = a.t[1] - a.t[0]
                cut = holdout_start_for(a.t)
                edges = fold_edges_for(a.t, N_FOLDS, cut)
                fold_of = fold_of_bar_times(a.t, edges)
                lr = log_returns(a)
                sd = rolling_std(lr, 500)
                for L_h in ([12] if quick else [6, 12, 24]):
                    for hold_h in ([48] if quick else [24, 48]):
                        for thr in ([2.0] if quick else [2.0, 2.5]):
                            L = L_h * 3600 // bar
                            H = hold_h * 3600 // bar
                            params = {"lead": lead_name, "lookback_h": L_h,
                                      "thr": thr, "hold_h": hold_h}
                            sig = catchup_signals(lr, sd, len(a), L, thr, False)
                            trades = make_trades(b, b, sig, H, fold_of,
                                                 FRICTION_BPS[fol], fol)
                            add(Candidate(f"spillover_{tf}", fol, params),
                                trades, edges, bar)

    # ---- family K: pooled follower basket on a BTC impulse ----
    # ONE instrument (equal-weight ETH+BNB basket), so this is a genuinely
    # separate sleeve rather than a re-labelling of the ETH cell. Its n is
    # the number of impulse events, not events x assets.
    for tf in (["spot_30m"] if quick else ["spot_15m", "spot_30m"]):
        lead = load_series("BTC", tf)
        if lead is None:
            continue
        grid = {a: load_series(a, tf) for a in ("ETH", "BNB")}
        if any(v is None for v in grid.values()):
            continue
        a, eth_a = align(lead, grid["ETH"])
        _, bnb_a = align(lead, grid["BNB"])
        bar = a.t[1] - a.t[0]
        cut = holdout_start_for(a.t)
        edges = fold_edges_for(a.t, N_FOLDS, cut)
        fold_of = fold_of_bar_times(a.t, edges)
        lr = log_returns(a)
        sd = rolling_std(lr, 500)
        for L_h in ([12] if quick else [6, 12, 24]):
            for hold_h in ([48] if quick else [24, 48]):
                for thr in ([2.0] if quick else [1.5, 2.0, 2.5]):
                    L = L_h * 3600 // bar
                    H = hold_h * 3600 // bar
                    sig = catchup_signals(lr, sd, len(a), L, thr, False)
                    trades = _basket_trades(
                        [eth_a, bnb_a], sig, H, fold_of, FRICTION_BPS["ETH"])
                    params = {"basket": "ETH+BNB", "lookback_h": L_h,
                              "thr": thr, "hold_h": hold_h}
                    add(Candidate(f"basket_{tf}", "BASKET", params), trades, edges, bar)

    # ---- purged walk-forward selection per asset family ----
    tested = len(cand_trades)
    if verbose:
        print(f"candidates with trades: {tested}")

    results = []
    # group candidates per (family, asset) so the sleeve is one asset+family
    grouped: dict[tuple[str, str], list[str]] = {}
    for key, m in meta.items():
        grouped.setdefault((m["family"], m["asset"]), []).append(key)

    for (family, asset), keys in sorted(grouped.items()):
        bar_s = meta[keys[0]]["bar_seconds"] or 3600
        # fold calendar shared by every candidate in the group
        ref_times = sorted({t.entry_t for k in keys for t in cand_trades[k]})
        if not ref_times:
            continue
        cut = holdout_start_for(ref_times)
        edges = fold_edges_for(ref_times, N_FOLDS, cut)
        hold_seconds = {
            k: max(bar_s, int(meta[k]["params"].get("hold_h", 1)) * 3600) for k in keys
        }

        # --- purged walk-forward inside the research window only ---
        research = {
            k: [t for t in cand_trades[k] if t.entry_t < cut] for k in keys
        }
        oof, picks = purged_select(research, edges, hold_seconds)
        if not oof:
            continue
        frozen = modal_pick(picks)
        holdout = []
        if frozen is not None:
            holdout = score_holdout(cand_trades, frozen, cut, hold_seconds)

        block = max(1, int(round(24 * 3600 / bar_s)))  # one day of bars
        g = sleeve_gate(
            f"{family}::{asset}",
            oof,
            asset=asset,
            family=family,
            params={f"fold{k}": v for k, v in picks.items()},
            block=block,
            tested_candidates=tested,
            holdout_trades=holdout,
            details={
                "bar_seconds": bar_s,
                "candidates_in_group": len(keys),
                "frozen_candidate": frozen,
                "holdout_start": cut,
            },
        )
        results.append(g)
        if verbose:
            hv = g.details.get("holdout_net_ev_bps")
            print(
                f"{g.sleeve_id:34s} {g.status:7s} n={g.n:5d} EV={g.net_ev_bps:+7.2f} "
                f"PF={g.profit_factor:5.2f} t_iid={g.t_iid:+5.2f} "
                f"t_bb={g.t_block_bootstrap:+5.2f} folds+={g.positive_folds}/{len(g.folds)} "
                f"OOS={hv if hv is None else f'{hv:+.1f}'}"
            )

    payload = {
        "generated_at": int(__import__("time").time()),
        "n_folds": N_FOLDS,
        "candidates_tested": tested,
        "holdout_fraction": HOLDOUT_FRACTION,
        "failures": failures,
        "summary": summarise(results),
        "sleeves": [r.as_dict() for r in results],
    }
    return payload


def _write_md(payload: dict, path: str) -> None:
    """Render the scan into a table a human can audit (and falsify)."""
    s = payload["summary"]
    lines = [
        "# Edge scan — certified sleeves",
        "",
        "Generated by `python -m backtest.edge_scan`. Do not hand-edit the numbers.",
        "",
        f"- folds (purged walk-forward): **{payload['n_folds']}**",
        f"- candidates measured: **{payload['candidates_tested']}**",
        f"- locked holdout: **last {int(HOLDOUT_FRACTION * 100)}% of each asset's own tape**, never used for selection",
        f"- open: **{s['open']}** · shadow: **{s['shadow']}** · closed: **{s['closed']}** of {s['total']}",
        "",
        "## Gate",
        "",
        "A sleeve is `open` only when ALL hold, out of fold:",
        "",
        "1. OOF Net EV > 0 bps (friction charged on every trade)",
        "2. OOF profit factor > 1.25",
        "3. strict majority of folds positive",
        "4. correlation-adjusted |t| >= 2.0 (conservative of iid and block-bootstrap t)",
        "5. locked holdout Net EV > 0 with at least 25 trades",
        "",
        "Fill model: decision at bar `t` close, fill at bar `t+1` **open**, exit at bar",
        "`t+H` close. Round-trip friction 12 bps (BTC/ETH/BNB/alts), 18 bps (PAXG).",
        "",
        "## Sleeves",
        "",
        "| sleeve | status | n | OOF Net EV (bp) | PF | t (iid) | t (block) | folds + | holdout EV (bp) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for x in sorted(payload["sleeves"], key=lambda r: (r["status"] != "open", -r["net_ev_bps"])):
        ho = x["details"].get("holdout_net_ev_bps")
        lines.append(
            f"| `{x['sleeve_id']}` | {x['status']} | {x['n']} | {x['net_ev_bps']:+.2f} | "
            f"{x['profit_factor']:.2f} | {x['t_iid']:+.2f} | {x['t_block_bootstrap']:+.2f} | "
            f"{x['positive_folds']}/{len(x['folds'])} | "
            f"{'--' if ho is None else f'{ho:+.2f}'} |"
        )
    lines += ["", "## Why the rest did not pass", ""]
    for x in payload["sleeves"]:
        if x["status"] != "open":
            lines.append(f"- `{x['sleeve_id']}` — {'; '.join(x['reasons']) or 'no reason recorded'}")
    if payload.get("failures"):
        lines += ["", "## Data gaps", ""] + [f"- {f}" for f in payload["failures"]]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="smaller grid, same rules")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "EDGE_SCAN.json"))
    args = ap.parse_args()

    payload = scan(quick=args.quick)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    s = payload["summary"]
    print()
    print(f"open={s['open']} shadow={s['shadow']} closed={s['closed']} of {s['total']}")
    print(f"open sleeves: {s['open_sleeves']}")
    print(f"open assets : {s['open_assets']}")

    # human-readable companion table, committed next to the JSON
    md = os.path.join(os.path.dirname(args.out), "EDGE_SCAN.md")
    _write_md(payload, md)
    print(f"written: {args.out}")
    print(f"written: {md}")

    # also emit the committed registry the live engine reads
    registry = {
        "generated_at": payload["generated_at"],
        "n_folds": payload["n_folds"],
        "candidates_tested": payload["candidates_tested"],
        "summary": s,
        "sleeves": [x for x in payload["sleeves"] if x["status"] in (STATUS_OPEN, STATUS_SHADOW)],
    }
    with open(os.path.join(os.path.dirname(__file__), "..", "app", "sleeve_registry.json"), "w") as f:
        json.dump(registry, f, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
