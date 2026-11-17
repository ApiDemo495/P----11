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
# trade generation — unbiased next_bar fill, greedy non-overlapping
# ---------------------------------------------------------------------------
def make_trades(
    entry_series: Series,
    exit_series: Series,
    sig: list,
    hold: int,
    fold_of_bar: list,
    friction_bps: float,
    asset: str,
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
                pnl = d * (exit_px / fill - 1.0) * 1e4 - friction_bps
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
