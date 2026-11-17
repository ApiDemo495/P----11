"""Event-driven alpha: CUSUM sampling, cascade capture, honest execution.

Why this module exists
----------------------
The deep walk-forward (`backtest/results/ALPHA_LAB_DEEP.md`) proved that
*per-bar* prediction cannot pay for itself: ~51% sign accuracy against
15–27 bp of measured friction leaves net EV negative on all 14 lanes. Two
independent findings say the productive structure is not "predict every
bar" but "trade rare, path-defined events":

  * López de Prado (AFML ch. 3) and the meta-labeling literature: label
    EVENTS sampled by a CUSUM filter, not every bar — otherwise overlapping
    windows make the sample redundant and the target untradeable. Barrier
    widths must be σ-scaled and may be asymmetric (2σ up / 1σ down is the
    documented crypto case study).
  * An independent 2,160-strategy study on 1-minute crypto data (none beat
    fees) found exactly one survivor under every stress test: the **crash
    bid** — resting bids 2% below the last close that only fill during
    liquidation cascades, exited at +2% or at market after four hours.

The second finding also settles the execution question with evidence:
resting/maker entries are *adversely selected* for mean reversion (fills
happen when price keeps going against you; reported fill rates 19–48% and
8–14× worse per trip than market orders). So this module models **taker
execution on the next bar open** — the conservative assumption — and never
books a maker fill that a real book might not have given us.

What is simulated
-----------------
  trigger   CUSUM-style excursion: |z| ≥ `z_thr` where z is the k-bar
            return scaled by the lane's realised σ.
  entry     next bar's OPEN (no same-bar close execution), taker fee.
  exit      first touch of PT / SL (SL wins ties), else market at the
            vertical barrier (time stop), taker fee.
  costs     round-trip bps charged on every trade, configurable.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULT_FEES_BPS = 12.0


# ───────────────────────── triggers ─────────────────────────


def sigma_series(bars, window: int = 240) -> np.ndarray:
    """Rolling σ of 1-bar returns (causal), plus a horizon scaling hook."""
    c = np.asarray(bars.c, dtype=float)
    r = np.zeros_like(c)
    r[1:] = np.diff(c) / np.maximum(c[:-1], 1e-12)
    out = np.zeros_like(c)
    if c.size == 0:
        return out
    cs = np.cumsum(r)
    cs2 = np.cumsum(r * r)
    for i in range(c.size):
        a = max(0, i - window + 1)
        n = i - a + 1
        if n < 12:
            continue
        s = cs[i] - (cs[a - 1] if a > 0 else 0.0)
        s2 = cs2[i] - (cs2[a - 1] if a > 0 else 0.0)
        var = max(0.0, s2 / n - (s / n) ** 2)
        out[i] = math.sqrt(var)
    return out


def cusum_events(bars, k_bar: int = 5, z_thr: float = 3.0,
                 sigma: np.ndarray | None = None,
                 cooldown: int | None = None) -> list[dict]:
    """Bars where a k-bar excursion exceeded `z_thr` σ.

    Returns dicts with the bar index, the side to take (fade the move), and
    the measured z. A `cooldown` (defaults to the lookback) prevents
    overlapping triggers of the same cascade.
    """
    c = np.asarray(bars.c, dtype=float)
    if c.size < k_bar + 20:
        return []
    sig = sigma if sigma is not None else sigma_series(bars)
    cool = k_bar * 2 if cooldown is None else int(cooldown)
    out: list[dict] = []
    last = -10 ** 9
    for i in range(k_bar, c.size - 1):
        s = sig[i]
        if not (s > 1e-9):
            continue
        z = (c[i] / c[i - k_bar] - 1.0) / (s * math.sqrt(k_bar))
        if abs(z) < z_thr:
            continue
        if i - last < cool:
            continue
        out.append({"i": int(i), "side": -1 if z > 0 else 1,
                    "z": float(z), "sigma": float(s)})
        last = i
    return out


# ───────────────────────── execution ─────────────────────────


def simulate_event(bars, ev: dict, pt_mult: float, sl_mult: float,
                   horizon: int, fees_bps: float = DEFAULT_FEES_BPS) -> dict:
    """Settle one event with taker execution on the next bar's open."""
    o = np.asarray(bars.o, dtype=float)
    h = np.asarray(bars.h, dtype=float)
    l = np.asarray(bars.l, dtype=float)
    c = np.asarray(bars.c, dtype=float)
    i = int(ev["i"])
    j = i + 1
    if j >= c.size:
        return {"taken": False}
    entry = float(o[j])
    if not (entry > 0):
        return {"taken": False}
    side = int(ev["side"])
    sig_scale = float(ev["sigma"]) * math.sqrt(max(1, horizon))
    tp = entry * (1 + side * pt_mult * sig_scale)
    sl = entry * (1 - side * sl_mult * sig_scale)
    end = min(j + horizon, c.size - 1)
    fee = fees_bps / 1e4
    for t in range(j, end + 1):
        if side > 0:
            if l[t] <= sl:                       # SL wins ties (conservative)
                return {"taken": True, "exit": "sl", "bars": t - j,
                        "gross": sl / entry - 1.0, "net": sl / entry - 1.0 - fee,
                        "r": -(sl_mult * sig_scale) - fee}
            if h[t] >= tp:
                return {"taken": True, "exit": "tp", "bars": t - j,
                        "gross": tp / entry - 1.0, "net": tp / entry - 1.0 - fee,
                        "r": pt_mult * sig_scale - fee}
        else:
            if h[t] >= sl:
                return {"taken": True, "exit": "sl", "bars": t - j,
                        "gross": 1.0 - sl / entry, "net": 1.0 - sl / entry - fee,
                        "r": -(sl_mult * sig_scale) - fee}
            if l[t] <= tp:
                return {"taken": True, "exit": "tp", "bars": t - j,
                        "gross": 1.0 - tp / entry, "net": 1.0 - tp / entry - fee,
                        "r": pt_mult * sig_scale - fee}
    px = float(c[end])
    g = (px / entry - 1.0) * side
    return {"taken": True, "exit": "time", "bars": end - j,
            "gross": g, "net": g - fee, "r": g - fee}


def trade_stats(trades: list[dict]) -> dict:
    """EV, profit factor, Sharpe, Sortino, drawdown from settled trades."""
    r = np.asarray([t["net"] for t in trades if t.get("taken")], dtype=float)
    if r.size == 0:
        return {"n": 0}
    wins = r[r > 0]
    losses = r[r <= 0]
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    dd = float(np.max(peak - eq)) if eq.size else 0.0
    sd = float(np.std(r, ddof=1)) if r.size > 2 else float("nan")
    if math.isfinite(sd) and sd <= 1e-9:
        sd = float("nan")
    # overlap correction: clustered events on one price path are not
    # independent observations, so the t-stat uses the *effective* count
    held = np.asarray([max(1.0, t.get("bars", 1.0)) for t in trades
                       if t.get("taken")], dtype=float)
    starts = np.asarray(sorted(t["i"] for t in trades if t.get("taken")),
                        dtype=float)
    gap = float(np.median(np.diff(starts))) if starts.size > 1 else float(held.mean())
    neff = float(r.size) / max(1.0, float(held.mean()) / max(1.0, gap))
    t_stat = (float(np.mean(r)) / (sd / math.sqrt(neff))
              if sd and math.isfinite(sd) and sd > 0 and neff > 0
              else float("nan"))
    downside = r[r < 0]
    dsd = float(np.std(downside, ddof=1)) if downside.size > 2 else float("nan")
    if math.isfinite(dsd) and dsd <= 1e-9:
        dsd = float("nan")
    return {
        "n": int(r.size),
        "win_rate": float(wins.size / r.size),
        "ev_bp": float(np.mean(r) * 1e4),
        "gross_ev_bp": float(np.mean([t["gross"] for t in trades if t.get("taken")]) * 1e4),
        "profit_factor": (float(wins.sum() / abs(losses.sum()))
                          if losses.sum() < 0 else float("inf")),
        "sharpe": (float(np.mean(r) / sd * math.sqrt(r.size))
                   if sd and math.isfinite(sd) and sd > 0 else float("nan")),
        "sortino": (float(np.mean(r) / dsd * math.sqrt(r.size))
                    if dsd and math.isfinite(dsd) and dsd > 0 else float("nan")),
        "max_dd_R": dd,
        "neff": round(neff, 1),
        "t_stat": round(t_stat, 2),
        "avg_bars": float(np.mean([t["bars"] for t in trades if t.get("taken")])),
        "tp_share": float(np.mean([t["exit"] == "tp" for t in trades if t.get("taken")])),
    }
