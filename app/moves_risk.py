"""Measured-moves TP/SL (docs11-charts Part B, adapted).

Why this exists, from the user's live screenshots: TP1 sat BELOW the
entry for a BUY and levels barely moved window to window. Root causes:
levels were `entry ± conf·ATR·√min·0.5` (confidence ~0.02 → a 2 bp
target), and an emergency re-lock moved the entry while the targets
stayed anchored to the OLD price.

This module computes levels the way the advice specifies — from the
MEASURED excursions of this asset at this horizon, frozen at the lock:

  moves   = |window move| and max-adverse-excursion (MAE, per side)
            over overlapping windows of the 1-minute tape
  σ_bps   = realised 1m vol scaled to the horizon, CAPPED at
            q80(|move|)/1.2816 so one bad print cannot inflate it
  sl_bps  = q80(MAE for the trade's side)          (+ spread/2)
  tp_bps  = clip(1.5 · sl_bps · f(edge), q50, q95) (+ spread/2)
            f(edge) = 0.8 + 0.4·edge, rr target 1.5, never < 1.0
  levels  = entry · (1 ± bps/1e4) — above entry for BUY, below for SELL

Below n=5 measured windows the levels are PROVISIONAL (ATR fallback)
and the block says so. Nothing here recomputes mid-window: callers
freeze the returned block into the cycle/lock; an emergency re-lock
re-anchors the SAME bps distances at the new entry (reanchor()).
"""
from __future__ import annotations

import math
from typing import Any

RR_TARGET = 1.5
MIN_WINDOWS = 5


def _q(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, max(0, int(q * len(sorted_vals))))
    return sorted_vals[i]


def moves_stats(closes: list[float], highs: list[float],
                lows: list[float], window_min: int,
                step: int = 5) -> dict[str, Any]:
    """Excursion quantiles for `window_min`-minute windows over the
    1-minute tape (overlapping, `step` minutes apart)."""
    n_bars = len(closes)
    W = max(1, int(window_min))
    moves: list[float] = []
    mae_long: list[float] = []
    mae_short: list[float] = []
    if n_bars > W:
        for i in range(0, n_bars - W, max(1, step)):
            e = closes[i]
            if e <= 0:
                continue
            j = i + W
            moves.append(abs(closes[j] - e) / e * 1e4)
            seg_hi = max(highs[i:j + 1]) if highs else closes[j]
            seg_lo = min(lows[i:j + 1]) if lows else closes[j]
            mae_long.append(max(0.0, (e - seg_lo) / e * 1e4))
            mae_short.append(max(0.0, (seg_hi - e) / e * 1e4))
    moves.sort(); mae_long.sort(); mae_short.sort()
    # realised vol scaled to the horizon
    rets = [(closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, n_bars) if closes[i - 1] > 0]
    sigma_bps = 0.0
    if len(rets) >= 10:
        m = sum(rets) / len(rets)
        var = sum((r - m) ** 2 for r in rets) / len(rets)
        sigma_bps = math.sqrt(var) * math.sqrt(W) * 1e4
    q80 = _q(moves, 0.80)
    if q80 > 0:                      # C-7: one bad print cannot inflate σ
        sigma_bps = min(sigma_bps, q80 / 1.2816)
    return {"n": len(moves), "q50": _q(moves, 0.50), "q80": q80,
            "q95": _q(moves, 0.95),
            "mae_long_q80": _q(mae_long, 0.80),
            "mae_short_q80": _q(mae_short, 0.80),
            "sigma_bps": sigma_bps}


def risk_block(entry: float, direction: int, minutes: int, edge: float,
               closes: list[float], highs: list[float], lows: list[float],
               atr: float = 0.0, spread_bps: float = 0.0) -> dict[str, Any]:
    """Frozen TP/SL block for one horizon. `edge` in [0,1]."""
    s = 1 if direction >= 0 else -1
    st = moves_stats(closes, highs, lows, minutes)
    n = st["n"]
    half_spread = max(0.0, float(spread_bps)) / 2.0
    provisional = n < MIN_WINDOWS
    if provisional:
        # ATR fallback — exists at window 1, labelled provisional (B-1)
        base = (atr / entry * math.sqrt(max(1, minutes)) * 1e4
                if entry > 0 and atr > 0 else 10.0 * math.sqrt(max(1, minutes)))
        sl_bps = max(2.0, 0.8 * base)
        tp_bps = RR_TARGET * sl_bps
    else:
        mae = st["mae_long_q80"] if s > 0 else st["mae_short_q80"]
        sl_bps = max(1.0, mae)
        f_edge = 0.8 + 0.4 * min(1.0, max(0.0, float(edge)))
        tp_bps = min(max(RR_TARGET * sl_bps * f_edge, st["q50"]),
                     st["q95"]) if st["q95"] > 0 else RR_TARGET * sl_bps * f_edge
        if sl_bps > 0:
            tp_bps = max(tp_bps, sl_bps * 1.0)       # rr never below 1.0
    sl_bps += half_spread
    tp_bps += half_spread
    tp = entry * (1 + s * tp_bps / 1e4)
    sl = entry * (1 - s * sl_bps / 1e4)
    rr = (tp_bps / sl_bps) if sl_bps > 0 else 0.0
    note = (f"{tp_bps:.0f} bps target / {sl_bps:.0f} bps stop = {rr:.2f}:1 — "
            + (f"provisional (n={n} windows < {MIN_WINDOWS}, ATR fallback)"
               if provisional else
               f"q80 MAE of {n} windows (q50 {st['q50']:.0f} / "
               f"q95 {st['q95']:.0f} bps, σ {st['sigma_bps']:.0f} bps)"))
    return {"entry": round(entry, 6), "direction": s, "minutes": int(minutes),
            "tp": round(tp, 6), "sl": round(sl, 6),
            "tp_bps": round(tp_bps, 2), "sl_bps": round(sl_bps, 2),
            "rr": round(rr, 3), "n": n, "provisional": provisional,
            "sigma_bps": round(st["sigma_bps"], 2),
            "q50": round(st["q50"], 2), "q80": round(st["q80"], 2),
            "q95": round(st["q95"], 2), "note": note}


def reanchor(block: dict[str, Any], new_entry: float) -> dict[str, Any]:
    """Emergency re-lock: SAME measured bps distances at the new entry.
    This is the fix for TP ending up on the wrong side of a re-locked
    entry — distances are frozen, the anchor moves."""
    s = int(block.get("direction") or 1)
    out = dict(block)
    out["entry"] = round(new_entry, 6)
    out["tp"] = round(new_entry * (1 + s * float(block["tp_bps"]) / 1e4), 6)
    out["sl"] = round(new_entry * (1 - s * float(block["sl_bps"]) / 1e4), 6)
    out["reanchored"] = True
    return out
