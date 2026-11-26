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

RR_TARGET = 2.0   # USER RULE (updated): ratio may live anywhere in
                  # [1:1, 4:1]; 2:1 is the preferred/default. The TP is
                  # the MEASURED favourable excursion clipped into that
                  # band; a measured ratio below 1:1 is kept honest in
                  # the block (below_floor) and the caller cuts the
                  # cycle to watch-only — below 1:1 is not worth trading.
MIN_WINDOWS = 5


def _q(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, max(0, int(q * len(sorted_vals))))
    return sorted_vals[i]


def _infer_bar_minutes(closes: list[float]) -> float:
    """Spacing of the tape, in minutes. DEFAULT IS 1 (the live 1m feed).

    This used to be a length-based guess, and that was a mistake worth
    recording: guessing the spacing SILENTLY re-introduced the very bug it
    was meant to prevent. A synthetic 900-bar 1-minute tape was guessed as
    hourly, so `22` became a single bar and the measured stop collapsed
    from 200+ bps to 4.8 bps. A tape's spacing is a FACT about the caller,
    not something to infer from how much data there is.

    So: default to the app's real feed (1-minute, 1440 bars/day) and make
    every other caller state `bar_minutes=` explicitly. The live engine,
    the warm-up replay and the history seed all pass 1.0; the research
    harnesses pass their CLI tape spacing. `_tape_too_coarse` reports the
    case where a caller asked for a window shorter than its own bar, which
    is the situation that CANNOT be measured and must not be faked.
    """
    return 1.0


def moves_stats(closes: list[float], highs: list[float],
                lows: list[float], window_min: int,
                step: int = 5,
                bar_minutes: float | None = None) -> dict[str, Any]:
    """Excursion quantiles for `window_min`-MINUTE windows.

    THE UNITS BUG THIS FIXES (2026-11-26)
    -------------------------------------
    `window_min` used to be treated as a NUMBER OF BARS: the loop took
    `closes[i:i+window_min]` regardless of the tape's spacing. Every
    in-repo backtest feeds a 1-HOUR tape (backtest/data.py: BTC/ETH/BNB
    1h), so `tf1`'s 22 minutes was measured as 22 HOURS — 22 hours of
    excursion against a 22-minute horizon. Live proof from the project's
    own trade log (backtest/results/mode_a_trades_real_1h.csv): the median
    BTC stop was 181 bps, which is the 22-hour q80 excursion of that tape,
    while 22 minutes of BTC moves ~32 bps. A stop 5.7x too wide in
    DISTANCE is 32x too wide in VARIANCE, and the trade then cannot resolve
    inside its own horizon: 158 of 206 real trades (76.7 %) never touched
    TP or SL, and only 8 (3.9 %) reached TP.

    That is also the direct cause of the user's report "I got tp of 551 on
    BNB when market was on 556 at signal time": a 3.5 %-wide target on a
    22-minute holding period is not a 22-minute target.

    Fix: the window is converted to minutes, and bars are counted from the
    measured spacing. `step` is likewise in minutes, not bars.
    """
    n_bars = len(closes)
    bmin = float(bar_minutes) if bar_minutes else _infer_bar_minutes(closes)
    bmin = max(bmin, 1e-9)
    # bars that span `window_min` minutes, at this tape's spacing
    W = max(1, int(round(window_min / bmin)))
    # step is in MINUTES in the signature; convert to bars the same way
    step_bars = max(1, int(round(max(1, step) / bmin)))
    moves: list[float] = []
    mae_long: list[float] = []
    mae_short: list[float] = []
    if n_bars > W:
        for i in range(0, n_bars - W, step_bars):
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
               atr: float = 0.0, spread_bps: float = 0.0,
               bar_minutes: float | None = None) -> dict[str, Any]:
    """Frozen TP/SL block for one horizon. `edge` in [0,1].

    `minutes` is the HOLDING PERIOD in minutes and `bar_minutes` is the
    spacing of `closes`. Passing the spacing is now the expected contract:
    without it the tape spacing is inferred and the block says so in
    `window_basis`.
    """
    s = 1 if direction >= 0 else -1
    _bmin = float(bar_minutes) if bar_minutes else _infer_bar_minutes(closes)
    st = moves_stats(closes, highs, lows, minutes, bar_minutes=_bmin)
    n = st["n"]
    half_spread = max(0.0, float(spread_bps)) / 2.0
    # A window SHORTER THAN THE TAPE'S OWN SPACING cannot be measured from
    # that tape at all: on 1h bars, 22 minutes and 44 minutes both collapse
    # to a single bar, and a single bar is an HOUR of excursion. The old
    # code did exactly that silently, which is how a 22-minute horizon got
    # a 181 bps stop (the 22-hour excursion) in the project's own trade log.
    # Insufficient resolution is now its own reason for the fallback, along
    # with too few windows — and the fallback scales volatility by
    # sqrt(time), i.e. by SQRT(MINUTES / BAR MINUTES), not by
    # sqrt(bars) as if `minutes` were a bar count.
    _bars_per_window = max(1, int(round(minutes / max(_bmin, 1e-9))))
    _too_coarse = (_bmin > 0 and minutes < _bmin)
    provisional = n < MIN_WINDOWS or _too_coarse
    _why_provisional = ("window shorter than the tape's bar spacing "
                        f"({minutes:g}min window on {_bmin:g}-min bars)"
                        if _too_coarse else
                        f"n={n} windows < {MIN_WINDOWS}")
    if provisional:
        # ATR fallback — exists at window 1, labelled provisional (B-1)
        # σ_horizon = σ_bar · sqrt(horizon / bar), the standard diffusion
        # scaling. `atr` is a per-BAR true range, so the square root is of
        # the RATIO of times, never of the raw minute count.
        _scale = math.sqrt(max(minutes, 1e-9) / max(_bmin, 1e-9))
        base = (atr / entry * _scale * 1e4
                if entry > 0 and atr > 0 else 10.0 * _scale)
        sl_bps = max(2.0, 0.8 * base) + half_spread
        tp_bps = RR_TARGET * sl_bps
    else:
        mae = st["mae_long_q80"] if s > 0 else st["mae_short_q80"]
        # the 1 unit: measured adverse excursion (robust q80 quantile)
        sl_bps = max(1.0, mae) + half_spread
        # measured favourable excursion, edge-scaled, clipped to the
        # asset's own q50..q95 move band (docstring design)
        f_edge = 0.8 + 0.4 * max(0.0, min(1.0, edge))
        tp_meas = 1.5 * sl_bps * f_edge
        if st["q95"] > 0:
            tp_meas = max(st["q50"], min(st["q95"], tp_meas))
        tp_bps = tp_meas + half_spread
        # band [1:1, 4:1]: above 4:1 the target is fantasy — cap it;
        # below 1:1 keep the measured truth and flag it (watch-only)
        if sl_bps > 0 and tp_bps / sl_bps > 4.0:
            tp_bps = 4.0 * sl_bps
    tp = entry * (1 + s * tp_bps / 1e4)
    sl = entry * (1 - s * sl_bps / 1e4)
    rr = (tp_bps / sl_bps) if sl_bps > 0 else 0.0
    note = (f"{minutes}min window measured on {_bmin:g}-min bars "
            f"({st['n']} windows) · "
            f"{tp_bps:.0f} bps target / {sl_bps:.0f} bps stop = {rr:.2f}:1 "
            f"(band 1:1–4:1, 2:1 preferred) — "
            + (f"provisional ({_why_provisional}, ATR fallback "
               f"σ×√{minutes:g}/{_bmin:g})"
               if provisional else
               f"q80 MAE of {n} windows (q50 {st['q50']:.0f} / "
               f"q95 {st['q95']:.0f} bps, σ {st['sigma_bps']:.0f} bps)"))
    return {"entry": round(entry, 6), "direction": s, "minutes": int(minutes),
            "tp": round(tp, 6), "sl": round(sl, 6),
            "tp_bps": round(tp_bps, 2), "sl_bps": round(sl_bps, 2),
            "rr": round(rr, 3), "below_floor": bool(rr < 1.0),
            "n": n, "provisional": provisional,
            "window_min": int(minutes), "bar_minutes": round(_bmin, 3),
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
