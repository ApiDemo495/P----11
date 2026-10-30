"""Section 8 — Hedge Fund and Dark Pool formulas.

HF1  IFR  Institutional Flow Reconstruction  in [-1, +1]
HF2  DPAD Dark Pool Accumulation/Distribution in [-1, +1]
HF3  SMDI Smart Money Divergence Index      in [-1, +1]

The three feed the CCI through ONE combined component (id 101) whose initial
weight is exactly 0.08 — Section 8: "Their initial combined weight is 0.08
(8% of total CCI)".  The combination is the plain mean of the three bounded
signals (the spec fixes the combined weight but not the mixer; mean keeps the
result inside [-1, +1] and is documented in the README).
"""
from __future__ import annotations

import math

from .models import FormulaContext

EPS = 1e-12


def institutional_flow(ctx: FormulaContext) -> dict:
    """HF1 IFR.

    A minute is *institutional* when V(i) > 3 * avg(V) AND the price move is
    below 1 * avg(move) (large volume, controlled impact = algorithmic
    execution).  Classification: close > open with the close in the upper 30%
    of the range -> buy; close < open with the close in the lower 30% -> sell.
    V_inst(i) = V(i) * (1 - |close - open| / (high - low + eps)) with
    exp(-0.05 * age) recency weights.
    """
    n = min(len(ctx.P), len(ctx.V))
    out = {"value": 0.0, "buy": 0.0, "sell": 0.0, "total": 0.0, "events": 0}
    if n < 10:
        return out
    P, V = ctx.P[-n:], ctx.V[-n:]
    O, H, L = ctx.candles()
    O, H, L = O[-n:], H[-n:], L[-n:]
    vols = [v for v in V[:-1]]
    avg_v = sum(vols) / len(vols) if vols else 0.0
    moves = [abs(P[i] - P[i - 1]) for i in range(1, len(P))]
    avg_move = sum(moves) / len(moves) if moves else 0.0
    if avg_v <= 0:
        return out

    buy = sell = 0.0
    events = 0
    for i in range(1, len(P)):
        if V[i] <= 3.0 * avg_v or abs(P[i] - P[i - 1]) >= 1.0 * avg_move:
            continue
        rng = H[i] - L[i]
        if rng <= EPS:
            continue
        v_inst = V[i] * (1.0 - abs(P[i] - O[i]) / (rng + EPS))
        upper = P[i] >= L[i] + 0.7 * rng      # close in upper 30% of range
        lower = P[i] <= L[i] + 0.3 * rng      # close in lower 30% of range
        age = (len(P) - 1 - i)                # minutes ago
        w = math.exp(-0.05 * age)
        if P[i] > O[i] and upper:
            buy += v_inst * w
            events += 1
        elif P[i] < O[i] and lower:
            sell += v_inst * w
            events += 1
    total = buy + sell
    out.update({
        "value": math.tanh((buy - sell) / (total + EPS)),
        "buy": buy, "sell": sell, "total": total, "events": events,
    })
    return out


def dark_pool_flow(ctx: FormulaContext) -> dict:
    """HF2 DPAD.

    Dark-pool print: V(i) > 5 * avg(V) AND |r(i)| < 0.3 * sigma_returns.
    Classification uses the NEXT 3 minutes: cumulative return > 0 ->
    accumulation, < 0 -> distribution.  A_dark / D_dark = count * average
    volume of those events.
    """
    n = min(len(ctx.P), len(ctx.V))
    out = {"value": 0.0, "accumulation": 0.0, "distribution": 0.0,
           "prints": 0}
    if n < 12:
        return out
    P, V = ctx.P[-n:], ctx.V[-n:]
    vols = V[:-1]
    avg_v = sum(vols) / len(vols) if vols else 0.0
    rets = [(P[i] - P[i - 1]) / P[i - 1] for i in range(1, len(P))
            if abs(P[i - 1]) > EPS]
    if len(rets) < 6:
        return out
    mu = sum(rets) / len(rets)
    sigma = math.sqrt(sum((r - mu) ** 2 for r in rets) / len(rets))
    if avg_v <= 0 or sigma <= EPS:
        return out

    acc_vols: list[float] = []
    dist_vols: list[float] = []
    # rets[i] is the move into minute i+1 of P/V indexing
    for i in range(1, len(P) - 3):
        r = (P[i] - P[i - 1]) / P[i - 1] if abs(P[i - 1]) > EPS else 0.0
        if not (V[i] > 5.0 * avg_v and abs(r) < 0.3 * sigma):
            continue
        # next 3 minutes net buying / selling
        if P[i + 3] > P[i]:
            acc_vols.append(V[i])
        elif P[i + 3] < P[i]:
            dist_vols.append(V[i])

    a_dark = (len(acc_vols) * (sum(acc_vols) / len(acc_vols))
              if acc_vols else 0.0)
    d_dark = (len(dist_vols) * (sum(dist_vols) / len(dist_vols))
              if dist_vols else 0.0)
    out.update({
        "value": math.tanh((a_dark - d_dark) / (a_dark + d_dark + EPS)),
        "accumulation": a_dark, "distribution": d_dark,
        "prints": len(acc_vols) + len(dist_vols),
    })
    return out


def smart_money_divergence(ifr: float, h_signal_avg: float) -> float:
    """HF3 SMDI = tanh((IFR - retail_flow) / 2) where retail_flow is the
    NEGATIVE of the average brain hormone signal (the brain simulates retail
    emotions and its hormone signal is contrarian)."""
    retail_flow = -h_signal_avg
    return math.tanh((ifr - retail_flow) / 2.0)


def combined_signal(ifr: float, dpad: float, smdi: float) -> float:
    """Single CCI component (id 101) = mean of the three bounded signals."""
    return max(-1.0, min(1.0, (ifr + dpad + smdi) / 3.0))
