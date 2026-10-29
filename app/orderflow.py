"""Order-flow seams reserved for Part 3 (Hedge Fund & Dark Pool formulas).

The specification's Part-2 call sites reference three order-flow quantities
whose exact definitions arrive with Part 3:

  * OFTI  — Order Flow Toxicity Index (F29 Phi_sl, E19 SMDM context)
  * CMP   — cross-asset momentum divergence (E19 SMDM)
  * news sentiment acceleration / price-narrative correlation (E20 NMS)

Until then this module provides clearly-labelled provisional proxies with the
same bounded ranges the exact formulas will occupy.  No other module defines
order-flow logic — Part 3 replaces exactly these three functions.
"""
from __future__ import annotations

import math


def ofti(
    closes: list[float],
    volumes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> float:
    """Provisional Order Flow Toxicity (VPIN-style bucketed imbalance) in [0, 1].

    Each minute's volume is split by the close's position inside its range;
    toxicity = mean over buckets of |V_buy - V_sell| / V_total.
    """
    n = min(len(closes), len(volumes))
    if n < 6:
        return 0.0
    closes = closes[-n:]
    volumes = volumes[-n:]
    if highs and lows and len(highs) >= n and len(lows) >= n:
        highs, lows = highs[-n:], lows[-n:]
    else:
        highs = [max(closes[i - 1], closes[i]) for i in range(n)]
        lows = [min(closes[i - 1], closes[i]) for i in range(n)]

    buckets = 10
    per_bucket = max(2, n // buckets)
    toxicities: list[float] = []
    for b in range(buckets):
        lo_i = b * per_bucket
        hi_i = min(n, lo_i + per_bucket)
        if hi_i - lo_i < 2:
            continue
        buy = sell = total = 0.0
        for i in range(lo_i, hi_i):
            rng = highs[i] - lows[i]
            pos = (closes[i] - lows[i]) / rng if rng > 0 else 0.5
            vol = volumes[i]
            buy += pos * vol
            sell += (1.0 - pos) * vol
            total += vol
        if total > 0:
            toxicities.append(abs(buy - sell) / total)
    if not toxicities:
        return 0.0
    return max(0.0, min(1.0, sum(toxicities) / len(toxicities)))


def cmp_divergence(peer_returns: list[list[float]]) -> float:
    """Provisional cross-asset momentum divergence in [0, 1].

    Mean over recent minutes of the fraction of asset pairs whose 1-minute
    returns disagree in sign (1 = fully divergent, 0 = fully in lockstep).
    """
    if not peer_returns:
        return 0.0
    fracs: list[float] = []
    for row in peer_returns:
        vals = [v for v in row if v != 0.0]
        if len(vals) < 2:
            continue
        same = total = 0
        for a in range(len(vals)):
            for b in range(a + 1, len(vals)):
                total += 1
                if (vals[a] > 0) == (vals[b] > 0):
                    same += 1
        if total:
            fracs.append(1.0 - same / total)
    if not fracs:
        return 0.0
    return max(0.0, min(1.0, sum(fracs) / len(fracs)))


def news_price_corr(
    price_returns: list[float], polarity_series: list[float]
) -> float:
    """Provisional price-narrative correlation in [-1, 1] (E20 seam)."""
    n = min(len(price_returns), len(polarity_series))
    if n < 4:
        return 0.0
    x = price_returns[-n:]
    y = polarity_series[-n:]
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx <= 1e-12 or dy <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, num / (dx * dy)))
