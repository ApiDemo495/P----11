"""Hedge-Flow Tracer (component id 102) — real-time rotation effect.

User request: 'a new hedge fund tracer with a weightage for determining
real time effect in the market'.

MEASURED inputs only (no synthetic flows):
  * volume-share rotation — each asset's share of the last30 minutes of
    tape volume vs its OWN previous30-minute baseline (where is the
    money rotating right now?)
  * relative return — this asset's30-minute return minus the reference
    (BTC) return (is this leg out/under-performing the market?)

    value = clip(tanh(K_ROT · rot + K_REL · rel), −1, +1)

  Scaling constants (K_ROT=4, K_REL=25) only map typical magnitudes
  (rot≈±0.2, rel≈±0.02) into tanh's useful range — the CONSTANT that
  matters is the CCI weightage, which the existing micro-learning earns
  and hindsight grades like every other component.

Returns (value, detail); on a thin tape it returns0 with a reason.
"""
from __future__ import annotations

from typing import Any

K_ROT = 4.0          # volume-share rotation gain (±0.2 → saturating)
K_REL = 25.0         # relative-return gain (±0.02 → ~0.4)
MIN_BARS = 61        # recent30 + baseline30 +1


def hedge_flow_tracer(feed: Any, symbol: str,
                      ref: str = "BTC") -> tuple[float, dict]:
    try:
        states = getattr(feed, "states", None) or {}
        if symbol not in states:
            return 0.0, {"reason": "no tape"}
        my = states[symbol]
        n = len(my.closes)
        if n < MIN_BARS:
            return 0.0, {"reason": f"needs ≥{MIN_BARS} minute bars, had {n}"}
        # volume shares: recent30 vs previous30 across all assets
        tot_rec = tot_base = 0.0
        rec: dict[str, float] = {}
        base: dict[str, float] = {}
        for s2, st in states.items():
            v = list(st.volumes)
            if len(v) < MIN_BARS - 1:
                continue
            r30 = float(sum(v[-30:]))
            b30 = float(sum(v[-60:-30]))
            rec[s2] = r30
            base[s2] = b30
            tot_rec += r30
            tot_base += b30
        if tot_rec <= 0 or tot_base <= 0:
            return 0.0, {"reason": "no volume on the tape"}
        share_now = rec.get(symbol, 0.0) / tot_rec
        share_base = base.get(symbol, 0.0) / tot_base
        rot = (share_now - share_base) / (share_base + 1e-9)
        # relative return vs reference over the same30-minute span
        def ret(s2: str) -> float:
            c = list(states[s2].closes)
            if len(c) < 31 or c[-31] == 0:
                return 0.0
            return c[-1] / c[-31] - 1.0
        rel = ret(symbol) - (ret(ref) if symbol != ref else 0.0)
        import math
        value = max(-1.0, min(1.0, math.tanh(K_ROT * rot + K_REL * rel)))
        return round(value, 6), {
            "share_now": round(share_now, 6),
            "share_base": round(share_base, 6),
            "rot": round(rot, 6),
            "rel_ret": round(rel, 6),
            "bars": n,
        }
    except Exception as exc:
        return 0.0, {"reason": f"error: {exc}"}
