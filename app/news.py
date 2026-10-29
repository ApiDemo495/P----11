"""News impact system.

PART 1 STATUS: the aggregation contract required by Section 1 is live:
per-direction impact aggregates with time decay, the 'News Impact Aggregate'
used by the Extreme Override (>0.8 opposing direction), and the news input to
Formula 28.  A seeded local headline store stands in until part 2 defines the
production news pipeline.
"""
from __future__ import annotations

import time

from .config import SYMBOL_ORDER

# Seeded local headlines (direction: +1 bullish / -1 bearish for the asset).
_SEED_HEADLINES: dict[str, list[tuple[str, float, int, float]]] = {
    # (headline, base_impact, direction, age_minutes_at_start)
    "BTC": [
        ("Spot ETF net inflows accelerate for a fourth session", 0.55, 1, 12),
        ("Miners rotate treasury reserves to exchanges", 0.35, -1, 40),
    ],
    "PAXG": [
        ("Real yields slip after soft labor print", 0.45, 1, 25),
        ("Central bank gold purchases slow QoQ", 0.40, -1, 70),
    ],
    "ETH": [
        ("L2 settlement volume hits monthly high", 0.42, 1, 18),
        ("Large validator queue exits observed", 0.38, -1, 55),
    ],
    "BNB": [
        ("Launchpool listing sparks platform demand", 0.44, 1, 30),
        ("Regulatory hearing scheduled next week", 0.50, -1, 90),
    ],
}


class NewsSystem:
    def __init__(self, now: float | None = None) -> None:
        self.ready = False
        self._t0 = now if now is not None else time.time()
        self.items: dict[str, list[dict]] = {}

    def initialize(self) -> None:
        self.ready = True
        self._seed()

    def _seed(self) -> None:
        now = time.time()
        for sym in SYMBOL_ORDER:
            self.items[sym] = [
                {
                    "headline": h,
                    "base_impact": imp,
                    "direction": d,
                    "ts": now - age * 60.0,
                }
                for (h, imp, d, age) in _SEED_HEADLINES.get(sym, [])
            ]

    def refresh(self) -> None:
        """Recomputed at each GRC; part 2 replaces this with the live feed."""
        if not self.items:
            self._seed()

    @staticmethod
    def _decay(age_seconds: float) -> float:
        return pow(2.718281828, -0.05 * max(0.0, age_seconds) / 60.0)

    def aggregate(self, symbol: str) -> dict:
        """Per-direction News Impact Aggregate in [0,1] with time decay."""
        now = time.time()
        bull = 0.0
        bear = 0.0
        for it in self.items.get(symbol, []):
            impact = it["base_impact"] * self._decay(now - it["ts"])
            if it["direction"] >= 0:
                bull += impact
            else:
                bear += impact
        bull = min(1.0, bull)
        bear = min(1.0, bear)
        opposing_of = {"bull": bear, "bear": bull}
        return {
            "bull": round(bull, 4),
            "bear": round(bear, 4),
            "polarity": 1 if bull > bear else (-1 if bear > bull else 0),
            "items": [
                {
                    "headline": it["headline"],
                    "direction": it["direction"],
                    "impact": round(
                        it["base_impact"] * self._decay(now - it["ts"]), 4
                    ),
                }
                for it in self.items.get(symbol, [])
            ],
            "_opposing": opposing_of,
        }

    def impact_against(self, symbol: str, direction: int) -> float:
        """News Impact Aggregate opposing `direction` (used by Extreme Override)."""
        agg = self.aggregate(symbol)
        if direction >= 0:
            return agg["bear"]
        return agg["bull"]

    def agrees_with(self, symbol: str, direction: int) -> bool:
        """For Formula 28: neutral (both near zero) counts as agreement."""
        agg = self.aggregate(symbol)
        if agg["bull"] < 0.05 and agg["bear"] < 0.05:
            return True
        dominant = 1 if agg["bull"] >= agg["bear"] else -1
        return dominant == direction or max(agg["bull"], agg["bear"]) < 0.15

    def status(self) -> dict:
        return {"ready": self.ready, "items": sum(len(v) for v in self.items.values())}
