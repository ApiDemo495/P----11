"""The app's own prediction logic, adapted to a slice of history.

Direction: app/engine.py PredictionEngine.compute_asset (CCI fusion
+ direction ladder — the same code that issues live calls), run on a
MarketFeed whose tape is the slice. Levels: app/moves_risk.risk_block
(measured q80-MAE stop, TP = 2x stop — the app's frozen 2:1 block).
The series' bars are treated as 1-minute bars, matching the app's
native tape; the horizon is the app's TF1 window (22 bars).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .slice import SliceView

HORIZON = 22          # bars (app TF1 window)


@dataclass
class Signal:
    direction: int    # +1 long / -1 short
    entry: float
    sl: float
    tp: float
    confidence: float
    tier: str = ""


class AppPredictor:
    def __init__(self, tf: str = "tf1"):
        from app.engine import PredictionEngine
        from app.feed import MarketFeed

        async def _emit(_msg):
            pass

        self.feed = MarketFeed()
        self.engine = PredictionEngine(self.feed, _emit)
        self.tf = tf
        self._booted = False

    def boot(self) -> None:
        async def _b():
            await self.engine.start()
            for _ in range(200):
                if self.engine.phase == "live":
                    break
                await asyncio.sleep(0.05)
        asyncio.run(_b())
        self._booted = True

    def predict(self, view: SliceView, symbol: str) -> Signal | None:
        if not self._booted:
            self.boot()
        st = self.feed.state(symbol)
        bars = view.bars
        st.times = [b.t for b in bars]
        st.opens = [b.o for b in bars]
        st.highs = [b.h for b in bars]
        st.lows = [b.l for b in bars]
        st.closes = [b.c for b in bars]
        st.volumes = [b.v for b in bars]
        st.sources = ["backtest"] * len(bars)
        st.live_price = bars[-1].c
        st.live_open, st.live_high = bars[-1].o, bars[-1].h
        st.live_low = bars[-1].l
        st.day_open = bars[0].o
        res = self.engine.compute_asset(symbol, record=False,
                                        display=False)[self.tf]
        direction = int(res.get("direction") or 0)
        if direction == 0:
            return None
        from app.moves_risk import risk_block
        entry = bars[-1].c
        blk = risk_block(entry, direction, HORIZON,
                         float(res.get("confidence") or 0.0),
                         st.closes, st.highs, st.lows,
                         atr=float(res.get("atr") or 0.0))
        if blk.get("tp") is None or blk.get("sl") is None:
            return None
        return Signal(direction, entry, float(blk["sl"]), float(blk["tp"]),
                      float(res.get("confidence") or 0.0),
                      str(res.get("tier") or ""))
