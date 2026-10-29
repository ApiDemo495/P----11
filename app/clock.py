"""World Clock Alignment System — the master clock.

The global clock is divided into 4-minute intervals: 15 Global Recheck Cycles
(GRC) per hour, firing at minutes :00, :04, :08, ... :56.  At each GRC the
entire system performs a complete recalculation of every formula, brain state,
emotion calculation, news impact score, weightage and prediction across all
four timeframes and all four assets simultaneously.

Between GRCs, live price data keeps streaming and the UI updates visually,
but no prediction values change until the next GRC fires (event-driven
exceptions defined by the spec — sub-target hit locks, SL protocol, cycle
rollover, manual override — are handled by the engine, not the clock).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

from .config import GRC_INTERVAL_SECONDS

log = logging.getLogger("clock")


def next_grc_ts(now: float | None = None) -> float:
    """Epoch seconds of the next GRC boundary (:00, :04, :08, ...)."""
    now = time.time() if now is None else now
    nxt = (int(now) // GRC_INTERVAL_SECONDS + 1) * GRC_INTERVAL_SECONDS
    return float(nxt)


def seconds_to_next_grc(now: float | None = None) -> float:
    now = time.time() if now is None else now
    return max(0.0, next_grc_ts(now) - now)


def grc_id(ts: float | None = None) -> int:
    """Monotonic GRC counter (sequence number since epoch)."""
    ts = time.time() if ts is None else ts
    return int(ts) // GRC_INTERVAL_SECONDS


class MasterClock:
    """Fires `on_grc(grc_ts)` exactly at every 4-minute boundary."""

    def __init__(self, on_grc: Callable[[float], Awaitable[None]]):
        self.on_grc = on_grc
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self.last_fired: float = 0.0
        self.fired_count: int = 0

    async def start(self) -> None:
        self._stop.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        # If we start exactly on a boundary, skip to the *next* one so a GRC
        # never fires twice for the same interval.
        while not self._stop.is_set():
            nxt = next_grc_ts()
            delay = nxt - time.time()
            if delay > 0:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                    break
                except asyncio.TimeoutError:
                    pass
            # small settle so we are never early
            while time.time() < nxt:
                await asyncio.sleep(0.01)
            try:
                self.last_fired = nxt
                self.fired_count += 1
                await self.on_grc(nxt)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("GRC handler failed")
