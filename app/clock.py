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
import contextlib
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
    """Fires `on_grc(grc_ts)` exactly at every 4-minute boundary.

    The scheduler must never be the silent single point of failure: if the
    task dies (exception in the loop, cancellation) or a handler hangs, the
    whole product looks "offline" while the socket still answers. `status()`
    reports measured liveness and `watchdog()` restarts a dead/silent
    scheduler, so a GRC outage self-heals instead of needing a restart.
    """

    #: a handler that runs longer than this is abandoned (the engine's own
    #: analysis budget is 175 s; a GRC pass gets the same room + margin)
    HANDLER_TIMEOUT_S = 240.0
    #: scheduler is declared silent when no fire happened for this long
    SILENT_AFTER_S = 2 * GRC_INTERVAL_SECONDS + 30.0

    def __init__(self, on_grc: Callable[[float], Awaitable[None]]):
        self.on_grc = on_grc
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self.last_fired: float = 0.0
        self.fired_count: int = 0
        self.started_at: float = 0.0
        self.restarts: int = 0
        self.last_error: str = ""
        self.handler_ms: float = 0.0

    def status(self) -> dict:
        """Measured scheduler state — never a guess."""
        now = time.time()
        alive = bool(self._task is not None and not self._task.done())
        return {
            "alive": alive,
            "fired_count": self.fired_count,
            "last_fired": self.last_fired,
            "last_fired_age_s": (round(now - self.last_fired, 1)
                                 if self.last_fired else None),
            "next_in_s": round(seconds_to_next_grc(now), 1),
            "interval_s": GRC_INTERVAL_SECONDS,
            "restarts": self.restarts,
            "handler_ms": self.handler_ms,
            "last_error": self.last_error,
            "uptime_s": round(now - self.started_at, 1) if self.started_at else 0.0,
        }

    async def watchdog(self) -> dict:
        """Restart the scheduler if it is dead or has gone silent.

        Called from the app's lag meter (every few seconds), so a clock that
        stopped firing is repaired within one interval instead of leaving a
        frozen board until someone restarts the process.
        """
        now = time.time()
        st = self.status()
        if self._task is None and not self.started_at:
            # never started yet (boot still in progress): nothing to heal
            return {**st, "silent_for_s": 0.0, "action": "not-started"}
        # Silence is measured from the LAST LIVENESS EVENT, which is either a
        # real fire or the moment the scheduler was (re)built. Measuring only
        # `last_fired` made the watchdog self-defeating: after one missed GRC
        # it rebuilt the task, but `last_fired` stayed old, so the very next
        # pass declared the fresh task silent again and cancelled it — up to
        # 240 s before its next boundary. One miss became a permanent outage
        # (observed as "clock watchdog: restarting scheduler (silent=649.7s)"
        # repeating forever while no GRC ever fired). The baseline must be
        # reset by the restart itself.
        silent_for = now - max(self.last_fired or 0.0, self.started_at or 0.0)
        healthy = st["alive"] and silent_for <= self.SILENT_AFTER_S
        st["silent_for_s"] = round(silent_for, 1)
        st["action"] = "healthy" if healthy else "restart"
        if healthy:
            return st
        # dead task, or alive but no fire for > 2 intervals: rebuild it
        log.warning("clock watchdog: restarting scheduler (alive=%s silent=%.1fs)",
                    st["alive"], silent_for)
        try:
            if self._task is not None and not self._task.done():
                self._task.cancel()
                with contextlib.suppress(BaseException):
                    await self._task
        except Exception:                      # pragma: no cover - defensive
            pass
        self.restarts += 1
        self._stop = asyncio.Event()
        self.started_at = time.time()      # new liveness baseline, see above
        self._task = asyncio.create_task(self._run())
        return {**self.status(), "action": "restarted",
                "silent_for_s": round(silent_for, 1)}

    async def start(self) -> None:
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run())
        self.started_at = time.time()

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
            t0 = time.perf_counter()
            self.last_fired = nxt
            self.fired_count += 1
            try:
                await asyncio.wait_for(self.on_grc(nxt),
                                       timeout=self.HANDLER_TIMEOUT_S)
            except asyncio.TimeoutError:
                self.last_error = (f"GRC handler exceeded "
                                   f"{self.HANDLER_TIMEOUT_S:.0f}s — abandoned")
                log.error(self.last_error)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"GRC handler failed: {exc!r}"
                log.exception("GRC handler failed")
            self.handler_ms = round((time.perf_counter() - t0) * 1000, 1)
