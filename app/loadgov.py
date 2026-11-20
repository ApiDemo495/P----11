"""Load governance: bulk background work must yield to the real-time loop.

The warm-history replay is minutes of pure-CPU analytics, and the engine's
live passes are timing-sensitive. A worker thread does not automatically
protect the event loop: CPython still hands the GIL back and forth, and a
tight numeric loop in one thread stretches the latency of everything on the
loop (measured: 1.8-2.6 s stalls in /api/health while the replay ran, which
the client painted as "ENGINE BUSY").

This module is the single place that decides how hard bulk work may push:

* ``note_lag(s)`` records the loop lag the lag-meter actually measured.
* ``cooperate()`` is called by bulk workers between units of work. It sleeps
  longer the worse the measured lag is, so the loop always gets its slice.

No timers, no threads, no state beyond one dict: a bulk worker that never
calls ``cooperate()`` simply keeps the old (bad) behaviour, and a process
with no lag pays ~0 ms.
"""
from __future__ import annotations

import time

_LAG: dict[str, float] = {"s": 0.0, "at": 0.0}
# how long to hand the CPU back, by measured loop lag
_STEPS = ((0.75, 0.25), (0.40, 0.15), (0.15, 0.05))


def note_lag(seconds: float) -> None:
    """Called by the loop's lag meter with the lag it measured."""
    _LAG["s"] = max(0.0, float(seconds))
    _LAG["at"] = time.time()


def current() -> float:
    """Measured loop lag, decaying to 0 if the meter stopped reporting."""
    if time.time() - _LAG["at"] > 10.0:
        return 0.0
    return _LAG["s"]


def cooperate(floor: float = 0.0) -> float:
    """Sleep a little, scaled by how starved the loop is. Returns the sleep."""
    lag = current()
    for threshold, pause in _STEPS:
        if lag >= threshold:
            time.sleep(max(floor, pause))
            return max(floor, pause)
    if floor:
        time.sleep(floor)
    return floor
