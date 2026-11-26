"""Counted best-effort guards.

Why this module exists
----------------------
The app has ~127 places that say `except Exception: pass`. Most are honest:
a calibration learner, a telemetry ring buffer or a pretty-printed log line
must never take down the settle loop because of one bad number. Nobody wants
those to raise.

The problem is that `pass` makes "this optional thing is working" and "this
optional thing has been broken for three days" look *identical* from the
outside. Measured on this repo: `MetaCognition.record()` (the calibration
learner) and `flywire.record()` (the attribution learner) are each wrapped in
a bare `except Exception: pass` inside `Engine._settle_and_lock`. If either
one started raising — a schema change, a NaN, a None — the dashboard would
keep showing a calibration panel that had stopped updating, and every
"learning" readout would be a fossil. That is the same class of silent
wrong answer as rendering a missing number as 0.00.

So: the exception is still swallowed (behaviour unchanged — these are
deliberately best-effort), but it is *counted*, the first message is kept,
and the totals are exposed on the diagnostics payload. A learner that has
been failing 400 times is now visible as failing 400 times.

Usage
-----
    from .silence import guard

    with guard("settle.meta_record"):
        self.meta.record(symbol, tf, prev["inputs"], up)

`guard` never raises. It records `{site: {"count", "last", "first_ts"}}`.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from contextlib import contextmanager

#: how many distinct sites we keep detail for before collapsing the rest
MAX_SITES = 256

_lock = threading.Lock()
_counts: "OrderedDict[str, dict]" = OrderedDict()
_suppressed = {"count": 0}


def reset() -> None:
    """Clear the counters (tests only)."""
    with _lock:
        _counts.clear()
        _suppressed["count"] = 0


@contextmanager
def guard(site: str, *, record: bool = True):
    """Swallow any exception from the block, counting it under `site`.

    `record=False` still swallows but does not count — use it only for a
    genuinely uninteresting hot path, and prefer leaving it on.
    """
    try:
        yield
    except Exception as exc:          # noqa: BLE001 - swallowing IS the point
        if not record:
            return
        key = str(site)
        now = time.time()
        with _lock:
            slot = _counts.get(key)
            if slot is None:
                if len(_counts) >= MAX_SITES:
                    _suppressed["count"] += 1
                    return
                slot = {"count": 0, "last": "", "first_ts": now, "last_ts": now}
                _counts[key] = slot
            slot["count"] += 1
            slot["last"] = "%s: %s" % (type(exc).__name__, exc)
            slot["last_ts"] = now


def report() -> dict:
    """Snapshot of every swallowed failure, worst first."""
    with _lock:
        items = [dict(site=k, **v) for k, v in _counts.items()]
        suppressed = _suppressed["count"]
    items.sort(key=lambda r: -r["count"])
    total = sum(r["count"] for r in items) + suppressed
    return {
        "total_swallowed": total,
        "sites": len(items),
        "suppressed_sites": suppressed,
        "detail": items[:40],
        "rule": ("a swallowed exception is still a failure; count it so "
                 "'learning quietly stopped' can never look like 'learning'"),
    }
