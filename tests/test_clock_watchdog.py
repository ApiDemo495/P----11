"""The clock watchdog must converge, not loop.

Live failure: after one missed GRC the log repeated

    clock watchdog: restarting scheduler (alive=True silent=649.7s)

every five seconds for ever, and no GRC ever fired again. The watchdog
rebuilt the task but measured silence from `last_fired` only, which the
restart did not update — so the brand-new scheduler was declared silent on
the very next pass and cancelled up to 240 s before its next boundary. A
self-*defeating* watchdog is worse than none: it guarantees the outage it is
supposed to repair.
"""
from __future__ import annotations

import asyncio
import time

from app.clock import MasterClock


def test_restart_resets_the_silence_baseline():
    async def go():
        fired: list[float] = []

        async def on_grc(ts):
            fired.append(ts)

        clk = MasterClock(on_grc)
        await clk.start()
        try:
            # The live state: built 700 s ago, last fire 700 s ago, so the
            # boundary was missed. Pass 1 must repair it; pass 2 (5 s later,
            # as the lag meter calls it) must then see a HEALTHY clock. With
            # only `last_fired` consulted, pass 2 still saw ~700 s of silence
            # and cancelled the task it had just created.
            clk.started_at = time.time() - 700
            clk.last_fired = time.time() - 700
            first = await clk.watchdog()
            assert first["action"] == "restarted", first
            assert clk.restarts == 1
            # THE BUG: the next pass must see a fresh baseline, not 0.0
            second = await clk.watchdog()
            assert second["action"] == "healthy", (
                "watchdog restarted again immediately — the silence baseline "
                f"was not reset by the restart: {second}")
            assert clk.restarts == 1, second
        finally:
            await clk.stop()

    asyncio.run(go())


def test_a_silent_but_running_scheduler_is_still_repaired():
    """The gate must still fire when it genuinely needs to."""
    async def go():
        async def on_grc(ts):
            pass

        clk = MasterClock(on_grc)
        await clk.start()
        try:
            clk.started_at = 1.0
            clk.last_fired = 1.0
            out = await clk.watchdog()
            assert out["action"] == "restarted"
            assert out["silent_for_s"] > clk.SILENT_AFTER_S
        finally:
            await clk.stop()

    asyncio.run(go())


def test_a_healthy_scheduler_is_not_disturbed():
    async def go():
        async def on_grc(ts):
            pass

        clk = MasterClock(on_grc)
        await clk.start()
        try:
            out = await clk.watchdog()
            assert out["action"] == "healthy", out
            assert clk.restarts == 0
        finally:
            await clk.stop()

    asyncio.run(go())
