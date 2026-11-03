import asyncio

import pytest

pytest_plugins = ("pytest_asyncio",)


def run_sync(coro):
    """Run a coroutine on a usable event loop.

    TestClient/anyio close the loop they used and clear it as *current*;
    older helpers then crash with "There is no current event loop" when
    they run after an HTTP test. Never assume get_event_loop() succeeds.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)
