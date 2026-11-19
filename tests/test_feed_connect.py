"""The CONNECT BINANCE button must DO something, and say what happened.

Live report: "i don't want simulation fake data i want everything real" —
while the header offered a CONNECT BINANCE button that only opened execution
settings, and `PUT /api/settings` silently dropped `feed_mode`. There was no
path from "the chip says SIMULATED" to "the tape is real" except restarting
with FEED_MODE=binance.

`MarketFeed.connect_real()` is that path. These tests pin the two outcomes
that matter: it adopts the real tape when the exchange answers, and it reports
the real error (and keeps the simulator) when it does not — never a silent
fallback, never a false "connected".
"""
from __future__ import annotations

import asyncio

import pytest


def _mkfeed(monkeypatch, tmp_path, *, works: bool):
    """A MarketFeed whose exchange either answers or is unreachable."""
    import app.feed as F

    def _state(sym):
        # the REAL SymbolState: a hand-written stub drifts from __slots__ and
        # fails on the first attribute the production code touches
        st = F.SymbolState(sym)
        st.closes = [68000.0, 68010.0, 68020.0] * 20
        st.times = list(range(60, 60 * 60, 60))
        st.volumes = [1.0] * 60
        st.opens = [68000.0] * 60
        st.highs = [68030.0] * 60
        st.lows = [67990.0] * 60
        st.sources = ["sim"] * 60
        return st

    feed = F.MarketFeed.__new__(F.MarketFeed)       # no I/O in __init__
    feed.mode = "sim"
    feed.source = "sim"
    feed.states = {s: _state(s) for s in F.SYMBOL_ORDER}
    feed._probing = False
    feed._sim_tasks = []
    feed._tasks = []
    feed._real_error = ""
    feed._real_attempts = 0
    feed.last_real_delivery = 0.0
    feed.rejected = feed.updates = feed.switches = feed.quote_prints = 0
    feed.last_delivery = 0.0
    feed._recent = []
    feed._sim_price = dict(F.SIM_BASE_PRICES)

    async def ok_start():
        for st in feed.states.values():
            st.last_tick = 1.0            # a real print landed
        if not works:
            raise ConnectionError("TLS/SSL connection has been closed (EOF)")

    def noop(*a, **k):     # note() is synchronous
        return None

    object.__setattr__(feed, "_start_binance", ok_start)
    feed.note = noop
    return feed, F


def test_connect_real_adopts_the_tape_when_the_exchange_answers(monkeypatch, tmp_path):
    feed, F = _mkfeed(monkeypatch, tmp_path, works=True)
    out = asyncio.run(feed.connect_real())
    assert out["ok"] is True, out
    assert feed.mode == "binance", "the tape family must actually switch"
    assert out["mode"] == "binance"
    assert feed._real_error == ""


def test_connect_real_reports_the_error_and_keeps_the_simulator(monkeypatch, tmp_path):
    feed, F = _mkfeed(monkeypatch, tmp_path, works=False)
    out = asyncio.run(feed.connect_real())
    assert out["ok"] is False
    assert "TLS/SSL" in out["error"]
    assert feed.mode == "sim", "a failed connect must not pretend to be live"
    # and the failure is recorded for the chip, not swallowed
    assert feed._real_attempts == 1
    stats_src = (F.__file__)
    assert "real_reason" in open(stats_src).read()


def test_connect_real_is_idempotent_and_reports_already_connected():
    feed, F = _mkfeed(None, None, works=True)
    feed.mode = "binance"
    out = asyncio.run(feed.connect_real())
    assert out["ok"] is True and out.get("already") is True


def test_settings_never_silently_swallow_a_feed_change():
    """`PUT /api/settings` dropped feed_mode, so a "saved" change did nothing."""
    src = open("app/main.py").read()
    i = src.index('@app.put("/api/settings")')
    block = src[i:i + 1200]
    assert '"ignored"' in block, "unapplied keys must be named, not dropped"


def test_connect_route_exists():
    src = open("app/main.py").read()
    assert '@app.post("/api/feeds/connect")' in src
    js = open("static/app.js").read()
    assert 'api("/api/feeds/connect"' in js, "the button must hit the route"
