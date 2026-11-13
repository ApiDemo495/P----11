"""Feed provenance regressions: live prices must not be paired with old sim candles."""
from __future__ import annotations

import time

from app.config import SYMBOLS
from app.feed import MarketFeed, _family


def test_unknown_writer_is_not_classified_as_sim():
    from app.feed import SymbolState

    assert _family("binance-ws") == "binance"
    assert _family("sim") == "sim"
    assert _family("test") == "unknown"
    state = SymbolState("BTC")
    state.push_bar(60, 100.0, 1.0, 100.0, 101.0, 99.0, source="test")
    assert state.indices_for_family("unknown") == []


def test_adopting_thin_binance_tape_discards_sim_bars_and_price():
    """The old >=25 guard kept 76k sim candles beside a 68k Binance tick."""
    feed = MarketFeed()
    now = int(time.time() // 60) * 60
    for st in feed.states.values():
        for i in range(4):
            t = now - (4 - i) * 60
            p = 75_900.0 + i
            st.push_bar(t, p, 1.0, p, p + 20, p - 20, source="sim")
        st.live_price = 75_903.0
    btc = feed.state("BTC")
    # One exchange candle is newer than the simulator; other symbols have
    # no Binance bars yet, which reproduces a thin/partial source handover.
    btc.push_bar(now, 68_769.0, 1.0, 68_770.0, 68_780.0,
                 68_750.0, source="binance-ws")
    btc.live_price = 68_769.0

    feed._switch_tape("binance", "binance-ws")

    assert feed.mode == "binance"
    assert btc.sources == ["binance-ws"]
    assert btc.closes == [68_769.0]
    assert btc.live_price == 68_769.0
    for symbol, st in feed.states.items():
        assert all(_family(source) == "binance" for source in st.sources)
        if symbol != "BTC":
            assert st.times == []
            assert st.live_price == 0.0  # never keep a simulator quote as live


def test_demoting_to_sim_discards_binance_history():
    feed = MarketFeed()
    feed.mode = "binance"
    now = int(time.time() // 60) * 60
    st = feed.state("BTC")
    st.push_bar(now - 60, 68_769.0, 1.0, 68_760.0, 68_780.0,
                68_750.0, source="binance-ws")
    st.push_bar(now, 68_800.0, 1.0, 68_790.0, 68_810.0,
                68_780.0, source="sim")
    st.live_price, st.last_tick = 68_800.0, time.time()

    feed._switch_tape("sim", "sim")

    assert st.sources == ["sim"]
    assert st.closes == [68_800.0]
    assert st.live_price == 68_800.0


def test_backfill_replaces_newer_sim_timestamps_with_sorted_binance_history(monkeypatch):
    """REST klines must be inserted/replaced, not ignored behind sim time."""
    from app import feed as feed_module
    from conftest import run_sync

    feed = MarketFeed()
    now = int(time.time() // 60) * 60
    for st in feed.states.values():
        for i in range(5):
            t = now - (4 - i) * 60
            p = 75_900.0 + i
            st.push_bar(t, p, 1.0, p, p + 20, p - 20, source="sim")
        st.live_price = 75_904.0

    rest_times = [now - 3 * 60, now - 2 * 60, now - 60]
    rows_by_symbol = {}
    for index, exchange_symbol in enumerate(SYMBOLS.values()):
        price = 68_760.0 + index * 100
        rows_by_symbol[exchange_symbol] = [
            [ts * 1000, str(price), str(price + 10), str(price - 10),
             str(price + i), "3.0"]
            for i, ts in enumerate(rest_times)
        ]

    class Response:
        def __init__(self, value):
            self.value = value

        def raise_for_status(self):
            pass

        def json(self):
            return self.value

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, params=None):
            if url.endswith("/api/v3/klines"):
                return Response(rows_by_symbol[params["symbol"]])
            return Response({"openPrice": "68000"})

    monkeypatch.setattr(feed_module.httpx, "AsyncClient", FakeAsyncClient)
    assert run_sync(feed._fetch_backfill()) == len(SYMBOLS) * len(rest_times)

    for exchange_symbol, symbol in zip(SYMBOLS.values(), SYMBOLS):
        st = feed.state(symbol)
        assert st.times == rest_times
        assert st.sources == ["binance-rest"] * len(rest_times)
        assert all(_family(source) == "binance" for source in st.sources)
        assert st.live_price == float(rows_by_symbol[exchange_symbol][-1][4])


def test_candles_endpoint_excludes_non_active_source_history(monkeypatch):
    """API candles are filtered to the active feed, even during a bad handover."""
    from fastapi.testclient import TestClient
    import app.main as main

    feed = MarketFeed()
    feed.mode, feed.source = "binance", "binance-ws"
    st = feed.state("BTC")
    now = int(time.time() // 60) * 60
    st.push_bar(now - 120, 75_920.0, 1.0, 75_910.0, 75_940.0,
                75_900.0, source="sim")
    st.push_bar(now - 60, 68_769.0, 1.0, 68_770.0, 68_780.0,
                68_750.0, source="binance-ws")
    st.live_price, st.live_open = 68_769.0, 68_770.0
    st.live_high, st.live_low, st.live_volume = 68_780.0, 68_750.0, 1.0
    monkeypatch.setattr(main.engine, "feed", feed)

    response = TestClient(main.app).get(
        "/api/candles?symbol=BTC&tf=tf1&n=260")
    payload = response.json()
    assert response.status_code == 200
    assert payload["bar_family"] == "binance"
    assert payload["excluded_foreign_bars"] == 1
    assert payload["bars"]
    assert all(bar["c"] < 70_000 for bar in payload["bars"])
