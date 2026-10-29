"""Real-time market data feed.

Modes (config FEED_MODE):
  auto    -> try Binance (REST 1m backfill + kline WebSocket), fall back to
             the built-in simulator when the exchange is unreachable.
  binance -> Binance only.
  sim     -> deterministic-quality simulator (regime-switching GBM, aligned
             1-minute bars, 1-second ticks) — also used for offline tests.

The feed maintains, per asset:
  * a minute bar series (times, closes, volumes) for the formula windows
  * a live 1-second tick stream (price + partial-minute volume) that keeps
    flowing between GRCs for the UI and hit/SL detection
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time

import httpx

from .config import (
    BACKFILL_BARS,
    BINANCE_REST,
    BINANCE_WS,
    FEED_MODE,
    SYMBOLS,
    SYMBOL_ORDER,
)

log = logging.getLogger("feed")

SIM_BASE_PRICES = {"BTC": 68000.0, "PAXG": 2650.0, "ETH": 3500.0, "BNB": 600.0}
SIM_PER_MIN_VOL = {"BTC": 120.0, "PAXG": 18.0, "ETH": 1500.0, "BNB": 4200.0}


class SymbolState:
    __slots__ = (
        "symbol", "times", "closes", "volumes",
        "opens", "highs", "lows",
        "live_price", "live_volume", "live_open", "live_high", "live_low",
        "day_open",
    )

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.times: list[int] = []
        self.closes: list[float] = []
        self.volumes: list[float] = []
        self.opens: list[float] = []
        self.highs: list[float] = []
        self.lows: list[float] = []
        self.live_price: float = 0.0
        self.live_volume: float = 0.0     # current partial minute volume
        self.live_open: float = 0.0       # current partial minute OHLC
        self.live_high: float = 0.0
        self.live_low: float = 0.0
        self.day_open: float = 0.0

    @property
    def last_bar_time(self) -> int:
        return self.times[-1] if self.times else 0

    def push_bar(
        self,
        ts: int,
        close: float,
        volume: float,
        open_: float | None = None,
        high: float | None = None,
        low: float | None = None,
    ) -> None:
        # derive missing OHLC from the close and the previous bar
        if open_ is None:
            open_ = self.closes[-1] if self.closes else close
        if high is None:
            high = max(open_, close)
        if low is None:
            low = min(open_, close)
        # replace a partially-recorded bar with the same timestamp, else append
        if self.times and self.times[-1] == ts:
            self.closes[-1] = close
            self.volumes[-1] = volume
            self.opens[-1] = open_
            self.highs[-1] = high
            self.lows[-1] = low
        elif self.times and ts < self.times[-1]:
            return
        else:
            self.times.append(ts)
            self.closes.append(close)
            self.volumes.append(volume)
            self.opens.append(open_)
            self.highs.append(high)
            self.lows.append(low)

    def aligned(self) -> dict[int, tuple[float, float]]:
        return dict(zip(self.times, zip(self.closes, self.volumes)))


class MarketFeed:
    def __init__(self) -> None:
        self.mode = "sim"
        self.states: dict[str, SymbolState] = {s: SymbolState(s) for s in SYMBOL_ORDER}
        self.on_tick = None            # async callback(symbol, price, volume)
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()
        self.ready = asyncio.Event()
        self._sim_rng = random.Random(495)
        # simulator internal price memory for continuity across minute bars
        self._sim_price = dict(SIM_BASE_PRICES)

    # ------------------------------------------------------------------ API
    def state(self, symbol: str) -> SymbolState:
        return self.states[symbol]

    def prices(self) -> dict[str, float]:
        return {s: self.states[s].live_price for s in SYMBOL_ORDER}

    def others_aligned(self, symbol: str) -> dict[str, tuple[list[int], list[float]]]:
        """Minute-epoch -> price series of the *other* assets (F19 input)."""
        out: dict[str, tuple[list[int], list[float]]] = {}
        for s, st in self.states.items():
            if s != symbol:
                out[s] = (list(st.times), list(st.closes))
        return out

    async def start(self) -> None:
        mode = FEED_MODE
        if mode in ("auto", "binance"):
            try:
                await self._start_binance()
                self.mode = "binance"
                log.info("feed: Binance mode active")
                return
            except Exception as exc:      # pragma: no cover - network dependent
                if mode == "binance":
                    raise
                log.warning("feed: Binance unavailable (%s); using simulator", exc)
        await self._start_sim()
        self.mode = "sim"

    async def stop(self) -> None:
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # ------------------------------------------------------------- Binance
    async def _start_binance(self) -> None:
        # 1m kline backfill so all four TF windows (248 bars) are ready instantly
        async with httpx.AsyncClient(base_url=BINANCE_REST, timeout=15.0) as client:
            for sym, ex_symbol in SYMBOLS.items():
                resp = await client.get(
                    "/api/v3/klines",
                    params={"symbol": ex_symbol, "interval": "1m", "limit": BACKFILL_BARS},
                )
                resp.raise_for_status()
                rows = resp.json()
                st = self.states[sym]
                prev_close = None
                for row in rows:
                    ts = int(row[0]) // 1000
                    o, h, l, c, v = (
                        float(row[1]), float(row[2]), float(row[3]),
                        float(row[4]), float(row[5]),
                    )
                    st.push_bar(ts, c, v, o, h, l)
                    prev_close = c
                if st.closes:
                    st.live_price = st.closes[-1]
                    st.live_open = st.opens[-1]
                    st.live_high = st.highs[-1]
                    st.live_low = st.lows[-1]
                st.day_open = st.closes[0] if st.closes else 0.0
        if all(len(self.states[s].closes) >= 30 for s in SYMBOL_ORDER):
            self.ready.set()

        self._tasks.append(asyncio.create_task(self._binance_ws_loop()))

    async def _binance_ws_loop(self) -> None:  # pragma: no cover - network
        streams = "/".join(f"{ex.lower()}@kline_1m" for ex in SYMBOLS.values())
        url = f"{BINANCE_WS}?streams={streams}"
        import websockets

        reverse = {v.lower(): k for k, v in SYMBOLS.items()}
        while not self._stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    async for raw in ws:
                        msg = json.loads(raw)
                        data = msg.get("data", msg)
                        if data.get("e") != "kline":
                            continue
                        k = data["k"]
                        ex = k["s"].lower()
                        sym = reverse.get(ex)
                        if not sym:
                            continue
                        st = self.states[sym]
                        price = float(k["c"])
                        st.live_price = price
                        st.live_volume = float(k["v"])
                        st.live_open = float(k["o"])
                        st.live_high = float(k["h"])
                        st.live_low = float(k["l"])
                        ts = int(k["t"]) // 1000
                        st.push_bar(
                            ts, price, float(k["v"]),
                            float(k["o"]), float(k["h"]), float(k["l"]),
                        )
                        await self._emit_tick(sym, price, float(k["v"]))
            except asyncio.CancelledError:
                raise
            except Exception as exc:                        # reconnect
                log.warning("feed: ws reconnect (%s)", exc)
                await asyncio.sleep(2.0)

    # ---------------------------------------------------------- Simulator
    async def _start_sim(self) -> None:
        rng = self._sim_rng
        now = int(time.time())
        start_ts = now - (BACKFILL_BARS + 5) * 60

        # regime-switching pre-history so baselines/regimes start populated
        for sym in SYMBOL_ORDER:
            st = self.states[sym]
            price = SIM_BASE_PRICES[sym] * (1.0 + rng.gauss(0, 0.01))
            base_vol = {"BTC": 0.0007, "PAXG": 0.00021, "ETH": 0.0009, "BNB": 0.0008}[sym]
            drift = 0.0
            regime_left = 0
            vol_mult = 1.0
            for i in range(BACKFILL_BARS + 5):
                if regime_left <= 0:
                    regime_left = rng.randint(30, 240)
                    vol_mult = rng.choice([0.7, 1.0, 1.6, 2.4])
                    # directional drift regimes so trending windows occur and
                    # the confidence floor is organically reachable
                    drift = rng.choice([-0.00045, -0.0002, 0.0, 0.0002, 0.00045])
                regime_left -= 1
                prev = price
                ret = rng.gauss(drift, base_vol * vol_mult)
                # PAXG is gold-pegged: strong mean reversion keeps it anchored
                if sym == "PAXG":
                    ret = 0.9 * ret - 0.02 * math.log(price / SIM_BASE_PRICES[sym])
                price *= math.exp(ret)
                volume = SIM_PER_MIN_VOL[sym] * rng.lognormvariate(0, 0.5) * vol_mult
                # candle OHLC: open at previous close, wick proportional to vol
                wick = abs(rng.gauss(0, 1.0)) * base_vol * (0.6 + 0.5 * vol_mult)
                o = prev
                h = max(o, price) * (1.0 + wick)
                l = min(o, price) * (1.0 - wick)
                ts = start_ts + i * 60
                st.push_bar(ts, price, volume, o, h, l)
            st.live_price = price
            st.live_open = st.opens[-1]
            st.live_high = st.highs[-1]
            st.live_low = st.lows[-1]
            self._sim_price[sym] = price
            # day_open for percent display: bar from ~24h ago if available
            st.day_open = st.closes[0] if st.closes else price

        self.ready.set()
        self._tasks.append(asyncio.create_task(self._sim_tick_loop()))

    async def _sim_tick_loop(self) -> None:
        rng = self._sim_rng
        per_sec_vol = {s: SIM_PER_MIN_VOL[s] / 60.0 for s in SYMBOL_ORDER}
        last_minute = int(time.time() // 60)
        # seed live candle fields from each symbol's last state
        for sym in SYMBOL_ORDER:
            st = self.states[sym]
            st.live_open = st.live_price or (st.closes[-1] if st.closes else 0.0)
            st.live_high = st.live_open
            st.live_low = st.live_open
        while not self._stop.is_set():
            await asyncio.sleep(1.0)
            now = time.time()
            minute = int(now // 60)
            rolled = minute != last_minute
            for sym in SYMBOL_ORDER:
                st = self.states[sym]
                price = self._sim_price[sym]
                sigma = {"BTC": 7e-5, "PAXG": 2e-5, "ETH": 9e-5, "BNB": 8e-5}[sym]
                price *= math.exp(rng.gauss(0, sigma))
                if sym == "PAXG":
                    price += 0.02 * (SIM_BASE_PRICES[sym] - price)
                self._sim_price[sym] = price
                st.live_price = price
                vol = per_sec_vol[sym] * rng.lognormvariate(0, 0.6)
                if rolled:
                    # close out the minute that just ended with its OHLC
                    st.push_bar(
                        last_minute * 60, price, st.live_volume,
                        st.live_open or price,
                        max(st.live_high or price, price),
                        min(st.live_low or price, price),
                    )
                    st.live_volume = 0.0
                    st.live_open = price
                    st.live_high = price
                    st.live_low = price
                else:
                    st.live_volume += vol
                    st.live_high = max(st.live_high or price, price)
                    st.live_low = min(st.live_low or price, price) if st.live_low else price
                await self._emit_tick(sym, price, vol)
            last_minute = minute

    async def _emit_tick(self, symbol: str, price: float, volume: float) -> None:
        if self.on_tick is not None:
            try:
                result = self.on_tick(symbol, price, volume)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                log.exception("feed: tick callback failed")
