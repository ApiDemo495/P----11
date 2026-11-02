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

# D-3: Binance refuses some Codespaces regions -> data-stream.binance.vision
# mirrors (same API shape) are the documented rotation target.
BINANCE_REST_VISION = "https://data-stream.binance.vision"
BINANCE_WS_VISION = "wss://data-stream.binance.vision"
# §0.1: a real feed gets a grace period before the simulator may start
# (advice: 2–3 min at 22–247 min horizons), and the simulator is cancelled
# the instant a real feed is healthy again (D-1).
FEED_GRACE_S = 120.0
REAL_STALE_S = 120.0          # binance silent this long -> demote to sim
SIM_BASE_PRICES = {"BTC": 68000.0, "PAXG": 2650.0, "ETH": 3500.0, "BNB": 600.0}
SIM_PER_MIN_VOL = {"BTC": 120.0, "PAXG": 18.0, "ETH": 1500.0, "BNB": 4200.0}


def _family(source: str) -> str:
    """Writer family: every binance path is one family, the simulator another."""
    return "binance" if str(source).startswith("binance") else "sim"


class SymbolState:
    __slots__ = (
        "symbol", "times", "closes", "volumes",
        "opens", "highs", "lows", "sources",
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
        self.sources: list[str] = []   # which source wrote each bar (D-1)
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
        source: str = "unknown",
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
            if self.sources:
                self.sources[-1] = source
        elif self.times and ts < self.times[-1]:
            return
        else:
            self.times.append(ts)
            self.closes.append(close)
            self.volumes.append(volume)
            self.opens.append(open_)
            self.highs.append(high)
            self.lows.append(low)
            self.sources.append(source)

    def flush_foreign(self, family: str) -> int:
        """Drop every bar whose writer is not in `family` (D-1: any source
        change flushes the tape so two feeds can never alternate in one)."""
        keep = [i for i, src in enumerate(self.sources)
                if _family(src) == family]
        dropped = len(self.times) - len(keep)
        if dropped > 0:
            self.times = [self.times[i] for i in keep]
            self.closes = [self.closes[i] for i in keep]
            self.volumes = [self.volumes[i] for i in keep]
            self.opens = [self.opens[i] for i in keep]
            self.highs = [self.highs[i] for i in keep]
            self.lows = [self.lows[i] for i in keep]
            self.sources = [self.sources[i] for i in keep]
        return dropped

    def aligned(self) -> dict[int, tuple[float, float]]:
        return dict(zip(self.times, zip(self.closes, self.volumes)))


class MarketFeed:
    def __init__(self) -> None:
        # honesty counters (advice D: never a bare "online")
        self.source = "boot"
        self.last_delivery = 0.0
        self.updates = 0
        self.rejected = 0
        self.quote_prints = 0            # volume<=0 prints (excluded from vol stats)
        self.last_real_delivery = 0.0    # last binance-path write (adopt/demote)
        self._reconcile_task: asyncio.Task | None = None
        self._sim_tasks: list[asyncio.Task] = []
        self._next_revive = 0.0
        self._tick_times: list[float] = []
        self.switches = 0
        self.mode = "sim"
        self.states: dict[str, SymbolState] = {s: SymbolState(s) for s in SYMBOL_ORDER}
        self.on_tick = None            # async callback(symbol, price, volume)
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()
        self.ready = asyncio.Event()
        self._sim_rng = random.Random(495)
        # simulator internal price memory for continuity across minute bars
        self._sim_price = dict(SIM_BASE_PRICES)
        self._sim_regime: dict[str, dict] = {}   # live drift regimes (sim)

    # ------------------------------------------------------------------ API
    def state(self, symbol: str) -> SymbolState:
        return self.states[symbol]

    def note(self, source: str) -> None:
        """Single-writer honesty: record which source is delivering NOW."""
        now = time.time()
        if source != self.source:
            if self.source not in ("boot",):
                self.switches += 1
            self.source = source
        self.last_delivery = now
        self.updates += 1
        self._tick_times.append(now)
        del self._tick_times[:-200]

    def stats(self) -> dict:
        """Feed chip payload — source, age, ticks/s, switches (advice D)."""
        now = time.time()
        recent = [t for t in self._tick_times if now - t <= 10.0]
        tps = len(recent) / 10.0
        age = round(now - self.last_delivery, 1) if self.last_delivery else None
        stale = (age is None) or age > 30.0
        return {
            "source": self.source,
            "mode": self.mode,
            "simulated": self.mode != "binance",
            "last_delivery_age_s": age,
            "ticks_per_s": round(tps, 2),
            "updates": self.updates,
            "switches": self.switches,
            "rejected": self.rejected,
            "quote_prints": self.quote_prints,
            "real_age_s": (round(now - self.last_real_delivery, 1)
                           if self.last_real_delivery else None),
            "bars": sum(len(st.times) for st in self.states.values()),
            "bars_sim": sum(1 for st in self.states.values()
                            for src in st.sources if _family(src) == "sim"),
            "stale": stale,
            "label": (f"SIMULATED TAPE" if self.mode != "binance"
                      else f"live · {self.source} · {tps:.1f} ticks/s"),
        }

    def prices(self) -> dict[str, float]:
        return {s: self.states[s].live_price for s in SYMBOL_ORDER}

    def others_aligned(self, symbol: str) -> dict[str, tuple[list[int], list[float]]]:
        """Minute-epoch -> price series of the *other* assets (F19 input)."""
        out: dict[str, tuple[list[int], list[float]]] = {}
        for s, st in self.states.items():
            if s != symbol:
                out[s] = (list(st.times), list(st.closes))
        return out

    def _ingest_ok(self, source: str) -> bool:
        """D-1 single writer: only the active family may write the tape;
        foreign writes are rejected and COUNTED (shown in /api/feeds/diagnose)."""
        if _family(source) == _family(self.mode):
            return True
        self.rejected += 1
        return False

    def _switch_tape(self, new_mode: str, source: str) -> None:
        """Mode change. On ADOPT (sim -> real) every synthetic bar is flushed
        (D-1). On DEMOTE the real history is KEPT — bars are time-ordered and
        never alternate between writers, so dropping them would only starve
        the formula windows; the chip turns SIMULATED TAPE instead."""
        if new_mode == self.mode:
            return
        if new_mode == "binance":
            for st in self.states.values():
                st.flush_foreign("binance")
        self.mode = new_mode
        self.note(source)               # counts switches in stats()

    async def _reconcile_loop(self) -> None:
        """§1.1 source order: adopt a delivering real feed (D-2) even when
        the state machine says sim; demote to the simulator when the real
        feed goes silent; revive attempts run under the same loop."""
        while not self._stop.is_set():
            await asyncio.sleep(10.0)
            now = time.time()
            if FEED_MODE == "sim":
                continue
            delivering = (self.last_real_delivery > 0
                          and now - self.last_real_delivery < 30.0)
            if self.mode == "sim" and delivering:
                log.info("feed: real feed delivering -> adopt (D-2)")
                for t in list(self._sim_tasks):
                    t.cancel()
                self._sim_tasks.clear()
                self._switch_tape("binance", "binance-ws")
            elif self.mode == "binance" and FEED_MODE == "auto" \
                    and now - self.last_real_delivery > REAL_STALE_S:
                log.warning("feed: real feed silent %.0fs -> simulator",
                            now - self.last_real_delivery)
                if not self._sim_tasks:
                    for sym, st in self.states.items():
                        if st.live_price:
                            self._sim_price[sym] = st.live_price
                    await self._start_sim()
                self._switch_tape("sim", "sim")
            # revive: while in sim under auto, keep probing Binance
            if (self.mode == "sim" and FEED_MODE == "auto"
                    and now >= self._next_revive):
                self._next_revive = now + 30.0
                try:
                    async with httpx.AsyncClient(base_url=BINANCE_REST,
                                                 timeout=5.0) as c:
                        r = await c.get("/api/v3/ping")
                        r.raise_for_status()
                    await self._start_binance()   # backfill + ws task
                    self.last_real_delivery = time.time()
                    for t in list(self._sim_tasks):
                        t.cancel()
                    self._sim_tasks.clear()
                    self._switch_tape("binance", "binance-rest")
                except Exception:
                    pass

    async def start(self) -> None:
        mode = FEED_MODE
        if mode in ("auto", "binance"):
            # real-feed grace (advice §1.1): retry the real feed for up to
            # FEED_GRACE_S before the simulator is allowed to start
            deadline = time.time() + (FEED_GRACE_S if mode == "auto" else 0.0)
            while True:
                try:
                    await self._start_binance()
                    self.mode = "binance"
                    log.info("feed: Binance mode active")
                    break
                except Exception as exc:  # pragma: no cover - network dependent
                    if mode == "binance":
                        raise
                    if time.time() >= deadline:
                        log.warning("feed: Binance unavailable after %.0fs "
                                    "grace (%s); simulator may start",
                                    FEED_GRACE_S, exc)
                        break
                    await asyncio.sleep(5.0)
        if self.mode != "binance":
            await self._start_sim()
            self.mode = "sim"
        self._reconcile_task = asyncio.create_task(self._reconcile_loop())
        self._tasks.append(self._reconcile_task)

    async def stop(self) -> None:
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # ------------------------------------------------------------- Binance
    async def _start_binance(self) -> None:
        # 1m kline backfill so all four TF windows (248 bars) are ready instantly.
        # D-3: rotate to data-stream.binance.vision when the main host refuses
        # (geo-block on some Codespaces regions).
        bases = [BINANCE_REST, BINANCE_REST_VISION]
        last_exc: Exception | None = None
        rows_by_sym: dict[str, list] = {}
        async with httpx.AsyncClient(timeout=15.0) as client:
            for base in bases:
                try:
                    for sym, ex_symbol in SYMBOLS.items():
                        resp = await client.get(
                            f"{base}/api/v3/klines",
                            params={"symbol": ex_symbol, "interval": "1m",
                                    "limit": BACKFILL_BARS},
                        )
                        resp.raise_for_status()
                        rows_by_sym[sym] = resp.json()
                    last_exc = None
                    break
                except Exception as exc:        # try the mirror host
                    last_exc = exc
            if last_exc is not None:
                raise last_exc
            for sym, rows in rows_by_sym.items():
                st = self.states[sym]
                prev_close = None
                for row in rows:
                    ts = int(row[0]) // 1000
                    o, h, l, c, v = (
                        float(row[1]), float(row[2]), float(row[3]),
                        float(row[4]), float(row[5]),
                    )
                    st.push_bar(ts, c, v, o, h, l, source="binance-rest")
                    prev_close = c
                if st.closes:
                    st.live_price = st.closes[-1]
                    st.live_open = st.opens[-1]
                    st.live_high = st.highs[-1]
                    st.live_low = st.lows[-1]
                st.day_open = st.closes[0] if st.closes else 0.0
        self.last_real_delivery = time.time()
        if all(len(self.states[s].closes) >= 30 for s in SYMBOL_ORDER):
            self.ready.set()

        self._tasks.append(asyncio.create_task(self._binance_ws_loop()))

    async def _binance_ws_loop(self) -> None:  # pragma: no cover - network
        streams = "/".join(f"{ex.lower()}@kline_1m" for ex in SYMBOLS.values())
        urls = [f"{BINANCE_WS}?streams={streams}",
                f"{BINANCE_WS_VISION}?streams={streams}"]   # D-3 rotation
        import websockets

        reverse = {v.lower(): k for k, v in SYMBOLS.items()}
        url_i = 0
        while not self._stop.is_set():
            try:
                async with websockets.connect(urls[url_i], ping_interval=20) as ws:
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
                        self.last_real_delivery = time.time()
                        if not self._ingest_ok("binance-ws"):
                            continue          # D-1: foreign writer rejected
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
                            source="binance-ws",
                        )
                        await self._emit_tick(sym, price, float(k["v"]))
            except asyncio.CancelledError:
                raise
            except Exception as exc:                        # reconnect + rotate
                log.warning("feed: ws reconnect (%s)", exc)
                url_i = (url_i + 1) % len(urls) if FEED_MODE == "auto" else url_i
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
                # never overwrite bars a real feed already wrote (D-1)
                if not (st.times and ts <= st.times[-1]):
                    st.push_bar(ts, price, volume, o, h, l, source="sim")
            st.live_price = price
            st.live_open = st.opens[-1]
            st.live_high = st.highs[-1]
            st.live_low = st.lows[-1]
            self._sim_price[sym] = price
            # day_open for percent display: bar from ~24h ago if available
            st.day_open = st.closes[0] if st.closes else price
            # live-regime seed: the tick loop CONTINUES these regimes (the old
            # driftless loop flattened every trend right after backfill, which
            # collapsed CCI/confidence to ~0 forever)
            self._sim_regime[sym] = {
                "drift": drift,
                "left": rng.randint(600, 2400),   # 10–40 min coherent windows
                "vol_mult": vol_mult,
            }

        self.ready.set()
        t = asyncio.create_task(self._sim_tick_loop())
        self._sim_tasks.append(t)
        self._tasks.append(t)

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
            if self.mode != "sim":
                continue               # real family owns the tape now (D-1)
            now = time.time()
            minute = int(now // 60)
            rolled = minute != last_minute
            for sym in SYMBOL_ORDER:
                st = self.states[sym]
                price = self._sim_price[sym]
                sigma = {"BTC": 7e-5, "PAXG": 2e-5, "ETH": 9e-5, "BNB": 8e-5}[sym]
                # persistent regime: coherent multi-minute drift windows so the
                # 22m+ formula windows see real trends (spec: confidence floor
                # is only organically reachable with trending data)
                reg = self._sim_regime[sym]
                reg["left"] -= 1
                if reg["left"] <= 0:
                    reg["left"] = rng.randint(600, 2400)
                    reg["drift"] = rng.choice([-0.00045, -0.0002, 0.0,
                                               0.0002, 0.00045])
                    if sym == "PAXG":
                        reg["drift"] *= 0.25          # gold stays peg-anchored
                    reg["vol_mult"] = rng.choice([0.7, 1.0, 1.6, 2.4])
                price *= math.exp(rng.gauss(reg["drift"],
                                            sigma * reg["vol_mult"]))
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
                        source="sim",
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
        if volume <= 0:
            self.quote_prints += 1      # D-7: quote print, excluded from vol stats
        self.note("binance-ws" if self.mode == "binance" else "sim")
        if self.on_tick is not None:
            try:
                result = self.on_tick(symbol, price, volume)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                log.exception("feed: tick callback failed")
