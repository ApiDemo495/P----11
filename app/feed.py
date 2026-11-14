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
    """Return a verified writer family; unknown lookalikes fail closed."""
    name = str(source or "").lower()
    if name in {"binance", "binance-rest", "binance-ws"}:
        return "binance"
    if name == "sim":
        return "sim"
    return "unknown"


def _valid_ohlcv(ts, open_, high, low, close, volume, *,
                 require_minute_boundary: bool = True) -> bool:
    """Validate one OHLCV row; exchange-family bars are epoch-minute aligned."""
    try:
        raw_ts = float(ts)
        stamp = int(raw_ts)
        values = tuple(float(value) for value in
                       (open_, high, low, close, volume))
    except (TypeError, ValueError, OverflowError):
        return False
    o, h, lo, c, v = values
    return (math.isfinite(raw_ts) and raw_ts == stamp and stamp > 0
            and (not require_minute_boundary or stamp % 60 == 0)
            and all(math.isfinite(value) for value in values)
            and min(o, h, lo, c) > 0 and v >= 0
            and h >= max(o, lo, c) and lo <= min(o, h, c))


class SymbolState:
    __slots__ = (
        "symbol", "times", "closes", "volumes",
        "opens", "highs", "lows", "sources",
        "live_price", "live_volume", "live_open", "live_high", "live_low",
        "day_open", "last_tick",
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
        self.last_tick: float = 0.0   # wall time of last live update

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
        if not _valid_ohlcv(
                ts, open_, high, low, close, volume,
                require_minute_boundary=_family(source) != "unknown"):
            return
        ts = int(float(ts))
        open_, high, low, close, volume = map(
            float, (open_, high, low, close, volume))
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
        keep = [i for i, src in enumerate(self.sources[:len(self.times)])
                if _family(src) == family]
        dropped = len(self.times) - len(keep)
        self.times = [self.times[i] for i in keep]
        self.closes = [self.closes[i] for i in keep]
        self.volumes = [self.volumes[i] for i in keep]
        self.opens = [self.opens[i] for i in keep]
        self.highs = [self.highs[i] for i in keep]
        self.lows = [self.lows[i] for i in keep]
        self.sources = [self.sources[i] for i in keep]
        return dropped

    def indices_for_family(self, family: str) -> list[int]:
        """Indices with explicit provenance; missing/unknown tags fail closed."""
        if family not in ("binance", "sim"):
            return []
        return [i for i, source in enumerate(self.sources[:len(self.times)])
                if _family(source) == family]

    def replace_history(
        self, bars: list[tuple[int, float, float, float, float, float, str]],
    ) -> None:
        """Atomically replace the tape with a sorted, deduplicated OHLC set.

        Used when REST backfill arrives after a simulator has already built a
        newer tape. `push_bar` is intentionally append/last-bucket oriented and
        must not be used to insert an old backfill behind that tape.
        """
        by_time: dict[int, tuple[float, float, float, float, float, str]] = {}
        duplicate_times: set[int] = set()
        for ts, open_, high, low, close, volume, source in bars:
            if not _valid_ohlcv(ts, open_, high, low, close, volume):
                continue
            stamp = int(float(ts))
            if stamp in by_time:
                duplicate_times.add(stamp)
                continue
            by_time[stamp] = (float(open_), float(high), float(low),
                              float(close), float(volume), str(source))
        ordered = sorted((ts, bar) for ts, bar in by_time.items()
                         if ts not in duplicate_times)
        self.times = [ts for ts, _ in ordered]
        self.opens = [bar[0] for _, bar in ordered]
        self.highs = [bar[1] for _, bar in ordered]
        self.lows = [bar[2] for _, bar in ordered]
        self.closes = [bar[3] for _, bar in ordered]
        self.volumes = [bar[4] for _, bar in ordered]
        self.sources = [bar[5] for _, bar in ordered]

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
        self._probing = False
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
        # exchange-server physics telemetry (Chrono-Sieve-inspired gauges)
        from .micro import MicroPhysics
        from .book import DepthBook
        self.micro = MicroPhysics()
        self.book = DepthBook()                    # live top-20 L2 book
        self._fo_task: asyncio.Task | None = None   # forceOrder listener

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
        del self._tick_times[:-2000]     # depth@100ms alone is ~40 msg/s

    def history_coverage(self, now: float | None = None) -> dict[str, dict]:
        """Measure active-family closed 1m candles; never infer missing bars."""
        now = time.time() if now is None else float(now)
        current = int(now // 60) * 60
        cutoff = current - 1440 * 60
        family = _family(self.mode)
        out = {}
        for symbol, state in self.states.items():
            stamps = set()
            for ts, source in zip(state.times, state.sources):
                try:
                    raw = float(ts)
                    stamp = int(raw)
                except (TypeError, ValueError, OverflowError):
                    continue
                if (_family(source) == family and math.isfinite(raw)
                        and raw == stamp and stamp > 0 and stamp % 60 == 0
                        and stamp < current):
                    stamps.add(stamp)
            ordered = sorted(stamps)
            recent = [ts for ts in ordered if cutoff <= ts < current]
            gaps = [b - a for a, b in zip(recent, recent[1:]) if b - a != 60]
            complete = (len(recent) == 1440 and recent[0] == cutoff
                        and recent[-1] == current - 60 and not gaps)
            out[symbol] = {
                "family": family, "bars": len(ordered),
                "last_24h_bars": len(recent), "expected_24h_bars": 1440,
                "continuous_24h": complete,
                "from": ordered[0] if ordered else None,
                "to": ordered[-1] if ordered else None,
                "gaps_24h": len(gaps),
                "largest_gap_minutes": max(gaps, default=0) // 60,
                **({} if ordered else {"why": "no active-family source candles"}),
            }
        return out

    def stats(self) -> dict:
        """Feed state with provenance and measured historical coverage."""
        now = time.time()
        recent = [t for t in self._tick_times if now - t <= 10.0]
        tps = len(recent) / 10.0
        age = round(now - self.last_delivery, 1) if self.last_delivery else None
        stale = (age is None) or age > 30.0
        history = self.history_coverage(now)
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
            "bootstrap_span_min": max(
                (st.times[-1] - st.times[0]) / 60.0
                for st in self.states.values() if st.times) if any(
                st.times for st in self.states.values()) else 0.0,
            "bars_sim": sum(1 for st in self.states.values()
                            for src in st.sources if _family(src) == "sim"),
            "history_24h": history,
            "history_complete_24h": bool(history) and all(
                row.get("continuous_24h") for row in history.values()),
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
        """Switch writer families without ever mixing their candle history.

        Thin live history is an empty/short live tape, not permission to label
        simulator candles as Binance. Likewise, on demotion only simulator
        candles remain visible. Missing live prices are cleared until the new
        writer supplies a print.
        """
        if new_mode == self.mode:
            return
        if new_mode not in ("binance", "sim"):
            raise ValueError(f"unknown feed mode: {new_mode}")
        for st in self.states.values():
            st.flush_foreign(new_mode)
            if new_mode == "binance":
                st.last_tick = 0.0
                if st.times:
                    st.live_price = st.closes[-1]
                    st.live_volume = st.volumes[-1]
                    st.live_open = st.opens[-1]
                    st.live_high = st.highs[-1]
                    st.live_low = st.lows[-1]
                else:
                    st.live_price = st.live_volume = 0.0
                    st.live_open = st.live_high = st.live_low = 0.0
                    st.day_open = 0.0
            elif st.times and not st.live_price:
                st.live_price = st.closes[-1]
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
            # backfill retry: adopted but klines came up short (or the
            # first fetch was geo-refused) — keep pulling until every
            # window is full, so formulas never sit on a 4-bar tape
            if (self.mode == "binance"
                    and (any(len(st.indices_for_family("binance")) < BACKFILL_BARS
                             for st in self.states.values())
                         or not all(row.get("continuous_24h")
                                    for row in self.history_coverage(now).values()))
                    and now >= getattr(self, "_next_backfill", 0.0)):
                self._next_backfill = now + 30.0
                try:
                    await self._fetch_backfill()
                    log.info("feed: backfill retry delivered %s",
                             {k: len(v.closes)
                              for k, v in self.states.items()})
                except Exception as exc:
                    log.info("feed: backfill retry failed (%s)", exc)
            # revive: while in sim under auto, keep probing Binance
            if (self.mode == "sim" and FEED_MODE == "auto"
                    and now >= self._next_revive and not self._probing):
                self._next_revive = now + 30.0
                self._probing = True
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
                finally:
                    self._probing = False

    async def start(self) -> None:
        mode = FEED_MODE
        if mode == "binance":
            # strict: the real feed or nothing (retries under the grace)
            deadline = time.time() + FEED_GRACE_S
            while True:
                try:
                    await self._start_binance()
                    self.mode = "binance"
                    log.info("feed: Binance mode active")
                    break
                except Exception as exc:  # pragma: no cover - network dependent
                    if time.time() >= deadline:
                        raise
                    await asyncio.sleep(5.0)
        else:
            # auto: the simulator starts IMMEDIATELY so the tape is never
            # empty at boot (formula/emotion windows must exist from the
            # first compute — an empty tape made every reading 0.00).
            # The real feed is probed in the background and ADOPTED the
            # moment it delivers (advice D-1/D-2: sim runs under real).
            await self._start_sim()
            self.mode = "sim"
            self._tasks.append(asyncio.create_task(self._try_binance_once()))
        self._reconcile_task = asyncio.create_task(self._reconcile_loop())
        self._tasks.append(self._reconcile_task)

    async def _try_binance_once(self) -> None:
        """Background probe: adopt Binance as soon as it answers — never
        blocking the tape the UI and formulas already depend on."""
        await asyncio.sleep(2.0)          # let the sim tape settle first
        if FEED_MODE != "auto" or self.mode == "binance":
            return
        if self._probing:
            return
        self._probing = True
        try:
            await self._start_binance()
            self.last_real_delivery = time.time()
            for t in list(self._sim_tasks):
                t.cancel()
            self._sim_tasks.clear()
            self._switch_tape("binance", "binance-rest")
            log.info("feed: real feed adopted over the simulator (D-2)")
        except Exception as exc:
            log.info("feed: real feed not available yet (%s) — "
                     "simulator keeps the tape; retrying in reconcile", exc)
        finally:
            self._probing = False

    async def stop(self) -> None:
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # ------------------------------------------------------------- Binance
    async def _fetch_symbol_klines(self, client, base: str, exchange_symbol: str,
                                   *, end_ms: int | None = None,
                                   count: int | None = None) -> list[tuple]:
        """Paginate verified 1m klines; absent provider minutes stay absent."""
        wanted = max(0, int(BACKFILL_BARS if count is None else count))
        if not wanted:
            return []
        if end_ms is None:
            current_ms = int(time.time() // 60) * 60_000
            end_ms = current_ms - 1            # closed candles only
        end_ms = int(end_ms)
        end_bucket = end_ms // 60_000 * 60_000
        cursor = max(0, end_bucket - (wanted - 1) * 60_000)
        accepted: dict[int, tuple] = {}
        while cursor <= end_ms and len(accepted) < wanted:
            limit = min(1000, wanted - len(accepted))
            response = await client.get(
                f"{base}/api/v3/klines",
                params={"symbol": exchange_symbol, "interval": "1m",
                        "startTime": cursor, "endTime": end_ms,
                        "limit": limit})
            response.raise_for_status()
            rows = response.json()
            if not isinstance(rows, list) or not rows:
                break
            max_stamp = None
            added = 0
            for row in rows:
                try:
                    raw_ms = float(row[0])
                    stamp_ms = int(raw_ms)
                    stamp = stamp_ms // 1000
                    o, high, low, close, volume = (float(row[i])
                                                   for i in (1, 2, 3, 4, 5))
                except (IndexError, TypeError, ValueError, OverflowError):
                    continue
                max_stamp = max(max_stamp or stamp_ms, stamp_ms)
                if (not math.isfinite(raw_ms) or raw_ms != stamp_ms
                        or stamp_ms % 60_000 or stamp_ms < cursor
                        or stamp_ms > end_ms or stamp in accepted
                        or not _valid_ohlcv(stamp, o, high, low, close, volume)):
                    continue
                accepted[stamp] = (stamp, o, high, low, close, volume,
                                   "binance-rest")
                added += 1
            if max_stamp is None or max_stamp < cursor:
                break
            next_cursor = max_stamp + 60_000
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if added == 0 and max_stamp < end_ms - 60_000:
                break
        return [accepted[ts] for ts in sorted(accepted)][-wanted:]

    async def _fetch_backfill(self) -> int:
        """Paginated real 1m history from Binance, then its official mirror."""
        bases = [BINANCE_REST, BINANCE_REST_VISION]
        rows_by_sym: dict[str, list[tuple]] | None = None
        used_base = bases[0]
        last_exc: Exception | None = None
        end_ms = int(time.time() // 60) * 60_000 - 1
        async with httpx.AsyncClient(timeout=15.0) as client:
            for base in bases:
                try:
                    candidate = {}
                    for sym, ex_symbol in SYMBOLS.items():
                        candidate[sym] = await self._fetch_symbol_klines(
                            client, base, ex_symbol, end_ms=end_ms,
                            count=BACKFILL_BARS)
                    rows_by_sym = candidate
                    used_base = base
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
            if rows_by_sym is None:
                if last_exc is not None:
                    raise last_exc
                return 0
            pushed = 0
            for sym, rest_bars in rows_by_sym.items():
                st = self.states[sym]
                pushed += len(rest_bars)
                rest_latest = max((bar[0] for bar in rest_bars), default=None)
                by_time = {bar[0]: bar for bar in rest_bars}
                # Retain verified WS candles newer than the REST snapshot,
                # and let a WS update own the overlapping final minute.
                for i, ts in enumerate(st.times[:len(st.sources)]):
                    src = st.sources[i]
                    if (_family(src) == "binance" and
                            (rest_latest is None or ts > rest_latest or
                             (ts == rest_latest and src == "binance-ws"))):
                        row = (ts, st.opens[i], st.highs[i], st.lows[i],
                               st.closes[i], st.volumes[i], src)
                        if _valid_ohlcv(ts, *row[1:6]):
                            by_time[ts] = row
                st.replace_history(list(by_time.values()))
                if st.closes:
                    st.live_price = st.closes[-1]
                    st.live_volume = st.volumes[-1]
                    st.live_open, st.live_high, st.live_low = (
                        st.opens[-1], st.highs[-1], st.lows[-1])
                    st.day_open = (st.closes[-1440] if len(st.closes) >= 1440
                                   else st.closes[0])
                else:
                    st.live_price = st.live_volume = st.day_open = 0.0
                    st.live_open = st.live_high = st.live_low = 0.0
            # Binance's ticker is the authoritative daily anchor when available.
            try:
                async with httpx.AsyncClient(timeout=10.0) as ticker_client:
                    for sym, ex_symbol in SYMBOLS.items():
                        response = await ticker_client.get(
                            f"{used_base}/api/v3/ticker/24hr",
                            params={"symbol": ex_symbol})
                        response.raise_for_status()
                        open_price = float(response.json().get("openPrice") or 0)
                        if math.isfinite(open_price) and open_price > 0:
                            self.states[sym].day_open = open_price
            except Exception:
                pass                         # measured bar fallback remains
        return pushed

    async def _start_binance(self) -> None:
        # 1m kline backfill so all four TF windows (248 bars) are ready
        # instantly; D-3 mirror rotation lives in _fetch_backfill.
        await self._fetch_backfill()
        self.last_real_delivery = time.time()
        if all(len(self.states[s].closes) >= 30 for s in SYMBOL_ORDER):
            self.ready.set()

        self._tasks.append(asyncio.create_task(self._binance_ws_loop()))
        if self._fo_task is None or self._fo_task.done():
            self._fo_task = asyncio.create_task(self._forceorder_loop())
            self._tasks.append(self._fo_task)

    async def _binance_ws_loop(self) -> None:  # pragma: no cover - network
        # kline (tape) + aggTrade (micro telemetry: T/E clocks + trade-id
        # gaps — the clock-skew tomography input). aggTrade never writes
        # the tape; it only feeds the gauges.
        streams = "/".join(
            f"{ex.lower()}@kline_1m/{ex.lower()}@aggTrade"
            f"/{ex.lower()}@depth20@100ms"
            for ex in SYMBOLS.values())
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
                        ev = data.get("e")
                        # every stream message is a tick for the rate
                        # meter (was: kline only -> "0.0 ticks/s" while
                        # depth/aggTrade poured in)
                        self.note("binance-ws")
                        # Partial-book depth20 sends "bids"/"asks" (full
                        # top-20 replacement each 100 ms); diff-depth
                        # sends "b"/"a". The old handler only accepted
                        # b/a, so the LIVE book never populated and
                        # F32-F35 read 0.000 on every live tape.
                        _bb = data.get("bids", data.get("b"))
                        _aa = data.get("asks", data.get("a"))
                        if isinstance(_bb, list) and isinstance(_aa, list):
                            # partial-book payloads carry NO "s" field —
                            # the symbol lives in the combined-stream
                            # name ("btcusdt@depth20@100ms")
                            _sym_raw = str(data.get("s") or "").lower() or \
                                str(msg.get("stream") or "").split("@")[0]
                            _bs = reverse.get(_sym_raw)
                            if _bs:
                                try:
                                    self.book.on_depth(
                                        _bs, _bb or [], _aa or [],
                                        int(data.get("E") or 0))
                                except Exception:
                                    pass
                            continue
                        if ev == "aggTrade":
                            # exchange physics: T (engine) vs E (gateway)
                            _sym = reverse.get(str(data.get("s") or "").lower())
                            if _sym:
                                try:
                                    self.micro.on_aggtrade(
                                        _sym,
                                        int(data.get("T") or 0),
                                        int(data.get("E") or 0),
                                        int(data.get("a") or 0),
                                        float(data.get("p") or 0),
                                        float(data.get("q") or 0))
                                    self.book.on_aggtrade(
                                        _sym,
                                        int(data.get("T") or 0),
                                        float(data.get("q") or 0),
                                        bool(data.get("m")))
                                except Exception:
                                    pass
                            continue
                        if ev != "kline":
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
                        st.last_tick = time.time()
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

    async def _forceorder_loop(self) -> None:  # pragma: no cover - network
        """Public USDⓈ-M futures liquidation prints (@forceOrder) —
        telemetry only: never touches the tape or its honesty gates."""
        import websockets
        url = ("wss://fstream.binance.com/stream"
               "?streams=!forceOrder@arr")
        reverse = {v.lower(): k for k, v in SYMBOLS.items()}
        while not self._stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    async for raw in ws:
                        msg = json.loads(raw)
                        data = msg.get("data", msg)
                        if data.get("e") != "forceOrder":
                            continue
                        o = data.get("o") or {}
                        sym = reverse.get(str(o.get("s") or "").lower())
                        if not sym:
                            continue
                        px = float(o.get("ap") or o.get("p") or 0)
                        qty = float(o.get("q") or 0)
                        self.micro.on_forceorder(sym, str(o.get("S") or ""),
                                                 px, qty)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.debug("feed: forceOrder stream reconnect (%s)", exc)
                await asyncio.sleep(10.0)

    def micro_status(self, symbol: str) -> dict:
        """Clock-skew / gap / liquidation gauges for one asset — honest
        absence when the tape has no live exchange packets (sim)."""
        try:
            out = self.micro.status(symbol,
                                    self.prices().get(symbol, 0.0))
            if not out.get("agg_count") and not out.get("liq_count"):
                sim = self.mode == "sim" or not str(self.source).startswith(
                    "binance")
                out.update({
                    "available": False,
                    "reason": ("sim tape — exchange microstructure "
                               "(clock-skew, forceOrder) does not exist "
                               "in the simulator" if sim else
                               "waiting for the first live micro packet "
                               "(aggTrade / forceOrder)"),
                    "mode": self.mode,
                })
            else:
                out.update({"available": True, "reason": "",
                            "mode": self.mode})
            return out
        except Exception as exc:
            return {"available": False, "reason": f"error: {exc}",
                    "mode": self.mode}

    def book_status(self, symbol: str) -> dict:
        """Live depth physics for the systems payload (honest cold state)."""
        try:
            return self.book.status(symbol,
                                    self.prices().get(symbol, 0.0))
        except Exception as exc:
            return {"book": False, "reason": f"error: {exc}"}

    def book_view(self, symbol: str) -> dict:
        """Rings handed to FormulaContext (F32/F33/F35 read these)."""
        try:
            return self.book.view(symbol)
        except Exception:
            return {"dbi_ring": [], "dev_ring": [], "lambda_ring": [],
                    "latest": {"book": False}}

    def micro_flow(self, symbol: str) -> float | None:
        """Net forced-flow share in [-1,1] or None (no data) — feeds the
        confidence micro-band; never sets a direction on its own."""
        if not self.micro.agg_count and not self.micro.liq_count:
            return None
        return self.micro.flow(symbol)

    # ---------------------------------------------------------- Simulator
    async def _start_sim(self) -> None:
        rng = self._sim_rng
        # Align synthetic bars to the same epoch-minute buckets as Binance.
        # Without this, the final sim bar could be :47, so Binance's :00 bar
        # was rejected as out-of-order during handover.
        now_minute = int(time.time() // 60) * 60
        start_ts = now_minute - (BACKFILL_BARS + 5) * 60

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
            st.last_tick = time.time()
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
                st.last_tick = time.time()
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
                # the partial minute lives ON the tape, continuously —
                # same semantics as the Binance kline handler (the sim
                # used to hide it until the roll, so every consumer of
                # closes[-1] lagged the live price by up to a minute)
                st.push_bar(
                    minute * 60, price, st.live_volume,
                    st.live_open or price, st.live_high or price,
                    st.live_low or price, source="sim",
                )
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
