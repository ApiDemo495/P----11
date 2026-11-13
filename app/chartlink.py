"""Chart-link monitor — NEW (docs11-charts Part D, productised).

One question, measured continuously on the SERVER (not inferred in the
browser): does the candle tape end at the live price? For every asset
this reports the live price, the last bar's close/time, the bar's age
and the delta in bps. A background watchdog samples it every 5 s and
keeps an incident log, so a desync is a recorded event with a
timestamp — not a screenshot argument.

Nothing here trades or predicts; it only measures the link between
the tape the chart draws and the price the ticker shows.
"""
from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from typing import Any

OK_BPS = 20.0          # ≤ 0.20 %: link healthy (doc D-4 threshold)
MAX_BAR_JUMP_BPS = 300.0  # fail closed on a ≥3% 1m gap/body (handover guard)
CHECK_S = 5.0


def candle_tape_integrity(
    bars: list[dict[str, Any]], *, live_price: float, family: str,
    foreign_bars: int = 0, now: float | None = None,
    bucket_minutes: int = 1,
) -> dict[str, Any]:
    """Fail closed on unknown provenance, stale/desynced quotes, or tape jumps.

    ``bars`` must be the actual OHLC rows intended for the chart. The
    per-candle continuity check catches a single-source-looking history that
    still jumps across a feed handover; matching only its final close is not
    sufficient. The threshold expands with bucket duration for aggregated
    candles, using the same rule for every symbol.
    """
    now = time.time() if now is None else float(now)
    bucket_minutes = max(1, int(bucket_minutes or 1))
    live = float(live_price or 0.0)
    reasons: list[str] = []
    if family not in ("binance", "sim"):
        reasons.append("unknown-active-source")
    if int(foreign_bars or 0) > 0:
        reasons.append("foreign-or-unknown-bars")
    if live <= 0 or not math.isfinite(live):
        reasons.append("live-quote-unavailable")
    if not bars:
        reasons.append("no-active-candles")

    last_close = 0.0
    last_t = 0
    delta_bps = None
    age_s = None
    max_jump_bps = 0.0
    ordered = True
    valid = True
    normalized: list[tuple[int, float, float, float, float]] = []
    for bar in bars:
        try:
            ts = int(bar["t"])
            o, h, low, close = (float(bar[k]) for k in ("o", "h", "l", "c"))
        except (KeyError, TypeError, ValueError, OverflowError):
            valid = False
            continue
        if (ts <= 0 or not all(math.isfinite(x) and x > 0
                               for x in (o, h, low, close))
                or h < max(o, low, close) or low > min(o, h, close)):
            valid = False
            continue
        if normalized and ts <= normalized[-1][0]:
            ordered = False
        normalized.append((ts, o, h, low, close))
    if not valid:
        reasons.append("invalid-ohlc")
    if not ordered:
        reasons.append("unordered-timestamps")

    if normalized:
        last_t, _, _, _, last_close = normalized[-1]
        age_s = max(0.0, now - last_t)
        max_age_s = max(180.0, bucket_minutes * 120.0)
        if last_t > now + 5.0:
            reasons.append("future-candles")
        if age_s > max_age_s:
            reasons.append("stale-candles")
        if live > 0 and last_close > 0:
            delta_bps = abs(last_close - live) / live * 10_000.0
            if delta_bps > OK_BPS:
                reasons.append("quote-candle-desync")
        for prev, curr in zip(normalized, normalized[1:]):
            prev_t, _, _, _, prev_close = prev
            curr_t, o, h, low, close = curr
            body_bps = abs(close - o) / o * 10_000.0
            gap_bps = (abs(o - prev_close) / prev_close * 10_000.0
                       if curr_t > prev_t else 0.0)
            range_bps = (h - low) / o * 10_000.0
            max_jump_bps = max(max_jump_bps, body_bps, gap_bps, range_bps)
        jump_limit = MAX_BAR_JUMP_BPS * math.sqrt(bucket_minutes)
        if max_jump_bps > jump_limit:
            reasons.append("extreme-candle-jump")

    return {
        "ok": not reasons,
        "reasons": reasons,
        "family": family if family in ("binance", "sim") else "unknown",
        "foreign_bars": int(foreign_bars or 0),
        "bars": len(normalized),
        "live_price": live,
        "last_bar_close": last_close,
        "last_bar_t": last_t,
        "bar_age_s": round(age_s, 1) if age_s is not None else None,
        "delta_bps": round(delta_bps, 2) if delta_bps is not None else None,
        "history_jump_bps": round(max_jump_bps, 2),
        "jump_limit_bps": round(MAX_BAR_JUMP_BPS * math.sqrt(bucket_minutes), 2),
    }


def link_report(feed, symbol: str, now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    st = feed.state(symbol)
    mode = str(getattr(feed, "mode", "unknown"))
    active_family = mode if mode in ("binance", "sim") else "unknown"
    indices = st.indices_for_family(active_family)
    foreign_bars = len(st.times) - len(indices)
    live = float(st.live_price or 0.0)
    chart_indices = indices[-260:]  # same maximum raw window as the dashboard
    chart_bars = [
        {"t": st.times[i], "o": st.opens[i], "h": st.highs[i],
         "l": st.lows[i], "c": st.closes[i]}
        for i in chart_indices
        if i < len(st.opens) and i < len(st.highs)
        and i < len(st.lows) and i < len(st.closes)
    ]
    source_mix_ok = active_family in ("binance", "sim") and foreign_bars == 0
    integrity = candle_tape_integrity(
        chart_bars, live_price=live, family=active_family,
        foreign_bars=foreign_bars, now=now, bucket_minutes=1)
    return {
        "symbol": symbol,
        **integrity,
        "ok": bool(source_mix_ok and integrity["ok"]),
        "bars": len(indices),
        "raw_bars": len(st.times),
        "foreign_bars": foreign_bars,
        "active_family": active_family,
        "source_mix_ok": source_mix_ok,
        "source": getattr(feed, "source", "?"),
        "mode": mode,
    }


def all_reports(feed, symbols, now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    reps = [link_report(feed, s, now) for s in symbols]
    return {
        "ok": all(r["ok"] for r in reps),
        "checked_at": now,
        "threshold_bps": OK_BPS,
        "assets": {r["symbol"]: r for r in reps},
    }


class LinkWatchdog:
    """Samples the link every CHECK_S; a desync that persists for 3
    consecutive checks becomes a logged incident (with the measured
    delta), so 'the chart froze' has a server-side paper trail."""

    def __init__(self, feed, symbols) -> None:
        self.feed = feed
        self.symbols = list(symbols)
        self.incidents: deque = deque(maxlen=50)
        self.checks = 0
        self.desync_checks = 0
        self._streak: dict[str, int] = {}
        self._task: asyncio.Task | None = None
        self.last: dict[str, Any] = {}

    async def _loop(self) -> None:
        while True:
            try:
                self.last = all_reports(self.feed, self.symbols)
                self.checks += 1
                for sym, r in self.last["assets"].items():
                    if r["ok"]:
                        self._streak[sym] = 0
                        continue
                    self.desync_checks += 1
                    self._streak[sym] = self._streak.get(sym, 0) + 1
                    if self._streak[sym] == 3:
                        self.incidents.append({
                            "ts": time.time(), "symbol": sym,
                            "delta_bps": r["delta_bps"],
                            "bar_age_s": r["bar_age_s"],
                            "live_price": r["live_price"],
                            "last_bar_close": r["last_bar_close"],
                            "foreign_bars": r["foreign_bars"],
                            "history_jump_bps": r["history_jump_bps"],
                            "integrity_reasons": r["reasons"],
                            "mode": r["mode"], "source": r["source"],
                        })
            except Exception:
                pass                      # the monitor must never crash the app
            await asyncio.sleep(CHECK_S)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    def status(self) -> dict[str, Any]:
        return {
            "checks": self.checks,
            "desync_checks": self.desync_checks,
            "incidents": list(self.incidents)[-10:],
            "last": self.last,
        }
