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
import time
from collections import deque
from typing import Any

OK_BPS = 20.0          # ≤ 0.20 %: link healthy (doc D-4 threshold)
CHECK_S = 5.0


def link_report(feed, symbol: str, now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    st = feed.state(symbol)
    mode = str(getattr(feed, "mode", "unknown"))
    active_family = mode if mode in ("binance", "sim") else "unknown"
    indices = st.indices_for_family(active_family)
    foreign_bars = len(st.times) - len(indices)
    last_i = indices[-1] if indices else None
    live = float(st.live_price or 0.0)
    last_t = int(st.times[last_i]) if last_i is not None else 0
    last_c = float(st.closes[last_i]) if last_i is not None else 0.0
    delta_bps = (abs(last_c - live) / live * 1e4) if live > 0 and last_c > 0 else None
    bar_age = (now - last_t) if last_t else None
    source_mix_ok = active_family in ("binance", "sim") and foreign_bars == 0
    ok = bool(source_mix_ok and delta_bps is not None and delta_bps <= OK_BPS
              and bar_age is not None and bar_age <= 180)
    return {
        "symbol": symbol,
        "live_price": live,
        "last_bar_t": last_t,
        "last_bar_close": last_c,
        "bar_age_s": round(bar_age, 1) if bar_age is not None else None,
        "delta_bps": round(delta_bps, 2) if delta_bps is not None else None,
        "ok": ok,
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
