"""FastAPI application — serves the dashboard and streams real-time state.

Single-process design (uvicorn --workers 1): the master clock, GRC state and
prediction engine are in-process singletons.  Between GRCs clients receive
1-second tick updates; full prediction snapshots are pushed exactly at GRC
boundaries and on the spec's event-driven exceptions.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .clock import MasterClock, seconds_to_next_grc
from .config import SYMBOLS, TF_ORDER, TIMEFRAMES
from .engine import PredictionEngine
from .feed import MarketFeed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("main")


class ConnectionManager:
    def __init__(self) -> None:
        self.clients: set[asyncio.Queue] = set()

    def connect(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self.clients.add(q)
        return q

    def disconnect(self, q: asyncio.Queue) -> None:
        self.clients.discard(q)

    async def broadcast(self, msg: dict[str, Any]) -> None:
        for q in list(self.clients):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()      # drop oldest on slow consumers
                with contextlib.suppress(asyncio.QueueFull):
                    q.put_nowait(msg)


manager = ConnectionManager()
feed = MarketFeed()


async def engine_emit(msg: dict[str, Any]) -> None:
    await manager.broadcast(msg)


engine = PredictionEngine(feed, emit=engine_emit)
clock = MasterClock(on_grc=engine.on_grc)

app = FastAPI(title="Crypto Prediction 4", version="1.0.0")


@app.on_event("startup")
async def _startup() -> None:
    feed.on_tick = _on_feed_tick
    await feed.start()
    await engine.start()
    await clock.start()
    log.info("application started (feed=%s)", feed.mode)


@app.on_event("shutdown")
async def _shutdown() -> None:
    await clock.stop()
    await engine.stop()
    await feed.stop()


_last_tick_broadcast = 0.0


async def _on_feed_tick(symbol: str, price: float, volume: float) -> None:
    """1 Hz-ish tick streaming for the UI between GRCs (no prediction changes)."""
    global _last_tick_broadcast
    now = time.time()
    if now - _last_tick_broadcast < 1.0:
        return
    _last_tick_broadcast = now
    await manager.broadcast({
        "type": "tick",
        "ts": now,
        "prices": feed.prices(),
    })


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "phase": engine.phase,
        "feed": feed.mode,
        "grc_count": engine.grc_count,
        "seconds_to_grc": round(seconds_to_next_grc(), 1),
    }


@app.get("/api/snapshot")
async def snapshot() -> JSONResponse:
    return JSONResponse(engine.build_snapshot())


@app.get("/api/formula-history")
async def formula_history(symbol: str = "BTC", tf: str = "tf1") -> JSONResponse:
    """Sparklines + accuracy rings for all 31 core / 22 emotion / 22
    physics formulas of one (asset, timeframe) — Section 9 panels."""
    if symbol not in SYMBOLS or tf not in TIMEFRAMES:
        return JSONResponse({"ok": False, "error": "unknown symbol/tf"},
                            status_code=400)
    return JSONResponse({"ok": True,
                         **engine.formula_history(symbol, tf)})


@app.get("/api/brain-geometry")
async def brain_geometry() -> JSONResponse:
    """Static 3D connectome geometry for the WebGL brain visualization."""
    return JSONResponse({"ok": True, **engine.brain_geometry()})


@app.get("/api/brain-activations")
async def brain_activations(symbol: str = "BTC", tf: str = "tf1") -> JSONResponse:
    """Per-instance neuron activations for one (asset, timeframe)."""
    if symbol not in SYMBOLS or tf not in TIMEFRAMES:
        return JSONResponse({"ok": False, "error": "unknown symbol/tf"},
                            status_code=400)
    return JSONResponse({"ok": True, **engine.brain_activations(symbol, tf)})


@app.get("/api/news/{symbol}")
async def news_feed(symbol: str) -> JSONResponse:
    """Section-7 news feed: categories, aggregate, per-TF impacts, items."""
    if symbol not in SYMBOLS:
        return JSONResponse({"ok": False, "error": "unknown symbol"},
                            status_code=400)
    return JSONResponse({"ok": True, **engine.news.aggregate(symbol)})


@app.get("/api/candles")
async def candles(symbol: str = "BTC", n: int = 180) -> JSONResponse:
    """OHLC bars for the candlestick chart (incl. the live partial minute)."""
    if symbol not in SYMBOLS:
        return JSONResponse({"ok": False, "error": "unknown symbol"},
                            status_code=400)
    n = max(10, min(500, n))
    st = engine.feed.state(symbol)
    count = len(st.times)
    start = max(0, count - n)
    bars = [
        {"t": st.times[i], "o": st.opens[i], "h": st.highs[i],
         "l": st.lows[i], "c": st.closes[i], "v": st.volumes[i]}
        for i in range(start, count)
    ]
    if st.live_price > 0 and st.live_open > 0:
        bars.append({"t": (bars[-1]["t"] + 60) if bars else int(time.time() // 60) * 60,
                     "o": st.live_open, "h": st.live_high, "l": st.live_low,
                     "c": st.live_price, "v": st.live_volume, "live": True})
    return JSONResponse({"ok": True, "symbol": symbol, "bars": bars})


@app.get("/api/systems")
async def systems(symbol: str = "BTC") -> JSONResponse:
    """Per-timeframe display systems (brain/emotion/physics/news/hedge)."""
    if symbol not in SYMBOLS:
        return JSONResponse({"ok": False, "error": "unknown symbol"}, status_code=400)
    data = {f"{symbol}|{tf}": engine.display_systems.get((symbol, tf))
            for tf in TF_ORDER if (symbol, tf) in engine.display_systems}
    return JSONResponse({"ok": True, "symbol": symbol, "systems": data})


@app.post("/api/override/{symbol}/{tf}")
async def override(symbol: str, tf: str) -> JSONResponse:
    """Manual override: forces an immediate fresh cycle for one timeframe,
    canceling the current countdown and starting fresh."""
    ok = await engine.override(symbol, tf)
    if not ok:
        return JSONResponse({"ok": False, "error": "unknown symbol/tf or not live"},
                            status_code=400)
    return JSONResponse({"ok": True, "cycle": engine.cycles[(symbol, tf)].to_dict(time.time())})


# ---------------------------------------------------------------------------
# Part 4 — learning, history, agents, settings (Sections 10/11/13/15)
# ---------------------------------------------------------------------------
@app.get("/api/learning")
async def learning_state() -> JSONResponse:
    snap = engine.last_snapshot or engine.build_snapshot()
    return JSONResponse({"ok": True, "learning": snap.get("learning", {}),
                         "shares": engine.learning.subsystem_shares,
                         "state_file": "data/auto_learning_state.json"})


@app.post("/api/learning/macro")
async def learning_macro() -> JSONResponse:
    """Manual/cron trigger for the macro learning cycle (15 GRCs)."""
    try:
        engine.learning.macro_cycle(engine)
        engine.learning.save(engine)
    except Exception as e:
        log.exception("macro trigger failed")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return JSONResponse({"ok": True, "macro_count": engine.learning.macro_count,
                         "shares": engine.learning.subsystem_shares})


@app.post("/api/learning/deep")
async def learning_deep() -> JSONResponse:
    """Manual trigger for the deep learning cycle (247 minutes)."""
    try:
        engine.learning.deep_cycle(engine)
        engine.learning.save(engine)
    except Exception as e:
        log.exception("deep trigger failed")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return JSONResponse({"ok": True, "deep_count": engine.learning.deep_count,
                         "shares": engine.learning.subsystem_shares})


@app.post("/api/learning/reset")
async def learning_reset() -> JSONResponse:
    """Reset ALL learned weights to the uniform prior (confirmation)."""
    engine.learning.reset(engine)
    engine.learning.save(engine)
    return JSONResponse({"ok": True, "message": "learning state reset"})


@app.get("/api/history/days")
async def history_days() -> JSONResponse:
    return JSONResponse({"ok": True, "days": engine.historydb.days()})


@app.get("/api/history/day")
async def history_day(date: str) -> JSONResponse:
    rows = engine.historydb.day(date)
    return JSONResponse({"ok": True, "date": date, "records": rows})


@app.get("/api/history/replay")
async def history_replay(date: str, i: int = 0) -> JSONResponse:
    rows = engine.historydb.day(date)
    if not rows or i < 0 or i >= len(rows):
        return JSONResponse({"ok": False, "error": "no record"}, status_code=404)
    return JSONResponse({"ok": True, "date": date, "i": i,
                         "total": len(rows), "record": rows[i]})


@app.get("/api/history/leaderboard")
async def history_leaderboard() -> JSONResponse:
    """Formula accuracy leaderboard (rolling 500, Section 13)."""
    from .formulas import FORMULA_NAMES
    from .config import WEIGHT_IDS as _WIDS
    rows = []
    for fid in _WIDS:
        acc = engine.learning.formula_accuracy(fid)
        if acc is not None:
            rows.append({"formula": fid,
                         "name": FORMULA_NAMES.get(fid, f"F{fid}"),
                         "accuracy": round(acc, 4)})
    rows.sort(key=lambda r: -r["accuracy"])
    return JSONResponse({"ok": True, "leaderboard": rows})


@app.get("/api/settings")
async def settings_get() -> JSONResponse:
    s = dict(engine.agents.settings)
    return JSONResponse({"ok": True, "settings": s})


@app.put("/api/settings")
async def settings_put(body: dict) -> JSONResponse:
    engine.agents.configure(**{k: v for k, v in body.items()
                               if k in engine.agents.settings})
    return JSONResponse({"ok": True, "settings": engine.agents.settings})


@app.get("/api/agents/status")
async def agents_status() -> JSONResponse:
    return JSONResponse({"ok": True, "agents": engine.agents.status(),
                         "signal": engine.last_agent_signal or {}})


@app.post("/api/agents/test/{slot}")
async def agents_test(slot: str) -> JSONResponse:
    if slot not in ("gemini", "backup1", "backup2", "backup3", "local"):
        return JSONResponse({"ok": False, "error": "unknown slot"}, status_code=400)
    res = await engine.agents.test(slot)
    return JSONResponse({"ok": True, "slot": slot, "result": res})


@app.get("/api/state/export")
async def state_export() -> JSONResponse:
    """Full learned state as a downloadable JSON object."""
    data = engine.learning.state_dict(engine)
    data["agents_settings"] = {
        k: v for k, v in engine.agents.settings.items() if "key" not in k
    }
    return JSONResponse({"ok": True, "state": data})


@app.post("/api/state/import")
async def state_import(body: dict) -> JSONResponse:
    state = body.get("state") or {}
    try:
        import json as _json
        import os as _os
        _os.makedirs("data", exist_ok=True)
        with open("data/auto_learning_state.json", "w") as fh:
            _json.dump(state, fh)
        engine.learning.load_state(engine=engine, brains=engine.brains)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return JSONResponse({"ok": True, "message": "state imported + reloaded"})


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    """Server -> client stream: snapshot / tick / phase / sl_event messages.
    Detection of a dead client happens on send (ping frames every 20s)."""
    await ws.accept()
    q = manager.connect()
    try:
        # initial state on connect
        await ws.send_json(engine.last_snapshot or engine.build_snapshot())
        if engine.phase == "analysis":
            await ws.send_json({
                "type": "phase",
                "phase": "analysis",
                "remaining_s": max(
                    0, int(engine.phase_deadline - time.time())
                ),
                "readiness": dict(engine.readiness),
                "warning": False,
            })
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=20.0)
                await ws.send_json(msg)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping", "ts": time.time()})
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("websocket error")
    finally:
        manager.disconnect(q)


# ---------------------------------------------------------------------------
# static dashboard
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")
