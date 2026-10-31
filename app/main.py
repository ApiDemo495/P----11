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
from fastapi.responses import JSONResponse, HTMLResponse
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

app = FastAPI(title="Crypto Prediction 4", version="1.0.0",
              docs_url="/api-docs", redoc_url="/api-redoc")


@app.on_event("startup")
async def _startup() -> None:
    feed.on_tick = _on_feed_tick
    await feed.start()
    await engine.start()
    await clock.start()
    global _learning_task
    _learning_task = asyncio.create_task(_hourly_learning_loop())
    log.info("application started (feed=%s)", feed.mode)


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _learning_task is not None:
        _learning_task.cancel()
    await clock.stop()
    await engine.stop()
    await feed.stop()


_last_tick_broadcast = 0.0

_learning_task: asyncio.Task | None = None


async def _hourly_learning_loop() -> None:
    """Section 10 background safety net: guarantees the macro cycle fires
    at least once per wall-clock hour even if GRCs stall (the engine
    itself also triggers it every 15 GRCs).  Mirrors scripts/learning_cron.sh
    without needing system cron."""
    while True:
        await asyncio.sleep(600)          # check every 10 minutes
        try:
            now = time.time()
            lm = engine.learning
            if now - (lm.last_macro_ts or 0.0) >= 3600.0:
                lm.macro_cycle(engine)
                lm.maybe_deep(engine)     # wall-clock deep cycle too
                lm.save(engine)
                log.info("hourly learning safety net fired "
                         "(macro=%d deep=%d)", lm.macro_count, lm.deep_count)
        except Exception:
            log.exception("hourly learning loop error")


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
    engine.news.refresh()   # top up the sim stream — never show an empty feed
    return JSONResponse({"ok": True, **engine.news.aggregate(symbol)})


@app.get("/api/candles")
async def candles(symbol: str = "BTC", n: int = 180, tf: str = "tf1") -> JSONResponse:
    """OHLC bars (incl. the live partial minute). tf=tf1 → raw 1-minute
    bars; tf2/tf3/tf4 aggregate into 44/130/247-minute windows per config."""
    if symbol not in SYMBOLS:
        return JSONResponse({"ok": False, "error": "unknown symbol"},
                            status_code=400)
    n = max(10, min(500, n))
    st = engine.feed.state(symbol)
    count = len(st.times)
    W = int(TIMEFRAMES.get(tf, {}).get("minutes", 1) or 1)
    if tf == "tf1":
        W = 1        # tf1 keeps raw 1-minute candles (the 22m window is the
                     # prediction horizon; traders watch 1m bars inside it)
    if W <= 1:
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
    else:
        # epoch-aligned buckets of W minutes
        bucket: dict[int, dict] = {}
        for i in range(count):
            bkt = st.times[i] // (W * 60) * (W * 60)
            b = bucket.get(bkt)
            if b is None:
                bucket[bkt] = {"t": bkt, "o": st.opens[i], "h": st.highs[i],
                               "l": st.lows[i], "c": st.closes[i], "v": st.volumes[i]}
            else:
                b["h"] = max(b["h"], st.highs[i])
                b["l"] = min(b["l"], st.lows[i])
                b["c"] = st.closes[i]
                b["v"] += st.volumes[i]
        bars = list(bucket.values())[-n:]
        if st.live_price > 0 and bars:
            bkt = int(time.time() // (W * 60)) * (W * 60)
            if bkt == bars[-1]["t"]:
                bars[-1]["h"] = max(bars[-1]["h"], st.live_price, st.live_high or 0)
                bars[-1]["l"] = min(bars[-1]["l"], st.live_price, st.live_low or 1e18)
                bars[-1]["c"] = st.live_price
                bars[-1]["live"] = True
            else:
                bars.append({"t": bkt, "o": st.live_open or st.live_price,
                             "h": st.live_high or st.live_price,
                             "l": st.live_low or st.live_price,
                             "c": st.live_price, "v": st.live_volume, "live": True})
    return JSONResponse({"ok": True, "symbol": symbol, "tf": tf,
                         "bucket_minutes": W, "bars": bars})


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
# /docs — in-app markdown viewer (README + API_SETUP). Works with ZERO
# editor extensions — fixes "markdown.showPreview not found" in any editor.
# ---------------------------------------------------------------------------
def _md(text: str) -> str:
    """Minimal safe markdown → HTML (headers, code, lists, tables, styles)."""
    import html as _h
    import re as _re
    out, in_code, in_ul, in_tbl = [], False, False, False

    def _close_lists():
        nonlocal in_ul, in_tbl
        if in_ul:
            out.append("</ul>"); in_ul = False
        if in_tbl:
            out.append("</table>"); in_tbl = False

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line.strip().startswith("```"):
            _close_lists()
            out.append("<pre><code>" if not in_code else "</code></pre>")
            in_code = not in_code
            continue
        if in_code:
            out.append(_h.escape(line)); continue
        if not line.strip():
            _close_lists(); continue
        esc = _h.escape(line)
        esc = _re.sub(r"`([^`]+)`", r"<code>\1</code>", esc)
        esc = _re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", esc)
        esc = _re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<i>\1</i>", esc)
        esc = _re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', esc)
        m = _re.match(r"^(#{1,6})\s+(.*)$", esc)
        if m:
            _close_lists(); lvl = min(len(m.group(1)) + 2, 6)
            out.append(f"<h{lvl}>{m.group(2)}</h{lvl}>"); continue
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(_re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue
            tag = "th" if not in_tbl and False else "td"
            if not in_tbl:
                out.append("<table>"); in_tbl = True
            out.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_tbl:
            out.append("</table>"); in_tbl = False
        m = _re.match(r"^\s*[-*+]\s+(.*)$", esc)
        if m:
            if not in_ul:
                out.append("<ul>"); in_ul = True
            out.append(f"<li>{m.group(1)}</li>"); continue
        if in_ul:
            out.append("</ul>"); in_ul = False
        out.append(f"<p>{esc}</p>")
    _close_lists()
    if in_code:
        out.append("</code></pre>")
    return "\n".join(out)


@app.get("/docs", response_class=HTMLResponse)
async def docs_page() -> HTMLResponse:
    """README + API_SETUP rendered in-app — no editor extension required."""
    import os as _os
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    parts = []
    for name in ("README.md", "API_SETUP.md"):
        fp = _os.path.join(root, name)
        try:
            with open(fp, encoding="utf-8", errors="replace") as fh:
                parts.append(f'<article><h1 class="doc-title">{name}</h1>{_md(fh.read())}</article>')
        except OSError:
            parts.append(f"<article><p>{name} not found</p></article>")
    page = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>PREDICT · docs</title><style>
body{background:#07070d;color:#d8d8e8;font:14px/1.6 -apple-system,"Segoe UI",sans-serif;
     max-width:900px;margin:0 auto;padding:40px 24px}
h1,h2,h3,h4,h5,h6{color:#c9a227;font-family:ui-monospace,monospace;letter-spacing:.5px;margin:1.4em 0 .5em}
h1.doc-title{border-bottom:1px solid #1d1d2e;padding-bottom:8px;font-size:20px}
pre{background:#0e0e18;border:1px solid #1d1d2e;border-radius:8px;padding:14px;overflow-x:auto}
code{font-family:ui-monospace,monospace;background:#12121f;padding:1px 5px;border-radius:3px;font-size:12.5px}
a{color:#00d4f0} table{border-collapse:collapse;margin:12px 0}
th,td{border:1px solid #1d1d2e;padding:6px 12px;font-size:12.5px} th{color:#6e6e8c}
ul{padding-left:22px} li{margin:4px 0} i{color:#9aa}
a.back{display:inline-block;margin-bottom:20px;color:#c9a227}
</style></head><body><a class="back" href="/">← back to app</a>""" + \
    "".join(parts) + "</body></html>"
    return HTMLResponse(page)


# ---------------------------------------------------------------------------
# static dashboard
# ---------------------------------------------------------------------------
@app.middleware("http")
async def _no_store_ui(request, call_next):
    """Preview proxies must never serve a stale app.js (frozen 4:00 bug)."""
    resp = await call_next(request)
    if request.url.path in ("/", "/app.js", "/style.css", "/index.html"):
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp

app.mount("/", StaticFiles(directory="static", html=True), name="static")
