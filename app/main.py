"""FastAPI application — serves the dashboard and streams real-time state.

Single-process design (uvicorn --workers 1): the master clock, GRC state and
prediction engine are in-process singletons.  Between GRCs clients receive
1-second tick updates; full prediction snapshots are pushed exactly at GRC
boundaries and on the spec's event-driven exceptions.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import math
import json
import logging
import os
import threading
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from .clock import MasterClock, seconds_to_next_grc
from .config import SYMBOLS, SYMBOL_ORDER, TF_ORDER, TIMEFRAMES, BACKFILL_BARS
from .engine import PredictionEngine
from .feed import MarketFeed, _family

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


from . import loadgov  # noqa: E402  (bulk-work throttle, see loadgov.py)
from .engine import run_pass  # noqa: E402  (dedicated heavy-pass pool)

async def engine_emit(msg: dict[str, Any]) -> None:
    await manager.broadcast(msg)


engine = PredictionEngine(feed, emit=engine_emit)
clock = MasterClock(on_grc=engine.on_grc)

app = FastAPI(title="Crypto Prediction 4", version="1.0.0",
              docs_url="/api-docs", redoc_url="/api-redoc")

# §6 warmup: the 7-day candle payload is ~600 KB raw — gzip it for the
# boot fetch and every 30 s refresh (starlette built-in, 1 KB threshold)
from starlette.middleware.gzip import GZipMiddleware as _GZipMiddleware  # noqa: E402
app.add_middleware(_GZipMiddleware, minimum_size=1024)


async def _vault_loop() -> None:
    """§9 state vault (E-7): persist learned state every 30s — formula
    states, hindsight rows, meta/forward fits, outcomes, brain weights —
    so a restart never forgets the streak."""
    while True:
        try:
            await asyncio.sleep(30.0)
            engine.learning.save(engine)
            engine.cci.save()
            engine.physics.save()
            engine.forward.save()
            engine.genesis.save()
            engine.vault_last_write = time.time()
            engine.hindsight()          # reads rings; keeps payload warm
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("vault save failed")


async def _startup_watchdog() -> None:
    """§9/J-5: under --supervise, exit 75 if readiness never lands within
    30s — supervisors see a real failure instead of a hung boot."""
    import asyncio as _a
    import os as _os
    if _os.environ.get("SUPERVISE") != "1":
        return
    # cold boots (bundled-brain restore + initial analysis phase)
    # take minutes on a 4-core Codespace; the old 30 s deadline
    # killed healthy boots and left the port answering 502
    for _i in range(20):
        await _a.sleep(30.0)
        if engine.phase == "live" and all(engine.readiness.values()):
            return
        # a busy-but-progressing boot is not a failure: report it instead of
        # exiting 75 (an exit here restarts the process, which replays the
        # warm history, which delays readiness again — a restart loop)
        print(f"startup watchdog: {30 * (_i + 1)}s phase={engine.phase} "
              f"readiness={engine.readiness} warmup={getattr(engine, 'seed_info', {}).get('state')}",
              flush=True)
    print("startup watchdog: not ready after 600s — exiting 75",
          flush=True)
    _os._exit(75)


_loop_lag = {"s": 0.0}
_selfheal_gate = {"at": 0.0}


_STALL_N = {"n": 0}


def _dump_stacks(reason: str) -> None:
    """Write every thread's stack to the log when the loop goes silent.

    A lag number alone says "something is slow"; this says WHERE. Rate
    limited, and it never raises into the meter.
    """
    try:
        import faulthandler
        import sys as _sys
        if _STALL_N["n"] > 20:
            return
        _STALL_N["n"] += 1
        log.warning("event loop blocked (%s) — thread stacks follow", reason)
        faulthandler.dump_traceback(file=_sys.stderr, all_threads=True)
    except Exception:
        pass


async def _lag_meter() -> None:
    """Event-loop lag: if sync work ever blocks the loop again, this
    number (in /api/health) says so instead of a silent 502."""
    import asyncio as _a
    while True:
        t0 = time.time()
        await _a.sleep(1.0)
        _loop_lag["s"] = round(max(0.0, time.time() - t0 - 1.0), 2)
        # tell the bulk workers how starved the loop is; they yield more
        loadgov.note_lag(_loop_lag["s"])
        if _loop_lag["s"] >= 1.5:
            _dump_stacks(f"lag {_loop_lag['s']}s")
        # SELF-HEAL every 5 s: a scheduler that died or went silent is
        # restarted here, and the brain is re-armed off the event loop.
        # Neither ever needs a manual restart again.
        try:
            if time.time() - _selfheal_gate["at"] > 5.0:
                _selfheal_gate["at"] = time.time()
                await clock.watchdog()
                _heal["last"] = engine.heal()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("self-heal pass failed")


async def _boot() -> None:
    """Full engine boot, OFF the startup path: uvicorn serves the app
    shell + /api/health immediately and the boot overlay reports
    progress; a slow brain restore / feed backfill can no longer
    leave the port answering 502 (fresh-Codespace bug, 22 reports)."""
    try:
        await feed.start()
        await engine.start()
        _spawn_history_seed()
        _spawn_agent_autostart()
        await clock.start()
        log.info("application booted (feed=%s)", feed.mode)
    except Exception:
        log.exception("background boot failed — shell stays up")


def _spawn_agent_autostart() -> None:
    """Arm the local-agent downloader so it arrives without a click.

    Runs in its own thread (the installer delays then retries with
    backoff), never under pytest, and can be switched off with
    PREDICT_NO_AUTO_AGENT=1 for constrained hosts.
    """
    import os as _os
    import threading as _th
    import time as _time

    def work() -> None:
        _time.sleep(20)                     # provisioning/settle first
        if _os.environ.get("PYTEST_CURRENT_TEST"):
            return
        if _os.environ.get("PREDICT_NO_AUTO_AGENT") == "1":
            log.info("agent auto-download disabled by PREDICT_NO_AUTO_AGENT")
            return
        try:
            st = engine.agents.installer.start_auto()
            log.info("agent auto-download armed: %s", st.get("state"))
        except Exception:
            log.exception("agent auto-download could not start")

    _th.Thread(target=work, name="agent-autostart", daemon=True).start()


def _spawn_history_seed() -> None:
    """Warm-history replay (app/warmup.py), in the background at boot.

    Every subsystem is replayed walk-forward over the backfilled tape so
    the ledger, forward P(up) buckets, advanced module board, subsystem
    scorecards, meta-combiner and calibration rings are populated on the
    first render instead of showing ``n=0 (needs ≥8)`` for days.  Runs once
    per boot (rings are in-memory) and once a day for the durable tables.
    """
    import os as _os
    import threading as _th
    import time as _time

    def work():
        # Wait for the FIRST LOCK before replaying. The replay is minutes of
        # CPU and used to start 8 s after boot — exactly when the initial
        # analysis pass runs — so the two fought for the same core and the
        # board sat on "ENGINE BUSY" during the one moment the user is
        # looking at it. Ring-based panels stay empty for a few seconds
        # longer; the lock is never delayed. Capped so a board that cannot
        # lock still gets its history.
        _deadline = _time.time() + 120.0
        while _time.time() < _deadline:
            if getattr(engine, "locked", False) or engine.phase == "live":
                break
            _time.sleep(0.5)
        _time.sleep(3)                      # let the first frames settle
        if _os.environ.get("PYTEST_CURRENT_TEST"):
            engine.seed_info = {"state": "skipped-test"}
            return
        if _os.environ.get("PREDICT_SKIP_WARMUP") == "1":
            engine.seed_info = {"state": "skipped-env"}
            return
        try:
            stamp = "data/.history_seed_stamp"
            fresh = (_os.path.exists(stamp)
                     and _time.time() - _os.path.getmtime(stamp) < 86400)
            # rings reset on restart, so the replay is always worth running;
            # only the durable ledger/forward write is skipped when fresh
            rows = sum(len(v) for v in engine.ledger.rows.values())
            engine.seed_info = {"state": "seeding"}
            from .warmup import warm_history
            info = warm_history(engine, hours=168)
            if fresh and rows >= 200:
                info["state"] = ("fresh-sim" if info.get("family") == "sim"
                                 else "fresh")
                info["why"] = ("rings re-seeded on boot; durable tables "
                               "already fresh")
            engine.seed_info = info
            _os.makedirs("data", exist_ok=True)
            open(stamp, "a").close()
        except Exception:
            log.exception("warm history replay failed")
            engine.seed_info = {"state": "failed"}

    _th.Thread(target=work, daemon=True, name="warm-history").start()


@app.get("/api/history/seed")
async def history_seed_status() -> JSONResponse:
    return JSONResponse({"ok": True,
                         "seed": getattr(engine, "seed_info",
                                         {"state": "pending"}),
                         "ledger_rows": sum(
                             len(v) for v in engine.ledger.rows.values())})


@app.on_event("startup")
async def _startup() -> None:
    feed.on_tick = _on_feed_tick
    asyncio.create_task(_boot())
    asyncio.create_task(_lag_meter())
    global _learning_task
    _learning_task = asyncio.create_task(_hourly_learning_loop())
    asyncio.create_task(_startup_watchdog())
    # guide §8: bring the LOCAL agent up at boot — probe the daemon,
    # spawn `ollama serve` detached, kick the prebuilt wheel install and
    # refresh the status. Daemon thread: the event loop never waits.
    # (Never under pytest: TestClient boots the app, and the worker would
    # spawn a real installer thread mid-suite.)
    try:
        import os as _bw_os
        from .agents import start_boot_worker
        if not _bw_os.environ.get("PYTEST_CURRENT_TEST"):
            start_boot_worker(getattr(engine, "agents", None))
    except Exception as _exc:                    # pragma: no cover
        print(f"agent boot worker skipped: {_exc}")
    from .chartlink import LinkWatchdog as _LW
    app.state.link_watchdog = _LW(feed, SYMBOL_ORDER)
    app.state.link_watchdog.start()
    app.state._vault_task = asyncio.create_task(_vault_loop())
    app.state._news_ticker = asyncio.create_task(_news_ticker())
    log.info("application shell up — engine booting in background")


@app.on_event("shutdown")
async def _shutdown() -> None:
    vt = getattr(app.state, "_vault_task", None)
    if vt is not None:
        vt.cancel()
    if _learning_task is not None:
        _learning_task.cancel()
    await clock.stop()
    await engine.stop()
    await feed.stop()


_last_tick_broadcast = 0.0

_learning_task: asyncio.Task | None = None


async def _news_ticker() -> None:
    """docs11 §6 — the 4-second publisher: pure evaluation per asset,
    emergency trigger, NEWS WS ticker (REST mirrors: /api/news/impact,
    /api/news/status)."""
    await asyncio.sleep(2.0)
    while True:
        try:
            # ingest pass every tick: sim keeps dripping (rate-aware for
            # the live source) so the News tab is a continuous stream.
            # OFF the event loop: refresh() may do a blocking httpx.get
            # (timeout=8s) when a live key is configured — inline it froze
            # the whole app behind the "ENGINE BUSY" banner (issue 1).
            await asyncio.to_thread(engine.news.refresh)
            tick: dict[str, Any] = {"type": "NEWS", "ts": time.time(),
                                    "count": len(engine.news.items),
                                    "latest": [{
                                        "ts": it.get("ts"),
                                        "headline": it.get("headline"),
                                        "source": it.get("source"),
                                        "synthetic": bool(it.get("synthetic")),
                                    } for it in list(engine.news.items)[-3:]],
                                    "assets": {}}
            for sym in SYMBOLS:
                ev = engine.news.evaluate(sym)
                tick["assets"][sym] = {
                    "coverage": engine.news.coverage_string(),
                    "horizons": {tf: {
                        "impact": h.get("impact"), "niv": h.get("niv"),
                        "smd": h.get("smd"), "n_items": h.get("n_items"),
                    } for tf, h in (ev.get("horizons") or {}).items()},
                    "top_theme": ((ev.get("horizons") or {})
                                  .get("tf1", {}).get("drivers") or [{}]
                                  )[0].get("theme"),
                    "critical": ev.get("critical_count", 0),
                }
                trig = ev.get("emergency_trigger")
                if trig:
                    out = engine.emergency_relock(
                        sym, "tf1", trig.get("reason") or "", automatic=True)
                    tick["assets"][sym]["emergency"] = out
            await manager.broadcast(tick)
        except asyncio.CancelledError:
            return
        except Exception:
            log.debug("news ticker error", exc_info=True)
        await asyncio.sleep(4.0)


async def _hourly_learning_loop() -> None:
    """Section 10 background safety net: guarantees the macro cycle fires
    at least once per wall-clock hour even if GRCs stall (the engine
    itself also triggers it every 15 GRCs).  in-process hourly loop
    without needing system cron."""
    while True:
        await asyncio.sleep(600)          # check every 10 minutes
        try:
            now = time.time()
            lm = engine.learning
            if now - (lm.last_macro_ts or 0.0) >= 3600.0:
                # off the loop: macro/deep cycles walk the whole ledger
                await asyncio.to_thread(lm.macro_cycle, engine)
                await asyncio.to_thread(lm.maybe_deep, engine)
                await asyncio.to_thread(lm.save, engine)
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
def _git_commit() -> str:
    try:
        import subprocess as _sp
        from pathlib import Path as _P
        return _sp.run(["git", "rev-parse", "--short", "HEAD"],
                       cwd=str(_P(__file__).parent.parent),
                       capture_output=True, text=True, timeout=5).stdout.strip() or "?"
    except Exception:
        return "?"


_COMMIT = _git_commit()


_BOOT_TS = time.time()
_heal: dict[str, Any] = {"last": None}


@app.get("/api/health")
async def health() -> dict:
    import time as _t
    age = round(_t.time() - getattr(engine, "last_window_ts", _t.time()), 1)
    fw = getattr(getattr(engine, "brains", None), "fw", None)
    return {
        "ok": True,
        "commit": _COMMIT,
        "uptime_s": round(_t.time() - _BOOT_TS, 1),
        "loop_lag_s": _loop_lag["s"],
        # scheduler liveness: a dead/silent GRC clock is the difference
        # between "live board" and "frozen board that still answers HTTP"
        "clock": clock.status(),
        "busy": bool(getattr(engine, "_busy", None) is not None
                     and engine._busy.locked()),
        "brain": {"state": getattr(fw, "state", None),
                  "pass_age_s": (round(_t.time() - fw.last_pass_at, 1)
                                 if getattr(fw, "last_pass_at", 0) else None),
                  "load_ms": getattr(fw, "load_ms", None)},
        "snapshot_ms": _snap_cache["ms"],
        # full wall time of the last GRC pass (the clock's handler_ms only
        # sees the coroutine, and the CPU now lives on worker threads)
        "grc_ms": getattr(engine, "grc_ms", None),
        # per-stage cost of the last GRC: the banner is only ever as honest as
        # this breakdown is visible
        "grc_profile": getattr(engine, "grc_profile", {}),
        # visible proof that a stuck readiness dry-run backs off instead of
        # re-burning the CPU every GRC
        "readiness": {"flags": dict(engine.readiness),
                      "dry_run_tries": getattr(engine, "_readiness_tries", 0),
                      "next_dry_run_in_s": round(
                          max(0.0, getattr(engine, "_readiness_next_at", 0.0)
                              - _t.time()), 1),
                      "last_error": getattr(engine, "_readiness_error", "")},
        "warmup": getattr(engine, "seed_info", {}),
        "heal": _heal["last"],
        "phase": engine.phase,
        "feed": feed.mode,
        "feed_stats": feed.stats(),
        "grc_count": engine.grc_count,
        "seconds_to_grc": round(seconds_to_next_grc(), 1),
        # watchdog (advice 9): a stalled window must be visible
        "seconds_since_window": age,
        "stalled": age > (TIMEFRAMES["tf4"]["minutes"] * 60 + 45),
        "locks": len(engine.locks),
        "ledger_n": sum(1 for rows in engine.ledger.rows.values()
                        for r in rows if r.get("result") in ("win", "loss")),
    }


@app.get("/api/watchdog")
async def watchdog() -> JSONResponse:
    """§9/J-5 watchdog: stalled window detection with an explicit alert
    state (feeds /api/degrade L4)."""
    now = time.time()
    age = now - engine.last_window_ts
    stalled = age > (TIMEFRAMES["tf4"]["minutes"] * 60 + 45)
    engine._watchdog = {
        "alert": stalled,
        "seconds_since_window": round(age, 1),
        "threshold_s": TIMEFRAMES["tf4"]["minutes"] * 60 + 45,
        "last_window_ts": engine.last_window_ts,
        "checked_at": now,
    }
    return JSONResponse({
        "ok": True, "stalled": stalled,
        "seconds_since_window": round(age, 1),
        "threshold_s": engine._watchdog["threshold_s"],
        "alert": stalled,
        "action": ("window advance missed — L4 shown on the panel"
                   if stalled else "healthy"),
    })


def _clean_json(obj, bad: set[str] | None = None, path: str = ""):
    """Replace NaN/±Inf with None (JSON has no such literals).

    One stray NaN used to 500 the ENTIRE snapshot — every panel blank.
    A single unavailable number must degrade to "—", not take the page
    down; the offending path is logged once per boot so the cause is
    still visible instead of silently papered over.
    """
    if isinstance(obj, float) and not math.isfinite(obj):
        if bad is not None:
            bad.add(path or "<root>")
        return None
    if isinstance(obj, dict):
        return {k: _clean_json(v, bad, f"{path}.{k}" if path else str(k))
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v, bad, f"{path}[{i}]") for i, v in enumerate(obj)]
    return obj


_NONFINITE_SEEN: set[str] = set()


def _json_safe(payload):
    bad: set[str] = set()
    out = _clean_json(payload, bad)
    new = {b.split(".")[:3].__str__() for b in bad} - _NONFINITE_SEEN
    if new:
        _NONFINITE_SEEN.update(new)
        log.warning("non-finite numbers replaced with null at: %s",
                    sorted(bad)[:8])
    return out


_snap_cache: dict[str, Any] = {"at": 0.0, "body": None, "bytes": b"",
                               "ms": 0.0, "building": False}
_snap_lock = threading.Lock()
# One build in flight, shared. When /api/snapshot and the GRC broadcast ask
# for a snapshot at the same moment, the second caller WAITS for the first
# result instead of starting a duplicate build (a 2.8 s build run twice is
# what starved /api/health into a 1.6 s round-trip).
_snap_flight: dict[str, Any] = {"owner": None, "done": None, "body": None,
                                "raw": b"", "ms": 0.0}
# The snapshot is rebuilt on EVERY GRC (4 min) and a few seconds after it, so
# the TTL only has to survive a burst of panel polls — not to be a refresh
# interval. It used to be 0.75 s against a 1.4 s build, i.e. every single REST
# poll started a rebuild under the lock, and every request queued behind it.
# That treadmill is the "ENGINE BUSY" stall: /api/health and the websocket
# waited on it. 3 s (still well under the 4-min contract) plus single-flight
# means a burst of polls costs ONE build.
SNAPSHOT_TTL_S = 3.0


def _snapshot_payload() -> dict:
    """Build (or reuse a ``SNAPSHOT_TTL_S``-young) snapshot, single-flight.

    Snapshot assembly is pure CPU (levels, charts, subsystems). Running it
    inline on the event loop stalled every other request — including /ws
    ticks and /api/health — for tens of seconds under load, and the UI
    banner then said "ENGINE OFFLINE" while the process was healthy. The
    route is a plain `def` (FastAPI runs it in its threadpool); the result
    is memoised and concurrent callers SHARE one in-flight build.
    """
    now = time.time()
    if (_snap_cache["body"] is not None
            and now - _snap_cache["at"] < SNAPSHOT_TTL_S):
        return _snap_cache["body"]
    # Single-flight: the first caller builds, later callers SHARE that build.
    # Before this they each started their own, so a GRC broadcast plus a panel
    # poll meant two full builds (and, with the old 0.75 s TTL, a chain of
    # them) competing with /api/health for the same core.
    with _snap_lock:
        if (_snap_flight["owner"] is not None
                and _snap_flight["owner"] != threading.get_ident()):
            done = _snap_flight["done"]
            _snap_lock.release()
            try:
                done.wait(timeout=10.0)
            finally:
                _snap_lock.acquire()
            if _snap_flight["raw"]:
                return _snap_flight["body"]
        owner = threading.get_ident()
        _snap_flight.update(owner=owner, done=threading.Event(),
                            body=None, raw=b"", ms=0.0)
        _snap_cache["building"] = True
    body = None
    raw = b""
    ms = 0.0
    try:
        t0 = time.perf_counter()
        body = _json_safe(engine.build_snapshot())
        raw = json.dumps(body, separators=(",", ":")).encode()
        ms = round((time.perf_counter() - t0) * 1000, 1)
    finally:
        with _snap_lock:
            _snap_cache["building"] = False
            if _snap_flight["owner"] == threading.get_ident():
                ev = _snap_flight["done"]
                _snap_flight.update(owner=None, body=body, raw=raw, ms=ms,
                                    done=None)
                if ev is not None:
                    ev.set()
    with _snap_lock:
        _snap_cache.update(at=time.time(), body=body, bytes=raw, ms=ms)
    return body


def prime_snapshot_cache(snap: dict) -> None:
    """Adopt a snapshot the engine just built for a broadcast.

    Called from ``engine._broadcast_full``. The GRC payload and the REST
    payload are the same object graph, so encoding it once here means the
    next panel poll is a memcpy instead of a second CPU build.
    """
    try:
        body = _json_safe(snap)
        raw = json.dumps(body, separators=(",", ":")).encode()
    except Exception:
        return
    with _snap_lock:
        # `ms` stays whatever the last REAL build cost — priming is an encode,
        # not a build. Writing it here made /api/health report a fake 0.0 ms
        # build time, i.e. a measurement that lied about itself.
        _snap_cache.update(at=time.time(), body=body, bytes=raw)


@app.get("/api/snapshot")
async def snapshot() -> Response:    # async + dedicated pool, see run_pass
    # A sync route runs in Starlette's shared request pool; a multi-second
    # snapshot build there starved every other request (measured: 3.1 s
    # /api/health round-trips during a GRC). The engine's own bounded pool
    # keeps request handling independent of analytics.
    await run_pass(_snapshot_payload)
    with _snap_lock:
        raw = _snap_cache["bytes"]
    # pre-serialised: ~1 MB of JSON used to be re-encoded per request
    return Response(content=raw, media_type="application/json")


@app.get("/api/formula-traces")
async def formula_traces(symbol: str = "BTC", tf: str = "tf1") -> JSONResponse:
    """§2 per-formula trace (µs/span/guard/zero reason/double check) —
    fetched on demand by the formulas view, never broadcast."""
    return JSONResponse({"ok": True,
                         "traces": engine.formula_traces_payload(symbol, tf)})


@app.get("/api/formula-history")
async def formula_history(symbol: str = "BTC", tf: str = "tf1") -> JSONResponse:
    """Sparklines + accuracy rings for all 31 core / 22 emotion / 22
    physics formulas of one (asset, timeframe) — Section 9 panels."""
    if symbol not in SYMBOLS or tf not in TIMEFRAMES:
        return JSONResponse({"ok": False, "error": "unknown symbol/tf"},
                            status_code=400)
    return JSONResponse({"ok": True,
                         **engine.formula_history(symbol, tf)})


@app.get("/api/rl/status")
async def rl_status() -> JSONResponse:
    """Reinforcement-learning console: error-finder actions, active
    weight dampers, issue-threshold bandit, and the HONEST accuracy
    target tracker (real number + gap — never a faked90%)."""
    from .rlsys import get_rl
    rl = get_rl()
    payload = rl.status()
    try:
        payload["target"] = rl.target_status(engine.ledger)
    except Exception:
        payload["target"] = {"target": 0.90, "current": None, "n": 0,
                             "status": "ledger unavailable"}
    try:
        payload["settled_rows"] = sum(len(v) for v in
                                      (engine.ledger.rows or {}).values())
    except Exception:
        payload["settled_rows"] = 0
    return JSONResponse({"ok": True, **payload})


@app.post("/api/explain")
async def explain_tab(body: dict) -> JSONResponse:
    """Plain-language readout for a tab (qwen2.5-coder local model,
    deterministic fallback when the model is not up yet)."""
    from . import explainer
    view = str(body.get("view") or "dashboard")
    symbol = str(body.get("symbol") or "BTC")
    tf = str(body.get("tf") or "tf1")
    if symbol not in SYMBOLS:
        return JSONResponse({"ok": False, "error": "unknown symbol"},
                            status_code=400)
    ctx = engine.explain_ctx(view, symbol, tf)

    async def _alt(prompt: str) -> dict:
        # loaded llama_cpp (or a configured cloud slot) via the agent —
        # review_text already offloads inference to a worker thread
        return await engine.agents.review_text(prompt, max_tokens=160)

    out = await explainer.explain(view, symbol, tf, ctx, alt=_alt)
    return JSONResponse(out)


@app.get("/api/think")
async def think_trace(symbol: str = "BTC", tf: str = "tf1",
                      narrate: bool = False) -> JSONResponse:
    """The app's thought process for a window: six steps, its assumptions, the
    outcome it rejected, its falsifier, and what it thinks of its own
    competence. `narrate=1` adds a language-model retelling - the model may
    only re-tell the structured trace, and its output is checked against it.
    """
    if symbol not in SYMBOLS:
        return JSONResponse({"ok": False, "error": "unknown symbol"},
                            status_code=400)
    out = engine.cognitive(symbol, tf)
    if narrate:
        out["narration"] = await _narrate(out)
    return JSONResponse(out)


async def _narrate(payload: dict) -> dict:
    """Ask the local model to re-tell the trace; verify it did not invent."""
    from .cognition import (deterministic_narration, guard_narration,
                            narration_prompt)
    fallback = deterministic_narration(payload)
    try:
        prompt = narration_prompt(payload)
        res = await engine.agents.review_text(prompt, max_tokens=320)
        text = (res.get("text") or "").strip() if res.get("ok") else ""
        if not text:
            return {"ok": True, "source": "deterministic", "text": fallback,
                    "why": res.get("reason") or "no model answered",
                    "guard": {"ok": True, "note": "no model text to check"}}
        guard = guard_narration(text, payload)
        if not guard["ok"]:
            # an invented figure is not a stylistic problem: publish the trace
            return {"ok": True, "source": "deterministic",
                    "rejected_model_text": text[:600],
                    "text": fallback, "guard": guard,
                    "why": "the model's retelling disagreed with the trace"}
        return {"ok": True, "source": res.get("backend") or "model",
                "text": text, "guard": guard}
    except Exception as exc:
        return {"ok": True, "source": "deterministic", "text": fallback,
                "why": f"narration failed: {type(exc).__name__}"}


@app.get("/api/think/history")
async def think_history(symbol: str = "BTC", tf: str = "tf1") -> JSONResponse:
    """The thought ledger: every recorded trace and how it turned out."""
    key = f"{symbol}|{tf}"
    items = list((engine._thoughts or {}).get(key) or [])
    graded = [r for r in items if r.get("outcome") in ("win", "loss")]
    wins = sum(1 for r in graded if r["outcome"] == "win")
    abstained = sum(1 for r in items if r.get("outcome") == "abstain")
    open_n = sum(1 for r in items if r.get("outcome") is None)
    return JSONResponse({
        "ok": True, "symbol": symbol, "tf": tf, "records": items[-40:],
        "n": len(items), "graded": len(graded), "wins": wins,
        "abstained": abstained, "open": open_n,
        # scope-first: the Mind tab for BTC/tf1 must describe BTC/tf1's
        # reasoning, not every scope's records pooled together
        "thought_stats": engine.thought_stats(symbol=symbol, tf=tf),
        "thought_stats_all": engine.thought_stats(),
        "rule": ("every analysis records a signature and its confidence; the "
                 "sealed prediction's settlement grades the thought that "
                 "produced it; abstentions are counted, never scored as wins"),
    })


@app.get("/api/explain/status")
async def explain_status() -> JSONResponse:
    from . import explainer
    # status() pings Ollama with a 2 s sync httpx call — off the loop (issue 1)
    return JSONResponse({"ok": True, **await asyncio.to_thread(explainer.status)})


@app.get("/api/brain/status")
def brain_status() -> JSONResponse:
    """§3.4 five-step startup verification with reasons."""
    out = dict(engine.brains.status())
    # brain.md: label the odour channels with the formulas they carry
    try:
        from .formulas import FORMULA_FUNCTIONS, FORMULA_NAMES
        fids = [f for f in sorted(FORMULA_FUNCTIONS) if f != 20]
        out.setdefault("flywire", {})["channel_map"] = [
            {"ch": i, "fid": fid,
             "name": FORMULA_NAMES.get(fid, f"F{fid}")}
            for i, fid in enumerate(fids)
        ]
    except Exception:
        pass
    return JSONResponse({"ok": True, **out})


@app.get("/api/system/status")
def system_status() -> JSONResponse:
    """One glance: what is running, what is waiting, and why
    (old-app inspiration) — assembled from the subsystems' own state,
    never from configuration."""
    eng = engine
    rows: list[dict] = []

    def row(label, ok, detail, state="ok"):
        rows.append({"label": label, "ok": bool(ok), "state": state,
                     "detail": str(detail)[:220]})

    # TAPE
    try:
        st = eng.feed.state("BTC")
        age = int(time.time() - (st.times[-1] if st.times else 0)) \
            if st.times else -1
        bars = len(st.closes)
        row("tape", bars >= 30,
            f"{eng.feed.mode} · {bars} BTC bars · last tick {age}s ago"
            + (" · live" if bars >= 30 else " · warming up"),
            "ok" if bars >= 30 else "warn")
    except Exception as exc:
        row("tape", False, f"error: {exc}", "err")
    # FORMULAS
    try:
        tr = eng.last_traces.get(("BTC", "tf1")) or {}
        vals = [t for k, t in tr.items()
                if isinstance(t, dict) and "value" in t]
        nz = sum(1 for t in vals if abs(t.get("value") or 0) > 1e-9)
        us = tr.get("_pass_us")
        row("formulas", nz > 0,
            f"{nz}/{len(vals)} non-zero this pass"
            + (f" · {us} µs" if us else ""),
            "ok" if nz else "warn")
    except Exception as exc:
        row("formulas", False, f"error: {exc}", "err")
    # EMOTIONS / PHYSICS (display systems)
    try:
        disp = eng.display_systems.get(("BTC", "tf1")) or {}
        em = disp.get("emotion") or {}
        row("emotions", eng.emotions.ready,
            f"CERC {em.get('cerc', '—')} · aggregate "
            f"{em.get('aggregate', '—')} · "
            f"{len(em.get('emotions') or {})} channels",
            "ok" if eng.emotions.ready else "warn")
        ph = disp.get("physics") or {}
        mechs = ph.get("mechanisms") or []
        act = sum(1 for m in mechs if m.get("active"))
        row("physics", eng.physics.ready,
            f"{act}/{len(mechs)} mechanisms active · UFP "
            f"{ph.get('ufp', '—')}",
            "ok" if act else "warn")
    except Exception as exc:
        row("emotions/physics", False, f"error: {exc}", "err")
    # NEWS
    try:
        ns = eng.news.status()
        row("news", bool(ns.get("items")),
            f"{ns.get('items', 0)} items · source {ns.get('source')} · "
            f"dropped {ns.get('relevance_dropped', 0)}",
            "ok" if ns.get("items") else "warn")
    except Exception as exc:
        row("news", False, f"error: {exc}", "err")
    # BRAIN
    try:
        chip = eng.brains.status().get("chip") or {}
        row("brain", chip.get("kind") == "ok",
            chip.get("text", "—"),
            "ok" if chip.get("kind") == "ok" else
            ("warn" if chip.get("kind") == "build" else "err"))
    except Exception as exc:
        row("brain", False, f"error: {exc}", "err")
    # AGENTS (optional — keyless is fine)
    try:
        ag = eng.agents.status()
        slots = ag.get("slots") or []
        if isinstance(slots, dict):
            slot_list = list(slots.values())
        else:
            slot_list = list(slots)          # list-of-slots shape
        nconn = sum(1 for v in slot_list
                    if isinstance(v, dict) and v.get("status") == "connected")
        # local-agent download state belongs on this row: the user asked
        # for it to arrive by itself, so "is it downloading, and why not"
        # must be answerable without opening another tab
        try:
            _au = eng.agents.installer.auto_status()
            _inst = ag.get("installer") or {}
            _backend = ag.get("backend") or _inst.get("backend") or ""
            # report the RUNTIME, not just the autoloader: "local agent
            # running" while nothing can answer was the misleading part
            _state = _inst.get("state") or _au.get("state")
            _loc = (f" · local agent {_state}"
                    + (f" via {_backend}" if _backend else "")
                    + (f", phase {_inst.get('phase')}"
                       if _inst.get("phase") else "")
                    + (f" (retry in {_au['next_try_in_s']}s)"
                       if _au.get("next_try_in_s") else "")
                    + (f": {(_inst.get('error') or _au.get('last_error'))[:90]}"
                       if (_inst.get("error") or _au.get("last_error"))
                       and _state == "failed" else ""))
        except Exception:
            _loc = ""
        row("agents", True,
            ("keyless mode — brain + formulas carry every vote"
             if not nconn else f"{nconn} slot(s) connected · "
             f"deadline {ag.get('deadline_s', '—')}s") + _loc,
            "ok")
    except Exception as exc:
        row("agents", False, f"error: {exc}", "err")
    # HINDSIGHT / AUTO
    try:
        changes = eng.hindsight_changes[-1] if eng.hindsight_changes             else None
        row("hindsight", bool(changes),
            (f"last change: {changes.get('formula')} "
             f"{changes.get('from')}→{changes.get('to')}"
             if changes else
             "waiting for enough graded windows (n≥8)"),
            "ok" if changes else "idle")
        row("auto-predict",
            eng.agents.settings.get("auto_predict", False),
            "on — analysis re-runs itself every ≥45 s (user-enabled)"
            if eng.agents.settings.get("auto_predict", False)
            else "off — manual ▶ ANALYSE only (default; user-controlled)",
            "ok" if eng.agents.settings.get("auto_predict", False)
            else "idle")
    except Exception as exc:
        row("hindsight", False, f"error: {exc}", "err")
    # HISTORY / SETTLEMENT — is the ledger actually grading windows?
    try:
        led = eng.ledger
        ago = (round(time.time() - led.last_settle_ts, 1)
               if led.last_settle_ts else None)
        tot = led.totals
        ls = led.last_settle or {}
        last_txt = (f"last {ago}s ago ({ls.get('tf', '')} "
                    f"{ls.get('result', '')})" if ago is not None
                    else "none yet — the first window boundary fills it "
                         "(tf1 ≤22 m, windows are epoch-aligned)")
        row("history", led.settle_calls > 0,
            f"{tot.get('win', 0)}W/{tot.get('loss', 0)}L/"
            f"{tot.get('skip', 0)}skip · {led.settle_calls} settles · "
            f"{last_txt}",
            "ok" if led.last_settle_ts else "idle")
    except Exception as exc:
        row("history", False, f"error: {exc}", "err")
    # CODE — answers 'why is the app not updating': stale checkout?
    try:
        build = eng.ops_status().get("build_id", "?")
        upd = ""
        try:
            import subprocess as _sp
            cnt = _sp.check_output(
                ["git", "rev-list", "--count", "HEAD..@{u}"],
                stderr=_sp.DEVNULL, timeout=2).decode().strip()
            if cnt.isdigit() and int(cnt) > 0:
                upd = (f" · UPDATE PENDING — {cnt} commit(s) behind "
                       f"(pull, then python3 start.py --detach)")
        except Exception:
            pass
        row("code", not upd, f"build {build}{upd or ' · up to date'}",
            "warn" if upd else "ok")
    except Exception as exc:
        row("code", False, f"error: {exc}", "err")
    # OPS / WATCHDOG
    try:
        ops = eng.ops_status()
        wd = ops.get("watchdog") or {}
        alert = bool(wd.get("alert"))
        sw = wd.get("seconds_since_window")
        vw = ops.get("vault_last_write_ago_s")
        row("watchdog", not alert,
            f"last window {f'{sw}s ago' if sw is not None else '—'} · "
            f"uptime {int(ops.get('uptime_s') or 0)}s · "
            f"build {ops.get('build_id')} · "
            f"vault {f'{vw}s' if vw is not None else '—'}",
            "warn" if alert else "ok")
    except Exception as exc:
        row("watchdog", False, f"error: {exc}", "err")
    return JSONResponse({"ok": True, "rows": rows,
                         "ts": time.time()})


@app.get("/api/brain/wiring")
async def brain_wiring(symbol: str = "BTC", tf: str = "tf1") -> JSONResponse:
    """§3.3 formula → channel wiring + polarity/odd-symmetry probes."""
    return JSONResponse({"ok": True, **engine.brain_wiring(symbol, tf)})


@app.post("/api/brain/reconnect")
async def brain_reconnect() -> JSONResponse:
    """A-6: re-run verification on demand (never a stale ❌)."""
    return JSONResponse({"ok": True, **engine.brains.reconnect()})


@app.get("/api/brain/explain")
async def brain_explain(symbol: str = "BTC", tf: str = "tf1") -> JSONResponse:
    """§3.5 show the wiring: activations, hormones, dominant state."""
    return JSONResponse({"ok": True,
                         "activations": engine.brain_activations(symbol, tf),
                         "status": engine.brains.status()})


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


@app.get("/api/news/status")
async def news_status() -> JSONResponse:
    """docs11 §3.6/§7 — coverage string, per-source backoff lines,
    per-horizon indices and the emergency audit trail."""
    await asyncio.to_thread(engine.news.refresh)   # may block 8 s (issue 1)
    return JSONResponse({"ok": True, **engine.news.stats(),
                         **engine.news.status()})


@app.post("/api/news/poll")
async def news_poll() -> JSONResponse:
    """docs11 §7 — manual poll NOW (rate-limit aware) + evaluate."""
    await asyncio.to_thread(engine.news.refresh)   # one honest ingest pass
    for sym in SYMBOLS:
        await asyncio.to_thread(engine.news.evaluate, sym)
    return JSONResponse({"ok": True, "items": len(engine.news.items),
                         "coverage": engine.news.coverage_string(),
                         "stats": engine.news.stats()})


@app.get("/api/news/feed")
async def news_feed_items(asset: str = "BTC", limit: int = 30) -> JSONResponse:
    """docs11 §7 — newest-first feed with per-item coverage label."""
    ns = engine.news
    out = [{
        "ts": it.get("ts"), "headline": it.get("headline"),
        "source": it.get("source"), "tier": it.get("tier"),
        "theme": it.get("theme"),
        "relevance": (it.get("relevance_by_asset") or {}).get(asset, 0.0),
        "sentiment": it.get("sentiment"),
        "synthetic": bool(it.get("synthetic")),
        "coverage": "synthetic" if it.get("synthetic") else "live",
    } for it in list(reversed(ns.items))[:max(1, min(200, limit))]]
    return JSONResponse({"ok": True, "asset": asset, "items": out,
                         "filtered": ns.filtered,
                         "filtered_examples": ns.filtered_examples})


@app.get("/api/news/impact")
async def news_impact(asset: str = "BTC", horizon: str = "tf1") -> JSONResponse:
    """docs11 §4 — per-horizon impact/NIV/SMD + top-4 drivers + theme
    inputs. Defined before /api/news/{symbol} so 'impact' never parses
    as a symbol."""
    return JSONResponse(engine.news.impact(asset, horizon))


@app.post("/api/news/emergency")
async def news_emergency(payload: dict) -> JSONResponse:
    """docs11 §4.9 — drill (no open lock) or live emergency re-lock.
    Acceptance 6: re-locks at today's price, original entry preserved in
    the reason, 30 s halt; >1/hour rejected as a bug (N-8)."""
    symbol = str(payload.get("asset") or "BTC").upper()
    if symbol not in SYMBOLS:
        return JSONResponse({"ok": False, "error": "unknown asset"},
                            status_code=400)
    tf = str(payload.get("tf") or "tf1")
    reason = str(payload.get("reason") or "manual emergency drill")[:300]
    out = engine.emergency_relock(symbol, tf, reason)
    if out.get("ok"):
        await manager.broadcast({"type": "news_emergency_ack", **out})
    return JSONResponse(out)


@app.get("/api/news/{symbol}")
async def news_feed(symbol: str) -> JSONResponse:
    """Section-7 news feed: categories, aggregate, per-TF impacts, items."""
    if symbol not in SYMBOLS:
        return JSONResponse({"ok": False, "error": "unknown symbol"},
                            status_code=400)
    await asyncio.to_thread(engine.news.refresh)   # top up the sim stream
    return JSONResponse({"ok": True, **engine.news.aggregate(symbol)})


def _candle_history_revision(bars: list[dict], *, mode: str = "",
                             source: str = "", switches: int = 0) -> str:
    """Revision for completed OHLC history, independent of list length/end time.

    A feed handover can replace every candle while keeping the same bucket
    timestamps and count. The browser must still replace its rendered series.
    The current partial candle is intentionally excluded; live ticks update it.
    """
    completed = bars[:-1]
    normalized = [[int(b["t"]), round(float(b["o"]), 8),
                   round(float(b["h"]), 8), round(float(b["l"]), 8),
                   round(float(b["c"]), 8), round(float(b.get("v") or 0), 8)]
                  for b in completed]
    payload = json.dumps({"mode": mode, "source": source,
                          "switches": int(switches), "bars": normalized},
                         sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


@app.get("/api/genesis/walkforward")
async def genesis_walkforward(symbol: str = "BTC", horizon: int = 22) -> JSONResponse:
    """guide §3.4 — replay the 1-minute tape through the pool and score it
    strictly out-of-sample (60 % IS / 40 % OOS)."""
    if symbol not in SYMBOLS:
        return JSONResponse({"ok": False, "error": "unknown symbol"},
                            status_code=400)
    try:
        bars = engine._bars_1m(symbol)
        rep = engine.genesis.evaluate_pool_walkforward(
            bars, horizon=max(2, min(240, int(horizon))))
    except Exception as exc:
        return JSONResponse({"ok": False, "error": repr(exc)}, status_code=500)
    return JSONResponse({"ok": True, "symbol": symbol, **rep})


@app.get("/api/tf-charts")
async def tf_charts(symbol: str = "BTC") -> JSONResponse:
    """guide §1 — all four mini-chart series in one payload.

    Each TF carries its own step, lookback, revision and epoch, so a card
    binds to the series the server publishes instead of re-aggregating a
    tf1 tape with its own private window."""
    if symbol not in SYMBOLS:
        return JSONResponse({"ok": False, "error": "unknown symbol"},
                            status_code=400)
    try:
        charts = engine.tf_charts(symbol)
    except Exception as exc:                     # never break the chart
        return JSONResponse({"ok": False, "error": repr(exc)},
                            status_code=500)
    return JSONResponse({
        "ok": True, "symbol": symbol, "charts": charts,
        "epochs": (getattr(engine, "epoch_start_tf", {}) or {}).get(symbol, {}),
        "server_time": int(time.time()),
    })


@app.get("/api/candles")
async def candles(symbol: str = "BTC", n: int = 180, tf: str = "tf1") -> JSONResponse:
    """OHLC bars (incl. the live partial minute), capped at 1000 source
    minutes for chart context. tf=tf1 → raw 1-minute bars; wider TFs
    aggregate using the configured horizon buckets."""
    if symbol not in SYMBOLS:
        return JSONResponse({"ok": False, "error": "unknown symbol"},
                            status_code=400)
    # §6 7-10-day warmup: serve the full backfill window (BACKFILL_BARS
    # = 7 days of 1m tape) — the old 1000-bar cap starved the main chart
    # to 16 h and left boot in WARMING_UP / n=0 states
    n = max(10, min(BACKFILL_BARS, n))
    st = engine.feed.state(symbol)
    active_family = _family(getattr(engine.feed, "mode", "unknown"))
    indices = st.indices_for_family(active_family)
    foreign_bars = len(st.times) - len(indices)
    count = len(indices)
    W = int(TIMEFRAMES.get(tf, {}).get("minutes", 1) or 1)
    if tf == "tf1":
        W = 1        # tf1 keeps raw 1-minute candles (the 22m window is the
                     # prediction horizon; traders watch 1m bars inside it)
    if W <= 1:
        start = max(0, count - n)
        bars = [
            {"t": st.times[i], "o": st.opens[i], "h": st.highs[i],
             "l": st.lows[i], "c": st.closes[i], "v": st.volumes[i]}
            for i in indices[start:]
        ]
        if st.live_price > 0 and active_family in ("sim", "binance"):
            # the live partial bar belongs to the CURRENT bucket: merge
            # into it when the tape already has it, else open it at the
            # previous close. (Was: appended at last_bar+60 — a future
            # candle whose close the chart then chased a minute late.)
            bkt = int(time.time() // 60) * 60
            bars = [b for b in bars if b["t"] <= bkt]      # never future
            if bars and bars[-1]["t"] == bkt:
                bars[-1]["h"] = max(bars[-1]["h"], st.live_high or 0, st.live_price)
                bars[-1]["l"] = min(bars[-1]["l"], st.live_low or bars[-1]["l"], st.live_price)
                bars[-1]["c"] = st.live_price
                bars[-1]["live"] = True
            else:
                prev_c = bars[-1]["c"] if bars else st.live_price
                bars.append({"t": bkt, "o": prev_c,
                             "h": max(st.live_high or 0, st.live_price, prev_c),
                             "l": min(st.live_low or st.live_price, st.live_price, prev_c),
                             "c": st.live_price, "v": st.live_volume, "live": True})
    else:
        # epoch-aligned buckets of W minutes
        bucket: dict[int, dict] = {}
        for i in indices:
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
        if st.live_price > 0 and active_family in ("sim", "binance"):
            bkt = int(time.time() // (W * 60)) * (W * 60)
            if bars and bkt == bars[-1]["t"]:
                bars[-1]["h"] = max(bars[-1]["h"], st.live_price, st.live_high or 0)
                bars[-1]["l"] = min(bars[-1]["l"], st.live_price, st.live_low or 1e18)
                bars[-1]["c"] = st.live_price
                bars[-1]["live"] = True
            else:
                bars.append({"t": bkt, "o": st.live_open or st.live_price,
                             "h": st.live_high or st.live_price,
                             "l": st.live_low or st.live_price,
                             "c": st.live_price, "v": st.live_volume, "live": True})
    fstats = engine.feed.stats()
    revision = _candle_history_revision(
        bars, mode=str(fstats.get("mode") or ""),
        source=str(fstats.get("source") or ""),
        switches=int(fstats.get("switches") or 0))
    from .chartlink import candle_tape_integrity
    integrity = candle_tape_integrity(
        bars, live_price=st.live_price, family=active_family,
        foreign_bars=foreign_bars, now=time.time(), bucket_minutes=W)
    return JSONResponse(
        {"ok": True, "symbol": symbol, "tf": tf,
         "bucket_minutes": W, "revision": revision,
         "bar_family": active_family, "excluded_foreign_bars": foreign_bars,
         "integrity": integrity,
         "feed": {"mode": fstats.get("mode"), "source": fstats.get("source"),
                  "switches": fstats.get("switches")},
         "bars": bars},
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@app.get("/api/backtest")
async def backtest_report() -> dict:
    """Real-backtest results (written by scripts/run_backtest_once.sh
    where Binance is reachable). Honest empty state until a real run
    exists — never synthetic numbers dressed as results."""
    import csv as _csv
    import os as _os
    path = "backtest/out/mode_a_trades.csv"
    if not _os.path.exists(path):
        return {"ok": False, "reason": "no real backtest yet — it runs "
                "automatically (max daily) where Binance is reachable"}
    from backtest.analytics import by_asset, metrics
    from backtest.runner import Trade
    raw = []
    with open(path) as f:
        for r in _csv.DictReader(f):
            raw.append(Trade(r["asset"], int(r["ts"]), r["direction"],
                             float(r["entry"]), float(r["sl"]),
                             float(r["tp"]), int(r["exit_ts"]),
                             r["exit_reason"], float(r["r_gross"]),
                             float(r["r_net"]), float(r["pnl"]),
                             r.get("tier", ""), r.get("source", "raw")))
    return {"ok": True, "source": "Binance 15m via CCXT",
            "per_asset": by_asset(raw), "combined": metrics(raw)}


@app.get("/api/chart-link")
async def chart_link(symbol: str = "") -> JSONResponse:
    """NEW: measured chart<->price link per asset (docs11 Part D).
    The browser's Chart Link panel reads this; delta_bps is the
    server-side truth about whether candles end at the live price."""
    from .chartlink import all_reports, link_report
    if symbol:
        if symbol not in SYMBOLS:
            return JSONResponse({"ok": False, "error": "unknown symbol"},
                                status_code=400)
        return JSONResponse({"ok": True, **link_report(feed, symbol)})
    wd = getattr(app.state, "link_watchdog", None)
    out = all_reports(feed, SYMBOL_ORDER)
    out["watchdog"] = (wd.status() if wd else {"checks": 0, "incidents": []})
    return JSONResponse(out)


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
    # never build a snapshot on the loop for one sub-block: the last
    # broadcast is authoritative, and an empty dict is honest on a cold boot
    snap = engine.last_snapshot or {}
    return JSONResponse({"ok": True, "learning": snap.get("learning", {}),
                         "shares": engine.learning.subsystem_shares,
                         "state_file": "data/auto_learning_state.json"})


@app.post("/api/learning/macro")
async def learning_macro() -> JSONResponse:
    """Manual/cron trigger for the macro learning cycle (15 GRCs)."""
    try:
        # off the event loop — the cycle walks every settled row (issue 1)
        await asyncio.to_thread(engine.learning.macro_cycle, engine)
        await asyncio.to_thread(engine.learning.save, engine)
    except Exception as e:
        log.exception("macro trigger failed")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return JSONResponse({"ok": True, "macro_count": engine.learning.macro_count,
                         "shares": engine.learning.subsystem_shares})


@app.post("/api/learning/deep")
async def learning_deep() -> JSONResponse:
    """Manual trigger for the deep learning cycle (247 minutes)."""
    try:
        await asyncio.to_thread(engine.learning.deep_cycle, engine)
        await asyncio.to_thread(engine.learning.save, engine)
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


@app.get("/api/history")
async def history_console() -> JSONResponse:
    """guide §7.5 — the HX HISTORY console in one payload: ledger totals,
    the 41-subject hindsight grade, the ten-module advanced scoreboard and
    the 448-row meta grid."""
    from .advanced import scoreboard
    payload: dict = {"ok": True}
    try:
        payload["ledger"] = engine.ledger.api_payload(SYMBOL_ORDER, TF_ORDER)
    except Exception as exc:
        payload["ledger"] = {"error": repr(exc)}
    try:
        # engine.hindsight is a METHOD (the §8.3 re-grade builder).  The
        # old `getattr(engine, "hindsight", None) or …` returned the bound
        # METHOD object itself — a truthy callable — which is not JSON
        # serializable, so GET /api/history 500'd for every caller.
        hz = getattr(engine, "hindsight", None)
        if callable(hz):
            payload["hindsight"] = hz()
        else:
            payload["hindsight"] = hz or engine.ledger.hindsight_regrade()
    except Exception as exc:
        payload["hindsight"] = {"error": repr(exc)}
    try:
        payload["advanced"] = scoreboard(engine, SYMBOL_ORDER, TF_ORDER)
    except Exception as exc:
        payload["advanced"] = {"error": repr(exc)}
    try:
        from .meta import meta_payload
        payload["meta"] = meta_payload(engine)
    except Exception as exc:
        payload["meta"] = {"error": repr(exc)}
    payload["rows"] = sum(len(v) for v in engine.ledger.rows.values())
    # belt & braces: a stray non-JSON leaf (function, bytes, set…) in ANY
    # section becomes a named placeholder instead of a 500 for the whole
    # console — panels keep rendering, the placeholder is visible.
    def _safe(o):
        if isinstance(o, dict):
            return {str(k): _safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_safe(v) for v in o]
        try:
            json.dumps(o)
            return o
        except (TypeError, ValueError):
            return f"<{type(o).__name__} not serializable>"
    return JSONResponse(_safe(payload))


@app.get("/api/history/advanced")
async def history_advanced(symbol: str = "BTC",
                           tf: str = "tf1") -> JSONResponse:
    """Per-module walk-forward accuracy for the advanced layer.

    Accepts either the UI asset key (BTC) or the exchange symbol
    (BTCUSDT); the old defaults (BTCUSDT/tf3) never matched an engine key,
    so the scoreboard stayed on "Loading…" forever.
    """
    from .config import SYMBOLS, TIMEFRAMES
    sym = str(symbol or "").upper()
    if sym not in SYMBOLS:
        sym = str(symbol or "").upper().replace("USDT", "")
    if sym not in SYMBOLS:
        sym = "BTC"
    t = str(tf or "").lower()
    if t not in TIMEFRAMES:
        t = "tf1"
    return JSONResponse({"ok": True, **engine.advanced_history(sym, t)})


@app.get("/api/history/day")
async def history_day(date: str) -> JSONResponse:
    rows = engine.historydb.day(date)
    return JSONResponse({"ok": True, "date": date, "records": rows})


@app.get("/api/history/calibration")
async def history_calibration() -> JSONResponse:
    """Confidence calibration: for every settled window, the LOCKED
    confidence bucket vs the realised strict-sign hit. A well-calibrated
    engine's hit-rate rises with the stated confidence — the single
    best accuracy audit. Honest 'needs n' below MIN_N."""
    MIN_N = 40
    pairs: list[tuple[float, float]] = []
    for symbol in SYMBOLS:
        for tf in TF_ORDER:
            ps = list(engine.hist.ring(
                engine.hist.tf_key(symbol, tf, "calib_p"), 400))
            hs = list(engine.hist.ring(
                engine.hist.tf_key(symbol, tf, "calib_h"), 400))
            n = min(len(ps), len(hs))
            pairs.extend((float(ps[i]), float(hs[i])) for i in range(n))
    buckets = {}
    for b, h in pairs:
        e = buckets.setdefault(int(round(b * 20)),
                               {"p": round(b, 2), "n": 0, "hits": 0})
        e["n"] += 1
        e["hits"] += h
    out = []
    for k in sorted(buckets):
        e = buckets[k]
        out.append({"conf": e["p"], "n": e["n"],
                    "hit_rate": round(e["hits"] / e["n"], 4)
                    if e["n"] else None})
    n = len(pairs)
    brier = (round(sum((b - h) ** 2 for b, h in pairs) / n, 4)
             if n else None)
    hit_all = (round(sum(h for _b, h in pairs) / n, 4) if n else None)
    return JSONResponse({
        "ok": True, "n": n, "enough": n >= MIN_N, "min_n": MIN_N,
        "bins": out, "brier": brier, "hit_rate": hit_all,
        "note": ("hit-rate should RISE with stated confidence; "
                 "brier → 0 is perfect, 0.25 is a coin flip")})


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
    from .config import (WEIGHT_IDS as _WIDS, BRAIN_COMPONENT_ID,
                         EMOTION_COMPONENT_ID, PHYSICS_COMPONENT_ID,
                         NEWS_COMPONENT_ID, HEDGE_COMPONENT_ID)
    # ids97-101 are SUBSYSTEM components (not formulas) — never label
    # them F97…; show what they actually are
    from .config import TRACER_COMPONENT_ID
    SUBSYS_NAMES = {
        TRACER_COMPONENT_ID: "hedge-flow tracer (subsystem)",
        BRAIN_COMPONENT_ID: "brain vote (subsystem)",
        EMOTION_COMPONENT_ID: "emotion CERC (subsystem)",
        PHYSICS_COMPONENT_ID: "physics UFP (subsystem)",
        NEWS_COMPONENT_ID: "news impact (subsystem)",
        HEDGE_COMPONENT_ID: "hedge HF (subsystem)",
    }
    rows = []
    for fid in _WIDS:
        acc = engine.learning.formula_accuracy(fid)
        n = len(engine.learning.acc500.get(fid) or [])
        # always list every formula — the UI renders accuracy=None as an
        # honest "— n=<n> (needs ≥8)" instead of a scary red 0.0000
        rows.append({"formula": fid,
                     "name": FORMULA_NAMES.get(fid)
                             or SUBSYS_NAMES.get(fid, f"F{fid}"),
                     "subsystem": fid in SUBSYS_NAMES,
                     "accuracy": round(acc, 4) if acc is not None else None,
                     "n": n})
    # undecided (accuracy=None, n<8) sink to the bottom — never crash sort
    rows.sort(key=lambda r: -(r["accuracy"]
                              if r["accuracy"] is not None else -1.0))
    return JSONResponse({"ok": True, "leaderboard": rows})


def _public_settings() -> dict:
    """Settings safe for the browser: secrets are never sent back.

    The old endpoint returned the live Gemini key in clear text on every
    dashboard poll path. A key belongs in the server's secret store, not in
    a JSON response (and not in a screenshot of the network tab).
    """
    from .agents import KEY_FIELDS
    out = {}
    for k, v in engine.agents.settings.items():
        if k in KEY_FIELDS:
            out[k] = ("•••• set" if v else "")
        else:
            out[k] = v
    return out


@app.get("/api/settings")
async def settings_get() -> JSONResponse:
    return JSONResponse({"ok": True, "settings": _public_settings()})


@app.put("/api/settings")
async def settings_put(body: dict) -> JSONResponse:
    from .agents import KEY_ALIASES, KEY_FIELDS
    rejected, applied = [], []
    for k, v in body.items():
        if k == "system_mode":
            # this IS a real switch — it used to be dropped while the header
            # button claimed it had changed the model
            engine.set_system_mode(str(v))
            applied.append(k)
        elif k == "feed_mode":
            rejected.append(k)      # applied through /api/feeds/connect
        elif k in engine.agents.settings or k in KEY_ALIASES:
            # blank key field = keep: never reported as applied
            if (KEY_ALIASES.get(k, k) in KEY_FIELDS
                    and isinstance(v, str) and not v.strip()):
                continue
            engine.agents.configure(**{k: v})
            applied.append(k)
        else:
            rejected.append(k)
    return JSONResponse({"ok": True, "settings": _public_settings(),
                         "applied": sorted(set(applied)),
                         "system_mode": engine.agents.settings.get(
                             "system_mode", "old"),
                         "ignored": sorted(set(rejected)),
                         "note": ("feed_mode is applied with "
                                  "POST /api/feeds/connect"
                                  if rejected else "all keys applied")})


@app.post("/api/system/mode")
async def system_mode(body: dict) -> JSONResponse:
    """Dedicated switch for OLD ladder ⇄ NEW horizon forecast."""
    return JSONResponse(engine.set_system_mode(str(body.get("mode", "old"))))


@app.get("/api/system/mode")
async def system_mode_get() -> JSONResponse:
    return JSONResponse({"ok": True,
                         "system_mode": engine.agents.settings.get(
                             "system_mode", "old")})


@app.get("/api/predictions")
async def predictions(symbol: str | None = None, tf: str | None = None,
                      limit: int = 50) -> JSONResponse:
    """Sealed prediction locks — the immutable record ANALYSE writes."""
    if symbol is not None and symbol not in SYMBOLS:
        return JSONResponse({"ok": False, "error": "unknown symbol"},
                            status_code=400)
    if tf is not None and tf not in TF_ORDER:
        return JSONResponse({"ok": False, "error": "unknown timeframe"},
                            status_code=400)
    return JSONResponse(engine.predlocks.payload(symbol, tf, limit))


@app.get("/api/predictions/stats")
async def prediction_stats(symbol: str | None = None,
                           tf: str | None = None) -> JSONResponse:
    return JSONResponse({"ok": True,
                         **engine.predlocks.stats(symbol, tf)})


@app.get("/api/predictions/audit")
async def prediction_audit() -> JSONResponse:
    """The sealed journal's SELF-EXAMINATION.

    The user's third report was that the app ships its own contradictions
    without noticing. This endpoint asks the record set whether it is
    coherent — wrong-side ladders, results recorded at a barrier the
    market had already passed, settlements younger than one bar — and
    names every break. ``ok: true`` with an empty ``issues`` list is the
    only state that means "nothing to report"; a populated list is the
    in-app error the page renders instead of hiding.
    """
    try:
        return JSONResponse({"ok": True, **engine.predlocks.audit()})
    except Exception as exc:                      # never take the page down
        return JSONResponse({"ok": False,
                             "error": f"{type(exc).__name__}: {exc}",
                             "issues": [], "checked": 0})


@app.get("/api/predictions/{lock_id}")
async def prediction_one(lock_id: str) -> JSONResponse:
    for row in engine.predlocks.rows:
        if row.get("id") == lock_id:
            return JSONResponse({"ok": True, "lock": row})
    return JSONResponse({"ok": False, "error": "unknown lock"}, status_code=404)


@app.get("/api/demo-sync/status")
async def demo_sync_status() -> JSONResponse:
    """Read-only account-link status; no-key mode is explicit and offline."""
    from . import demo_sync as _ds
    cfg = _ds.config()
    if not cfg.get("has_keys"):
        return JSONResponse(_ds.SERVICE.status())
    try:
        snapshot = await asyncio.to_thread(_ds.SERVICE.status)
        internal = engine.predlocks.latest(limit=100)
        snapshot["reconciliation"] = _ds.reconcile(internal, snapshot)
        snapshot["internal_locks"] = internal
        snapshot["read_only"] = True
        snapshot["badge"] = ("READ-ONLY BINANCE "
                             + ("TESTNET" if cfg["testnet"] else "MAINNET")
                             + f" {cfg['market'].upper()} LINKED")
        snapshot["orders_today"] = engine.trader.status().get("orders_today", 0)
        return JSONResponse(snapshot)
    except Exception as exc:
        # Avoid returning request URLs, signed query strings, or credentials.
        return JSONResponse({"ok": False, "linked": True,
                             "read_only": True, "market": cfg["market"],
                             "testnet": cfg["testnet"],
                             "error": f"account read failed: {type(exc).__name__}",
                             "reconciliation": []}, status_code=502)


@app.put("/api/demo-sync/config")
async def demo_sync_config(body: dict) -> JSONResponse:
    """Save separate, read-only sync credentials; never arms execution."""
    from .demo_sync import save_config, SERVICE
    try:
        result = save_config(
            api_key=str(body.get("api_key") or ""),
            api_secret=str(body.get("api_secret") or ""),
            testnet=bool(body.get("testnet", True)),
            market=str(body.get("market") or "futures"),
            clear_keys=bool(body.get("clear_keys")))
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    SERVICE.reset()
    return JSONResponse({"ok": True, "config": result,
                         "note": "GET-only account sync; order placement is not implemented"})


@app.get("/api/trading/status")
async def trading_status() -> JSONResponse:
    return JSONResponse({"ok": True, **engine.trader.status()})


@app.put("/api/trading/config")
async def trading_config(body: dict) -> JSONResponse:
    from .trading import load_cfg, save_cfg, public_cfg
    cfg = load_cfg()
    for k in ("enabled", "testnet", "max_order_usdt", "max_orders_day",
              "sizing_mode", "equity_usdt", "risk_pct"):
        if k in body:
            cfg[k] = body[k]
    # blank key fields mean "keep the stored key" — never wipe by accident
    if body.get("api_key"):
        cfg["api_key"] = str(body["api_key"]).strip()
    if body.get("api_secret"):
        cfg["api_secret"] = str(body["api_secret"]).strip()
    if body.get("clear_keys"):
        cfg["api_key"] = cfg["api_secret"] = ""
        cfg["enabled"] = False
        cfg["armed_until"] = 0.0
    save_cfg(cfg)
    engine.trader.cfg = cfg
    return JSONResponse({"ok": True, "config": public_cfg(cfg)})


@app.post("/api/trading/arm")
async def trading_arm(body: dict) -> JSONResponse:
    from .trading import load_cfg, save_cfg, public_cfg
    import time as _t
    cfg = load_cfg()
    minutes = max(0, min(240, int(body.get("minutes") or 0)))
    cfg["armed_until"] = (_t.time() + minutes * 60) if minutes else 0.0
    save_cfg(cfg)
    engine.trader.cfg = cfg
    return JSONResponse({"ok": True, "config": public_cfg(cfg)})


@app.get("/api/agents/status")
async def agents_status() -> JSONResponse:
    # agents.status() probes Ollama (sync httpx, 2 s cap) and auditor.status()
    # pings the daemon — both off the event loop (issue 1)
    ag = await asyncio.to_thread(engine.agents.status)
    try:
        ag["auditor"] = await asyncio.to_thread(engine.auditor.status)
    except Exception:                              # the Agents tab span
        ag["auditor"] = {"available": False,       # on "checking daemon…"
                          "hint": "auditor status unavailable"}
    return JSONResponse({"ok": True, "agents": ag,
                         "signal": engine.last_agent_signal or {}})


@app.post("/api/feeds/connect")
async def feeds_connect() -> JSONResponse:
    """The CONNECT BINANCE button: try the real feed NOW and report the truth.

    It used to be a settings link, and `PUT /api/settings` dropped `feed_mode`
    entirely, so the header could not actually change the tape. This performs
    the switch and returns the measured outcome — success (with the source
    that delivered) or the real error, never a silent fallback.
    """
    out = await engine.feed.connect_real()
    out["feed"] = engine.feed.stats()
    return JSONResponse(out)


@app.get("/api/feeds/diagnose")
async def feeds_diagnose() -> JSONResponse:
    """§1.1: per-source honesty — delivery ages, rejected writes, tape mix."""
    st = engine.feed.stats()
    return JSONResponse({
        "ok": True,
        "feed": st,
        "sources": {
            "binance-ws": {"family": "binance",
                           "last_real_delivery_age_s": st.get("real_age_s")},
            "sim": {"family": "sim"},
        },
        "rules": {
            "single_writer": "only the active family may write; foreign "
                             "writes are rejected and counted",
            "grace_s": 120.0,
            "switch_policy": "adopt flushes synthetic bars; demote keeps "
                             "real history, chip goes SIMULATED TAPE",
        },
    })


@app.post("/api/analyse/{symbol}")
async def analyse_start(symbol: str, tf: str = "tf1",
                        request_id: str | None = None) -> JSONResponse:
    """Staged ANALYSE: formulas → emotions → physics → news → hedge →
    AI review (Gemini + local qwen2.5-coder) → final box (signal, four
    TPs, SL). Read-only — locks never change here.

    ``request_id`` (optional) is stamped onto every analyse_* event so
    the UI can discard superseded work atomically (latest-wins)."""
    out = engine.start_analyse(symbol, tf, request_id=request_id)
    status = 200 if out.get("ok") else 409
    return JSONResponse(out, status_code=status)


@app.get("/api/analyse/status")
async def analyse_status() -> JSONResponse:
    job = engine.analyse_job or {"status": "idle"}
    return JSONResponse({"ok": True, "job": job})


@app.get("/api/cells")
async def hold_cells() -> JSONResponse:
    """The certified hold policy's 16 live cells (4 assets × 4 TFs):
    tier badge (HOLD-OUT VERIFIED / IN-SAMPLE SHADOW / UNGRADEABLE ·
    WARMUP), measured win rate + Wilson confidence, the per-cell
    ANALYSE countdown (one click locks THAT cell for its tf minutes —
    the other cells stay free), the era position with real friction,
    and the last sealed lock."""
    from . import hold_policy as _hp
    cells = engine.hold_cells_block()
    return JSONResponse({
        "ok": True,
        "policy": "certified hold policy (backtest/results/hold_wr.json)",
        "cells": cells,
        "count": len(cells),
        "tiers": _hp.tier_counts(),
        "manual_default": True,       # only the user can enable auto
        "auto_predict": bool(engine.agents.settings.get(
            "auto_predict", False)),
        "rule": ("one ANALYSE click = one tf-minute lock on that cell "
                 "only; 4 predictions per asset, 16 total"),
    })


@app.get("/api/analyses")
async def analyses_journal(limit: int = 100) -> JSONResponse:
    """Issue 3 — durable memory of every analysis run: what was decided,
    what it learned, what needs to change (ai_correction/hindsight), and
    the lock id that ties it to settlement."""
    import json as _json
    path = __import__("os").environ.get(
        "ANALYSE_JOURNAL_PATH", "data/analyses.jsonl")
    rows: list = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(_json.loads(line))
                    except ValueError:
                        continue
    except OSError:
        pass
    n = max(1, min(500, int(limit)))
    return JSONResponse({"ok": True, "records": rows[-n:],
                         "total": len(rows),
                         "path": path,
                         "rule": ("every ANALYSE run is appended here — "
                                  "success, duplicate, queue or failure")})


@app.get("/api/genesis/status")
async def genesis_status() -> JSONResponse:
    """§7 pool counts, next generation time, rules (NumPy only)."""
    return JSONResponse({"ok": True, **engine.genesis.status(),
                         "vote": engine.genesis.live_score({})})


@app.get("/api/genesis/formulas")
async def genesis_formulas() -> JSONResponse:
    """§7 every candidate with its seven metrics + lifecycle state."""
    return JSONResponse({"ok": True, "formulas": engine.genesis.formulas_payload()})


@app.get("/api/genesis/formulas/{fid:path}")
async def genesis_formula(fid: str) -> JSONResponse:
    d = engine.genesis.formula_detail(fid)
    if not d:
        return JSONResponse({"ok": False, "error": "unknown formula"},
                            status_code=404)
    return JSONResponse({"ok": True, "formula": d})


@app.get("/api/genesis/domains")
async def genesis_domains() -> JSONResponse:
    """§7 the ten domains with the maths of each."""
    return JSONResponse({"ok": True, "domains": engine.genesis.domains_payload()})


@app.get("/api/ops")
async def ops() -> JSONResponse:
    """§12 ops row: degradation + vault + watchdog + uptime + build id."""
    return JSONResponse({"ok": True, **engine.ops_status()})


@app.get("/api/degrade")
async def degrade() -> JSONResponse:
    """§9.4 honest degradation levels with reasons + what still serves."""
    return JSONResponse({"ok": True, **engine.degrade_status()})


@app.get("/api/secrets/status")
async def secrets_status() -> JSONResponse:
    """§9.1 vault view: which secret SOURCES exist — values never shown,
    never written to any file in the repo (I-5)."""
    import os as _os
    from .agents import KEY_FIELDS, SECRETS_FILE, SETTINGS_FILE
    env_present = {k: bool(_os.environ.get(k)) for k in
                   ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY",
                    "ANTHROPIC_API_KEY", "NEWS_API_KEY", "NEUPRINT_TOKEN")}
    configured = {k: k in engine.agents.settings and
                  bool(engine.agents.settings.get(k)) for k in KEY_FIELDS}
    on_disk = False
    try:
        txt = open(SETTINGS_FILE, encoding="utf-8").read()
        on_disk = any(f'"{k}"' in txt for k in KEY_FIELDS)
    except OSError:
        pass
    secrets_file = bool(_os.path.exists(SECRETS_FILE))
    # keyring probe runs agents.status() (sync Ollama ping) — off the loop
    keyring = (await asyncio.to_thread(engine.agents.status)).get(
        "keyring") or {}
    # Git-ignored, owner-only secret store. This is intentional and is NOT
    # "a leak": it is the only way a pasted key survives a restart without
    # putting it in a tracked file.
    return JSONResponse({
        "ok": True,
        "env": env_present,
        "settings_file_exists": _os.path.exists(SETTINGS_FILE),
        "secrets_file": secrets_file,
        "configured": configured,
        "keys_on_disk": on_disk,
        "keyring": keyring,
        "rule": ("tracked settings never contain a key; the local store "
                 "data/secrets.json is chmod 600 and git-ignored. "
                 "configured=true means the agent will try that key now."),
    })


@app.get("/placeholder")
async def placeholder() -> HTMLResponse:
    """§9/J-2: a real page (never an empty body) when the app is not
    ready — explains readiness level and links health/watchdog."""
    d = engine.degrade_status()
    rows = "".join(f"<li><b>L{k}</b> — {v}</li>"
                   for k, v in d["all_levels"].items())
    reasons = "".join(f"<li>{r}</li>" for r in d["reasons"]) or "<li>none</li>"
    return HTMLResponse(f"""<!doctype html><html><head><meta charset="utf-8">
<title>app not ready</title>
<style>body{{font:15px/1.6 system-ui;background:#0b0e14;color:#dfe6f0;
padding:40px;max-width:760px;margin:auto}} .chip{{display:inline-block;
border:1px solid #2c3542;border-radius:999px;padding:3px 10px;
font-size:12px}} a{{color:#5ad1ff}}</style></head><body>
<h1>System warming up — L{d['level']}: {d['name']}</h1>
<p class="chip">{engine.phase}</p>
<p>Reasons:</p><ul>{reasons}</ul>
<p>Never-ready page (advice J-2): health is always served at
<a href="/api/health">/api/health</a>, watchdog at <a href="/api/watchdog">/api/watchdog</a>,
degradation at <a href="/api/degrade">/api/degrade</a>.</p>
<p>Levels:</p><ul>{rows}</ul></body></html>""")


@app.get("/api/settings/effects")
async def settings_effects() -> JSONResponse:
    """§10.5 (D-5): which settings change data behaviour — the UI refetches
    when one of these changes instead of showing stale data."""
    return JSONResponse({
        "ok": True,
        "data_settings": ["feed_mode", "symbol", "tf", "focus_tf",
                          "news_enabled"],
        "cosmetic_settings": ["haptics", "chart_style"],
        "rule": ("data-affecting changes force a full refetch before the "
                 "next render; cosmetic changes apply immediately"),
        "feed_note": ("FEED_MODE=binance is strict (refuses sim); auto "
                      "graces 120s then falls back with the chip flagged"),
    })


@app.get("/api/forward")
async def forward_tables(symbol: str | None = None) -> JSONResponse:
    """§2.7 forward expectancy: P(up | reading bucket) per formula × horizon."""
    return JSONResponse({"ok": True, **engine.forward.payload(symbol)})


@app.get("/api/ledger")
async def ledger_api() -> JSONResponse:
    """Strict-sign settlement ledger — the ONLY accuracy source (G-1)."""
    from .config import SYMBOLS, TIMEFRAMES
    return JSONResponse({"ok": True,
                         **engine.ledger.api_payload(list(SYMBOLS),
                                                     list(TIMEFRAMES))})


@app.get("/api/auditor/status")
async def auditor_status() -> JSONResponse:
    """Local Ollama formula auditor (qwen2.5-coder:3b) status + reports."""
    return JSONResponse({"ok": True,
                         **await asyncio.to_thread(engine.auditor.status)})


@app.post("/api/agents/triage")
async def agents_triage(req: dict) -> JSONResponse:
    """Fast Groq (8B) sentiment triage of one news headline."""
    text = str(req.get("text", ""))[:500]
    if not text:
        return JSONResponse({"ok": False, "error": "empty text"}, status_code=400)
    return JSONResponse(await engine.agents.triage_headline(text))


@app.get("/api/agents/catalog")
async def agents_catalog() -> JSONResponse:
    """Downloadable agent packs + auto-best scorecard."""
    return JSONResponse({"ok": True, **engine.agents.catalog_status()})


@app.post("/api/agents/import")
async def agents_import(req: dict) -> JSONResponse:
    """Import a downloadable agent pack (paste the key once)."""
    out = engine.agents.import_pack(str(req.get("pack_id", "")),
                                    str(req.get("key", "")),
                                    req.get("slot"))
    return JSONResponse(out, status_code=200 if out.get("ok") else 400)


@app.get("/api/agents/pack/{pack_id}")
async def agents_pack(pack_id: str) -> JSONResponse:
    """Download an agent pack definition (JSON, key never included)."""
    pack = engine.agents.export_pack(pack_id)
    if pack is None:
        return JSONResponse({"ok": False, "error": "unknown pack"},
                            status_code=404)
    return JSONResponse({"ok": True, "pack": pack})


@app.post("/api/agents/auto")
async def agents_auto(req: dict) -> JSONResponse:
    """Enable/disable best-of-best auto-routing."""
    return JSONResponse(engine.agents.set_auto(bool(req.get("enabled"))))


@app.get("/api/agents/local/install")
async def agents_local_install_status() -> JSONResponse:
    """In-app local-agent download: state, phase, plain-word error, log."""
    return JSONResponse({"ok": True,
                         "install": engine.agents.installer.status()})


@app.post("/api/agents/local/install")
async def agents_local_install_start() -> JSONResponse:
    """Start (or report) the background local-agent download.

    Runs on a daemon thread and reports a phase for every step (runtime →
    model → load). The runtime ladder is prebuilt wheel → reachable source
    build → Ollama, and the model download is independent of it, so a host
    that cannot compile llama.cpp still ends up with a usable agent.
    """
    return JSONResponse({"ok": True,
                         "install": engine.agents.installer.start()})


@app.get("/api/agents/local/endpoints")
async def agents_local_endpoints() -> JSONResponse:
    """Which local server is answering, and what else could run right now.

    This is the answer to a blocked GGUF download: no transfer is needed if
    the user already has Ollama / LM Studio / llama-server running, and this
    reports every port it probed plus the routes that need nothing at all.
    """
    ag = engine.agents
    out = await asyncio.to_thread(ag.local_endpoint_status)
    out["alternatives"] = await asyncio.to_thread(
        engine.agents.installer._alternatives)
    out["ok"] = True
    return JSONResponse(out)


@app.post("/api/agents/local/endpoints")
async def agents_local_endpoints_set(req: dict) -> JSONResponse:
    """Point the app at a running server (``{"endpoint": "http://host:port"}``).

    An empty endpoint clears the override and returns to autodetection. The
    value is kept in the environment of the running process AND reported, so
    the Agents tab can show exactly what is being probed.
    """
    ep = str((req or {}).get("endpoint") or "").strip()
    from app import agents as _ag
    if ep:
        os.environ[_ag.LOCAL_ENDPOINT_ENV] = ep
    else:
        os.environ.pop(_ag.LOCAL_ENDPOINT_ENV, None)
    _ag.invalidate_endpoint_cache()
    ag = engine.agents
    local = ag.local
    local.reset()
    await asyncio.to_thread(local.try_load, True)
    out = await asyncio.to_thread(ag.local_endpoint_status)
    out.update(ok=True, endpoint=ep,
               local=local.describe())
    return JSONResponse(out)


@app.post("/api/agents/local/retry")
async def agents_local_retry() -> JSONResponse:
    """Probe the runtime again WITHOUT re-downloading.

    Useful after `ollama serve` or a wheel has been installed outside the
    app: it drops the cached load state and re-runs the ladder's detection
    step only, which is seconds instead of a 2 GB download.
    """
    local = engine.agents.local
    local.reset()
    inst = engine.agents.installer
    try:
        import asyncio as _a
        loop = _a.get_event_loop()
        await loop.run_in_executor(None, local.try_load, True)
    except Exception:
        pass
    return JSONResponse({"ok": local.available,
                         "backend": local.backend,
                         "label": local.label,
                         "error": local.error,
                         "install": inst.status()})


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
        # off the loop: building the snapshot is CPU, not I/O
        await ws.send_json(engine.last_snapshot
                           or await run_pass(engine.build_snapshot))
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
