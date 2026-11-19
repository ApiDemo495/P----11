"""Regression tests for the two faults that kept coming back on the phone:
the red "ENGINE BUSY — a long pass is running" banner and the false
"brain offline — vote withheld (RECONNECT)" chip.

Both were measured, not guessed:

* ``/api/snapshot`` was rebuilt on every poll (1.4 s of CPU) because it
  probed Ollama 16 times per build and each probe built a fresh httpx
  client, which re-read the whole CA bundle (~24 ms). The route's 0.75 s
  TTL was shorter than the build, so every single request started a
  rebuild while holding the lock.
* A stalled or killed connectome build was indistinguishable from a working
  one: ``ensure()`` returns early on "building", ``heal()`` only acted on
  "absent", and there was no heartbeat to detect a dead worker — so the
  chip showed "building" (or a bare "offline") forever.

These tests fail if either regression returns.
"""
import ast
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _src(rel: str) -> str:
    return (ROOT / rel).read_text()


# ── the probe no longer rebuilds an SSL context per call ────────────────
def test_ollama_probe_is_cached_and_reuses_one_client():
    src = _src("app/agents.py")
    assert "_HTTP_LOCAL = httpx.Client(" in src
    assert "def _probe_uncached(" in src
    assert "_PROBE_CACHE" in src
    # a per-call httpx.get() would recreate the transport every time
    assert "r = httpx.get(url" not in src


def test_probe_cache_returns_quickly_and_can_be_invalidated():
    from app import agents

    agents.invalidate_probe()
    t0 = time.perf_counter()
    first = agents.probe_ollama(timeout=0.05)
    first_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    second = agents.probe_ollama(timeout=0.05)
    second_ms = (time.perf_counter() - t0) * 1000
    assert second["cached"] is True
    assert first["ok"] is second["ok"] is False       # nothing listens here
    # the memo must actually skip the socket attempt, not merely relabel it
    assert second_ms < max(5.0, first_ms)
    agents.invalidate_probe()
    assert agents.probe_ollama(timeout=0.05)["cached"] is False


# ── the snapshot route cannot become a self-feeding rebuild loop ────────
def test_snapshot_ttl_longer_than_a_build_and_single_flight():
    src = _src("app/main.py")
    tree = ast.parse(src)
    ttl = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if getattr(t, "id", None) == "SNAPSHOT_TTL_S":
                    ttl = ast.literal_eval(node.value)
    assert ttl is not None and ttl >= 3.0, (
        "the TTL must survive several panel polls; a TTL shorter than the "
        "build made every poll start a rebuild under the lock"
    )
    # single-flight: concurrent callers get the last payload instead of
    # queueing behind the CPU build
    assert '_snap_cache["building"]' in src
    assert "single-flight" in src.lower()


def test_snapshot_payload_is_memoised_within_ttl():
    os_environ_backup = None
    import os
    os_environ_backup = os.environ.get("PREDICT_SKIP_WARMUP")
    os.environ["PREDICT_SKIP_WARMUP"] = "1"
    try:
        from app import main as M

        M._snap_cache.update(at=0.0, body=None, bytes=b"", ms=0.0,
                            building=False)
        t0 = time.perf_counter()
        M._snapshot_payload()
        cold = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        M._snapshot_payload()
        warm = (time.perf_counter() - t0) * 1000
        assert warm < max(2.0, cold / 4), (
            f"second payload call must be a cache hit, got {warm:.1f} ms "
            f"vs {cold:.1f} ms")
        assert M._snap_cache["building"] is False
    finally:
        if os_environ_backup is None:
            os.environ.pop("PREDICT_SKIP_WARMUP", None)
        else:
            os.environ["PREDICT_SKIP_WARMUP"] = os_environ_backup


# ── heavy GRC work must not run on the event loop ───────────────────────
def test_grc_compute_runs_off_the_event_loop():
    src = _src("app/engine.py")
    i = src.index("async def on_grc(")
    body = src[i:i + 4200]
    assert "await run_pass(" in body, (
        "compute_asset is seconds of CPU per asset; inline in an async def it "
        "freezes /api/health and the websocket (the ENGINE BUSY banner)")
    assert "res_pack = self.compute_asset(" not in body, (
        "GRC must not call compute_asset synchronously on the loop")
    broadcast = src[src.index("async def _broadcast_full("):]
    assert "await run_pass(self.build_snapshot)" in broadcast[:600]
    # and the pool must be the engine's own, never asyncio.to_thread's
    # default executor — that is the SAME pool Starlette serves sync routes
    # from, so heavy passes there just moved the stall into /api/health
    assert "asyncio.to_thread" not in src.split("class PredictionEngine")[1]
    assert "_pass_pool = _fut.ThreadPoolExecutor(" in src


def test_readiness_dry_run_backs_off_instead_of_repeating_every_grc():
    src = _src("app/engine.py")
    i = src.index("def _update_readiness(")
    body = src[i:i + 2600]
    assert "_readiness_next_at" in body and "backoff" in body.lower(), (
        "an unbounded dry-run re-burned ~10 s of CPU on every GRC while "
        "physics warmed up")
    assert "self._readiness_tries" in src


# ── a stalled brain build is measurable and self-healing ────────────────
def _fake_fw():
    from app.flywire import FlyWireBrain
    fw = FlyWireBrain.__new__(FlyWireBrain)          # no file I/O, no numpy load
    fw.state = "building"
    fw.progress = {"stage": "downloading", "pct": 4.0}
    fw.build_started_at = time.time()
    fw.build_beat_at = time.time()
    fw.attempt = 1
    fw.build_pid = None
    fw.last_error = ""
    fw.last_pass_at = 0.0
    fw.last_pass_ms = None
    fw.meta = {}
    fw._acquire_thread = None
    fw._last_supervise_at = 0.0
    return fw


def test_building_brain_is_blue_not_offline():
    fw = _fake_fw()
    chip = fw.chip()
    assert chip["kind"] == "build", "a loading brain must never be red"
    assert "vote withheld" not in chip["text"]
    assert "4 %" in chip["text"] or "4%" in chip["text"]


def test_lost_build_worker_is_detected():
    fw = _fake_fw()
    assert fw._classify_worker() == "dead"      # thread attribute is None
    assert "worker lost" in fw.progress_line()
    assert fw.chip()["stalled"] is True


def test_wedged_build_is_detected_from_the_heartbeat():
    fw = _fake_fw()
    th = threading.Thread(target=time.sleep, args=(30,), daemon=True)
    th.start()
    fw._acquire_thread = th
    assert fw._classify_worker() == "working"
    fw.build_beat_at = time.time() - (fw.BEAT_TIMEOUT_S + 5)
    assert fw._classify_worker() == "wedged"
    assert fw.stalled_for() is not None
    assert "stalled" in fw.progress_line()
    th.join(timeout=0.01)


def test_supervise_restarts_a_lost_build(monkeypatch):
    from app import flywire as F

    fw = _fake_fw()
    started = {"n": 0}

    def _fake_ensure():
        started["n"] += 1
        fw.state = "building"

    monkeypatch.setattr(fw, "ensure", _fake_ensure)
    out = fw.supervise()
    assert out["action"] == "restarted"
    assert started["n"] == 1
    assert "self-heal" in fw.last_error
    # rate limited: an instantly-failing build must not spin
    fw.state = "building"
    fw._acquire_thread = None
    assert fw.supervise()["action"] == "restart-suppressed"


def test_engine_heal_covers_building_state():
    src = _src("app/engine.py")
    i = src.index("def heal(")
    body = src[i:i + 2200]
    assert 'fw.state == "building"' in body and "supervise()" in body, (
        "heal() only reacted to 'absent', so a build that died in 'building' "
        "was never repaired — the chip stayed unproven indefinitely")


# ── the GRC broadcast feeds the REST cache (one build, not two) ─────────
def test_broadcast_priming_avoids_a_second_build(monkeypatch):
    import os
    os.environ.setdefault("PREDICT_SKIP_WARMUP", "1")
    from app import main as M

    M._snap_cache.update(at=0.0, body=None, bytes=b"", ms=0.0, building=False)
    calls = {"n": 0}
    real_build = M.engine.build_snapshot

    def _counting_build():
        calls["n"] += 1
        return real_build()

    monkeypatch.setattr(M.engine, "build_snapshot", _counting_build)
    snap = M.engine.build_snapshot()
    calls["n"] = 0
    M.prime_snapshot_cache(snap)
    assert calls["n"] == 0, "priming must not rebuild the snapshot"
    import time as _t
    t0 = _t.perf_counter()
    body = M._snapshot_payload()
    assert (_t.perf_counter() - t0) < 0.05
    assert calls["n"] == 0, "a primed cache must serve the REST poll directly"
    assert body.get("brain") is not None
    assert cache_is_fresh(M)


def cache_is_fresh(M) -> bool:
    import time as _t
    return _t.time() - M._snap_cache["at"] < 1.0


def test_concurrent_snapshot_callers_share_one_build(monkeypatch):
    """Two panels polling at the same instant must not run two builds."""
    import os
    import time as _t
    os.environ.setdefault("PREDICT_SKIP_WARMUP", "1")
    from app import main as M

    M._snap_cache.update(at=0.0, body=None, bytes=b"", ms=0.0, building=False)
    M._snap_flight.update(owner=None, done=None, body=None, raw=b"", ms=0.0)
    calls = {"n": 0}
    real = M.engine.build_snapshot

    def _slow_build():
        calls["n"] += 1
        _t.sleep(0.4)                     # stand in for a multi-second build
        return real()

    monkeypatch.setattr(M.engine, "build_snapshot", _slow_build)
    results = []
    import threading as _th
    threads = [_th.Thread(target=lambda: results.append(M._snapshot_payload()))
               for _ in range(4)]
    t0 = _t.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    elapsed = _t.perf_counter() - t0
    assert calls["n"] == 1, f"expected ONE shared build, got {calls['n']}"
    assert len(results) == 4 and all(results)
    assert elapsed < 2.0                    # not 4 x 0.4 s serialised
    assert M._snap_flight["owner"] is None  # no stuck flight marker


def test_run_pass_executes_off_loop_on_its_own_pool():
    """run_pass is the one gateway for heavy passes — it must actually run."""
    import asyncio
    import threading
    from app.engine import run_pass

    async def main():
        me = threading.get_ident()
        where = {}

        def _work(x, y=0):
            where["thread"] = threading.get_ident()
            return x + y

        out = await run_pass(_work, 2, y=3)
        return out, where["thread"] != me

    out, off_loop = asyncio.run(main())
    assert out == 5
    assert off_loop, "a heavy pass must not run on the event-loop thread"


# ── bulk work yields to the measured loop lag (the GIL tax) ────────────
def test_load_governor_scales_yield_to_measured_lag():
    from app import loadgov

    loadgov.note_lag(0.0)
    t0 = time.perf_counter()
    loadgov.cooperate()
    idle = time.perf_counter() - t0
    assert idle < 0.02, "a healthy loop must not be taxed"

    loadgov.note_lag(0.9)
    t0 = time.perf_counter()
    loadgov.cooperate()
    loaded = time.perf_counter() - t0
    assert loaded >= 0.2, f"a starved loop must get the CPU back, got {loaded:.3f}s"

    # a stale reading must decay, or one bad measurement taxes forever
    loadgov._LAG["at"] = time.time() - 30
    assert loadgov.current() == 0.0
    loadgov.note_lag(0.0)


def test_bulk_workers_call_the_governor():
    """The warm replay and the analyse stages must actually yield."""
    warm = _src("app/warmup.py")
    assert "loadgov.cooperate()" in warm
    eng = _src("app/engine.py")
    assert eng.count("_gov.cooperate()") >= 4
    # the meter must feed it, or cooperate() is blind
    assert "loadgov.note_lag(" in _src("app/main.py")


def test_stall_is_self_diagnosing():
    """A blocked loop must dump stacks, not just report a number."""
    main = _src("app/main.py")
    assert "def _dump_stacks(" in main
    assert "faulthandler.dump_traceback(" in main
    assert "_dump_stacks(f\"lag {_loop_lag['s']}s\")" in main
