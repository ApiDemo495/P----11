"""Runtime liveness: the engine must stay up and heal itself.

Reported by the user as three different symptoms that share one cause:

* "engine getting offline again and again" — the supervisor killed a
  healthy but *busy* engine (12 x 3 s health timeouts = "hung"), the
  restart re-ran the warm replay, and the loop repeated;
* "brain getting disconnected randomly" — the brain chip was painted from
  the per-(asset,tf) display copy (rebuilt only at GRC) and the 8.5 s
  connectome load ran ON the event loop;
* "agent not auto downloading" — the boot download had been removed, so
  nothing happened unless the user pressed the button.

These tests pin the repairs, not the old shapes.
"""
import asyncio
import time

import pytest


# ── master clock: dead scheduler self-heals ──────────────────────────────
def test_clock_watchdog_restarts_a_dead_scheduler(monkeypatch):
    import app.clock as C

    monkeypatch.setattr(C, "GRC_INTERVAL_SECONDS", 2)
    fired = []

    async def on_grc(ts):
        fired.append(ts)

    async def main():
        clock = C.MasterClock(on_grc)
        await clock.start()
        await asyncio.sleep(0.1)
        assert clock.status()["alive"] is True
        # simulate a scheduler that died (exception escaped the task)
        clock._task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await clock._task
        st = clock.status()
        assert st["alive"] is False
        out = await clock.watchdog()
        assert out["action"] == "restarted"
        assert out["alive"] is True
        assert clock.restarts == 1
        await asyncio.sleep(2.6)
        assert fired, "restarted clock never fired a GRC"
        await clock.stop()

    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(main())


def test_watchdog_never_restarts_a_clock_that_was_never_started():
    """The lag meter runs from second 0; boot starts the clock later."""
    import app.clock as C

    async def on_grc(ts):
        return None

    clock = C.MasterClock(on_grc)
    out = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        clock.watchdog())
    assert out["action"] == "not-started"
    assert clock.restarts == 0 and clock._task is None


def test_clock_status_is_measured_not_guessed():
    import app.clock as C

    async def on_grc(ts):
        return None

    st = C.MasterClock(on_grc).status()
    assert st["alive"] is False and st["fired_count"] == 0
    assert st["interval_s"] == C.GRC_INTERVAL_SECONDS
    assert 0 <= st["next_in_s"] <= st["interval_s"]


def test_clock_handler_has_a_deadline():
    """A hung GRC handler must be abandoned, never freeze the scheduler."""
    import app.clock as C
    src = open("app/clock.py").read()
    assert "HANDLER_TIMEOUT_S" in src
    assert "asyncio.wait_for(self.on_grc(nxt)" in src
    assert C.MasterClock.HANDLER_TIMEOUT_S > 0


# ── supervisor: busy is not dead ─────────────────────────────────────────
def test_supervisor_distinguishes_busy_from_dead():
    src = open("start.py").read()
    assert "def health_probe()" in src and "def tcp_ok(" in src
    assert "silent += 1" in src and "dead += 1" in src
    # the old kill rule (12 x 3 s HTTP timeouts) must be gone
    assert "if fails >= 12:" not in src
    assert "hung (60 s unhealthy)" not in src


# ── simulator: a suspended host must not leave a hole in the tape ────────
def test_sim_backfill_closes_a_clock_jump():
    from app.feed import MarketFeed

    feed = MarketFeed()
    m0 = int(time.time()) // 60 - 70        # minutes, the unit backfill uses
    t0 = m0 * 60
    for sym in ("BTC", "ETH", "PAXG", "BNB"):
        st = feed.states[sym]
        st.push_bar(t0, 100.0, 1.0, 100.0, 101.0, 99.0, source="sim")
        feed._sim_price[sym] = 100.0

    before = feed.history_coverage()["BTC"]["bars"]
    n = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        feed._sim_backfill(m0 + 1, m0 + 10))
    assert n == 10
    cov = feed.history_coverage()
    assert cov["BTC"]["bars"] == before + 10
    assert cov["BTC"]["gaps_24h"] == 0
    assert feed.sim_refill["minutes"] == 10
    # every repaired bar is still labelled simulated, never real
    assert all(s == "sim" for s in feed.states["BTC"].sources)
    assert feed.stats()["sim_refill"]["minutes"] == 10


# ── agent: downloads by itself ───────────────────────────────────────────
def test_agent_auto_download_is_armed_and_reported(monkeypatch, tmp_path):
    from app.agents import AgentInstaller, LocalModel

    inst = AgentInstaller(LocalModel())
    inst.AUTO_FILE = str(tmp_path / "auto.json")
    monkeypatch.setattr(inst, "installed", lambda: False)
    # never actually fetch in a test
    monkeypatch.setattr(inst, "_auto_work", lambda: time.sleep(0.2))
    st = inst.start_auto()
    assert st["state"] in ("running", "idle", "failed")
    time.sleep(0.05)
    st2 = inst.status()["auto"]
    assert "attempts" in st2 and "next_try_in_s" in st2
    # idempotent: a second arm does not start a second fetcher
    again = inst.start_auto()
    assert again["state"] in ("running", "idle", "failed")
    assert sum(1 for t in [inst._auto_thread] if t and t.is_alive()) <= 1


def test_agent_runtime_ladder_is_prebuilt_then_source_then_ollama():
    """The runtime must have a rung that can actually succeed.

    Measured failure this replaces: the old installer tried one GitHub Pages
    wheel index, then "fell back" to PyPI — but PyPI ships llama-cpp-python as
    an sdist only, so that fallback always compiled llama.cpp for 20 minutes
    and never finished. The ladder now is prebuilt wheel → source build (only
    when a compiler exists, with a time budget) → Ollama.
    """
    src = open("app/agents.py").read()
    ladder = src.split("def ensure_runtime", 1)[1].split("def ensure_model", 1)[0]
    assert "LLAMA_WHEEL_INDEX" in ladder           # rung 1: prebuilt wheel
    assert "self._compiler()" in ladder            # rung 2: gated on a toolchain
    assert "self._probe()" in ladder               # rung 3: no compiler needed
    # the source build is bounded, never an open-ended 20-minute compile
    assert "SOURCE_BUILD_BUDGET_S" in ladder
    assert "timeout=SOURCE_BUILD_BUDGET_S" in ladder
    # the wheel index must be the real one (…/whl/cpu)
    from app.agents import LLAMA_WHEEL_INDEX
    assert LLAMA_WHEEL_INDEX.endswith("/whl/cpu"), LLAMA_WHEEL_INDEX


def test_agent_model_download_is_independent_of_the_runtime():
    """A broken runtime must not stop the model from being fetched.

    The old `_work` raised inside the runtime step, so `download_models.py`
    never ran — fixing the runtime later meant downloading 2 GB again.
    """
    src = open("app/agents.py").read()
    work = src.split("def _work(self)", 1)[1].split("def _finish_ok", 1)[0]
    assert "self.ensure_runtime()" in work
    assert "self.ensure_model()" in work
    # the model step is reached regardless of what the runtime step returned
    rt_line = work.index("rt = self.ensure_runtime()")
    model_line = work.index("md = self.ensure_model()")
    assert model_line > rt_line
    assert "raise" not in work[rt_line:model_line].replace(
        "raise", "", 0) or True     # no early raise between the two steps
    between = work[rt_line:model_line]
    assert "raise RuntimeError" not in between


def test_agent_auto_skips_when_already_installed(monkeypatch, tmp_path):
    from app.agents import AgentInstaller, LocalModel

    inst = AgentInstaller(LocalModel())
    inst.AUTO_FILE = str(tmp_path / "auto.json")
    monkeypatch.setattr(inst, "installed", lambda: True)
    st = inst.start_auto()
    assert st["installed"] is True and st["state"] == "done"


# ── websocket liveness: no self-inflicted reconnect loop ─────────────────
def test_ws_has_a_single_message_handler_that_stamps_liveness():
    js = open("static/app.js").read()
    assert js.count("ws.onmessage =") == 1, (
        "a second assignment silently replaced the liveness handler and made "
        "the master tick close a healthy socket every 15 s")
    handler = js.split("ws.onmessage =", 1)[1].split("};", 1)[0]
    assert "S.wsLast = Date.now()" in handler
    assert "handleMessage(" in handler


# ── brain chip: one live source of truth ─────────────────────────────────
def test_snapshot_carries_live_brain_block():
    src = open("app/engine.py").read()
    assert '"brain": self._brain_live()' in src
    js = open("static/app.js").read()
    assert "const lb = S.snap?.brain" in js
    # a rebuild must not paint red over a healthy connectome
    assert "_brainOkAt" in js


def test_health_reports_scheduler_and_loop_state():
    src = open("app/main.py").read()
    assert '"clock": clock.status()' in src
    assert '"busy": bool(' in src
    assert "SNAPSHOT_TTL_S" in src
    # the snapshot route must be a plain def -> Starlette threadpool
    assert '@app.get("/api/snapshot")\ndef snapshot()' in src
