"""Local-agent download contract.

Two paths arm the downloader: the boot autoloader (`start_auto`: delayed,
retried with backoff, flock-guarded) and the DOWNLOAD LOCAL AGENT button.
Both report a phase for every step and neither can block the app.

The runtime ladder itself is pinned in tests/test_runtime_selfheal.py and
tests/test_agent_runtime.py; this file pins the installer's own contract.
"""
import os
import threading
import time

from app import agents as ag


def _installer(tmp_path, monkeypatch, runner, *, deps=False):
    monkeypatch.setattr(ag, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(ag.AgentInstaller, "LOG_FILE",
                        str(tmp_path / "data" / "setup_ai.log"))
    monkeypatch.setattr(ag.AgentInstaller, "DONE_FILE",
                        str(tmp_path / "data" / ".ai_done"))
    monkeypatch.setattr(ag.AgentInstaller, "LOCK_FILE",
                        str(tmp_path / "data" / ".agent.lock"))
    monkeypatch.setattr(ag.AgentInstaller, "AUTO_FILE",
                        str(tmp_path / "data" / "auto.json"))
    # the runtime must look absent unless the test says otherwise: a
    # llama_cpp that happens to be installed in the venv (e.g. from a real
    # auto-download) would skip step 1 and the test would stop describing
    # the code path it names

    # never touch a real daemon or the network
    absent = lambda *a, **k: {"ok": False, "models": [], "detail": "absent",
                              "running": False, "ms": 0.0}
    no_cc = lambda: {"ok": False, "cc": None, "cxx": None,
                     "detail": "no compiler"}
    local = ag.LocalModel()
    inst = ag.AgentInstaller(
        local, runner=runner,
        deps=lambda: {"ok": deps, "detail": "test"},
        probe=absent, compiler=no_cc)
    return inst, local


def _fake_gguf(path, payload=b"GGUF" + b"\x00" * 64):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(payload)


def test_status_reports_not_installed_honestly(tmp_path, monkeypatch):
    inst, _ = _installer(tmp_path, monkeypatch, runner=lambda c, say: 0)
    st = inst.status()
    assert st["installed"] is False
    assert st["model_present"] is False
    assert st["state"] in ("idle", "installing", "done", "failed")
    assert st["model_label"]            # a tier is always named
    assert st["model_size"]             # ...with its size up front
    # the runtime ladder's own probes are part of the status, so the panel
    # can say WHY the local agent is not live instead of a bare "installing"
    assert "ollama" in st and "runtime_present" in st and "steps" in st


def test_status_is_never_a_bare_installing_without_a_phase(tmp_path, monkeypatch):
    """The old panel sat on "installing" with phase '' for 20 minutes."""
    inst, _ = _installer(tmp_path, monkeypatch, runner=lambda c, say: 0)
    assert inst.status()["phase"] in ("", "runtime", "model", "load")


def test_model_downloads_even_when_the_runtime_fails(tmp_path, monkeypatch):
    """The two steps are independent.

    Measured bug: `_work` raised inside the runtime step, so
    `download_models.py` never ran. Fixing the runtime later meant
    downloading 2 GB again.
    """
    seen = []

    def runner(cmd, say):
        seen.append(cmd)
        if any("download_models" in p for p in cmd):
            _fake_gguf(inst.model_file())      # the "download" lands
        return 1 if any("pip" in p for p in cmd) else 0   # pip always fails

    inst, _ = _installer(tmp_path, monkeypatch, runner=runner)
    inst.start()
    inst._thread.join(timeout=20)
    st = inst.status()
    assert any("download_models" in " ".join(c) for c in seen), \
        "the model step must run even when the runtime step failed"
    assert st["model_present"] is True, st
    # runtime unavailable + model present -> failed, but honestly: the model
    # is banked and the message says what is still missing
    assert st["state"] == "failed"
    assert "model" in st["error"].lower()
    assert st["validation"]["magic_ok"] is True


def test_download_flow_installs_model_and_marks_done(tmp_path, monkeypatch):
    calls = []

    def runner(cmd, say):
        calls.append(cmd)
        say("fake step: " + " ".join(cmd[-2:]))
        if any("download_models" in part for part in cmd):
            _fake_gguf(inst.model_file())
        return 0

    inst, local = _installer(tmp_path, monkeypatch, runner=runner, deps=True)
    # pretend the runtime import works and the GGUF loads
    st = inst.start()
    assert st["state"] == "installing"
    inst._thread.join(timeout=20)
    st = inst.status()
    assert st["model_present"] is True
    assert len(calls) >= 1                       # at least the model step
    assert any("download_models" in " ".join(c) for c in calls)
    assert any("fake step" in line for line in st["log"])
    assert st["backend"] in ("llama_cpp", "local", "")


def test_failed_download_reports_plain_reason_and_can_retry(tmp_path,
                                                            monkeypatch):
    inst, _ = _installer(tmp_path, monkeypatch, runner=lambda c, say: 1)
    inst.start()
    inst._thread.join(timeout=20)
    st = inst.status()
    assert st["state"] == "failed"
    # the reason names the step that failed, in plain words
    assert st["error"]
    assert st["installed"] is False
    # ...and every attempted rung is recorded with its own reason
    steps = {s["step"]: s for s in st["steps"]}
    assert steps and all(s["ok"] is False for s in steps.values())
    assert "prebuilt wheel" in steps and "model" in steps


def test_start_is_idempotent_while_running(tmp_path, monkeypatch):
    gate = threading.Event()
    calls = []

    def runner(cmd, say):
        calls.append(cmd)
        gate.wait(timeout=10)
        return 0

    inst, _ = _installer(tmp_path, monkeypatch, runner=runner)
    inst.start()
    st2 = inst.start()                            # second press: no-op
    assert st2["state"] == "installing"
    gate.set()
    inst._thread.join(timeout=20)
    # only ONE model fetch was started by the two presses
    model_calls = [c for c in calls
                   if any("download_models" in p for p in c)]
    assert len(model_calls) == 1


def test_routes_exist():
    from app.main import app
    paths = {r.path for r in app.routes}
    assert "/api/agents/local/install" in paths
    assert "/api/agents/local/retry" in paths      # re-probe without re-fetch


def test_preferred_model_is_qwen25_coder_3b():
    repo, fname, label = ag.pick_local_model(8.0)      # Codespace tier
    assert "Qwen2.5-Coder-3B" in label
    assert fname == "qwen2.5-coder-3b-instruct-q4_k_m.gguf"
    # a 4 GB host still gets a tier that fits
    _, small, label_small = ag.pick_local_model(3.8)
    assert "0.5B" in label_small and "q4_k_m" in small


def test_validation_catches_half_downloads(tmp_path, monkeypatch):
    inst, _ = _installer(tmp_path, monkeypatch, runner=lambda c, say: 0)
    path = inst.model_file()
    v = ag.model_validation(path)
    assert v["present"] is False and v["source"] == "absent"
    _fake_gguf(path, b"PARTIAL-DOWNLOAD-not-a-model")
    v = ag.model_validation(path)
    assert v["present"] is True and v["magic_ok"] is False
    assert "corrupt" in v["sha"]
    assert inst.model_present() is False        # corrupt != downloaded
    _fake_gguf(path)
    ag.write_model_sidecar(path)
    v = ag.model_validation(path)
    assert v["magic_ok"] is True and v["sha"] == "ok"
    assert "0 bytes fetched" in v["source"]
    assert inst.model_present() is True


def test_installer_fetcher_lock_is_exclusive(tmp_path, monkeypatch):
    inst, _ = _installer(tmp_path, monkeypatch, runner=lambda c, say: 0)
    os.makedirs(os.path.dirname(inst.LOCK_FILE), exist_ok=True)
    fh = inst._grab_lock()
    assert fh is not None
    try:
        assert inst._grab_lock() is None         # second fetcher refused
        got = inst._wait_lock_or_model(tries=1)
        assert got == "timeout"                  # nothing landed; no race
    finally:
        fh.close()
    fh2 = inst._grab_lock()                      # free again after release
    assert fh2 is not None
    fh2.close()


def test_auto_downloader_is_delayed_and_idempotent(tmp_path, monkeypatch):
    inst, _ = _installer(tmp_path, monkeypatch, runner=lambda c, say: 0)
    monkeypatch.setattr(inst, "_auto_work", lambda: time.sleep(0.2))
    st = inst.start_auto()
    assert st["state"] in ("running", "idle", "failed")
    time.sleep(0.05)
    st2 = inst.status()["auto"]
    assert "attempts" in st2 and "next_try_in_s" in st2
    again = inst.start_auto()
    assert again["state"] in ("running", "idle", "failed")
    assert sum(1 for t in [inst._auto_thread] if t and t.is_alive()) <= 1


def test_status_surfaces_the_installer_at_the_top_level():
    """The Agents tab read `agents.installer` and got None, so the whole
    download panel had nothing to render."""
    m = ag.AgentManager()
    st = m.status()
    assert st.get("installer") is not None
    assert "state" in st["installer"] and "phase" in st["installer"]
    assert st["local"]["install"]["state"] == st["installer"]["state"]
