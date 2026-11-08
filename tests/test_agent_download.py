"""In-app local-agent download (Agents tab button).

Boot no longer downloads the agent; the user presses DOWNLOAD
LOCAL AGENT and AgentInstaller does it in the background with
honest state/log reporting. These tests pin that contract.
"""
import os
import threading

from app import agents as ag


def _installer(tmp_path, monkeypatch, runner):
    monkeypatch.setattr(ag, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(ag.AgentInstaller, "LOG_FILE",
                        str(tmp_path / "data" / "setup_ai.log"))
    monkeypatch.setattr(ag.AgentInstaller, "DONE_FILE",
                        str(tmp_path / "data" / ".ai_done"))
    local = ag.LocalModel()
    return ag.AgentInstaller(local, runner=runner), local


def test_status_reports_not_installed_honestly(tmp_path, monkeypatch):
    inst, _ = _installer(tmp_path, monkeypatch, runner=lambda c, say: 0)
    st = inst.status()
    assert st["installed"] is False
    assert st["model_present"] is False
    assert st["state"] in ("idle", "installing", "done", "failed")
    assert st["model_label"]            # a tier is always named
    assert st["model_size"]             # ...with its size up front


def test_download_flow_installs_model_and_marks_done(tmp_path, monkeypatch):
    calls = []

    def runner(cmd, say):
        calls.append(cmd)
        say("fake step: " + " ".join(cmd[-2:]))
        if any("download_models" in part for part in cmd):
            path = None
            # create exactly the file the installer expects
            path = inst.model_file()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").write(b"GGUF-fake")
        return 0

    inst, _ = _installer(tmp_path, monkeypatch, runner=runner)
    st = inst.start()
    assert st["state"] == "installing"
    inst._thread.join(timeout=20)
    st = inst.status()
    assert st["state"] == "done", st
    assert st["model_present"] is True
    assert len(calls) == 2                       # runtime, then model
    assert any("requirements-agents" in " ".join(c) for c in calls)
    assert os.path.exists(inst.DONE_FILE)
    assert any("fake step" in line for line in st["log"])


def test_failed_download_reports_plain_reason_and_can_retry(tmp_path, monkeypatch):
    inst, _ = _installer(tmp_path, monkeypatch, runner=lambda c, say: 1)
    inst.start()
    inst._thread.join(timeout=20)
    st = inst.status()
    assert st["state"] == "failed"
    assert "runtime install failed" in st["error"]
    assert st["installed"] is False


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
    pip_calls = [c for c in calls if any("pip" in part for part in c)]
    assert len(pip_calls) == 1                    # the second press started
                                                  # no duplicate download


def test_routes_exist():
    from app.main import app
    paths = {r.path for r in app.routes}
    assert "/api/agents/local/install" in paths


def test_catalog_default_is_qwen25_coder_3b():
    assert ag.DEFAULT_MODEL_ID == "qwen25-coder-3b"
    best = ag.model_by_id(None)
    assert "Qwen2.5-Coder-3B" in best["label"]
    ids = [m["id"] for m in ag.MODEL_CATALOG]
    assert len(ids) >= 4 and len(set(ids)) == len(ids)   # multiple options
    repo, fname, _ = ag.pick_local_model(8.0)            # Codespace tier
    assert fname == best["file"]


def test_download_specific_model_by_id(tmp_path, monkeypatch):
    seen = {}

    def runner(cmd, say):
        if any("download_models" in part for part in cmd):
            seen["cmd"] = cmd
            path = inst.model_file("qwen25-coder-1.5b")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").write(b"GGUF-fake")
        return 0

    inst, local = _installer(tmp_path, monkeypatch, runner=runner)
    inst.start("qwen25-coder-1.5b")
    inst._thread.join(timeout=20)
    st = inst.status()
    assert st["state"] == "done", st
    assert st["target"] == "qwen25-coder-1.5b"
    assert local.model_id == "qwen25-coder-1.5b"
    assert "--model" in seen["cmd"]
    assert "qwen25-coder-1.5b" in seen["cmd"]
    by_id = {m["id"]: m for m in st["models"]}
    assert by_id["qwen25-coder-1.5b"]["present"] is True
    assert by_id["qwen25-coder-3b"]["recommended"] is True
    assert by_id["qwen25-coder-3b"]["present"] is False
