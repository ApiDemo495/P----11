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
