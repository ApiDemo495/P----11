"""Automation tests — self-bootstrapping setup, model tiering, auto-start."""
from __future__ import annotations

import json
import os
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bash_syntax(*paths: str) -> None:
    for p in paths:
        full = os.path.join(ROOT, p)
        assert os.path.exists(full), f"missing {p}"
        r = subprocess.run(["bash", "-n", full], capture_output=True, text=True)
        assert r.returncode == 0, f"bash -n {p}: {r.stderr}"


def test_shell_scripts_valid():
    _bash_syntax("run.sh", "scripts/setup.sh", "scripts/auto_start.sh",
                 "scripts/learning_cron.sh")


def test_run_sh_bootstraps_before_start():
    src = open(os.path.join(ROOT, "run.sh")).read()
    assert "scripts/setup.sh" in src          # auto-setup hook
    assert "SKIP_SETUP" in src                # escape hatch
    assert "uvicorn app.main:app" in src


def test_setup_sh_covers_all_stages():
    src = open(os.path.join(ROOT, "scripts/setup.sh")).read()
    for needle in (".venv", "requirements.txt", "requirements-agents.txt",
                   "download_models.py", "learning_cron.sh", "data/history",
                   "SETUP SUMMARY"):
        assert needle in src, f"setup.sh missing {needle}"
    # AI/model stages must NOT gate startup: they run in the background
    assert "background" in src
    assert "setup_ai.log" in src
    assert "nohup" in src


def test_auto_start_is_fast_when_fresh():
    src = open(os.path.join(ROOT, "scripts/auto_start.sh")).read()
    # setup runs ONLY when the environment is stale; fresh boots skip it
    assert "environment stale — running setup" in src
    assert "environment fresh — skipping setup" in src
    assert "app already healthy" in src          # health gate before anything


def test_auto_start_supervisor():
    src = open(os.path.join(ROOT, "scripts/auto_start.sh")).read()
    assert "scripts/setup.sh" in src
    assert "/api/health" in src               # waits for health
    assert "--stop" in src and "--restart" in src
    assert "nohup" in src                     # background process


def test_devcontainer_full_automation():
    cfg = json.load(open(os.path.join(ROOT, ".devcontainer", "devcontainer.json")))
    assert cfg["postCreateCommand"] == "bash scripts/setup.sh"
    assert cfg["postStartCommand"] == "bash scripts/auto_start.sh"
    assert 8000 in cfg["forwardPorts"]


def test_download_models_tier_selection():
    py = os.path.join(ROOT, ".venv", "bin", "python")
    if not os.path.exists(py):
        py = "python3"
    script = os.path.join(ROOT, "scripts", "download_models.py")
    out16 = subprocess.run([py, script, "--dry-run", "--ram", "16"],
                           capture_output=True, text=True, cwd=ROOT)
    assert out16.returncode == 0
    assert "Phi-3-mini" in out16.stdout
    out8 = subprocess.run([py, script, "--dry-run", "--ram", "8"],
                          capture_output=True, text=True, cwd=ROOT)
    assert "TinyLlama" in out8.stdout
    out2 = subprocess.run([py, script, "--dry-run", "--ram", "2"],
                          capture_output=True, text=True, cwd=ROOT)
    assert "Qwen2-0.5B" in out2.stdout
    assert "DONE" not in out16.stdout         # dry-run never claims a download


def test_agents_local_model_tiers_match_spec():
    from app.agents import LOCAL_MODELS
    tiers = [(t[0], t[3]) for t in LOCAL_MODELS]
    assert tiers[0][0] == 14 and "Phi-3-mini" in tiers[0][1]
    assert tiers[1][0] == 7 and "TinyLlama" in tiers[1][1]
    assert tiers[2][0] == 0 and "Qwen2-0.5B" in tiers[2][1]


def test_hourly_learning_safety_loop_wired():
    import importlib
    main = importlib.import_module("app.main")
    assert hasattr(main, "_hourly_learning_loop")
    src = open(os.path.join(ROOT, "app", "main.py")).read()
    assert "_hourly_learning_loop()" in src
    assert "asyncio.create_task(_hourly_learning_loop())" in src
    # loop must persist macro timestamps
    assert "last_macro_ts" in open(os.path.join(ROOT, "app", "learning.py")).read()


def test_learning_state_roundtrips_macro_ts(monkeypatch, tmp_path):
    from app import config
    from app.learning import LearningSystem
    monkeypatch.setattr(config, "LEARN_STATE_FILE", str(tmp_path / "s.json"))
    monkeypatch.setattr("app.learning.LEARN_STATE_FILE", str(tmp_path / "s.json"))
    ls = LearningSystem()
    ls.last_macro_ts = 1712345678.5
    ls.save()
    ls2 = LearningSystem()
    ls2.load_state()
    assert ls2.last_macro_ts == pytest.approx(1712345678.5)


def test_requirements_agents_has_build_toolchain():
    src = open(os.path.join(ROOT, "requirements-agents.txt")).read()
    assert "llama-cpp-python" in src
    assert "huggingface_hub" in src
    assert "cmake" in src          # enables source builds anywhere
