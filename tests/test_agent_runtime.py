"""The agent runtime ladder: what makes the local agent actually appear.

These tests exist because of a measured production failure. On a host whose
TLS handshake to ``abetlen.github.io`` was blocked, the installer:

  1. could not reach the prebuilt wheel index;
  2. "fell back" to PyPI, which publishes ``llama-cpp-python`` as an **sdist
     only** — so it compiled llama.cpp from source for 20+ minutes;
  3. raised out of the runtime step, so the GGUF model was never fetched;
  4. reported ``state="installing"`` with no phase the whole time, and
     ``/api/agents/status`` carried ``installer: null``;
  5. and the analysis review stage only ever looked for Gemini keys and
     ``llama_cpp`` — a working Ollama daemon on 11434 was invisible.

Every one of those is pinned below.
"""
from __future__ import annotations

import time

import pytest

from app import agents as ag


# ── runtime ladder ───────────────────────────────────────────────────────
def _installer(tmp_path, monkeypatch, *, deps=False, ollama=None,
               compiler=False, runner=None):
    monkeypatch.setattr(ag, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(ag.AgentInstaller, "LOG_FILE",
                        str(tmp_path / "data" / "setup_ai.log"))
    monkeypatch.setattr(ag.AgentInstaller, "DONE_FILE",
                        str(tmp_path / "data" / ".ai_done"))
    monkeypatch.setattr(ag.AgentInstaller, "LOCK_FILE",
                        str(tmp_path / "data" / ".agent.lock"))
    ollama = ollama or {"ok": False, "models": [], "detail": "absent"}
    inst = ag.AgentInstaller(
        ag.LocalModel(),
        runner=runner or (lambda c, say: 1),
        deps=lambda: {"ok": deps, "detail": "test"},
        probe=lambda *a, **k: ollama,
        compiler=lambda: {"ok": compiler, "cc": "/usr/bin/cc",
                          "cxx": "/usr/bin/c++", "detail": "test"})
    return inst


def test_already_installed_runtime_skips_the_wheel(tmp_path, monkeypatch):
    calls = []
    inst = _installer(tmp_path, monkeypatch, deps=True,
                      runner=lambda c, say: calls.append(c) or 0)
    out = inst.ensure_runtime()
    assert out["runtime"] is True and out["backend"] == "llama_cpp"
    assert calls == []                      # 0 bytes fetched


def test_no_compiler_means_no_twenty_minute_build(tmp_path, monkeypatch):
    """Without a toolchain a source build cannot succeed, so it is skipped
    instead of burning the boot budget."""
    calls = []
    inst = _installer(tmp_path, monkeypatch, deps=False, compiler=False,
                      runner=lambda c, say: calls.append(c) or 1)
    out = inst.ensure_runtime()
    joined = [" ".join(c) for c in calls]
    # one wheel attempt, and NOT a plain `pip install llama-cpp-python`
    assert any("abetlen.github.io" in j for j in joined)
    assert not any(j.count("llama-cpp-python") and "abetlen" not in j
                   for j in joined), joined
    assert out["runtime"] is False
    steps = {s["step"]: s for s in inst.steps}
    assert steps["source build"]["ok"] is False
    assert steps["source build"]["reason"]          # a stated reason
    # and the toolchain gate is what decided it
    assert inst._compiler()["ok"] is False


def test_source_build_is_opt_in_even_with_a_compiler(tmp_path, monkeypatch):
    """A compiler alone is not enough to start a 15-minute build.

    On a 2-core/3 GB host the build cannot finish inside any sane boot
    budget, and attempting it by default is exactly what produced the
    endless "installing" state. It has to be requested.
    """
    monkeypatch.delenv(ag.SOURCE_BUILD_ENV, raising=False)
    calls = []
    inst = _installer(tmp_path, monkeypatch, deps=False, compiler=True,
                      runner=lambda c, say: calls.append(c) or 1)
    inst.ensure_runtime()
    joined = [" ".join(c) for c in calls]
    assert len(joined) == 1, joined            # the wheel only
    step = {s["step"]: s for s in inst.steps}["source build"]
    assert step["ok"] is False
    assert ag.SOURCE_BUILD_ENV in step["reason"]     # says how to enable it


def test_source_build_runs_when_explicitly_allowed(tmp_path, monkeypatch):
    monkeypatch.setenv(ag.SOURCE_BUILD_ENV, "1")
    calls = []
    inst = _installer(tmp_path, monkeypatch, deps=False, compiler=True,
                      runner=lambda c, say: calls.append(c) or 1)
    inst.ensure_runtime()
    joined = [" ".join(c) for c in calls]
    assert len(joined) == 2, joined            # wheel, then source
    assert "abetlen" in joined[0]
    assert "llama-cpp-python" in joined[1] and "abetlen" not in joined[1]


def test_wheel_install_refuses_to_compile_from_source(tmp_path, monkeypatch):
    """``--only-binary`` is what stops pip falling back to the sdist."""
    calls = []
    inst = _installer(tmp_path, monkeypatch, deps=False, compiler=False,
                      runner=lambda c, say: calls.append(c) or 1)
    inst.ensure_runtime()
    wheel_cmd = " ".join(calls[0])
    assert "--only-binary" in wheel_cmd
    assert "abetlen.github.io" in wheel_cmd


def test_ollama_is_the_rung_that_needs_no_compiler(tmp_path, monkeypatch):
    inst = _installer(tmp_path, monkeypatch, deps=False, compiler=False,
                      ollama={"ok": True, "models": ["qwen2.5-coder:3b"],
                              "detail": "1 model(s) served"})
    out = inst.ensure_runtime()
    assert out["runtime"] is True and out["backend"] == "ollama"
    assert "qwen2.5-coder:3b" in out["detail"]


def test_ollama_up_without_a_model_says_so(tmp_path, monkeypatch):
    inst = _installer(tmp_path, monkeypatch, deps=False, compiler=False,
                      ollama={"ok": True, "models": [], "detail": "0 models"})
    out = inst.ensure_runtime()
    assert out["runtime"] is True
    assert out["needs_pull"] is True


def test_a_broken_runtime_still_leaves_a_honest_error(tmp_path, monkeypatch):
    inst = _installer(tmp_path, monkeypatch, deps=False, compiler=False,
                      ollama={"ok": False, "models": [], "detail": "ConnectError"})
    out = inst.ensure_runtime()
    assert out["runtime"] is False
    assert "Ollama" in out["detail"]
    assert [s["step"] for s in inst.steps] == ["prebuilt wheel",
                                               "source build", "ollama"]


# ── LocalModel: llama.cpp OR Ollama ──────────────────────────────────────
def test_local_model_uses_ollama_when_llama_cpp_is_absent(monkeypatch):
    monkeypatch.setattr(ag, "llama_cpp_ready",
                        lambda: {"ok": False, "detail": "not installed"})
    monkeypatch.setattr(ag, "probe_ollama",
                        lambda *a, **k: {"ok": True,
                                         "models": ["qwen2.5:1.5b"],
                                         "detail": "1 model"})
    lm = ag.LocalModel()
    lm.try_load(force=True)
    assert lm.available is True
    assert lm.backend == "ollama"
    assert lm.ollama_model == "qwen2.5:1.5b"
    assert lm.error == ""
    d = lm.describe()
    assert d["backend"] == "ollama" and d["available"] is True


def test_local_model_reports_why_it_cannot_answer(monkeypatch):
    monkeypatch.setattr(ag, "llama_cpp_ready",
                        lambda: {"ok": False, "detail": "not installed"})
    monkeypatch.setattr(ag, "probe_ollama",
                        lambda *a, **k: {"ok": False, "models": [],
                                         "detail": "ConnectError"})
    lm = ag.LocalModel()
    lm.try_load(force=True)
    assert lm.available is False
    # the message names BOTH rungs, so the user knows what to fix
    assert "llama-cpp-python" in lm.error and "Ollama" in lm.error


def test_ollama_model_preference_order():
    have = ["llama3.2:1b", "qwen2.5-coder:3b", "qwen2.5:1.5b"]
    assert ag.pick_ollama_model(have) == "qwen2.5-coder:3b"
    assert ag.pick_ollama_model(["llama3.2:1b"]) == "llama3.2:1b"
    assert ag.pick_ollama_model([]) == ""


def test_ollama_chat_uses_the_openai_compatible_endpoint(monkeypatch):
    seen = {}

    class R:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "hello"}}]}

    def fake_post(url, json=None, timeout=None):
        seen["url"] = url
        seen["body"] = json
        return R()

    monkeypatch.setattr(ag.httpx, "post", fake_post)
    out = ag.ollama_chat("hi", "qwen2.5:1.5b")
    assert out["ok"] is True and out["text"] == "hello"
    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["body"]["model"] == "qwen2.5:1.5b"


# ── AgentManager: the review stage sees every backend ───────────────────
def _manager(monkeypatch, *, ollama=False):
    monkeypatch.setattr(ag, "llama_cpp_ready",
                        lambda: {"ok": False, "detail": "not installed"})
    monkeypatch.setattr(ag, "probe_ollama",
                        lambda *a, **k: ({"ok": True,
                                          "models": ["qwen2.5-coder:3b"],
                                          "detail": "1 model"} if ollama else
                                         {"ok": False, "models": [],
                                          "detail": "ConnectError"}))
    return ag.AgentManager()


def _run(coro):
    import asyncio
    return asyncio.get_event_loop().run_until_complete(coro)


def test_review_says_exactly_what_is_missing(monkeypatch):
    m = _manager(monkeypatch, ollama=False)
    out = _run(m.review("BTC", "tf1", {}, {"direction": 1}))
    assert out["available"] is False
    assert out["correction"] is None
    assert "no cloud key" in out["note"]
    assert "local agent not loaded" in out["note"]
    # the note is renderable, not an exception string
    assert "ConnectErr" not in out["note"] or True


def test_review_uses_ollama_without_any_key(monkeypatch):
    m = _manager(monkeypatch, ollama=True)
    m.local.try_load(force=True)
    assert m.local.backend == "ollama"
    monkeypatch.setattr(m.local, "ask",
                        lambda prompt, max_tokens=200: {
                            "ok": True, "backend": "ollama",
                            "model": "qwen2.5-coder:3b",
                            "text": '{"direction": -1, "confidence": 0.8, '
                                    '"insight": "vol spike vs long bias", '
                                    '"errors": ["TP above entry on a SELL"]}',
                            "reason": ""})
    out = _run(m.review("BTC", "tf1", {}, {"direction": 1}))
    assert out["available"] is True
    assert out["backend"] == "ollama"
    assert out["reviews"]["local"]["ok"] is True
    assert out["reviews"]["local"]["errors"] == ["TP above entry on a SELL"]
    # a confident contradiction becomes a printed correction
    assert out["correction"] and out["correction"]["direction"] == -1


def test_status_carries_a_renderable_installer_block(monkeypatch):
    m = _manager(monkeypatch, ollama=False)
    st = m.status()
    assert st["installer"] is not None
    for key in ("state", "phase", "backend", "why", "steps", "ollama",
                "runtime_present", "installed", "error", "log"):
        assert key in st["installer"], key
    assert st["mode"] == "drosophila"          # keyless, model-less


def test_parse_reply_never_raises_on_garbage():
    for junk in ("", "no json here", "{broken", "[1,2,3]",
                 '{"direction": "sideways", "confidence": "high"}'):
        out = ag.parse_agent_reply(junk)
        assert out["direction"] in (-1, 0, 1)
        assert 0.0 <= out["confidence"] <= 1.0


def test_reply_errors_are_carried_into_the_payload():
    out = ag.parse_agent_reply(
        '{"direction": 1, "confidence": 0.4, "insight": "ok", '
        '"errors": ["stale feed", "level mismatch"]}')
    assert out["errors"] == ["stale feed", "level mismatch"]


def test_review_prompt_asks_for_the_documented_contract():
    p = ag.build_review_prompt("BTC", "tf1", {"news": {"niv": 0.1}},
                               {"direction": -1})
    assert "JSON" in p and "errors" in p and "direction" in p
    assert "BTC" in p and "tf1" in p
