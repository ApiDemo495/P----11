"""The download-free routes to a working local agent.

Background: the GGUF route needs huggingface.co reachable, a 2 GB transfer
that survives, and a llama.cpp build for the host's platform. On a network
that blocks the hub — where TLS is closed at CONNECT, so every retry looks
like an endless "downloading…" — none of that is fixable, and the only
advice the UI used to give was "press download again".

Almost every local runner people install speaks the same wire format
(POST /v1/chat/completions, GET /v1/models): Ollama, LM Studio, llama.cpp's
own llama-server, Jan, GPT4All, LocalAI, vLLM, text-generation-webui,
KoboldCpp. So one prober reaches all of them, needs no download, and works
behind the same blocked network. These tests pin that path, the honest
diagnosis when the hub is blocked, and the free-cloud escape hatch.
"""
import json

import pytest

from app import agents as ag


# ── endpoint discovery ───────────────────────────────────────────────────
class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p


def test_a_running_server_is_found_without_any_download(monkeypatch):
    """Discovery must be the FIRST rung: instant and download-free."""
    monkeypatch.delenv(ag.LOCAL_ENDPOINT_ENV, raising=False)
    monkeypatch.setenv(ag.LOCAL_AUTODETECT_ENV, "1")
    calls = []

    def fake_get(url, timeout=None):
        calls.append(url)
        if ":1234" in url:
            return _Resp({"data": [{"id": "qwen2.5-coder:3b"}]})
        raise ConnectionError("refused")

    monkeypatch.setattr(ag._HTTP_LOCAL, "get", fake_get)
    ag.invalidate_endpoint_cache()
    found = ag.discover_local_endpoints(force=True)
    assert found["ok"] is True
    assert found["base"] == "http://127.0.0.1:1234"
    assert found["label"] == "LM Studio"
    assert found["model"] == "qwen2.5-coder:3b"
    # it stopped probing once something answered
    assert len(calls) == 2


def test_no_server_running_names_every_port_it_tried(monkeypatch):
    """'not loaded' is not an answer; the panel needs the port list."""
    monkeypatch.delenv(ag.LOCAL_ENDPOINT_ENV, raising=False)
    monkeypatch.setenv(ag.LOCAL_AUTODETECT_ENV, "1")
    monkeypatch.setattr(ag._HTTP_LOCAL, "get",
                        lambda url, timeout=None: (_ for _ in ()).throw(
                            ConnectionError("refused")))
    ag.invalidate_endpoint_cache()
    found = ag.discover_local_endpoints(force=True)
    assert found["ok"] is False
    assert len(found["tried"]) == len(ag.LOCAL_ENDPOINTS)
    assert all(t["ok"] is False for t in found["tried"])
    assert ag.LOCAL_ENDPOINT_ENV in found["detail"]


def test_an_explicit_endpoint_is_probed_first(monkeypatch):
    monkeypatch.setenv(ag.LOCAL_ENDPOINT_ENV, "192.168.1.9:8080")
    ag.invalidate_endpoint_cache()
    eps = ag.configured_endpoints()
    assert eps[0]["base"] == "http://192.168.1.9:8080"
    assert eps[0]["source"] == "env"


def test_autodetect_can_be_turned_off(monkeypatch):
    monkeypatch.setenv(ag.LOCAL_ENDPOINT_ENV, "http://box:9999")
    monkeypatch.setenv(ag.LOCAL_AUTODETECT_ENV, "0")
    eps = ag.configured_endpoints()
    assert [e["base"] for e in eps] == ["http://box:9999"]


def test_a_refused_sweep_costs_no_time(monkeypatch):
    """A refused localhost port answers in microseconds; a boot path may not
    spend 9 x timeout discovering that nothing is running."""
    monkeypatch.delenv(ag.LOCAL_ENDPOINT_ENV, raising=False)
    monkeypatch.setenv(ag.LOCAL_AUTODETECT_ENV, "1")
    monkeypatch.setattr(ag._HTTP_LOCAL, "get",
                        lambda url, timeout=None: (_ for _ in ()).throw(
                            ConnectionRefusedError("refused")))
    ag.invalidate_endpoint_cache()
    import time
    t0 = time.perf_counter()
    ag.discover_local_endpoints(force=True)
    assert time.perf_counter() - t0 < 2.0


# ── the LocalModel backend ───────────────────────────────────────────────
def test_local_model_uses_a_running_server_before_any_download(monkeypatch):
    monkeypatch.delenv(ag.LOCAL_ENDPOINT_ENV, raising=False)
    monkeypatch.setenv(ag.LOCAL_AUTODETECT_ENV, "1")

    def fake_get(url, timeout=None):
        if ":8080" in url:
            return _Resp({"data": [{"id": "qwen2.5-coder-3b"}]})
        raise ConnectionError("refused")

    monkeypatch.setattr(ag._HTTP_LOCAL, "get", fake_get)
    monkeypatch.setattr(ag, "llama_cpp_ready",
                        lambda: {"ok": False, "detail": "absent"})
    ag.invalidate_endpoint_cache()
    lm = ag.LocalModel()
    lm.try_load(force=True)
    assert lm.available is True
    assert lm.backend == "openai"
    assert lm.endpoint == "http://127.0.0.1:8080"
    assert "llama.cpp server" in lm.label


def test_a_runaway_download_error_says_the_hub_is_blocked(monkeypatch):
    """URLError wraps the SSL close; the hint must still be the useful one."""
    import urllib.error

    def boom(req, timeout=None):
        raise urllib.error.URLError(
            ConnectionResetError("TLS/SSL connection has been closed (EOF)"))

    monkeypatch.setattr("urllib.request.urlopen", boom)
    monkeypatch.setattr(ag, "resolve_hf_file", lambda r, f: f)
    res = ag.download_gguf("Qwen/x-GGUF", "x.gguf", "/tmp/none.gguf", timeout=1)
    assert res["ok"] is False
    assert res["blocked"] is True
    assert "huggingface.co is not reachable" in res["hint"]
    # and it must point at the routes that DO work
    assert ag.LOCAL_ENDPOINT_ENV in res["hint"]


def test_a_plain_network_flake_is_not_called_a_block(monkeypatch):
    import urllib.error

    def boom(req, timeout=None):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr("urllib.request.urlopen", boom)
    monkeypatch.setattr(ag, "resolve_hf_file", lambda r, f: f)
    res = ag.download_gguf("Qwen/x-GGUF", "x.gguf", "/tmp/none.gguf", timeout=1)
    assert res["ok"] is False
    assert res["blocked"] is False
    assert "huggingface.co is not reachable" not in res["hint"]


# ── the free-cloud escape hatch ──────────────────────────────────────────
@pytest.mark.parametrize("setting,want", [
    ("", "https://api.openai.com/v1/chat/completions"),
    (None, "https://api.openai.com/v1/chat/completions"),
    ("openai", "https://api.openai.com/v1/chat/completions"),
    ("groq", "https://api.groq.com/openai/v1/chat/completions"),
    ("deepseek", "https://api.deepseek.com/v1/chat/completions"),
    ("openrouter", "https://openrouter.ai/api/v1/chat/completions"),
    ("http://127.0.0.1:1234", "http://127.0.0.1:1234/v1/chat/completions"),
    ("http://x/v1", "http://x/v1/chat/completions"),
    ("http://x/v1/chat/completions", "http://x/v1/chat/completions"),
])
def test_openai_slot_can_be_pointed_anywhere(setting, want):
    assert ag.openai_base_url(setting) == want


def test_an_unknown_nickname_falls_back_to_openai_rather_than_a_dead_url():
    assert ag.openai_base_url("nonsense") == ag.FREE_OPENAI_BASES["openai"]
