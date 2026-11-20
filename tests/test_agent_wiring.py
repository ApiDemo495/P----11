"""Agent wiring regressions: keys that survive a restart, current models,
and a downloader that needs nothing beyond the standard library.
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app import agents as ag


def test_stored_key_survives_reload_without_entering_tracked_settings(
        tmp_path, monkeypatch):
    monkeypatch.setattr(ag, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    monkeypatch.setattr(ag, "SECRETS_FILE", str(tmp_path / "secrets.json"))
    monkeypatch.delenv("PREDICT_NO_SECRET_FILE", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    m = ag.AgentManager()
    m.configure(gemini_key="  AIza-test-key  ")
    assert m.settings["gemini_key"] == "AIza-test-key"
    settings_txt = (tmp_path / "settings.json").read_text()
    assert "AIza-test-key" not in settings_txt      # I-5 still honoured
    secrets_txt = (tmp_path / "secrets.json").read_text()
    assert "AIza-test-key" in secrets_txt
    assert oct(os.stat(tmp_path / "secrets.json").st_mode & 0o777) == "0o600"
    m2 = ag.AgentManager()
    assert m2.settings["gemini_key"] == "AIza-test-key"


def test_blank_key_keeps_the_stored_one_and_alias_resolves(tmp_path,
                                                           monkeypatch):
    monkeypatch.setattr(ag, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    monkeypatch.setattr(ag, "SECRETS_FILE", str(tmp_path / "secrets.json"))
    monkeypatch.delenv("PREDICT_NO_SECRET_FILE", raising=False)
    m = ag.AgentManager()
    m.configure(openai_key="sk-openai")
    assert m.settings["backup2_key"] == "sk-openai"
    m.configure(backup2_key="")
    assert m.settings["backup2_key"] == "sk-openai"


def test_choose_model_prefers_a_current_flash_and_returns_a_bare_id():
    models = [
        {"name": "models/gemini-1.5-flash",
         "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-2.5-pro",
         "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-2.5-flash",
         "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3.5-flash",
         "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/text-embedding-004",
         "supportedGenerationMethods": ["embedContent"]},
    ]
    chosen = ag.AgentManager.choose_model(models)
    assert chosen == "gemini-2.5-flash"      # GA workhorse, no "models/" prefix


def test_gemini_default_is_a_live_model_not_the_retired_1_5():
    assert ag.GEMINI_DEFAULT_MODEL.startswith("gemini-2.5")
    assert "1.5" not in ag.GEMINI_URL


@pytest.fixture()
def hf_server():
    payload = b"GGUF" + b"\x00" * 4096
    state = {"served": 0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/api/models/"):
                body = json.dumps({"siblings": [
                    {"rfilename": "model-q4_k_m.gguf"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if "/resolve/main/" in self.path:
                if "missing" in self.path:
                    self.send_error(404)
                    return
                rng = self.headers.get("Range")
                if rng:
                    start = int(rng.split("=")[1].split("-")[0])
                    chunk = payload[start:]
                else:
                    chunk = payload
                state["served"] += 1
                self.send_response(206 if rng else 200)
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
                return
            self.send_error(404)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        t.join(timeout=5)


def test_stdlib_downloader_resolves_the_real_filename_and_resumes(
        tmp_path, monkeypatch, hf_server):
    monkeypatch.setenv("PREDICT_HF_BASE", hf_server)
    dest = str(tmp_path / "model.gguf")
    # a partial file from an earlier interrupted run is resumed, not discarded
    with open(dest + ".part", "wb") as fh:
        fh.write(b"GGUF")
    done = []
    res = ag.download_gguf("org/repo", "renamed-q4_k_m.gguf", dest,
                           progress=lambda d, t: done.append((d, t)))
    assert res["ok"], res
    assert os.path.getsize(dest) == 4100
    assert open(dest, "rb").read(4) == b"GGUF"
    assert os.path.exists(dest + ".sha256.json")   # sidecar recorded
    assert done


def test_stdlib_downloader_rejects_an_error_page(tmp_path, monkeypatch,
                                                 hf_server):
    monkeypatch.setenv("PREDICT_HF_BASE", hf_server)
    # the server only knows model-q4_k_m.gguf; asking for a missing id still
    # resolves, so point the resolver at a 404 route instead
    monkeypatch.setattr(ag, "resolve_hf_file", lambda repo, fname: "missing.gguf")
    dest = str(tmp_path / "bad.gguf")
    res = ag.download_gguf("org/repo", "missing.gguf", dest)
    assert res["ok"] is False
    assert res["reason"]
