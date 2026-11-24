"""Section 11 — agent integration, rebuilt around a runtime that actually runs.

Design rules (each one is a bug this file used to have)
------------------------------------------------------
1. **The app never depends on an agent.** With no key, no compiler and no
   model the engine trades on formulas + brains + physics alone. Every agent
   path is optional and reports its own state.

2. **The local runtime is a *ladder*, never a single source.**
   ``llama_cpp`` → prebuilt CPU wheel → reachable source build → **Ollama**.
   The previous version tried one GitHub-Pages wheel index, and when that
   host's TLS handshake was blocked it fell back to PyPI. PyPI publishes
   ``llama-cpp-python`` as an **sdist only** (checked against
   pyproject/pypi JSON), so that "fallback" always compiled llama.cpp for
   twenty minutes and never finished. Ollama is the rung that needs neither a
   compiler nor that wheel index — it only needs a binary or a running
   daemon, and it serves an OpenAI-compatible HTTP API.

3. **The model download is independent of the runtime install.** A failed
   runtime used to raise before the model was ever fetched, so fixing the
   runtime later meant downloading everything again. Now the GGUF lands
   regardless, and the load step reports what it still needs.

4. **Status is never a bare "installing".** Every state carries a ``phase``,
   a ``backend`` that won, a ``why`` for what did not, and a log tail. A
   download in progress reports elapsed seconds and bytes on disk.

5. **Every provider shares one path** (``AgentRuntime``) for timeouts,
   cooldowns, failover and realized-accuracy scoring, so a local model, an
   Ollama daemon and a cloud key are interchangeable in the panel and in
   ``_analyse_ai_review``.

6. **Nothing blocks the event loop.** Installer and autoload run on daemon
   threads guarded by an flock; inference is pushed to a worker.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import Any

import httpx

from .config import (
    AGENT_COMBINED_WEIGHT,
    AGENT_EPS,
    AGENT_FALLBACK_GRCS,
    AGENT_FREE_RPH,
    AGENT_GEMINI_WEIGHT,
    AGENT_LOCAL_WEIGHT,
    AGENT_PAID_PER_GRC,
    SYMBOL_ORDER,
    TF_ORDER,
)

log = logging.getLogger("agents")

# Quiet the hub's retry spam: agent models are optional and a blocked
# huggingface.co must never look like an application error.
for _lg in ("huggingface_hub", "huggingface_hub.utils._http"):
    logging.getLogger(_lg).setLevel(logging.ERROR)

SETTINGS_FILE = "data/settings.json"
#: local secret store (outside git, chmod 600). The old contract said
#: "stored locally in data/settings.json" in the UI while save_settings
#: stripped every key — so a pasted Gemini key vanished on the next restart.
SECRETS_FILE = os.environ.get("PREDICT_SECRETS_FILE", "data/secrets.json")
KEY_FIELDS = frozenset({"gemini_key", "backup1_key", "backup2_key",
                        "backup3_key", "groq_key", "openai_key",
                        "anthropic_key", "newsapi_key", "neuprint_key"})
#: accepted spellings from older clients / packs
KEY_ALIASES = {"openai_key": "backup2_key", "anthropic_key": "backup3_key",
               "gemini-backup_key": "backup1_key", "groq_api_key": "groq_key"}
MODELS_DIR = "models"
#: the model Gemini free keys can actually call today (1.5 was retired and
#: the hard-coded id made a valid key answer HTTP 404)
GEMINI_DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_MODEL_ORDER = ("gemini-2.5-flash", "gemini-2.5-flash-lite",
                      "gemini-2.0-flash", "gemini-2.0-flash-lite",
                      "gemini-flash-latest")

# ── local runtime endpoints ─────────────────────────────────────────────
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.environ.get("OLLAMA_PORT", "11434"))
OLLAMA_TIMEOUT_S = 2.0
#: how long a source build of llama.cpp may run before it is abandoned.
#: A 20-minute silent compile is what made the old installer look dead.
SOURCE_BUILD_BUDGET_S = 900.0
#: Source builds are OPT-IN. On a 2-core/3 GB Codespace-class host a full
#: llama.cpp build does not finish inside any reasonable boot budget, so
#: attempting it by default is what produced the endless "installing" state.
#: Set PREDICT_AGENT_SOURCE_BUILD=1 to allow it (with the budget above).
SOURCE_BUILD_ENV = "PREDICT_AGENT_SOURCE_BUILD"
#: prebuilt CPU wheel index (correct name is whl/cpu, not whl/cpu-only)
LLAMA_WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"


# ══════════════════════════════════════════════════════════════════════
# model tiers
# ══════════════════════════════════════════════════════════════════════
LOCAL_MODELS = [
    # (min_ram_gb, hf repo, filename, label) — RAM tiers. The top tier is
    # the preferred Qwen2.5-Coder-3B (also the formula auditor's model);
    # the floor tier still fits a 4 GB Codespace.
    (5.0, "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
     "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
     "Qwen2.5-Coder-3B-Instruct (Q4_K_M)"),
    (0.0, "Qwen/Qwen2-0.5B-Instruct-GGUF", "qwen2-0_5b-instruct-q4_k_m.gguf",
     "Qwen2-0.5B-Instruct (Q4_K_M)"),
]

MODEL_SIZE_HINT = {
    "qwen2.5-coder-3b-instruct-q4_k_m.gguf": "~2.0 GB",
    "qwen2-0_5b-instruct-q4_k_m.gguf": "~400 MB",
}

#: Ollama tags to prefer for the local agent (first one present wins).
OLLAMA_TAGS = ("qwen2.5-coder:3b", "qwen2.5-coder:1.5b", "qwen2.5:3b",
               "qwen2.5:1.5b", "llama3.2:3b", "llama3.2:1b")


def pick_local_model(ram_gb: float):
    """(repo, filename, label) for the RAM tier — one source of truth."""
    for min_ram, repo, fname, label in LOCAL_MODELS:
        if ram_gb >= min_ram:
            return repo, fname, label
    last = LOCAL_MODELS[-1]
    return last[1], last[2], last[3]


def resolve_model_path(local: "LocalModel | None" = None) -> str:
    """THE one place the preferred model becomes a path (installer, loader
    and UI all agree through here)."""
    ram = local.ram_gb if local is not None else 8.0
    _, fname, _ = pick_local_model(ram)
    return os.path.join(MODELS_DIR, fname)


# ══════════════════════════════════════════════════════════════════════
# GGUF validation (local only, never a network call)
# ══════════════════════════════════════════════════════════════════════
def model_sidecar(path: str) -> str:
    return path + ".sha256.json"


def write_model_sidecar(path: str) -> dict:
    """Hash a downloaded GGUF ONCE and record {sha256,size,mtime} beside it,
    so later starts validate locally without re-hashing 2 GB."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    st = os.stat(path)
    rec = {"sha256": h.hexdigest(), "size": st.st_size,
           "mtime": int(st.st_mtime)}
    try:
        with open(model_sidecar(path), "w") as fh:
            json.dump(rec, fh)
    except OSError:
        pass
    return rec


# paths currently being hashed by the TOFU recorder (issue 2: a model
# downloaded by an older build or a failed sidecar write stayed
# "unverified (no checksum record)" forever)
_sidecar_writing: set[str] = set()


def record_sidecar_async(path: str) -> bool:
    """Trust-on-first-use: when the GGUF magic is good but no checksum
    record exists, hash it ONCE on a daemon thread and write the sidecar.
    Returns True if a writer was started (or is already running)."""
    if not path or not os.path.exists(path):
        return False
    if path in _sidecar_writing:
        return True
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != b"GGUF":
                return False
        if os.path.exists(model_sidecar(path)):
            return False
    except OSError:
        return False
    _sidecar_writing.add(path)

    def _work() -> None:
        try:
            write_model_sidecar(path)
        except Exception:
            pass
        finally:
            _sidecar_writing.discard(path)

    threading.Thread(target=_work, name="sidecar-tofu", daemon=True).start()
    return True


def model_validation(path: str) -> dict:
    """Local-only validation: exists, GGUF magic at offset 0, recorded sha
    still matches size+mtime. A half-finished download fails HERE and is
    re-fetched instead of failing mysteriously at load time."""
    out = {"present": False, "magic_ok": False, "bytes": 0,
           "sha": "absent", "source": "absent"}
    if not os.path.exists(path):
        return out
    out["present"] = True
    out["bytes"] = os.path.getsize(path)
    try:
        with open(path, "rb") as fh:
            out["magic_ok"] = fh.read(4) == b"GGUF"
    except OSError:
        pass
    if not out["magic_ok"]:
        out["sha"] = "corrupt — not a GGUF file"
        return out
    out["source"] = "on disk — 0 bytes fetched"
    try:
        with open(model_sidecar(path)) as fh:
            rec = json.load(fh)
        st = os.stat(path)
        out["sha"] = ("ok" if rec.get("size") == st.st_size
                      and rec.get("mtime") == int(st.st_mtime)
                      else "unverified (file changed since download)")
    except (OSError, ValueError):
        # issue 2: a good GGUF with no sidecar used to stay "unverified"
        # forever — record the checksum now (background hash of the file)
        if record_sidecar_async(path):
            out["sha"] = "verifying (recording checksum now)"
        else:
            out["sha"] = "unverified (no checksum record)"
    return out


def _ram_gb() -> float:
    """Usable RAM in GB (MemTotal, falling back to `free`)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except (OSError, ValueError):
        pass
    try:
        out = subprocess.run(["free", "-m"], capture_output=True, text=True,
                             timeout=3).stdout
        for line in out.splitlines():
            if line.lower().startswith("mem:"):
                return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 4.0


def sys_executable() -> str:
    import sys
    return sys.executable


def _mask(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "•" * len(key)
    return key[:4] + "•" * (len(key) - 8) + key[-4:]


# ══════════════════════════════════════════════════════════════════════
# prompt / reply contract
# ══════════════════════════════════════════════════════════════════════
REPLY_CONTRACT = (
    'Reply with ONLY a JSON object: {"direction": 1 or -1 or 0 '
    '(confirm/contradict/uncertain), "confidence": 0..1, '
    '"insight": "one short sentence"}.')


def build_prompt(ctx: dict) -> str:
    """The structured GRC prompt (same shape for every provider)."""
    lines = [
        "You are a crypto market confirmation agent. Analyze the state below.",
        f"Prices: {json.dumps(ctx.get('prices', {}))}",
        f"10-minute returns: {json.dumps(ctx.get('returns', {}))}",
        f"CCI prediction: direction={ctx.get('cci_dir')} "
        f"confidence={ctx.get('cci_conf')}",
        f"Top-5 formula signals by weight: {json.dumps(ctx.get('top5', []))}",
        f"Drosophila brain consensus: {ctx.get('brain_consensus')}",
        f"Latest news: {ctx.get('headline')} "
        f"(sentiment={ctx.get('news_sentiment')})",
        f"Emotion composite (CERC): {ctx.get('cerc')}",
        REPLY_CONTRACT,
    ]
    return "\n".join(lines)


def build_review_prompt(symbol: str, tf: str, store: dict,
                        result: dict | None = None) -> str:
    """Prompt for the ANALYSIS REVIEW stage (stage 6 of ▶ ANALYSE).

    It asks for the same JSON contract as the GRC prompt plus an optional
    ``errors`` list, so a reviewer can both vote and flag arithmetic it
    believes is wrong.
    """
    compact = {
        "symbol": symbol,
        "timeframe": tf,
        "signal": (result or {}).get("signal"),
        "confidence": (result or {}).get("confidence"),
        "emotions": {k: (v or {}).get("cerc")
                     for k, v in (store.get("emotions") or {}).items()},
        "physics": {k: (v or {}).get("ufp")
                    for k, v in (store.get("physics") or {}).items()},
        "news": (store.get("news") or {}).get("niv"),
        "hedge": {k: (v or {}).get("combined")
                  for k, v in (store.get("hedge") or {}).items()},
    }
    return (
        "You are the review agent of a crypto prediction engine. Read the "
        "assembled analysis below and answer with JSON only: "
        '{"direction": 1 or -1 or 0, "confidence": 0..1, '
        '"insight": "one short sentence", "errors": ["..."]} — '
        "errors lists concrete arithmetic or data contradictions you can "
        "see, or an empty list.\nAnalysis: "
        + json.dumps(compact, separators=(",", ":"))[:1600])


def parse_agent_reply(text: str) -> dict:
    """Pull the direction/confidence/insight triple (and errors) out of a
    model reply. Never raises; unparseable text degrades to a heuristic."""
    out: dict[str, Any] = {"direction": 0, "confidence": 0.0, "insight": "",
                           "errors": []}
    if not text:
        return out
    m = re.search(r"\{.*\}", text, re.DOTALL)
    blob = m.group(0) if m else text
    try:
        cleaned = (blob.replace("−", "-").replace("–", "-")
                   .replace("’", "'").replace("'", '"'))
        d = json.loads(cleaned)
        if not isinstance(d, dict):
            # a bare list/scalar is not a reply — fall through to the
            # heuristic instead of calling .get() on a list
            raise TypeError("reply is not an object")
        try:
            direction = float(d.get("direction", 0) or 0)
        except (TypeError, ValueError):
            direction = 0.0
        out["direction"] = 1 if direction > 0 else (-1 if direction < 0 else 0)
        try:
            out["confidence"] = max(0.0, min(1.0,
                                             float(d.get("confidence", 0.0))))
        except (TypeError, ValueError):
            out["confidence"] = 0.0
        out["insight"] = str(d.get("insight", ""))[:300]
        errs = d.get("errors")
        if isinstance(errs, list):
            out["errors"] = [str(x)[:200] for x in errs][:6]
        return out
    except (ValueError, TypeError):
        pass
    # heuristic fallback: confirm/contradict words + a 0..1 number
    low = text.lower()
    if any(w in low for w in ("confirm", "agree", "bullish")):
        out["direction"] = 1
    elif any(w in low for w in ("contradict", "disagree", "bearish")):
        out["direction"] = -1
    nums = re.findall(r"0?\.\d+|\b\d{1,3}\b", text)
    if nums:
        try:
            v = float(nums[0])
            out["confidence"] = min(1.0, v / 100.0 if v > 1 else v)
        except ValueError:
            pass
    out["insight"] = text.strip()[:300]
    return out


# ══════════════════════════════════════════════════════════════════════
# Ollama (the rung that needs no compiler)
# ══════════════════════════════════════════════════════════════════════
# One client, built once. Constructing ``httpx.get`` per probe forced a fresh
# ``ssl.create_default_context()`` each time — ``load_verify_locations`` walked
# the whole CA bundle (~24 ms) on EVERY probe. ``status()`` probes 16 times and
# ``build_snapshot()`` calls ``status()`` twice, so one snapshot build spent
# 1.15 s doing nothing but re-reading certificates. That CPU sat on the same
# box as the GRC pass and /api/health, which is what produced the "ENGINE BUSY"
# banner and the false offline chip. The client is shared; the probe is cached.
_HTTP_LOCAL = httpx.Client(timeout=OLLAMA_TIMEOUT_S, trust_env=False,
                           limits=httpx.Limits(max_keepalive_connections=2,
                                               max_connections=4))
_PROBE_CACHE: dict[str, Any] = {}
_PROBE_TTL_S = 5.0


def _probe_uncached(host: str, port: int, timeout: float) -> dict:
    url = f"http://{host}:{port}/api/tags"
    out: dict[str, Any] = {"url": url, "ok": False, "models": [],
                           "detail": "", "ms": None, "running": False}
    t0 = time.perf_counter()
    try:
        r = _HTTP_LOCAL.get(url, timeout=timeout)
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        out["ok"] = bool(r.status_code == 200)
        if out["ok"]:
            try:
                out["models"] = [m.get("name") for m in
                                 (r.json().get("models") or [])][:20]
            except ValueError:
                out["models"] = []
            out["detail"] = f"{len(out['models'])} model(s) served"
            out["running"] = True
        else:
            out["detail"] = f"HTTP {r.status_code}"
    except Exception as exc:                    # refused == not running
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        out["detail"] = type(exc).__name__
    return out


def probe_ollama(host: str = OLLAMA_HOST, port: int = OLLAMA_PORT,
                 timeout: float = OLLAMA_TIMEOUT_S,
                 max_age: float = _PROBE_TTL_S) -> dict:
    """Liveness probe for the local LLM daemon, memoised for ``max_age`` s.

    The engine asks "is Ollama up?" from several places per build. Without
    the memo each ask opened a socket to a port that is almost never bound,
    so the answer cost a TLS/connection attempt every time. Pass
    ``max_age=0`` to force a real probe (the installer and the Settings
    "reconnect" button do exactly that).
    """
    key = f"{host}:{port}"
    if max_age > 0:
        hit = _PROBE_CACHE.get(key)
        if hit and time.time() - hit[0] < max_age:
            out = dict(hit[1])
            out["cached"] = True
            out["age_s"] = round(time.time() - hit[0], 1)
            return out
    out = _probe_uncached(host, port, timeout)
    out["cached"] = False
    _PROBE_CACHE[key] = (time.time(), dict(out))
    return out


def invalidate_probe() -> None:
    """Drop the memo — call after starting/stopping the daemon."""
    _PROBE_CACHE.clear()


def ensure_ollama_running(timeout: float = OLLAMA_TIMEOUT_S) -> dict:
    """Probe the daemon; start `ollama serve` detached if the binary exists.

    Returns the probe result plus ``started``/``binary`` keys. Never raises
    and never waits on the daemon beyond ``timeout``.
    """
    probe = probe_ollama(timeout=timeout)
    if probe["ok"]:
        probe["started"] = False
        probe["binary"] = None
        return probe
    binp = shutil.which("ollama") or shutil.which("/usr/local/bin/ollama")
    probe["binary"] = binp
    if not binp:
        probe["started"] = False
        probe["detail"] = (probe.get("detail") or "") + " · no ollama binary"
        return probe
    try:
        subprocess.Popen([binp, "serve"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                         start_new_session=True, close_fds=True)
        probe["started"] = True
        probe["detail"] = "ollama serve spawned (detached)"
    except Exception as exc:
        probe["started"] = False
        probe["detail"] = f"spawn failed: {exc}"
    return probe


def pick_ollama_model(models: list[str], preferred: str = "") -> str:
    """First available tag, preferring ``preferred`` then OLLAMA_TAGS."""
    have = [str(m) for m in (models or []) if m]
    if preferred:
        for m in have:
            if m == preferred or m.split(":")[0] == preferred.split(":")[0]:
                return m
    for want in OLLAMA_TAGS:
        for m in have:
            if m == want or m.split(":")[0] == want.split(":")[0]:
                return m
    return have[0] if have else ""


def ollama_chat(prompt: str, model: str = "", *,
                host: str = OLLAMA_HOST, port: int = OLLAMA_PORT,
                timeout: float = 60.0, max_tokens: int = 200) -> dict:
    """POST an OpenAI-compatible chat completion to Ollama.

    Ollama serves ``/v1/chat/completions`` with the same schema as OpenAI,
    so one code path covers the daemon and (later) any compatible host.
    """
    if not model:
        probe = probe_ollama(host=host, port=port)
        model = pick_ollama_model(probe.get("models") or [])
    if not model:
        return {"ok": False, "reason": "no ollama model pulled",
                "detail": "run: ollama pull " + OLLAMA_TAGS[0]}
    url = f"http://{host}:{port}/v1/chat/completions"
    body = {"model": model, "temperature": 0.2,
            "max_tokens": int(max_tokens),
            "messages": [{"role": "user", "content": prompt}]}
    try:
        r = httpx.post(url, json=body, timeout=timeout)
    except Exception as exc:
        return {"ok": False, "reason": type(exc).__name__, "model": model}
    if r.status_code >= 400:
        return {"ok": False, "reason": f"http {r.status_code}",
                "model": model, "detail": r.text[:200]}
    try:
        text = r.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        return {"ok": False, "reason": "unparseable response", "model": model}
    return {"ok": True, "text": str(text), "model": model}


# ══════════════════════════════════════════════════════════════════════
# runtime readiness probes
# ══════════════════════════════════════════════════════════════════════
def llama_cpp_ready() -> dict:
    """Is llama-cpp-python importable, and which version?"""
    try:
        import importlib.util
        if importlib.util.find_spec("llama_cpp") is None:
            return {"ok": False, "detail": "not installed"}
        from llama_cpp import __version__ as v  # type: ignore
        return {"ok": True, "version": str(v), "detail": f"llama_cpp {v}"}
    except Exception as exc:                    # pragma: no cover - optional
        return {"ok": False, "detail": type(exc).__name__}


def source_build_allowed() -> bool:
    """Is a from-source llama.cpp build permitted on this host?

    Opt-in by environment variable *and* a real toolchain. Both conditions
    print into the ladder's step record so the panel can say which one
    failed instead of showing a silent stub.
    """
    return os.environ.get(SOURCE_BUILD_ENV) == "1"


def compiler_ready() -> dict:
    """A source build needs a C/C++ toolchain; say so instead of compiling
    for twenty minutes and failing at the link step."""
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    cxx = (shutil.which("c++") or shutil.which("g++") or shutil.which("clang++"))
    return {"ok": bool(cc and cxx), "cc": cc, "cxx": cxx,
            "detail": ("toolchain present" if (cc and cxx)
                       else "no C/C++ compiler on this host")}


def ollama_pull(model: str = "", *, timeout: float = 1800.0,
                progress: "callable | None" = None) -> dict:
    """Pull a model through the daemon's HTTP API (no CLI needed).

    The old installer only *printed* `ollama pull …` and called it "a manual
    step", which is exactly why the agent never arrived on a host where the
    daemon was already running. The daemon can fetch its own model; this
    drives that endpoint and streams progress.
    """
    model = model or OLLAMA_TAGS[0]
    try:
        with httpx.stream("POST", f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/pull",
                          json={"model": model, "stream": True},
                          timeout=httpx.Timeout(timeout, connect=5.0)) as r:
            if r.status_code >= 400:
                return {"ok": False, "reason": f"http {r.status_code}",
                        "model": model}
            last = ""
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                last = str(obj.get("status") or last)
                if progress:
                    progress(last)
    except Exception as exc:
        return {"ok": False, "reason": type(exc).__name__ + ": " + str(exc)[:200],
                "model": model}
    probe = probe_ollama(max_age=0)
    have = pick_ollama_model(probe.get("models") or [], preferred=model)
    return {"ok": bool(probe.get("ok") and have), "model": have or model,
            "last": last, "reason": "" if have else "daemon did not list it"}


# ══════════════════════════════════════════════════════════════════════
# stdlib GGUF downloader (works without huggingface_hub / pip)
# ══════════════════════════════════════════════════════════════════════
def _hf_base() -> str:
    return (os.environ.get("PREDICT_HF_BASE") or "https://huggingface.co"
            ).rstrip("/")


def _http_json(url: str, timeout: float = 15.0) -> dict:
    import urllib.request
    req = urllib.request.Request(url, headers={
        "User-Agent": "predict-agent/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def list_hf_gguf(repo: str) -> list[str]:
    """List the .gguf files in a HF repo using only the standard library."""
    base = _hf_base()
    try:
        data = _http_json(f"{base}/api/models/{repo}")
        files = [str(s.get("rfilename")) for s in
                 (data.get("siblings") or []) if s.get("rfilename")]
    except Exception:
        files = []
    if not files:
        try:
            tree = _http_json(f"{base}/api/models/{repo}/tree/main")
            files = [str(s.get("path")) for s in tree if s.get("path")]
        except Exception:
            files = []
    return [f for f in files if f.lower().endswith(".gguf")]


def resolve_hf_file(repo: str, fname: str) -> str:
    """Pick the real quantisation file (repos rename them constantly)."""
    files = list_hf_gguf(repo)
    if not files:
        return fname                    # let the download URL say why
    if fname in files:
        return fname
    tag = fname.lower().split(".")[0].rsplit("-", 1)[-1]
    preferred = [f for f in files if tag and tag in f.lower()]
    if preferred:
        return sorted(preferred, key=len)[0]
    small = [f for f in files if "q4" in f.lower()]
    return sorted(small or files, key=len)[0]


def download_gguf(repo: str, fname: str, dest: str, *,
                  progress: "callable | None" = None,
                  timeout: float = 60.0) -> dict:
    """Stream a GGUF from HF (or a mirror) to `dest`, resuming if possible.

    Returns ``{ok, path, bytes, total, reason}``. A truncated/invalid file is
    removed and reported, never left to fail mysteriously at load time.
    """
    import urllib.error
    import urllib.request
    base = _hf_base()
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    total = 0
    try:
        real = resolve_hf_file(repo, fname)
    except Exception:
        real = fname
    url = f"{base}/{repo}/resolve/main/{real}?download=true"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"User-Agent": "predict-agent/1.0"}
    if have:
        headers["Range"] = f"bytes={have}-"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status = getattr(r, "status", 200)
            if have and status != 206:      # server ignored Range
                have = 0
                mode = "wb"
            else:
                mode = "ab" if have else "wb"
            length = r.headers.get("Content-Length")
            total = (int(length) + have) if length and length.isdigit() else 0
            with open(part, mode) as fh:
                done = have
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
                    if total and done >= total:
                        break
    except urllib.error.HTTPError as exc:
        return {"ok": False, "reason": f"HTTP {exc.code} for {url}",
                "bytes": have, "total": total, "path": dest}
    except Exception as exc:
        return {"ok": False, "reason": type(exc).__name__ + ": " + str(exc)[:200],
                "bytes": have, "total": total, "path": dest}
    size = os.path.getsize(part) if os.path.exists(part) else 0
    try:
        with open(part, "rb") as fh:
            magic = fh.read(4)
    except OSError:
        magic = b""
    if magic != b"GGUF":
        try:
            os.remove(part)
        except OSError:
            pass
        return {"ok": False, "bytes": size, "total": total, "path": dest,
                "reason": (f"downloaded {size} bytes but the file is not a "
                           f"GGUF — the URL answered an error page or the "
                           f"transfer was cut")}
    if total and size < total:
        return {"ok": False, "bytes": size, "total": total, "path": dest,
                "reason": f"truncated transfer ({size}/{total} bytes)"}
    try:
        os.replace(part, dest)
        write_model_sidecar(dest)
    except OSError as exc:
        return {"ok": False, "bytes": size, "total": total, "path": dest,
                "reason": "could not finalise the file: " + str(exc)}
    return {"ok": True, "path": dest, "bytes": size, "total": total or size,
            "reason": ""}


# ══════════════════════════════════════════════════════════════════════
# LocalModel — one object the engine asks "can you answer?"
# ══════════════════════════════════════════════════════════════════════
class LocalModel:
    """The local agent: llama.cpp GGUF first, Ollama second.

    ``available`` is True when EITHER backend can answer. ``backend`` names
    the one in use so the UI can print it, and ``error`` always explains why
    the local slot is not live yet.
    """

    #: how often a failed load may be retried (seconds)
    RETRY_S = 180.0

    def __init__(self) -> None:
        self.available = False
        self.backend = ""            # "llama_cpp" | "ollama" | ""
        self.label = "not loaded"
        self.llm = None              # llama_cpp.Llama instance
        self.ollama_model = ""       # chosen ollama tag
        self.path = None
        self.ram_gb = _ram_gb()
        self.download_progress = 0.0
        self.last_ms = None
        self.error = ""
        self._last_attempt = 0.0
        self._net_fails = 0
        self._lock = threading.Lock()

    # -- capability -----------------------------------------------------
    @property
    def ready(self) -> bool:
        return bool(self.available)

    def describe(self) -> dict:
        return {
            "available": self.available,
            "backend": self.backend,
            "label": self.label,
            "model": self.ollama_model or (
                os.path.basename(self.path) if self.path else ""),
            "ram_gb": round(self.ram_gb, 1),
            "error": self.error,
            "last_ms": self.last_ms,
            "download_progress": self.download_progress,
        }

    # -- loading --------------------------------------------------------
    def try_load(self, force: bool = False) -> None:
        """Try llama.cpp, then Ollama. Idempotent and rate-limited."""
        if self.available:
            return
        now = time.time()
        if force:
            self._last_attempt = 0.0
            self._net_fails = 0
        if not force and self._last_attempt and now - self._last_attempt < self.RETRY_S:
            return
        if self._net_fails >= 2 and not force and now - self._last_attempt < 3600:
            return
        self._last_attempt = now
        self.error = ""
        if self._try_llama_cpp():
            return
        if self._try_ollama():
            return
        # neither: keep the most useful explanation
        self.label = "unavailable"

    def _try_llama_cpp(self) -> bool:
        probe = llama_cpp_ready()
        if not probe["ok"]:
            self.error = (f"llama-cpp-python not installed ({probe['detail']}); "
                          "checking Ollama next")
            return False
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as exc:            # pragma: no cover - optional
            self.error = f"llama_cpp import failed: {type(exc).__name__}"
            return False
        repo, fname, label = pick_local_model(self.ram_gb)
        self.label = label
        path = resolve_model_path(self)
        if os.path.exists(path) and not model_validation(path)["magic_ok"]:
            self.error = ("downloaded model file is corrupt (truncated?) — "
                          "re-download it from the Agents tab")
            return False
        if not os.path.exists(path):
            # "press download" was the whole message, so a user whose hub is
            # blocked pressed it forever. Name the step AND the likely cause.
            self.error = (f"no model file at {path}: only the download step "
                          "fetches it, and it fails if huggingface.co (or its "
                          "CDN) is unreachable from this host")
            return False
        try:
            # leave one core for the event loop: n_threads=all cores made
            # every inference starve the loop behind the ENGINE BUSY banner
            _cores = max(1, (os.cpu_count() or 4) - 1)
            self.llm = Llama(model_path=path, n_ctx=2048, n_threads=_cores,
                             verbose=False)
            self.path = path
            self.available = True
            self.backend = "llama_cpp"
            self.label = f"{label} (llama.cpp)"
            self.error = ""
            return True
        except Exception as exc:
            self.error = f"model load failed: {type(exc).__name__}"
            self.llm = None
            return False

    def _try_ollama(self) -> bool:
        probe = probe_ollama()
        if not probe["ok"]:
            self.error = (self.error + " · " if self.error else "") + (
                f"Ollama not reachable on {OLLAMA_HOST}:{OLLAMA_PORT} "
                f"({probe['detail']})")
            return False
        model = pick_ollama_model(probe.get("models") or [])
        if not model:
            self.error = (f"Ollama is up but serves no model — run: "
                          f"ollama pull {OLLAMA_TAGS[0]}")
            return False
        self.ollama_model = model
        self.available = True
        self.backend = "ollama"
        self.label = f"Ollama {model}"
        self.error = ""
        return True

    # -- inference ------------------------------------------------------
    def query(self, prompt: str, max_tokens: int = 120) -> str:
        """Blocking text completion. Empty string means "no answer" — the
        caller must treat that as an abstention, never as neutral."""
        if not self.available:
            return ""
        with self._lock:
            t0 = time.time()
            try:
                if self.backend == "llama_cpp" and self.llm is not None:
                    out = self.llm(prompt, max_tokens=max_tokens,
                                   temperature=0.2)
                    text = out["choices"][0]["text"]
                elif self.backend == "ollama":
                    res = ollama_chat(prompt, self.ollama_model,
                                      max_tokens=max_tokens)
                    if not res.get("ok"):
                        self.error = f"ollama: {res.get('reason')}"
                        return ""
                    text = res.get("text", "")
                else:
                    return ""
                self.last_ms = int((time.time() - t0) * 1000)
                return str(text).strip()
            except Exception as exc:
                self.error = f"inference failed: {type(exc).__name__}"
                return ""

    def ask(self, prompt: str, max_tokens: int = 200) -> dict:
        """Structured ask: {ok, text, backend, model, reason}."""
        if not self.available:
            return {"ok": False, "reason": self.error or "local model not loaded",
                    "backend": "", "text": ""}
        text = self.query(prompt, max_tokens=max_tokens)
        return {"ok": bool(text), "text": text, "backend": self.backend,
                "model": self.ollama_model or os.path.basename(self.path or ""),
                "reason": "" if text else (self.error or "empty reply")}

    def sentiment(self, text: str) -> float | None:
        """News sentiment in [-1, +1] via the local model (Section 11)."""
        if not self.available or not text:
            return None
        reply = self.query(
            "Rate the sentiment of this crypto headline from -1 to +1. "
            f"Reply with only the number.\nHeadline: {text[:200]}",
            max_tokens=8)
        m = re.search(r"-?\d+(?:\.\d+)?", reply)
        if not m:
            return None
        v = float(m.group(0))
        return max(-1.0, min(1.0, v / 10.0 if abs(v) > 1 else v))

    def reset(self) -> None:
        """Drop the loaded backend so the next try_load re-probes."""
        with self._lock:
            self.available = False
            self.backend = ""
            self.llm = None
            self.ollama_model = ""
            self.error = ""
            self._last_attempt = 0.0
            self._net_fails = 0


# ══════════════════════════════════════════════════════════════════════
# AgentInstaller — the download that used to never finish
# ══════════════════════════════════════════════════════════════════════
class AgentInstaller:
    """Downloads and wires the local agent, best-effort, without ever
    blocking the app.

    Phases, in order, each independent:
      1. ``runtime``  — llama.cpp wheel (prebuilt first, then a *reachable*
                        source build), or Ollama, or nothing at all;
      2. ``model``    — the GGUF file for this host's RAM tier;
      3. ``load``     — hand the model to LocalModel.

    A failure in 1 does not stop 2. A failure in either is reported in plain
    words with the exact missing host/binary, and ``start()`` may be called
    again at any time.
    """

    LOG_FILE = os.path.join("data", "setup_ai.log")
    DONE_FILE = os.path.join("data", ".ai_done")
    LOCK_FILE = os.path.join("data", ".agent_download.lock")
    AUTO_FILE = os.path.join("data", ".agent_auto.json")
    #: first attempt 90 s after boot, then 10 min, then 30 min
    AUTO_DELAYS_S = (90.0, 600.0, 1800.0)

    def __init__(self, local: "LocalModel", runner=None, *,
                 deps=None, probe=None, compiler=None) -> None:
        self.local = local
        self._runner = runner            # injectable for tests
        # the runtime probes are injectable so the installer's own contract
        # can be tested without touching a real daemon, compiler or index
        self._deps = deps or llama_cpp_ready
        self._probe = probe or probe_ollama
        self._compiler = compiler or compiler_ready
        self._lock = threading.Lock()
        self.state = "idle"              # idle|installing|done|failed
        self.phase = ""                  # runtime|model|load
        self.backend = ""                # what made the local slot live
        self.why = ""                    # why a rung was skipped
        self.steps: list[dict] = []      # one record per attempted rung
        self.started_at = 0.0
        self.finished_at = 0.0
        self.error = ""
        self.log: deque = deque(maxlen=60)
        self._thread: threading.Thread | None = None
        self._auto_thread: threading.Thread | None = None
        self.auto: dict[str, Any] = {"state": "idle", "attempts": 0,
                                     "last_result": "", "last_error": "",
                                     "next_try_at": 0.0}

    # -- detection ------------------------------------------------------
    def deps_installed(self) -> bool:
        try:
            return bool(self._deps()["ok"])
        except Exception:
            return False

    def ollama_ready(self) -> dict:
        try:
            return self._probe()
        except Exception as exc:            # pragma: no cover - defensive
            return {"ok": False, "models": [], "detail": type(exc).__name__}

    def model_file(self) -> str:
        return resolve_model_path(self.local)

    def model_present(self) -> bool:
        v = model_validation(self.model_file())
        return bool(v["present"] and v["magic_ok"])

    def runtime_present(self) -> bool:
        """Any usable runtime (llama.cpp OR a live Ollama daemon)."""
        return self.deps_installed() or bool(self.ollama_ready().get("ok"))

    def installed(self) -> bool:
        """Full local agent: a runtime AND a model (GGUF) — OR Ollama with a
        model already pulled, which needs no local GGUF at all."""
        if self.deps_installed() and self.model_present():
            return True
        probe = self.ollama_ready()
        return bool(probe.get("ok") and pick_ollama_model(probe.get("models") or []))

    # -- status ---------------------------------------------------------
    def status(self) -> dict:
        _, fname, label = pick_local_model(self.local.ram_gb)
        running = self._thread is not None and self._thread.is_alive()
        state = "installing" if running else self.state
        if state == "idle" and self.installed():
            state = "done"
        target = self.model_file()
        ollama = self.ollama_ready()
        runtime = bool(self.deps_installed() or ollama.get("ok"))
        have = model_validation(target)["bytes"]
        want = MODEL_SIZE_HINT.get(fname, "")
        return {
            "state": state,
            "installed": self.installed(),
            "deps_installed": self.deps_installed(),
            "runtime_present": runtime,
            "ollama": ollama,
            "compiler": self._compiler(),
            "model_present": self.model_present(),
            "model_label": label,
            "model_file": target,
            "model_size": want,
            "bytes_on_disk": have,
            "validation": model_validation(target),
            # the phase the run ENDED in is the useful fact ("it failed
            # during the model download"), so never blank it on completion
            "phase": self.phase,
            "backend": self.backend or self.local.backend,
            "why": self.why,
            "steps": list(self.steps),
            "error": self.error,
            "log": list(self.log),
            "elapsed_s": (int(time.time() - self.started_at)
                          if running and self.started_at else 0),
            "auto": self.auto_status(),
        }

    # -- the button -----------------------------------------------------
    def start(self) -> dict:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self.status()                 # already downloading
            if self.installed():
                self.state = "done"
                return self.status()
            self.state, self.phase, self.error = "installing", "runtime", ""
            self.why = ""
            self.steps = []
            self.log.clear()
            self.started_at, self.finished_at = time.time(), 0.0
            self._thread = threading.Thread(target=self._work, daemon=True,
                                            name="agent-installer")
            self._thread.start()
            return self.status()

    # -- automatic downloader -------------------------------------------
    def _auto_save(self) -> None:
        try:
            d = os.path.dirname(self.AUTO_FILE)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.AUTO_FILE, "w") as fh:
                json.dump(self.auto, fh)
        except OSError:
            pass

    def auto_status(self) -> dict:
        now = time.time()
        running = (self._auto_thread is not None
                   and self._auto_thread.is_alive())
        return {
            "state": "running" if running else self.auto.get("state", "idle"),
            "attempts": self.auto.get("attempts", 0),
            "last_result": self.auto.get("last_result", ""),
            "last_error": self.auto.get("last_error", ""),
            "next_try_in_s": (max(0, round(self.auto.get("next_try_at", 0) - now))
                              if self.auto.get("next_try_at") else None),
            "installed": self.installed(),
            "rule": ("downloads by itself after boot: a runtime (prebuilt "
                     "wheel, reachable source build, or Ollama), then the "
                     "GGUF model; retries with backoff and reports exactly "
                     "what failed"),
        }

    def start_auto(self) -> dict:
        """Arm the automatic downloader (boot path; idempotent)."""
        with self._lock:
            if self._auto_thread is not None and self._auto_thread.is_alive():
                return self.auto_status()
            if self.installed():
                self.auto.update(state="done", last_result="already installed")
                self._auto_save()
                return self.auto_status()
            self.auto = {"state": "running", "attempts": 0,
                         "last_result": "", "last_error": "", "next_try_at": 0.0}
            self._auto_thread = threading.Thread(
                target=self._auto_work, name="agent-autodownload", daemon=True)
            self._auto_thread.start()
            return self.auto_status()

    def _auto_work(self) -> None:
        for i, delay in enumerate(self.AUTO_DELAYS_S):
            if self.installed():
                break
            self.auto["next_try_at"] = time.time() + delay
            time.sleep(delay)
            self.auto["next_try_at"] = 0.0
            if self.installed():
                break
            self.auto["attempts"] = i + 1
            try:
                self.start()
                th = self._thread
                if th is not None:
                    th.join(timeout=60 * 20)
            except Exception as exc:            # pragma: no cover - defensive
                self.auto["last_error"] = repr(exc)[:300]
            if self.installed():
                self.auto["state"] = "done"
                self.auto["last_result"] = "installed automatically"
                self._auto_save()
                return
            self.auto["last_result"] = (f"attempt {i + 1} failed: "
                                        + (self.error or "not installed"))
            self.auto["last_error"] = (self.error or "")[:300]
            self._say("auto-download: " + self.auto["last_result"])
            self._auto_save()
        self.auto["state"] = "done" if self.installed() else "failed"
        if not self.installed():
            self.auto["last_result"] = (
                self.auto.get("last_result", "")
                + " — press DOWNLOAD LOCAL AGENT to retry now")
        self._auto_save()

    # -- plumbing -------------------------------------------------------
    def _say(self, line: str) -> None:
        """Append one log line, keeping its head AND tail.

        `line[-300:]` chopped the FRONT of a long failure, so the panel showed
        lines starting mid-word ("appened while trying to locate the file…")
        with the actual reason gone. Keep a generous window and elide the
        middle instead.
        """
        line = str(line).strip()
        if not line:
            return
        if len(line) > 420:
            line = line[:200].rstrip() + " … " + line[-200:].lstrip()
        self.log.append(line)
        try:
            d = os.path.dirname(self.LOG_FILE)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.LOG_FILE, "a") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

    def _exec(self, cmd: list, timeout: float | None = None) -> int:
        if self._runner is not None:
            return self._runner(cmd, self._say)
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    bufsize=1)
        except OSError as exc:
            self._say(f"could not run {cmd[0]}: {exc}")
            return 127
        ended = threading.Event()

        def _kill() -> None:
            # a source build that runs past its budget is abandoned instead
            # of holding the installer open for half an hour
            if not ended.is_set():
                self._say(f"step exceeded {timeout:.0f}s — stopping it")
                try:
                    proc.kill()
                except OSError:
                    pass
        watchdog = None
        if timeout:
            watchdog = threading.Timer(timeout, _kill)
            watchdog.daemon = True
            watchdog.start()
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                self._say(line)
        finally:
            rc = proc.wait()
            ended.set()
            if watchdog is not None:
                watchdog.cancel()
        return rc

    def _grab_lock(self):
        """flock handle or None — the boot script takes the same lock, so
        two fetchers can never run at once."""
        try:
            import fcntl
            os.makedirs(os.path.dirname(self.LOCK_FILE) or "data",
                        exist_ok=True)
            fh = open(self.LOCK_FILE, "a")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                fh.close()
                raise
            return fh
        except (OSError, ImportError):
            return None

    def _wait_lock_or_model(self, tries: int = 360):
        """While another fetcher holds the lock: wait for it, but bail out
        early if the model we want lands valid on disk.
        Returns 'model' | (\"lock\", handle) | 'timeout'."""
        for _ in range(tries):
            if self.model_present():
                return "model"
            fh = self._grab_lock()
            if fh is not None:
                return ("lock", fh)
            time.sleep(10)
        return "timeout"

    def _step(self, name: str, ok: bool, reason: str = "") -> None:
        self.steps.append({"step": name, "ok": bool(ok),
                           "reason": (reason or "")[:300],
                           "at": round(time.time(), 3)})

    # -- the runtime ladder ---------------------------------------------
    def ensure_runtime(self) -> dict:
        """Walk the runtime ladder; return {runtime, backend, detail}.

        ``runtime`` is True when ANY rung worked (or is already present).
        Nothing here raises: each rung records why it did not work.
        """
        # rung 0 — already importable
        probe = self._deps()
        if probe["ok"]:
            self._say(f"llama.cpp runtime already installed ({probe['detail']})")
            self._step("llama_cpp", True, probe["detail"])
            return {"runtime": True, "backend": "llama_cpp",
                    "detail": probe["detail"]}

        # rung 1 — prebuilt CPU wheel (correct index path, no source build)
        self._say("installing prebuilt llama.cpp CPU wheel from "
                  f"{LLAMA_WHEEL_INDEX} …")
        rc = self._exec([sys_executable(), "-m", "pip", "install",
                         "--only-binary", "llama-cpp-python",
                         "--extra-index-url", LLAMA_WHEEL_INDEX,
                         "llama-cpp-python>=0.3"],
                        timeout=SOURCE_BUILD_BUDGET_S)
        if rc == 0 and self._deps()["ok"]:
            self._step("prebuilt wheel", True, LLAMA_WHEEL_INDEX)
            return {"runtime": True, "backend": "llama_cpp",
                    "detail": "prebuilt CPU wheel"}
        self._step("prebuilt wheel", False,
                   f"pip exit {rc} — index unreachable or no wheel for this "
                   "Python/platform")

        # rung 2 — a source build ONLY when explicitly allowed and a
        # toolchain exists, with a budget
        tool = self._compiler()
        if tool["ok"] and source_build_allowed():
            self._say("prebuilt wheel unavailable — compiling from source "
                      f"(budget {SOURCE_BUILD_BUDGET_S / 60:.0f} min, "
                      f"{tool['cxx']})")
            rc = self._exec([sys_executable(), "-m", "pip", "install",
                             "llama-cpp-python>=0.3"],
                            timeout=SOURCE_BUILD_BUDGET_S)
            if rc == 0 and self._deps()["ok"]:
                self._step("source build", True, tool["cxx"] or "")
                return {"runtime": True, "backend": "llama_cpp",
                        "detail": "source build"}
            self._step("source build", False, f"pip exit {rc}")
        else:
            why = (tool["detail"] if not tool["ok"] else
                   f"source build is opt-in — set {SOURCE_BUILD_ENV}=1 to "
                   f"allow up to {SOURCE_BUILD_BUDGET_S / 60:.0f} min of "
                   "compiling")
            self._say(f"skipping the source build: {why}")
            self._step("source build", False, why)

        # rung 3 — Ollama: no compiler, no wheel index, just a daemon
        self._say("trying Ollama (no compiler or wheel index needed) …")
        probe = self._probe()
        if probe.get("ok"):
            model = pick_ollama_model(probe.get("models") or [])
            detail = (f"daemon up, model {model}" if model else
                      "daemon up but no model pulled")
            self._step("ollama", True, detail)
            return {"runtime": True, "backend": "ollama", "detail": detail,
                    "needs_pull": (not model)}
        self._step("ollama", False, probe.get("detail") or "")
        return {"runtime": False, "backend": "",
                "detail": ("no usable runtime: llama.cpp could not be "
                           "installed and Ollama is not reachable — the app "
                           "runs on formulas + brains + physics alone")}

    def ensure_model(self) -> dict:
        """Download the GGUF for this RAM tier — stdlib, no pip required.

        The old step shelled out to `scripts/download_models.py`, which imports
        `huggingface_hub`. That package is in requirements-agents.txt, which
        the boot installer does NOT install, so on a normal host the script
        exited 0 with "SKIPPED huggingface_hub unavailable" and the panel
        blamed the network. The downloader below needs nothing but urllib.
        """
        if self.model_present():
            self._say("model already on disk — 0 bytes fetched")
            self._step("model", True, "already present")
            return {"model": True, "detail": "already present"}
        repo, fname, label = pick_local_model(self.local.ram_gb)
        path = self.model_file()
        if self._runner is not None:
            # an injected runner IS the downloader in tests and embedders;
            # the stdlib path below is for the real host
            rc = self._exec([sys_executable(),
                             os.path.join("scripts", "download_models.py")],
                            timeout=60 * 30)
            if self.model_present():
                self._step("model", True, "downloaded")
                return {"model": True, "detail": "downloaded"}
            why = ("no model on disk after the download step "
                   f"(exit {rc}) — the hub is unreachable or the download "
                   "was truncated")
            self._step("model", False, why)
            return {"model": False, "detail": why}

        def _prog(done: int, total: int) -> None:
            if total:
                self.local.download_progress = round(min(1.0, done / total), 4)
                if int(done / (1 << 20)) % 64 == 0:
                    self._say(f"model {done / 1e6:.0f}/{total / 1e6:.0f} MB "
                              f"({self.local.download_progress * 100:.0f}%)")

        self._say(f"downloading {repo}/{fname} with the stdlib downloader …")
        res = download_gguf(repo, fname, path, progress=_prog)
        if res.get("ok") and self.model_present():
            self.local.download_progress = 1.0
            self._step("model", True, f"{res['bytes'] / 1e9:.2f} GB from {repo}")
            self._say(f"model ready at {path} — 0 further bytes needed")
            return {"model": True, "detail": "downloaded"}
        why = res.get("reason") or "no valid model landed on disk"
        # Second chance: if huggingface_hub IS installed, its resumable
        # downloader may succeed where a plain GET was cut.
        try:
            import huggingface_hub  # noqa: F401
            self._say("stdlib download failed — retrying with huggingface_hub")
            rc = self._exec([sys_executable(),
                             os.path.join("scripts", "download_models.py")],
                            timeout=60 * 30)
            if self.model_present():
                self.local.download_progress = 1.0
                self._step("model", True, "downloaded via huggingface_hub")
                return {"model": True, "detail": "downloaded"}
            why += f" · huggingface_hub fallback exit {rc}"
        except Exception:
            pass
        v = model_validation(path)
        if v["present"] and not v["magic_ok"]:
            why = (f"file on disk is not a GGUF ({v['bytes']} bytes) — a "
                   f"truncated download; it will be re-fetched. {why}")
        self._step("model", False, why)
        self._say("model download failed: " + why)
        return {"model": False, "detail": why}

    def ensure_ollama_model(self) -> dict:
        """Ask the daemon to pull its preferred tag (no manual CLI step)."""
        probe = self._probe()
        if not probe.get("ok"):
            return {"ok": False, "reason": "daemon down"}
        have = pick_ollama_model(probe.get("models") or [])
        if have:
            return {"ok": True, "model": have, "reason": "already pulled"}
        model = OLLAMA_TAGS[0]
        self._say(f"Ollama is up without a model — pulling {model} via the API")
        res = ollama_pull(model, progress=lambda s: self._say("ollama: " + s))
        if res.get("ok"):
            self.local.ollama_model = res.get("model") or model
            self._step("ollama pull", True, res.get("model") or model)
        else:
            self._step("ollama pull", False, res.get("reason") or "")
        return res

    def _work(self) -> None:
        lock_fh = self._grab_lock()
        if lock_fh is None:
            self._say("another agent download is already running — waiting")
            got = self._wait_lock_or_model()
            if got == "model":
                lock_fh = self._grab_lock()
            elif isinstance(got, tuple):
                lock_fh = got[1]
            else:
                self.state = "failed"
                self.error = ("another download held the lock for over an hour "
                              "— press download to try again")
                self.phase = ""
                self.finished_at = time.time()
                return
        try:
            self.phase = "runtime"
            rt = self.ensure_runtime()
            self.backend = rt.get("backend") or ""
            self.why = "" if rt["runtime"] else rt.get("detail", "")
            if rt["runtime"] and rt.get("needs_pull"):
                # never leave "run this yourself" as the final answer when
                # the daemon can fetch its own model
                self.phase = "model"
                pulled = self.ensure_ollama_model()
                if pulled.get("ok"):
                    self._finish_ok("ollama")
                    return

            # the model step runs even when the runtime failed: the two are
            # decoupled so a later runtime fix is instant
            self.phase = "model"
            md = self.ensure_model()

            if not rt["runtime"] and not md["model"]:
                # Ollama with a model already pulled is still a working agent
                probe = self.ollama_ready()
                if probe.get("ok") and pick_ollama_model(probe.get("models") or []):
                    self._finish_ok("ollama")
                    return
                raise RuntimeError(
                    "no runtime and no model: " + rt.get("detail", "")
                    + " · model: " + md.get("detail", ""))

            self.phase = "load"
            self._say("wiring the local agent …")
            self.local.reset()
            self.local.try_load(force=True)
            if self.local.available:
                self._finish_ok(self.local.backend or self.backend or "local")
            elif md["model"] and not rt["runtime"]:
                # model is on disk, runtime missing: an honest, resumable state
                self.state = "failed"
                self.error = ("model downloaded and ready, but no runtime can "
                              "load it: " + rt.get("detail", ""))
                self._say(self.error)
            else:
                self.state = "failed"
                self.error = self.local.error or "model loaded but unavailable"
        except Exception as exc:                 # never break the app
            self.state = "failed"
            self.error = str(exc)[:400]
            self._say("download stopped: " + self.error)
        finally:
            self.phase = ""
            self.finished_at = time.time()
            if lock_fh is not None:
                try:
                    lock_fh.close()               # releases the flock
                except OSError:
                    pass

    def _finish_ok(self, backend: str) -> None:
        try:
            d = os.path.dirname(self.DONE_FILE)
            if d:
                os.makedirs(d, exist_ok=True)
            open(self.DONE_FILE, "a").close()
        except OSError:
            pass
        self.backend = backend
        self.state = "done"
        self._say(f"local agent ready via {backend}: {self.local.label}")


# ══════════════════════════════════════════════════════════════════════
# agent catalog (downloadable packs)
# ══════════════════════════════════════════════════════════════════════
AGENT_CATALOG = [
    {"id": "gemini", "name": "Google Gemini", "slot": "gemini",
     "model": "gemini-2.0-flash", "setup": "https://aistudio.google.com/apikey"},
    {"id": "openai", "name": "OpenAI GPT", "slot": "backup2",
     "model": "gpt-4o", "setup": "https://platform.openai.com/api-keys"},
    {"id": "anthropic", "name": "Anthropic Claude", "slot": "backup3",
     "model": "claude-3-5-sonnet", "setup": "https://console.anthropic.com"},
    {"id": "gemini-free", "name": "Gemini backup", "slot": "backup1",
     "model": "gemini-2.0-flash", "setup": "https://aistudio.google.com/apikey"},
    {"id": "groq-70b", "name": "Groq Llama 3.3 70B", "slot": "groq",
     "model": "llama-3.3-70b-versatile",
     "setup": "https://console.groq.com/keys"},
    {"id": "groq-8b", "name": "Groq Llama 3.1 8B (fast triage)",
     "slot": "groq", "model": "llama-3.1-8b-instant",
     "setup": "https://console.groq.com/keys"},
]

GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "{model}:generateContent?key={key}")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


async def _to_thread(fn, *a):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *a)


def _extract_text(slot: str, r: httpx.Response) -> str:
    try:
        d = r.json()
    except ValueError:
        return r.text[:400]
    try:
        if slot == "backup2":
            return d["choices"][0]["message"]["content"]
        if slot == "backup3":
            return d["content"][0]["text"]
        return d["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return json.dumps(d)[:400]


class AgentManager:
    """Owns keys, rate limits, failover and the agent signals.

    Every provider — Gemini family, OpenAI, Anthropic, Groq, the local GGUF
    and an Ollama daemon — is reached through the same ``AgentRuntime``
    helper so timeouts, cooldowns, failure streaks and realized-accuracy
    scoring cannot drift apart per slot.

    Keyless operation is the default and is fully supported: the engine
    trades on formulas + brains + physics and ``status()["mode"]`` says
    ``drosophila``.
    """

    GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_PRIMARY = "llama-3.3-70b-versatile"     # deep judgment / regime
    GROQ_TRIAGE = "llama-3.1-8b-instant"         # low-latency news scoring
    GEMINI_LIST_URL = ("https://generativelanguage.googleapis.com/v1beta/"
                       "models?key={key}")
    #: per-call HTTP budget. 8 s was too tight for a 70B Groq completion and
    #: too generous for a dead host; 12 s with one retry is the compromise.
    HTTP_TIMEOUT_S = 12.0

    def __init__(self) -> None:
        self.settings: dict[str, Any] = {
            "gemini_key": "", "backup1_key": "", "backup2_key": "",
            "backup3_key": "", "groq_key": "", "newsapi_key": "",
            "neuprint_key": "",
            "gemini_enabled": True, "local_enabled": True,
            "groq_enabled": True, "openai_enabled": True,
            "anthropic_enabled": True,
            # model ids are settings so a provider rename is a field edit,
            # not a code patch
            "gemini_model": GEMINI_DEFAULT_MODEL,
            "openai_model": "gpt-4o",
            "anthropic_model": "claude-3-5-sonnet-latest",
            "theme": "dark", "haptics": True, "sound": True,
            # AUTO-PREDICT: MANUAL IS THE DEFAULT (user directive) —
            # analysis runs only on ▶ ANALYSE; only the user can turn
            # auto mode on (and it then sticks: auto_predict_touched)
            "auto_predict": False,
            "auto_predict_touched": False,
            # two prediction systems: "old" (TP/SL ladder, default)
            # or "new" (horizon forecast: price at timeframe end IS the
            # target, SL kept). Toggled from the header button.
            "system_mode": "old",
            "assets": list(SYMBOL_ORDER), "timeframes": list(TF_ORDER),
        }
        self.local = LocalModel()
        self.installer = AgentInstaller(self.local)
        self.calls: dict[str, deque] = {}        # key -> timestamps (1h window)
        self.paid: dict[str, bool] = {}
        self.last_good = {"gemini": None, "local": None}
        self.fail_streak = {"gemini": 0, "local": 0}
        self.warning = ""
        self.last_replies: dict[str, dict] = {}
        self.scores: dict[str, dict] = {}   # slot -> {"acc": ewma, "n": samples}
        self.auto_best: bool = False
        self.catalog = [dict(p) for p in AGENT_CATALOG]
        self.active_slot = "gemini"
        # §11.1 key ring: one record per slot — test = a REAL API call
        # (never a prefix check, I-1); states untested/ok/rate-limited/
        # invalid/error with last error, cooldown, calls, failures.
        self.keyring: dict[str, dict] = {
            slot: {"state": "untested", "last_error": "",
                   "cooldown_until": 0.0, "calls": 0, "failures": 0,
                   "model": "", "last_ok": 0.0, "latency_ms": None}
            for slot in ("gemini", "backup1", "backup2", "backup3", "local",
                         "groq")
        }
        self.load_settings()

    # ---------------- settings ----------------
    def load_settings(self) -> None:
        try:
            with open(SETTINGS_FILE) as fh:
                self.settings.update(json.load(fh))
            # MANUAL-DEFAULT migration: settings written by the old
            # auto-on default are not a user choice — flip them to
            # manual once.  A user who explicitly toggled (touched) or
            # re-saved auto mode keeps their value.
            if (self.settings.get("auto_predict") is True
                    and not self.settings.get("auto_predict_touched")):
                self.settings["auto_predict"] = False
        except (OSError, ValueError):
            pass
        # local secret store: the user pastes a key once and it survives a
        # restart. data/ is git-ignored and the file is chmod 600.
        try:
            with open(SECRETS_FILE) as fh:
                secrets = json.load(fh)
            if isinstance(secrets, dict):
                for k, v in secrets.items():
                    key = KEY_ALIASES.get(k, k)
                    if key in KEY_FIELDS and isinstance(v, str):
                        self.settings[key] = v.strip()
        except (OSError, ValueError):
            pass
        self.scores = {k: dict(v) for k, v in
                       (self.settings.get("agent_scores") or {}).items()}
        self.auto_best = bool(self.settings.get("auto_best"))
        for slot, st in (self.settings.get("keyring") or {}).items():
            if slot in self.keyring:
                self.keyring[slot].update(
                    {k: v for k, v in st.items() if k in self.keyring[slot]})
        # environment mirrors (spec: stored in a local environment variable)
        for slot, env in (("gemini_key", "GEMINI_API_KEY"),
                          ("backup1_key", "GEMINI_BACKUP_KEY"),
                          ("backup2_key", "OPENAI_API_KEY"),
                          ("backup3_key", "ANTHROPIC_API_KEY"),
                          ("newsapi_key", "NEWS_API_KEY"),
                          ("groq_key", "GROQ_API_KEY"),
                          ("neuprint_key", "NEUPRINT_TOKEN")):
            if not self.settings.get(slot) and os.environ.get(env):
                self.settings[slot] = os.environ[env]

    def save_settings(self) -> None:
        """Persist preferences and (separately, chmod 600) the user's keys.

        I-5 is honoured literally: no key is ever written into the repo's
        tracked files. `data/secrets.json` is git-ignored, owner-readable
        only, and never returned by any API.
        """
        try:
            safe = {k: v for k, v in self.settings.items()
                    if k not in KEY_FIELDS and k != "keyring"}
            safe["keyring"] = {slot: {kk: vv for kk, vv in st.items()
                                      if kk != "key"}
                               for slot, st in self.keyring.items()}
            os.makedirs(os.path.dirname(SETTINGS_FILE) or ".", exist_ok=True)
            with open(SETTINGS_FILE, "w") as fh:
                json.dump(safe, fh)
        except OSError:
            pass
        if os.environ.get("PREDICT_NO_SECRET_FILE") == "1":
            return
        try:
            secrets = {k: v for k, v in self.settings.items()
                       if k in KEY_FIELDS and v}
            d = os.path.dirname(SECRETS_FILE)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = SECRETS_FILE + ".tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as fh:
                json.dump(secrets, fh)
            os.replace(tmp, SECRETS_FILE)
            try:
                os.chmod(SECRETS_FILE, 0o600)
            except OSError:
                pass
        except OSError:
            pass

    def configure(self, **kv) -> None:
        """Apply a settings patch — keys included, aliases resolved.

        The old method silently ignored every key that was not already in the
        defaults (so `groq_key` could never be saved) and returned the live
        key on the next GET. Both are fixed here.
        """
        changed_keys = []
        for raw_k, v in kv.items():
            k = KEY_ALIASES.get(raw_k, raw_k)
            if k in KEY_FIELDS and v is None:
                # explicit None clears the stored key ("" means "keep")
                self.settings[k] = ""
                changed_keys.append(k)
                continue
            if isinstance(v, str) and k in KEY_FIELDS:
                v = v.strip().strip('"').strip("'")
                if not v:
                    continue          # blank means "keep the stored key"
                if len(v) > 4096:
                    continue
            if k in self.settings or k in KEY_FIELDS or k in KEY_ALIASES:
                self.settings[k] = v
                if k == "auto_predict":
                    # the USER moved it — the manual-default migration
                    # must never undo a deliberate toggle
                    self.settings["auto_predict_touched"] = True
                if k in KEY_FIELDS:
                    changed_keys.append(k)
        for k in changed_keys:
            slot = k[:-4] if k.endswith("_key") else k
            if slot in self.keyring:
                self.keyring[slot].update(state="untested", last_error="",
                                          cooldown_until=0.0, model="")
        self.save_settings()

    # ---------------- rate limiting ----------------
    def _under_limit(self, key: str) -> bool:
        now = time.time()
        q = self.calls.setdefault(key, deque())
        while q and now - q[0] > 3600:
            q.popleft()
        limit = 60 if self.paid.get(key) else AGENT_FREE_RPH
        if len(q) >= limit:
            return False
        q.append(now)
        return True

    def usage(self, key: str) -> tuple[int, int]:
        now = time.time()
        q = self.calls.get(key, deque())
        used = sum(1 for t in q if now - t <= 3600)
        return used, (60 if self.paid.get(key) else AGENT_FREE_RPH)

    # ---------------- HTTP providers ----------------
    async def _call_provider(self, slot: str, key: str, prompt: str) -> dict:
        if not key:
            return {"ok": False, "reason": "no key"}
        kr = self.keyring.get(slot) or {}
        if kr.get("cooldown_until", 0.0) > time.time():
            return {"ok": False, "reason": "rate limited"}
        kr["calls"] = int(kr.get("calls", 0)) + 1
        if not self._under_limit(key):
            kr.update(state="rate-limited", cooldown_until=time.time() + 60)
            return {"ok": False, "reason": "rate limited"}
        headers: dict[str, str] = {}
        if slot == "backup2":                     # OpenAI
            url = OPENAI_URL
            headers = {"Authorization": f"Bearer {key}",
                       "Content-Type": "application/json"}
            model = str(self.settings.get("openai_model") or "gpt-4o")
            body = {"model": model, "temperature": 0.2, "max_tokens": 400,
                    "messages": [{"role": "user", "content": prompt}]}
        elif slot == "backup3":                   # Anthropic
            url = ANTHROPIC_URL
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
                       "Content-Type": "application/json"}
            model = str(self.settings.get("anthropic_model")
                        or "claude-3-5-sonnet-latest")
            body = {"model": model, "max_tokens": 400,
                    "messages": [{"role": "user", "content": prompt}]}
        else:                                     # Gemini (primary + backup1)
            # I-2: the model id comes from ListModels when the key is tested;
            # the default is a CURRENT flash model. Gemini 1.5 was retired,
            # and the old hard-coded id made a perfectly valid key answer 404.
            model = ((self.keyring.get(slot) or {}).get("model")
                     or self.settings.get("gemini_model")
                     or GEMINI_DEFAULT_MODEL)
            url = GEMINI_URL.format(model=model, key=key)
            headers = {"Content-Type": "application/json"}
            body = {"contents": [{"parts": [{"text": prompt}]}],
                    # Gemini 2.5/3.x are thinking models: a 400-token cap can
                    # be fully consumed by internal reasoning, returning empty
                    # text. 1024 leaves room for the JSON answer.
                    "generationConfig": {"temperature": 0.2,
                                         "maxOutputTokens": 1024}}
        try:
            async with httpx.AsyncClient(timeout=self.HTTP_TIMEOUT_S) as client:
                r = await client.post(url, json=body, headers=headers)
        except Exception as e:
            # OFFLINE BUDGET: a dead/unreachable endpoint must not be
            # re-dialled on the very next analysis — cool the slot so
            # "mostly offline" providers cost one fast failure per
            # 90 s instead of a 12 s stall on every run
            kr.update(state="offline",
                      last_error=e.__class__.__name__,
                      failures=int(kr.get("failures", 0)) + 1,
                      cooldown_until=time.time() + 90.0)
            return {"ok": False, "reason": e.__class__.__name__}
        if r.status_code == 429:
            kr.update(state="rate-limited", cooldown_until=time.time() + 60,
                      failures=int(kr.get("failures", 0)) + 1,
                      last_error="http 429")
            return {"ok": False, "reason": "rate limited"}
        if r.status_code in (400, 401, 403):
            # 400 can be a renamed model id — mark the slot so the next call
            # falls back to the pinned id (I-2)
            kr.update(state="invalid", last_error=f"http {r.status_code}",
                      failures=int(kr.get("failures", 0)) + 1)
            return {"ok": False, "reason": f"http {r.status_code}"}
        if r.status_code >= 400:
            _cool = 60.0 if r.status_code >= 500 else 0.0
            kr.update(state="error", last_error=f"http {r.status_code}",
                      failures=int(kr.get("failures", 0)) + 1,
                      **({"cooldown_until": time.time() + _cool}
                         if _cool else {}))
            return {"ok": False, "reason": f"http {r.status_code}"}
        rem = r.headers.get("x-ratelimit-remaining")
        if rem is not None:
            try:
                self.paid[key] = float(rem) > 50
            except ValueError:
                pass
        text = _extract_text(slot, r)
        parsed = parse_agent_reply(text)
        parsed["ok"] = True
        parsed["slot"] = slot
        kr.update(state="ok", last_error="", last_ok=time.time())
        return parsed

    # ---------------- local provider (llama.cpp or Ollama) --------------
    async def _call_local(self, prompt: str, max_tokens: int = 200) -> dict:
        """Ask the local agent without blocking the event loop."""
        if not self.settings.get("local_enabled", True):
            return {"ok": False, "reason": "local agent disabled in settings"}
        if not self.local.available and not self.local.error:
            await _to_thread(self.local.try_load)
        if not self.local.available:
            return {"ok": False, "backend": "", "reason":
                    self.local.error or "local model not loaded"}
        # the review is ADVISORY — it must never be the reason the app
        # freezes. If the loop is already starving, skip the local leg.
        try:
            from . import loadgov as _gov
            if _gov.current() >= 0.75:
                return {"ok": False, "backend": "",
                        "reason": "skipped — event loop busy (advisory only)"}
        except Exception:
            pass
        # the local model is ADVISORY — cap it: a CPU-bound llama on a
        # starved box must not eat the analysis (paired with the engine
        # stage's 8 s wall; timeout ⇒ honest "too slow" skip)
        try:
            res = await asyncio.wait_for(
                _to_thread(self.local.ask, prompt, max_tokens), timeout=5.0)
        except (asyncio.TimeoutError, TimeoutError):
            res = {"ok": False, "backend": self.local.backend or "local",
                   "reason": "local review timed out (5 s) — backend too "
                             "slow or offline"}
        self.last_replies["local"] = {k: v for k, v in res.items()
                                      if k != "text"}
        if not res.get("ok"):
            return {"ok": False, "backend": res.get("backend", ""),
                    "reason": res.get("reason", "empty reply")}
        parsed = parse_agent_reply(res["text"])
        parsed.update(ok=True, slot="local", backend=res.get("backend", ""),
                      model=res.get("model", ""))
        return parsed

    # ---------------- main query ----------------
    async def query(self, ctx: dict) -> dict:
        """Run the failover chain; returns the merged agent signal."""
        prompt = build_prompt(ctx)
        slots = [("gemini", self.settings.get("gemini_key")),
                 ("backup1", self.settings.get("backup1_key")),
                 ("backup2", self.settings.get("backup2_key")),
                 ("backup3", self.settings.get("backup3_key"))]
        if self.auto_best:
            best = self.best_slot()
            if best:
                slots.sort(key=lambda kv: 0 if kv[0] == best else 1)
        cloud = None
        for slot, key in slots:
            # per-slot enable flags: a disabled Gemini must not silence a
            # configured OpenAI/Anthropic key (the old code checked the
            # gemini flag for every provider)
            flag = {"gemini": "gemini_enabled", "backup1": "gemini_enabled",
                    "backup2": "openai_enabled",
                    "backup3": "anthropic_enabled"}[slot]
            if not key or not self.settings.get(flag, True):
                continue
            res = await self._call_provider(slot, key, prompt)
            self.last_replies[slot] = {k: v for k, v in res.items()
                                       if k != "text"}
            if res.get("ok"):
                cloud = res
                self.active_slot = slot
                self.fail_streak["gemini"] = 0
                self.last_good["gemini"] = res
                self.warning = ""
                break
            if res.get("reason") == "rate limited":
                self.warning = f"{slot} rate-limited — trying next agent"
                continue
            self.fail_streak["gemini"] += 1

        local_res = await self._call_local(prompt) if cloud is None else None
        if local_res is not None:
            self.last_replies["local"] = local_res
            if local_res.get("ok"):
                self.fail_streak["local"] = 0
                self.last_good["local"] = local_res
                self.active_slot = "local"
            else:
                self.fail_streak["local"] += 1

        # fallback: reuse the previous cloud signal for a few failed GRCs
        if cloud is None and self.last_good["gemini"] is not None:
            if self.fail_streak["gemini"] <= AGENT_FALLBACK_GRCS:
                cloud = dict(self.last_good["gemini"])
                cloud["fallback"] = True
            else:
                self.warning = ("agent system offline after "
                                f"{self.fail_streak['gemini']} failures")
        return self._merge(cloud, local_res)

    def _merge(self, cloud: dict | None, local: dict | None) -> dict:
        g_c = float(cloud.get("confidence", 0.0)) if cloud else 0.0
        g_d = int(cloud.get("direction", 0)) if cloud else 0
        l_c = float(local.get("confidence", 0.0)) if local else 0.0
        l_d = int(local.get("direction", 0)) if local else 0
        if g_c > 0 and l_c > 0:
            direction = ((g_d * g_c + l_d * l_c) / (g_c + l_c + AGENT_EPS))
            confidence = (g_c + l_c) / 2.0
            weight = AGENT_COMBINED_WEIGHT
            source = "combined"
        elif g_c > 0:
            direction, confidence, weight, source = (g_d, g_c,
                                                     AGENT_GEMINI_WEIGHT,
                                                     "gemini")
        elif l_c > 0:
            direction, confidence, weight, source = (l_d, l_c,
                                                     AGENT_LOCAL_WEIGHT,
                                                     "local")
        else:
            return {"direction": 0, "confidence": 0.0, "weight": 0.0,
                    "source": "none", "insight": "", "gemini": cloud,
                    "local": local, "warning": self.warning,
                    "backend": self._backend_label()}
        direction = 1 if direction > 0.1 else (-1 if direction < -0.1 else 0)
        insight = ((cloud or {}).get("insight")
                   or (local or {}).get("insight") or "")
        return {
            "direction": direction,
            "confidence": round(min(1.0, confidence), 4),
            "weight": weight,
            "source": source,
            "insight": insight,
            "gemini": cloud,
            "local": local,
            "warning": self.warning,
            "backend": self._backend_label(),
        }

    def _backend_label(self) -> str:
        """Which agent actually answered (for the UI chip)."""
        if self.local.available:
            return self.local.backend or "local"
        if self.active_slot and self.settings.get(f"{self.active_slot}_key"):
            return self.active_slot
        return ""

    # ---------------- analysis review (stage 6) ----------------
    async def review_text(self, prompt: str, max_tokens: int = 320) -> dict:
        """Free-form completion for NARRATION only (never a decision).

        The cognition layer builds the thought trace; a model is allowed to
        re-tell it. This exposes that one path without giving the model any
        way to feed a number back into the engine: the caller validates the
        text against the trace and discards it on mismatch.
        """
        if not self.settings.get("local_enabled", True):
            return {"ok": False, "reason": "local agent disabled in settings",
                    "backend": "", "text": ""}
        if not self.local.available and not self.local.error:
            await _to_thread(self.local.try_load)
        if not self.local.available:
            cloud = self.best_slot()
            if cloud:
                key = self.settings.get(f"{cloud}_key")
                if key:
                    res = await self._call_provider(cloud, key, prompt)
                    if res.get("ok"):
                        return {"ok": True, "text": res.get("insight", ""),
                                "backend": cloud, "model": cloud}
            return {"ok": False, "backend": "",
                    "reason": (self.local.error or "no local model loaded")
                    + "; no cloud slot answered"}
        # same 5 s wall as _call_local — a hung llama must not stall
        # any advisory path either
        try:
            res = await asyncio.wait_for(
                _to_thread(self.local.ask, prompt, max_tokens), timeout=5.0)
        except (asyncio.TimeoutError, TimeoutError):
            return {"ok": False, "backend": self.local.backend or "local",
                    "reason": "local review timed out (5 s) — backend too "
                              "slow or offline", "text": ""}
        if not res.get("ok"):
            return {"ok": False, "backend": res.get("backend", ""),
                    "reason": res.get("reason", "empty reply"), "text": ""}
        return {"ok": True, "text": res.get("text", ""),
                "backend": res.get("backend", ""),
                "model": res.get("model", "")}

    async def review(self, symbol: str, tf: str, store: dict,
                     result: dict | None = None) -> dict:
        """The `ai_review` stage of ▶ ANALYSE.

        Returns a payload that is always renderable:
        ``{available, backend, note, reviews: {name: {...}}, correction}``.
        ``available`` is False only when nothing at all could answer, and
        ``note`` then says exactly what is missing.
        """
        prompt = build_review_prompt(symbol, tf, store, result)
        reviews: dict[str, dict] = {}
        tried: list[str] = []

        # 1) cloud slots (same failover order as query)
        for slot, keyname in (("gemini", "gemini_key"),
                              ("backup1", "backup1_key"),
                              ("backup2", "backup2_key"),
                              ("backup3", "backup3_key")):
            key = self.settings.get(keyname)
            flag = {"gemini": "gemini_enabled", "backup1": "gemini_enabled",
                    "backup2": "openai_enabled",
                    "backup3": "anthropic_enabled"}[slot]
            if not key or not self.settings.get(flag, True):
                continue
            tried.append(slot)
            res = await self._call_provider(slot, key, prompt)
            self.last_replies[slot] = {k: v for k, v in res.items()
                                       if k != "text"}
            reviews[slot] = ({"ok": True, "reason": res.get("insight", ""),
                              "errors": res.get("errors") or [],
                              "direction": res.get("direction"),
                              "confidence": res.get("confidence"),
                              "backend": slot}
                             if res.get("ok") else
                             {"ok": False, "error": res.get("reason", "")})
            if res.get("ok"):
                break

        # 2) local agent (llama.cpp GGUF or Ollama) — short JSON only:
        # 200 tokens on CPU took ~132 s and starved the loop (banner)
        local_res = await self._call_local(prompt, max_tokens=120)
        if local_res.get("ok"):
            reviews["local"] = {"ok": True,
                                "reason": local_res.get("insight", ""),
                                "errors": local_res.get("errors") or [],
                                "direction": local_res.get("direction"),
                                "confidence": local_res.get("confidence"),
                                "backend": local_res.get("backend") or "local"}
        elif self.settings.get("local_enabled", True) and self.local.available:
            reviews["local"] = {"ok": False,
                                "error": local_res.get("reason", "no answer")}

        ok = {k: v for k, v in reviews.items() if v.get("ok")}
        if not ok:
            missing = []
            if not any(self.settings.get(k) for k in
                       ("gemini_key", "backup1_key", "backup2_key",
                        "backup3_key")):
                missing.append("no cloud key")
            if not self.local.available:
                missing.append("local agent not loaded"
                               + (f" ({self.local.error})" if self.local.error
                                  else ""))
            return {"available": False,
                    "backend": "",
                    "note": ("no AI answered — deterministic pipeline only ("
                             + "; ".join(missing) + "). The call above stands "
                             "on formulas + brains + physics."),
                    "reviews": reviews, "tried": tried, "correction": None}

        # a correction needs a confident, contradictory review
        correction = None
        for name, rv in ok.items():
            d = int(rv.get("direction") or 0)
            c = float(rv.get("confidence") or 0.0)
            cur = (result or {}).get("direction") or 0
            if d and cur and d != cur and c >= 0.6:
                correction = {"by": name, "direction": d, "confidence": c,
                              "reason": (rv.get("reason") or "")[:200]}
                break
        backend = next(iter(ok.values())).get("backend", "")
        return {"available": True, "backend": backend,
                "note": f"reviewed by {', '.join(ok)} ({backend})",
                "reviews": reviews, "tried": tried,
                "correction": correction}

    # ---------------- UI helpers ----------------
    def status(self) -> dict:
        cat = self.catalog_status()

        def slot_status(slot: str, key: str) -> dict:
            used, limit = self.usage(key) if key else (0, AGENT_FREE_RPH)
            reply = self.last_replies.get(slot) or {}
            st = ("disabled" if (slot == "gemini" and
                                 not self.settings.get("gemini_enabled", True))
                  else "connected" if reply.get("ok")
                  else "rate-limited" if reply.get("reason") == "rate limited"
                  else "disconnected" if key else "not configured")
            return {"key": _mask(key), "status": st,
                    "last": {k: reply.get(k) for k in
                             ("direction", "confidence", "insight", "reason")},
                    "usage": used, "limit": limit}

        keys = ["gemini_key", "backup1_key", "backup2_key", "backup3_key"]
        slots = ["gemini", "backup1", "backup2", "backup3"]
        has_cloud = any(self.settings.get(k) for k in keys)
        local_ok = bool(self.local.available)
        # every agent is OPTIONAL: without keys the app runs fully on the
        # Drosophila brain + formulas + math + physics
        mode = ("cloud" if has_cloud and local_ok else
                "cloud" if has_cloud else
                "local" if local_ok else "drosophila")
        return {
            "keyring": {k: dict(v) for k, v in self.keyring.items()},
            "catalog": cat,
            "groq": {"configured": bool(self.groq_key),
                     "primary": self.GROQ_PRIMARY,
                     "triage": self.GROQ_TRIAGE},
            "auditor": None,   # filled by main.py from engine.auditor
            "slots": [slot_status(s, self.settings.get(k))
                      for s, k in zip(slots, keys)],
            "local": {
                "label": self.local.label,
                "available": self.local.available,
                "backend": self.local.backend,
                "download_progress": self.local.download_progress,
                "ram_gb": round(self.local.ram_gb, 1),
                "last_ms": self.local.last_ms,
                "error": self.local.error,
                "last": self.last_replies.get("local"),
                "enabled": self.settings.get("local_enabled", True),
                "install": self.installer.status(),
            },
            # the installer is ALSO top-level: the Agents tab used to read
            # agents.installer and find null, so it had nothing to render
            "installer": self.installer.status(),
            "active": self.active_slot,
            "backend": self._backend_label(),
            "warning": self.warning,
            "mode": mode,
            "has_cloud_keys": has_cloud,
        }

    # ---------------- catalog / import / auto-best ----------------
    def catalog_status(self) -> dict:
        packs = []
        for pack in self.catalog:
            slot = pack["slot"]
            p2 = dict(pack)
            p2["imported"] = bool(self.settings.get(f"{slot}_key"))
            sc = self.scores.get(slot) or {}
            p2["score"] = round(sc.get("acc", 0.5), 4) if sc.get("n", 0) else None
            p2["samples"] = sc.get("n", 0)
            packs.append(p2)
        return {"packs": packs, "auto_best": self.auto_best,
                "best": self.best_slot(), "scores": self.scoreboard()}

    def import_pack(self, pack_id: str, key: str, slot: str | None = None) -> dict:
        pack = next((p for p in self.catalog if p["id"] == pack_id), None)
        if pack is None:
            return {"ok": False, "error": "unknown pack"}
        slot = slot or pack["slot"]
        if slot not in ("gemini", "backup1", "backup2", "backup3", "groq"):
            return {"ok": False, "error": "unknown slot"}
        self.settings[f"{slot}_key"] = key.strip()
        if slot != "groq":
            self.settings["gemini_enabled"] = True
        self.save_settings()
        return {"ok": True, "slot": slot, "pack": pack["id"]}

    def export_pack(self, pack_id: str) -> dict | None:
        pack = next((p for p in self.catalog if p["id"] == pack_id), None)
        if pack is None:
            return None
        out = dict(pack)
        out["format"] = "predict-agent-pack-v1"
        out["import"] = ('POST /api/agents/import {"pack_id": "%s", "key": "…"}'
                         % pack_id)
        out["key"] = ""     # never embedded — download then import with your key
        return out

    def set_auto(self, enabled: bool) -> dict:
        self.auto_best = bool(enabled)
        self.settings["auto_best"] = self.auto_best
        self.save_settings()
        return {"ok": True, "auto_best": self.auto_best,
                "best": self.best_slot()}

    def scoreboard(self) -> dict:
        return {slot: {"acc": round(sc.get("acc", 0.5), 4), "n": sc.get("n", 0)}
                for slot, sc in self.scores.items() if sc.get("n", 0)}

    def record_outcome(self, correct: bool) -> None:
        """Rolling realized-accuracy score for the slot that answered."""
        slot = self.active_slot
        if not slot:
            return
        sc = self.scores.setdefault(slot, {"acc": 0.5, "n": 0})
        sc["acc"] = 0.90 * sc["acc"] + 0.10 * (1.0 if correct else 0.0)
        sc["n"] = sc["n"] + 1
        self.settings["agent_scores"] = {k: v for k, v in self.scores.items()}
        self.save_settings()

    # ---------------- Groq synthesis (Prompt 2) ----------------
    @property
    def groq_key(self) -> str:
        return str(self.settings.get("groq_key") or "")

    async def _groq_call(self, model: str, messages: list,
                         max_tokens: int = 400,
                         temperature: float = 0.2) -> dict:
        """Exponential-backoff Groq call (free tier: 30 RPM caps)."""
        if not self.groq_key:
            return {"ok": False, "reason": "no GROQ_API_KEY"}
        delay = 1.5
        last = "unknown"
        for attempt in range(4):
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    r = await client.post(
                        self.GROQ_URL,
                        headers={"Authorization": f"Bearer {self.groq_key}"},
                        json={"model": model, "messages": messages,
                              "temperature": temperature,
                              "max_tokens": max_tokens,
                              "response_format": {"type": "json_object"}},
                    )
                if r.status_code == 429:
                    last = "rate limited"
                elif r.status_code >= 500:
                    last = f"server {r.status_code}"
                elif r.status_code >= 400:
                    return {"ok": False, "reason": f"http {r.status_code}"}
                else:
                    data = r.json()
                    text = data["choices"][0]["message"].get("content", "")
                    return {"ok": True, "text": text, "model": model}
            except Exception as e:
                last = e.__class__.__name__
            await asyncio.sleep(delay)
            delay *= 2                       # 1.5 → 3 → 6 s
        return {"ok": False, "reason": last}

    async def synthesize(self, compact: dict) -> dict:
        """Qualitative synthesis: regime + weight adjustments + confidence.

        `compact` is the PRE-AGGREGATED formula summary (top components,
        multipliers, emotion/physics/news/brain votes) — never the raw 75+
        series, to respect rate limits.
        """
        prompt = (
            "You are the risk committee of a crypto prediction engine. "
            "Given this pre-aggregated snapshot of 75+ formulas, output "
            "strict JSON: "
            '{"regime": "Trending" | "Mean-Reverting" | "High-Volatility Shock", '
            '"weight_adjustments": {formula_id: float override 0.5..1.5, '
            "ONLY for formulas you would up/down-weight}, "
            '"confidence_score": float 0..1, "rationale": "<40 words"}.\n'
            "Snapshot: " + json.dumps(compact, separators=(",", ":"))[:1500]
        )
        res = await self._groq_call(
            self.GROQ_PRIMARY, [{"role": "user", "content": prompt}],
            max_tokens=400, temperature=0.2)
        if not res.get("ok"):
            return {"ok": False, "reason": res.get("reason")}
        try:
            out = json.loads(res["text"])
        except ValueError:
            return {"ok": False, "reason": "unparseable JSON"}
        regime = str(out.get("regime", ""))
        if regime not in ("Trending", "Mean-Reverting", "High-Volatility Shock"):
            return {"ok": False, "reason": f"bad regime {regime!r}"}
        try:
            conf = max(0.0, min(1.0, float(out.get("confidence_score", 0.0))))
        except (TypeError, ValueError):
            conf = 0.0
        raw_adj = out.get("weight_adjustments") or {}
        adj: dict[str, float] = {}
        if isinstance(raw_adj, dict):
            for k, v in list(raw_adj.items())[:12]:
                try:
                    adj[str(k)] = max(0.5, min(1.5, float(v)))
                except (TypeError, ValueError):
                    continue
        return {"ok": True, "regime": regime, "weight_adjustments": adj,
                "confidence_score": conf,
                "rationale": str(out.get("rationale", ""))[:300],
                "model": self.GROQ_PRIMARY, "ts": time.time()}

    async def triage_headline(self, text: str) -> dict:
        """Fast sentiment triage of a live news headline (8B model)."""
        res = await self._groq_call(
            self.GROQ_TRIAGE,
            [{"role": "user", "content":
              'Score this crypto headline sentiment as JSON {"sentiment": '
              'float -1..1, "reliability": float 0..1}:\n' + text[:400]}],
            max_tokens=120, temperature=0.0)
        if not res.get("ok"):
            return {"ok": False, "reason": res.get("reason")}
        try:
            d = json.loads(res["text"])
            return {"ok": True,
                    "sentiment": max(-1.0, min(1.0, float(d.get("sentiment", 0)))),
                    "reliability": max(0.0, min(1.0, float(d.get("reliability", 0)))),
                    "model": self.GROQ_TRIAGE}
        except (ValueError, TypeError):
            return {"ok": False, "reason": "unparseable"}

    def best_slot(self) -> str | None:
        """Highest realized accuracy with >= 3 samples."""
        cands = [(sc.get("acc", 0.0), slot) for slot, sc in self.scores.items()
                 if sc.get("n", 0) >= 3 and self.settings.get(f"{slot}_key")]
        if not cands:
            return None
        cands.sort(reverse=True)
        return cands[0][1]

    # ---------------- key ring: ListModels-driven model choice (I-2) -----
    @staticmethod
    def choose_model(models: list[dict]) -> str | None:
        """The ListModels response is the ONLY truth about which models the
        key may call (I-2): keep generateContent models, drop vision/
        embedding/tts/image/audio ids, rank by family and generation.
        A pinned id is honoured only if the key can actually use it."""
        usable = []
        for m in models:
            mid = str(m.get("name") or m.get("id") or "")
            methods = m.get("supportedGenerationMethods") or []
            if "generateContent" not in methods:
                continue
            low = mid.lower()
            if any(bad in low for bad in ("vision", "embedding", "tts",
                                          "image", "audio", "aqa", "guard")):
                continue
            # ListModels returns "models/gemini-2.5-flash"; the generateContent
            # URL already carries /models/, so keep the bare id.
            usable.append(mid.split("/")[-1])
        if not usable:
            return None

        def rank(mid: str) -> tuple:
            low = mid.lower().split("/")[-1]
            # stable, current-first: 2.5 flash is the default workhorse; a
            # retired 1.5 model must never outrank it just because the key
            # still lists it.
            try:
                pos = next(i for i, m in enumerate(GEMINI_MODEL_ORDER)
                           if low.startswith(m))
            except StopIteration:
                pos = len(GEMINI_MODEL_ORDER)
            flash = 0 if "flash" in low else 1      # flash first (speed)
            lite = 1 if "lite" in low else 0         # full flash before lite
            preview = 1 if "preview" in low else 0   # GA before preview
            pro = 1 if "pro" in low else 0
            return (preview, pos, flash, lite, pro, low)
        usable.sort(key=rank)
        return usable[0]

    async def test(self, slot: str) -> dict:
        kr = self.keyring.setdefault(slot, {"state": "untested"})
        kr["calls"] = int(kr.get("calls", 0)) + 1
        key = self.settings.get(f"{slot}_key", "")
        if slot == "local":
            self.local.reset()
            await _to_thread(self.local.try_load, True)   # manual retry
            st = self.local.describe()
            if self.local.available:
                st["detail"] = self.local.label
            return {"ok": self.local.available, "detail": st,
                    "error": self.local.error,
                    "backend": self.local.backend}
        t0 = time.monotonic()
        # Gemini family: ListModels first — the response decides which model
        # id this key may call (I-2). No prefix checks anywhere (I-1).
        if slot in ("gemini", "backup1") and key:
            pinned = str(self.settings.get("gemini_model")
                         or GEMINI_DEFAULT_MODEL).split("/")[-1]
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    r = await client.get(self.GEMINI_LIST_URL.format(key=key))
                if r.status_code in (400, 401, 403):
                    kr.update(state="invalid", last_error=f"http {r.status_code}",
                              cooldown_until=time.time() + 300)
                    self.save_settings()
                    return {"ok": False,
                            "detail": {"state": "invalid",
                                       "error": f"key rejected: http {r.status_code}"}}
                if r.status_code == 429:
                    kr.update(state="rate-limited", last_error="429 on ListModels",
                              cooldown_until=time.time() + 60)
                    self.save_settings()
                    return {"ok": False, "detail": {"state": "rate-limited"}}
                r.raise_for_status()
                models = (r.json().get("models") or [])
                chosen = self.choose_model(models)
                names = [str(m.get("name") or "") for m in models]
                pinned_ok = any(pinned in n for n in names)
                if pinned_ok:
                    chosen = pinned            # honour pinned only if usable
                if not chosen:
                    kr.update(state="error",
                              last_error="no generateContent model")
                    return {"ok": False,
                            "detail": {"state": "error",
                                       "error": "no usable model"}}
                kr["model"] = chosen
                kr["pinned_ok"] = pinned_ok
            except Exception as e:
                # sandbox/network blocks are documented as such (G-8) —
                # not presented as "API down"
                kr.update(state="error",
                          last_error=f"ListModels {e.__class__.__name__}")
                kr["model"] = pinned
        res = await self._call_provider(
            slot, key, "Reply with only this JSON: "
                       '{"direction": 0, "confidence": 0.5, "insight": "ok"}')
        ms = round((time.monotonic() - t0) * 1000, 1)
        kr["latency_ms"] = ms
        if res.get("ok"):
            kr.update(state="ok", last_error="", last_ok=time.time())
        else:
            kr["failures"] = int(kr.get("failures", 0)) + 1
            reason = str(res.get("reason") or "error")
            if "rate" in reason:
                kr.update(state="rate-limited", last_error=reason,
                          cooldown_until=time.time() + 60)
            elif "http 40" in reason:
                kr.update(state="invalid", last_error=reason,
                          cooldown_until=time.time() + 300)
            else:
                kr.update(state="error", last_error=reason)
        self.save_settings()
        return {"ok": bool(res.get("ok")), "detail": res, "keyring": dict(kr),
                "model": kr.get("model"), "latency_ms": ms}


# ══════════════════════════════════════════════════════════════════════
# boot glue (kept for main.py / start.py)
# ══════════════════════════════════════════════════════════════════════
_boot_lock = threading.Lock()
_boot_started = False
_boot_state: dict[str, Any] = {"stage": "idle", "detail": "", "at": 0.0,
                               "ollama": None, "install": None}


def _boot_worker(manager=None) -> None:
    """Daemon-first: probe/start Ollama, report the install state, refresh
    the agent status. Never blocks boot and never raises."""
    _boot_state["stage"] = "probing ollama"
    _boot_state["at"] = time.time()
    try:
        probe = ensure_ollama_running()
        _boot_state["ollama"] = probe
        _boot_state["detail"] = probe.get("detail") or ""
        inst = getattr(manager, "installer", None) if manager else None
        if inst is not None:
            _boot_state["stage"] = "checking local model install"
            st = inst.status()
            _boot_state["install"] = {k: st.get(k) for k in
                                      ("state", "installed", "model_present",
                                       "backend", "error")}
            # NEVER blocks boot: start() only spawns its own thread
            if not st.get("installed"):
                try:
                    inst.start()
                except Exception as exc:
                    _boot_state["install"]["error"] = str(exc)
        if manager is not None:
            # issue 2: bring the local model UP at boot (not on first
            # query) — a downloaded GGUF + installed runtime must show
            # LOADED immediately after a restart, not "not loaded"
            local = getattr(manager, "local", None)
            if local is not None and not local.available:
                _boot_state["stage"] = "loading local model"
                try:
                    local.try_load()
                    _boot_state["loaded"] = bool(local.available)
                except Exception as exc:
                    _boot_state["loaded_error"] = str(exc)
            _boot_state["stage"] = "refreshing agent status"
            try:
                st = manager.status()
                _boot_state["agents"] = {"mode": st.get("mode"),
                                         "backend": st.get("backend")}
            except Exception as exc:
                _boot_state["agents"] = {"error": str(exc)}
    except Exception as exc:                    # never kill the thread silently
        _boot_state["detail"] = f"boot worker error: {exc}"
    finally:
        _boot_state["stage"] = "done"
        _boot_state["at"] = time.time()


def start_boot_worker(manager=None) -> dict:
    """Idempotent: spawn the agent boot worker once per process."""
    global _boot_started
    with _boot_lock:
        if _boot_started:
            return {"started": False, "state": dict(_boot_state)}
        _boot_started = True
        _boot_state["stage"] = "starting"
        t = threading.Thread(target=_boot_worker, args=(manager,),
                             name="agent-boot", daemon=True)
        t.start()
    return {"started": True, "state": dict(_boot_state)}


def boot_state() -> dict:
    return dict(_boot_state)

# [removed 2026-11-23] get_status — dead code: zero references anywhere
