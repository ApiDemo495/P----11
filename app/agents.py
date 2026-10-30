"""Section 11 — agent integration (Gemini primary + 3 backups + local model).

Everything is optional: with no keys and no local model the app runs on the
core formulas alone. Failover: Gemini → Backup1 (Gemini) → Backup2 (OpenAI)
→ Backup3 (Anthropic); a failed signal is reused for AGENT_FALLBACK_GRCS
cycles and then forced neutral with a UI warning.
"""
from __future__ import annotations

import json
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

SETTINGS_FILE = "data/settings.json"
MODELS_DIR = "models"

GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "gemini-1.5-flash:generateContent?key={key}")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

LOCAL_MODELS = [
    # (min_ram_gb, repo, file, label)
    (14, "microsoft/Phi-3-mini-4k-instruct-gguf", "Phi-3-mini-4k-instruct-q4k.gguf",
     "Phi-3-mini-4k-instruct (Q4_K_M)"),
    (7, "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
     "TinyLlama-1.1B-Chat-v1.0 (Q4_K_M)"),
    (0, "Qwen/Qwen2-0.5B-Instruct-GGUF", "qwen2-0_5b-instruct-q4_k_m.gguf",
     "Qwen2-0.5B-Instruct (Q4_K_M)"),
]


def _mask(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "•" * len(key)
    return key[:4] + "•" * (len(key) - 8) + key[-4:]


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
        f"Latest news: {ctx.get('headline')} (sentiment={ctx.get('news_sentiment')})",
        f"Emotion composite (CERC): {ctx.get('cerc')}",
        "Reply with ONLY a JSON object: "
        '{"direction": +1|−1|0 (confirm/contradict/uncertain), '
        '"confidence": 0..1, "insight": "one short sentence"}.',
    ]
    return "\n".join(lines)


def parse_agent_reply(text: str) -> dict:
    """Pull the direction/confidence/insight triple out of a model reply."""
    out = {"direction": 0, "confidence": 0.0, "insight": ""}
    if not text:
        return out
    m = re.search(r"\{.*\}", text, re.DOTALL)
    blob = m.group(0) if m else text
    try:
        # tolerate unicode minus and trailing commas
        cleaned = (blob.replace("−", "-").replace("–", "-")
                   .replace("'", '"'))
        d = json.loads(cleaned)
        out["direction"] = (1 if float(d.get("direction", 0)) > 0 else
                            (-1 if float(d.get("direction", 0)) < 0 else 0))
        out["confidence"] = max(0.0, min(1.0, float(d.get("confidence", 0.0))))
        out["insight"] = str(d.get("insight", ""))[:300]
        return out
    except (ValueError, TypeError):
        pass
    # heuristic fallback: look for confirm/contradict words + a 0..1 number
    low = text.lower()
    if any(w in low for w in ("confirm", "agree", "bullish", "up")):
        out["direction"] = 1
    elif any(w in low for w in ("contradict", "disagree", "bearish", "down")):
        out["direction"] = -1
    nums = re.findall(r"0?\.\d+|\b\d{1,2}\b", text)
    if nums:
        v = float(nums[0])
        out["confidence"] = min(1.0, v / 100.0 if v > 1 else v)
    out["insight"] = text.strip()[:300]
    return out


class LocalModel:
    """llama.cpp wrapper — selected automatically from available RAM."""

    def __init__(self) -> None:
        self.available = False
        self.label = "not loaded"
        self.llm = None
        self.path = None
        self.ram_gb = _ram_gb()
        self.download_progress = 0.0
        self.last_ms = None
        self.error = ""
        self._lock = threading.Lock()

    def try_load(self) -> None:
        if self.llm is not None:
            return
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            self.error = f"llama-cpp-python not installed ({e.__class__.__name__})"
            self.label = "unavailable"
            return
        repo, fname, label = None, None, None
        for min_ram, r, f, lab in LOCAL_MODELS:
            if self.ram_gb >= min_ram:
                repo, fname, label = r, f, lab
                break
        assert repo and fname and label
        self.label = label
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            try:
                os.makedirs(MODELS_DIR, exist_ok=True)
                from huggingface_hub import hf_hub_download  # type: ignore
                self.download_progress = 0.05
                got = hf_hub_download(repo_id=repo, filename=fname,
                                      local_dir=MODELS_DIR)
                path = got
                self.download_progress = 1.0
            except Exception as e:
                self.error = f"model download failed: {e.__class__.__name__}"
                return
        try:
            self.llm = Llama(model_path=path, n_ctx=2048, n_threads=4,
                             verbose=False)
            self.path = path
            self.available = True
        except Exception as e:
            self.error = f"model load failed: {e.__class__.__name__}"

    def query(self, prompt: str, max_tokens: int = 120) -> str:
        if not self.available or self.llm is None:
            return ""
        with self._lock:
            t0 = time.time()
            try:
                out = self.llm(prompt, max_tokens=max_tokens, temperature=0.2)
                self.last_ms = int((time.time() - t0) * 1000)
                return out["choices"][0]["text"].strip()
            except Exception as e:
                self.error = f"inference failed: {e.__class__.__name__}"
                return ""

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


def _ram_gb() -> float:
    try:
        out = subprocess.run(["free", "-m"], capture_output=True, text=True,
                             timeout=3).stdout
        for line in out.splitlines():
            if line.lower().startswith("mem:"):
                return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 4.0


class AgentManager:
    """Owns keys, rate limits, failover and the two agent signals."""

    def __init__(self) -> None:
        self.settings: dict[str, Any] = {
            "gemini_key": "", "backup1_key": "", "backup2_key": "",
            "backup3_key": "", "newsapi_key": "", "neuprint_key": "",
            "gemini_enabled": True, "local_enabled": True,
            "theme": "dark", "haptics": True, "sound": True,
            "assets": list(SYMBOL_ORDER), "timeframes": list(TF_ORDER),
        }
        self.local = LocalModel()
        self.calls: dict[str, deque] = {}        # key -> timestamps (1h window)
        self.paid: dict[str, bool] = {}
        self.last_good = {"gemini": None, "local": None}
        self.fail_streak = {"gemini": 0, "local": 0}
        self.warning = ""
        self.last_replies: dict[str, dict] = {}
        self.active_slot = "gemini"
        self.load_settings()

    # ---------------- settings ----------------
    def load_settings(self) -> None:
        try:
            with open(SETTINGS_FILE) as fh:
                self.settings.update(json.load(fh))
        except (OSError, ValueError):
            pass
        import os as _os
        # environment mirrors (spec: stored in a local environment variable)
        for slot, env in (("gemini_key", "GEMINI_API_KEY"),
                          ("backup1_key", "GEMINI_BACKUP_KEY"),
                          ("backup2_key", "OPENAI_API_KEY"),
                          ("backup3_key", "ANTHROPIC_API_KEY"),
                          ("newsapi_key", "NEWSAPI_KEY")):
            if not self.settings.get(slot) and _os.environ.get(env):
                self.settings[slot] = _os.environ[env]

    def save_settings(self) -> None:
        try:
            os.makedirs(os.path.dirname(SETTINGS_FILE) or ".", exist_ok=True)
            with open(SETTINGS_FILE, "w") as fh:
                json.dump(self.settings, fh)
        except OSError:
            pass

    def configure(self, **kv) -> None:
        for k, v in kv.items():
            if k in self.settings:
                self.settings[k] = v
        self.save_settings()

    # ---------------- rate limiting ----------------
    def _under_limit(self, key: str) -> bool:
        now = time.time()
        q = self.calls.setdefault(key, deque())
        while q and now - q[0] > 3600:
            q.popleft()
        limit = 60 if self.paid.get(key) else AGENT_FREE_RPH * AGENT_PAID_PER_GRC
        # free tier: 15/hour total (1 per GRC); paid: 60/hour (4 per GRC)
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
        if not self._under_limit(key):
            return {"ok": False, "reason": "rate limited"}
        headers: dict[str, str] = {}
        if slot in ("backup2",):          # OpenAI
            url = OPENAI_URL
            headers = {"Authorization": f"Bearer {key}",
                       "Content-Type": "application/json"}
            body = {"model": "gpt-4o", "temperature": 0.2,
                    "messages": [{"role": "user", "content": prompt}]}
        elif slot in ("backup3",):        # Anthropic
            url = ANTHROPIC_URL
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
                       "Content-Type": "application/json"}
            body = {"model": "claude-3-5-sonnet-latest", "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}]}
        else:                             # Gemini (primary + backup1)
            url = GEMINI_URL.format(key=key)
            headers = {"Content-Type": "application/json"}
            body = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.post(url, json=body, headers=headers)
        except Exception as e:
            return {"ok": False, "reason": e.__class__.__name__}
        if r.status_code == 429:
            return {"ok": False, "reason": "rate limited"}
        if r.status_code >= 400:
            return {"ok": False, "reason": f"http {r.status_code}"}
        # paid-tier detection (Gemini headers expose remaining quota)
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
        return parsed

    # ---------------- main query ----------------
    async def query(self, ctx: dict) -> dict:
        """Run the failover chain; returns the merged agent signal."""
        prompt = build_prompt(ctx)
        slots = [("gemini", self.settings.get("gemini_key")),
                 ("backup1", self.settings.get("backup1_key")),
                 ("backup2", self.settings.get("backup2_key")),
                 ("backup3", self.settings.get("backup3_key"))]
        cloud = None
        for slot, key in slots:
            if not key or not self.settings.get("gemini_enabled", True):
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
        # local model (threaded so the event loop never blocks)
        local_res = None
        if self.settings.get("local_enabled", True):
            loop = threading.Thread(target=self.local.try_load, daemon=True)
            if not self.local.available and not self.local.error:
                loop.start()
                loop.join(timeout=0.05)
            if self.local.available:
                text = await _to_thread(self.local.query, prompt)
                local_res = parse_agent_reply(text)
                local_res["ok"] = bool(text)
                self.last_replies["local"] = local_res
                if text:
                    self.fail_streak["local"] = 0
                    self.last_good["local"] = local_res
                else:
                    self.fail_streak["local"] += 1
        # fallback: reuse previous cloud signal for up to 3 failed GRCs
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
            direction, confidence, weight, source = g_d, g_c, AGENT_GEMINI_WEIGHT, "gemini"
        elif l_c > 0:
            direction, confidence, weight, source = l_d, l_c, AGENT_LOCAL_WEIGHT, "local"
        else:
            return {"direction": 0, "confidence": 0.0, "weight": 0.0,
                    "source": "none", "insight": "", "gemini": cloud,
                    "local": local, "warning": self.warning}
        direction = 1 if direction > 0.1 else (-1 if direction < -0.1 else 0)
        insight = ((cloud or {}).get("insight") or (local or {}).get("insight") or "")
        return {
            "direction": direction,
            "confidence": round(min(1.0, confidence), 4),
            "weight": weight,
            "source": source,
            "insight": insight,
            "gemini": cloud,
            "local": local,
            "warning": self.warning,
        }

    # ---------------- UI helpers ----------------
    def status(self) -> dict:
        def slot_status(slot: str, key: str) -> dict:
            used, limit = self.usage(key) if key else (0, AGENT_FREE_RPH)
            reply = self.last_replies.get(slot) or {}
            st = "disabled" if (slot == "gemini" and
                                not self.settings.get("gemini_enabled", True)) else \
                 ("connected" if reply.get("ok") else
                  ("rate-limited" if reply.get("reason") == "rate limited" else
                   ("disconnected" if key else "not configured")))
            return {"key": _mask(key), "status": st,
                    "last": {k: reply.get(k) for k in
                             ("direction", "confidence", "insight", "reason")},
                    "usage": used, "limit": limit}

        keys = ["gemini_key", "backup1_key", "backup2_key", "backup3_key"]
        slots = ["gemini", "backup1", "backup2", "backup3"]
        return {
            "slots": [slot_status(s, self.settings.get(k))
                      for s, k in zip(slots, keys)],
            "local": {
                "label": self.local.label,
                "available": self.local.available,
                "download_progress": self.local.download_progress,
                "ram_gb": round(self.local.ram_gb, 1),
                "last_ms": self.local.last_ms,
                "error": self.local.error,
                "last": self.last_replies.get("local"),
                "enabled": self.settings.get("local_enabled", True),
            },
            "active": self.active_slot,
            "warning": self.warning,
        }

    async def test(self, slot: str) -> dict:
        key = self.settings.get(f"{slot}_key", "")
        if slot == "local":
            await _to_thread(self.local.try_load)
            return {"ok": self.local.available, "detail": self.local.label,
                    "error": self.local.error}
        res = await self._call_provider(
            slot, key, "Reply with only this JSON: "
                       '{"direction": 0, "confidence": 0.5, "insight": "ok"}')
        return {"ok": bool(res.get("ok")), "detail": res}


async def _to_thread(fn, *a):
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *a)


def _extract_text(slot: str, r: httpx.Response) -> str:
    try:
        d = r.json()
    except ValueError:
        return r.text[:400]
    try:
        if slot in ("backup2",):
            return d["choices"][0]["message"]["content"]
        if slot in ("backup3",):
            return d["content"][0]["text"]
        return d["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return json.dumps(d)[:400]
