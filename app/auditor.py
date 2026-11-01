"""Local LLM formula auditor — Ollama + qwen2.5-coder:3b.

User spec (Prompt 1): a 4-core / 8 GB Codespace runs qwen2.5-coder:3b
(~2 GB) at decent speed. The model is a QUANT DEBUGGER, invoked ONLY on:

  a) an unhandled exception or NaN in a formula (any of the 75+),
  b) a formula's rolling directional accuracy < 45% over a 50-GRC window.

It is NEVER used for bar-by-bar math or tick processing. Requests are
deterministic (temperature=0.1) and capped at 200 tokens to protect CPU.

The daemon is optional: if Ollama is not running the auditor records
"skipped (offline)" and the core app is unaffected.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import httpx

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:3b")
ACCURACY_FLOOR = 0.45          # trigger: rolling accuracy below this
ACCURACY_WINDOW = 50           # GRC samples required before judging
COOLDOWN_S = 600               # one audit per formula per 10 minutes
PING_TTL_S = 60                # re-probe daemon availability every minute

_QUANT_PROMPT = """You are a quant debugger for a crypto prediction engine.
Formula: {name}
Recent metric history (last {n} values): {history}
{error_block}
Check ONLY for: divide-by-zero, NaN/inf propagation, stale timeframe
windows, sign flips at boundaries, unit mismatches. Suggest at most 2
concrete code corrections. Be terse. Max 200 tokens."""


class FormulaAuditor:
    """Ollama-backed auditor with cooldowns; never blocks the engine."""

    def __init__(self) -> None:
        self.url = OLLAMA_URL
        self.model = OLLAMA_MODEL
        self._available = False
        self._ping_at = 0.0
        self._lock = threading.Lock()
        self._cooldown: dict[str, float] = {}
        self.reports: list[dict] = []
        self.triggers = {"exceptions": 0, "accuracy": 0, "audits": 0,
                         "skipped_offline": 0}

    # ---------------- daemon health ----------------
    @property
    def available(self) -> bool:
        now = time.time()
        if now - self._ping_at < PING_TTL_S:
            return self._available
        self._ping_at = now
        try:
            r = httpx.get(f"{self.url}/api/tags", timeout=2.0)
            tags = (r.json() or {}).get("models") or []
            names = " ".join(m.get("name", "") for m in tags)
            self._available = r.status_code == 200 and self.model.split(":")[0] in names
        except Exception:
            self._available = False
        return self._available

    # ---------------- triggers ----------------
    def report_anomalies(self, symbol: str, tf: str,
                         anomalies: list[tuple], values: dict) -> None:
        """Called from compute_all's collector: (fid, kind, err)."""
        if not anomalies:
            return
        for fid, kind, err in anomalies:
            name = f"F{fid}"
            history = [round(float(values.get(fid, 0.0)), 6)]
            if self._enqueue(symbol, tf, name, kind, history, err):
                self.triggers["exceptions"] += 1

    def check_accuracy(self, symbol: str, tf: str, fid: int,
                       ring: list[float]) -> None:
        """Trigger (b): full window and rolling accuracy < 45%."""
        if len(ring) < ACCURACY_WINDOW:
            return
        acc = (sum(ring) / len(ring) + 1.0) / 2.0      # ±1 rewards -> 0..1
        if acc >= ACCURACY_FLOOR:
            return
        if self._enqueue(symbol, tf, f"F{fid}", "low-accuracy",
                         [round(acc, 4), *[round(x, 2) for x in ring[-8:]]], None):
            self.triggers["accuracy"] += 1

    def _enqueue(self, symbol: str, tf: str, name: str, kind: str,
                 history: list, err: str | None) -> bool:
        """True when the trigger was actionable (not cooled down)."""
        key = f"{name}|{kind}"
        now = time.time()
        with self._lock:
            if now - self._cooldown.get(key, 0.0) < COOLDOWN_S:
                return False
            self._cooldown[key] = now
        if not self.available:
            self.triggers["skipped_offline"] += 1
            return True
        threading.Thread(target=self._run_audit, daemon=True,
                         args=(symbol, tf, name, kind, history, err)).start()
        return True

    # ---------------- the actual LLM call ----------------
    def _run_audit(self, symbol: str, tf: str, name: str, kind: str,
                   history: list, err: str | None) -> None:
        error_block = (f"Runtime error:\n{err[:600]}" if err
                       else f"Trigger: rolling accuracy {history[0]} "
                            f"< {ACCURACY_FLOOR} over {ACCURACY_WINDOW} GRCs")
        prompt = _QUANT_PROMPT.format(name=f"{name} ({symbol}/{tf})",
                                      n=len(history), history=history,
                                      error_block=error_block)
        t0 = time.time()
        try:
            r = httpx.post(f"{self.url}/api/generate",
                           json={"model": self.model, "prompt": prompt,
                                 "stream": False,
                                 "options": {"temperature": 0.1,
                                             "num_predict": 200}},
                           timeout=60.0)
            reply = ((r.json() or {}).get("response") or "").strip()
            ok = r.status_code == 200 and bool(reply)
        except Exception as e:
            reply, ok = f"auditor error: {e.__class__.__name__}", False
        self.triggers["audits"] += 1
        with self._lock:
            self.reports.append({
                "ts": time.time(), "symbol": symbol, "tf": tf,
                "formula": name, "trigger": kind, "model": self.model,
                "latency_ms": int((time.time() - t0) * 1000),
                "ok": ok, "reply": reply[:1200],
            })
            del self.reports[:-25]      # keep the last 25 reports

    # ---------------- status for the API/UI ----------------
    def status(self) -> dict:
        return {
            "available": self.available,
            "model": self.model,
            "url": self.url,
            "triggers": dict(self.triggers),
            "accuracy_floor": ACCURACY_FLOOR,
            "window": ACCURACY_WINDOW,
            "reports": list(reversed(self.reports[:10])),
            "hint": ("daemon down — run scripts/setup_local_llm.sh"
                     if not self.available else "watching 75+ formulas"),
        }
