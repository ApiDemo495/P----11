"""Tab explainer — plain-language "why this value, what it means, what
the signal is, and how TP/SL are calculated".

Two sources, in order:
  1. the LOCAL model — Ollama + qwen2.5-coder:3b (same daemon the
     formula auditor uses). Auto-import: if the daemon is up but the
     model is missing, we start `ollama pull` once in the background
     (never blocks a request; the fallback explains in the meantime).
  2. a deterministic fallback composed from the engine's own numbers —
     every claim grounded in the payload, never invented.

Requests are temperature=0.1, capped ~260 tokens, cached per
(view, symbol, tf, payload-hash) for 10 minutes so a tab switch never
hammers the CPU.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from typing import Any

import httpx

from .auditor import OLLAMA_MODEL, OLLAMA_URL

PULL_COOLDOWN_S = 1800
CACHE_TTL_S = 600
_cache: dict[str, dict] = {}
_pull_state = {"started": 0.0, "running": False, "done": False,
               "error": "", "pid": None}
_lock = threading.Lock()

SYSTEM = ("You are the explainer of a multi-asset crypto prediction app. "
          "Given the engine's JSON for one tab, write a short, factual "
          "readout: (1) WHY these values came out this way, (2) WHAT they "
          "mean in plain words, (3) the current SIGNAL and why, (4) if TP/"
          "SL numbers are present, how they were calculated. Use ONLY the "
          "numbers in the JSON — never invent figures. Plain text, short "
          "lines, max 260 tokens.")


# ------------------------------------------------------------------ status
def status() -> dict[str, Any]:
    available = False
    models: list[str] = []
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2.0)
        if r.status_code == 200:
            models = [m.get("name", "") for m in (r.json() or {}).get("models") or []]
            available = any(OLLAMA_MODEL.split(":")[0] in m for m in models)
    except Exception:
        pass
    pulling = _pull_state["running"]
    if not available and not pulling:
        _maybe_start_pull()
        pulling = _pull_state["running"]
    return {"available": available, "model": OLLAMA_MODEL, "url": OLLAMA_URL,
            "pulling": pulling, "pull_done": _pull_state["done"],
            "pull_error": _pull_state["error"],
            "source": ("qwen2.5-coder" if available else
                       "pulling model…" if pulling else
                       "local fallback (deterministic)"),
            "fallback_only": not available}


def _maybe_start_pull() -> None:
    """Auto-import: bring qwen2.5-coder:3b in when Ollama itself is up."""
    with _lock:
        now = time.time()
        if _pull_state["running"] or (
                _pull_state["started"]
                and now - _pull_state["started"] < PULL_COOLDOWN_S):
            return
        _pull_state["started"] = now
        _pull_state["running"] = True

    def _run() -> None:
        try:
            # daemon reachable? (model missing is why we are here)
            httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3.0).raise_for_status()
            proc = subprocess.Popen(
                ["ollama", "pull", OLLAMA_MODEL],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True)
            _pull_state["pid"] = proc.pid
            _, err = proc.communicate(timeout=1800)
            _pull_state["running"] = False
            if proc.returncode == 0:
                _pull_state["done"] = True
            else:
                _pull_state["error"] = (err or "").strip()[:300]
        except FileNotFoundError:
            _pull_state["running"] = False
            _pull_state["error"] = "ollama CLI not on PATH"
        except Exception as exc:
            _pull_state["running"] = False
            _pull_state["error"] = repr(exc)[:300]

    threading.Thread(target=_run, name="ollama-pull", daemon=True).start()


# ------------------------------------------------------------- fallbacks
def _fmt(v: Any, d: int = 4) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    a = abs(f)
    if a == 0:
        return "0.00"
    if a >= 10 ** -d:
        return f"{f:.{d}f}"
    if a >= 1e-6:
        return f"{f:.6f}".rstrip("0")
    return f"{f:.1e}"


def _tp_sl_block(ctx: dict) -> str:
    atr = ctx.get("atr")
    price = ctx.get("price")
    conf = ctx.get("confidence")
    direction = ctx.get("direction", 0)
    tps = ctx.get("tps") or []
    lines = []
    if atr and price and tps:
        side = "long (buy)" if direction > 0 else (
            "short (sell)" if direction < 0 else "flat")
        lines.append(
            f"TP/SL MATH — side {side}: every sub-target is "
            f"TP_k = entry + dir × |conf| × ATR × √minutes_k × 0.5 "
            f"(TARGET_SCALE=0.5). With entry {_fmt(price, 2)}, "
            f"conf {_fmt(conf, 3)}, ATR {_fmt(atr, 4)}:")
        for t in tps[:4]:
            lines.append(f"  TP{t.get('k')} · {t.get('minutes')}m → "
                         f"{_fmt(t.get('level'), 2)}")
        sl = ctx.get("sl")
        if sl:
            lines.append(
                f"  SL → {_fmt(sl, 2)}  (1 × ATR from entry, tightened by "
                f"dynamic level placement when a fractal boundary sits "
                f"closer; ratio TP:SL = {ctx.get('ratio', '—')})")
    return "\n".join(lines)


def fallback(view: str, symbol: str, tf: str, ctx: dict) -> str:
    """Deterministic readout — every sentence built from ctx numbers."""
    v = view or "dashboard"
    head = f"{symbol} · {tf.upper()} · {ctx.get('phase', 'live')} — "
    if v == "formulas":
        top = (ctx.get("top") or [])[:5]
        zeros = (ctx.get("zero_reasons") or {})
        parts = [
            head + f"CCI {_fmt(ctx.get('cci_adjusted'))} "
            f"(raw {_fmt(ctx.get('cci_raw'))}), direction "
            f"{ctx.get('direction_label', '—')}, confidence "
            f"{_fmt(ctx.get('confidence'))}.",
            "WHAT IT MEANS: CCI is the weighted sum of every core formula "
            "outputs (ω × F adjusted); its sign is the lean, |CCI| vs the "
            "noise floor decides if the lean is real.",
            "TOP CONTRIBUTORS (why this value): " +
            ("; ".join(f"{c.get('name')} {_fmt(c.get('contribution'))} "
                       f"(ω {_fmt(c.get('weight'), 3)} × "
                       f"F {_fmt(c.get('value'))})" for c in top)
             or "none this window"),
        ]
        if zeros:
            parts.append(
                "HONEST ZEROS (not bugs): " +
                "; ".join(f"F{k}: {str(val)[:70]}" for k, val in
                          list(zeros.items())[:4]))
        parts.append(_tp_sl_block(ctx))
        return "\n".join(parts)
    if v == "emotion":
        return "\n".join([
            head + f"CERC {_fmt(ctx.get('cerc'))} (aggregate "
            f"{_fmt(ctx.get('aggregate'))}), {ctx.get('flow', '')}.",
            "WHAT IT MEANS: CERC is the net emotion flow — positive = "
            "greed/fomo side in control, negative = fear side. It "
            "modulates confidence ±15% and NEVER sets the direction.",
            "HONEST ZEROS: " + ("; ".join(
                f"E{k}: {str(val)[:60]}" for k, val in
                list((ctx.get('zero_reasons') or {}).items())[:4])
                or "none — every channel has a reading"),
            _tp_sl_block(ctx),
        ])
    if v == "physics":
        active = [m for m in (ctx.get("mechanisms") or []) if m.get("active")]
        inactive = [m for m in (ctx.get("mechanisms") or []) if not m.get("active")]
        return "\n".join([
            head + f"UFP {_fmt(ctx.get('ufp'))} · WFCP "
            f"{_fmt(ctx.get('wfcp'))} · DRPT {_fmt(ctx.get('drpt'))} · "
            f"GSPI {_fmt(ctx.get('gspi'))} (confidence multipliers "
            f"steps 5-7).",
            f"WHAT IT MEANS: physics never sets direction — it scales "
            f"confidence. {len(active)}/21 mechanisms active this "
            f"window: " + (", ".join(str(m.get('name', ''))[:22]
                                     for m in active[:6]) or "none"),
            "INACTIVE WITH REASONS: " + ("; ".join(
                f"P{m.get('id')} {str(m.get('why', ''))[:70]}"
                for m in inactive[:4]) or "none"),
            _tp_sl_block(ctx),
        ])
    if v == "news":
        return "\n".join([
            head + f"news value {_fmt(ctx.get('value'))}, polarity "
            f"{_fmt(ctx.get('polarity'))}, {ctx.get('count', 0)} live "
            f"items, coverage: {str(ctx.get('coverage', ''))[:140]}.",
            "WHAT IT MEANS: each headline is scored for relevance/"
            "sentiment/impact; the aggregate enters as a subsystem vote "
            "and a theme prior — it never overrides the tape.",
            _tp_sl_block(ctx),
        ])
    # dashboard (default)
    parts = [
        head + f"SIGNAL {ctx.get('signal', '—')} · confidence "
        f"{_fmt(ctx.get('confidence'))} (floor 0.35) · "
        f"{ctx.get('reason', '')}",
        "WHY: " + "; ".join(
            f"{c.get('name')} {_fmt(c.get('contribution'))}"
            for c in (ctx.get("top") or [])[:5]),
        "VOTES (double-check): " + ", ".join(
            f"{k} {'▲' if v2 > 0 else '▼' if v2 < 0 else '·'}"
            for k, v2 in (ctx.get("votes") or {}).items()),
    ]
    cap = ctx.get("penalty_capped")
    if cap:
        parts.append(
            f"PENALTY CAP: RTP/PCV penalties ({_fmt(ctx.get('penalty'))}) "
            f"exceeded 90% of the available confidence — the residue is "
            f"kept so the number is never an exact 0.00.")
    parts.append(_tp_sl_block(ctx))
    return "\n".join(parts)


# ------------------------------------------------------------------ LLM
def _ctx_hash(view: str, symbol: str, tf: str, ctx: dict) -> str:
    raw = json.dumps(ctx, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


async def explain(view: str, symbol: str, tf: str,
                  ctx: dict) -> dict[str, Any]:
    key = f"{view}|{symbol}|{tf}|{_ctx_hash(view, symbol, tf, ctx)}"
    hit = _cache.get(key)
    if hit and time.time() - hit["t"] < CACHE_TTL_S:
        return {**hit["body"], "cached": True}
    fb = fallback(view, symbol, tf, ctx)
    body: dict[str, Any] = {
        "ok": True, "view": view, "symbol": symbol, "tf": tf,
        "text": fb, "source": "local-fallback", "model": None,
        "cached": False,
    }
    st = status()
    if st["available"]:
        try:
            prompt = (f"Tab: {view} | {symbol}/{tf}\nEngine JSON:\n"
                      f"{json.dumps(ctx, default=str)[:4000]}\n\n"
                      f"{SYSTEM}")
            async with httpx.AsyncClient(timeout=45.0) as client:
                r = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": OLLAMA_MODEL, "prompt": prompt,
                          "stream": False, "options": {
                              "temperature": 0.1, "num_predict": 260}})
            r.raise_for_status()
            text = ((r.json() or {}).get("response") or "").strip()
            if text:
                body.update(text=text, source="qwen2.5-coder",
                            model=OLLAMA_MODEL)
        except Exception as exc:
            body["llm_error"] = repr(exc)[:200]
    else:
        body["model_status"] = st["source"]
    _cache[key] = {"t": time.time(), "body": body}
    if len(_cache) > 200:
        for k in [k for k, v in _cache.items()
                  if time.time() - v["t"] > CACHE_TTL_S][:50]:
            _cache.pop(k, None)
    return body
