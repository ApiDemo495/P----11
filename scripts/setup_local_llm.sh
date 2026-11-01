#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# Local formula auditor — Ollama + qwen2.5-coder:3b (user Prompt 1).
# ~2 GB RAM · fits an 8 GB / 4-core Codespace with headroom.
# Idempotent: safe to re-run; skips work that is already done.
#
#   bash scripts/setup_local_llm.sh          install + pull + start
#   bash scripts/setup_local_llm.sh --status check daemon + model
# ══════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${OLLAMA_MODEL:-qwen2.5-coder:3b}"
PORT=11434

if [ "${1:-}" = "--status" ]; then
  curl -sf -m 3 "http://127.0.0.1:${PORT}/api/tags" | head -c 300; echo
  exit 0
fi

# 1) install runtime (skips when ollama already on PATH)
if ! command -v ollama >/dev/null 2>&1; then
  echo "[llm] installing Ollama runtime…"
  curl -fsSL https://ollama.com/install.sh | sh || {
    echo "[llm] WARN: install failed — auditor stays offline (core app unaffected)"
    exit 0
  }
fi

# 2) daemon (idempotent)
if ! curl -sf -m 2 "http://127.0.0.1:${PORT}/api/tags" >/dev/null 2>&1; then
  echo "[llm] starting ollama serve (logs: data/ollama.log)…"
  mkdir -p data
  nohup ollama serve > data/ollama.log 2>&1 &
  for i in $(seq 1 20); do
    curl -sf -m 2 "http://127.0.0.1:${PORT}/api/tags" >/dev/null 2>&1 && break
    sleep 1
  done
fi

# 3) pull the model (skips when present)
if ollama list 2>/dev/null | grep -q "${MODEL%%:*}"; then
  echo "[llm] model ${MODEL} already present"
else
  echo "[llm] pulling ${MODEL} (~1.9 GB, one-time)…"
  ollama pull "${MODEL}"
fi

echo "[llm] verifying…"
ollama list | grep -i qwen || true
echo "[llm] done — the app auditor picks it up automatically (OLLAMA_URL/MODEL env to relocate)."
