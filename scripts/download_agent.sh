#!/usr/bin/env bash
# Background local-agent download for Codespaces — spawned FIRST by
# auto_start (before the engine start finishes, long before the
# deferred backtest). Downloads the llama.cpp runtime + the default
# agent model (Qwen2.5-Coder-3B) WITHOUT ever blocking the app:
#   order at boot:  1) agent download (this, background, niced)
#                   2) app auto-start (placeholder -> engine)
#                   3) daily backtest (deferred, waits for this)
# It waits for core provisioning (.setup_stamp) before touching pip
# so two installers never corrupt each other's downloads, shares
# its pid/log with setup.sh's AI stage, and is a no-op once done.
set -u
cd "$(dirname "$0")/.." || exit 0
mkdir -p data models
LOG=data/setup_ai.log
MODEL_FILE="models/qwen2.5-coder-3b-instruct-q4_k_m.gguf"

if [ -f data/.ai_done ] && [ -f "$MODEL_FILE" ]; then
  echo "[agent] already downloaded — skipped" >> "$LOG"; exit 0
fi
if [ -f data/setup_ai.pid ] && kill -0 "$(cat data/setup_ai.pid 2>/dev/null)" 2>/dev/null; then
  echo "[agent] download already running — skipped" >> "$LOG"; exit 0
fi
echo $$ > data/setup_ai.pid
trap 'rm -f data/setup_ai.pid' EXIT

# wait (max 30 min) for core provisioning so pip never races pip
for _ in $(seq 1 360); do
  if [ -f .setup_stamp ] && [ -x .venv/bin/python ] && \
     .venv/bin/python -c "import fastapi" 2>/dev/null; then break; fi
  sleep 5
done
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3

echo "[agent] $(date -u +%FT%TZ) downloading local agent: llama.cpp runtime + Qwen2.5-Coder-3B (background, app unaffected)" >> "$LOG"
if "$PY" -c "import llama_cpp" 2>/dev/null; then
  echo "[agent] llama.cpp runtime already installed" >> "$LOG"
else
  nice -n 19 "$PY" -m pip install -q -r requirements-agents.txt >> "$LOG" 2>&1 \
    || { echo "[agent] runtime install failed (non-fatal; retry from the Agents tab)" >> "$LOG"; exit 0; }
fi
nice -n 19 "$PY" scripts/download_models.py >> "$LOG" 2>&1 || true
if [ -f "$MODEL_FILE" ]; then
  touch data/.ai_done
  echo "[agent] ✓ local agent ready: Qwen2.5-Coder-3B" >> "$LOG"
else
  echo "[agent] model not present after download attempt (hub unreachable? retry from the Agents tab)" >> "$LOG"
fi
exit 0
