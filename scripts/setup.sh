#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# One-shot, idempotent bootstrap for the whole app.
#   venv → core deps → agent/AI deps → data dirs → local-model download
#   → hourly learning cron (best-effort) → stamp for fast re-runs
# Every stage after the core dependencies is best-effort: the app still
# runs in formula-only mode when a stage cannot finish (e.g. offline).
# Usage:  bash scripts/setup.sh          (FORCE=1 to re-run everything)
# ══════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
STAMP=".setup_stamp"
FORCE="${FORCE:-0}"
log()  { printf '[setup] %s\n' "$*"; }
warn() { printf '[setup] WARN: %s\n' "$*"; }

req_hash() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$@" | sha256sum | cut -d' ' -f1
  else
    shasum -a 256 "$@" | shasum -a 256 | cut -d' ' -f1
  fi
}

# ── 1) virtualenv ─────────────────────────────────────────────────────
if [ ! -x .venv/bin/python ]; then
  log "creating virtualenv (.venv)…"
  if ! "$PYTHON_BIN" -m venv .venv; then
    warn "venv creation failed — falling back to system interpreter"
  fi
fi
VENV_PY=".venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY="$PYTHON_BIN"
log "interpreter: $VENV_PY ($("$VENV_PY" --version 2>&1))"

# ── 2) core requirements (hard requirement) ───────────────────────────
CUR_HASH="$(req_hash requirements.txt)"
PREV_HASH=""
[ -f "$STAMP" ] && PREV_HASH="$(cut -d' ' -f1 "$STAMP" 2>/dev/null || true)"
if [ "$FORCE" != "1" ] && [ "$CUR_HASH" = "$PREV_HASH" ] && [ -f "$STAMP" ]; then
  log "core requirements up to date — skipping pip"
else
  log "installing core requirements…"
  if ! "$VENV_PY" -m pip install -q -r requirements.txt; then
    warn "core dependency install FAILED — cannot start the app"
    exit 1
  fi
  log "core requirements installed ✓"
fi

# ── 3) agent / AI requirements (best-effort: llama-cpp, HF hub) ───────
AGENTS_STATUS="skipped"
if [ -f requirements-agents.txt ]; then
  log "installing agent/AI requirements (llama-cpp-python, huggingface-hub, cmake)…"
  if "$VENV_PY" -m pip install -q -r requirements-agents.txt \
       > data/setup_agents.log 2>&1; then
    AGENTS_STATUS="installed"
    log "agent/AI requirements installed ✓"
  else
    AGENTS_STATUS="failed (see data/setup_agents.log — app runs without local model)"
    warn "agent/AI requirements failed — core app unaffected"
  fi
fi

# ── 4) directories ────────────────────────────────────────────────────
mkdir -p data data/history models
[ -f data/settings.json ] || echo '{"theme":"dark","haptics":true,"sound":true}' > data/settings.json

# ── 5) local AI model download (RAM-tier, best-effort) ────────────────
MODEL_STATUS="skipped"
log "downloading the RAM-tier local model (best-effort)…"
if "$VENV_PY" scripts/download_models.py > data/setup_models.log 2>&1; then
  MODEL_STATUS="$(grep -oE 'DONE|SKIPPED[^ ]*' data/setup_models.log | tail -1 || echo done)"
  log "model stage: $MODEL_STATUS"
else
  MODEL_STATUS="failed (see data/setup_models.log — cloud/formula mode unaffected)"
  warn "model download did not complete — will retry on first use"
fi

# ── 6) hourly learning cron (best-effort; the app also has an
#       in-process hourly safety loop, so this is belt-and-braces) ─────
CRON_STATUS="not-installed"
CRON_LINE="0 * * * * $(pwd)/scripts/learning_cron.sh >> $(pwd)/data/learning_cron.log 2>&1"
if command -v crontab >/dev/null 2>&1; then
  if crontab -l 2>/dev/null | grep -q "learning_cron.sh"; then
    CRON_STATUS="present"
  else
    { crontab -l 2>/dev/null; echo "$CRON_LINE"; } | crontab - 2>/dev/null \
      && CRON_STATUS="installed" || CRON_STATUS="failed"
  fi
fi
log "learning cron: $CRON_STATUS"

# ── 7) stamp for fast re-runs ─────────────────────────────────────────
echo "$CUR_HASH $(date -u +%s)" > "$STAMP"

# ── 8) summary ────────────────────────────────────────────────────────
cat <<EOF
[setup] ═══════════ SETUP SUMMARY ═══════════
[setup] core deps      : ok
[setup] agent/AI deps  : $AGENTS_STATUS
[setup] local model    : $MODEL_STATUS
[setup] learning cron  : $CRON_STATUS
[setup] data dirs      : data/, data/history/, models/
[setup] next step      : ./run.sh   (or auto-started via scripts/auto_start.sh)
[setup] ═══════════════════════════════════════
EOF
exit 0
