#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# One-shot, idempotent bootstrap for the whole app.
#
#   FOREGROUND (gates startup, seconds):   venv + core requirements +
#       data dirs + stamp + cron check
#   BACKGROUND (never gates startup):      agent/AI deps (llama-cpp build)
#       + RAM-tier model download  ->  data/setup_ai.log
#
# Set BLOCKING=1 to wait for the background stages (e.g. CI).
# Usage:  bash scripts/setup.sh     (FORCE=1 re-runs, fresh ~1s re-run)
# ══════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
STAMP=".setup_stamp"
FORCE="${FORCE:-0}"
BLOCKING="${BLOCKING:-0}"
T0=$(date +%s)
log()  { printf '[setup %4ss] %s\n' "$(( $(date +%s) - T0 ))" "$*"; }
warn() { printf '[setup %4ss] WARN: %s\n' "$(( $(date +%s) - T0 ))" "$*"; }

req_hash() {
  # raw sha256 of the file contents (must match `sha256sum | cut` in
  # auto_start.sh / run.sh so the staleness check agrees with the stamp)
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
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

# ── 2) core requirements (hard requirement, foreground) ───────────────
CORE_HASH="$(req_hash requirements.txt)"
AI_HASH="$(req_hash requirements-agents.txt 2>/dev/null || echo none)"
PREV_CORE="" PREV_AI=""
if [ -f "$STAMP" ]; then
  PREV_CORE="$(awk '{print $1}' "$STAMP")"
  PREV_AI="$(awk '{print $2}' "$STAMP" 2>/dev/null)"
fi
if [ "$FORCE" != "1" ] && [ -x .venv/bin/python ] \
   && [ "$CORE_HASH" = "$PREV_CORE" ]; then
  log "core requirements up to date — skipping pip"
else
  log "installing core requirements…"
  if ! "$VENV_PY" -m pip install -q -r requirements.txt; then
    warn "core dependency install FAILED — cannot start the app"
    exit 1
  fi
  log "core requirements installed ✓"
fi

# ── 3) directories + settings seed (foreground, instant) ──────────────
mkdir -p data data/history models
[ -f data/settings.json ] || echo '{"theme":"dark","haptics":true,"sound":true}' > data/settings.json

# ── 4) agent/AI deps + model download — BACKGROUND by default ─────────
run_ai_stages() {
  "$VENV_PY" -m pip install -q -r requirements-agents.txt \
    && "$VENV_PY" scripts/download_models.py
}
ai_state="skipped"
if [ "$FORCE" = "1" ] || [ "$AI_HASH" != "$PREV_AI" ] \
   || [ ! -f data/.ai_done ]; then
  if [ "$BLOCKING" = "1" ]; then
    log "installing AI/agent requirements + model (BLOCKING=1)…"
    if run_ai_stages > data/setup_ai.log 2>&1; then
      ai_state="installed"
      touch data/.ai_done
    else
      ai_state="failed (data/setup_ai.log)"
      warn "AI stage failed — core app unaffected"
    fi
  else
    # lock so parallel boots don't spawn duplicate builds
    if [ -f data/setup_ai.pid ] && kill -0 "$(cat data/setup_ai.pid)" 2>/dev/null; then
      ai_state="already building in background"
    else
      log "spawning background AI stage (llama-cpp build + model) → data/setup_ai.log"
      nohup bash -c "
        '$VENV_PY' -m pip install -q -r requirements-agents.txt \
          && '$VENV_PY' scripts/download_models.py \
          && touch data/.ai_done \
          || echo 'AI stage failed (non-fatal)' >> data/setup_ai.log
      " >> data/setup_ai.log 2>&1 &
      echo $! > data/setup_ai.pid
      ai_state="running in background (log: data/setup_ai.log)"
    fi
  fi
else
  ai_state="up to date"
fi

# ── 5) hourly learning cron (best-effort; in-process loop also exists) ─
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

# ── 6) stamp (core hash + AI hash) ────────────────────────────────────
echo "$CORE_HASH $AI_HASH $(date -u +%s)" > "$STAMP"

# ── 7) summary ────────────────────────────────────────────────────────
cat <<EOF
[setup] ═══════════ SETUP SUMMARY ($(( $(date +%s) - T0 ))s total) ═══════════
[setup] core deps      : ok (foreground)
[setup] agent/AI deps  : $ai_state
[setup] learning cron  : $CRON_STATUS
[setup] data dirs      : data/, data/history/, models/
[setup] next step      : ./run.sh   (or auto-started via scripts/auto_start.sh)
[setup] ══════════════════════════════════════════════════════════════
EOF
exit 0
