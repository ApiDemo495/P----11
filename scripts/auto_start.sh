#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# Idempotent auto-start supervisor — used by the devcontainer
# postStartCommand and safe to run any number of times.
#   1. health-check first: if the app is already up, exit in <1 s
#   2. runs scripts/setup.sh ONLY when the environment is stale
#      (AI/model stages are backgrounded inside setup.sh, never blocking)
#   3. starts uvicorn in the background and waits for /api/health
# Usage:  bash scripts/auto_start.sh [--stop|--restart]
# ══════════════════════════════════════════════════════════════════════
# markdown preview fix (idempotent, silent): the editor's built-in
# markdown.showPreview may be absent — Markdown Preview Enhanced bundles
# its own preview engine.
if command -v code >/dev/null 2>&1; then
  code --install-extension shd101wyy.markdown-preview-enhanced --force >/dev/null 2>&1 || true
  code --install-extension yzhang.markdown-all-in-one --force >/dev/null 2>&1 || true
fi
set -uo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"
HEALTH_URL="http://127.0.0.1:${PORT}/api/health"
PIDFILE="data/server.pid"
LOG="data/server.log"
ACTION="${1:-start}"
T0=$(date +%s)
log() { printf '[auto-start %4ss] %s\n' "$(( $(date +%s) - T0 ))" "$*"; }

stop_app() {
  if [ -f "$PIDFILE" ]; then
    pid="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null; sleep 1
      kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
      log "stopped pid $pid"
    fi
    rm -f "$PIDFILE"
  fi
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" 2>/dev/null && log "cleared port $PORT listener"
  fi
}

case "$ACTION" in
  --stop)    stop_app; log "stopped"; exit 0 ;;
  --restart) stop_app ;;
  --start|start|"") : ;;
esac

if curl -sf -m 3 "$HEALTH_URL" >/dev/null 2>&1; then
  log "app already healthy on :$PORT — nothing to do"
  exit 0
fi

# ── staleness check: only invoke setup when venv/stamp are missing ────
if [ "${SKIP_SETUP:-0}" != "1" ]; then
  need_setup=0
  [ -x .venv/bin/python ] || need_setup=1
  [ -f .setup_stamp ] || need_setup=1
  if [ "$need_setup" = "0" ] && [ -f .setup_stamp ]; then
    H=$(sha256sum requirements.txt 2>/dev/null | cut -d' ' -f1 || echo stale)
    S=$(awk '{print $1}' .setup_stamp)
    [ "$H" = "$S" ] || need_setup=1
  fi
  if [ "$need_setup" = "1" ]; then
    log "environment stale — running setup…"
    bash scripts/setup.sh || { log "setup failed — cannot start"; exit 1; }
  else
    log "environment fresh — skipping setup"
  fi
fi

# fast self-heal: missing real-brain data or docs.html (background, ~2s)
if [ ! -f data/brain_real.json ] && [ -x .venv/bin/python ]; then
  nohup .venv/bin/python scripts/fetch_real_brain.py >> data/setup.log 2>&1 &
fi
if [ -x .venv/bin/python ]; then
  .venv/bin/python scripts/make_docs_html.py >> data/setup.log 2>&1 || true
fi

VENV_PY=".venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY="python3"
mkdir -p data

log "starting uvicorn on 0.0.0.0:$PORT (log: $LOG)…"
nohup "$VENV_PY" -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" \
  --workers 1 >>"$LOG" 2>&1 &
echo $! > "$PIDFILE"

# wait for health (up to 60 s)
for i in $(seq 1 60); do
  if curl -sf -m 2 "$HEALTH_URL" >/dev/null 2>&1; then
    log "✓ app healthy on http://0.0.0.0:$PORT (pid $(cat "$PIDFILE")) — total $(( $(date +%s) - T0 ))s"
    exit 0
  fi
  sleep 1
done

log "✗ app did not become healthy in 60 s — last log lines:"
tail -n 25 "$LOG" 2>/dev/null
exit 1
