#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# Idempotent auto-start supervisor — used by the devcontainer
# postStartCommand and safe to run any number of times.
#   1. runs scripts/setup.sh on first use (fast no-op when fresh)
#   2. starts uvicorn in the background if the app isn't already healthy
#   3. waits for /api/health and prints the public URL hint
# Usage:  bash scripts/auto_start.sh [--stop|--restart]
# ══════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"
HEALTH_URL="http://127.0.0.1:${PORT}/api/health"
PIDFILE="data/server.pid"
LOG="data/server.log"
ACTION="${1:-start}"

log() { printf '[auto-start] %s\n' "$*"; }

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
  # also clear any stray listener on the port (best-effort)
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

# first run → full bootstrap (fast no-op once stamped)
if [ "${SKIP_SETUP:-0}" != "1" ]; then
  log "checking setup…"
  bash scripts/setup.sh || { log "setup failed — cannot start"; exit 1; }
fi

VENV_PY=".venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY="python3"
mkdir -p data

log "starting uvicorn on 0.0.0.0:$PORT (log: $LOG)…"
nohup "$VENV_PY" -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" \
  --workers 1 >>"$LOG" 2>&1 &
echo $! > "$PIDFILE"

# wait for health (up to 90 s — first boot compiles models/caches)
for i in $(seq 1 90); do
  if curl -sf -m 2 "$HEALTH_URL" >/dev/null 2>&1; then
    log "✓ app is healthy on http://0.0.0.0:$PORT  (pid $(cat "$PIDFILE"))"
    exit 0
  fi
  sleep 1
done

log "✗ app did not become healthy in 90 s — last log lines:"
tail -n 25 "$LOG" 2>/dev/null
exit 1
