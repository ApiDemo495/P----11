#!/usr/bin/env bash
# Production launcher — fully automated: bootstraps venv + all
# requirements (incl. AI/agent deps + RAM-tier model) when needed,
# then starts the app in the foreground.
# Single worker is required: the master clock, GRC state and prediction
# engine are in-process singletons.
#
#   ./run.sh                 auto-setup + start (PORT=8000 by default)
#   SKIP_SETUP=1 ./run.sh    start without re-checking dependencies
#   ./scripts/auto_start.sh  background auto-start (devcontainer postStart)
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8000}"

# ── automated bootstrap (fast no-op once stamped) ─────────────────────
if [ "${SKIP_SETUP:-0}" != "1" ]; then
  H=$(sha256sum requirements.txt 2>/dev/null | cut -d' ' -f1 || echo stale)
  S=$(cut -d' ' -f1 .setup_stamp 2>/dev/null || echo stale)
  V=ok; [ -x .venv/bin/python ] || V=missing
  if [ "$H" != "$S" ] || [ "$V" = "missing" ]; then
    bash scripts/setup.sh
  fi
fi

PYTHON_BIN=".venv/bin/python"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="python3"

# ── --supervise: single-instance lock + startup watchdog (advice §9/J-1) ──
# flock held by THIS parent on fd 9; the child runs with fd 9 CLOSED (the
# J-1 leak: a forked worker inheriting fd9 kept the lock forever and every
# crash-loop attempt printed 'already running'). Watchdog: if the app is
# not answering /api/health within 30s, kill the child and exit 75 —
# supervisor/health checks see a real failure, never a hung boot.
if [ "${1:-}" = "--supervise" ]; then
  exec 9> .app.lock
  if ! flock -n 9; then
    echo "run.sh --supervise: another instance holds .app.lock" >&2
    exit 75
  fi
  "$0" --run 9>&- &
  child=$!
  ready=0
  for _ in $(seq 1 60); do
    sleep 0.5
    if ! kill -0 "$child" 2>/dev/null; then
      wait "$child"; rc=$?
      echo "run.sh --supervise: child exited early rc=$rc" >&2
      exit "$rc"
    fi
    if curl -sf "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then
      ready=1; break
    fi
  done
  if [ "$ready" != "1" ]; then
    echo "run.sh --supervise: readiness timeout after 30s — exiting 75" >&2
    kill "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
    exit 75
  fi
  echo "run.sh --supervise: ready (child pid $child), holding lock"
  # §9 supervisor: health-check every 10s; six consecutive failures ->
  # TERM, KILL, relaunch (lock stays held by this parent on fd 9)
  fails=0
  while kill -0 "$child" 2>/dev/null; do
    sleep 10
    if curl -sf "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then
      fails=0
    else
      fails=$((fails + 1))
      if [ "$fails" -ge 6 ]; then
        echo "run.sh --supervise: 6 failed health checks — TERM/KILL/relaunch" >&2
        kill -TERM "$child" 2>/dev/null || true
        sleep 3
        kill -KILL "$child" 2>/dev/null || true
        "$0" --run 9>&- &
        child=$!
        fails=0
      fi
    fi
  done
  wait "$child"; exit $?
fi
if [ "${1:-}" = "--run" ]; then
  SUPERVISE=1 exec "$PYTHON_BIN" -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" --workers 1
fi
exec "$PYTHON_BIN" -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" --workers 1
