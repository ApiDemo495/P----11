#!/usr/bin/env bash
# NEW: eternal supervisor — the server must never stay dead.
# Started once by auto_start.sh (idempotent via pidfile); every 30 s
# it health-checks the app and resurrects it after 3 consecutive
# failures (crash, OOM-kill, wedge). Log: data/supervisor.log.
set -uo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8000}"
PIDF="data/supervisor.pid"
mkdir -p data
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; then
  exit 0                      # already supervising
fi
echo $$ > "$PIDF"
log() { echo "$(date -Is) $*" >> data/supervisor.log; }
log "supervisor started (pid $$)"
fails=0
while true; do
  sleep 30
  if ! flock -n data/start.lock -c true 2>/dev/null; then
    continue   # a starter holds the lock (provisioning/booting) — hands off
  fi
  if curl -sf -m 5 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
    fails=0
  else
    fails=$((fails + 1))
    log "health failed ($fails/3)"
    if [ "$fails" -ge 3 ]; then
      log "app down — resurrecting via auto_start --restart"
      bash scripts/auto_start.sh --restart >> data/supervisor.log 2>&1 || true
      fails=0
    fi
  fi
done
