#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# One starter for every hook (postCreate/postStart/postAttach) and
# for the supervisor. Design (docs11-codespaces.md): SOMETHING
# listens on the port within seconds, always — a stdlib placeholder
# first, the real engine after provisioning. Hooks return in <5 s;
# a detached, flock-serialized worker does the slow work.
#   hook mode (default): lock -> healthy? exit : placeholder + worker
#   --worker:  provision (if stale) -> handover -> uvicorn -> health
#   --restart: stop, then worker flow inline (supervisor path)
#   --stop:    stop engine + placeholder
# Logs: data/auto_start.log, data/server.log, data/supervisor.log
# ══════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"
HEALTH_URL="http://127.0.0.1:${PORT}/api/health"
PIDFILE="data/server.pid"
PHPIDFILE="data/placeholder.pid"
LOG="data/server.log"
LOCK="data/start.lock"
ACTION="${1:-start}"
T0=$(date +%s)
mkdir -p data
log() { printf '[auto-start %4ss] %s\n' "$(( $(date +%s) - T0 ))" "$*" | tee -a data/auto_start.log; }

healthy() { curl -sf -m 3 "$HEALTH_URL" >/dev/null 2>&1; }

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

start_placeholder() {
  if healthy; then return 0; fi
  if curl -sf -m 2 "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
    return 0                                   # placeholder already up
  fi
  if [ -f "$PHPIDFILE" ] && kill -0 "$(cat "$PHPIDFILE" 2>/dev/null)" 2>/dev/null; then
    return 0
  fi
  setsid nohup python3 scripts/placeholder_page.py \
    >>data/placeholder.log 2>&1 9>&- &
  echo $! > "$PHPIDFILE"
  for _ in 1 2 3 4 5; do
    curl -sf -m 1 "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1 && break
    sleep 1
  done
  log "placeholder serving on :$PORT (pid $(cat "$PHPIDFILE"))"
}

stop_placeholder() {
  if [ -f "$PHPIDFILE" ]; then
    pid="$(cat "$PHPIDFILE" 2>/dev/null || true)"
    [ -n "$pid" ] && kill "$pid" 2>/dev/null
    rm -f "$PHPIDFILE"
  fi
}

# ── editor markdown preview fix (background, never blocks) ───────────
ensure_md_preview() {
  mkdir -p data
  command -v code >/dev/null 2>&1 || {
    echo "no 'code' CLI — extensions come from the devcontainer" \
      >> data/md_preview.log; return 0; }
  _have=$(code --list-extensions 2>/dev/null \
            | grep -ix "shd101wyy.markdown-preview-enhanced" || true)
  if [ -z "$_have" ]; then
    code --install-extension shd101wyy.markdown-preview-enhanced \
      >> data/md_preview.log 2>&1 || true
  fi
  command -v python3 >/dev/null 2>&1 || return 0
  python3 scripts/ensure_md_keybindings.py >> data/md_preview.log 2>&1 || true
}

run_worker() {
  ( ensure_md_preview ) >/dev/null 2>&1 &

  # ── auto-update: clean tree fast-forwards; dirty tree untouched ──
  {
    if command -v git >/dev/null 2>&1 && [ -z "$(git status --porcelain 2>/dev/null)" ]; then
      _br="$(git symbolic-ref --short HEAD 2>/dev/null || true)"
      if [ -n "$_br" ]; then
        _before="$(git rev-parse HEAD 2>/dev/null || true)"
        if git fetch -q origin "$_br" 2>>data/auto_update.log \
           && git pull -q --ff-only origin "$_br" 2>>data/auto_update.log; then
          _after="$(git rev-parse HEAD 2>/dev/null || true)"
          if [ -n "$_before" ] && [ "$_before" != "$_after" ]; then
            log "code updated ${_before} -> ${_after} — restarting app"
            stop_app
          fi
        fi
      fi
    fi
  } || true

  bash scripts/make_port_public.sh || true
  setsid nohup bash scripts/supervise.sh >/dev/null 2>&1 9>&- &
  setsid nohup bash scripts/run_backtest_once.sh >>data/backtest.log 2>&1 9>&- &

  if healthy; then
    stop_placeholder
    log "app already healthy on :$PORT — nothing to do"
    return 0
  fi

  start_placeholder

  # ── provision only when stale ──
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
      log "environment stale — running setup… (niced)"
      if ! nice -n 10 bash scripts/setup.sh >>data/setup.log 2>&1; then
        # NON-FATAL (docs11 cause 1): a failed optional/pip step must
        # never leave nothing listening. Try the engine anyway; if
        # it cannot import, the placeholder returns with the logs.
        log "WARN: setup failed — attempting engine start anyway (see data/setup.log)"
      fi
    else
      log "environment fresh — skipping setup"
    fi
  fi

  if [ ! -f data/brain_real.json ] && [ ! -f assets/brain_real.json ] \
     && [ -x .venv/bin/python ]; then
    setsid nohup .venv/bin/python scripts/fetch_real_brain.py >> data/setup.log 2>&1 9>&- &
  fi
  if [ -x .venv/bin/python ]; then
    ( .venv/bin/python scripts/make_docs_html.py >> data/setup.log 2>&1 || true ) &
  fi

  VENV_PY=".venv/bin/python"
  [ -x "$VENV_PY" ] || VENV_PY="python3"

  stop_placeholder
  log "starting uvicorn on 0.0.0.0:$PORT (log: $LOG)…"
  setsid nohup "$VENV_PY" -m uvicorn app.main:app --host 0.0.0.0 \
    --port "$PORT" --workers 1 >>"$LOG" 2>&1 9>&- &
  echo $! > "$PIDFILE"

  for _ in $(seq 1 240); do
    if healthy; then
      log "✓ app healthy on http://0.0.0.0:$PORT (pid $(cat "$PIDFILE")) — total $(( $(date +%s) - T0 ))s"
      bash scripts/make_port_public.sh || true
      return 0
    fi
    sleep 1
  done

  log "✗ app did not become healthy in 240 s — placeholder back up with logs"
  tail -n 25 "$LOG" 2>/dev/null
  start_placeholder
  return 1
}

case "$ACTION" in
  --stop)
    exec 9>"$LOCK"; flock 9
    stop_app; stop_placeholder; log "stopped"; exit 0 ;;
  --restart)
    exec 9>"$LOCK"; flock 9          # serialize with any running worker
    stop_app; stop_placeholder
    run_worker; exit $? ;;
  --worker)
    exec 9>"$LOCK"; flock 9          # one worker at a time, hooks queue
    run_worker; exit $? ;;
  --start|start|--attach|--provision|"")
    exec 9>"$LOCK"
    if ! flock -n 9; then
      log "another starter is already working — exiting"
      exit 0
    fi
    if healthy; then
      stop_placeholder
      bash scripts/make_port_public.sh || true
      setsid nohup bash scripts/supervise.sh >/dev/null 2>&1 9>&- &
      log "app already healthy on :$PORT — nothing to do"
      exit 0
    fi
    start_placeholder
    setsid nohup bash scripts/auto_start.sh --worker \
      >>data/auto_start.log 2>&1 9>&- &
    log "worker launched — placeholder is serving on :$PORT meanwhile"
    exit 0 ;;
  *)
    log "unknown action: $ACTION"; exit 2 ;;
esac
