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

# ── editor markdown preview fix (never blocks startup) ─────────────────
# "markdown.showPreview not found" = no built-in markdown in the editor.
# Markdown Preview Enhanced bundles its OWN preview engine; when it is
# present we also remap Ctrl+Shift+V / Ctrl+K V to MPE's command, so the
# built-in shortcut no longer errors. Everything below runs in the
# BACKGROUND — a slow marketplace used to add seconds to EVERY boot.
ensure_md_preview() {
  # failures are LOGGED (data/md_preview.log) — the old version sent
  # everything to /dev/null, so a broken install looked identical to a
  # working one and the error kept coming back with no trace.
  mkdir -p data
  command -v code >/dev/null 2>&1 || {
    echo "no 'code' CLI on PATH — editor extensions are installed by the" \
         "devcontainer (customizations.vscode.extensions) or run" \
         "scripts/fix_markdown_preview.sh from an editor terminal" \
         >> data/md_preview.log; return 0; }
  _have=$(code --list-extensions 2>/dev/null \
            | grep -ix "shd101wyy.markdown-preview-enhanced" || true)
  if [ -z "$_have" ]; then
    code --install-extension shd101wyy.markdown-preview-enhanced \
      >> data/md_preview.log 2>&1 || true
    code --list-extensions 2>/dev/null \
      | grep -qix "shd101wyy.markdown-preview-enhanced" || {
        echo "MPE install did not land — see scripts/fix_markdown_preview.sh" \
          >> data/md_preview.log; return 0; }
  fi
  # merge user keybindings idempotently (never clobber what exists);
  # the script detects the REAL user dir (Codespaces ~/.vscode-remote,
  # local ~/.config/Code, macOS, Windows) — see its docstring.
  command -v python3 >/dev/null 2>&1 || return 0
  python3 scripts/ensure_md_keybindings.py >> data/md_preview.log 2>&1 || true
}
# spawn immediately, ACTUALLY in the background (&): runs while
# setup/boot proceed in parallel
( ensure_md_preview ) >/dev/null 2>&1 &

# ── auto-update (no commands, ever): a clean tree fast-forwards to
# the branch tip; if the commit moved, the app restarts on new code.
# A dirty tree is never touched. Log: data/auto_update.log ──
mkdir -p data
{
  if command -v git >/dev/null 2>&1 && [ -z "$(git status --porcelain 2>/dev/null)" ]; then
    _br="$(git symbolic-ref --short HEAD 2>/dev/null || true)"
    if [ -n "$_br" ]; then
      _before="$(git rev-parse HEAD 2>/dev/null || true)"
      if git fetch -q origin "$_br" 2>>data/auto_update.log          && git pull -q --ff-only origin "$_br" 2>>data/auto_update.log; then
        _after="$(git rev-parse HEAD 2>/dev/null || true)"
        if [ -n "$_before" ] && [ "$_before" != "$_after" ]; then
          log "code updated ${_before} -> ${_after} — restarting app"
          stop_app
        fi
      fi
    fi
  fi
} || true

# port visibility on EVERY invocation — even (especially) when the
# app is already healthy; the old order skipped this on early exit
bash scripts/make_port_public.sh || true

# the supervisor keeps the app alive from here on (crash/OOM/wedge);
# started before the health early-exit so it runs in every path
nohup bash scripts/supervise.sh >/dev/null 2>&1 &

# real backtest, at most daily, fully in the background (no-op where
# Binance is unreachable — never fakes numbers)
nohup bash scripts/run_backtest_once.sh >>data/backtest.log 2>&1 &

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
    log "environment stale — running setup… (niced)"
    nice -n 10 bash scripts/setup.sh || { log "setup failed — cannot start"; exit 1; }
  else
    log "environment fresh — skipping setup"
  fi
fi

# fast self-heal: missing real-brain data or docs.html (background, ~2s)
# stats already ship in assets/ (tracked) — only fetch when BOTH missing
if [ ! -f data/brain_real.json ] && [ ! -f assets/brain_real.json ] \
   && [ -x .venv/bin/python ]; then
  nohup .venv/bin/python scripts/fetch_real_brain.py >> data/setup.log 2>&1 &
fi
# docs.html regen is NOT on the boot path (background, best-effort)
if [ -x .venv/bin/python ]; then
  ( .venv/bin/python scripts/make_docs_html.py >> data/setup.log 2>&1 || true ) &
fi

VENV_PY=".venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY="python3"
mkdir -p data

log "starting uvicorn on 0.0.0.0:$PORT (log: $LOG)…"
nohup "$VENV_PY" -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" \
  --workers 1 >>"$LOG" 2>&1 &
echo $! > "$PIDFILE"

# wait for health (up to 240 s — a cold boot restores the bundled
# brain and runs the initial analysis before uvicorn serves)
for i in $(seq 1 240); do
  if curl -sf -m 2 "$HEALTH_URL" >/dev/null 2>&1; then
    log "✓ app healthy on http://0.0.0.0:$PORT (pid $(cat "$PIDFILE")) — total $(( $(date +%s) - T0 ))s"
    exit 0
  fi
  sleep 1
done

log "✗ app did not become healthy in 240 s — last log lines:"
tail -n 25 "$LOG" 2>/dev/null
exit 1
