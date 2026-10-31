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
exec "$PYTHON_BIN" -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" --workers 1
