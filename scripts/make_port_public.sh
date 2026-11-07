#!/usr/bin/env bash
# Force the app port PUBLIC — idempotent, best-effort, always exits 0.
# Called by auto_start.sh on EVERY start/attach (including the
# already-healthy early-exit path, which used to skip visibility and
# leave the port Private -> GitHub auth wall breaks WebSockets).
# Result is logged to data/port_visibility.log so doctor.sh can show it.
set -uo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8000}"
mkdir -p data
LOGF="data/port_visibility.log"
stamp() { echo "$(date -Is) $*" >> "$LOGF"; }

if [ -z "${CODESPACE_NAME:-}" ]; then
  stamp "not a Codespace (CODESPACE_NAME unset) — nothing to do"
  exit 0
fi
if ! command -v gh >/dev/null 2>&1; then
  stamp "gh CLI not found — set port $PORT to Public once in the Ports tab"
  exit 0
fi
for i in 1 2 3; do
  if gh codespace ports visibility "$PORT:public" -c "$CODESPACE_NAME" >/dev/null 2>&1; then
    stamp "port $PORT set PUBLIC (attempt $i)"
    exit 0
  fi
  sleep 2
done
stamp "FAILED to set port $PORT public after 3 attempts — Ports tab: right-click $PORT -> Port Visibility -> Public"
exit 0
