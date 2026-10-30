#!/usr/bin/env bash
# Section 10 — hourly macro-learning cron for the background process.
# Add to crontab:  0 * * * *  /path/to/scripts/learning_cron.sh
# The engine already runs macro learning every 15 GRCs (~60 min) inside its
# own loop; this script is the external safety net that guarantees the cycle
# fires even if the app was restarted mid-window.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8017}"
URL="http://127.0.0.1:${PORT}/api/learning/macro"
LOG="data/learning_cron.log"
if curl -sf -X POST "$URL" -o /tmp/learning_macro_resp.json -m 60; then
  echo "$(date -u +%FT%TZ) macro ok" >>"$LOG"
else
  echo "$(date -u +%FT%TZ) macro FAILED (is the app running on :${PORT}?)" >>"$LOG"
  exit 1
fi
# keep the log small
tail -n 200 "$LOG" >"${LOG}.tmp" && mv "${LOG}.tmp" "$LOG"
