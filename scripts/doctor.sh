#!/usr/bin/env bash
# NEW: one-command diagnosis for "the page isn't working" in a Codespace.
#   bash scripts/doctor.sh
# Checks process, port, health, chart link, log tail.
set -uo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8000}"
echo "── doctor: $(date) ─────────────────────────────"
if [ -f data/server.pid ] && kill -0 "$(cat data/server.pid)" 2>/dev/null; then
  echo "✓ server process alive (pid $(cat data/server.pid))"
else
  echo "✗ server process NOT running → python3 start.py --detach"
fi
curl -sf -m 3 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1 \
  && echo "✓ /api/health answers on :$PORT" \
  || echo "✗ /api/health silent — server still booting or wedged (see log below)"
curl -sf -m 5 "http://127.0.0.1:$PORT/api/chart-link" 2>/dev/null \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('✓ chart link:', {k: (r['delta_bps'], r['ok']) for k, r in d['assets'].items()})" 2>/dev/null \
  || echo "· chart-link not up yet"
if command -v gh >/dev/null 2>&1 && [ -n "${CODESPACE_NAME:-}" ]; then
  gh codespace ports -c "$CODESPACE_NAME" 2>/dev/null | grep -E "^\s*$PORT\b" || true
fi
echo "── port visibility ─────────────────────────────"
echo "private by design (auto-public removed by user request)"
echo "── last server log lines ───────────────────────"
tail -n 15 data/server.log 2>/dev/null || echo "(no data/server.log)"
