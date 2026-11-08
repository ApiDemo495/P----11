#!/usr/bin/env bash
# Automatic REAL backtest: runs at most once per 24h, in the
# background, only where Binance is reachable. Writes the trade CSV
# + a plain-text report the app serves at /api/backtest.
# The sandbox that builds this repo cannot reach Binance — there the
# ccxt import/connect fails and this exits quietly (no fake numbers).
set -u
cd "$(dirname "$0")/.."
STAMP=data/.backtest_stamp
mkdir -p data backtest/out
if [ -f data/.backtest_running ] && kill -0 "$(cat data/.backtest_running 2>/dev/null)" 2>/dev/null; then
  echo "backtest: already running — skipped"; exit 0
fi
echo $$ > data/.backtest_running
trap 'rm -f data/.backtest_running' EXIT
if [ -f "$STAMP" ] && [ $(( $(date +%s) - $(stat -c %Y "$STAMP" 2>/dev/null || echo 0) )) -lt 86400 ]; then
  echo "backtest: fresh enough, skipped"; exit 0
fi
[ -f .setup_stamp ] || { echo "backtest: env not ready — skipped"; exit 0; }
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
"$PY" -c "import ccxt" 2>/dev/null || "$PY" -m pip install -q -r requirements-backtest.txt 2>/dev/null
"$PY" -c "import ccxt" 2>/dev/null || { echo "backtest: ccxt unavailable, skipped"; exit 0; }
"$PY" - <<'PYEOF'
import socket
socket.setdefaulttimeout(6)
try:
    socket.create_connection(("api.binance.com", 443)).close()
except OSError:
    print("backtest: Binance unreachable here, skipped (no fake numbers)")
    raise SystemExit(0)
from backtest import data as d
from backtest.predictor import AppPredictor
from backtest.runner import run_mode_a, trades_to_csv
from backtest.analytics import by_asset, metrics
from backtest.report import comparison_report
from backtest.replay import load_filtered
series = {s: d.fetch_ccxt(s, "15m", limit=1500) for s in d.SYMBOLS}
trades = run_mode_a(series, AppPredictor())
trades_to_csv(trades, "backtest/out/mode_a_trades.csv")
rep = comparison_report(trades, load_filtered())
open("backtest/out/report.txt", "w").write(
    "REAL BACKTEST — Binance 15m via CCXT, app logic, friction applied\n" + rep)
print(rep)
PYEOF
[ $? -eq 0 ] && touch "$STAMP" && echo "backtest: done $(date -u +%FT%TZ)"
