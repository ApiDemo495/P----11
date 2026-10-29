#!/usr/bin/env bash
# Production launcher for the prediction application.
# Single worker is required: the master clock, GRC state and prediction
# engine are in-process singletons.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
exec python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" --workers 1
