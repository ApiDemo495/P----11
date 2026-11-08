#!/bin/sh
# The whole boot system is start.py now (one Python file):
# placeholder -> provision -> import-check -> uvicorn -> watchdog.
exec python3 "$(dirname "$0")/start.py" --foreground
