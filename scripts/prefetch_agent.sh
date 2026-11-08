#!/usr/bin/env bash
# Foreground, idempotent agent prefetch — for Codespaces PREBUILDS.
# A prebuild runs onCreateCommand/updateContentCommand once and
# snapshots the disk, so a model fetched HERE is already on disk in
# every Codespace created from that prebuild (downloaded once,
# zero bytes at start). To use it:
#   1. GitHub repo → Settings → Codespaces → Set up prebuild for
#      this branch (this step is yours — it uses Actions minutes).
#   2. Point the prebuild-captured command at this script, e.g.
#      "updateContentCommand": "bash scripts/prefetch_agent.sh"
# It is deliberately NOT wired into the default devcontainer
# hooks: without a prebuild it would block Codespace creation for
# the length of the llama.cpp build — the lag this design avoids.
# Safe to run anywhere else too: no-ops when the model is present
# and always exits 0.
exec bash "$(dirname "$0")/download_agent.sh"
