#!/usr/bin/env bash
# Regenerate the in-repo connectome bundle (run after a local build):
#   bash scripts/bundle_brain.sh
# Compresses data/connectome/flywire_783.npz -> assets/connectome/
# flywire_783.npz.gz (~62 MB, under GitHub's 100 MB file limit) so a
# fresh Codespace restores the brain from the repo with NO download.
set -eu
cd "$(dirname "$0")/.."
mkdir -p assets/connectome
gzip -9 -c data/connectome/flywire_783.npz > assets/connectome/flywire_783.npz.gz
ls -la assets/connectome/
