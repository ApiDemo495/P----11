#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# Permanent fix for:  Could not open markdown preview:
#                     command 'markdown.showPreview' not found
#
# WHY IT HAPPENS: markdown.showPreview belongs to VS Code's BUILT-IN
# "Markdown Language Features" extension. When that built-in is
# disabled/absent in your editor profile, the command does not exist —
# no workspace file can resurrect it. This script applies the durable
# two-part cure:
#   1. installs Markdown Preview Enhanced (ships its OWN preview
#      engine, independent of the built-in), and
#   2. remaps Ctrl+Shift+V / Ctrl+K V to MPE's command in the REAL
#      user keybindings file for THIS editor (Codespaces, local,
#      macOS, Windows — detected, not guessed).
# Run it ONCE from a terminal inside your editor:
#     bash scripts/fix_markdown_preview.sh
# It is idempotent — re-running changes nothing.
# ══════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "$0")/.."

CODE=""
for c in code code-insiders codium cursor; do
  command -v "$c" >/dev/null 2>&1 && { CODE="$c"; break; }
done
if [ -z "$CODE" ]; then
  echo "✗ no editor CLI (code/code-insiders/codium/cursor) on PATH."
  echo "  Run this from a terminal INSIDE VS Code / Codespaces, or install"
  echo "  'Markdown Preview Enhanced' from the Extensions panel manually."
  exit 1
fi
echo "editor CLI: $CODE"

fail=0
for ext in shd101wyy.markdown-preview-enhanced yzhang.markdown-all-in-one; do
  if "$CODE" --list-extensions 2>/dev/null | grep -qix "$ext"; then
    echo "✓ already installed: $ext"
  elif "$CODE" --install-extension "$ext"; then
    echo "✓ installed: $ext"
  else
    echo "✗ install failed: $ext (check network / marketplace access)"
    fail=1
  fi
done

# verify MPE really landed before touching keybindings
if "$CODE" --list-extensions 2>/dev/null \
     | grep -qix "shd101wyy.markdown-preview-enhanced"; then
  python3 scripts/ensure_md_keybindings.py || fail=1
  echo "✓ Ctrl+Shift+V / Ctrl+K V now open Markdown Preview Enhanced"
else
  echo "✗ MPE not present after install — keybinding remap skipped"
  fail=1
fi

cat <<'EOF'

If Ctrl+Shift+V still errors after a reload, the built-in itself is
disabled — re-enable it (the only cure for the built-in command):
  Extensions panel → search "@builtin markdown" →
  "Markdown Language Features" → Enable  →  reload the window.
Either path works: MPE via the remapped shortcut, or the restored
built-in. The app's own docs also render without any extension —
open docs.html, or run the app and visit /docs.
EOF
exit "$fail"
