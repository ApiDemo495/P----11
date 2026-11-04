#!/usr/bin/env python3
"""Merge the markdown-preview keybinding remap into the USER keybindings.

The editor error "markdown.showPreview not found" means the built-in
markdown command is missing; Markdown Preview Enhanced (auto-installed
by setup/auto_start) ships its own preview. Once MPE is present we map
the muscle-memory shortcuts to its command:

    Ctrl+Shift+V  -> markdown-preview-enhanced.openPreview
    Ctrl+K V      -> markdown-preview-enhanced.openPreview

Rules: idempotent, never clobber existing bindings, keep a .bak if the
file is not valid JSON, exit 0 always (this must never break a boot).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

TARGET_CMD = "markdown-preview-enhanced.openPreview"
KEYS = [
    {"key": "ctrl+shift+v", "command": TARGET_CMD,
     "when": "editorLangId == markdown"},
    {"key": "ctrl+k v", "command": TARGET_CMD,
     "when": "editorLangId == markdown"},
]


def main() -> int:
    p = (pathlib.Path(os.path.expanduser("~"))
         / ".config" / "Code" / "User" / "keybindings.json")
    arr: list = []
    if p.exists():
        try:
            txt = p.read_text().strip()
            arr = json.loads(txt) if txt else []
            if not isinstance(arr, list):
                raise ValueError("not a list")
        except Exception:
            try:
                p.rename(p.with_suffix(".json.bak"))
            except OSError:
                pass
            arr = []
    have = {(x.get("key"), x.get("command"))
            for x in arr if isinstance(x, dict)}
    added = [k for k in KEYS if (k["key"], k["command"]) not in have]
    if added:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(arr + added, indent=2) + "\n")
        print(f"keybindings: +{len(added)} markdown-preview remap(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # never raises past here
