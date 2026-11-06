#!/usr/bin/env python3
"""Merge the markdown-preview keybinding remap into the USER keybindings.

The editor error "markdown.showPreview not found" means the built-in
markdown command is missing/disabled; Markdown Preview Enhanced
(auto-installed by setup/auto_start) ships its own preview. Once MPE is
present we map the muscle-memory shortcuts to its command:

    Ctrl+Shift+V  -> markdown-preview-enhanced.openPreview
    Ctrl+K V      -> markdown-preview-enhanced.openPreview

PERMANENT-FIX NOTES (why the old version silently did nothing):
the user keybindings file lives in a DIFFERENT place per editor —
  * Codespaces / VS Code Remote : ~/.vscode-remote/data/User/
  * local VS Code (Linux)       : ~/.config/Code/User/
  * Insiders / OSS / VSCodium   : ~/.config/Code - Insiders/User/ etc.
  * macOS                       : ~/Library/Application Support/Code/User/
  * Windows                     : %APPDATA%/Code/User/
Writing only to ~/.config/Code/User meant the remap landed where the
running editor never looks. This version detects every plausible user
dir, writes to the ones that belong to an installed editor (plus the
primary one for the current environment), and prints where it wrote.

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
PRODUCTS = ["Code", "Code - Insiders", "Code - OSS", "VSCodium",
            "VSCodium - Insiders", "Cursor"]


def candidate_user_dirs() -> list[pathlib.Path]:
    """Every plausible VS Code-family *User* dir, primary first."""
    home = pathlib.Path(os.path.expanduser("~"))
    dirs: list[pathlib.Path] = []
    # Codespaces / remote server: user data lives under ~/.vscode-remote
    remote = home / ".vscode-remote" / "data" / "User"
    if os.environ.get("CODESPACES") or (home / ".vscode-remote").exists():
        dirs.append(remote)
    dirs.append(remote)  # harmless if unused; only written when primary
    if os.environ.get("APPDATA"):                       # Windows
        for p in PRODUCTS:
            dirs.append(pathlib.Path(os.environ["APPDATA"]) / p / "User")
    for p in PRODUCTS:                                  # Linux
        dirs.append(home / ".config" / p / "User")
    for p in PRODUCTS:                                  # macOS
        dirs.append(home / "Library" / "Application Support" / p / "User")
    # de-dupe, keep order
    seen: set[str] = set()
    out = []
    for d in dirs:
        if str(d) not in seen:
            seen.add(str(d))
            out.append(d)
    return out


def merge_into(user_dir: pathlib.Path) -> bool:
    """Idempotently add the remap to one keybindings.json. True if the
    file now contains the remap."""
    p = user_dir / "keybindings.json"
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
        print(f"keybindings: +{len(added)} markdown-preview remap(s) -> {p}")
    return True


def main() -> int:
    dirs = candidate_user_dirs()
    # Write to every candidate whose editor install actually exists on
    # this machine (its product dir is present), plus the primary dir
    # for this environment so a fresh Codespace gets it before the
    # editor first writes its own files.
    written: list[str] = []
    for i, d in enumerate(dirs):
        product_dir_exists = d.parent.exists() or d.exists()
        if i == 0 or product_dir_exists:
            try:
                if merge_into(d):
                    written.append(str(d))
            except OSError as e:
                print(f"keybindings: skipped {d}: {e}")
    if written:
        print("keybindings: markdown preview remap active in: "
              + "; ".join(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())  # never raises past here
