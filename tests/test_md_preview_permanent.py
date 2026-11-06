"""Permanent markdown-preview + auto-start fixes.

Bug history: 'markdown.showPreview not found' kept returning because
(a) the keybinding remap wrote only to ~/.config/Code/User — a path
Codespaces/Remote editors never read, (b) extension installs failed
silently into /dev/null, (c) auto-start only existed as devcontainer
lifecycle commands, which never fire in local VS Code.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_keybindings_land_in_codespaces_remote_user_dir():
    home = tempfile.mkdtemp()
    env = dict(os.environ, HOME=home, CODESPACES="true")
    script = ROOT / "scripts" / "ensure_md_keybindings.py"
    r = subprocess.run([sys.executable, str(script)], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    kb = Path(home) / ".vscode-remote" / "data" / "User" / "keybindings.json"
    assert kb.exists(), "remap must land where Codespaces actually reads"
    arr = json.loads(kb.read_text())
    cmds = {(x["key"], x["command"]) for x in arr}
    assert ("ctrl+shift+v", "markdown-preview-enhanced.openPreview") in cmds
    # idempotent: a second run adds nothing
    r2 = subprocess.run([sys.executable, str(script)], env=env,
                        capture_output=True, text=True)
    assert r2.returncode == 0
    assert json.loads(kb.read_text()) == arr


def test_keybindings_never_clobber_existing():
    home = tempfile.mkdtemp()
    user = Path(home) / ".vscode-remote" / "data" / "User"
    user.mkdir(parents=True)
    mine = [{"key": "ctrl+shift+v", "command": "workbench.action.terminal.focus",
             "when": "editorLangId == markdown"}]
    (user / "keybindings.json").write_text(json.dumps(mine))
    env = dict(os.environ, HOME=home, CODESPACES="true")
    subprocess.run([sys.executable,
                    str(ROOT / "scripts" / "ensure_md_keybindings.py")],
                   env=env, capture_output=True, text=True, check=True)
    arr = json.loads((user / "keybindings.json").read_text())
    assert mine[0] in arr                      # user's binding preserved
    assert any(x.get("command") == "markdown-preview-enhanced.openPreview"
               for x in arr)                   # ours added alongside


def test_folder_open_task_is_the_everywhere_autostart():
    tasks = json.loads((ROOT / ".vscode" / "tasks.json").read_text())
    auto = [t for t in tasks["tasks"]
            if t.get("runOptions", {}).get("runOn") == "folderOpen"]
    assert auto and "auto_start.sh" in auto[0]["command"]
    fix = [t for t in tasks["tasks"]
           if "fix_markdown_preview.sh" in t.get("command", "")]
    assert fix, "a one-click Fix Markdown Preview task must exist"


def test_devcontainer_attach_and_autodownload():
    cfg = json.loads((ROOT / ".devcontainer" / "devcontainer.json").read_text())
    # postStart only fires on container start; postAttach covers every
    # re-attach (the gap that made auto-start look dead)
    assert cfg["postAttachCommand"] == "bash scripts/auto_start.sh"
    assert cfg["postStartCommand"] == "bash scripts/auto_start.sh"
    # auto-download of agent deps + model is ON in the container
    # (still background/non-blocking inside setup.sh; the script's own
    # default stays opt-in for bare-metal runs)
    assert cfg["remoteEnv"]["INSTALL_AGENTS"] == "1"


def test_autostart_md_fix_is_backgrounded_and_logged():
    auto = (ROOT / "scripts" / "auto_start.sh").read_text()
    assert "( ensure_md_preview ) >/dev/null 2>&1 &" in auto   # real &
    assert "md_preview.log" in auto                            # not silent
    setup = (ROOT / "scripts" / "setup.sh").read_text()
    assert "md_preview.log" in setup
    assert "fix_markdown_preview.sh" in setup


def test_fix_script_exists_and_is_valid():
    p = ROOT / "scripts" / "fix_markdown_preview.sh"
    assert p.exists()
    r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    src = p.read_text()
    assert "markdown-preview-enhanced" in src and "--install-extension" in src
    assert "@builtin markdown" in src   # prints the built-in re-enable cure
