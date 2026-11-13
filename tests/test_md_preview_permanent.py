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


def test_autostart_is_devcontainer_hooks_not_a_task():
    # the folderOpen task made VS Code show a trust popup in every
    # new Codespace; start.py via devcontainer hooks replaced it
    tasks = json.loads((ROOT / ".vscode" / "tasks.json").read_text())
    auto = [t for t in tasks["tasks"]
            if t.get("runOptions", {}).get("runOn") == "folderOpen"]
    assert not auto
    fix = [t for t in tasks["tasks"]
           if "fix_markdown_preview.sh" in t.get("command", "")]
    assert fix, "a one-click Fix Markdown Preview task must exist"


def test_devcontainer_hooks_cover_attach_and_start():
    cfg = json.loads((ROOT / ".devcontainer" / "devcontainer.json").read_text())
    # postStart only fires on container start; postAttach covers every
    # re-attach (the gap that made auto-start look dead)
    for hook in ("onCreateCommand", "postCreateCommand",
                 "postStartCommand", "postAttachCommand"):
        assert cfg[hook] == "python3 start.py --detach"
    # agent deps + model are NOT auto-downloaded at boot (user order);
    # they download from the Agents tab button only
    assert "INSTALL_AGENTS" not in json.dumps(cfg)


def test_editor_extras_live_in_start_py_background():
    src = (ROOT / "start.py").read_text()
    assert "def editor_extras" in src
    assert "--install-extension" in src          # MPE/Markdown All-in-One
    assert "daemon=True" in src                  # thread, never gates boot
    assert "make_docs_html.py" in src            # docs regen off critical path


def test_fix_script_exists_and_is_valid():
    p = ROOT / "scripts" / "fix_markdown_preview.sh"
    assert p.exists()
    r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    src = p.read_text()
    assert "markdown-preview-enhanced" in src and "--install-extension" in src
    assert "@builtin markdown" in src   # prints the built-in re-enable cure
