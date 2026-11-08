"""Everything automatic: code auto-update, auto-reload. Port stays PRIVATE."""
import pathlib

ROOT = pathlib.Path(__file__).parent.parent


def test_auto_start_updates_and_restarts_itself():
    sh = (ROOT / "scripts" / "auto_start.sh").read_text()
    assert "git pull -q --ff-only" in sh
    assert "git status --porcelain" in sh          # never touches a dirty tree
    assert "stop_app" in sh                        # restarts when the tip moves
    assert "make_port_public" not in sh            # never auto-public (user)


def test_health_reports_commit_and_client_reloads():
    assert '"commit": _COMMIT' in (ROOT / "app" / "main.py").read_text()
    js = (ROOT / "static" / "app.js").read_text()
    assert "S._commit" in js and "location.reload()" in js


def test_nothing_makes_the_port_public_automatically():
    assert not (ROOT / "scripts" / "make_port_public.sh").exists()
    for f in ("scripts/auto_start.sh", "scripts/supervise.sh", "run.sh"):
        assert "visibility" not in (ROOT / f).read_text()


def test_placeholder_binds_first_and_hooks_return_fast():
    ph = ROOT / "scripts" / "placeholder_page.py"
    assert ph.exists()
    src = ph.read_text()
    assert "http.server" in src and "import app" not in src  # stdlib only
    assert '"/healthz"' in src or "healthz" in src
    sh = (ROOT / "scripts" / "auto_start.sh").read_text()
    assert "flock" in sh                       # one starter at a time
    assert "placeholder_page.py" in sh         # placeholder before engine
    assert "--worker" in sh                    # slow work detached
    assert "setsid nohup" in sh                # survives the hook shell


def test_devcontainer_opens_editor_immediately():
    import json
    d = json.loads((ROOT / ".devcontainer" / "devcontainer.json").read_text())
    assert d["waitFor"] == "onCreateCommand"   # never stuck on provisioning
    for hook in ("onCreateCommand", "postCreateCommand",
                 "postStartCommand", "postAttachCommand"):
        assert "auto_start.sh" in d[hook]
        assert "|| true" in d[hook]     # a hook can never ⚠ the creation
    assert d["portsAttributes"]["8000"]["visibility"] == "private"
    assert (ROOT / ".github" / "workflows" / "boot.yml").exists()


def test_setup_can_never_gate_or_kill_the_boot():
    sh = (ROOT / "scripts" / "setup.sh").read_text()
    # marketplace installs are detached, never in the boot path
    assert "install_editor_extensions ) >> data/md_preview.log 2>&1 &" in sh
    assert "timeout 120 code --install-extension" in sh
    auto = (ROOT / "scripts" / "auto_start.sh").read_text()
    assert "attempting engine start anyway" in auto   # setup failure != dead port
    assert "cannot start" not in auto


def test_engine_crash_restores_placeholder_fast():
    auto = (ROOT / "scripts" / "auto_start.sh").read_text()
    assert "engine process died during boot" in auto
    assert auto.count("start_placeholder") >= 3   # boot, provision, crash paths


def test_boot_downloads_are_slim_and_deferred():
    import json
    d = json.loads((ROOT / ".devcontainer" / "devcontainer.json").read_text())
    assert "INSTALL_AGENTS" not in json.dumps(d)      # no forced AI build/download
    auto = (ROOT / "scripts" / "auto_start.sh").read_text()
    assert "sleep 600" in auto                        # backtest deferred past boot
    assert "launch_backtest_deferred" in auto
    bt = (ROOT / "scripts" / "run_backtest_once.sh").read_text()
    assert ".setup_stamp" in bt                       # never races provisioning
    assert ".backtest_running" in bt                  # no duplicate runners
    setup = (ROOT / "scripts" / "setup.sh").read_text()
    assert "nice -n 19 bash -c" in setup              # opt-in AI stage is niced



def test_boot_has_no_agent_download():
    auto = (ROOT / "scripts" / "auto_start.sh").read_text()
    assert "download_agent" not in auto        # agent is in-app only (old)
    assert not (ROOT / "scripts" / "download_agent.sh").exists()
    bt = (ROOT / "scripts" / "run_backtest_once.sh").read_text()
    assert "agent download" not in bt          # backtest: deferred only


def test_no_editor_warning_generators():
    import json
    # markdownlint stays installed (preview bundle) but our own docs
    # must not spam the Problems ⚠ counter
    cfg = json.loads((ROOT / ".markdownlint.json").read_text())
    for rule in ("MD013", "MD033", "MD026", "MD036", "MD040", "MD041"):
        assert cfg[rule] is False
    tasks = json.loads((ROOT / ".vscode" / "tasks.json").read_text())
    # no folderOpen auto task: it made VS Code show a trust popup in
    # EVERY new Codespace; devcontainer hooks do the auto-start
    assert not any(t.get("runOptions", {}).get("runOn") == "folderOpen"
                   for t in tasks["tasks"])
    import json as _j
    dc = _j.loads((ROOT / ".devcontainer" / "devcontainer.json").read_text())
    assert all("auto_start.sh" in dc[h] for h in
               ("onCreateCommand", "postCreateCommand",
                "postStartCommand", "postAttachCommand"))



def test_auto_update_survives_generated_docs_dirt():
    auto = (ROOT / "scripts" / "auto_start.sh").read_text()
    assert "grep -vx docs.html" in auto          # generated-only dirt discarded
    assert "auto-update skipped — dirty tree" in auto  # blockers are named
    docs = (ROOT / "docs.html").read_text()
    readme = (ROOT / "README.md").read_text()
    import re
    n = re.search(r"(\d+) tests", readme)
    assert n and (n.group(1) + " tests") in docs  # committed docs in sync
