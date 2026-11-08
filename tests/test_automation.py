"""Everything automatic: code auto-update, public port, auto-reload."""
import pathlib

ROOT = pathlib.Path(__file__).parent.parent


def test_auto_start_updates_and_restarts_itself():
    sh = (ROOT / "scripts" / "auto_start.sh").read_text()
    assert "git pull -q --ff-only" in sh
    assert "git status --porcelain" in sh          # never touches a dirty tree
    assert "stop_app" in sh                        # restarts when the tip moves
    assert "make_port_public.sh" in sh


def test_health_reports_commit_and_client_reloads():
    assert '"commit": _COMMIT' in (ROOT / "app" / "main.py").read_text()
    js = (ROOT / "static" / "app.js").read_text()
    assert "S._commit" in js and "location.reload()" in js


def test_port_public_script_exists_and_never_fails():
    sh = (ROOT / "scripts" / "make_port_public.sh").read_text()
    assert "ports visibility" in sh and "exit 0" in sh


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
    assert d["portsAttributes"]["8000"]["visibility"] == "public"
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


def test_boot_order_agent_then_app_then_backtest():
    auto = (ROOT / "scripts" / "auto_start.sh").read_text()
    assert "launch_agent_download" in auto
    worker = auto.split("run_worker() {", 1)[1]
    assert worker.index("launch_agent_download") < worker.index("start_placeholder")
    dl = (ROOT / "scripts" / "download_agent.sh").read_text()
    assert "qwen2.5-coder-3b-instruct-q4_k_m.gguf" in dl   # specified best model
    assert ".setup_stamp" in dl          # waits for core pip — no pip race
    assert "setup_ai.pid" in dl          # single shared download lock
    bt = (ROOT / "scripts" / "run_backtest_once.sh").read_text()
    assert "agent download still running" in bt   # backtest runs LAST
    dm = (ROOT / "scripts" / "download_models.py").read_text()
    assert "--model" in dm
