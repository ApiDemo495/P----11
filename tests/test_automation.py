"""The boot system is ONE Python file now: start.py.

The old bash chain (auto_start.sh + setup.sh + supervise.sh +
placeholder_page.py + flock + curl) was deleted on user order
after it kept wedging in real Codespaces. These tests pin the
new contract so it cannot regress:
  - devcontainer hooks call only `python3 start.py --detach`
  - no curl, no flock, no editor CLI on the boot path
  - placeholder first, import checked before it stops, watchdog
  - old chain files are gone
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).parent.parent


def _start_src() -> str:
    return (ROOT / "start.py").read_text()


def test_old_bash_chain_is_deleted():
    for f in ("scripts/auto_start.sh", "scripts/supervise.sh",
              "scripts/placeholder_page.py", "scripts/setup.sh",
              "scripts/http_check.py"):
        assert not (ROOT / f).exists(), f


def test_start_py_is_stdlib_only_and_self_contained():
    src = _start_src()
    assert "ThreadingHTTPServer" in src            # placeholder
    assert "urllib.request" in src                 # health, no curl
    assert '"curl"' not in src and "curl -" not in src  # never invoked
    assert "flock(" not in src and "fcntl" not in src  # no lock files
    assert "import app.main" in src                # import pre-check
    assert "--detach" in src and "--stop" in src and "--status" in src
    # provisioning bounded, pip bounded
    assert '"--retries", "2", "--timeout", "30"' in src
    assert 'timeout=timeout_s' in src or "subprocess.run" in src


def test_placeholder_answers_before_engine():
    """Unit-level: the placeholder handler serves /, /healthz and
    honest 503s for /api/* without any venv."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("start_mod",
                                                  ROOT / "start.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    srv = mod._PlaceholderServer()
    # bind on an ephemeral port for the test
    import socket
    from http.server import ThreadingHTTPServer
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), mod._Placeholder)
    port = httpd.server_address[1]
    import threading
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/",
                                    timeout=3) as r:
            assert r.status == 200
            assert b"starting" in r.read()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz",
                                    timeout=3) as r:
            assert json.load(r)["stage"]
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health",
                                   timeout=3)
            raise AssertionError("api must be honest 503 pre-engine")
        except urllib.error.HTTPError as e:
            assert e.code == 503
    finally:
        httpd.shutdown()


def test_devcontainer_hooks_are_start_py_only():
    d = json.loads((ROOT / ".devcontainer" / "devcontainer.json")
                   .read_text())
    assert d["waitFor"] == "onCreateCommand"   # editor never blocked
    for hook in ("onCreateCommand", "postCreateCommand",
                 "postStartCommand", "postAttachCommand"):
        assert d[hook] == "python3 start.py --detach || true"
    # user order: the port is ALWAYS public so the -8000 URL works
    # in any browser tab (private ports 502 outside the editor auth)
    assert d["portsAttributes"]["8000"]["visibility"] == "public"
    assert "protocol" not in d["portsAttributes"]["8000"]  # plain HTTP app
    assert d["portsAttributes"]["8000"]["onAutoForward"] == "openBrowser"
    assert (ROOT / ".github" / "workflows" / "boot.yml").exists()
    assert (ROOT / "run.sh").read_text().count("start.py") >= 1


def test_no_folderopen_task_popup():
    tasks = json.loads((ROOT / ".vscode" / "tasks.json").read_text())
    assert not any(t.get("runOptions", {}).get("runOn") == "folderOpen"
                   for t in tasks["tasks"])


def test_health_reports_commit_and_client_reloads():
    assert '"commit": _COMMIT' in (ROOT / "app" / "main.py").read_text()
    js = (ROOT / "static" / "app.js").read_text()
    assert "S._commit" in js and "location.reload()" in js


def test_port_made_public_from_inside():
    src = _start_src()
    assert "def make_port_public" in src
    assert '"ports", "visibility"' in src
    assert "CODESPACE_NAME" in src       # no-op outside Codespaces
    assert "ensure_md_keybindings.py" in src   # orphan relinked


def test_watchdog_and_deferred_backtest_exist():
    src = _start_src()
    assert "fails >= 12" in src                    # hung engine restarted
    assert "run_backtest_once.sh" in src           # daily backtest kept
    assert "time.sleep(600)" in src                # …deferred past boot
    assert "stop_engine(_child)" in src
