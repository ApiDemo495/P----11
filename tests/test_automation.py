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
