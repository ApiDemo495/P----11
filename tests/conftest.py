import asyncio

import pytest

pytest_plugins = ("pytest_asyncio",)


def run_sync(coro):
    """Run a coroutine on a usable event loop.

    TestClient/anyio close the loop they used and clear it as *current*;
    older helpers then crash with "There is no current event loop" when
    they run after an HTTP test. Never assume get_event_loop() succeeds.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Persisted runtime state must not leak between test sessions
# ---------------------------------------------------------------------------
# The engine loads its learned state from ``data/`` at start-up
# (app/config.py: LEARN_STATE_FILE, PHYSICS_STATE_FILE, locks, genesis ...)
# and test runs WRITE it back. The directory is git-ignored, so a dirty copy
# survives from one run to the next and silently changes what the engine
# computes: three exact-value TP-ladder tests and one SL safe-branch test
# passed on a clean checkout and failed on the second run, with no code
# change at all. A test suite whose result depends on the previous test run
# is not a test suite.
#
# Snapshot the directory for the session and restore it at the end, so every
# run starts from the same state and leaves the working tree as it found it.
import os
import shutil
import tempfile

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolate_lock_file(tmp_path_factory):
    """Keep manufactured locks out of data/locks.json.

    Engines load persisted locks at start-up. Within one session a lock
    written by an earlier test (old price, old levels) was inherited by the
    next engine, whose fresh cycle then saw TP1 already satisfied — two
    lock-protocol tests failed in the full run and passed in isolation.
    """
    path = tmp_path_factory.mktemp("locks") / "locks.json"
    old = os.environ.get("PRED_LOCKS_PATH")
    os.environ["PRED_LOCKS_PATH"] = str(path)
    yield path
    if old is None:
        os.environ.pop("PRED_LOCKS_PATH", None)
    else:
        os.environ["PRED_LOCKS_PATH"] = old


@pytest.fixture(autouse=True)
def _fresh_locks(_isolate_lock_file):
    """Every test starts with no inherited contract."""
    try:
        os.remove(_isolate_lock_file)
    except OSError:
        pass
    yield


@pytest.fixture(autouse=True)
def _isolate_pred_journal(monkeypatch, tmp_path):
    """A prediction seal written by one test must not make the next test's
    analyse take the duplicate-window path.

    PredictionJournal reads its default path from a module global captured
    at import, so pointing the env var at a tmp file is not enough — patch
    the global itself. Without this, an ETH|tf2 lock sealed by an earlier
    test made test_cognition's analyse skip its ledger write (session run
    failed where the file-only run passed).
    """
    from app import predlock
    monkeypatch.setattr(predlock, "JOURNAL_FILE",
                        str(tmp_path / "prediction_locks.json"))


@pytest.fixture(scope="session", autouse=True)
def _isolate_persisted_state():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    live = os.path.join(root, "data")
    backup = None
    if os.path.isdir(live):
        backup = tempfile.mkdtemp(prefix="data_snapshot_")
        shutil.copytree(live, os.path.join(backup, "data"))
    # Start the session from the state a fresh clone has (data/ is
    # git-ignored, so "absent" IS the canonical starting point). Without
    # this, the first run of a session behaves differently from the second
    # because the engine adopts state written by earlier tests in the run.
    shutil.rmtree(live, ignore_errors=True)
    yield
    if backup is not None:
        shutil.rmtree(live, ignore_errors=True)
        shutil.copytree(os.path.join(backup, "data"), live)
        shutil.rmtree(backup, ignore_errors=True)
    else:
        shutil.rmtree(live, ignore_errors=True)
