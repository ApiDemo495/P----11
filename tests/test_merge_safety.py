"""Regression coverage for branch-integration safety contracts."""
import math
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app import cost_gate
from app.engine import PredictionEngine


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("price,atr", [
    (103.0, 1.0),       # exactly three ATR is not strictly greater
    (104.0, 0.0),       # no measured ATR
    (104.0, math.nan),
    (math.nan, 1.0),    # non-finite quotes must not corrupt a lock
])
def test_emergency_never_reanchors_without_proven_structural_move(automatic, price, atr):
    lock = {"price": 100.0, "locked_at": 9000.0}
    engine = SimpleNamespace(
        news=SimpleNamespace(emergency_history=[], last_emergency={}),
        feed=SimpleNamespace(prices=lambda: {"BTC": price}),
        locks={("BTC", "tf1"): lock},
        build_ctx=lambda *args: None, _atr=lambda ctx: atr,
    )
    with patch("app.engine.time.time", return_value=10000.0):
        result = PredictionEngine.emergency_relock(engine, "BTC", automatic=automatic)
    assert not result["ok"]
    assert lock == {"price": 100.0, "locked_at": 9000.0}
    assert engine.news.emergency_history == []
    assert not getattr(engine, "_halt_until", {})


def test_rejected_sleeve_never_qualifies_even_with_favourable_numbers(monkeypatch):
    sleeve = dict(asset="BTC", tf="swing", status="rejected", gross_bp=50.0,
                  boot_t=3.0, profit_factor=2.0, n=300, folds="5/5",
                  trigger="test", execution="test", note="retracted")
    monkeypatch.setitem(cost_gate.SLEEVES, "retracted_test", sleeve)
    result = cost_gate.sleeve_gate("retracted_test", 8.0)
    assert result["state"] == "blocked"
    assert result["qualifies"] is False


@pytest.mark.parametrize("value,expected", [("sim", "sim"), (" BINANCE ", "binance"), ("auto", "auto")])
def test_feed_mode_environment_is_honoured(value, expected):
    env = dict(os.environ, FEED_MODE=value)
    result = subprocess.run([sys.executable, "-c", "from app.config import FEED_MODE; print(FEED_MODE)"],
                            env=env, text=True, capture_output=True, check=True)
    assert result.stdout.strip() == expected


def test_invalid_feed_mode_fails_loudly():
    env = dict(os.environ, FEED_MODE="simulated-typo")
    result = subprocess.run([sys.executable, "-c", "import app.config"],
                            env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "FEED_MODE must be one of" in result.stderr


def test_paper_book_never_fabricates_a_fill():
    """The rule this pins has always been "no invented fills".

    The book used to satisfy it by being frozen outright. It now satisfies it
    by only filling on a verified next-bar OPEN over a gap-free Binance tape
    (app/exec_live.py), and by reporting the precise fact that blocks a fill
    when one cannot be proven — so a pause still names its cause, and an
    existing position is still never marked against unverified tape.
    """
    class Book:
        def snapshot(self):
            return {"open_positions": [{"sleeve_id": "existing"}],
                    "closed_trades": 1}
        def manage(self, *args, **kwargs):
            raise AssertionError("must not mark or close unverified positions")
        def try_enter(self, *args, **kwargs):
            raise AssertionError("must not create unverified fills")

    # a feed that cannot price a certified fill: simulated tape
    class Feed:
        mode = "sim"
        def state(self, symbol):
            raise AssertionError("simulated tape must be refused before resampling")

    engine = SimpleNamespace(portfolio=Book(), feed=Feed())
    book = PredictionEngine._step_portfolio(
        engine, {"signals": [{"state": "long"}]})
    assert book["execution_status"] == "paused"
    # the reason is the specific fact, per asset — never a bare "paused"
    assert "simulated tape is excluded" in book["execution_reason"]
    assert set(book["paused_assets"]) == {"ETH", "BTC", "BNB", "PAXG"}
    # and the contract it would fill under is published either way
    assert "next 15m bar open" in book["execution_contract"]["fill"]
    assert book["entered_this_bar"] == []
    assert book["open_positions"] == [{"sleeve_id": "existing"}]


def test_incomplete_and_discontinuous_buckets_do_not_create_a_signal():
    from app.sleeves import compute_impulse_z, resample_bars

    times = list(range(0, 20 * 900 + 60, 60))
    closes = [100.0 + 0.01 * i + math.sin(i / 5) for i in range(len(times))]
    assert compute_impulse_z(times, closes, impulse_hours=1, std_window=10) is not None
    # A single missing minute in the last complete research window must not
    # turn 15m bars into a multi-period return while pretending continuity.
    i = times.index(18 * 900 + 5 * 60)
    del times[i]
    del closes[i]
    bt, _ = resample_bars(times, closes, 900)
    assert 18 * 900 not in bt
    assert compute_impulse_z(times, closes, impulse_hours=1, std_window=10) is None


def test_certified_trigger_rejects_simulated_and_mixed_source_tape():
    from app.sleeves import impulse_z_for
    times = list(range(0, 20 * 900 + 60, 60))
    closes = [100.0 + 0.01 * i + math.sin(i / 5) for i in range(len(times))]
    state = SimpleNamespace(times=times, closes=closes,
                            sources=["binance-rest"] * len(times))
    feed = SimpleNamespace(mode="binance", state=lambda symbol: state)
    z, _ = impulse_z_for(feed, "ETH", std_window=10, impulse_hours=1)
    assert z is not None
    feed.mode = "sim"
    assert impulse_z_for(feed, "ETH", std_window=10, impulse_hours=1) == (None, "warming_up")
    feed.mode = "binance"
    state.sources[-20] = "sim"
    assert impulse_z_for(feed, "ETH", std_window=10, impulse_hours=1) == (None, "warming_up")


def test_meta_evidence_constant_channels_stay_finite(tmp_path):
    import json
    import warnings
    from app.meta import CALIB_MIN, SOURCES, MetaCombiner

    meta = MetaCombiner(path=str(tmp_path / "meta.json"))
    meta.rows = [{"inputs": {key: (float(i % 3) if key == "score" else 0.0)
                              for key in SOURCES}, "up": i % 2}
                 for i in range(CALIB_MIN)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        evidence = meta.evidence()
    json.dumps(evidence, allow_nan=False)
    assert evidence["ready"]
    assert evidence["rho_bar"] == 0.0
    assert set(evidence["correlation_unavailable"]) == set(SOURCES) - {"score"}


def test_failed_provisioning_does_not_stamp_environment_fresh(tmp_path, monkeypatch):
    import start

    monkeypatch.setattr(start, "ROOT", str(tmp_path))
    monkeypatch.setattr(start, "DATA", str(tmp_path / "data"))
    monkeypatch.setattr(start, "STAMP", str(tmp_path / "stamp"))
    monkeypatch.setattr(start, "VENV_PY", sys.executable)
    monkeypatch.setattr(start, "_req_hash", lambda: "new-hash")
    monkeypatch.delenv("SKIP_SETUP", raising=False)
    (tmp_path / "stamp").write_text("old-hash 1\n")
    attempts = []

    def run(*args, **kwargs):
        attempts.append(1)
        return 1 if len(attempts) == 1 else 0

    monkeypatch.setattr(start, "_run", run)
    start.ensure_env()
    assert not (tmp_path / "stamp").exists()
    start.ensure_env()
    assert len(attempts) == 2
    assert (tmp_path / "stamp").read_text().startswith("new-hash ")
    start.ensure_env()
    assert len(attempts) == 2
