"""Deep additions: Wilson intervals + expectancy on probabilities,
probability-forecast auditing, feed-staleness trading gate, ledger
expectancy. The theme: every number ships with its uncertainty, and
the forecast system itself is graded."""
from pathlib import Path

from app import trading
from app.ledger import SettlementLedger
from tests.contract_fixtures import locked_contract_signal

ROOT = Path(__file__).resolve().parent.parent


def test_wilson_and_ev_in_branch_payload():
    src = (ROOT / "app" / "engine.py").read_text()
    i = src.index("def branch_stats")
    seg = src[i:i + 16000]
    assert "_wilson(_tpn, _n)" in seg and '"ev_bps"' in seg
    assert "prob_audit_summary" in src
    assert '"prob_audit": self.prob_audit_summary()' in src


def test_stale_tape_refused_by_trading_gate():
    import time
    cfg = {"enabled": True, "testnet": True, "api_key": "k",
           "api_secret": "s", "max_order_usdt": 25, "max_orders_day": 4,
           "armed_until": time.time() + 600}
    sig = locked_contract_signal(entry=100.0, tp_bps=200.0,
                                 sl_bps=100.0, cycle_id="deep-audit",
                                 feed_age_s=45.0)
    ok, why = trading.gate(cfg, sig, set(), 0)
    assert not ok and "stale tape" in why
    sig["feed_age_s"] = 1.0
    assert trading.gate(cfg, sig, set(), 0)[0]


def test_feed_stamps_last_tick():
    src = (ROOT / "app" / "feed.py").read_text()
    assert src.count("st.last_tick = time.time()") >= 3
    assert '"last_tick"' in src


def test_ledger_expectancy_and_payoff(tmp_path):
    led = SettlementLedger(path=str(tmp_path / "l.json"))
    led.settle("BTC", "tf1", 100.0, 110.0, 1)    # +1000 bp win
    led.settle("BTC", "tf1", 100.0, 95.0, 1)     # -500 bp loss
    st = led.stats("BTC", "tf1")
    assert st["exp_bps"] == 250.0
    assert st["payoff"] == 2.0


def test_ledger_first_touch_is_resolved_only_and_seed_retry_is_idempotent(tmp_path):
    led = SettlementLedger(path=str(tmp_path / "l.json"))
    first = led.settle("BTC", "tf1", 100.0, 101.0, 1,
                       record_id="seed-v1:BTC:tf1:60", first_touch="target",
                       barriers_available=True)
    duplicate = led.settle("BTC", "tf1", 100.0, 99.0, 1,
                           record_id="seed-v1:BTC:tf1:60", first_touch="stop",
                           barriers_available=True)
    led.settle("BTC", "tf1", 100.0, 100.0, 1, first_touch="neither",
               barriers_available=True)
    led.settle("BTC", "tf1", 100.0, 100.0, 1,
               first_touch="unavailable", barriers_available=False)
    ambiguous = led.settle("BTC", "tf1", 100.0, 100.0, 1,
                           first_touch="target", barriers_available=True,
                           ambiguous=True)

    assert first["_persisted"] is True
    assert ambiguous["first_touch"] == "ambiguous"
    assert duplicate["_duplicate"] is True
    assert duplicate["first_touch"] == "target"
    stats = led.stats("BTC", "tf1")
    assert stats["tp_first_n"] == 1
    assert stats["tp_first_pct"] == 1.0
    assert stats["first_touch"]["neither"] == 1
    assert stats["first_touch"]["unavailable"] == 1
    assert stats["first_touch"]["ambiguous"] == 1
    assert led.api_payload(["BTC"], ["tf1"])["rows_total"] == 4
    restored = SettlementLedger(path=str(tmp_path / "l.json"))
    assert restored.stats("BTC", "tf1")["tp_first_n"] == 1


def test_lock_checkpoint_is_atomic_json(tmp_path, monkeypatch):
    import json
    from app.engine import PredictionEngine
    from app.feed import MarketFeed

    monkeypatch.chdir(tmp_path)
    eng = PredictionEngine(MarketFeed(), emit=lambda _message: None)
    eng.locks.clear()
    eng.locks[("BTC", "tf1")] = {"symbol": "BTC", "tf": "tf1",
                                  "window_start": 123.0}
    assert eng._save_locks() is True
    with open(tmp_path / "data" / "locks.json", encoding="utf-8") as fh:
        assert json.load(fh)["locks"][0]["window_start"] == 123.0


def test_engine_first_touch_requires_a_valid_contract_and_grades_probability(tmp_path):
    import time
    from app.engine import PredictionEngine
    from app.feed import MarketFeed
    from app.models import TFCycle

    def settle_case(name, events, *, tamper_contract=False):
        eng = PredictionEngine(MarketFeed(), emit=lambda _message: None)
        eng.locks.clear()
        eng.ledger.path = str(tmp_path / f"{name}.json")
        eng.ledger.rows.clear()
        eng.ledger.settle_calls = 0
        eng.ledger.totals = {"win": 0, "loss": 0, "skip": 0}
        eng._save_locks = lambda: None
        eng._save_prob_audit = lambda: None
        eng.rl.on_settle = lambda **_kwargs: None
        eng.rl.q_on_settle = lambda *_args: None
        eng.rl.audit = lambda *_args: None
        eng._prob_audit = {"n": 0, "sum_pred": 0.0, "tp_first": 0,
                           "brier_sum": 0.0, "brier_cal_sum": 0.0}
        now = time.time()
        start, end = eng._grid_window("tf1", now)
        locked_at = now - 100
        locked = locked_contract_signal(cycle_id=f"touch-{name}")
        contract = dict(locked["contract"])
        if tamper_contract:
            contract["tp"] += 1.0
        eng.cycles[("BTC", "tf1")] = TFCycle(
            symbol="BTC", tf="tf1", W=22, direction=1,
            cycle_id=locked["cycle_id"], lock_id=locked["lock_id"],
            level_revision=1, level_contract=contract, events=events)
        window_start = start - 1320
        eng.locks[("BTC", "tf1")] = {
            "symbol": "BTC", "tf": "tf1", "side": 1, "price": 100.0,
            "locked_at": locked_at, "window_start": window_start,
            "window_end": end - 1, "relocks": 0,
        }
        eng.feed.state("BTC").live_price = 101.0
        eng._halt_until = {"BTC": time.time() + 600}
        eng._prob_pred[("BTC", "tf1")] = {
            "p_target": 0.7, "cycle_start": window_start,
        }
        eng._settle_and_lock("BTC", "tf1", now, {"direction": 1})
        return eng.ledger.rows["BTC|tf1"][0], eng._prob_audit

    target, target_audit = settle_case(
        "target", [{"ts": time.time() - 10, "type": "target_hit"}])
    assert target["first_touch"] == "target"
    assert target["barriers_available"] is True
    assert target_audit["n"] == 1 and target_audit["tp_first"] == 1

    same_ts = time.time() - 10
    ambiguous, ambiguous_audit = settle_case(
        "ambiguous", [
            {"ts": same_ts, "type": "target_hit"},
            {"ts": same_ts, "type": "tp_hit"},
            {"ts": same_ts, "type": "sl_crossed"},
        ])
    assert ambiguous["first_touch"] == "ambiguous"
    assert ambiguous["ambiguous"] is True
    assert ambiguous_audit["n"] == 0

    invalid, invalid_audit = settle_case(
        "invalid", [{"ts": time.time() - 10, "type": "target_hit"}],
        tamper_contract=True)
    assert invalid["first_touch"] == "unavailable"
    assert invalid["barriers_available"] is False
    assert invalid_audit["n"] == 0


def test_ledger_failed_atomic_write_rolls_back(tmp_path, monkeypatch):
    import app.ledger as ledger_module

    led = SettlementLedger(path=str(tmp_path / "l.json"))

    def fail_replace(_src, _dst):
        raise OSError("simulated disk error")

    monkeypatch.setattr(ledger_module.os, "replace", fail_replace)
    result = led.settle("BTC", "tf1", 100.0, 101.0, 1)
    assert result["_persisted"] is False
    assert led.rows == {}
    assert led.settle_calls == 0


def test_prob_audit_ui_wired():
    js = (ROOT / "static" / "app.js").read_text()
    assert "forecast audit (all windows)" in js
    assert "expectancy (barrier pair)" in js
    assert "P(target) 95% interval" in js
