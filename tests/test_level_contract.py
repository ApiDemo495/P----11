"""Contract identity, locked-entry barriers and account-safety regressions."""
from app import levels
from app.models import TFCycle
from tests.contract_fixtures import locked_contract_signal


def test_contract_is_bound_to_lock_cycle_side_and_revision():
    sig = locked_contract_signal(cycle_id="identity")
    contract = sig["contract"]
    cycle = TFCycle(symbol=sig["symbol"], tf=sig["tf"], W=22,
                    direction=sig["direction"], cycle_id=sig["cycle_id"],
                    lock_id=sig["lock_id"], level_revision=1,
                    contract_lock=sig["contract_lock"],
                    level_contract=contract)
    assert levels.validate_cycle_contract(cycle) == contract
    cycle.level_contract = {**contract, "tp": contract["tp"] + 1}
    assert levels.validate_cycle_contract(cycle) is None
    cycle.level_contract = contract
    cycle.direction = -1
    assert levels.validate_cycle_contract(cycle) is None


def test_contract_ratios_keep_floor_and_cap_explicit():
    lock_id = "lock-test"
    below = levels.make_contract(
        levels.freeze(100.0, 1, {"tp_bps": 50.0, "sl_bps": 100.0}),
        symbol="BTC", timeframe="tf1", cycle="c1", side=1,
        revision=1, lock_id=lock_id)
    assert below and below["below_floor"] and below["ratio"] == 0.5
    above = levels.make_contract(
        levels.freeze(100.0, 1, {"tp_bps": 500.0, "sl_bps": 100.0}),
        symbol="BTC", timeframe="tf1", cycle="c2", side=1,
        revision=1, lock_id=lock_id)
    assert above is None


def test_simulated_feed_cannot_be_marked_execution_eligible():
    sig = locked_contract_signal(cycle_id="sim-feed", feed_mode="sim",
                                 feed_simulated=True)
    from app import trading
    cfg = {"enabled": True, "api_key": "k", "api_secret": "s",
           "armed_until": 9999999999, "max_orders_day": 4}
    ok, why = trading.gate(cfg, sig, set(), 0)
    assert not ok and "live Binance feed required" in why


def test_first_touch_probability_is_withheld_after_a_contract_barrier():
    from app.engine import PredictionEngine
    payload = {"symbol": "BTC", "direction": 1, "entry": 100.0,
               "tp": 102.0, "sl": 99.0, "countdown_s": 60,
               "level_contract": {"entry": 100.0},
               "contract_tp_hit": False, "sl_observed": False}
    result = PredictionEngine.branch_stats(None, payload, 102.0, 2_000.0, "tf1")
    assert result["p_target"] is None and result["p_stop"] is None
    assert "probability withheld" in result["method"]


def test_contract_snapshot_and_analysis_share_one_pair():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    engine = (root / "app" / "engine.py").read_text()
    assert "_install_cycle_contract(cyc, now" in engine
    assert 'd["level_contract"]' in engine and 'd["contract_id"]' in engine
    assert "_levels.validate_cycle_contract" in engine
