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


def test_first_touch_is_reported_as_resolved_after_a_contract_barrier():
    """A touched barrier is a RESOLVED window, not a blank panel.

    The old contract returned `p_target=None` and dropped the whole
    distribution, which left the Probability-Branches chart empty for the
    rest of the window. A first touch of the target IS P(target before
    stop) = 100% for this window, so it is reported as such, together
    with the distribution of the time that is left.
    """
    from app.engine import PredictionEngine
    payload = {"symbol": "BTC", "direction": 1, "entry": 100.0,
               "tp": 102.0, "sl": 99.0, "countdown_s": 60,
               "level_contract": {"entry": 100.0},
               "contract_tp_hit": False, "sl_observed": False}
    result = PredictionEngine.branch_stats(None, payload, 102.0, 2_000.0, "tf1")
    assert result["resolved"]["tp_first"] is True
    assert result["p_target"] == 1.0 and result["p_stop"] == 0.0
    assert result["p_between"] == 0.0
    assert "already observed" in result["method"]
    assert result["fan"] and result["bands"]        # chart still renders
    # a stop-first touch is the mirror case
    payload_sl = {**payload, "sl_observed": True}
    r2 = PredictionEngine.branch_stats(None, payload_sl, 98.5, 2_000.0, "tf1")
    assert r2["resolved"]["sl_first"] is True
    assert r2["p_target"] == 0.0 and r2["p_stop"] == 1.0


def test_contract_snapshot_and_analysis_share_one_pair():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    engine = (root / "app" / "engine.py").read_text()
    assert "_install_cycle_contract(cyc, now" in engine
    assert 'd["level_contract"]' in engine and 'd["contract_id"]' in engine
    assert "_levels.validate_cycle_contract" in engine
