"""Reusable valid lock-bound signal for execution-gate tests."""
from app import levels


def locked_contract_signal(*, symbol="BTC", tf="tf1", side=1,
                           entry=100.0, tp_bps=200.0, sl_bps=100.0,
                           cycle_id="fixture-cycle", feed_age_s=1.0,
                           feed_mode="binance", feed_simulated=False):
    lock = {
        "symbol": symbol, "tf": tf, "side": side, "price": entry,
        "locked_at": 1_000.0, "window_start": 1_000.0,
        "window_end": 2_320.0, "relocks": 0,
    }
    lock["lock_id"] = levels.make_lock_id(lock)
    frozen = levels.freeze(entry, side,
                           {"tp_bps": tp_bps, "sl_bps": sl_bps})
    cycle_id = f"{symbol}:{tf}:{cycle_id}"
    contract = levels.make_contract(
        frozen, symbol=symbol, timeframe=tf, cycle=cycle_id, side=side,
        revision=1, lock_id=lock["lock_id"])
    return {
        "symbol": symbol, "tf": tf, "issued": True,
        "execution_eligible": True, "direction": side,
        "entry": contract["entry"], "tp": contract["tp"],
        "sl": contract["sl"], "contract": contract,
        "contract_lock": lock, "contract_id": contract["id"],
        "cycle_id": cycle_id, "lock_id": lock["lock_id"],
        "level_revision": 1, "feed_age_s": feed_age_s,
        "feed_mode": feed_mode, "feed_simulated": feed_simulated,
        "tp_bps": tp_bps, "sl_bps": sl_bps,
    }
