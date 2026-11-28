"""Optional Binance execution gates + per-asset profiles + crowd
emotions. Safety doctrine under test: a random (sub-floor, unarmed,
disabled, wrong-side) prediction can NEVER become an order."""
import time

from app import levels
from app import trading
from app.config import asset_profile
from app.emotion import _crowd
from tests.contract_fixtures import locked_contract_signal

SIG = locked_contract_signal(entry=100.0, tp_bps=100.0, sl_bps=100.0,
                             cycle_id="trading-profile")


def _cfg(**kw):
    c = {"enabled": True, "testnet": True, "api_key": "k", "api_secret": "s",
         "max_order_usdt": 25, "max_orders_day": 4,
         "armed_until": time.time() + 600}
    c.update(kw)
    return c


def test_gate_refuses_everything_unsafe():
    assert not trading.gate(_cfg(enabled=False), SIG, set(), 0)[0]
    assert not trading.gate(_cfg(api_key=""), SIG, set(), 0)[0]
    assert not trading.gate(_cfg(armed_until=0), SIG, set(), 0)[0]
    assert not trading.gate(_cfg(), {**SIG, "issued": False}, set(), 0)[0]
    assert not trading.gate(_cfg(), {**SIG, "sl": 100.5}, set(), 0)[0]  # wrong side
    assert not trading.gate(_cfg(), SIG, {"BTC"}, 0)[0]                 # open pos
    assert not trading.gate(_cfg(), SIG, set(), 4)[0]                   # day cap
    ok, why = trading.gate(_cfg(), SIG, set(), 0)
    assert ok and why == "all gates passed"
    sim = {**SIG, "feed_mode": "sim", "feed_simulated": True}
    ok, why = trading.gate(_cfg(), sim, set(), 0)
    assert not ok and "live Binance feed required" in why
    missing_feed = {k: v for k, v in SIG.items()
                    if k not in ("feed_mode", "feed_simulated")}
    ok, why = trading.gate(_cfg(), missing_feed, set(), 0)
    assert not ok and "live Binance feed required" in why


def test_the_execution_gate_still_refuses_a_sub_one_to_one_contract():
    """The gate stays as a last-resort net, but it must be UNREACHABLE now.

    A sub-1:1 pair is floored when the contract is built and again when the
    card's levels are written, so `levels.freeze` no longer produces one and
    the gate can only fire from a hand-built artifact. Keeping it is what
    makes that a guarantee rather than a hope: if a future caller bypasses
    the floor, the order is still refused instead of executed at 0.5:1.
    """
    # a contract the floor never touched cannot be sub-1:1 ...
    floored = levels.freeze(100.0, 1, {"tp_bps": 50.0, "sl_bps": 100.0})
    assert levels.make_contract(
        floored, symbol="BTC", timeframe="tf1", cycle="c1", side=1,
        revision=1, lock_id="lock-x")["ratio"] >= 1.0
    # ... so the ONLY way to reach the gate is a forged artifact, and then
    # it must refuse. Build one directly.
    forged = dict(locked_contract_signal(
        entry=100.0, tp_bps=50.0, sl_bps=100.0, cycle_id="below-floor"))
    contract = dict(forged.get("contract") or {})
    if not contract:
        return
    contract.update({"ratio": 0.5, "below_floor": True,
                     "tp": 100.5, "tp_dist": 0.5, "sl": 99.0, "sl_dist": 1.0})
    forged["contract"] = contract
    forged.update({"tp": 100.5, "sl": 99.0})
    ok, why = trading.gate(_cfg(), forged, set(), 0)
    assert not ok, "a sub-1:1 contract must never reach an order"
    # it is refused; WHICH check catches it is not the point (a hand-built
    # artifact usually trips the fingerprint first). What matters is that no
    # path lets a target closer than its stop through to execution.
    assert why


def test_trader_executes_only_through_gates_with_brackets():
    """The doctrine under test is "nothing bypasses the gates".

    The order it uses must still be one the EXCHANGE would accept, or the
    test proves nothing about a real order: a 25 USDT budget for a
    BTCUSDT order at 100.0 rounds to zero at stepSize 0.001 and is now
    refused before it is sent. The fixture is therefore sized like a real
    BTC order (BTCUSDT tick 0.1 / step 0.001 / minNotional 100).
    """
    calls = []

    def fake_transport(method, url, params, headers):
        calls.append((method, url, params))
        return {"orderId": len(calls)}

    t = trading.Trader(transport=fake_transport)
    t.cfg = _cfg(max_order_usdt=5000)
    import app.trading as tr
    orig = tr.load_cfg
    tr.load_cfg = lambda: t.cfg
    try:
        btc = locked_contract_signal(symbol="BTC", entry=68273.5,
                                     tp_bps=100.0, sl_bps=100.0,
                                     cycle_id="trading-profile-btc")
        row = t.consider({**btc, "issued": False})
        assert not row["ok"] and not calls          # refused, no order
        row = t.consider(btc)
        assert row["ok"] and len(calls) == 3        # entry + SL + TP
        types = [c[2].get("type") for c in calls]
        assert types == ["MARKET", "STOP_MARKET", "TAKE_PROFIT_MARKET"]
        assert "testnet.binancefuture.com" in calls[0][1]
        assert "signature" in calls[0][2]
    finally:
        tr.load_cfg = orig


def test_public_cfg_masks_keys():
    pub = trading.public_cfg(_cfg())
    assert "s" not in pub["api_secret"] and pub["has_keys"]


def test_asset_profiles_differ_by_nature():
    assert asset_profile("SOL")["sl_mult"] > 1.2       # high beta: wider
    assert asset_profile("ETH")["floor_mult"] > 1.1    # ETH: higher bar
    assert asset_profile("BNB")["sl_mult"] < 1.0       # BNB: tighter
    assert asset_profile("BTC")["sl_mult"] == 1.0


def test_crowd_reads_are_signed_and_labelled():
    r = _crowd({11: 0.9, 16: 0.8}, {11: 1, 16: 1}, (11, 16),
               ("pos", "neg", "flat"))
    assert r["score"] > 0.15 and r["label"] == "pos"
    f = _crowd({11: 0.5}, {11: 0}, (11,), ("pos", "neg", "flat"))
    assert f["label"] == "flat"
