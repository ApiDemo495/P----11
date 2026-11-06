"""Optional Binance execution gates + per-asset profiles + crowd
emotions. Safety doctrine under test: a random (sub-floor, unarmed,
disabled, wrong-side) prediction can NEVER become an order."""
import time

from app import trading
from app.config import asset_profile
from app.emotion import _crowd

SIG = {"symbol": "BTC", "tf": "tf1", "issued": True, "direction": 1,
       "entry": 100.0, "tp": 101.0, "sl": 99.0, "tp_bps": 100, "sl_bps": 100}


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


def test_trader_executes_only_through_gates_with_brackets():
    calls = []

    def fake_transport(method, url, params, headers):
        calls.append((method, url, params))
        return {"orderId": len(calls)}

    t = trading.Trader(transport=fake_transport)
    t.cfg = _cfg()
    import app.trading as tr
    orig = tr.load_cfg
    tr.load_cfg = lambda: t.cfg
    try:
        row = t.consider({**SIG, "issued": False})
        assert not row["ok"] and not calls          # refused, no order
        row = t.consider(SIG)
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
    assert asset_profile("PAXG")["sl_mult"] > 1.2      # gold: wider
    assert asset_profile("ETH")["floor_mult"] > 1.1    # ETH: higher bar
    assert asset_profile("BNB")["sl_mult"] < 1.0       # BNB: tighter
    assert asset_profile("BTC")["sl_mult"] == 1.0


def test_crowd_reads_are_signed_and_labelled():
    r = _crowd({11: 0.9, 16: 0.8}, {11: 1, 16: 1}, (11, 16),
               ("pos", "neg", "flat"))
    assert r["score"] > 0.15 and r["label"] == "pos"
    f = _crowd({11: 0.5}, {11: 0}, (11,), ("pos", "neg", "flat"))
    assert f["label"] == "flat"
