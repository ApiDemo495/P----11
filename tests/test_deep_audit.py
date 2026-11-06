"""Deep additions: Wilson intervals + expectancy on probabilities,
probability-forecast auditing, feed-staleness trading gate, ledger
expectancy. The theme: every number ships with its uncertainty, and
the forecast system itself is graded."""
from pathlib import Path

from app import trading
from app.ledger import SettlementLedger

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
    sig = {"symbol": "BTC", "tf": "tf1", "issued": True, "direction": 1,
           "entry": 100.0, "tp": 101.0, "sl": 99.0,
           "feed_age_s": 45.0}
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


def test_prob_audit_ui_wired():
    js = (ROOT / "static" / "app.js").read_text()
    assert "forecast audit (all windows)" in js
    assert "expectancy (barrier pair)" in js
    assert "P(target) 95% interval" in js
