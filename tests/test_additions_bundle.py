"""Self-directed additions: empirical fan, hero crosshair, risk
calculator, issued alerts, trade close-out bookkeeping."""
from pathlib import Path

from app import trading
from tests.contract_fixtures import locked_contract_signal

ROOT = Path(__file__).resolve().parent.parent


def test_branch_payload_carries_empirical_fan():
    src = (ROOT / "app" / "engine.py").read_text()
    assert '"fan": _fan' in src and "_q(_rets[_f], 0.05)" in src
    assert '"fan": [{"f": f,' in src          # labelled fallback cone


def test_charts_use_fan_crosshair_riskcalc_alerts():
    js = (ROOT / "static" / "app.js").read_text()
    assert "p2.q95" in js and "p2.q05" in js  # engine fan keys are q05/q95 (q90/q10 was the blank-chart bug)
    lc = (ROOT / "static" / "LineChart.js").read_text()
    assert "this.cross" in lc  # PriceChart magnet crosshair lives in LineChart.js
    assert "function renderRiskCalc()" in js
    assert "riskUsdt / dist" in js
    assert "checkIssuedAlerts" in js and "was === false" in js
    html = (ROOT / "static" / "index.html").read_text()
    assert 'id="risk-eq"' in html and 'id="risk-table"' in html


def test_close_position_books_pnl_and_frees_symbol():
    import time
    calls = []

    def fake(method, url, params, headers):
        calls.append(params)
        return {"orderId": len(calls)}

    t = trading.Trader(transport=fake)
    t.cfg = {"enabled": True, "testnet": True, "api_key": "k",
             "api_secret": "s", "max_order_usdt": 25, "max_orders_day": 4,
             "armed_until": time.time() + 600}
    import app.trading as tr
    orig = tr.load_cfg
    tr.load_cfg = lambda: t.cfg
    try:
        sig = locked_contract_signal(symbol="ETH", side=-1, entry=100.0,
                                     tp_bps=200.0, sl_bps=100.0,
                                     cycle_id="close-position")
        assert t.consider(sig)["ok"]
        assert "ETH" in t.open_positions
        row = t.close_position("ETH", 97.0, "win")
        assert row["pnl_bps_est"] == 300.0     # short, -3% = +300bp
        assert "ETH" not in t.open_positions
        assert t.close_position("ETH", 97.0, "win") is None
    finally:
        tr.load_cfg = orig
