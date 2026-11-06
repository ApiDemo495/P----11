"""User's five: quiet brain errors, own LineChart v2 (no ChartX), header
Binance connect, functional equity sizing."""
from app.flywire import short_error
from app import trading


def test_brain_error_never_raw():
    raw = "RuntimeError('could not deserialize thrift: page header...')"
    out = short_error(raw)
    assert "thrift" not in out and "deserialize" not in out
    assert "RECONNECT" in out
    assert short_error(None) is None
    assert "memory" in short_error("build worker died exit -9 OOM")


def test_chart_engine_replaced():
    js = open("static/app.js").read()
    assert "class LineChart" in js
    assert "new ChartX" not in js and "class ChartX" not in js
    assert "last price" in js.lower() or "lp.p" in js  # axis chip drawn
    html = open("static/index.html").read()
    assert 'id="btn-binance"' in html and "CONNECT BINANCE" in html
    assert 'id="risk-apply"' in html


def test_risk_sizing_functional(monkeypatch):
    calls = []
    tr = trading.Trader()
    tr.cfg = {**trading.DEFAULTS, "sizing_mode": "risk",
              "equity_usdt": 1000.0, "risk_pct": 1.0,
              "max_order_usdt": 100000.0, "testnet": True,
              "api_key": "k", "api_secret": "s"}
    monkeypatch.setattr(tr, "_signed",
                        lambda m, p, params: calls.append(params) or {"orderId": 1})
    out = tr.execute({"symbol": "BTC", "direction": 1, "entry": 100.0, "sl": 99.0, "tp": 101.0})
    # risk 10 USDT / 1% stop distance = 1000 notional -> qty 10 (capped by equity)
    assert out["qty"] == 10.0
    assert calls[0]["quantity"] == 10.0
