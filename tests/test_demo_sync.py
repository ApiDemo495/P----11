"""Read-only exchange sync, credential masking, and lock reconciliation."""
import json
from pathlib import Path

import pytest

from app import demo_sync as ds


def test_no_keys_mode_uses_explicit_simulated_badge_without_api_calls(tmp_path):
    svc = ds.DemoSync(config_provider=lambda: {
        "has_keys": False, "market": "futures", "testnet": True,
        "api_key": "", "api_secret": ""},
        client_factory=lambda *_: pytest.fail("must not poll without keys"))
    result = svc.status()
    assert result["linked"] is False
    assert result["badge"] == ds.NO_KEYS_BADGE
    assert "SIMULATED" in result["badge"]


def test_config_uses_nested_trading_file_and_env_without_exposing_secret(tmp_path):
    path = tmp_path / "data" / "trading.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"enabled": True, "api_key": "execution-key",
                                "api_secret": "execution-secret"}))
    saved = ds.save_config(api_key="sync-key", api_secret="sync-secret",
                           market="spot", testnet=True, path=path)
    assert saved["has_keys"] and saved["api_key"] == "••••-key"
    raw = json.loads(path.read_text())
    assert raw["api_key"] == "execution-key"  # execution config preserved
    assert raw["demo_sync"]["api_secret"] == "sync-secret"
    cfg = ds.config(path, environ={})
    assert cfg["api_key"] == "sync-key" and cfg["market"] == "spot"
    assert "sync-secret" not in json.dumps(saved)
    assert path.stat().st_mode & 0o077 == 0

    cleared = ds.save_config(clear_keys=True, path=path)
    assert not cleared["has_keys"]
    assert json.loads(path.read_text())["api_key"] == "execution-key"


def test_env_fallback_and_env_file(tmp_path):
    root = tmp_path / ".env"
    root.write_text("BINANCE_DEMO_API_KEY=env-key\n"
                    "BINANCE_DEMO_API_SECRET='env-secret'\n"
                    "BINANCE_DEMO_MARKET=spot\n")
    cfg = ds.config(tmp_path / "data" / "trading.json", environ={})
    assert cfg["api_key"] == "env-key" and cfg["api_secret"] == "env-secret"
    assert cfg["market"] == "spot" and cfg["has_keys"]


def test_signed_requests_are_get_only_and_endpoint_allowlisted():
    calls = []
    def transport(url, headers):
        calls.append((url, headers))
        if "/fapi/v2/account?" in url:
            return {"positions": [{"symbol": "BTCUSDT", "positionAmt": "0.2",
                                   "entryPrice": "60000", "markPrice": "61000",
                                   "unRealizedProfit": "200"}]}
        if "/fapi/v1/userTrades?" in url:
            sym = url.split("symbol=")[1].split("&")[0]
            return [{"symbol": sym, "time": 1234, "buyer": True,
                     "price": "60000", "qty": "0.2", "realizedPnl": "10",
                     "commission": "0.01", "commissionAsset": "USDT",
                     "orderId": 7, "clientOrderId": ""}]
        if "/fapi/v1/income?" in url:
            return [{"time": 1234, "symbol": "BTCUSDT", "incomeType": "COMMISSION",
                     "income": "-0.01", "asset": "USDT"}]
        raise AssertionError(url)
    client = ds.BinanceReadOnly({"api_key": "k", "api_secret": "s",
                                 "has_keys": True, "market": "futures",
                                 "testnet": True}, transport=transport)
    snap = client.snapshot()
    assert len(calls) == 6  # account + four userTrades + income
    assert all(url.startswith(ds.BASES[(True, "futures")]) for url, _ in calls)
    assert all("signature=" in url and h["X-MBX-APIKEY"] == "k"
               for url, h in calls)
    assert snap["positions"][0]["side"] == "BUY"
    assert snap["positions"][0]["entry_price"] == 60000
    assert snap["fills"][0]["realized_pnl"] == 10
    assert snap["fills"][0]["commission"] == 0.01
    assert snap["income"][0]["type"] == "COMMISSION"
    with pytest.raises(ValueError):
        client._get("/fapi/v1/order", {"symbol": "BTCUSDT"})


def test_spot_trade_sync_does_not_invent_realized_pnl():
    def transport(url, headers):
        if "/api/v3/account?" in url:
            return {"balances": [{"asset": "BTC", "free": "0.1", "locked": "0"}]}
        return [{"symbol": "BTCUSDT", "time": 1000, "isBuyer": True,
                 "price": "50000", "qty": "0.1", "commission": "0.001",
                 "commissionAsset": "BTC", "orderId": 1}]
    client = ds.BinanceReadOnly({"api_key": "k", "api_secret": "s",
                                 "has_keys": True, "market": "spot",
                                 "testnet": False}, transport=transport)
    snap = client.snapshot()
    assert snap["positions"][0]["asset"] == "BTC"
    assert snap["fills"][0]["realized_pnl"] is None
    assert snap["fills"][0]["commission"] == 0.001
    assert snap["realized_pnl_available"] is False


def test_reconcile_reports_heuristic_match_and_side_mismatch():
    locks = [
        {"id": "lock-1", "symbol": "BTC", "tf": "tf1", "side": "BUY",
         "entry": 60000, "status": "open"},
        {"id": "lock-2", "symbol": "ETH", "tf": "tf2", "side": "SELL",
         "entry": 3000, "status": "open"},
    ]
    snapshot = {"market": "futures", "positions": [
        {"asset": "BTC", "side": "BUY", "entry_price": 60500, "quantity": 0.2},
        {"asset": "ETH", "side": "BUY", "entry_price": 3000, "quantity": 1}],
        "fills": []}
    rows = ds.reconcile(locks, snapshot)
    assert rows[0]["status"].startswith("asset+side match")
    assert rows[1]["status"].startswith("SIDE MISMATCH")
    assert all("heuristic" in r["linkage"] for r in rows)


def test_reconcile_labels_absent_position_not_a_filled_trade():
    rows = ds.reconcile([{"symbol": "BNB", "tf": "tf4", "side": "SELL",
                          "status": "open"}],
                        {"market": "futures", "positions": [], "fills": []})
    assert rows[0]["account_side"] == "FLAT"
    assert "no matching account position" in rows[0]["status"]
