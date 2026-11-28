"""Optional read-only Binance Testnet/account fill synchronization.

This module is intentionally separate from ``app.trading.Trader``. It
implements GET-only signed requests and has no method capable of placing,
modifying, or cancelling an order. Credentials live in ``data/trading.json``
under ``demo_sync`` (distinct from the execution key fields) or in
``BINANCE_DEMO_API_KEY`` / ``BINANCE_DEMO_API_SECRET`` environment variables.

Spot ``myTrades`` do not carry realized PnL; spot PnL is explicitly null
rather than inferred without cost-basis/inventory history. Futures
``userTrades`` do report realized PnL and commission. This synchronizer is
an account/fill snapshot, not an exchange order router and not proof that a
fill belongs to a particular prediction lock. Reconciliation is keyed by
asset and side, and is labeled heuristic unless a client order identity is
available (the prediction-lock system currently supplies none).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CONFIG_PATH = DATA_DIR / "trading.json"
SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "BNB": "BNBUSDT",
           "SOL": "SOLUSDT"}
BASES = {
    (True, "futures"): "https://testnet.binancefuture.com",
    (False, "futures"): "https://fapi.binance.com",
    (True, "spot"): "https://testnet.binance.vision",
    (False, "spot"): "https://api.binance.com",
}
NO_KEYS_BADGE = "APP PREDICTION LOCKS (SIMULATED) — NO TESTNET KEYS LINKED"


def _read_json(path: Path = CONFIG_PATH) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def config(path: Path = CONFIG_PATH, environ: dict | None = None) -> dict:
    env = dict(os.environ if environ is None else environ)
    # Minimal .env support for the three documented names; process env always
    # wins. Parse only these variables and never return their values publicly.
    env_path = path.parent.parent / ".env"
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            if k in ("BINANCE_DEMO_API_KEY", "BINANCE_DEMO_API_SECRET",
                     "BINANCE_DEMO_TESTNET", "BINANCE_DEMO_MARKET"):
                env.setdefault(k, v)
    except OSError:
        pass
    root = _read_json(path)
    cfg = root.get("demo_sync") if isinstance(root.get("demo_sync"), dict) else {}
    key = str(cfg.get("api_key") or env.get("BINANCE_DEMO_API_KEY") or "").strip()
    secret = str(cfg.get("api_secret") or env.get("BINANCE_DEMO_API_SECRET") or "").strip()
    env_test = str(env.get("BINANCE_DEMO_TESTNET", "true")).strip().lower()
    testnet = bool(cfg.get("testnet", env_test not in ("0", "false", "no")))
    market = str(cfg.get("market") or env.get("BINANCE_DEMO_MARKET") or "futures").lower()
    if market not in ("spot", "futures"):
        market = "futures"
    return {"api_key": key, "api_secret": secret,
            "testnet": testnet, "market": market,
            "has_keys": bool(key and secret)}


def save_config(*, api_key: str = "", api_secret: str = "",
                testnet: bool = True, market: str = "futures",
                clear_keys: bool = False, path: Path = CONFIG_PATH) -> dict:
    """Persist separate read-only credentials; blank key inputs preserve."""
    if market not in ("spot", "futures"):
        raise ValueError("market must be spot or futures")
    root = _read_json(path)
    old = root.get("demo_sync") if isinstance(root.get("demo_sync"), dict) else {}
    if clear_keys:
        key = secret = ""
    else:
        key = str(api_key or old.get("api_key") or "").strip()
        secret = str(api_secret or old.get("api_secret") or "").strip()
    root["demo_sync"] = {"api_key": key, "api_secret": secret,
                         "testnet": bool(testnet), "market": market}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(root, indent=1), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)
    return {"has_keys": bool(key and secret), "testnet": bool(testnet),
            "market": market, "api_key": ("••••" + key[-4:] if key else ""),
            "badge": None if key and secret else NO_KEYS_BADGE}


class BinanceReadOnly:
    """Small signed GET-only API client. ``transport`` aids deterministic tests."""
    def __init__(self, cfg: dict | None = None,
                 transport: Callable | None = None):
        self.cfg = cfg or config()
        self.transport = transport or self._http

    @staticmethod
    def _http(url: str, headers: dict) -> Any:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=12) as response:
            return json.loads(response.read().decode("utf-8"))

    def _get(self, path: str, params: dict | None = None) -> Any:
        # Fixed endpoints only. Never accept a caller-provided URL or method.
        allowed = {"/fapi/v2/account", "/fapi/v1/positionRisk",
                   "/fapi/v1/userTrades", "/fapi/v1/income",
                   "/api/v3/account", "/api/v3/myTrades"}
        if path not in allowed:
            raise ValueError("endpoint not on read-only allowlist")
        if not self.cfg.get("has_keys"):
            raise RuntimeError(NO_KEYS_BADGE)
        args = {**(params or {}), "timestamp": int(time.time() * 1000),
                "recvWindow": 5000}
        query = urllib.parse.urlencode(args)
        signature = hmac.new(str(self.cfg["api_secret"]).encode(),
                             query.encode(), hashlib.sha256).hexdigest()
        base = BASES[(bool(self.cfg.get("testnet", True)),
                      str(self.cfg.get("market", "futures")))]
        url = f"{base}{path}?{query}&signature={signature}"
        return self.transport(url, {"X-MBX-APIKEY": self.cfg["api_key"]})

    def snapshot(self) -> dict:
        market = str(self.cfg.get("market", "futures"))
        positions = []
        fills = []
        income = []
        if market == "futures":
            account = self._get("/fapi/v2/account")
            for p in account.get("positions", []):
                amount = float(p.get("positionAmt") or 0)
                symbol = p.get("symbol")
                if amount and symbol in SYMBOLS.values():
                    positions.append({
                        "symbol": symbol, "asset": next(k for k, v in SYMBOLS.items() if v == symbol),
                        "side": "BUY" if amount > 0 else "SELL",
                        "quantity": abs(amount), "entry_price": float(p.get("entryPrice") or 0),
                        "unrealized_pnl": float(p.get("unrealizedProfit") or 0),
                        "mark_price": float(p.get("markPrice") or 0),
                    })
            for asset, symbol in SYMBOLS.items():
                rows = self._get("/fapi/v1/userTrades", {"symbol": symbol, "limit": 100})
                for row in rows:
                    fills.append({
                        "asset": asset, "symbol": symbol,
                        "time_ms": int(row.get("time") or 0),
                        "side": "BUY" if row.get("buyer") else "SELL",
                        "price": float(row.get("price") or 0),
                        "quantity": float(row.get("qty") or 0),
                        "realized_pnl": float(row.get("realizedPnl") or 0),
                        "commission": float(row.get("commission") or 0),
                        "commission_asset": row.get("commissionAsset"),
                        "order_id": str(row.get("orderId") or ""),
                        "client_order_id": str(row.get("clientOrderId") or ""),
                    })
            rows = self._get("/fapi/v1/income", {"limit": 1000})
            income = [{"time_ms": int(r.get("time") or 0),
                       "symbol": r.get("symbol"), "type": r.get("incomeType"),
                       "amount": float(r.get("income") or 0),
                       "asset": r.get("asset"),
                       "info": str(r.get("info") or "")[:120]}
                      for r in rows]
        else:
            account = self._get("/api/v3/account")
            for b in account.get("balances", []):
                free, locked = float(b.get("free") or 0), float(b.get("locked") or 0)
                if free + locked > 0:
                    positions.append({"asset": b.get("asset"), "symbol": b.get("asset"),
                                      "side": "LONG" if free + locked > 0 else "FLAT",
                                      "quantity": free + locked,
                                      "free": free, "locked": locked,
                                      "entry_price": None, "unrealized_pnl": None})
            for asset, symbol in SYMBOLS.items():
                rows = self._get("/api/v3/myTrades", {"symbol": symbol, "limit": 100})
                for row in rows:
                    fills.append({
                        "asset": asset, "symbol": symbol,
                        "time_ms": int(row.get("time") or 0),
                        "side": "BUY" if row.get("isBuyer") else "SELL",
                        "price": float(row.get("price") or 0),
                        "quantity": float(row.get("qty") or 0),
                        "realized_pnl": None,
                        "commission": float(row.get("commission") or 0),
                        "commission_asset": row.get("commissionAsset"),
                        "order_id": str(row.get("orderId") or ""),
                        "client_order_id": "",
                    })
        fills.sort(key=lambda x: x["time_ms"], reverse=True)
        return {"market": market, "testnet": bool(self.cfg.get("testnet", True)),
                "positions": positions, "fills": fills[:400],
                "income": income, "realized_pnl_available": market == "futures",
                "fetched_at": time.time()}


def reconcile(locks: list[dict], snapshot: dict) -> list[dict]:
    """Compare locks with account positions without claiming exact linkage."""
    positions = {p.get("asset"): p for p in snapshot.get("positions", [])
                 if p.get("asset") in SYMBOLS}
    rows = []
    for lock in locks:
        asset = str(lock.get("symbol") or "")
        pos = positions.get(asset)
        want = str(lock.get("side") or "")
        have = str(pos.get("side") or "") if pos else "FLAT"
        # Spot balances cannot reliably reveal short exposure or average entry.
        if snapshot.get("market") == "spot" and pos:
            status = "balance present; trade/lock linkage unavailable"
        elif not pos and lock.get("status") == "open":
            status = "prediction open; no matching account position"
        elif pos and want == have:
            status = "asset+side match only; not a fill-level link"
        elif pos:
            status = "SIDE MISMATCH — inspect account fills"
        else:
            status = "no account position (lock may already be closed)"
        rows.append({"lock_id": lock.get("id"), "asset": asset,
                     "timeframe": lock.get("tf"), "lock_side": want,
                     "lock_entry": lock.get("entry"),
                     "lock_status": lock.get("status"),
                     "account_side": have,
                     "account_entry": pos.get("entry_price") if pos else None,
                     "account_quantity": pos.get("quantity") if pos else 0,
                     "status": status,
                     "linkage": "heuristic only: app lock has no exchange clientOrderId"})
    return rows


class DemoSync:
    """API-facing service with a short snapshot cache to protect rate limits."""
    def __init__(self, config_provider=config, client_factory=BinanceReadOnly,
                 ttl_s: float = 15.0):
        self.config_provider = config_provider
        self.client_factory = client_factory
        self.ttl_s = ttl_s
        self._cache: dict | None = None
        self._cached_at = 0.0

    def status(self) -> dict:
        cfg = self.config_provider()
        if not cfg.get("has_keys"):
            return {"ok": True, "linked": False, "badge": NO_KEYS_BADGE,
                    "market": cfg["market"], "testnet": cfg["testnet"],
                    "positions": [], "fills": [], "reconciliation": [],
                    "note": "Internal prediction locks are app measurements, not account trades."}
        if self._cache and time.time() - self._cached_at < self.ttl_s:
            return {"ok": True, "linked": True, **self._cache,
                    "cached": True}
        snapshot = self.client_factory(cfg).snapshot()
        self._cache = snapshot
        self._cached_at = time.time()
        return {"ok": True, "linked": True, **snapshot, "cached": False}

    def reset(self) -> None:
        self._cache = None
        self._cached_at = 0.0


SERVICE = DemoSync()
