"""trading.py — OPTIONAL Binance execution, safe by construction.

Doctrine: predictions must never be able to fire trades by accident.

  * DISABLED by default; no keys -> nothing can execute, ever.
  * TESTNET by default (testnet.binancefuture.com). Mainnet requires
    the user to explicitly turn testnet OFF in settings.
  * ARMED window: even with keys + enabled, orders only fire while an
    ARM timer (default 30 min, set by the user) is running.
  * Only ISSUED signals (confidence above the asset's own floor,
    direction set, TP/SL present and on the correct side of entry —
    re-checked with app/levels.py) may execute. Sub-floor "weak lean"
    calls are refused with the reason logged.
  * One open position per symbol; a hard cap on order size (USDT) and
    on orders per day; every gate decision (fire or refuse) is logged
    to data/trading_log.json — silence is not an option.
  * Keys live only in data/trading.json (runtime state, never in the
    repo) and are masked in every API response.

Orders: USDT-M futures MARKET entry + STOP_MARKET (SL) +
TAKE_PROFIT_MARKET (TP), closePosition brackets, reduceOnly safety.
Symbol mapping: BTC->BTCUSDT, ETH->ETHUSDT, BNB->BNBUSDT,
PAXG->PAXGUSDT.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from . import levels as _levels

DATA = Path(__file__).resolve().parent.parent / "data"
CFG_PATH = DATA / "trading.json"
LOG_PATH = DATA / "trading_log.json"
SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "BNB": "BNBUSDT",
           "PAXG": "PAXGUSDT"}
TESTNET = "https://testnet.binancefuture.com"
MAINNET = "https://fapi.binance.com"

DEFAULTS = {
    "enabled": False,
    "testnet": True,
    "api_key": "",
    "api_secret": "",
    "max_order_usdt": 25.0,
    # FUNCTIONAL equity sizing (the POSITION SIZE panel drives this):
    # mode "risk" sizes every order as equity x risk_pct% / stop-distance,
    # capped by max_order_usdt and by the equity itself. "fixed" = legacy.
    "sizing_mode": "fixed",
    "equity_usdt": 0.0,
    "risk_pct": 1.0,
    "max_orders_day": 4,
    "armed_until": 0.0,
}


def load_cfg() -> dict:
    try:
        cfg = {**DEFAULTS, **json.loads(CFG_PATH.read_text())}
    except Exception:
        cfg = dict(DEFAULTS)
    return cfg


def save_cfg(cfg: dict) -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    CFG_PATH.write_text(json.dumps(cfg, indent=1))
    return cfg


def public_cfg(cfg: dict) -> dict:
    out = dict(cfg)
    out["api_key"] = ("••••" + cfg["api_key"][-4:]) if cfg.get("api_key") else ""
    out["api_secret"] = "••••" if cfg.get("api_secret") else ""
    out["has_keys"] = bool(cfg.get("api_key") and cfg.get("api_secret"))
    out["armed"] = float(cfg.get("armed_until") or 0) > time.time()
    return out


def _log(row: dict) -> None:
    try:
        rows = json.loads(LOG_PATH.read_text()) if LOG_PATH.exists() else []
    except Exception:
        rows = []
    rows.append(row)
    DATA.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(rows[-200:], indent=1))


def read_log(n: int = 20) -> list:
    try:
        return json.loads(LOG_PATH.read_text())[-n:]
    except Exception:
        return []


def gate(cfg: dict, sig: dict, open_symbols: set,
         orders_today: int) -> tuple[bool, str]:
    """Every refusal is a reason, never silence. Order matters."""
    if not cfg.get("enabled"):
        return False, "trading disabled in settings"
    if not (cfg.get("api_key") and cfg.get("api_secret")):
        return False, "no API keys configured"
    if float(cfg.get("armed_until") or 0) <= time.time():
        return False, "not ARMED (arm timer expired — press ARM to trade)"
    if float(sig.get("feed_age_s") or 0) > 20.0:
        return False, (f"stale tape ({sig.get('feed_age_s'):.0f}s since last "
                       "tick) — never trade a dead feed")
    if not sig.get("issued"):
        return False, "signal not ISSUED (below confidence floor — no trade)"
    if not sig.get("direction"):
        return False, "no direction"
    # execution is STRICTER than display: levels must be valid as
    # given — never repaired, never recomputed at order time
    if not _levels.valid(
            {"tp": sig.get("tp"), "sl": sig.get("sl"),
             "entry": sig.get("entry"), "direction": sig.get("direction")}):
        return False, "TP/SL missing or wrong side of entry — refused"
    if sig["symbol"] in open_symbols:
        return False, f"position already open on {sig['symbol']}"
    if orders_today >= int(cfg.get("max_orders_day") or 4):
        return False, "daily order cap reached"
    return True, "all gates passed"


class Trader:
    def __init__(self, transport=None):
        self.cfg = load_cfg()
        self.open_positions: dict[str, dict] = {}
        self.orders_today = 0
        self._day = time.strftime("%Y-%m-%d")
        # transport(method, url, params, headers) -> dict (injectable)
        self.transport = transport or self._http

    # -- plumbing ------------------------------------------------------
    def _roll_day(self):
        today = time.strftime("%Y-%m-%d")
        if today != self._day:
            self._day, self.orders_today = today, 0

    def _http(self, method, url, params, headers):
        data = urllib.parse.urlencode(params).encode() \
            if method != "GET" else None
        if method == "GET" and params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method=method)
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())

    def _signed(self, method, path, params):
        cfg = self.cfg
        params = {**params, "timestamp": int(time.time() * 1000),
                  "recvWindow": 5000}
        qs = urllib.parse.urlencode(params)
        sig = hmac.new(cfg["api_secret"].encode(), qs.encode(),
                       hashlib.sha256).hexdigest()
        base = TESTNET if cfg.get("testnet", True) else MAINNET
        return self.transport(method, base + path,
                              {**params, "signature": sig},
                              {"X-MBX-APIKEY": cfg["api_key"]})

    # -- decision + execution ------------------------------------------
    def consider(self, sig: dict) -> dict:
        """Evaluate one issued lock. Logs the decision either way;
        executes only when every gate passes."""
        self.cfg = load_cfg()
        self._roll_day()
        ok, why = gate(self.cfg, sig, set(self.open_positions),
                       self.orders_today)
        row = {"ts": time.time(), "symbol": sig.get("symbol"),
               "tf": sig.get("tf"), "side": sig.get("direction"),
               "entry": sig.get("entry"), "ok": ok, "why": why,
               "testnet": bool(self.cfg.get("testnet", True))}
        _log(row)
        if not ok:
            return row
        try:
            row["order"] = self.execute(sig)
            self.orders_today += 1
            self.open_positions[sig["symbol"]] = {
                "tf": sig.get("tf"), "entry": sig.get("entry"),
                "side": sig.get("direction"),
                "window_end": sig.get("window_end")}
        except Exception as exc:                       # never crash engine
            row["ok"] = False
            row["why"] = f"order failed: {exc}"[:200]
            _log(row)
        return row

    def execute(self, sig: dict) -> dict:
        cfg = self.cfg
        sym = SYMBOLS[sig["symbol"]]
        entry = float(sig["entry"])
        notional = float(cfg["max_order_usdt"])
        if cfg.get("sizing_mode") == "risk" and float(cfg.get("equity_usdt") or 0) > 0:
            sl = float(sig.get("sl") or 0)
            dist = abs(entry - sl) / entry if sl > 0 else 0.0
            if dist > 0:
                risk_usdt = float(cfg["equity_usdt"]) * float(cfg.get("risk_pct") or 1.0) / 100.0
                notional = min(risk_usdt / dist, notional,
                               float(cfg["equity_usdt"]))
        qty = round(notional / entry, 6)
        side = "BUY" if sig["direction"] > 0 else "SELL"
        close = "SELL" if side == "BUY" else "BUY"
        entry_o = self._signed("POST", "/fapi/v1/order", {
            "symbol": sym, "side": side, "type": "MARKET",
            "quantity": qty})
        sl_o = self._signed("POST", "/fapi/v1/order", {
            "symbol": sym, "side": close, "type": "STOP_MARKET",
            "stopPrice": round(float(sig["sl"]), 2), "closePosition": "true"})
        tp_o = self._signed("POST", "/fapi/v1/order", {
            "symbol": sym, "side": close, "type": "TAKE_PROFIT_MARKET",
            "stopPrice": round(float(sig["tp"]), 2), "closePosition": "true"})
        return {"symbol": sym, "qty": qty, "entry_id": entry_o.get("orderId"),
                "sl_id": sl_o.get("orderId"), "tp_id": tp_o.get("orderId")}

    def close_position(self, symbol: str, exit_price: float,
                       result: str | None) -> dict | None:
        """Settlement reached: close the book on an open position and
        log the estimated outcome (bps vs entry, side-aware)."""
        pos = self.open_positions.pop(symbol, None)
        if not pos:
            return None
        entry = float(pos.get("entry") or 0)
        side = int(pos.get("side") or 0)
        pnl_bps = round(side * (exit_price / entry - 1) * 1e4, 1) \
            if entry > 0 and exit_price > 0 else None
        row = {"ts": time.time(), "symbol": symbol, "tf": pos.get("tf"),
               "side": side, "entry": entry, "exit": exit_price,
               "ok": True, "why": f"position closed at settle ({result})",
               "pnl_bps_est": pnl_bps, "closed": True}
        _log(row)
        return row

    def status(self) -> dict:
        self.cfg = load_cfg()
        self._roll_day()
        return {"config": public_cfg(self.cfg),
                "open_positions": self.open_positions,
                "orders_today": self.orders_today,
                "log": read_log(12)}
