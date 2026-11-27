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
from .exchange_filters import cheapest_legal_notional, legal_order

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
    # Binance futures rejects any order below the symbol's MIN_NOTIONAL,
    # and the real floor is higher than the raw filter because quantity
    # must also be a multiple of stepSize. Measured 2026-11-27:
    #
    #   BTC    minNotional 100 USDT, stepSize 0.001 -> 0.002 BTC = ~136 USDT
    #   ETH    20 USDT, stepSize 0.001              -> ~22 USDT
    #   BNB    20 USDT, stepSize 0.010              -> ~22 USDT
    #   PAXG   20 USDT, stepSize 0.001              -> ~22 USDT
    #
    # The old default was 25.0, which cannot be filled on ANY of the four
    # symbols (BTC fails on three filters at once). 150 clears BTC's
    # effective floor with room for price drift; exchange_filters
    # re-derives the true floor from live /exchangeInfo at execution time
    # and the order path refuses an illegal size instead of sending it.
    "max_order_usdt": 150.0,
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


def _valid_contract_signal(sig: dict) -> tuple[bool, str]:
    contract, lock = sig.get("contract"), sig.get("contract_lock")
    if not _levels.lock_id_valid(lock):
        return False, "contract lock fingerprint missing or invalid"
    if not _levels.contract_valid(
            contract, symbol=sig.get("symbol"), timeframe=sig.get("tf"),
            cycle=sig.get("cycle_id"), side=sig.get("direction"),
            revision=sig.get("level_revision"), lock_id=lock.get("lock_id"),
            entry=sig.get("entry")):
        return False, "frozen TP/SL contract identity or fingerprint invalid"
    for field in ("entry", "tp", "sl"):
        try:
            if abs(float(sig.get(field)) - float(contract[field])) > max(
                    1e-7, abs(float(contract[field])) * 1e-10):
                return False, f"{field} differs from the frozen contract"
        except (TypeError, ValueError, KeyError, OverflowError):
            return False, f"{field} missing from the frozen contract"
    if contract.get("below_floor") or not (1.0 <= float(contract["ratio"]) <= 4.0):
        return False, "TP:SL below 1:1 or above 4:1 — watch-only"
    return True, ""


def _live_feed_signal(sig: dict) -> bool:
    """Only an explicitly confirmed live Binance tape may authorize orders."""
    return sig.get("feed_mode") == "binance" and sig.get("feed_simulated") is False


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
    if not _live_feed_signal(sig):
        return False, "live Binance feed required — simulated or unknown tape cannot execute"
    if not sig.get("issued"):
        return False, "signal not ISSUED (below confidence floor — no trade)"
    if not sig.get("direction"):
        return False, "no direction"
    if not sig.get("execution_eligible"):
        return False, "cycle is not execution-eligible (watch-only or barrier already hit)"
    valid_contract, why = _valid_contract_signal(sig)
    if not valid_contract:
        return False, why
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
               "contract_id": sig.get("contract_id"),
               "cycle_id": sig.get("cycle_id"),
               "lock_id": sig.get("lock_id"),
               "level_revision": sig.get("level_revision"),
               "feed_mode": sig.get("feed_mode"),
               "feed_simulated": sig.get("feed_simulated"),
               "testnet": bool(self.cfg.get("testnet", True))}
        _log(row)
        if not ok:
            return row
        try:
            row["order"] = self.execute(sig)
            self.orders_today += 1
            o = row["order"]
            # The position record must carry what a close actually needs:
            # the FILLED quantity and the two bracket order ids. Without
            # them the app could log a settlement but had no way to send a
            # reduce-only close, so the position stayed open at the
            # exchange with orphaned TP/SL orders attached.
            self.open_positions[sig["symbol"]] = {
                "tf": sig.get("tf"), "entry": sig.get("entry"),
                "side": sig.get("direction"),
                "window_end": sig.get("window_end"),
                "qty": o.get("qty"), "sl_id": o.get("sl_id"),
                "tp_id": o.get("tp_id"),
                "tp_px": o.get("tp_px"), "sl_px": o.get("sl_px"),
                "exchange_closed": False}
        except Exception as exc:                       # never crash engine
            row["ok"] = False
            row["why"] = f"order failed: {exc}"[:200]
            _log(row)
        return row

    def execute(self, sig: dict) -> dict:
        # Defense in depth: even internal callers may not bypass the same
        # account, live-feed, barrier and frozen-contract gates as consider().
        ok, why = gate(self.cfg, sig, set(self.open_positions), self.orders_today)
        if not ok:
            raise ValueError(why)
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
        # advanced sizing dampener (HAR-RV vol + EVT tail + RMT
        # market mode): can only shrink the order, never lever it up
        _sm = float(sig.get("size_mult") or 1.0)
        notional *= max(0.25, min(1.0, _sm))
        # EVT catastrophic-stop bound: the protective stop actually
        # placed is never wider than the measured 1-in-100 tail
        _sl_price = float(sig["sl"])
        _max_bps = float(sig.get("sl_max_bps") or 0.0)
        if _max_bps > 0 and _sl_price > 0:
            _dist = abs(entry - _sl_price) / entry * 1e4
            if _dist > _max_bps:
                _sl_price = entry * (1 - sig["direction"]
                                     * _max_bps / 1e4)
        # ── EXCHANGE FILTERS ─────────────────────────────────────────
        # `round(notional / entry, 6)` and `round(price, 2)` are not
        # orders, they are rejections. Binance enforces LOT_SIZE %stepSize,
        # PRICE_FILTER %tickSize and notional >= minNotional per symbol; at
        # the old 25 USDT default BTC failed all three (qty 0.000366 vs a
        # 0.001 step, stop 67590.76 vs a 0.1 tick, 25 USDT vs the 100 USDT
        # floor). Every value is now rounded DOWN to a legal one, so the
        # sent order can only be smaller than the risk already approved.
        _ord = legal_order(sym, entry, notional / entry,
                           int(sig["direction"]), tp=float(sig["tp"]),
                           sl=_sl_price, budget_notional=notional)
        if not _ord["ok"]:
            raise ValueError(
                "order would be rejected by the exchange: "
                + "; ".join(_ord["why"])
                + f" (max_order_usdt must be at least "
                  f"{_ord['filters']['min_notional']:g} for {sym})")
        qty = _ord["qty"]
        side = "BUY" if sig["direction"] > 0 else "SELL"
        close = "SELL" if side == "BUY" else "BUY"
        entry_o = self._signed("POST", "/fapi/v1/order", {
            "symbol": sym, "side": side, "type": "MARKET",
            "quantity": qty})
        sl_o = self._signed("POST", "/fapi/v1/order", {
            "symbol": sym, "side": close, "type": "STOP_MARKET",
            "stopPrice": _ord["sl"], "closePosition": "true"})
        tp_o = self._signed("POST", "/fapi/v1/order", {
            "symbol": sym, "side": close, "type": "TAKE_PROFIT_MARKET",
            "stopPrice": _ord["tp"], "closePosition": "true"})
        return {"symbol": sym, "qty": qty, "entry_id": entry_o.get("orderId"),
                "sl_id": sl_o.get("orderId"), "tp_id": tp_o.get("orderId"),
                "filters": _ord["filters"], "tp_px": _ord["tp"],
                "sl_px": _ord["sl"]}

    def _cancel(self, symbol: str, order_id) -> dict | None:
        """Cancel one resting order. Never raises — a dead order is not news."""
        try:
            return self._signed("DELETE", "/fapi/v1/order",
                                {"symbol": symbol, "orderId": order_id})
        except Exception as exc:                       # pragma: no cover
            return {"error": str(exc)[:120]}

    def reconcile(self, symbol: str, price: float) -> dict | None:
        """Close a position AT THE EXCHANGE, not just in this account.

        `close_position` used to pop the local book and log a P&L estimate
        while the position stayed open on Binance with its brackets
        attached. That is the worst possible failure for an execution
        layer: the app reports "closed", the exchange reports an open
        position, and the daily order cap plus the one-position-per-symbol
        gate then keep the app from ever managing it again.

        This sends a real reduce-only MARKET order, then cancels the TP/SL
        brackets (which would otherwise sit as orphan orders). It is
        deliberately separate from close_position so the settlement path
        cannot accidentally start trading.
        """
        pos = self.open_positions.get(symbol)
        if not pos:
            return None
        sym = SYMBOLS.get(symbol, symbol)
        side = int(pos.get("side") or 1)
        qty = float(pos.get("qty") or 0.0)
        if qty <= 0:
            return {"ok": False, "why": "no filled quantity on this position",
                    "symbol": sym, "closed": False}
        close_side = "SELL" if side > 0 else "BUY"
        intent = {"ts": time.time(), "symbol": symbol, "tf": pos.get("tf"),
                  "side": side, "entry": pos.get("entry"),
                  "exit": price, "close_side": close_side, "qty": qty}
        try:
            o = self._signed("POST", "/fapi/v1/order", {
                "symbol": sym, "side": close_side, "type": "MARKET",
                "quantity": qty, "reduceOnly": "true"})
        except Exception as exc:
            row = {**intent, "ok": False, "closed": False,
                   "why": f"close order rejected: {exc}"[:200]}
            _log(row)
            return row
        # The position is flat now. Remove the brackets so they cannot
        # later fire against a position that no longer exists.
        cancelled = []
        for k in ("sl_id", "tp_id"):
            oid = pos.get(k)
            if oid:
                self._cancel(sym, oid)
                cancelled.append(oid)
        self.open_positions.pop(symbol, None)
        row = {**intent, "ok": True, "closed": True,
               "close_order_id": (o or {}).get("orderId"),
               "cancelled_brackets": cancelled,
               "why": "position closed at the exchange (reduce-only market)"}
        _log(row)
        return row

    def close_position(self, symbol: str, exit_price: float,
                       result: str | None) -> dict | None:
        """Settlement reached: record the outcome of an open position.

        This books the RESULT only. It deliberately does not send an order
        — the settlement path must never become a trade trigger. A caller
        that needs the position actually flattened must call
        ``reconcile()`` first, which is the exchange-facing half.
        """
        pos = self.open_positions.pop(symbol, None)
        if not pos:
            return None
        entry = float(pos.get("entry") or 0)
        side = int(pos.get("side") or 0)
        pnl_bps = round(side * (exit_price / entry - 1) * 1e4, 1) \
            if entry > 0 and exit_price > 0 else None
        row = {"ts": time.time(), "symbol": symbol, "tf": pos.get("tf"),
               "side": side, "entry": entry, "exit": exit_price,
               "ok": True, "why": f"position settled ({result})",
               "pnl_bps_est": pnl_bps, "closed": True,
               "exchange_closed": bool(pos.get("exchange_closed"))}
        _log(row)
        return row

    def status(self) -> dict:
        self.cfg = load_cfg()
        self._roll_day()
        return {"config": public_cfg(self.cfg),
                "open_positions": self.open_positions,
                "orders_today": self.orders_today,
                "log": read_log(12)}
