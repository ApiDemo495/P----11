"""Exchange symbol filters — the difference between an order and a rejection.

WHY THIS MODULE EXISTS
----------------------
``app/trading.py`` used to build orders like this::

    qty = round(notional / entry, 6)
    "stopPrice": round(float(_sl_price), 2)

Both lines are wrong on Binance USDⓈ-M futures, which enforces three
separate per-symbol filters:

    LOT_SIZE      quantity % stepSize == 0, and quantity >= minQty
    PRICE_FILTER  price % tickSize == 0
    MIN_NOTIONAL  price * quantity >= minNotional

Measured on real BTCUSDT futures (tickSize 0.1, stepSize 0.001,
minNotional 100.0 USDT) at the app's default ``max_order_usdt = 25.0``
and a BTC price near 68273::

    qty  = round(25 / 68273, 6)   = 0.000366   -> not a multiple of 0.001
    stop = round(68273 * 0.99, 2) = 67590.76   -> not a multiple of 0.1
    notional = 25 USDT                          -> below the 100 USDT floor

Three of three filters failed. The entry order is rejected (so no trade
happens at all), the protective stop is rejected (so a position that did
open would be unbracketed), and the take-profit is rejected the same way.

This module makes every order *round DOWN to a legal value*, which is the
documented best practice: rounding down can only shrink an order below the
risk the sizing already approved. Rounding up would quietly exceed it.

Filters are fetched live from ``/fapi/v1/exchangeInfo`` and cached, because
exchanges change them. The bundled defaults below are the values measured
on 2026-11-27 and exist so a cold start or an offline test still produces a
legal order instead of crashing — they are labelled as defaults everywhere
they surface, never presented as the live schedule.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Any, NamedTuple


class Filters(NamedTuple):
    """The three constraints that decide whether an order is accepted."""
    tick_size: float = 0.01      # PRICE_FILTER.tickSize
    step_size: float = 0.001     # LOT_SIZE.stepSize
    min_qty: float = 0.001       # LOT_SIZE.minQty
    min_notional: float = 5.0    # MIN_NOTIONAL.minNotional
    source: str = "bundled default"
    fetched_at: float = 0.0

    # -- rounding -------------------------------------------------------
    def floor_step(self, qty: float) -> float:
        """Largest legal quantity <= qty. Never rounds up."""
        if self.step_size <= 0:
            return self.floor_qty(qty)
        n = math.floor(float(qty) / self.step_size + 1e-9)
        return self.floor_qty(n * self.step_size)

    def floor_qty(self, qty: float) -> float:
        """Trim float dust so a legal quantity is not lost to 0.1+0.2."""
        if qty <= 0:
            return 0.0
        # decimals implied by the step, then round to it
        dec = max(0, -int(math.floor(math.log10(self.step_size)))) \
            if self.step_size > 0 else 8
        return round(float(qty), min(12, dec + 2))

    def floor_tick(self, px: float) -> float:
        """Largest legal price <= px. Never rounds up."""
        if self.tick_size <= 0:
            return float(px)
        n = math.floor(float(px) / self.tick_size + 1e-9)
        dec = max(0, -int(math.floor(math.log10(self.tick_size))))
        return round(n * self.tick_size, min(12, dec))

    def ceil_tick(self, px: float) -> float:
        """Smallest legal price >= px (used only where up is the safe side)."""
        if self.tick_size <= 0:
            return float(px)
        n = math.ceil(float(px) / self.tick_size - 1e-9)
        dec = max(0, -int(math.floor(math.log10(self.tick_size))))
        return round(n * self.tick_size, min(12, dec))

    # -- verdicts -------------------------------------------------------
    def qty_ok(self, qty: float) -> bool:
        if qty <= 0 or qty < self.min_qty - 1e-12:
            return False
        if self.step_size <= 0:
            return True
        n = qty / self.step_size
        return abs(n - round(n)) < 1e-6

    def price_ok(self, px: float) -> bool:
        if px <= 0 or self.tick_size <= 0:
            return px > 0
        n = px / self.tick_size
        return abs(n - round(n)) < 1e-6

    def notional_ok(self, qty: float, px: float) -> bool:
        return float(qty) * float(px) >= self.min_notional - 1e-9

    def describe(self) -> dict[str, Any]:
        return {"tick_size": self.tick_size, "step_size": self.step_size,
                "min_qty": self.min_qty, "min_notional": self.min_notional,
                "source": self.source,
                "fetched_at": self.fetched_at or None}


#: Measured 2026-11-27. Used when /exchangeInfo cannot be reached.
BUNDLED: dict[str, Filters] = {
    "BTCUSDT": Filters(0.1, 0.001, 0.001, 100.0),
    "ETHUSDT": Filters(0.01, 0.001, 0.001, 20.0),
    "BNBUSDT": Filters(0.01, 0.01, 0.01, 20.0),
    "PAXGUSDT": Filters(0.01, 0.001, 0.001, 20.0),
}

#: The cheapest legal order on the majors. BTC alone needs 100 USDT.
DEFAULT_MIN_NOTIONAL = 100.0

_CACHE: dict[str, Filters] | None = None
_CACHE_AT = 0.0
_TTL_S = 3600.0
_lock = threading.RLock()


def _parse(sym: str, filters: list[dict]) -> Filters | None:
    got: dict[str, float] = {}
    for f in filters or []:
        t = str(f.get("filterType") or "").upper()
        try:
            if t == "PRICE_FILTER":
                got["tick_size"] = float(f.get("tickSize") or 0.0)
            elif t == "LOT_SIZE":
                got["step_size"] = float(f.get("stepSize") or 0.0)
                got["min_qty"] = float(f.get("minQty") or 0.0)
            elif t in ("MIN_NOTIONAL", "NOTIONAL"):
                got["min_notional"] = float(
                    f.get("minNotional") or f.get("notional") or 0.0)
        except (TypeError, ValueError):
            continue
    if not got:
        return None
    base = BUNDLED.get(sym, Filters())
    return Filters(
        tick_size=got.get("tick_size") or base.tick_size,
        step_size=got.get("step_size") or base.step_size,
        min_qty=got.get("min_qty") or base.min_qty,
        min_notional=got.get("min_notional") or base.min_notional,
        source="exchangeInfo", fetched_at=time.time())


def refresh(transport=None, force: bool = False) -> dict[str, Filters]:
    """Pull live filters. Falls back to BUNDLED on any failure.

    ``transport(method, url, params, headers)`` matches the Trader's own
    injectable transport, so this is testable without a network.
    """
    global _CACHE, _CACHE_AT
    with _lock:
        if (_CACHE is not None and not force
                and time.time() - _CACHE_AT < _TTL_S):
            return dict(_CACHE)
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    doc: Any = None
    try:
        if transport is not None:
            doc = transport("GET", url, {}, {})
        else:
            import json
            import urllib.request
            with urllib.request.urlopen(url, timeout=8) as r:
                doc = json.loads(r.read().decode())
    except Exception:
        doc = None
    out: dict[str, Filters] = dict(BUNDLED)
    for s in ((doc or {}).get("symbols") or []):
        name = str(s.get("symbol") or "")
        if name not in BUNDLED:
            continue
        parsed = _parse(name, s.get("filters") or [])
        if parsed is not None:
            out[name] = parsed
    with _lock:
        _CACHE, _CACHE_AT = dict(out), time.time()
    return dict(out)


def for_symbol(symbol: str, transport=None) -> Filters:
    """Filters for one futures symbol, live if possible, bundled if not."""
    sym = str(symbol).upper()
    if not sym.endswith("USDT"):
        sym = sym + "USDT"
    table = refresh(transport)
    return table.get(sym) or BUNDLED.get(sym) or Filters()


#: Reference prices for turning a notional floor into a *quantity* floor.
#: The exchange states minNotional in USDT and stepSize in coin, so the
#: smallest order you can actually place is
#:     ceil(minNotional / price / stepSize) * stepSize * price
#: which is strictly greater than minNotional whenever that division is not
#: already a step multiple. BTC is the clean example: minNotional 100 USDT
#: with stepSize 0.001 at ~68273 a coin needs 0.002 BTC = 136.5 USDT, so a
#: 100 USDT budget cannot be filled at all. Quoting "the minimum is 100"
#: would be true of the filter and false of the order.
REFERENCE_PRICE = {"BTCUSDT": 68000.0, "ETHUSDT": 2450.0,
                   "BNBUSDT": 440.0, "PAXGUSDT": 3735.0}


def min_order_usdt(symbol: str = "BTC", price: float | None = None,
                   transport=None) -> float:
    """The smallest notional that will actually be ACCEPTED, in USDT."""
    sym = str(symbol).upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    f = for_symbol(sym, transport)
    px = float(price or REFERENCE_PRICE.get(sym) or 1.0)
    if px <= 0 or f.step_size <= 0:
        return float(f.min_notional)
    need = f.floor_step(f.min_notional / px)
    if need * px < f.min_notional - 1e-9:
        need = f.floor_qty(need + f.step_size)
    return round(max(float(f.min_notional), need * px), 2)


def cheapest_legal_notional(transport=None) -> float:
    """The dearest of the per-symbol effective floors — a safe default."""
    table = refresh(transport)
    return max(min_order_usdt(sym, transport=transport) for sym in table)


def legal_order(symbol: str, entry: float, qty: float, side: int,
                tp: float | None = None, sl: float | None = None,
                budget_notional: float | None = None,
                transport=None) -> dict:
    """Make an order legal, or say exactly why it cannot be.

    Returns ``{"ok", "qty", "tp", "sl", "why", "filters"}``. Every price and
    quantity is rounded DOWN, so the result never exceeds the risk the
    caller already approved. Stop and target are rounded on the side that
    keeps them *away* from the entry (a long's stop moves down, a long's
    target moves up) — shrinking a stop toward the entry would tighten the
    risk after sizing, and widening a target never makes a trade worse.
    """
    f = for_symbol(symbol, transport)
    entry = float(entry or 0.0)
    qty = f.floor_step(float(qty or 0.0))
    # Rounding DOWN a step can drop a legal order under minNotional: BTC at
    # a 100 USDT budget and 68273 a coin wants 0.001465, and flooring to the
    # 0.001 step leaves 68 USDT, which the exchange rejects. When the budget
    # can still afford it, step UP to the smallest legal quantity instead —
    # never above what the caller approved.
    if entry > 0 and qty > 0 and not f.notional_ok(qty, entry):
        up = f.floor_step(f.min_notional / entry)
        if up * entry < f.min_notional - 1e-9:
            up = f.floor_qty(up + f.step_size)
        cap = (float(budget_notional) if budget_notional
               else float("inf"))
        if up * entry >= f.min_notional - 1e-9 and up * entry <= cap + 1e-9:
            qty = up
    d = 1 if int(side) >= 0 else -1
    out_tp = out_sl = None
    if tp is not None and float(tp) > 0:
        # away from entry = up for a long, down for a short
        out_tp = (f.ceil_tick(float(tp)) if d > 0
                  else f.floor_tick(float(tp)))
    if sl is not None and float(sl) > 0:
        # away from entry = down for a long, up for a short
        out_sl = (f.floor_tick(float(sl)) if d > 0
                  else f.ceil_tick(float(sl)))
    why: list[str] = []
    if qty <= 0:
        why.append(
            f"quantity rounds to 0 at stepSize {f.step_size:g} "
            f"(notional too small for {symbol})")
    elif qty < f.min_qty - 1e-12:
        why.append(f"quantity {qty:g} below minQty {f.min_qty:g}")
    if entry > 0 and qty > 0 and not f.notional_ok(qty, entry):
        why.append(
            f"notional {qty * entry:.2f} USDT below the {f.min_notional:g} "
            f"USDT minimum for {symbol}")
    return {"ok": not why, "qty": qty, "tp": out_tp, "sl": out_sl,
            "why": why, "filters": f.describe()}
