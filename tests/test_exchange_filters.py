"""Every order the app publishes must survive the exchange's filters.

WHY THIS FILE EXISTS
--------------------
``app/trading.py`` used to size and price orders like this::

    qty = round(notional / entry, 6)
    "stopPrice": round(float(_sl_price), 2)

Neither value is legal on Binance USDⓈ-M futures, which enforces three
per-symbol filters. At the app's then-default ``max_order_usdt = 25.0``
with BTC near 68273 and real BTCUSDT filters (tickSize 0.1,
stepSize 0.001, minNotional 100):

    qty     0.000366   not a multiple of 0.001   -> LOT_SIZE reject
    stop    67590.76   not a multiple of 0.1     -> PRICE_FILTER reject
    notional 25 USDT   below 100                 -> MIN_NOTIONAL reject

All three failed, on every symbol, at every default setting. The user's
standing rule is that the app must publish an order actually executable
at the broker; a rejected order is not one.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import exchange_filters as ef  # noqa: E402
from app import trading  # noqa: E402
from tests.contract_fixtures import locked_contract_signal  # noqa: E402


# ───────────────────────── the filter primitives ─────────────────────────
def test_floor_never_rounds_up():
    f = ef.Filters(tick_size=0.1, step_size=0.001, min_qty=0.001,
                   min_notional=100.0)
    for q in (0.0019, 0.002999, 1.2345678):
        assert f.floor_step(q) <= q + 1e-12
    for p in (67590.76, 68273.5, 0.09):
        assert f.floor_tick(p) <= p + 1e-12


def test_floor_step_produces_a_legal_quantity():
    f = ef.Filters(step_size=0.001, min_qty=0.001, min_notional=100.0)
    q = f.floor_step(0.0349999)
    assert f.qty_ok(q)
    assert q == pytest.approx(0.034)


def test_floor_tick_produces_a_legal_price():
    f = ef.Filters(tick_size=0.1)
    p = f.floor_tick(67590.76)
    assert f.price_ok(p)
    assert p == pytest.approx(67590.7)


def test_the_exact_values_the_app_used_to_send_are_illegal():
    """The regression itself, pinned as numbers."""
    f = ef.Filters(tick_size=0.1, step_size=0.001, min_qty=0.001,
                   min_notional=100.0)
    old_qty = round(25.0 / 68273.5, 6)          # 0.000366
    old_stop = round(68273.5 * 0.99, 2)         # 67590.76
    assert not f.qty_ok(old_qty)
    assert not f.price_ok(old_stop)
    assert not f.notional_ok(old_qty, 68273.5)
    # and the legal replacements are accepted
    assert f.qty_ok(f.floor_step(150.0 / 68273.5))
    assert f.price_ok(f.floor_tick(old_stop))


# ───────────────────────── the effective floor ─────────────────────────
def test_effective_floor_exceeds_the_raw_filter_for_btc():
    """minNotional is a USDT filter; the order floor is a step multiple.

    BTC: 100 USDT / 68000 / 0.001 is not a whole number of steps, so the
    smallest placeable order is 0.002 BTC = 136 USDT, not 100.
    """
    raw = ef.BUNDLED["BTCUSDT"].min_notional
    eff = ef.min_order_usdt("BTC")
    assert eff > raw
    assert eff == pytest.approx(136.0, abs=1.0)


def test_shipped_default_clears_every_symbol_floor():
    """The default max_order_usdt must be receivable on all four."""
    floor = ef.cheapest_legal_notional()
    assert trading.DEFAULTS["max_order_usdt"] >= floor, (
        f"default {trading.DEFAULTS['max_order_usdt']} sits below the "
        f"{floor} USDT floor — no order could be placed at all")
    for sym in ("BTC", "ETH", "BNB", "PAXG"):
        assert ef.min_order_usdt(sym) <= trading.DEFAULTS["max_order_usdt"]


def test_the_old_default_of_25_was_unfillable_on_every_symbol():
    """Not always for the same reason — but never fillable.

    BTC failed all three filters at once. ETH/BNB/PAXG cleared their
    20 USDT minNotional but still failed LOT_SIZE, because
    round(25 / price, 6) is not a multiple of the 0.001 step. The bug was
    never one filter; it was that neither quantity nor price was ever
    made filter-aware.
    """
    prices = {"BTCUSDT": 68273.5, "ETHUSDT": 2450.0,
              "BNBUSDT": 439.45, "PAXGUSDT": 3735.0}
    for sym, px in prices.items():
        f = ef.for_symbol(sym)
        old_qty = round(25.0 / px, 6)
        assert not f.qty_ok(old_qty), (sym, old_qty, f.step_size)
        # and BTC additionally busts the notional floor
        if sym == "BTCUSDT":
            assert not f.notional_ok(old_qty, px)


# ───────────────────────── the order path ─────────────────────────
def _exec(symbol, px, budget, tp_bps=120.0, sl_bps=60.0):
    tr = trading.Trader()
    tr.cfg = {**trading.DEFAULTS, "enabled": True, "armed_until": 9e9,
              "max_orders_day": 4, "testnet": True, "api_key": "k",
              "api_secret": "s", "max_order_usdt": budget}
    sent: list[dict] = []
    tr._signed = lambda m, p, params: (sent.append(dict(params))
                                       or {"orderId": 1})
    sig = locked_contract_signal(symbol=symbol, entry=px, tp_bps=tp_bps,
                                 sl_bps=sl_bps,
                                 cycle_id=f"{symbol}-{budget}-{px}")
    tr.execute(sig)
    return sent


@pytest.mark.parametrize("symbol,px", [("BTC", 68273.5), ("ETH", 2450.0),
                                       ("BNB", 439.45), ("PAXG", 3735.0)])
def test_every_sent_order_is_legal(symbol, px):
    sent = _exec(symbol, px, trading.DEFAULTS["max_order_usdt"])
    f = ef.for_symbol(symbol)
    qty = float(sent[0]["quantity"])
    sl = float(sent[1]["stopPrice"])
    tp = float(sent[2]["stopPrice"])
    assert f.qty_ok(qty), (symbol, qty, f.step_size)
    assert f.price_ok(sl), (symbol, sl, f.tick_size)
    assert f.price_ok(tp), (symbol, tp, f.tick_size)
    assert f.notional_ok(qty, px), (symbol, qty * px, f.min_notional)


def test_an_unfillable_budget_is_refused_not_sent():
    """Better a stated refusal than a silent exchange rejection."""
    with pytest.raises(ValueError) as e:
        _exec("BTC", 68273.5, 25.0)
    msg = str(e.value)
    assert "rejected by the exchange" in msg
    # and the message says what to change
    assert "max_order_usdt" in msg and "BTCUSDT" in msg


def test_sizing_steps_up_to_the_minimum_rather_than_dropping_under_it():
    """Rounding down must not turn a fillable budget into a reject."""
    sent = _exec("BTC", 68273.5, 150.0)
    qty = float(sent[0]["quantity"])
    # 150/68273.5 = 0.002197 -> floor 0.002, which IS legal (136 USDT)
    assert qty == pytest.approx(0.002)
    assert qty * 68273.5 >= 100.0


def test_stop_moves_away_from_entry_never_toward_it():
    """Rounding a stop toward entry would tighten risk after sizing."""
    sent = _exec("ETH", 2450.0, 150.0, sl_bps=60.0)
    sl = float(sent[1]["stopPrice"])
    assert sl <= 2450.0 * (1 - 60 / 1e4) + 1e-9   # long: stop rounds down


def test_short_stop_rounds_up():
    tr = trading.Trader()
    tr.cfg = {**trading.DEFAULTS, "enabled": True, "armed_until": 9e9,
              "max_orders_day": 4, "testnet": True, "api_key": "k",
              "api_secret": "s", "max_order_usdt": 150.0}
    sent: list[dict] = []
    tr._signed = lambda m, p, params: (sent.append(dict(params))
                                       or {"orderId": 1})
    sig = locked_contract_signal(symbol="ETH", side=-1, entry=2450.0,
                                 tp_bps=120.0, sl_bps=60.0,
                                 cycle_id="short-legal")
    tr.execute(sig)
    sl = float(sent[1]["stopPrice"])
    assert sl >= 2450.0 * (1 + 60 / 1e4) - 1e-9, (sl, sent[1])


def test_filters_are_attached_to_the_order_result():
    tr = trading.Trader()
    tr.cfg = {**trading.DEFAULTS, "enabled": True, "armed_until": 9e9,
              "max_orders_day": 4, "testnet": True, "api_key": "k",
              "api_secret": "s", "max_order_usdt": 150.0}
    tr._signed = lambda m, p, params: {"orderId": 1}
    sig = locked_contract_signal(symbol="ETH", entry=2450.0,
                                 cycle_id="filters-attached")
    out = tr.execute(sig)
    assert out["filters"]["step_size"] > 0
    assert out["filters"]["tick_size"] > 0
    assert out["sl_px"] and out["tp_px"]


# ───────────────────────── live / offline behaviour ─────────────────────
def test_offline_startup_still_produces_legal_orders(monkeypatch):
    """No network is not an excuse to send an illegal order."""
    monkeypatch.setattr(ef, "_CACHE", None)
    monkeypatch.setattr(ef, "_CACHE_AT", 0.0)

    def boom(*a, **k):
        raise OSError("no network")

    table = ef.refresh(transport=boom, force=True)
    assert table["BTCUSDT"].source == "bundled default"
    assert table["BTCUSDT"].min_notional == 100.0
    r = ef.legal_order("BTCUSDT", 68273.5, 150.0 / 68273.5, 1)
    assert r["ok"] and r["filters"]["source"] == "bundled default"


def test_live_filters_override_the_bundled_defaults(monkeypatch):
    monkeypatch.setattr(ef, "_CACHE", None)
    monkeypatch.setattr(ef, "_CACHE_AT", 0.0)
    payload = {"symbols": [{
        "symbol": "BTCUSDT",
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.50"},
            {"filterType": "LOT_SIZE", "stepSize": "0.010", "minQty": "0.010"},
            {"filterType": "MIN_NOTIONAL", "minNotional": "250.0"},
        ]}]}
    table = ef.refresh(transport=lambda *a: payload, force=True)
    f = table["BTCUSDT"]
    assert f.tick_size == 0.5 and f.step_size == 0.01
    assert f.min_notional == 250.0
    assert f.source == "exchangeInfo"
    # a 150 USDT order is no longer legal under the live schedule
    assert not ef.legal_order("BTCUSDT", 68273.5, 150.0 / 68273.5, 1)["ok"]


# ──────────────── the position must actually close at the exchange ────────
def _open_btc():
    """Open one real, legal BTC position and return (trader, calls)."""
    import app.trading as tr_mod
    trader = trading.Trader()
    cfg = {**trading.DEFAULTS, "enabled": True, "armed_until": 9e9,
           "max_orders_day": 4, "testnet": True, "api_key": "k",
           "api_secret": "s", "max_order_usdt": 150.0}
    trader.cfg = cfg
    tr_mod.load_cfg = lambda: cfg
    calls: list[tuple] = []
    trader._signed = lambda m, p, params: (
        calls.append((m, p, dict(params))) or {"orderId": len(calls)})
    sig = locked_contract_signal(symbol="BTC", entry=68273.5,
                                 cycle_id="reconcile-test")
    row = trader.consider(sig)
    assert row["ok"], row["why"]
    return trader, calls


def test_an_open_position_remembers_how_to_be_closed():
    """qty and both bracket ids must survive the fill.

    Without them the app could log a settlement but had no way to send a
    reduce-only close, so the position stayed open at the exchange.
    """
    trader, _ = _open_btc()
    pos = trader.open_positions["BTC"]
    assert pos["qty"] == pytest.approx(0.002)
    assert pos["sl_id"] and pos["tp_id"]
    assert pos["sl_px"] and pos["tp_px"]


def test_reconcile_sends_a_reduce_only_close_and_cancels_brackets():
    trader, calls = _open_btc()
    n_before = len(calls)
    out = trader.reconcile("BTC", 68900.0)
    assert out["ok"] and out["closed"]
    new = calls[n_before:]
    # exactly one closing order, reduce-only market, opposite side
    markets = [c for c in new if c[2].get("type") == "MARKET"]
    assert len(markets) == 1
    m = markets[0]
    assert m[2]["side"] == "SELL"          # long position closes with a sell
    assert m[2]["reduceOnly"] == "true"    # never opens a new position
    assert m[2]["quantity"] == pytest.approx(0.002)
    # and both brackets are cancelled so they cannot fire later
    deletes = [c for c in new if c[0] == "DELETE"]
    assert len(deletes) == 2
    assert out["cancelled_brackets"]
    assert "BTC" not in trader.open_positions


def test_reconcile_does_not_pop_the_book_when_the_close_is_rejected():
    """A failed close must leave the position visible, not forgotten."""
    trader, calls = _open_btc()

    def boom(method, path, params):
        if params.get("type") == "MARKET":
            raise OSError("insufficient margin")
        return {"orderId": 99}

    trader._signed = boom
    out = trader.reconcile("BTC", 68900.0)
    assert out["ok"] is False and not out["closed"]
    assert "rejected" in out["why"]
    assert "BTC" in trader.open_positions, (
        "the position was dropped from the book even though the exchange "
        "never confirmed the close")


def test_close_position_alone_never_sends_an_order():
    """Settlement books the result; it must not become a trade trigger."""
    trader, calls = _open_btc()
    n_before = len(calls)
    row = trader.close_position("BTC", 68900.0, "target")
    assert len(calls) == n_before, (
        "close_position sent an order — the settlement path must never "
        "become a trade trigger")
    assert row["pnl_bps_est"] is not None
    assert row["exchange_closed"] is False   # honestly not flattened
