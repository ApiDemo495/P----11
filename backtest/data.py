"""Data pipeline: CCXT fetch + CSV cache; synthetic smoke data.

Real runs use CCXT (Binance) and cache every pull under
backtest/cache/ so a series is downloaded once. `sim_bars` exists
ONLY so the harness mechanics can be smoke-tested offline; its
output is labelled SYNTHETIC everywhere and is never a result.
"""
from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Bar:
    t: int          # epoch seconds (bar open)
    o: float
    h: float
    l: float
    c: float
    v: float = 0.0


SYMBOLS = {"BTC": "BTC/USDT", "ETH": "ETH/USDT",
           "BNB": "BNB/USDT", "PAXG": "PAXG/USDT"}

# §18.1 per-side friction in bps of entry, charged once per side (era
# accounting charges 2x this per era = the full round trip).
#
# THE VENUE IS PART OF THE MODEL, NOT AN ASSUMPTION (fixed 2026-11-27).
# The stack used to be hardcoded to Binance SPOT taker: commission 10
# (VIP0) + slippage + half the Roll(1984) spread. But `app/trading.py`
# routes with ``"type": "MARKET"`` and ``"closePosition": "true"`` on
# ``/fapi/v1/order`` — i.e. Binance USDⓈ-M FUTURES, whose real taker fee
# is 0.05 % = 5 bps, not 10. Every EV number in this repo was therefore
# charged 18.9 bps a round trip MORE than the order path actually pays.
# The repo already recorded the correct figures in
# ``backtest/results/winrate_venue.json``; they were simply never wired
# into the cost model.
#
# VENUE_RT_BPS: full round trip, taker/taker, per venue. Sources are the
# ones recorded in winrate_venue.json (venue fee schedules, 2026).
VENUE_RT_BPS: dict[str, float] = {
    "lighter": 0.0,          # docs.lighter.xyz, Standard tier (2026-08)
    "dydx_btc": 0.0,         # dYdX v4, BTC-USD fees waived
    "mexc_fut": 4.0,         # MEXC futures 0 %/0.02 %
    "extended": 5.0,         # Extended 0 %/0.025 % (Starknet)
    "hyperliquid4": 8.0,     # Hyperliquid T4 0 %/0.04 % (needs $500M/14d)
    "aster": 8.0,            # Aster VIP1 0 %/0.04 %
    "binance_fut": 10.0,     # Binance USDⓈ-M 0.02 %/0.05 % — THE ORDER PATH
    "binance_spot": 20.0,    # Binance spot 0.10 %/0.10 % taker
    "legacy_spot": 28.872,   # the old hardcoded stack incl. Roll spread
}
#: What the app's own order path pays: ``app/trading.py`` posts MARKET
#: orders to ``/fapi/v1/order``, so both legs are futures takers.
#: Override with ``VENUE=binance_spot`` (or any key above) to price a
#: different execution; ``PENNY_VENUE=1`` prices the cheapest schedule the
#: repo has evidence for.
DEFAULT_VENUE = "binance_fut"

# per-side slippage beyond the fee, measured from the tape (bps of entry)
SLIPPAGE_BPS = {"BTC": 1.0, "ETH": 1.0, "BNB": 2.0, "PAXG": 5.0}


def venue_rt(symbol: str) -> float:
    """Full round trip for this asset on the configured venue, in bps.

    An unrecognised symbol gets the MOST expensive measured slippage, not
    the cheapest: a cost model may fail safe (over-charge) but must never
    fail flattering.  (It previously inherited a 2 bps default and so
    priced an unknown asset BELOW BTC.)
    """
    name = os.environ.get("VENUE") or DEFAULT_VENUE
    if os.environ.get("PENNY_VENUE") == "1":
        name = "lighter"
    rt = float(VENUE_RT_BPS.get(name, VENUE_RT_BPS[DEFAULT_VENUE]))
    slip = SLIPPAGE_BPS.get(str(symbol).upper())
    if slip is None:
        slip = max(SLIPPAGE_BPS.values())
    return rt + 2.0 * float(slip)


def venue_name() -> str:
    name = os.environ.get("VENUE") or DEFAULT_VENUE
    return "lighter" if os.environ.get("PENNY_VENUE") == "1" else name


#: per-side friction derived from the configured venue + measured slippage
FRICTION_BPS = {s: venue_rt(s) / 2.0 for s in SLIPPAGE_BPS}
# full round trip per asset (bps) — the real "cost of a trade"
ROUND_TRIP_BPS = {s: venue_rt(s) for s in SLIPPAGE_BPS}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def _cache_path(symbol: str, tf: str) -> str:
    return os.path.join(CACHE_DIR, f"{symbol}_{tf}.csv")


def save_cache(symbol: str, tf: str, bars: list[Bar]) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(symbol, tf)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "o", "h", "l", "c", "v"])
        for b in bars:
            w.writerow([b.t, b.o, b.h, b.l, b.c, b.v])
    return path


def load_cache(symbol: str, tf: str) -> list[Bar]:
    path = _cache_path(symbol, tf)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        r = csv.reader(f)
        next(r, None)
        return [Bar(int(float(x[0])), *(float(v) for v in x[1:6])) for x in r if x]


def fetch_ccxt(symbol: str, tf: str = "15m", limit: int = 1500,
               exchange_id: str = "binance") -> list[Bar]:
    """Fetch via CCXT and cache. Requires `pip install ccxt` and
    network access to the exchange (works in a Codespace)."""
    try:
        import ccxt
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("ccxt not installed — pip install ccxt") from e
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    raw = ex.fetch_ohlcv(SYMBOLS[symbol], timeframe=tf, limit=limit)
    bars = [Bar(int(r[0]) // 1000, float(r[1]), float(r[2]),
                float(r[3]), float(r[4]), float(r[5])) for r in raw]
    save_cache(symbol, tf, bars)
    return bars


def sim_bars(symbol: str, n: int = 1200, seed: int = 7,
             start_t: int = 1_700_000_000) -> list[Bar]:
    """SYNTHETIC GBM-with-regimes bars for offline smoke tests only."""
    rng = random.Random(hash((symbol, seed)) & 0xFFFFFFFF)
    base = {"BTC": 67000.0, "ETH": 3500.0, "BNB": 600.0, "PAXG": 2650.0}[symbol]
    vol = {"BTC": 0.0011, "ETH": 0.0014, "BNB": 0.0013, "PAXG": 0.0004}[symbol]
    price, drift, left, bars = base, 0.0, 0, []
    for i in range(n):
        if left <= 0:
            left = rng.randint(60, 300)
            drift = rng.choice([-0.0004, -0.00015, 0.0, 0.00015, 0.0004])
        left -= 1
        o = price
        price *= math.exp(rng.gauss(drift, vol))
        if symbol == "PAXG":
            price += 0.01 * (base - price)
        wick = abs(rng.gauss(0, vol)) * 0.8
        bars.append(Bar(start_t + i * 60, o,
                        max(o, price) * (1 + wick), min(o, price) * (1 - wick),
                        price, rng.lognormvariate(0, 0.5) * 100))
    return bars
