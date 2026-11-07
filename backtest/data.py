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

# per-asset round-trip friction in bps of entry (fee + slippage)
FRICTION_BPS = {"BTC": 12.0, "ETH": 12.0, "BNB": 12.0, "PAXG": 18.0}

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
