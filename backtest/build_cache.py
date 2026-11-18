"""Rebuild ``backtest/cache`` from the documented real-data mirrors.

The cache is git-ignored (it is ~93 MB of exchange tape), so a fresh clone
or a reset sandbox has no data. This script is the reproducible path back:
it reads the raw mirrors listed in ``docs/ALPHA_RESEARCH.md`` and writes
the exact CSV files ``backtest/edge_scan.py`` expects.

Raw inputs (passed on the command line, never baked into the repo):

* ``--perp-dir``  a checkout of ``KevinSherry2016/Crypto`` containing
  ``CleanData/kline{SYM}USDT.csv`` (Binance USD-M perpetual 1h klines).
* ``--spot-root`` a directory whose children are checkouts of
  ``parshant231234-eng/{BTC,ETH,BNB}USDT-Historical-Data``, each containing
  ``1m/YYYY/MM/{SYM}USDT_1m_YYYY-MM.csv`` (Binance spot 1m klines).

Usage::

    python -m backtest.build_cache --perp-dir /tmp/rebuild/mirror \\
        --spot-root /tmp/rebuild
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from datetime import datetime, timezone

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
PERP_SYMBOLS = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "LINK", "PAXG"]
SPOT_SYMBOLS = ["BTC", "ETH", "BNB"]


def _write(symbol: str, tf: str, bars: list[tuple]) -> tuple[str, int]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{symbol}_{tf}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "o", "h", "l", "c", "v"])
        for b in bars:
            w.writerow([b[0]] + [f"{x:.10g}" for x in b[1:6]])
    return path, len(bars)


def read_perp_hourly(perp_dir: str, symbol: str) -> list[tuple]:
    """Binance USD-M 1h klines, open_time stored as YYYYMMDDHHMMSS."""
    path = os.path.join(perp_dir, "CleanData", f"kline{symbol}USDT.csv")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            t = int(datetime.strptime(
                row["open_time"], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).timestamp())
            out.append((t, float(row["open"]), float(row["high"]),
                        float(row["low"]), float(row["close"]), float(row["volume"])))
    out.sort()
    return out


def read_spot_resampled(spot_dir: str, symbol: str, minutes: int) -> list[tuple]:
    """Aggregate Binance spot 1m klines onto fixed UTC ``minutes`` buckets."""
    files = sorted(glob.glob(os.path.join(spot_dir, "1m", "*", "*", "*.csv")))
    if not files:
        return []
    out: list[tuple] = []
    cur = None
    bucket = minutes * 60
    for path in files:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                ts = int(row["timestamp"]) // 1000
                o = float(row["open"]); h = float(row["high"])
                l = float(row["low"]); c = float(row["close"]); v = float(row["volume"])
                slot = ts - (ts % bucket)
                if cur is None or cur[0] != slot:
                    if cur:
                        out.append(tuple(cur))
                    cur = [slot, o, h, l, c, v]
                else:
                    cur[2] = max(cur[2], h)
                    cur[3] = min(cur[3], l)
                    cur[4] = c
                    cur[5] += v
    if cur:
        out.append(tuple(cur))
    out.sort()
    return out


def read_gold_monthly(gold_csv: str) -> list[tuple]:
    """Gold monthly benchmark (datasets/gold-prices, LBMA-style, 1833-present).

    Written as a DAILY grid by forward-filling the monthly price onto the
    first day of each month. It is a macro benchmark, not a tradable tick
    series: any strategy using it must size its holds in months and treat
    the missing intra-month path as a real limitation (see
    docs/ALPHA_RESEARCH.md).
    """
    out = []
    with open(gold_csv, newline="") as f:
        for row in csv.DictReader(f):
            ym = row["Date"]
            y, m = int(ym[:4]), int(ym[5:7])
            t = int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp())
            px = float(row["Price"])
            out.append((t, px, px, px, px, 0.0))
    out.sort()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perp-dir", required=True)
    ap.add_argument("--spot-root", required=True)
    ap.add_argument("--gold-csv", default="",
                    help="optional path to datasets/gold-prices data/monthly.csv")
    args = ap.parse_args()

    for sym in PERP_SYMBOLS:
        bars = read_perp_hourly(args.perp_dir, sym)
        if bars:
            print(_write(sym, "perp_1h", bars))
        else:
            print(f"!! missing perp {sym}")

    for sym in SPOT_SYMBOLS:
        root = os.path.join(args.spot_root, f"{sym.lower()}1m")
        for minutes, tf in ((15, "spot_15m"), (30, "spot_30m"),
                            (60, "spot_1h"), (240, "spot_4h")):
            bars = read_spot_resampled(root, sym, minutes)
            if bars:
                print(_write(sym, tf, bars))
            else:
                print(f"!! missing spot {sym} {minutes}m (looked in {root})")
    if args.gold_csv:
        bars = read_gold_monthly(args.gold_csv)
        if bars:
            print(_write("GOLD", "monthly", bars))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
