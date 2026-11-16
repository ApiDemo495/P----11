"""Download real Binance 1h klines from the GitHub mirror of Binance Vision.

Why this exists: the sandbox (and some Codespaces regions) cannot reach
`data.binance.vision` directly, but GitHub can.  The mirror
`KevinSherry2016/Crypto` holds byte-identical daily CSV exports under
`marketData/unzipped/{SYMBOL}/{SYMBOL}-1h-{DATE}/{SYMBOL}-1h-{DATE}.csv`.

Each row carries `taker_buy_volume` / `quote_volume` / `count`, which is
what makes real order-flow features (taker imbalance, average trade size)
computable **offline** instead of only from the live book.

Usage
    python3 scripts/fetch_binance_mirror.py                       # defaults
    python3 scripts/fetch_binance_mirror.py --start 2026-01-01 --end 2026-09-30
    python3 scripts/fetch_binance_mirror.py --symbols BTC ETH

Writes one cache file per symbol: `backtest/cache/{SYM}_1h.csv` with the
app's column order (t,o,h,l,c,v) plus the extra order-flow columns.
Synthetic data is never written here.
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "KevinSherry2016/Crypto"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "backtest", "cache")
COLUMNS = ["t", "o", "h", "l", "c", "v",
           "quote_v", "trades", "taker_buy_v", "taker_buy_qv"]


def _api(url: str, token: str | None) -> dict:
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "alpha-lab-fetch"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read().decode())


def _csv_path(symbol: str, day: str) -> str:
    f = f"{symbol}-1h-{day}/{symbol}-1h-{day}.csv"
    return f"marketData/unzipped/{symbol}/{f}"


def fetch_day(symbol: str, day: str, token: str | None) -> list[list]:
    url = (f"https://api.github.com/repos/{REPO}/contents/"
           + _csv_path(symbol, day))
    try:
        d = _api(url, token)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise
    if d.get("encoding") != "base64" or not d.get("content"):
        return []
    text = base64.b64decode(d["content"]).decode("utf-8", "replace")
    rows = []
    for r in csv.reader(text.splitlines()[1:]):
        if len(r) < 11:
            continue
        try:
            rows.append([int(r[0]) // 1000, float(r[1]), float(r[2]),
                         float(r[3]), float(r[4]), float(r[5]),
                         float(r[7]), float(r[8]),
                         float(r[9]), float(r[10])])
        except ValueError:
            continue
    return rows


def daterange(start: str, end: str):
    a = dt.date.fromisoformat(start)
    b = dt.date.fromisoformat(end)
    while a <= b:
        yield a.isoformat()
        a += dt.timedelta(days=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--symbols", nargs="*",
                    default=["BTC", "ETH", "BNB", "PAXG"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    os.makedirs(CACHE, exist_ok=True)
    lock = threading.Lock()
    days = list(daterange(args.start, args.end))
    for sym in args.symbols:
        pair = f"{sym}USDT"
        out = os.path.join(CACHE, f"{sym}_1h.csv")
        got: list[list] = []
        missing = []

        def job(day: str):
            try:
                rows = fetch_day(pair, day, token)
            except Exception as e:  # network hiccup: record, keep going
                with lock:
                    missing.append(day)
                _ = e
                return
            with lock:
                got.extend(rows)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(job, days))
        got.sort(key=lambda r: r[0])
        # de-duplicate on timestamp (mirror occasionally overlaps)
        seen, clean = set(), []
        for r in got:
            if r[0] in seen:
                continue
            seen.add(r[0])
            clean.append(r)
        if not clean:
            print(f"{sym}: NO DATA (mirror reachable?)", file=sys.stderr)
            continue
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            w.writerows(clean)
        t0 = dt.datetime.utcfromtimestamp(clean[0][0]).isoformat()
        t1 = dt.datetime.utcfromtimestamp(clean[-1][0]).isoformat()
        print(f"{sym}: {len(clean)} bars {t0} → {t1} "
              f"({len(missing)} empty days) → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
