"""Download real Binance klines from GitHub mirrors of Binance Vision.

Two sources, both byte-identical to `data.binance.vision` exports:

  1. `vision` — daily 1h CSV files (unzipped) in `KevinSherry2016/Crypto`,
     `marketData/unzipped/{SYM}/{SYM}-1h-{DATE}/…csv`. Cheap and complete
     for TF3/TF4.
  2. `zips` — MONTHLY zipped klines, which is the only accessible route to
     sub-hour bars:
        · 1m → `JienWeng/crypto-execution-timing`, `data/raw/{SYM}-1m-YYYY-MM.zip`
        · 5m → `thanhnha39/Crypto-Tail-Risk-…`, `data/raw/binance/{SYM}/…zip`
     Sub-hour data is what makes TF1 (22 m) and TF2 (44 m) testable
     instead of guessed.

Files larger than 1 MB cannot come back through the contents API, so the
zip path resolves each month to a git blob and decodes it (`git/blobs`
supports up to 100 MB).

Every row keeps `taker_buy_volume`, `quote_volume` and `count`, which is
what makes order-flow features computable offline.

Usage
    python3 scripts/fetch_binance_mirror.py --source vision
    python3 scripts/fetch_binance_mirror.py --source zips --interval 1m \
        --symbols BTC ETH BNB --months 2026-01 2026-02 … 2026-08

Output: `backtest/cache/{SYM}_{interval}.csv` with the app's column order
(t,o,h,l,c,v) plus quote_v, trades, taker_buy_v, taker_buy_qv.
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import io
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "backtest", "cache")
COLUMNS = ["t", "o", "h", "l", "c", "v",
           "quote_v", "trades", "taker_buy_v", "taker_buy_qv"]

VISION_REPO = "KevinSherry2016/Crypto"
ZIP_REPOS = {
    "1m": ("JienWeng/crypto-execution-timing", "data/raw"),
    "5m": ("thanhnha39/Crypto-Tail-Risk-Volatility-Forecasting-Engine",
           "data/raw/binance"),
}


def _api(url: str) -> dict:
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "alpha-lab-fetch"})
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode())


def _norm_ts(raw: str) -> int:
    """Binance switched to MICROSECOND open_time in 2025+ exports; older
    files are milliseconds and the oldest are seconds. Normalise to
    seconds so resampling and joins are unambiguous."""
    v = int(float(raw))
    if v > 10**14:      # microseconds
        return v // 10**6
    if v > 10**11:      # milliseconds
        return v // 10**3
    return v


def _rows_from_csv(text: str) -> list[list]:
    out = []
    for r in csv.reader(text.splitlines()):
        if not r or not r[0].strip().lstrip("-").isdigit():
            continue                      # header or junk
        if len(r) < 11:
            continue
        try:
            out.append([_norm_ts(r[0]), float(r[1]), float(r[2]),
                        float(r[3]), float(r[4]), float(r[5]),
                        float(r[7]), float(r[8]),
                        float(r[9]), float(r[10])])
        except (ValueError, IndexError):
            continue
    return out


# ─────────────────────────── source 1: daily 1h CSVs ───────────────────────────


def _vision_day(symbol: str, day: str) -> list[list]:
    f = f"{symbol}-1h-{day}/{symbol}-1h-{day}.csv"
    url = (f"https://api.github.com/repos/{VISION_REPO}/contents/"
           f"marketData/unzipped/{symbol}/{f}")
    try:
        d = _api(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise
    if d.get("encoding") != "base64" or not d.get("content"):
        return []
    return _rows_from_csv(base64.b64decode(d["content"]).decode("utf-8", "replace"))


def fetch_vision(symbols: list[str], start: str, end: str,
                 workers: int = 8) -> dict[str, str]:
    days, d = [], dt.date.fromisoformat(start)
    stop = dt.date.fromisoformat(end)
    while d <= stop:
        days.append(d.isoformat())
        d += dt.timedelta(days=1)
    out = {}
    for sym in symbols:
        got: list[list] = []

        pair = f"{sym}USDT"          # the mirror keys files by the PAIR
        def job(day: str, _p=pair):
            got.extend(_vision_day(_p, day))

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(job, days))
        out[sym] = _write(sym, "1h", got)
    return out


# ─────────────────────────── source 2: monthly zips ───────────────────────────


def _listing(repo: str, folder: str) -> dict[str, str]:
    """name → blob sha for one directory."""
    try:
        d = _api(f"https://api.github.com/repos/{repo}/contents/{folder}")
    except urllib.error.HTTPError:
        return {}
    if not isinstance(d, list):
        return {}
    return {x["name"]: x.get("sha") for x in d
            if x.get("type") == "file" and isinstance(x.get("size"), int)}


def _blob(repo: str, sha: str) -> bytes:
    d = _api(f"https://api.github.com/repos/{repo}/git/blobs/{sha}")
    if d.get("encoding") != "base64":
        raise RuntimeError("blob not base64")
    return base64.b64decode(d["content"])


def _zip_month(repo: str, folder: str, pair: str, interval: str,
               month: str) -> list[list]:
    name = f"{pair}-{interval}-{month}.zip"
    files = _listing(repo, folder)
    sha = files.get(name)
    if not sha:
        return []
    raw = _blob(repo, sha)
    rows: list[list] = []
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for n in z.namelist():
            if n.lower().endswith(".csv"):
                rows.extend(_rows_from_csv(
                    z.read(n).decode("utf-8", "replace")))
                break
    return rows


def fetch_zips(symbols: list[str], interval: str, months: list[str],
               workers: int = 4) -> dict[str, str]:
    repo, folder = ZIP_REPOS[interval]
    out = {}
    for sym in symbols:
        pair = f"{sym}USDT"
        sub = f"{folder}/{pair}" if interval == "5m" else folder

        def job(m: str, _p=pair, _sub=sub):
            return _zip_month(repo, _sub, _p, interval, m)

        got: list[list] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for rows in ex.map(job, months):
                got.extend(rows)
        out[sym] = _write(sym, interval, got)
    return out


# ─────────────────────────── shared writer ───────────────────────────


def _write(sym: str, interval: str, rows: list[list]) -> str:
    if not rows:
        return ""
    rows.sort(key=lambda r: r[0])
    seen, clean = set(), []
    for r in rows:
        if r[0] in seen:
            continue
        seen.add(r[0])
        clean.append(r)
    path = os.path.join(CACHE, f"{sym}_{interval}.csv")
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(clean)
    t0 = dt.datetime.utcfromtimestamp(clean[0][0]).isoformat()
    t1 = dt.datetime.utcfromtimestamp(clean[-1][0]).isoformat()
    print(f"{sym} {interval}: {len(clean)} bars {t0} → {t1} → {path}")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["vision", "zips"], default="vision")
    ap.add_argument("--interval", default="1h", choices=["1m", "5m", "1h"])
    ap.add_argument("--symbols", nargs="*", default=["BTC", "ETH", "BNB", "PAXG"])
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--months", nargs="*", default=None,
                    help="YYYY-MM months for --source zips")
    a = ap.parse_args(argv)
    if a.source == "vision":
        res = fetch_vision(a.symbols, a.start, a.end)
    else:
        months = a.months or [f"2026-{m:02d}" for m in range(1, 13)]
        res = fetch_zips(a.symbols, a.interval, months)
    if not any(res.values()):
        print("no data written — mirror layout changed?", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
