"""Fetch deterministic historical OHLCV inputs via GitHub's allowed API.

This is a reproducible opt-in data bootstrap for price_action_oos. BTC,
ETH, and BNB spot 1m monthly klines use the documented public parshant
mirrors; PAXG uses the documented KevinSherry USD-M hourly CSV. The data
is ignored under backtest/cache and is never committed. Missing PAXG
sub-hour OHLC remains ungradeable; this script does not synthesize it.
"""
from __future__ import annotations
import base64, csv, datetime as dt, io, json, os, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "backtest" / "cache"
SYMBOLS = ("BTC", "ETH", "BNB")

def api(url: str):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "P----11-price-action-oos"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for retry in range(5):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as res:
                return json.loads(res.read())
        except urllib.error.HTTPError as e:
            if e.code in (403, 429) and retry < 4:
                time.sleep(min(60, 2 ** (retry + 1)))
                continue
            if e.code == 404:
                return None
            raise

def norm_ts(value: str) -> int:
    n = int(float(value))
    if n > 10**14: return n // 10**6
    if n > 10**11: return n // 10**3
    return n


def fetch_spot():
    tasks=[]
    for sym in SYMBOLS:
        repo=f"parshant231234-eng/{sym}USDT-Historical-Data"
        years=api(f"https://api.github.com/repos/{repo}/contents/1m") or []
        for entry in years:
            if entry.get("type")!="dir" or not str(entry.get("name","")).isdigit(): continue
            year=int(entry["name"])
            months=api(f"https://api.github.com/repos/{repo}/contents/1m/{year}") or []
            for m in months:
                if m.get("type")=="dir" and str(m.get("name","")).isdigit():
                    tasks.append((sym,year,int(m["name"])))
    CACHE.mkdir(parents=True,exist_ok=True)
    for sym in SYMBOLS:
        own=sorted((t for t in tasks if t[0]==sym),key=lambda x:(x[1],x[2]))
        path=CACHE/f"{sym}_1m.csv"
        last_ts=-1
        with path.open("w",newline="") as fh:
            w=csv.writer(fh); w.writerow(["t","o","h","l","c","v"])
            with ThreadPoolExecutor(max_workers=6) as pool:
                # executor.map preserves chronological task order, while
                # downloading month blobs concurrently within each asset.
                for name,y,m,rows in pool.map(fetch_one_month,own):
                    for row in rows:
                        if row[0] > last_ts:
                            w.writerow(row); last_ts=row[0]
                    print(f"{name} {y}-{m:02d}: {len(rows)} bars",flush=True)
        print(f"wrote {path}: through epoch {last_ts}",flush=True)

def fetch_one_month(task):
    sym,year,month=task
    repo=f"parshant231234-eng/{sym}USDT-Historical-Data"
    folder=f"1m/{year:04d}/{month:02d}"
    listing=api(f"https://api.github.com/repos/{repo}/contents/{folder}")
    if not isinstance(listing,list): return sym,year,month,[]
    match=next((x for x in listing if x.get("name","").endswith(".csv") and "1m" in x.get("name","")),None)
    if not match: return sym,year,month,[]
    data=api(f"https://api.github.com/repos/{repo}/git/blobs/{match['sha']}")
    if not data or data.get("encoding")!="base64": raise RuntimeError(f"no blob {sym} {year}-{month}")
    text=base64.b64decode(data["content"]).decode("utf-8","replace")
    out=[]
    for r in csv.DictReader(io.StringIO(text)):
        try:
            ts=norm_ts(r.get("timestamp") or r.get("open_time") or r.get("openTime"))
            out.append((ts,float(r["open"]),float(r["high"]),float(r["low"]),float(r["close"]),float(r.get("volume") or 0)))
        except (KeyError,TypeError,ValueError): continue
    return sym,year,month,out


def fetch_paxg_hourly():
    repo="KevinSherry2016/Crypto"
    p=api(f"https://api.github.com/repos/{repo}/contents/CleanData/klinePAXGUSDT.csv")
    if not p: return
    d=api(f"https://api.github.com/repos/{repo}/git/blobs/{p['sha']}")
    text=base64.b64decode(d["content"]).decode("utf-8","replace")
    rows=[]
    for r in csv.DictReader(io.StringIO(text)):
        try:
            stamp=dt.datetime.strptime(r["open_time"],"%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc).timestamp()
            rows.append((int(stamp),float(r["open"]),float(r["high"]),float(r["low"]),float(r["close"]),float(r.get("volume") or 0)))
        except (KeyError,ValueError): continue
    rows.sort()
    if rows:
        with (CACHE/"PAXG_1h.csv").open("w",newline="") as fh:
            w=csv.writer(fh); w.writerow(["t","o","h","l","c","v"]); w.writerows(rows)
        print(f"PAXG 1h: {len(rows)} rows",flush=True)

if __name__=="__main__":
    fetch_spot()
    fetch_paxg_hourly()
