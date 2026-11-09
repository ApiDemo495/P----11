"""Collect full-engine decision features on REAL BTC 1-minute tape
(the app's native timeframe — TF1=22m ... TF4=247m as designed).

Data: parshant231234-eng/BTCUSDT-Historical-Data on GitHub —
Binance Vision spot 1m monthly files. Cross-validated against the
KevinSherry2016 Binance Vision mirror: hour-open prices agree
within 6 bps (basis level; direction outcomes unaffected).
Continuity asserted (exact 60s steps) before any bar is used.

Usage: CF_1M_DIR=/tmp/btc1m .venv/bin/python -m backtest.collect_btc1m
"""
from __future__ import annotations

import csv
import glob
import json
import os

from .collect_features import flat, regime
from .data import Bar, FRICTION_BPS
from .per_tf_real import TF_W
from .predictor import AppPredictor
from .slice import DataSlice

ROOT = os.environ.get("CF_1M_DIR", "/tmp/btc1m")
MONTHS = os.environ.get("CF_1M_MONTHS", "2026/08,2026/09").split(",")
STEP = int(os.environ.get("CF_1M_STEP", "22"))
CTX = int(os.environ.get("CF_1M_CTX", "2000"))
OUT = os.path.join(os.path.dirname(__file__), "results",
                   "decision_features_btc1m.jsonl")


def load_1m() -> list[Bar]:
    bars: list[Bar] = []
    for mo in MONTHS:
        for path in sorted(glob.glob(os.path.join(
                ROOT, "1m", mo, "*.csv"))):
            for r in csv.DictReader(open(path)):
                bars.append(Bar(t=int(r["timestamp"]) // 1000,
                                o=float(r["open"]), h=float(r["high"]),
                                l=float(r["low"]), c=float(r["close"]),
                                v=float(r["volume"])))
    bars.sort(key=lambda b: b.t)
    for a, b in zip(bars, bars[1:]):
        assert b.t - a.t == 60, f"gap {a.t}->{b.t}"
    return bars


def main() -> None:
    bars = load_1m()
    print(f"BTC 1m: {len(bars)} bars", flush=True)
    pred = AppPredictor(); pred.boot()
    ds = DataSlice(bars)
    n = 0
    with open(OUT, "w") as fh:
        t = CTX
        while t + max(TF_W.values()) < len(bars):
            pack = pred.pack(ds.at(t), "BTC")
            entry = bars[t].c
            for tf, W in TF_W.items():
                res = pack.get(tf) or {}
                d = int(res.get("direction") or 0)
                if not d:
                    continue
                f = {}
                flat("pk", res, f)
                f.update(regime(bars, t, W))
                settle = bars[t + W].c
                bps = d * (settle / entry - 1) * 1e4
                fh.write(json.dumps({
                    "asset": "BTC", "tf": tf, "t": bars[t].t, "dir": d,
                    "win": 1 if bps > 0 else 0,
                    "bps_net": round(bps - FRICTION_BPS["BTC"], 2),
                    "f": f}) + "\n")
                n += 1
            t += STEP
    print("rows", n, "->", OUT, flush=True)


if __name__ == "__main__":
    main()
