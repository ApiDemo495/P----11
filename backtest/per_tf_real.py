"""REAL per-asset x per-TF win rate of the FULL app engine.

Answers 'what is the real win rate using all app features, for each
asset, for each timeframe?' with a measurement, not a claim:

  - Data: Binance Vision 1h klines mirrored on GitHub
    (KevinSherry2016/Crypto, marketData/unzipped — byte-format
    identical to data.binance.vision daily files; the sandbox cannot
    reach Binance directly). Validated: zero timestamp gaps, spot
    anchors match known history.
  - Predictor: the app's own PredictionEngine.compute_asset — ALL
    features that can run on a price tape vote (37 formulas, CCI
    fusion + direction ladder, four FlyWire brains, physics/emotion
    stacks). Learning is frozen between decisions (record=False,
    the harness default). Order-book / news features need live
    state and cannot be replayed from klines — they are absent
    here exactly as they are absent from any kline backtest, and
    this file will not pretend otherwise.
  - Bars are 1h, so TF windows are W bars = 22/44/130/247 hours.
    The app treats its tape as 1-minute bars; the TF *logic* (window
    lengths, formulas, fusion) is identical, the wall-clock meaning
    of a 'bar' is what changes. Read TF1 as 'the 22-bar horizon'.
  - Settlement per TF over its own window: strict sign at the
    window close (tie = loss — the settlement ledger's rule), plus
    TP/SL first-touch from the app's measured risk_block (SL wins
    ties). Friction (12 bps BTC/ETH/BNB, 18 bps PAXG round trip)
    charged on the bps column, never hidden.
  - Decision points every 22 bars over the last ~83 days of tape
    (one regime — not a full market cycle). Windows overlap for
    TF3/TF4, so n overstates independence there; Wilson 95% CIs
    are printed so the uncertainty is visible, not buried.

Usage: .venv/bin/python -m backtest.per_tf_real --data-root /tmp/crypto
Writes backtest/results/per_tf_full_engine_real.csv + prints table.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import time

from .data import Bar, FRICTION_BPS
from .predictor import AppPredictor
from .slice import DataSlice

SYMS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT",
        "BNB": "BNBUSDT", "PAXG": "PAXGUSDT"}
TF_W = {"tf1": 22, "tf2": 44, "tf3": 130, "tf4": 247}
KEEP_BARS = 3500          # context + evaluation window
EVAL_BARS = 2000          # decisions live in the last ~2000 bars
STEP = 22


def load_bars(root: str, pair: str) -> list[Bar]:
    bars: list[Bar] = []
    for path in sorted(glob.glob(os.path.join(
            root, "marketData", "unzipped", pair, f"{pair}-1h-*",
            "*.csv"))):
        with open(path) as fh:
            rd = csv.reader(fh)
            header = next(rd, None)
            for row in rd:
                if len(row) < 6:
                    continue
                bars.append(Bar(t=int(row[0]) // 1000, o=float(row[1]),
                                h=float(row[2]), l=float(row[3]),
                                c=float(row[4]), v=float(row[5])))
    bars.sort(key=lambda b: b.t)
    # validation: strictly hourly, no gaps, no duplicates
    for a, b in zip(bars, bars[1:]):
        assert b.t - a.t == 3600, f"gap/dup at {a.t}->{b.t}"
    return bars


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/tmp/crypto")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "results",
        "per_tf_full_engine_real.csv"))
    args = ap.parse_args()

    series = {}
    for sym, pair in SYMS.items():
        bars = load_bars(args.data_root, pair)[-KEEP_BARS:]
        series[sym] = bars
        print(f"{sym}: {len(bars)} bars "
              f"{time.strftime('%Y-%m-%d', time.gmtime(bars[0].t))} -> "
              f"{time.strftime('%Y-%m-%d', time.gmtime(bars[-1].t))}",
              flush=True)

    pred = AppPredictor()
    pred.boot()
    print(f"engine booted phase={pred.engine.phase}", flush=True)

    def set_state(sym2: str, bars2: list[Bar]) -> None:
        st = pred.feed.state(sym2)
        st.times = [b.t for b in bars2]
        st.opens = [b.o for b in bars2]
        st.highs = [b.h for b in bars2]
        st.lows = [b.l for b in bars2]
        st.closes = [b.c for b in bars2]
        st.volumes = [b.v for b in bars2]
        st.sources = ["backtest"] * len(bars2)
        st.live_price = bars2[-1].c
        st.live_open, st.live_high = bars2[-1].o, bars2[-1].h
        st.live_low = bars2[-1].l
        st.day_open = bars2[0].o

    rows = []
    for sym, bars in series.items():
        ds = DataSlice(bars)
        eval_start = len(bars) - EVAL_BARS
        t = eval_start
        npts = 0
        t0 = time.time()
        while t + max(TF_W.values()) < len(bars):
            # cross-asset features must see every asset AS OF bar t —
            # never a state left over from another asset's later bar
            # (that would be look-ahead through the back door)
            for s2, b2 in series.items():
                set_state(s2, b2[:t + 1])
            pack = pred.pack(ds.at(t), sym)
            entry = bars[t].c
            for tf, W in TF_W.items():
                res = pack.get(tf) or {}
                direction = int(res.get("direction") or 0)
                if direction == 0:
                    continue
                settle = bars[t + W].c
                move_bps = direction * (settle / entry - 1.0) * 1e4
                win = 1 if move_bps > 0 else 0      # tie = loss
                # TP/SL first-touch, app risk block, SL wins ties
                lvl = ""
                try:
                    from app.moves_risk import risk_block
                    blk = risk_block(
                        entry, direction, W,
                        float(res.get("confidence") or 0.0),
                        [b.c for b in bars[:t + 1]],
                        [b.h for b in bars[:t + 1]],
                        [b.l for b in bars[:t + 1]],
                        atr=float(res.get("atr") or 0.0))
                    tp, sl = blk.get("tp"), blk.get("sl")
                    if tp and sl:
                        for fb in bars[t + 1:t + W + 1]:
                            if direction > 0:
                                if fb.l <= sl:
                                    lvl = "SL"; break
                                if fb.h >= tp:
                                    lvl = "TP"; break
                            else:
                                if fb.h >= sl:
                                    lvl = "SL"; break
                                if fb.l <= tp:
                                    lvl = "TP"; break
                except Exception:
                    lvl = ""
                rows.append({
                    "asset": sym, "tf": tf, "t": bars[t].t,
                    "direction": direction, "entry": entry,
                    "settle": settle, "win": win,
                    "bps_gross": round(move_bps, 2),
                    "bps_net": round(move_bps - FRICTION_BPS[sym], 2),
                    "level": lvl,
                    "conf": round(float(res.get("confidence") or 0.0), 4),
                    "tier": str(res.get("tier") or ""),
                })
            npts += 1
            if npts == 3:
                print(f"  {sym}: ~{time.time() - t0:.1f}s / 3 packs",
                      flush=True)
            t += STEP
        print(f"{sym}: {npts} decision packs done", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} graded calls -> {args.out}\n")

    print(f"{'asset':6} {'tf':4} {'win%':>7} {'95% CI':>15} {'n':>4} "
          f"{'TP/SL/none':>12} {'bps net':>8}")
    summary = {}
    for sym in SYMS:
        for tf in TF_W:
            sel = [r for r in rows if r["asset"] == sym and r["tf"] == tf]
            n = len(sel)
            wins = sum(r["win"] for r in sel)
            lo, hi = wilson(wins, n)
            tp = sum(1 for r in sel if r["level"] == "TP")
            sl = sum(1 for r in sel if r["level"] == "SL")
            none = n - tp - sl
            bps = (sum(r["bps_net"] for r in sel) / n) if n else 0.0
            summary[f"{sym}|{tf}"] = {
                "win_pct": round(100 * wins / n, 2) if n else None,
                "ci": [round(100 * lo, 1), round(100 * hi, 1)], "n": n,
                "tp": tp, "sl": sl, "neither": none,
                "bps_net": round(bps, 2)}
            print(f"{sym:6} {tf:4} {100 * wins / n:6.2f}% "
                  f"{100 * lo:6.1f}–{100 * hi:<7.1f} {n:4d} "
                  f"{tp:3d}/{sl:3d}/{none:<5d} {bps:+8.2f}")
    for tf in TF_W:
        sel = [r for r in rows if r["tf"] == tf]
        n = len(sel)
        wins = sum(r["win"] for r in sel)
        lo, hi = wilson(wins, n)
        print(f"{'ALL':6} {tf:4} {100 * wins / n:6.2f}% "
              f"{100 * lo:6.1f}–{100 * hi:<7.1f} {n:4d}")
    print("\n" + json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
