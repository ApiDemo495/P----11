"""CLI: purged walk-forward evaluation of the fusion upgrades.

    python3 -m backtest.alpha_lab                 # cached bars, all assets
    python3 -m backtest.alpha_lab --assets BTC ETH --steps 1 2 4

Prints a BEFORE (classic technical composite, fixed 2:1 barriers) vs
AFTER (order-flow block + decorrelated IC fusion + meta-label filter +
isotonic confidence + vol-fitted barriers) table per asset and horizon,
and writes `backtest/results/ALPHA_LAB.md`.

Nothing here is synthetic. If a cache file is missing the asset is
skipped and named as skipped — no fabricated numbers.
"""
from __future__ import annotations

import argparse
import os
import sys

from app.alpha_lab import evaluate_symbol, load_csv, resample

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
RESULTS = os.path.join(HERE, "results")

FRICTION = {"BTC": 12.0, "ETH": 12.0, "BNB": 12.0, "PAXG": 18.0}

# horizon in base bars (1h base) per timeframe
TF_STEPS = {"TF1": 1, "TF2": 2, "TF3": 2, "TF4": 4}
TF_HORIZON = {"TF1": 22, "TF2": 44, "TF3": 130, "TF4": 247}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="*", default=["BTC", "ETH", "BNB", "PAXG"])
    ap.add_argument("--horizons", nargs="*", type=int, default=[2, 4],
                    help="forward horizon in HOURS (2 ≈ TF3 130m, "
                         "4 ≈ TF4 247m; TF1/TF2 need sub-hour bars the "
                         "mirror does not carry)")
    ap.add_argument("--write", action="store_true", default=True)
    args = ap.parse_args(argv)

    os.makedirs(RESULTS, exist_ok=True)
    rows, lines = [], []
    lines.append("# Alpha lab — purged walk-forward, real Binance 1h history")
    lines.append("")
    lines.append("Data: `backtest/cache/{SYM}_1h.csv` (Binance Vision mirror, "
                 "2026-01-01 → 2026-09-30, 6,552 bars/asset).")
    lines.append("")
    lines.append("BEFORE = classic technical composite (momentum / efficiency "
                 "/ range position), |conf| from the composite, fixed 2:1 "
                 "vol barriers.")
    lines.append("AFTER = + taker-imbalance & trade-size & VWAP-deviation "
                 "block, orthogonalised against the TA block, IC-shrunk "
                 "weights fitted on TRAIN folds only, meta-label success "
                 "filter, isotonic confidence, barrier multiples fitted on "
                 "TRAIN folds only.")
    lines.append("")
    hdr = ("| Asset | step | n | B acc | A acc | B acc@floor | A acc@gated | "
           "Brier raw→iso | slope | B EV bp | A EV bp | gated EV bp | "
           "|IC| ta/flow | IC before→after | gate cov |")
    sep = "|" + "---|" * 16
    lines.append(hdr)
    lines.append(sep)
    print(hdr)
    market = None
    mpath = os.path.join(CACHE, "BTC_1h.csv")
    if os.path.exists(mpath):
        market = load_csv(mpath)
    for sym in args.assets:
        path = os.path.join(CACHE, f"{sym}_1h.csv")
        if not os.path.exists(path):
            print(f"{sym}: no cache at {path} — SKIPPED", file=sys.stderr)
            lines.append(f"| {sym} | — | — | — | — | — | — | — | — | — | — "
                         f"| — | no cache | — |")
            continue
        base = load_csv(path)
        for horizon in args.horizons:
            bars = base
            try:
                r = evaluate_symbol(bars, horizon=horizon,
                                    friction_bps=FRICTION.get(sym, 12.0),
                                    market=market)
            except Exception as e:  # never fake a number
                print(f"{sym} step{step}: ERROR {e}", file=sys.stderr)
                continue
            if "error" in r:
                print(f"{sym} h{horizon}: {r['error']}")
                continue
            b, bh = r["before"], r["before_hi"]
            a, ag, cal = r["after"], r["after_gated"], r["calib"]
            line = (f"| {sym} | {horizon}h | {r['n']} | "
                    f"{b['sign_acc']*100:.1f}% | {a['sign_acc']*100:.1f}% | "
                    f"{bh['sign_acc']*100:.1f}% | {ag['sign_acc']*100:.1f}% | "
                    f"{cal['raw_brier']:.3f}→{cal['iso_brier']:.3f} | "
                    f"{cal['iso_slope']:.2f} | {b['ev_bp']:+.1f} | "
                    f"{a['ev_bp']:+.1f} | {ag['ev_bp']:+.1f} | "
                    f"{r['ic_ta']:.3f}/{r['ic_flow']:.3f} | "
                    f"{b['ic']:+.3f}→{a['ic']:+.3f} | "
                    f"{r['after_gated']['coverage']*100:.0f}% |")
            print(line)
            lines.append(line)
            rows.append({"asset": sym, "horizon_h": horizon, **r})
    if args.write and rows:
        with open(os.path.join(RESULTS, "ALPHA_LAB.md"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nwrote {os.path.join(RESULTS, 'ALPHA_LAB.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
