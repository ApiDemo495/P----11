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

from app.alpha_lab import (evaluate_deep, evaluate_symbol, load_csv,
                           resample)

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
    ap.add_argument("--mode", choices=["classic", "deep"], default="classic")
    ap.add_argument("--write", action="store_true", default=True)
    args = ap.parse_args(argv)

    os.makedirs(RESULTS, exist_ok=True)
    if args.mode == "deep":
        return _deep_report(args)
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


# per-timeframe geometry: (base interval, resample step from 1m, horizon bars,
# decision stride). A 22-minute window cannot be traded at 12 bp friction —
# the σ column in the report shows exactly that.
DEEP_GEOM = {
    "TF1": (5, 4, 4),     # 5-minute bars, 20-minute horizon, stride 4
    "TF2": (5, 9, 4),     # 5-minute bars, 45-minute horizon
    "TF3": (15, 9, 3),    # 15-minute bars, 135-minute horizon
    "TF4": (60, 4, 2),    # 1-hour bars, 240-minute horizon
}


def _deep_report(args) -> int:
    lines = ["# Alpha lab — DEEP ensemble, real sub-hour Binance data", "",
             "Data: `backtest/cache/{SYM}_1m.csv` (Binance Vision monthly "
             "archives via GitHub mirrors), 1-minute bars with taker "
             "volume / quote volume / trade counts, 2026-01-01 → "
             "2026-08-31 (BTC, ETH, BNB). PAXG has no accessible 1m "
             "archive and is evaluated on the 1h mirror (TF3/TF4 only).",
             "",
             "BEFORE = equal-weight technical blend (momentum signs), "
             "fixed 2:1 vol barriers, 12 bp friction, no filter.",
             "AFTER = noise-screened, prior-signed ridge ensemble over "
             "technical + order-flow + microstructure features "
             "(OFI, VPIN, Kyle λ, Amihud, size skew, Hawkes), adaptive "
             "friction from Kyle λ, cost-aware meta-label P(net > hurdle), "
             "and a tradability gate requiring the lane's σ to clear "
             "3× friction.",
             "",
             "| Lane | n | σ bp | tradable | acc B→A | gated acc | IC B→A | "
             "EV B→A | EV sym A | gated EV | net-win | hurdle | kept |",
             "|" + "---|" * 14]
    print(lines[-4])
    for sym in args.assets:
        for tf, (step, hz, stride) in DEEP_GEOM.items():
            base_path = os.path.join(CACHE, f"{sym}_1m.csv")
            if tf in ("TF3", "TF4") and not os.path.exists(base_path):
                base_path = os.path.join(CACHE, f"{sym}_1h.csv")
                step, hz = (1, 4 if tf == "TF4" else 9)
            if not os.path.exists(base_path):
                print(f"{sym} {tf}: no cache — SKIPPED", file=sys.stderr)
                continue
            bars = load_csv(base_path)
            bars = resample(bars, step) if step > 1 else bars
            try:
                r = evaluate_deep(bars, horizon=hz, tf=tf,
                                  friction_bps=FRICTION.get(sym, 12.0),
                                  market=resample(load_csv(
                                      os.path.join(CACHE, f"{sym}_1m.csv")), 5)
                                  if "1m" in base_path else None,
                                  stride=stride)
            except Exception as e:
                print(f"{sym} {tf}: ERROR {e}", file=sys.stderr)
                continue
            if "error" in r:
                print(f"{sym} {tf}: {r['error']}")
                continue
            g, a, b, c = r["after_gated"], r["after"], r["before"], r["calib"]
            line = (f"| {sym} {tf} | {r['n']} | {r['sigma_bps_mean']:.1f} | "
                    f"{r['tradable_share']*100:.0f}% | "
                    f"{b['sign_acc']*100:.1f}→{a['sign_acc']*100:.1f}% | "
                    f"{(g['sign_acc']*100 if g['sign_acc']==g['sign_acc'] else float('nan')):.1f}%"
                    f"({g['coverage']*100:.0f}%) | "
                    f"{b['ic']:+.3f}→{a['ic']:+.3f} | "
                    f"{b['ev_bp']:+.0f}→{a['ev_bp']:+.0f} | "
                    f"{a['ev_sym']:+.0f} | {g['ev_sym']:+.0f} | "
                    f"{c['net_win_rate']*100:.0f}% | {c['hurdle_mean']:.1f}bp | "
                    f"{r['ensemble']['kept_total']}/{r['ensemble']['features']} |")
            print(line, flush=True)
            lines.append(line)
    with open(os.path.join(RESULTS, "ALPHA_LAB_DEEP.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote backtest/results/ALPHA_LAB_DEEP.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
