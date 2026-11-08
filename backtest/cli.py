"""CLI:  python -m backtest.cli --mode A --source sim --bars 900
        python -m backtest.cli --mode A --source ccxt --tf 15m
        python -m backtest.cli --mode B --source sim
        python -m backtest.cli --report out_trades.csv
"""
from __future__ import annotations

import argparse
import os

from . import data as data_mod
from .analytics import by_asset, metrics
from .predictor import AppPredictor
from .report import comparison_report
from .runner import run_mode_a, trades_to_csv

OUT = os.path.join(os.path.dirname(__file__), "out")


def _load(source: str, tf: str, bars: int):
    series = {}
    for sym in data_mod.SYMBOLS:
        if source == "ccxt":
            cached = data_mod.load_cache(sym, tf)
            series[sym] = cached if len(cached) >= bars else \
                data_mod.fetch_ccxt(sym, tf, limit=max(bars, 500))
        else:
            series[sym] = data_mod.sim_bars(sym, n=bars)
    return series


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "B"], default="A")
    ap.add_argument("--source", choices=["ccxt", "sim"], default="sim")
    ap.add_argument("--tf", default="15m")
    ap.add_argument("--bars", type=int, default=900)
    ap.add_argument("--policy", choices=["baseline", "validated"],
                    default="baseline")
    ap.add_argument("--report", default="")
    args = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    series = _load(args.source, args.tf, args.bars)
    label = "SYNTHETIC (sim) — harness smoke data, NOT real market results" \
        if args.source == "sim" else f"CCXT {args.tf} cached/fetched"
    print(f"[data] {label}")
    pred = AppPredictor()
    if getattr(args, "policy", "baseline") == "validated":
        from .policy import PolicyPredictor
        pred = PolicyPredictor()
        print("[policy] validated: inverse-TF4 + high-vol gate + k=0.75")

    if args.report:
        from .replay import load_filtered
        import csv
        raw = []
        with open(args.report) as f:
            for r in csv.DictReader(f):
                from .runner import Trade
                raw.append(Trade(r["asset"], int(r["ts"]), r["direction"],
                                 float(r["entry"]), float(r["sl"]), float(r["tp"]),
                                 int(r["exit_ts"]), r["exit_reason"],
                                 float(r["r_gross"]), float(r["r_net"]),
                                 float(r["pnl"]), r.get("tier", ""), r.get("source", "raw")))
        print(comparison_report(raw, load_filtered()))
        return 0

    if args.mode == "A":
        trades = run_mode_a(series, pred)
        path = trades_to_csv(trades, os.path.join(OUT, "mode_a_trades.csv"))
        print(f"[mode A] {len(trades)} trades → {path}")
        for a, m in by_asset(trades).items():
            print(" ", a, m)
        print("  COMBINED", metrics(trades))
        if args.source == "sim":
            print("  (synthetic data — rerun with --source ccxt for real numbers)")
        return 0

    from .replay import replay
    replay(series, pred)
    print("session logged to backtest/replay.db — compare with "
          "`python -m backtest.cli --report backtest/out/mode_a_trades.csv`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
