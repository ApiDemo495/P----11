#!/usr/bin/env python3
"""Edge scan — settle every candidate structure on a sequential book.

This is the falsification harness for the 2026-11 programme.  Each row is
one *fixed* configuration, evaluated out-of-fold on a single-position book
(entry at the next bar open, taker fills, stop wins ties, no pyramiding)
with purged folds and an overlap-aware t-statistic.

    python -m backtest.edge_scan                     # canonical table
    python -m backtest.edge_scan --only cascade -v
    python -m backtest.edge_scan --write             # refresh the report

Nothing here searches parameters inside the folds: a searched number is a
property of the search, not of the market, and the previous round proved
exactly that (a per-fold grid selector degenerated to the rarest cell and
reported -6 to -23 bp on every asset).
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.alpha_lab import build_matrix, load_csv, purged_folds  # noqa: E402
from app.ensemble import ic_weights, zscore_train  # noqa: E402
from app.events import sigma_series  # noqa: E402
from app.portfolio import run_sleeve, trade_stats  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
FEE = {"BTC": 12.0, "ETH": 12.0, "BNB": 12.0, "PAXG": 18.0}


def _load(sym: str, interval: str):
    return load_csv(os.path.join(CACHE, f"{sym}_{interval}.csv"))


def _finish(trades: list[dict], fold_ev: list[float], fee: float) -> dict:
    st = trade_stats(trades, fees_bps=fee)
    st["folds"] = fold_ev
    st["folds_positive"] = f"{sum(1 for f in fold_ev if f > 0)}/{len(fold_ev)}"
    return st


def oof_ensemble(sym: str, hz: int, pt: float = 2.0, sl: float = 1.0,
                 gate_q: float = 0.7, interval: str = "1h",
                 folds: int = 5) -> dict:
    """TA + flow ensemble, ridge fitted on train folds, settled sequentially."""
    b = _load(sym, interval)
    m = build_matrix(b, hz, market=b, micro=False)
    X, idx, fwd, prior = m["X"], m["idx"], m["fwd"], m["prior"]
    n = len(b.c)
    sg = sigma_series(b, 240)
    score = np.zeros(n)
    gate = np.zeros(n, bool)
    fold_range = []
    for tr, te in purged_folds(X.shape[0], folds=folds, embargo=hz):
        Z = zscore_train(X, tr)
        fit = ic_weights(Z[tr], fwd[tr], prior, n_eff=tr.size)
        sc = Z[te] @ fit["w"]
        score[idx[te]] = sc
        thr = (np.quantile(np.abs(Z[tr] @ fit["w"]), gate_q)
               if gate_q > 0 else -1.0)
        gate[idx[te]] = np.abs(sc) >= thr
        fold_range.append((tr, te))
    trades, fev = [], []
    for _tr, te in fold_range:
        mm = np.zeros(n, bool)
        mm[idx[te]] = True
        t = run_sleeve(b, score, gate & mm, pt=pt, sl=sl, hz=hz, sigma=sg,
                       fees_bps=FEE[sym])
        trades += t
        fev.append(round(trade_stats(t, fees_bps=FEE[sym])["ev_bp"], 1) if t else 0.0)
    return _finish(trades, fev, FEE[sym])


def oof_classic(sym: str, name: str, hz: int = 24, pt: float = 3.0,
                sl: float = 1.5, interval: str = "1h") -> dict:
    """A textbook signal with the same sequential settlement."""
    b = _load(sym, interval)
    c = np.asarray(b.c, dtype=float)
    n = c.size
    sg = sigma_series(b, 240)
    ma = np.convolve(c, np.ones(20) / 20, mode="full")[:n]
    sd = np.sqrt(np.maximum(1e-18, np.convolve(c * c, np.ones(20) / 20,
                                               mode="full")[:n] - ma ** 2))
    bb = (c - ma) / np.maximum(1e-12, sd)
    span = np.zeros(n)
    for i in range(n):
        span[i] = (c[max(0, i - 55):i + 1].max() - c[max(0, i - 55):i + 1].min())
    don = np.zeros(n)
    for i in range(n):
        w = c[max(0, i - 55):i + 1]
        don[i] = (c[i] - w.min()) / max(1e-12, w.max() - w.min()) - 0.5
    sig = {
        "bollinger_fade": -np.sign(bb),
        "bollinger_follow": np.sign(bb),
        "donchian_follow": np.sign(don),
        "donchian_fade": -np.sign(don),
    }[name]
    gate = np.abs(bb) >= 2.0 if name.startswith("bollinger") else np.ones(n, bool)
    trades, fev = [], []
    for _tr, te in purged_folds(n, folds=5, embargo=max(hz, 120)):
        m = np.zeros(n, bool)
        m[te[0]:te[-1] + 1] = True
        t = run_sleeve(b, sig, gate & m, pt=pt, sl=sl, hz=hz, sigma=sg,
                       fees_bps=FEE[sym])
        trades += t
        fev.append(round(trade_stats(t, fees_bps=FEE[sym])["ev_bp"], 1) if t else 0.0)
    return _finish(trades, fev, FEE[sym])


def oof_cascade(sym: str, params=(15, 4.0, 1.0, 1.0, 240)) -> dict:
    from app.alpha_lab import evaluate_cascade
    b = _load(sym, "1m")
    r = evaluate_cascade(b, params, fees_bps=FEE[sym])
    if "error" in r:
        return {"n": 0, "note": r["error"]}
    return r


def oof_leadlag(target: str, src: str = "BTC", k: int = 3, hz: int = 5,
                q: float = 0.9, pt: float = 2.0, sl: float = 1.0) -> dict:
    tb, sb = _load(target, "1m"), _load(src, "1m")
    n = min(len(tb.c), len(sb.c))
    sc = np.asarray(sb.c, dtype=float)[:n]
    sg = sigma_series(tb, 240)[:n]
    r = np.zeros(n)
    r[k:] = sc[k:] / sc[:-k] - 1.0
    thr = np.quantile(np.abs(r[n // 10:]), 1.0 - q)
    score = np.zeros(n)
    score[1:] = np.sign(r[:-1])
    gate = np.zeros(n, bool)
    gate[1:] = np.abs(r[:-1]) >= thr
    trades, fev = [], []
    for _tr, te in purged_folds(n, folds=5, embargo=240):
        m = np.zeros(n, bool)
        m[te[0]:te[-1] + 1] = True
        t = run_sleeve(tb, score, gate & m, pt=pt, sl=sl, hz=hz, sigma=sg,
                       fees_bps=FEE[target])
        trades += t
        fev.append(round(trade_stats(t, fees_bps=FEE[target])["ev_bp"], 1) if t else 0.0)
    return _finish(trades, fev, FEE[target])


def _fmt(label: str, st: dict) -> str:
    if not st.get("n"):
        return f"| {label} | 0 | — | — | — | — | {st.get('note', 'no trades')} |"
    return ("| {l} | {n} | {g:+.1f} | {e:+.1f} | {pf} | {t:+.2f} | {pos} |"
            .format(l=label, n=st["n"], g=st["gross_ev_bp"], e=st["ev_bp"],
                    pf=(f"{st['profit_factor']:.2f}"
                        if math.isfinite(st["profit_factor"]) else "∞"),
                    t=st["t_stat"], pos=st.get("folds_positive", "—")))


def canonical(assets=("BTC", "ETH", "BNB", "PAXG"), only=None) -> list[str]:
    rows = ["| sleeve | n | gross bp | net bp | PF | t | folds + |",
            "|---|---:|---:|---:|---:|---:|---|"]
    want = lambda tag: only is None or tag in only
    if want("ensemble"):
        for sym in assets:
            for hz, pt, sl in ((12, 2.0, 1.0), (48, 3.0, 1.5)):
                rows.append(_fmt(f"{sym} ensemble h={hz}h pt{pt}/sl{sl}",
                                 oof_ensemble(sym, hz, pt=pt, sl=sl)))
    if want("classic"):
        for sym in assets:
            rows.append(_fmt(f"{sym} bollinger follow h=24",
                             oof_classic(sym, "bollinger_follow")))
    if want("cascade"):
        for sym in assets[:3]:
            rows.append(_fmt(f"{sym} cascade fade k15 z4",
                             oof_cascade(sym)))
    if want("leadlag"):
        for sym in ("ETH", "BNB"):
            rows.append(_fmt(f"{sym}<-BTC lead-lag k3",
                             oof_leadlag(sym)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="ensemble / classic / cascade / leadlag")
    ap.add_argument("--assets", nargs="*", default=["BTC", "ETH", "BNB", "PAXG"])
    ap.add_argument("--write", action="store_true",
                    help="write backtest/results/EDGE_SCAN.md")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    rows = canonical(tuple(args.assets), args.only)
    text = "\n".join(rows)
    print(text)
    if args.write:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "EDGE_SCAN.md")
        header = ("# Edge scan — sequential out-of-fold settlement\n\n"
                  "Every sleeve is a *fixed* configuration settled on a "
                  "single-position book: entry at the next bar open, taker "
                  "fills, stop wins ties, purged 5-fold walk-forward, "
                  "friction 12 bp (18 bp PAXG). `folds +` counts how many of "
                  "the five test folds were profitable.\n\n")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(header + text + "\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
