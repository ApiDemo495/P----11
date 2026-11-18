#!/usr/bin/env python3
"""Edge scan — settle every candidate structure on a sequential book.

This is the falsification harness for the 2026-11 programme.  Each row is
one *fixed* configuration, evaluated out-of-fold on a single-position book
(entry at the next bar open, taker fills, stop wins ties, no pyramiding)
with purged folds and an overlap-aware t-statistic.

    python -m backtest.derivs_scan                     # canonical table
    python -m backtest.derivs_scan --only cascade -v
    python -m backtest.derivs_scan --write             # refresh the report

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

from app import derivs, event_alpha  # noqa: E402
from app.alpha_lab import build_matrix, load_csv, purged_folds  # noqa: E402
from app.ensemble import ic_weights, zscore_train  # noqa: E402
from app.events import sigma_series  # noqa: E402
from app.portfolio import block_bootstrap, run_sleeve, trade_stats  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
FEE = {"BTC": 12.0, "ETH": 12.0, "BNB": 12.0, "PAXG": 18.0}


def _load(sym: str, interval: str, kind: str | None = None):
    """Load a cached series; derivatives live under cache/derivs/."""
    if kind:
        return load_csv(os.path.join(CACHE, "derivs", f"{sym}_{interval}.csv"))
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


# ───────────────── event sleeves (derivatives + convex payoffs) ─────────────────


def oof_event(bars, score, mask, sigma, *, sl=1.0, hz=48, trail=3.0,
              pullback=0.25, fill_mode="next_bar", fees_bps=8.0,
              maker_fee_bps=2.0, taker_fee_bps=5.0, penalty_bps=1.0,
              entry="maker_pullback", folds=5) -> dict:
    """One event sleeve, settled sequentially and scored out-of-fold."""
    n = len(bars.c)
    trades, fev = [], []
    for _tr, te in purged_folds(n, folds=folds, embargo=max(96, hz)):
        m = np.zeros(n, bool)
        m[te[0]:te[-1] + 1] = True
        t = run_sleeve(bars, score, np.asarray(mask, bool) & m, sigma=sigma,
                       pt=6.0, sl=sl, hz=hz, trail=trail, runner=True,
                       entry=entry, pullback=pullback,
                       maker_fee_bps=maker_fee_bps, taker_fee_bps=taker_fee_bps,
                       maker_penalty_bps=penalty_bps, min_penetration=1e-4,
                       fill_mode=fill_mode, fees_bps=fees_bps)
        trades += t
        st = trade_stats(t, fees_bps=fees_bps)
        fev.append(round(st["ev_bp"], 1) if st["n"] else 0.0)
    st = trade_stats(trades, fees_bps=fees_bps)
    st["folds"] = fev
    st["folds_positive"] = f"{sum(1 for f in fev if f > 0)}/{len(fev)}"
    st["boot"] = block_bootstrap([x["net"] for x in trades], block=10)
    st["trades"] = trades
    return st


def event_arms(assets=("BTC", "ETH", "BNB", "PAXG")) -> list[tuple]:
    """Every measured sleeve, with its rejection recorded, not hidden."""
    arms = []
    if "ETH" in assets:
        eth, btc = _load("ETH", "1h"), _load("BTC", "1h")
        arms.append(("ETH catch-up (BTC impulse)", eth,
                     event_alpha.catchup(eth, btc),
                     sigma_series(eth, 240), "leadlag", {}))
    if "BNB" in assets:
        bnb, btc = _load("BNB", "1h"), _load("BTC", "1h")
        arms.append(("BNB catch-up (BTC impulse)", bnb,
                     event_alpha.catchup(bnb, btc),
                     sigma_series(bnb, 240), "leadlag", {}))
    if "PAXG" in assets:
        paxg, btc = _load("PAXG", "1h"), _load("BTC", "1h")
        arms.append(("PAXG catch-up (gold flight)", paxg,
                     event_alpha.catchup(paxg, btc),
                     sigma_series(paxg, 240), "leadlag", {}))
    if "BTC" in assets:
        perp = _load("BTC", "perp_1h", kind="derivs")
        d = derivs.load("BTC")
        f = derivs.align(d, np.asarray(perp.t, float))
        sg = sigma_series(perp, 240)
        arms.append(("BTC squeeze (compression+funding)",
                     perp, event_alpha.squeeze(perp, f), sg, "squeeze", {}))
        arms.append(("BTC basis exhaustion fade", perp,
                     event_alpha.basis_fade(perp, f), sg, "basis_fade", {}))
    return arms


def events_table(assets=("BTC", "ETH", "BNB", "PAXG"), verbose=False) -> list[str]:
    rows = ["| sleeve | n | gross bp | net bp | fee bp | PF | boot t | folds + |",
            "|---|---:|---:|---:|---:|---:|---:|---|"]
    for label, bars, sig, sg, plan_name, kw in event_arms(assets):
        p = event_alpha.plan(plan_name)
        st = oof_event(bars, sig["score"], sig["mask"], sg,
                       sl=p["sl"], hz=p["hz"], trail=p["trail"],
                       pullback=p["pullback"], entry=p["entry"], **kw)
        b = st["boot"]
        rows.append("| {l} | {n} | {g:+.1f} | {e:+.1f} | {f:.1f} | {pf} | "
                    "{t:+.2f} | {pos} |".format(
                        l=label, n=st["n"], g=st["gross_ev_bp"],
                        e=st["ev_bp"], f=st["fee_bp"],
                        pf=(f"{st['profit_factor']:.2f}"
                            if math.isfinite(st["profit_factor"]) else "∞"),
                        t=b["t"], pos=st["folds_positive"]))
        if verbose:
            print(f"    {label}: folds {st['folds']} CI "
                  f"[{b['lo_bp']:+.0f},{b['hi_bp']:+.0f}]")
    return rows


def fill_bias_probe() -> list[str]:
    """Prove the maker fill model is not manufacturing the edge.

    A coin-flip side on a synthetic random walk must earn ~zero gross.  The
    naive "fill at the limit inside the touched bar" mode pays ~+15 bp —
    free money that no venue offers — which is why `next_bar` is the default.
    """
    rng = np.random.default_rng(3)
    n = 20000
    r = rng.normal(0, 0.006, n)
    c = 100 * np.exp(np.cumsum(r))
    o = c * np.exp(rng.normal(0, 0.006 * 0.4, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, n)))
    bars = type("B", (), {"o": o, "h": h, "l": lo, "c": c,
                          "t": np.arange(n) * 3600.0, "v": np.ones(n)})()
    sig = np.full(n, 0.006)
    side = np.sign(np.random.default_rng(0).normal(size=n))
    mask = np.ones(n, bool)
    rows = ["| fill model | n | gross bp | verdict |", "|---|---:|---:|---|"]
    for mode in ("wick", "next_bar"):
        tr = run_sleeve(bars, side, mask, sigma=sig, pt=6.0, sl=1.0, hz=48,
                        trail=3.0, runner=True, entry="maker_pullback",
                        pullback=0.5, maker_fee_bps=2.0, taker_fee_bps=5.0,
                        maker_penalty_bps=1.0, min_penetration=1e-4,
                        fill_mode=mode)
        st = trade_stats(tr, fees_bps=8.0)
        verdict = ("fiction — a coin flip cannot earn this"
                   if abs(st["gross_ev_bp"]) > 6 else "unbiased")
        rows.append(f"| {mode} | {st['n']} | {st['gross_ev_bp']:+.2f} | "
                    f"{verdict} |")
    return rows


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


HEADER = """# Edge scan — sequential out-of-fold settlement

> **Superseded in part (2026-11-18 merge).** Block 1's two positive
> readings (ETH catch-up +17.0 bp, BTC squeeze +6.4 bp) did **not**
> reproduce from a from-scratch rebuild — see `docs/ALPHA_RESEARCH.md`
> §0 (overlapping trades counted as independent, friction charged per
> episode in places, best-of-535 selection). They stay on the record so
> nobody re-discovers them as new; the live registry
> (`app/cost_gate.py::SLEEVES`) marks both `rejected`, and the certified
> book is `backtest/results/EDGE_SCAN.md` (§4). What still stands in this
> file: Block 2's fill-model bias probe (live in `--only bias`) and Block
> 3's per-bar falsifications (consistent with §4's closed list).

Every row is a *fixed* configuration, settled on a single-position book with
purged five-fold walk-forward and overlap-aware statistics. Parameters are
never searched inside a fold: a searched number describes the search, not the
market.

**Block 1 — event sleeves** (multi-year archive: spot 1h 2024-01 → 2026-09,
BTC USD-M perp 1h + funding/basis 2023-01 → 2025-11). Asymmetric convex exits
(tight structural stop, uncapped trailing runner), maker-pullback entries
filled **one bar later** at 8 bp round-trip (2 bp maker + 5 bp taker exit +
1 bp adverse-fill penalty).

**Block 2 — fill-model bias probe.** A coin flip on a synthetic random walk.
The naive "fill at the limit inside the bar that touched it" model pays it
~+15 bp per trade, which no venue pays for liquidity provision; the default
`next_bar` fill is the guard against that fiction.

**Block 3 — per-bar sleeves** (same archive, reference only): taker fills at
12 bp (18 bp PAXG). `folds +` counts how many of the five test folds were
profitable.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="ensemble / classic / cascade / leadlag / events / "
                         "bias")
    ap.add_argument("--assets", nargs="*", default=["BTC", "ETH", "BNB", "PAXG"])
    ap.add_argument("--write", action="store_true",
                    help="write backtest/results/DERIVS_SCAN.md")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    rows = []
    full = not args.only          # a bare run documents everything measured
    if full or "events" in (args.only or []):
        rows += events_table(tuple(args.assets), verbose=args.verbose)
    if full or "bias" in (args.only or []):
        rows += fill_bias_probe()
    if full or any(o not in ("events", "bias") for o in (args.only or [])):
        rows = (rows + ["", ""] if rows else []) + canonical(
            tuple(args.assets), [o for o in (args.only or []) if o not in
                                 ("events", "bias")] or None)
    text = "\n".join(rows)
    print(text)
    if args.write:
        suffix = "_EVENTS" if (args.only and any(
            o in ("events", "bias") for o in args.only)) else ""
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", f"DERIVS_SCAN{suffix}.md")
        header = HEADER
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(header + text + "\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
