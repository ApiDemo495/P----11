"""Audit every formula against real tape: is it alive, and does it duplicate?

Two kinds of dead weight hide in a 31-formula stack, and they need
different responses:

1. **Structurally silent** — the formula returns a constant because it
   reads state that only exists live (GRC history rings, order book,
   forceOrder prints). That is correct behaviour under the project's
   "cause-6 honesty" rule: no live book, no claim. It is NOT dead code,
   and deleting it would delete a real feature. This audit reports it
   separately so nobody mistakes silence for rot.

2. **Redundant** — the formula moves, but it is ~the same signal as
   another formula (|correlation| above ``DUP_CORR``). Redundancy is real
   cost in a weighted consensus: two copies of one idea vote twice.

The audit runs the app's own ``compute_all`` over the real cached tape,
so it measures the shipped formulas, not a copy.

Usage::

    python -m backtest.formula_audit                 # ETH, TF1 window
    python -m backtest.formula_audit --symbol BTC --tf tf2
"""
from __future__ import annotations

import argparse
import statistics

from app.formulas import FORMULA_FUNCTIONS, FORMULA_NAMES, compute_all
from app.models import FormulaContext
import csv
import os

from backtest.edge_scan import CACHE_DIR, load_series


def load_series_with_volume(symbol: str, tf: str):
    """The live tape carries volume; ``edge_scan.load_series`` drops it.

    Volume-reading formulas (F8 RTDI, F17 VAD) are meaningless on synthetic
    volume, so the audit reads the real ``v`` column itself rather than
    borrowing the price-only loader.
    """
    path = os.path.join(CACHE_DIR, f"{symbol}_{tf}.csv")
    if not os.path.exists(path):
        return None, None
    t, o, h, l, c, v = ([] for _ in range(6))
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) < 6:
                continue
            t.append(int(float(row[0])))
            o.append(float(row[1])); h.append(float(row[2]))
            l.append(float(row[3])); c.append(float(row[4]))
            v.append(float(row[5]))
    ser = load_series(symbol, tf)
    return ser, v

DUP_CORR = 0.85          # |corr| above this = effectively the same signal
MIN_VARYING_SAMPLES = 50

# state a formula needs that a kline replay cannot supply
_LIVE_STATE_TOKENS = ("ctx.hist", "ctx.book", "ctx.flow", "ctx.others",
                      "_book(ctx)", "liq_flow", "ctx.liq_flow", "getattr(ctx, \"book\"")


def _needs_live_state(fid: int) -> bool:
    import inspect

    fn = FORMULA_FUNCTIONS.get(fid)
    if fn is None:
        return False
    src = inspect.getsource(fn)
    return any(tok in src for tok in _LIVE_STATE_TOKENS)


def collect(symbol: str = "ETH", tf: str = "tf1", window_bars: int = 0,
            samples: int = 900) -> dict[int, list[float]]:
    """Sample ``compute_all`` over real minute tape.

    ``window_bars`` must match what the live engine hands the formulas:
    ``build_ctx`` sets ``W = TIMEFRAMES[tf]["minutes"]`` over **1-minute**
    samples. Auditing 1h bars instead shifts every tick-density and
    volume formula onto a tape they were never written for.
    """
    from app.config import TIMEFRAMES

    if window_bars <= 0:
        window_bars = TIMEFRAMES[tf]["minutes"]
    series, vol = load_series_with_volume(symbol, "spot_1m")
    if series is None:
        raise SystemExit("cache missing — run backtest/build_cache.py first")
    out: dict[int, list[float]] = {}
    start = window_bars + 1
    step = max(1, (len(series.c) - start) // samples)
    for k in range(samples):
        i = start + k * step
        if i + 1 > len(series.c):
            break
        lo, hi = i - window_bars, i + 1
        ctx = FormulaContext(
            symbol=symbol, tf=tf, W=window_bars,
            P=list(series.c[lo:hi]),
            V=vol[lo:hi],
            T=list(series.t[lo:hi]),
            hist=None, others={},
            O=list(series.o[lo:hi]),
            H=list(series.h[lo:hi]),
            L=list(series.l[lo:hi]),
        )
        try:
            vals = compute_all(ctx, record=False)
        except Exception:
            continue
        for fid, v in vals.items():
            out.setdefault(fid, []).append(v)
    return out


def audit(symbol: str = "ETH", tf: str = "tf1") -> dict:
    raw = collect(symbol=symbol, tf=tf)
    rows = []
    silent_by_design, dead, varying = [], [], []
    for fid in sorted(raw):
        vals = [x for x in raw[fid] if x == x]
        name = FORMULA_NAMES.get(fid, str(fid))
        if len(vals) < MIN_VARYING_SAMPLES:
            rows.append({"id": fid, "name": name, "n": len(vals),
                         "verdict": "insufficient samples"})
            continue
        mean = sum(vals) / len(vals)
        sd = statistics.pstdev(vals)
        live = _needs_live_state(fid)
        if sd < 1e-12:
            verdict = "silent (needs live state)" if live else "SILENT — dead"
            (silent_by_design if live else dead).append(fid)
        else:
            verdict = "alive"
            varying.append(fid)
        rows.append({"id": fid, "name": name, "n": len(vals),
                     "mean": mean, "sd": sd,
                     "abs_mean_over_sd": (abs(mean) / sd if sd else None),
                     "needs_live_state": live, "verdict": verdict})

    # redundancy among the formulas that actually move
    dup_pairs = []
    for i, a in enumerate(varying):
        for b in varying[i + 1:]:
            va, vb = raw[a], raw[b]
            n = min(len(va), len(vb))
            if n < MIN_VARYING_SAMPLES:
                continue
            x, y = va[:n], vb[:n]
            mx = sum(x) / n
            my = sum(y) / n
            sx = (sum((v - mx) ** 2 for v in x) / n) ** 0.5
            sy = (sum((v - my) ** 2 for v in y) / n) ** 0.5
            if sx <= 0 or sy <= 0:
                continue
            cov = sum((p - mx) * (q - my) for p, q in zip(x, y)) / n
            corr = cov / (sx * sy)
            if abs(corr) >= DUP_CORR:
                dup_pairs.append({
                    "a": a, "b": b, "corr": round(corr, 3),
                    "a_name": FORMULA_NAMES.get(a, str(a)),
                    "b_name": FORMULA_NAMES.get(b, str(b)),
                })

    return {
        "symbol": symbol, "tf": tf,
        "total": len(raw),
        "alive": varying,
        "silent_needs_live_state": silent_by_design,
        "dead": dead,
        "duplicate_pairs": sorted(dup_pairs, key=lambda d: -abs(d["corr"])),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="ETH")
    ap.add_argument("--tf", default="tf1")
    args = ap.parse_args()
    rep = audit(args.symbol, args.tf)

    print(f"{args.symbol} {args.tf}: {rep['total']} formulas")
    print(f"  alive                 : {len(rep['alive'])} {rep['alive']}")
    print(f"  silent (live state)   : {len(rep['silent_needs_live_state'])} "
          f"{rep['silent_needs_live_state']}")
    print(f"  DEAD (should be none) : {len(rep['dead'])} {rep['dead']}")
    print(f"  duplicate pairs >= {DUP_CORR}: {len(rep['duplicate_pairs'])}")
    for d in rep["duplicate_pairs"][:10]:
        print(f"    F{d['a']:<2} ~ F{d['b']:<2} corr {d['corr']:+.2f} "
              f"({d['a_name']} / {d['b_name']})")
    print()
    print(f"{'id':>3} {'name':34s} {'n':>4} {'mean':>8} {'sd':>8}  verdict")
    for r in rep["rows"]:
        if "mean" not in r:
            continue
        print(f"{r['id']:>3} {str(r['name'])[:34]:34s} {r['n']:>4} "
              f"{r['mean']:>8.3f} {r['sd']:>8.3f}  {r['verdict']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
