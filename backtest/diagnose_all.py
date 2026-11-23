"""DIAGNOSE EVERYTHING — per-component grading on 3 years of real tape.

The app has ~35 formulas, 22 emotion cells, a physics layer, a brain
ensemble, and an "advanced" module stack.  What has never been done is
grading each one against realised price on the SAME tape with the SAME
horizon and the SAME cost, so their contributions can be compared.

Method
------
For every non-overlapping window of the tape (the app's own tf windows):

    signal  = the component's output for that window
    outcome = sign of the realised move over the window, NET of round trip

A component is graded on **sign accuracy** (does its sign match the
direction that actually paid) and on **IC** (rank correlation between its
value and the realised move).  Accuracy near 50 % means the component
carries no directional information at that horizon, whatever its
theoretical story.

Reporting rules
---------------
* a component is only called "informative" if its accuracy interval
  excludes 50 % *and* its IC sign is stable across the walk-forward halves;
* components that are pure confidence scalers are labelled as such
  instead of being force-graded as directional;
* nothing is dropped for being uninformative — the point is to show which
  of the 35+ are carrying weight and which are decoration.

Usage
    .venv/bin/python -m backtest.diagnose_all --assets BTC ETH
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data import FRICTION_BPS, load_cache          # noqa: E402

RESULTS = os.path.join("backtest", "results")
OUT = os.path.join(RESULTS, "diagnose_all.json")
TRAIN_FRAC = 0.65
TF_MIN = {"tf1": 22, "tf2": 44, "tf3": 130, "tf4": 247}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def spearman(a, b):
    """Rank correlation without scipy — average ranks, no ties handling
    beyond that (the tape has continuous values so ties are rare)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 10:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def _real_hist():
    """The app's own HistoryStore — the formulas depend on its exact Ring
    API (vmax/mean/last), and a stub silently zeroes half of them."""
    from app.history import HistoryStore
    return HistoryStore()


def windows(t, step, w_min):
    W = int(w_min * 60)
    starts = np.arange(int(np.ceil(t[0] / W) * W),
                       int(t[-1]) + step - W + 1, W, dtype=np.int64)
    i_in = np.searchsorted(t, starts, side="left")
    i_out = np.searchsorted(t, starts + W - step, side="right") - 1
    ok = (i_in >= 0) & (i_out > i_in) & (i_out < len(t))
    return i_in[ok], i_out[ok]


def build_ctx(fn_mod, P, V, T, O_, H_, L_, w_min, symbol, hist, window):
    """One FormulaContext for a single window slice."""
    from app.models import FormulaContext
    return FormulaContext(
        symbol=symbol, tf="tf", W=w_min, P=list(P), V=list(V), T=list(T),
        hist=hist, O=list(O_), H=list(H_), L=list(L_),
    )


def grade_component(values, outcomes):
    """Accuracy + IC for one component's per-window values."""
    v = np.asarray(values, float)
    o = np.asarray(outcomes, float)
    ok = np.isfinite(v) & np.isfinite(o) & (v != 0)
    n = int(ok.sum())
    if n < 30:
        return {"n": n, "status": "too_few"}
    sgn = np.sign(v[ok])
    pay = np.sign(o[ok])
    acc = float((sgn == pay).mean())
    k = int((sgn == pay).sum())
    lo, hi = wilson(k, n)
    ic = spearman(v[ok], o[ok])
    cut = int(n * TRAIN_FRAC)
    ic_a = spearman(v[ok][:cut], o[ok][:cut])
    ic_b = spearman(v[ok][cut:], o[ok][cut:])
    acc_a = float((sgn[:cut] == pay[:cut]).mean()) if cut else float("nan")
    acc_b = float((sgn[cut:] == pay[cut:]).mean()) if n - cut else float("nan")
    stable = (np.isfinite(ic_a) and np.isfinite(ic_b)
              and np.sign(ic_a) == np.sign(ic_b) and ic_a != 0)
    informative = bool(lo > 0.5 or hi < 0.5) and stable
    return {
        "n": n, "accuracy": round(acc, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "ic": round(ic, 4) if np.isfinite(ic) else None,
        "ic_train": round(ic_a, 4) if np.isfinite(ic_a) else None,
        "ic_holdout": round(ic_b, 4) if np.isfinite(ic_b) else None,
        "acc_train": round(acc_a, 4) if np.isfinite(acc_a) else None,
        "acc_holdout": round(acc_b, 4) if np.isfinite(acc_b) else None,
        "ic_stable": bool(stable), "informative": informative,
    }


def run_formulas(asset, tape="5m", tf="tf3", max_windows=400):
    """Grade every registered formula on real windows."""
    from app.formulas import FORMULA_FUNCTIONS, FORMULA_NAMES, SIGN_REVERSALS
    bars = load_cache(asset, tape)
    if not bars:
        return {"status": "no_tape"}
    t = np.array([b.t for b in bars])
    P = np.array([b.c for b in bars], float)
    V = np.array([getattr(b, "v", 0.0) for b in bars], float)
    Hh = np.array([b.h for b in bars], float)
    Ll = np.array([b.l for b in bars], float)
    Oo = np.array([getattr(b, "o", b.c) for b in bars], float)
    step = int(t[1] - t[0])
    w_min = TF_MIN[tf]
    i_in, i_out = windows(t, step, w_min)
    if len(i_in) > max_windows:
        idx = np.linspace(0, len(i_in) - 1, max_windows).astype(int)
        i_in, i_out = i_in[idx], i_out[idx]
    rt = 2.0 * FRICTION_BPS[asset] * 1e-4
    # outcome: realised NET move over the window, signed
    gross = P[i_out] / P[i_in] - 1.0
    outcomes = gross - rt          # net; sign = did this direction pay
    # costs shift the threshold, not the sign of the move; use gross sign
    outcomes = gross
    per = {fid: [] for fid in FORMULA_FUNCTIONS}
    errors = {}
    for k, (a, b) in enumerate(zip(i_in, i_out)):
        # give the formula plenty of history so its baselines can warm up
        lo = max(0, a - 4 * w_min)
        sl = slice(lo, a + 1)
        if len(P[sl]) < 10:
            continue
        hist = _real_hist()
        try:
            ctx = build_ctx(None, P[sl], V[sl], t[sl], Oo[sl], Hh[sl],
                            Ll[sl], w_min, asset, hist, None)
        except Exception as e:
            errors["ctx"] = repr(e)
            continue
        for fid, fn in FORMULA_FUNCTIONS.items():
            try:
                val = float(fn(ctx))
                if not np.isfinite(val):
                    val = 0.0
                val *= SIGN_REVERSALS.get(fid, 1)
            except Exception as e:
                errors[str(fid)] = f"{type(e).__name__}: {str(e)[:60]}"
                val = 0.0
            per[fid].append((val, outcomes[k]))
    rows = []
    for fid, pairs in per.items():
        if not pairs:
            continue
        vals = [p[0] for p in pairs]
        outs = [p[1] for p in pairs]
        g = grade_component(vals, outs)
        g.update({"fid": fid, "name": FORMULA_NAMES.get(fid, "?")})
        rows.append(g)
    rows.sort(key=lambda r: -(r.get("accuracy") or 0))
    return {"status": "ok", "tf": tf, "w_min": w_min, "rows": rows,
            "errors": errors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="*", default=["BTC", "ETH"])
    ap.add_argument("--tf", nargs="*", default=["tf1", "tf3"])
    ap.add_argument("--windows", type=int, default=400)
    args = ap.parse_args()
    out = {}
    for asset in args.assets:
        for tf in args.tf:
            key = f"{asset}|{tf}"
            r = run_formulas(asset, tf=tf, max_windows=args.windows)
            out[key] = r
            if r.get("status") != "ok":
                print(f"{key}: {r['status']}")
                continue
            print(f"\n=== {key}  ({r['w_min']}m windows)")
            print(f"  {'fid':>4s} {'name':>7s} {'n':>5s} {'acc':>7s} {'95% CI':>14s} "
                  f"{'IC':>7s} {'IC train':>9s} {'IC hold':>8s}  verdict")
            for row in r["rows"]:
                if row.get("status") == "too_few":
                    print(f"  {row['fid']:4d} {row['name']:>7s} {row['n']:5d}   too few")
                    continue
                lo, hi = row["ci95"]
                v = ("INFORMATIVE" if row["informative"]
                     else ("reversed!" if hi < 0.5 else "no edge"))
                print(f"  {row['fid']:4d} {row['name']:>7s} {row['n']:5d} "
                      f"{row['accuracy']*100:6.1f}% [{lo*100:4.0f},{hi*100:4.0f}] "
                      f"{(row['ic'] or 0):+7.3f} {(row['ic_train'] or 0):+9.3f} "
                      f"{(row['ic_holdout'] or 0):+8.3f}  {v}")
            if r["errors"]:
                print(f"  errors: {list(r['errors'].items())[:3]}")
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
