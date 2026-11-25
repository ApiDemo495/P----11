"""Regenerate the trend-interaction gate's numbers from the real record.

Usage::

    python -m backtest.trend_gate_audit            # human summary
    python -m backtest.trend_gate_audit --json     # machine summary
    python -m backtest.trend_gate_audit --write    # refresh the committed json

The substrate is ``backtest/results/per_tf_full_engine_real.csv`` — the
engine's own 893 settled windows over 72 days of real tape, each carrying the
realised net bps and the win flag. Nothing here re-runs the engine, and no
row is invented: if the CSV is missing the audit refuses rather than guessing.

The rule under test is exactly ``app/trend_gate.py``'s::

    trend = sign(entry[i] - entry[i - LAG])     per cell, LAG = 2 sampled rows
    side opposes trend -> stands     side chases trend -> withheld

Everything is reported, including what fails: per-asset, per-cell, five
temporal folds, and a 65/35 time split of the gated rows. ``bps_net`` is
outcome-derived in this file, so it is never used as a filter — only the
recorded ``win`` column is scored.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict

CSV_PATH = os.path.join("backtest", "results", "per_tf_full_engine_real.csv")
OUT_PATH = os.path.join("backtest", "results", "trend_gate_audit.json")
LAG = 2


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def _slice(rs: list[dict]) -> dict:
    a = [r for r in rs if r["agrees"]]
    o = [r for r in rs if not r["agrees"]]
    sa, so = sum(r["win"] for r in a), sum(r["win"] for r in o)
    return {
        "n": len(rs),
        "agree_n": len(a), "agree_wr": round(sa / max(1, len(a)), 4),
        "agree_ci": wilson(sa, len(a)),
        "opp_n": len(o), "opp_wr": round(so / max(1, len(o)), 4),
        "opp_ci": wilson(so, len(o)),
    }


def build(path: str = CSV_PATH, lag: int = LAG) -> dict:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        raise SystemExit(f"{path}: no rows — cannot audit anything")
    by: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    for r in rows:
        by[(r["asset"], r["tf"])].append(
            (float(r["t"]), float(r["entry"]), int(float(r["direction"])),
             int(float(r["win"])), float(r["conf"])))
    recs: list[dict] = []
    for cell, seq in sorted(by.items()):
        seq.sort()
        for k in range(lag, len(seq)):
            move = seq[k][1] - seq[k - lag][1]
            if move == 0:
                continue
            trend = 1 if move > 0 else -1
            recs.append({"asset": cell[0], "tf": cell[1], "t": seq[k][0],
                         "entry": seq[k][1], "side": seq[k][2],
                         "win": seq[k][3], "conf": seq[k][4],
                         "trend": trend, "agrees": trend == seq[k][2]})
    wins = sum(int(float(r["win"])) for r in rows)
    out: dict = {
        "source": path, "lag_rows": lag,
        "baseline_n": len(rows),
        "baseline_wr": round(wins / len(rows), 4),
        "baseline_ci": wilson(wins, len(rows)),
        "pooled": _slice(recs),
    }
    trend_wins = sum(1 for r in recs
                     if (r["win"] == 1) == (r["trend"] == r["side"]))
    out["trend_as_side"] = {"n": len(recs),
                            "wr": round(trend_wins / max(1, len(recs)), 4),
                            "ci": wilson(trend_wins, len(recs))}
    out["per_asset"] = {a: _slice([r for r in recs if r["asset"] == a])
                        for a in sorted({r["asset"] for r in recs})}
    recs.sort(key=lambda r: r["t"])
    n = len(recs)
    out["folds"] = []
    for f in range(5):
        lo, hi = f * n // 5, (f + 1) * n // 5
        g = [r for r in recs[lo:hi] if not r["agrees"]]
        w = sum(r["win"] for r in g)
        out["folds"].append({"fold": f + 1, "n": len(g),
                             "wr": round(w / max(1, len(g)), 4),
                             "ci": wilson(w, len(g))})
    opp = [r for r in recs if not r["agrees"]]
    cut = int(len(opp) * 0.65)
    tr, ho = opp[:cut], opp[cut:]
    tw, hw = sum(r["win"] for r in tr), sum(r["win"] for r in ho)
    out["split"] = {
        "train_n": len(tr), "train_wr": round(tw / max(1, len(tr)), 4),
        "train_ci": wilson(tw, len(tr)),
        "hold_n": len(ho), "hold_wr": round(hw / max(1, len(ho)), 4),
        "hold_ci": wilson(hw, len(ho)),
    }
    # WHAT THE APP ACTUALLY ISSUES, end to end: confidence above the asset's
    # own issue floor (app/config.asset_issue_floor) AND the call not chasing
    # the trend. This is the number the user experiences, so it is reported
    # with its coverage — a higher win rate bought with silence is not a win.
    from app.config import asset_issue_floor
    deployed = []
    for cell, seq in sorted(by.items()):
        seq.sort()
        floor = asset_issue_floor(cell[0])
        for k in range(len(seq)):
            t, entry, side, win, conf = seq[k]
            if conf <= floor:
                continue
            if k >= lag:
                mv = seq[k][1] - seq[k - lag][1]
                if mv != 0 and (1 if mv > 0 else -1) == side:
                    continue
            deployed.append({"asset": cell[0], "tf": cell[1], "t": t,
                             "win": win})
    dw = sum(r["win"] for r in deployed)
    out["deployed"] = {
        "rule": "confidence > asset_issue_floor AND side does not chase the "
                f"{lag}-row trend",
        "n": len(deployed),
        "coverage": round(len(deployed) / len(rows), 4),
        "wr": round(dw / max(1, len(deployed)), 4),
        "ci": wilson(dw, len(deployed)),
        "per_asset": {},
    }
    for a in sorted({r["asset"] for r in deployed}):
        g = [r for r in deployed if r["asset"] == a]
        w = sum(r["win"] for r in g)
        out["deployed"]["per_asset"][a] = {
            "n": len(g), "wr": round(w / max(1, len(g)), 4), "ci": wilson(w, len(g))}
    deployed.sort(key=lambda r: r["t"])
    cut = int(len(deployed) * 0.65)
    tr, ho = deployed[:cut], deployed[cut:]
    tw, hw = sum(r["win"] for r in tr), sum(r["win"] for r in ho)
    out["deployed"]["split"] = {
        "train_n": len(tr), "train_wr": round(tw / max(1, len(tr)), 4),
        "train_ci": wilson(tw, len(tr)),
        "hold_n": len(ho), "hold_wr": round(hw / max(1, len(ho)), 4),
        "hold_ci": wilson(hw, len(ho)),
    }
    out["per_cell_gated"] = []
    for cell in sorted(by):
        g = [r for r in recs if r["asset"] == cell[0] and r["tf"] == cell[1]
             and not r["agrees"]]
        w = sum(r["win"] for r in g)
        out["per_cell_gated"].append({
            "cell": f"{cell[0]}|{cell[1]}", "n": len(g),
            "wr": round(w / max(1, len(g)), 4), "ci": wilson(w, len(g))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="print the summary")
    ap.add_argument("--write", action="store_true",
                    help=f"write {OUT_PATH}")
    args = ap.parse_args()
    rep = build()
    if args.write:
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
            fh.write("\n")
    if args.json:
        print(json.dumps(rep, indent=1))
        return
    p, s, t = rep["pooled"], rep["split"], rep["trend_as_side"]
    print(f"source          {rep['source']}  ({rep['lag_rows']} rows lag)")
    print(f"every window    n={rep['baseline_n']:4d}  "
          f"{100 * rep['baseline_wr']:5.1f} %  CI {rep['baseline_ci']}")
    print(f"side CHASES     n={p['agree_n']:4d}  "
          f"{100 * p['agree_wr']:5.1f} %  CI {p['agree_ci']}")
    print(f"side FADES      n={p['opp_n']:4d}  "
          f"{100 * p['opp_wr']:5.1f} %  CI {p['opp_ci']}   <- the call that "
          f"is kept")
    print(f"trend as side   n={t['n']:4d}  {100 * t['wr']:5.1f} %  CI {t['ci']}")
    print("folds (fading)  " + "  ".join(
        f"f{f['fold']} {100 * f['wr']:.1f}% (n={f['n']})" for f in rep["folds"]))
    print(f"65/35 split     train n={s['train_n']} {100 * s['train_wr']:.1f} % "
          f"CI {s['train_ci']} | holdout n={s['hold_n']} "
          f"{100 * s['hold_wr']:.1f} % CI {s['hold_ci']}")
    dep = rep.get("deployed") or {}
    if dep:
        ds = dep["split"]
        print(f"\nWHAT SHIPS     n={dep['n']:4d} ({100 * dep['coverage']:.0f}% of "
              f"windows)  {100 * dep['wr']:5.1f} %  CI {dep['ci']}")
        print(f"  vs all windows  {100 * rep['baseline_wr']:5.1f} %  CI "
              f"{rep['baseline_ci']}")
        print("  per asset    " + "  ".join(
            f"{a} {100 * v['wr']:.0f}% (n={v['n']})"
            for a, v in dep["per_asset"].items()))
        print(f"  65/35 split  train n={ds['train_n']} "
              f"{100 * ds['train_wr']:.1f} % | holdout n={ds['hold_n']} "
              f"{100 * ds['hold_wr']:.1f} % CI {ds['hold_ci']}")
    worst = sorted(rep["per_cell_gated"], key=lambda r: r["wr"])[:4]
    print("weakest cells   " + "  ".join(
        f"{r['cell']} {100 * r['wr']:.1f}% (n={r['n']})" for r in worst))
    print("\nHonest reading: in-sample and selection-aware. The pooled gap is "
          "large,\nbut the holdout interval includes 50 %, so this is an "
          "abstention rule\nbacked by a plausible mechanism — not a promised "
          "forward win rate.")


if __name__ == "__main__":
    main()
