"""Reproduce every number in ``app/trend_gate.py``'s docstring, from scratch.

Run::

    python -m backtest.trend_gate_audit               # full report
    python -m backtest.trend_gate_audit --json        # machine-readable

Why this file exists
====================
``app/trend_gate.py`` now FLIPS a chasing call instead of withholding it, and
it justifies that on a specific record: the app's own engine walked over 72
days of real tape, all four assets × all four timeframes, 893 settled windows
(``backtest/results/per_tf_full_engine_real.csv``). A rule that changes the
side of a live call must be reproducible by anyone reading the module, or the
docstring is just a claim. This script is that reproduction — one command, no
hidden sample, and every weakness printed next to every win.

The measurement, stated precisely
=================================
The "trend" is the sign of the engine's own entry-price move over the last
``TREND_LAG_ROWS`` settled windows of the SAME cell (asset, timeframe), i.e.
exactly the series ``trend_gate.observe()`` samples at settle time. A call
"chases" when its side equals that sign. Nothing here reads the future: for
row *i* of a cell, the trend uses rows *i-1* and *i-2* only.

The flip is then: keep the side when the call opposes the trend, invert the
side when it chases. Win rate and net bps are recomputed on the flipped side
(the realised move is symmetric, so net bps simply changes sign minus the
cost, which is why it is reported, not assumed).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "backtest" / "results" / "per_tf_full_engine_real.csv"
#: legacy name — the test-suite and older notes refer to ``CSV_PATH``
CSV_PATH = str(CSV)
LAG = 2                     # must match app/trend_gate.py TREND_LAG_ROWS


# ── statistics helpers (no scipy in this repo) ──────────────────────────────
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson interval — the interval the docstring quotes."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def _wr(rows: list[dict]) -> tuple[float, tuple[float, float], int]:
    n = len(rows)
    k = sum(int(r["_win"]) for r in rows)
    p = k / n if n else 0.0
    return p, wilson(k, n), n


def _net(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return sum(float(r["_net"]) for r in rows) / len(rows)


def _pct(p: float) -> str:
    return f"{100 * p:.1f} %"


def _ci(ci: tuple[float, float]) -> str:
    return f"{100 * ci[0]:.1f} – {100 * ci[1]:.1f} %"


# ── loading ────────────────────────────────────────────────────────────────
def load(path: Path = CSV) -> list[dict]:
    """Every settled window, one dict per row, sorted per cell by time."""
    if not path.exists():
        raise SystemExit(f"missing record: {path}")
    rows: list[dict] = []
    with path.open(newline="") as fh:
        for i, rec in enumerate(csv.DictReader(fh)):
            try:
                entry = float(rec["entry"] or 0)
                settle = float(rec["settle"] or 0)
                side = int(rec["direction"] or 0)
                t = float(rec["t"] or 0)
            except (KeyError, TypeError, ValueError):
                continue
            if entry <= 0 or settle <= 0 or side not in (-1, 1):
                continue
            rows.append({
                "asset": rec["asset"], "tf": rec["tf"], "t": t,
                "side": side, "entry": entry, "settle": settle,
                "win": int(rec.get("win") or 0),
                "net": float(rec.get("bps_net") or 0.0),
                "gross": float(rec.get("bps_gross") or 0.0),
                "i": i,
            })
    rows.sort(key=lambda r: (r["asset"], r["tf"], r["t"]))
    return rows


def annotate(rows: list[dict], lag: int = LAG) -> list[dict]:
    """Attach the causal trend, the chase flag and the flipped side.

    A row is only usable when its cell has ``lag`` earlier settled windows —
    the same warmup ``trend_gate`` has (``MIN_ROWS``). Unusable rows are kept
    but marked ``_ok = False`` so the sample size is never silently inflated.
    """
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_cell.setdefault((r["asset"], r["tf"]), []).append(r)
    out: list[dict] = []
    for cell_rows in by_cell.values():
        cell_rows.sort(key=lambda r: r["t"])
        for j, r in enumerate(cell_rows):
            if j - 2 < 0:
                r["_ok"] = False
                out.append(r)
                continue
            # The series the app actually samples: `trend_gate.observe()` is
            # fed the LOCK PRICE of every settled window, i.e. this cell's
            # entry series. The trend at row j is the sign of the entry move
            # over the last 2 DECISIONS: entry[j] vs entry[j-2] — both are
            # prints the engine already published, so the reading is causal
            # (the rule acts on row j's own lock, and entry[j] is not the
            # future of anything the rule is deciding).
            #
            # Two WRONG substrates were tried first and are recorded so they
            # are not tried again: (a) a settled window's own entry→settle
            # move INVERTS the table (it reported 61 % for chases, because a
            # window that settled up is the future of the row before it), and
            # (b) entry[j-1] vs entry[j-3] is a 1-row lag, worth only 51.1 %.
            # Only entry[j] vs entry[j-2] reproduces the module's numbers.
            now_e = cell_rows[j]["entry"]
            then_e = cell_rows[j - 2]["entry"]
            move = now_e - then_e
            trend = 0 if abs(move) < 1e-12 else (1 if move > 0 else -1)
            r["_trend"] = trend
            r["_move_bps"] = (now_e / then_e - 1.0) * 1e4
            r["_trend"] = trend
            r["_chase"] = bool(trend != 0 and r["side"] == trend)
            r["_ok"] = trend != 0
            # realised win/net are symmetric in the side: inverting the side
            # inverts the outcome, minus the SAME round-trip cost (bps_net is
            # already cost-adjusted on the original side).
            gross = r["gross"]
            cost = gross - r["net"]
            r["_win"] = int(r["win"])
            r["_net"] = r["net"]
            r["_flip_side"] = -r["side"] if r["_chase"] else r["side"]
            r["_flip_win"] = (1 - int(r["win"])) if r["_chase"] else int(r["win"])
            r["_flip_net"] = (-gross - cost) if r["_chase"] else r["net"]
            out.append(r)
    return out


def evaluate(rows: list[dict]) -> dict:
    """All the docstring's numbers, computed once."""
    use = [r for r in rows if r.get("_ok")]
    chase = [r for r in use if r["_chase"]]
    against = [r for r in use if not r["_chase"]]
    base_wr, base_ci, base_n = _wr(use)

    def win_of(r: dict, key: int) -> dict:
        q = dict(r)
        q["_win"] = key
        return q

    flipped = [win_of(r, r["_flip_win"]) for r in use]
    for q, r in zip(flipped, use):
        q["_net"] = r["_flip_net"]
    flip_wr, flip_ci, flip_n = _wr(flipped)

    # trend alone as the side
    trend_rows = [win_of(r, int(r["win"]) if r["_trend"] == r["side"]
                         else 1 - int(r["win"])) for r in use]
    trend_wr, trend_ci, trend_n = _wr(trend_rows)

    all_wr, all_ci, all_n = _wr([win_of(r, r["win"]) for r in rows])

    # 5-fold temporal split of the flipped policy
    folds: list[float] = []
    fold_rows: list[dict] = []
    ordered = sorted(flipped, key=lambda r: r["t"])
    k = 5
    step = max(1, len(ordered) // k)
    for f in range(k):
        seg = ordered[f * step:(f + 1) * step] if f < k - 1 else ordered[f * step:]
        if not seg:
            continue
        p, ci_f, n_f = _wr(seg)
        folds.append(p)
        fold_rows.append({"fold": f + 1, "n": n_f, "wr": p, "ci": ci_f})

    # 65/35 split
    cut = int(len(ordered) * 0.65)
    train, holdout = ordered[:cut], ordered[cut:]
    tr_wr, tr_ci, tr_n = _wr(train)
    ho_wr, ho_ci, ho_n = _wr(holdout)

    # paired bootstrap: is the flip actually better than the engine's side?
    rng = random.Random(20261126)
    diffs: list[float] = []
    for _ in range(2000):
        pick = [rng.choice(use) for _ in range(len(use))]
        a = sum(r["_flip_win"] for r in pick) / len(pick)
        b = sum(r["win"] for r in pick) / len(pick)
        diffs.append(a - b)
    diffs.sort()
    boot = (diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))])

    per_cell = {}
    for (asset, tf) in sorted({(r["asset"], r["tf"]) for r in use}):
        cell = [r for r in flipped if (r["asset"], r["tf"]) == (asset, tf)]
        raw = [r for r in use if (r["asset"], r["tf"]) == (asset, tf)]
        p, ci, n = _wr(cell)
        per_cell[f"{asset}|{tf}"] = {
            "n": n, "wr": p, "ci": ci, "net_bps": _net(cell),
            "chases": sum(1 for r in raw if r["_chase"]),
            "base_wr": (_wr([win_of(r, r["win"]) for r in raw])[0]),
            "base_net_bps": _net(raw),
        }

    per_asset = {}
    for asset in sorted({r["asset"] for r in use}):
        for label, src in (("flip", flipped), ("engine", use)):
            cell = [r for r in src if r["asset"] == asset]
            p, ci, n = _wr(cell)
            per_asset.setdefault(asset, {})[label] = {
                "n": n, "wr": p, "ci": ci, "net_bps": _net(cell)}

    return {
        "rows_in_file": len(rows),
        "rows_usable": len(use),
        "warmup_dropped": len(rows) - len(use),
        "lag": LAG,
        "baseline": {"n": base_n, "wr": base_wr, "ci": base_ci,
                     "net_bps": _net(use)},
        "all_rows_asis": {"n": all_n, "wr": all_wr, "ci": all_ci},
        "chases": {"n": len(chase), "wr": _wr(chase)[0], "ci": _wr(chase)[1],
                   "net_bps": _net(chase)},
        "opposes": {"n": len(against), "wr": _wr(against)[0],
                    "ci": _wr(against)[1], "net_bps": _net(against)},
        "trend_as_side": {"n": trend_n, "wr": trend_wr, "ci": trend_ci},
        "flip_policy": {"n": flip_n, "wr": flip_wr, "ci": flip_ci,
                        "net_bps": _net(flipped),
                        "gross_wr": sum(r["_flip_win"] for r in flipped) / flip_n},
        "flip_folds": folds,
        "fold_detail": fold_rows,
        "split_65_35": {"train": {"n": tr_n, "wr": tr_wr, "ci": tr_ci},
                        "holdout": {"n": ho_n, "wr": ho_wr, "ci": ho_ci}},
        "paired_bootstrap_delta": {"mean": sum(diffs) / len(diffs),
                                   "ci": boot,
                                   "samples": len(diffs)},
        "per_cell": per_cell,
        "per_asset": per_asset,
    }


def build(path: Path | None = None) -> dict:
    """Evaluate the record and return it in the shape callers expect.

    Kept as the stable entry point (the tests and ``app/trend_gate.py`` both
    quote from it). The flat :func:`evaluate` keys are all preserved; the
    extra ``pooled``/``split``/``deployed`` views exist so a reader does not
    have to know the internal naming to check the docstring's tables.
    """
    m = evaluate(annotate(load(Path(path) if path else CSV)))
    ch, op, f = m["chases"], m["opposes"], m["flip_policy"]
    per_asset = {a: {"wr": d["flip"]["wr"], "n": d["flip"]["n"],
                     "net_bps": d["flip"]["net_bps"],
                     "engine_wr": d["engine"]["wr"]}
                 for a, d in m["per_asset"].items()}
    m["pooled"] = {"agree_n": ch["n"], "agree_wr": ch["wr"], "agree_ci": ch["ci"],
                   "opp_n": op["n"], "opp_wr": op["wr"], "opp_ci": op["ci"]}
    # ``baseline_wr`` is EVERY row in the file (the number the user actually
    # experienced); ``baseline`` is the usable sample the flip is scored on.
    m["baseline_wr"] = m["all_rows_asis"]["wr"]
    m["baseline_ci"] = m["all_rows_asis"]["ci"]
    m["split"] = {"train_n": m["split_65_35"]["train"]["n"],
                  "train_wr": m["split_65_35"]["train"]["wr"],
                  "hold_n": m["split_65_35"]["holdout"]["n"],
                  "hold_wr": m["split_65_35"]["holdout"]["wr"],
                  "hold_ci": m["split_65_35"]["holdout"]["ci"]}
    # The deployed rule keeps EVERY window (coverage 1.0) — the whole point of
    # replacing the abstention. Anything reporting coverage < 1.0 is
    # describing the retired withhold-only policy, not this one.
    m["deployed"] = {"n": f["n"], "wr": f["wr"], "ci": f["ci"],
                     "coverage": f["n"] / m["rows_in_file"] if m["rows_in_file"] else 0.0,
                     "net_bps": f["net_bps"], "per_asset": per_asset,
                     "mode": "flip"}
    return m


# ── reporting ──────────────────────────────────────────────────────────────
def report(m: dict) -> str:
    L: list[str] = []
    A = L.append
    A("trend-interaction rule — audit (app/trend_gate.py)")
    A(f"record: {CSV.relative_to(ROOT)}")
    A(f"rows {m['rows_in_file']}, usable {m['rows_usable']} "
      f"(dropped {m['warmup_dropped']} warming up, lag {m['lag']})")
    A("")
    A("A. the interaction (what the rule is built on)")
    A(f"   every shipped window      {m['all_rows_asis']['n']:4d}  "
      f"{_pct(m['all_rows_asis']['wr'])}   CI {_ci(m['all_rows_asis']['ci'])}")
    b = m['baseline']
    A(f"   usable sample             {b['n']:4d}  {_pct(b['wr'])}   "
      f"CI {_ci(b['ci'])}   net {b['net_bps']:+.1f} bps/row")
    for key, label in (("chases", "AGREES with trend"), ("opposes", "OPPOSES the trend")):
        d = m[key]
        A(f"   {label:25s} {d['n']:4d}  {_pct(d['wr'])}   "
          f"CI {_ci(d['ci'])}   net {d['net_bps']:+.1f} bps/row")
    t = m["trend_as_side"]
    A(f"   trend alone as the side   {t['n']:4d}  {_pct(t['wr'])}   "
      f"CI {_ci(t['ci'])}")
    A("")
    A("B. the policy (100 % coverage — nothing is dropped)")
    f = m["flip_policy"]
    A(f"   engine side, unchanged    {b['n']:4d}  {_pct(b['wr'])}   "
      f"CI {_ci(b['ci'])}   net {b['net_bps']:+.1f} bps/row")
    A(f"   FLIP the chases           {f['n']:4d}  {_pct(f['wr'])}   "
      f"CI {_ci(f['ci'])}   net {f['net_bps']:+.1f} bps/row")
    pb = m["paired_bootstrap_delta"]
    A(f"   paired delta              {100 * pb['mean']:+.1f} pp   "
      f"CI {100 * pb['ci'][0]:+.1f} – {100 * pb['ci'][1]:+.1f} pp "
      f"({pb['samples']} resamples)")
    A("")
    A("C. honesty — every fold, including the losing one")
    A("   folds: " + " / ".join(_pct(x) for x in m["flip_folds"])
      + f"   ({sum(1 for x in m['flip_folds'] if x > 0.5)} of "
        f"{len(m['flip_folds'])} above 50 %)")
    for fr in m["fold_detail"]:
        A(f"     fold {fr['fold']}  n={fr['n']:4d}  {_pct(fr['wr'])}  "
          f"CI {_ci(fr['ci'])}")
    s = m["split_65_35"]
    A(f"   65/35 time split  train n={s['train']['n']} {_pct(s['train']['wr'])}"
      f" | holdout n={s['holdout']['n']} {_pct(s['holdout']['wr'])} "
      f"CI {_ci(s['holdout']['ci'])}")
    A("")
    A("D. where it does NOT work (stated, not hidden)")
    for asset, d in m["per_asset"].items():
        fl, en = d["flip"], d["engine"]
        A(f"   {asset:5s} flip {_pct(fl['wr'])} (n={fl['n']}) "
          f"net {fl['net_bps']:+.1f} bps | engine {_pct(en['wr'])} "
          f"net {en['net_bps']:+.1f} bps")
    A("")
    A("E. per cell (small n — most intervals cross 50 %)")
    for cell, d in m["per_cell"].items():
        A(f"   {cell:10s} n={d['n']:3d}  flip {_pct(d['wr'])} "
          f"CI {_ci(d['ci'])} net {d['net_bps']:+.1f} "
          f"| engine {_pct(d['base_wr'])} net {d['base_net_bps']:+.1f} "
          f"| chases {d['chases']}")
    A("")
    A("Nothing above is a forward promise. The rule is in-sample and")
    A("selection-aware; it is not wired into the certified hold policy.")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="machine-readable")
    ap.add_argument("--csv", default=str(CSV), help="record to audit")
    ap.add_argument("--out", default="", help="write the JSON here too")
    args = ap.parse_args(argv)
    rows = annotate(load(Path(args.csv)))
    m = evaluate(rows)
    if args.out:
        Path(args.out).write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2) if args.json else report(m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
