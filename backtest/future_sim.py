"""FUTURE-6y SIMULATION — explicitly SIMULATED, never measured.

The user asked for a "for fun" 6-year-forward analysis.  Honesty rule
(real-data-only standing directive): everything this script prints is a
MONTE CARLO over block-bootstrap resamples of the REAL 6-year tape —
it is a null distribution ("if the future behaves like resampled
history"), NOT a measurement and NOT a forecast.  Every output key is
prefixed ``sim_`` and the JSON carries ``"label": "SIMULATED ...\"``.

Method
------
* Take the real 5m tape (BTC/ETH, 2020-11 → 2026-09).
* Build each synthetic path by sampling REAL calendar days (288 5m
  bars) with replacement, in 6-year-length sequences — this preserves
  the real return distribution, fat tails, per-day vol clustering and
  intraday shape; only cross-day ordering is randomized.
* Grade the SAME continuous predictors the live benchmark uses
  (revW = fade of the previous window, mom44h shipped rule, the mom
  grid, up/down) on the same epoch windows / strict sign rules.
* Report the distribution of out-of-sample-style WRs across N paths,
  and the share of paths where each predictor reaches ≥66% continuous
  WR — the number the user cares about.

Run:  .venv/bin/python -m backtest.future_sim --asset BTC --asset ETH
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from .continuous import (CACHE, MOM_GRID, RESULTS, _hourly_final,
                         load_bars, wilson)

LABEL = ("SIMULATED — block-bootstrap Monte Carlo over the real tape; "
         "not a measurement, not a forecast")
DAY_BARS = 288              # 5m bars per day
N_PATHS = 200               # fun-scale, cheap enough to be honest
W_GRID = (22, 44, 130, 247)


def _day_blocks(c: np.ndarray, t: np.ndarray) -> list[np.ndarray]:
    day = (t // 86400).astype(np.int64)
    edges = np.flatnonzero(day[1:] != day[:-1]) + 1
    return list(np.split(c, edges))


def _synth_path(blocks: list[np.ndarray], n_days: int,
                rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(blocks), size=n_days)
    return np.concatenate([blocks[i] for i in idx])


def _sides_for(c: np.ndarray, t: np.ndarray, w_min: int, step_s: int):
    """(starts, i_entry, i_exit, side dicts, moves, entry) for one path."""
    W = w_min * 60
    t0 = int(math.ceil(int(t[0]) / W) * W)
    t1 = int(t[-1]) + step_s
    starts = np.arange(t0, t1 - W + 1, W, dtype=np.int64)
    if len(starts) < 200:
        return None
    i_entry = np.searchsorted(t, starts - step_s, side="right") - 1
    i_exit = np.searchsorted(t, starts + W - step_s, side="right") - 1
    ok = (i_entry >= 0) & (i_exit > i_entry) & (i_exit < len(t))
    starts, i_entry, i_exit = starts[ok], i_entry[ok], i_exit[ok]
    entry = c[i_entry]
    exit_ = c[i_exit]
    moves = exit_ - entry
    n = len(starts)
    sides = {"up": np.ones(n, dtype=int), "down": -np.ones(n, dtype=int)}

    # revW — fade of the previous completed window (live rule)
    iprev = np.searchsorted(t, starts - W, side="right") - 1
    prev_ok = iprev >= 0
    pm = np.zeros(n)
    pm[prev_ok] = entry[prev_ok] - c[iprev[prev_ok]]
    rev = np.zeros(n, dtype=int)
    rev[prev_ok] = np.where(pm[prev_ok] > 0, -1,
                            np.where(pm[prev_ok] < 0, 1, 0))
    sides["revW"] = rev

    # shipped mom44h — hourly resample, 44h lag
    h_ts, h_c = _hourly_final(t, c)
    hb = (t[i_entry] // 3600) * 3600
    j = np.searchsorted(h_ts, hb, side="right") - 1
    mom = np.ones(n, dtype=int)
    valid = j >= 44
    with np.errstate(divide="ignore", invalid="ignore"):
        d = entry - h_c[np.clip(j - 44, 0, None)]
    mom[valid] = np.where(d[valid] > 0, 1, np.where(d[valid] < 0, -1, 1))
    sides["mom44h"] = mom

    # momentum grid on the path itself
    for L in MOM_GRID:
        secs = int(L * 60)
        idx = np.searchsorted(t, starts - secs, side="right") - 1
        m = np.ones(n, dtype=int)
        good = idx >= 0
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = entry - c[np.clip(idx, 0, None)]
        m[good] = np.where(dd[good] > 0, 1, np.where(dd[good] < 0, -1, 1))
        sides[f"mom{L}"] = m
    return starts, entry, moves, sides


def run(asset: str, paths: int = N_PATHS, seed: int = 495) -> dict:
    bars = load_bars(asset, "5m")
    if bars is None:
        raise SystemExit(f"cache missing for {asset}")
    t, o, h, l, c, v, step_s = bars
    step_s = int(step_s)
    blocks = _day_blocks(c, t)
    n_days = int((int(t[-1]) - int(t[0])) // 86400) + 1
    rng = np.random.default_rng(seed)

    # collect per-(tf, predictor): list of WRs across paths
    agg: dict[str, dict[str, list]] = {}
    n_windows: dict[str, int] = {}
    for w_min in W_GRID:
        if step_s * 4 > w_min * 60:
            continue
        for p in range(paths):
            pc = _synth_path(blocks, n_days, rng)
            # synthetic epoch time on the same calendar grid as the real
            # data's start (5m grid) — same window alignment as benchmark
            pt = t[0] + np.arange(len(pc), dtype=np.int64) * step_s
            got = _sides_for(pc, pt, w_min, step_s)
            if got is None:
                continue
            starts, entry, moves, sides = got
            if p == 0:
                n_windows[w_min] = len(starts)
            for name, side in sides.items():
                wr = float(np.mean(side * moves > 0))   # tie=loss ✓
                agg.setdefault(f"{w_min}", {}).setdefault(name, []).append(wr)

    out: dict = {
        "label": LABEL,
        "method": (f"{paths}-path day-block bootstrap of the real 5m tape "
                   f"({n_days} real days), same epoch windows & strict-sign "
                   "scoring as backtest/continuous.py"),
        "asset": asset,
        "sim_paths": paths,
        "sim_seed": seed,
        "sim_cells": {},
    }
    for w_min, per in agg.items():
        cell = {"n_windows_per_path": n_windows.get(int(w_min))}
        for name, wrs in sorted(per.items()):
            a = np.asarray(wrs)
            lo, hi = wilson(int(round(a.mean() * 1000)), 1000)  # shape only
            cell[f"sim_wr_{name}"] = {
                "median": round(float(np.median(a)), 4),
                "p05": round(float(np.quantile(a, 0.05)), 4),
                "p95": round(float(np.quantile(a, 0.95)), 4),
                "share_ge_66": round(float(np.mean(a >= 0.66)), 4),
            }
        out["sim_cells"][w_min] = cell
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", action="append")
    ap.add_argument("--paths", type=int, default=N_PATHS)
    args = ap.parse_args()
    assets = args.asset or ["BTC", "ETH"]
    os.makedirs(RESULTS, exist_ok=True)
    doc = {"label": LABEL, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                         time.gmtime()),
           "assets": {}}
    for asset in assets:
        print(f"[SIMULATED] {asset}: {args.paths} bootstrap paths …")
        r = run(asset, paths=args.paths)
        doc["assets"][asset] = r
        for w_min, cell in sorted(r["sim_cells"].items(), key=lambda x: int(x[0])):
            bits = []
            for k_, v in cell.items():
                if not k_.startswith("sim_wr_"):
                    continue
                bits.append(f"{k_[7:]} med={v['median']:.3f} "
                            f"[{v['p05']:.3f},{v['p95']:.3f}] "
                            f"P(≥66%)={v['share_ge_66']:.2f}")
            print(f"  tf{w_min}min n/win={cell.get('n_windows_per_path')}")
            for b in bits:
                print(f"    {b}")
    out = os.path.join(RESULTS, "future_sim.json")
    with open(out + ".tmp", "w") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(out + ".tmp", out)
    print(f"[SIMULATED] → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
