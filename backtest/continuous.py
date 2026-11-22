"""Continuous-cycle win-rate benchmark — every window, no abstention.

Answers the directive: *"66% win rate for each tf for continuous cycle.
Don't be selective."* — measured on REAL bars only, never simulated.

Grading (identical to the live ledger's rules):

* Windows are the app's epoch-aligned grid: W ∈ {22, 44, 130, 247}
  minutes, ``start = floor(t / W) · W`` (engine._grid_window).
* EVERY complete window is graded — a predictor must emit a side for
  each one; skips do not exist in this harness.
* **sign WR** — win iff ``sign(exit − entry) == side`` (tie = loss),
  the ledger's ``_row_result`` strict-sign rule.
* **trade WR / EV** — continuous position with causal barriers:
  SL = m·ATR(W) from bars BEFORE the window, TP = k·SL; first touch
  settles (same-bar both-touch → SL wins, mandate rule); no touch →
  close at window end. Friction charged both ways. Geometry (k, m)
  selected per cell on TRAIN years only, reported on holdout + full.

Data resolution (reachable mirrors only — documented, never faked):

* ``{SYM}_1m.csv`` preferred → exact boundaries.
* else ``{SYM}_5m.csv`` → boundary price = last bar close at/before the
  boundary (age ≤ 4 min). Validated against 1m on the 2026 overlap
  (``approx_validation``) before any 5m-derived number is quoted.
* ``{SYM}_1h.csv`` alone → sub-hour windows are NOT gradeable
  (status ``ungradeable``) — no fabrication.

Predictors (all causal — features from bars strictly before start):

* ``up`` / ``down``   drift baselines
* ``mom{L}``          fixed-lookback momentum sign
* ``mom44h``          the shipped mandate rule (offline replica)
* ``revW``            fade the completed previous window
* ``mom_best``        best fixed rule picked on TRAIN years
* ``lr_oof``          logistic ensemble, purged yearly walk-forward,
                       reported strictly out-of-fold

Usage:
    .venv/bin/python -m backtest.continuous                 # all cells
    .venv/bin/python -m backtest.continuous --asset BTC --smoke
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import time

import numpy as np

try:                                    # optional JIT — same loops, ~50x
    from numba import njit              # faster; pure-Python fallback if
except Exception:                       # numba is unavailable
    def njit(*_a, **_k):                # pragma: no cover
        def deco(f):
            return f
        return deco if not (_a and callable(_a[0])) else _a[0]

CACHE = os.path.join(os.path.dirname(__file__), "cache")
RESULTS = os.path.join(os.path.dirname(__file__), "results")

TF_W = {"tf1": 22, "tf2": 44, "tf3": 130, "tf4": 247}   # minutes
W_MIN = list(TF_W.values())
FRICTION_BPS = {"BTC": 12.0, "ETH": 12.0, "BNB": 12.0, "PAXG": 18.0}

RET_L = (15, 60, 240, 720, 1440, 2640, 4320)           # feature lookbacks
MOM_GRID = (15, 60, 240, 720, 1440, 2640, 4320)        # momentum candidates
K_GRID = (0.6, 0.75, 1.0, 1.25, 1.5)                   # TP = k·SL
M_GRID = (0.5, 0.75, 1.0)                              # SL = m·q80|W-move|
TARGET_WR = 0.66
PURGE_DAYS = 7


# ───────────────────────────── data loading ──────────────────────────
def load_bars(asset: str, want: str | None = None) -> tuple | None:
    """(t, o, h, l, c, v, step_s) — finest reachable resolution
    (or ``want`` when a specific interval is requested)."""
    order = (want,) if want else ("1m", "5m", "1h")
    for interval in order:
        step = {"1m": 60, "5m": 300, "1h": 3600}[interval]
        path = os.path.join(CACHE, f"{asset}_{interval}.csv")
        if not os.path.exists(path):
            continue
        # np.genfromtxt peaks at ~1 KB/row transient (object-dtype
        # phase) — on the 4.7M-row parshant BNB tape that is a >4 GB
        # spike and the kernel OOM-kills the run (EXIT=137, no output).
        # loadtxt parses straight into float64: same shape, ~10x less.
        rows = np.loadtxt(path, delimiter=",", skiprows=1,
                          usecols=(0, 1, 2, 3, 4, 5))
        if rows.ndim != 2 or rows.size == 0:
            continue
        rows = rows[np.argsort(rows[:, 0], kind="stable")]
        keep = np.ones(len(rows), dtype=bool)
        keep[1:] = rows[1:, 0] > rows[:-1, 0]
        rows = rows[keep]
        t = rows[:, 0].astype(np.int64)
        return (t, rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4],
                rows[:, 5], step)
    return None


def _load_interval(asset: str, interval: str) -> tuple | None:
    step = {"1m": 60, "5m": 300, "1h": 3600}[interval]
    path = os.path.join(CACHE, f"{asset}_{interval}.csv")
    if not os.path.exists(path):
        return None
    rows = np.loadtxt(path, delimiter=",", skiprows=1,
                      usecols=(0, 1, 2, 3, 4, 5))
    if rows.ndim != 2 or rows.size == 0:
        return None
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    keep = np.ones(len(rows), dtype=bool)
    keep[1:] = rows[1:, 0] > rows[:-1, 0]
    rows = rows[keep]
    t = rows[:, 0].astype(np.int64)
    return (t, rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4],
            rows[:, 5], step)


def _hourly_final(t: np.ndarray, c: np.ndarray):
    """(hour_open_ts, final_close_of_that_hour) over the full series."""
    hour = (t // 3600) * 3600
    keep = np.ones(len(t), dtype=bool)
    keep[:-1] = hour[1:] != hour[:-1]
    return hour[keep], c[keep]


def _year_start(y: int) -> int:
    return int(dt.datetime(y, 1, 1, tzinfo=dt.timezone.utc).timestamp())


def _utc_years(starts: np.ndarray) -> np.ndarray:
    return (starts.astype("datetime64[s]")
            .astype("datetime64[Y]").astype(np.int32) + 1970)


# ───────────────────────────── metrics ───────────────────────────────
def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - half) / d), min(1.0, (centre + half) / d)


def sign_metrics(sides: np.ndarray, moves: np.ndarray) -> dict:
    """strict-sign: tie (move == 0) = loss."""
    n = int(len(sides))
    if n == 0:
        return {"n": 0, "wr": None}
    correct = int(np.sum(((moves > 0) & (sides > 0)) |
                         ((moves < 0) & (sides < 0))))
    ties = int(np.sum(moves == 0))
    lo, hi = wilson(correct, n)
    return {"n": n, "wins": correct, "ties": ties,
            "wr": round(correct / n, 4), "wilson": [round(lo, 4), round(hi, 4)]}


# ───────────────────────────── features ──────────────────────────────
def build_features(t: np.ndarray, o: np.ndarray, h: np.ndarray,
                   l: np.ndarray, c: np.ndarray, v: np.ndarray,
                   step: int, idx: np.ndarray,
                   btc_bars: tuple | None) -> tuple[np.ndarray, list[str]]:
    """Causal feature matrix at bar indices ``idx`` (last bar < start)."""
    n = len(idx)
    step_min = step / 60.0
    logret = np.zeros(len(c))
    logret[1:] = np.log(c[1:] / c[:-1])
    feats: list[np.ndarray] = []
    names: list[str] = []

    for L in RET_L:
        lb = max(1, int(round(L / step_min)))
        r = np.full(n, np.nan)
        ok = idx >= lb
        r[ok] = c[idx[ok]] / c[idx[ok] - lb] - 1.0
        feats.append(r)
        names.append(f"ret{L}")

    for short, long_, tag in ((60, 1440, "volr1h24h"), (15, 720, "volr15m12h")):
        s_b = max(2, int(round(short / step_min)))
        l_b = max(3, int(round(long_ / step_min)))
        vs = np.full(n, np.nan)
        vl = np.full(n, np.nan)
        ok = idx >= l_b
        sel = idx[ok]
        # vectorised windows via stride tricks
        vs[ok] = [np.std(logret[i - s_b + 1:i + 1]) for i in sel]
        vl[ok] = [np.std(logret[i - l_b + 1:i + 1]) for i in sel]
        with np.errstate(divide="ignore", invalid="ignore"):
            feats.append(np.where(vl > 0, vs / vl, np.nan))
        names.append(tag)

    rb = max(2, int(round(1440 / step_min)))
    rp = np.full(n, np.nan)
    ok = idx >= rb
    sel = idx[ok]
    hh = np.array([h[i - rb + 1:i + 1].max() for i in sel])
    ll = np.array([l[i - rb + 1:i + 1].min() for i in sel])
    rng = hh - ll
    rp[ok] = np.where(rng > 0, (c[sel] - ll) / rng, 0.5)
    feats.append(rp)
    names.append("range24")

    vb = max(2, int(round(1440 / step_min)))
    rv = np.full(n, np.nan)
    ok = idx >= vb
    sel = idx[ok]
    mv = np.array([v[i - vb + 1:i + 1].mean() for i in sel])
    with np.errstate(divide="ignore", invalid="ignore"):
        rv[ok] = np.where(mv > 0, v[sel] / mv, np.nan)
    feats.append(rv)
    names.append("relvol24")

    hours = ((t[idx] % 86400) // 3600).astype(float)
    feats.append(np.sin(2 * math.pi * hours / 24))
    feats.append(np.cos(2 * math.pi * hours / 24))
    names += ["tod_sin", "tod_cos"]

    if btc_bars is not None:
        bt, _bo, _bh, _bl, bc, _bv, bstep = btc_bars
        lead = np.full(n, np.nan)
        ts = t[idx]
        j_now = np.searchsorted(bt, ts, side="right") - 1
        j_prev = np.searchsorted(bt, ts - 3600, side="right") - 1
        valid = (j_now >= 0) & (j_prev >= 0) & (j_prev < j_now)
        with np.errstate(divide="ignore", invalid="ignore"):
            lead[valid] = bc[j_now[valid]] / bc[j_prev[valid]] - 1.0
        feats.append(lead)
        names.append("btc_ret60")

    X = np.column_stack(feats)
    return X, names


# ───────────────────────────── logistic ──────────────────────────────
def logistic_fit_predict(Xtr: np.ndarray, ytr: np.ndarray,
                         Xte: np.ndarray, iters: int = 300,
                         lr: float = 0.5, l2: float = 1e-4) -> np.ndarray:
    mu = np.nanmean(Xtr, axis=0)
    sd = np.nanstd(Xtr, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    mu = np.where(np.isfinite(mu), mu, 0.0)

    def prep(X):
        Z = (X - mu) / sd
        Z = np.where(np.isfinite(Z), Z, 0.0)
        return np.hstack([np.ones((len(Z), 1)), Z])

    A, B = prep(Xtr), prep(Xte)
    y = (ytr > 0).astype(float)
    w = np.zeros(A.shape[1])
    for _ in range(iters):
        z = A @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        w -= lr * (A.T @ (p - y) / len(y) + l2 * w)
    return B @ w


# ───────────────────────────── barriers ──────────────────────────────
def _scale_at(t: np.ndarray, c: np.ndarray, at_ts: np.ndarray,
              lb: int, samples: int = 100, q: float = 0.80) -> np.ndarray:
    """Causal barrier scale at each boundary: the q-quantile of
    |W-length moves| over the trailing ``samples`` spans (all end at or
    before the boundary). Native-bar ATR would measure a 5-minute
    horizon — the barrier has to live on the WINDOW's horizon."""
    n = len(at_ts)
    out = np.full(n, np.nan)
    idx = np.searchsorted(t, at_ts, side="right") - 1
    for j, i in enumerate(idx):
        if i < lb * 2:
            continue
        lo = max(lb, i - samples * lb)
        js = np.arange(lo, i + 1, lb)
        if len(js) < 8:
            continue
        moves = np.abs(c[js] - c[js - lb])
        out[j] = float(np.quantile(moves, q))
    return out


@njit(cache=True)
def _simulate_core(sides, entry, moves, sl, tp, fr, i_entry, i_exit, h, l):
    """The exact _simulate double loop, JIT-compiled.

    The pure-Python original ran 4.7M windows x ~22 bars x 756
    (k, m) x side combinations per BNB 1m cell — measured at 6-20
    hours per cell on the 9-year parshant tape (the kernel OOM and
    the multi-hour wall times were both this loop).  The logic below
    is a line-for-line copy: same order, same SL-wins-ties rule, same
    float ops — bitwise-identical output (verified against the Python
    reference on real BTC/BNB windows).
    """
    n = len(sides)
    wins = np.zeros(n, dtype=np.bool_)
    pnl = np.zeros(n)
    for i in range(n):
        a, b = int(i_entry[i]) + 1, int(i_exit[i])
        side = int(sides[i])
        e = float(entry[i])
        if b <= a or side == 0 or not np.isfinite(sl[i]):
            ret = moves[i] / e * side if e else 0.0
            wins[i] = ret > 0
            pnl[i] = ret - 2 * fr
            continue
        tp_px = e + tp[i] * side
        sl_px = e - sl[i] * side
        outcome = 0          # 0 = neither, +1 tp, -1 sl
        for j in range(a, b + 1):
            hi, lo = h[j], l[j]
            if side > 0:
                sl_hit = lo <= sl_px
                tp_hit = hi >= tp_px
            else:
                sl_hit = hi >= sl_px
                tp_hit = lo <= tp_px
            if sl_hit:       # SL wins same-bar ties (mandate rule)
                outcome = -1
                break
            if tp_hit:
                outcome = 1
                break
        if outcome < 0:
            wins[i] = False
            pnl[i] = -sl[i] / e - 2 * fr
        elif outcome > 0:
            wins[i] = True
            pnl[i] = tp[i] / e - 2 * fr
        else:
            ret = moves[i] / e * side
            wins[i] = ret > 0
            pnl[i] = ret - 2 * fr
    return wins, pnl


def _simulate(sides: np.ndarray, entry: np.ndarray, exit_: np.ndarray,
              moves: np.ndarray, atr: np.ndarray, k: float, m: float,
              starts: np.ndarray, t: np.ndarray, h: np.ndarray,
              l: np.ndarray, step_s: int, fr_bps: float,
              i_entry: np.ndarray, i_exit: np.ndarray) -> dict:
    """Continuous barrier trade: SL = m·ATR, TP = k·SL (price units)."""
    n = len(starts)
    med = float(np.nanmedian(atr))
    atr_f = np.where(np.isfinite(atr) & (atr > 0), atr, med)
    sl = m * atr_f
    tp = k * sl
    fr = fr_bps * 1e-4
    wins, pnl = _simulate_core(np.ascontiguousarray(sides, dtype=np.int64),
                               np.ascontiguousarray(entry, dtype=np.float64),
                               np.ascontiguousarray(moves, dtype=np.float64),
                               np.ascontiguousarray(sl, dtype=np.float64),
                               np.ascontiguousarray(tp, dtype=np.float64),
                               float(fr),
                               np.ascontiguousarray(i_entry, dtype=np.int64),
                               np.ascontiguousarray(i_exit, dtype=np.int64),
                               np.ascontiguousarray(h, dtype=np.float64),
                               np.ascontiguousarray(l, dtype=np.float64))
    return {"win": wins, "pnl": pnl}


# ───────────────────────────── cell grading ──────────────────────────
def grade_cell(asset: str, w_min: int, bars: tuple, *,
               smoke: bool = False, btc_bars: tuple | None = None) -> dict:
    t, o, h, l, c, v, step = bars
    step_s = int(step)
    out: dict = {"asset": asset, "tf_min": w_min, "step_s": step_s}
    if step_s * 4 > w_min * 60:
        out["status"] = "ungradeable"
        out["why"] = (f"{step_s // 60}m bars cannot resolve a {w_min}-min "
                      f"window honestly (boundary age up to "
                      f"{step_s // 60 - 1} min)")
        return out

    W = w_min * 60
    t0 = int(math.ceil(int(t[0]) / W) * W)
    t1 = int(t[-1]) + step_s
    starts = np.arange(t0, t1 - W + 1, W, dtype=np.int64)
    if smoke and len(starts) > 600:
        starts = starts[:: len(starts) // 600][:600]

    i_entry = np.searchsorted(t, starts - step_s, side="right") - 1
    i_exit = np.searchsorted(t, starts + W - step_s, side="right") - 1
    ok = (i_entry >= 0) & (i_exit > i_entry) & (i_exit < len(t))
    starts, i_entry, i_exit = starts[ok], i_entry[ok], i_exit[ok]
    n = len(starts)
    if n < 50:
        out["status"] = "too_few_windows"
        out["n"] = int(n)
        return out

    entry = c[i_entry]
    exit_ = c[i_exit]
    moves = exit_ - entry
    out["n_windows"] = int(n)
    out["span"] = [str(int(starts[0])), str(int(starts[-1]))]
    out["boundary_age_max_s"] = max(0, step_s - 60)

    # ── predictors ──────────────────────────────────────────────────
    sides: dict[str, np.ndarray] = {}
    sides["up"] = np.ones(n, dtype=int)
    sides["down"] = -np.ones(n, dtype=int)

    # shipped mom44 rule: sign(hourly close now − hourly close 44h ago)
    h_ts, h_c = _hourly_final(t, c)
    hb = (t[i_entry] // 3600) * 3600
    j = np.searchsorted(h_ts, hb, side="right") - 1
    mom = np.ones(n, dtype=int)
    okj = (j >= 44) & (np.arange(n) >= 0)
    # causal close now = entry close (inside current hour bucket)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = entry - h_c[np.clip(j - 44, 0, None)]
    valid = okj & (j - 44 >= 0)
    mom[valid] = np.where(d[valid] > 0, 1, np.where(d[valid] < 0, -1, 1))
    sides["mom44h"] = mom

    step_min = step / 60.0
    for L in MOM_GRID:
        lb = max(1, int(round(L / step_min)))
        s = np.ones(n, dtype=int)
        okm = i_entry >= lb
        r = np.full(n, 0.0)
        r[okm] = c[i_entry[okm]] / c[i_entry[okm] - lb] - 1.0
        s[okm] = np.where(r[okm] > 0, 1, np.where(r[okm] < 0, -1, 1))
        sides[f"mom{L}"] = s

    lbw = max(1, int(round(w_min / step_min)))
    s = np.ones(n, dtype=int)
    okr = i_entry >= lbw
    r_prev = np.full(n, 0.0)
    r_prev[okr] = c[i_entry[okr]] / c[i_entry[okr] - lbw] - 1.0
    s[okr] = np.where(r_prev[okr] > 0, -1, 1)
    sides["revW"] = s

    # ── features + purged yearly logistic OOF ───────────────────────
    X, feat_names = build_features(t, o, h, l, c, v, step, i_entry,
                                   btc_bars)
    X = np.column_stack([X, r_prev])
    feat_names = feat_names + ["prevW"]

    year_arr = _utc_years(starts)
    uniq_years = sorted(set(int(y) for y in year_arr))
    test_years = uniq_years[1:]

    lr_sides = np.zeros(n, dtype=int)
    fold_info = []
    for y in test_years:
        tr = starts < (_year_start(y) - PURGE_DAYS * 86400)
        te = year_arr == y
        if tr.sum() < 500 or te.sum() < 50:
            continue
        ytr = np.where(moves[tr] > 0, 1, np.where(moves[tr] < 0, -1, 0))
        good = ytr != 0
        score = logistic_fit_predict(X[tr][good], ytr[good], X[te])
        lr_sides[te] = np.where(score > 0, 1, -1)
        fold_info.append({"year": y, "train": int(tr.sum()),
                          "test": int(te.sum())})
    covered = lr_sides != 0
    out["lr_folds"] = fold_info
    out["lr_oof_n"] = int(covered.sum())
    out["features"] = feat_names

    # ── sign metrics ────────────────────────────────────────────────
    sign_tbl = {}
    for name, s in sides.items():
        m = sign_metrics(s, moves)
        m["yearly"] = {
            int(y): sign_metrics(s[year_arr == y], moves[year_arr == y])["wr"]
            for y in uniq_years if int((year_arr == y).sum()) >= 50}
        sign_tbl[name] = m
    if covered.any():
        m = sign_metrics(lr_sides[covered], moves[covered])
        m["yearly"] = {
            int(y): sign_metrics(lr_sides[(year_arr == y) & covered],
                                 moves[(year_arr == y) & covered])["wr"]
            for y in uniq_years
            if int(((year_arr == y) & covered).sum()) >= 50}
        m["oof"] = True
        sign_tbl["lr_oof"] = m
    out["sign"] = sign_tbl

    # ── train-selected best fixed rule → holdout ────────────────────
    if len(uniq_years) >= 2:
        last_y = uniq_years[-1]
        tr_mask = year_arr < last_y
        te_mask = year_arr == last_y
        best_name, best_wr = None, -1.0
        for name in [f"mom{L}" for L in MOM_GRID] + ["mom44h", "revW"]:
            wr = sign_metrics(sides[name][tr_mask], moves[tr_mask])["wr"]
            if wr is not None and wr > best_wr:
                best_wr, best_name = wr, name
        if best_name:
            out["mom_best_train"] = {"rule": best_name,
                                     "train_wr": round(best_wr, 4)}
            out["mom_best_holdout"] = sign_metrics(
                sides[best_name][te_mask], moves[te_mask])

    # ── continuous trade simulation (causal barriers) ───────────────
    fr_bps = FRICTION_BPS.get(asset, 12.0)
    atr = _scale_at(t, c, starts - step_s,
                    max(2, int(round(w_min / step_min))))
    trade_tbl = {}
    pred_for_trade = {"mom44h": sides["mom44h"], "up": sides["up"],
                      "down": sides["down"]}
    if "mom_best_train" in out:
        pred_for_trade["mom_best"] = sides[out["mom_best_train"]["rule"]]
    if covered.any():
        pred_for_trade["lr_oof"] = lr_sides

    for sname, s in pred_for_trade.items():
        mask = np.ones(n, dtype=bool) if sname != "lr_oof" else covered
        if mask.sum() < 100:
            continue
        sub = {k_: arr[mask] for k_, arr in
               dict(starts=starts, entry=entry, exit_=exit_, moves=moves,
                    atr=atr, i_entry=i_entry, i_exit=i_exit).items()}
        ysub = year_arr[mask]
        ssub = s[mask]
        last_y = uniq_years[-1]
        tr = ysub < last_y
        te = ysub == last_y
        single_year = not tr.any()
        if single_year:
            # e.g. BNB 2026-only: no earlier year to train on — grade
            # on all of it and SAY SO (holdout empty, not invented)
            tr = np.ones(len(ysub), dtype=bool)
            te = np.zeros(len(ysub), dtype=bool)
        best = None
        for k in K_GRID:
            for m in M_GRID:
                res = _simulate(ssub, sub["entry"], sub["exit_"],
                                sub["moves"], sub["atr"], k, m,
                                sub["starts"], t, h, l, step_s, fr_bps,
                                sub["i_entry"], sub["i_exit"])
                if not tr.any():
                    continue
                ev_tr = float(np.mean(res["pnl"][tr]))
                if ev_tr <= 0:
                    continue
                wr_tr = float(np.mean(res["win"][tr]))
                cand = (round(ev_tr, 6), round(wr_tr, 6), -k, -m)
                # prefer EV first; tie-break higher WR, then smaller k/m
                if best is None or cand > best[0]:
                    best = (cand, k, m, res)
        if best is None:
            # transparency: the best GROSS geometry (no friction) so the
            # reader sees how far friction is from closing the gap
            gross_best, gk, gm = None, None, None
            for k in K_GRID:
                for m in M_GRID:
                    res = _simulate(ssub, sub["entry"], sub["exit_"],
                                    sub["moves"], sub["atr"], k, m,
                                    sub["starts"], t, h, l, step_s,
                                    fr_bps, sub["i_entry"], sub["i_exit"])
                    if not tr.any():
                        continue
                    g = float(np.mean(res["pnl"][tr])) + 2 * fr_bps * 1e-4
                    if gross_best is None or g > gross_best:
                        gross_best, gk, gm = g, k, m
            trade_tbl[sname] = {
                "status": "no_EV>0_geometry_on_train",
                "gross_best_train_ev": (None if gross_best is None
                                        else round(gross_best, 6)),
                "gross_best_k": gk, "gross_best_m": gm,
                "net_friction_per_window_bps": round(2 * fr_bps, 1),
                "note": "no (k, m) produced positive train EV after "
                        "friction — honest failure, reported as such"}
            continue
        _, k, m, res = best
        def _agg(mask2):
            if not mask2.any():
                return {}
            w_ = res["win"][mask2]
            pnl = res["pnl"][mask2]
            lo, hi = wilson(int(w_.sum()), int(len(w_)))
            return {"n": int(len(w_)), "wr": round(float(w_.mean()), 4),
                    "wilson": [round(lo, 4), round(hi, 4)],
                    "ev": round(float(pnl.mean()), 5)}
        trade_tbl[sname] = {"k": k, "m_atr": m,
                            "train": _agg(tr), "holdout": _agg(te),
                            "full": _agg(np.ones(len(ssub), dtype=bool)),
                            "single_year_data": bool(single_year)}
    out["trade"] = trade_tbl
    out["status"] = "ok"
    return out


# ────────────────── approximation validation (1m vs coarse) ──────────
def approx_validation(asset: str, w_min: int) -> dict | None:
    """Same windows graded on exact 1m boundaries vs the coarse bars
    the long history uses — the honest error bar of the approximation."""
    fine = _load_interval(asset, "1m")
    coarse = _load_interval(asset, "5m")
    if fine is None or coarse is None:
        return None

    def moves_from(bars):
        t, _o, _h, _l, c, _v, step = bars
        step_s = int(step)
        W = w_min * 60
        t0 = max(int(t[0]), _year_start(2026))
        t0 = int(math.ceil(t0 / W) * W)        # re-align to the grid AFTER
        t1 = min(int(t[-1]) + step_s, _year_start(2027))
        t1 = int(t1 // W * W)
        if t1 - t0 < W * 50:
            return None, None
        starts = np.arange(t0, t1 - W + 1, W, dtype=np.int64)
        ie = np.searchsorted(t, starts - step_s, side="right") - 1
        ix = np.searchsorted(t, starts + W - step_s, side="right") - 1
        ok = (ie >= 0) & (ix > ie) & (ix < len(t))
        return starts[ok], c[ix[ok]] - c[ie[ok]]

    s1, m1 = moves_from(fine)
    if s1 is None:
        return None
    s5, m5 = moves_from(coarse)
    if s5 is None:
        return None
    common, i1, i5 = np.intersect1d(s1, s5, return_indices=True)
    if len(common) < 50:
        return None
    a, b = m1[i1], m5[i5]
    sign_flip = float(np.mean(np.sign(a) != np.sign(b)))
    wr_delta = float(abs(np.mean(np.sign(a) > 0) - np.mean(np.sign(b) > 0)))
    return {"asset": asset, "tf_min": w_min, "n": int(len(common)),
            "sign_flip_rate": round(sign_flip, 4),
            "up_rate_delta": round(wr_delta, 4)}


# ───────────────────────────── CLI ───────────────────────────────────
def run(assets: list[str], smoke: bool = False,
        interval: str | None = None) -> dict:
    btc_bars = load_bars("BTC", interval)
    table: dict = {}
    for asset in assets:
        bars = load_bars(asset, interval)
        if bars is None:
            print(f"{asset}: no cached bars — fetch first", flush=True)
            continue
        for w in W_MIN:
            t0 = time.time()
            cell = grade_cell(asset, w, bars, smoke=smoke,
                              btc_bars=btc_bars)
            cell["grade_s"] = round(time.time() - t0, 1)
            table[f"{asset}|{w}"] = cell
            if cell.get("status") != "ok":
                print(f"{asset} {w}min: {cell.get('status')} — "
                      f"{cell.get('why', '')}", flush=True)
                continue
            sg = cell["sign"]
            top = sorted(sg.items(),
                         key=lambda kv: -(kv[1]["wr"] or 0))[:6]
            row = "  ".join(f"{k}={v['wr'] * 100:.1f}%(n={v['n']})"
                            for k, v in top)
            tr = cell.get("trade", {})
            tr_bits = []
            for pn in ("mom44h", "lr_oof", "mom_best"):
                info = tr.get(pn)
                if info and "holdout" in info and info["holdout"]:
                    tr_bits.append(
                        f"{pn}: WR {info['holdout']['wr'] * 100:.1f}% "
                        f"EV {info['holdout']['ev']:+.4f} (k={info['k']})")
            print(f"{asset} {w}min n={cell['n_windows']} "
                  f"[{cell['grade_s']}s]: {row}", flush=True)
            for bit in tr_bits:
                print(f"    trade {bit}", flush=True)
    return table


def main() -> int:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", action="append", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--interval", default=None, choices=["1m", "5m", "1h"],
                    help="force a specific resolution (default: finest)")
    ap.add_argument("--approx", action="store_true",
                    help="validate 5m-vs-1m boundary approximation")
    args = ap.parse_args()
    if args.approx:
        for a in (args.asset or ["BTC", "ETH"]):
            for w in W_MIN:
                r = approx_validation(a, w)
                print(f"approx {a} {w}min: {r}")
        return 0
    assets = args.asset or ["BTC", "ETH", "BNB", "PAXG"]
    table = run(assets, smoke=args.smoke, interval=args.interval)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "continuous_wr.json")
    # flock-merged write so per-asset runs can go in parallel
    import fcntl
    with open(out + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            merged = {}
            if os.path.exists(out):
                try:
                    merged = json.load(open(out))
                except Exception:
                    merged = {}
            merged.update(table)
            with open(out + ".tmp", "w") as fh:
                json.dump(merged, fh, indent=1)
            os.replace(out + ".tmp", out)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
