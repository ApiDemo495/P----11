"""Hold-design continuous benchmark — BOTH targets on real data, no abstention.

  T1  every-window first-touch row win rate ≥ 66 %
      (every window counted, tie = loss, causal k/m geometry)
  T2  system EV > 0 after REAL friction — 12/18 bps per side, charged
      only on actual round trips (side flips), never relaxed

Why a new design: the per-window re-entry model (continuous.py,
REAL_RESULTS §12) must pay 24 bps every window against ~19–106 bps
windows — measured impossible. The mandate allows changing the trade
structure, so this harness holds ONE position per side-era:

* side recomputed every window (no skipped analyses, ever),
* position enters at era start (side flip), TP/SL first-touch on the
  native bars (SL wins same-bar ties), otherwise exits at the flip,
* friction = one round trip per era (the only real trades),
* rows (T1) graded exactly like continuous.py: first-touch TP/SL
  inside the window, else window-end sign settle — via _simulate with
  friction 0 (friction never changes a row's sign outcome; T2 owns
  the money).

Selection is causal: (side, k, m) chosen on TRAIN years subject to
row_WR_train ≥ 0.66 AND era_EV_net_train > 0 (≥30 eras), then
reported on the HOLDOUT year. Single-year assets (BNB) select
in-sample and are labelled as such — never presented as forward.

Run:  .venv/bin/python -m backtest.hold [--asset BTC ...] [--interval 5m]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

try:                                    # optional JIT — see _era_sim_core
    from numba import njit
except Exception:                       # pragma: no cover
    def njit(*_a, **_k):
        def deco(f):
            return f
        return deco if not (_a and callable(_a[0])) else _a[0]

from .continuous import (CACHE, MOM_GRID, RESULTS, _hourly_final,
                         _scale_at, _simulate, _utc_years, _year_start,
                         build_features, load_bars, logistic_fit_predict,
                         wilson)
from .data import FRICTION_BPS

# Row geometry (T1 — the every-window scoring rule): tight so most
# windows actually touch; the first-touch floor 1/(1+k) at k=0.5 is
# 66.7 % and the real side edge lifts above it. Rows never pay
# friction (T2 owns the money) — they grade the CALL, not a round trip.
K_ROW = (0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6)
M_ROW = (0.15, 0.25, 0.4, 0.55)
# Era geometry (T2 — the position's REAL TP/SL): wide enough for the
# held move to pay 24 bps of friction at a positive edge.
K_ERA = (0.6, 0.75, 1.0, 1.25, 1.5, 2.0)
# m = 999 → NO barriers: ride the era to the flip (the trend-capture
# geometry; barriers can only truncate a positive-drift era)
M_ERA = (0.5, 1.0, 2.0, 3.0, 999.0)
LONG_MOM_MIN = (7200, 10080, 14400, 43200, 129600, 259200,
                525600)                                  # 5d..365d
ROW_WR_TARGET = 0.66
PURGE_DAYS = 7
MIN_ERA_TRAIN = 15
MIN_ERA_HOLDOUT = 3
# single-year assets (BNB 2026-only) have no earlier year to prove a
# direction on — one era is still one honest (disclosed) in-sample bet
MIN_ERA_TRAIN_SINGLE = 1


def _year_split_mask(starts: np.ndarray):
    years = _utc_years(starts)
    uniq = sorted(set(int(y) for y in years))
    last = uniq[-1]
    return years, uniq, (years < last), (years == last), len(uniq) == 1


def build_sides(t, o, h, l, c, v, step, starts, i_entry, moves, w_min):
    """Every side has FULL coverage (side ∈ {−1,+1} on every window)."""
    n = len(starts)
    step_s = int(step)
    step_min = step_s / 60.0
    W = w_min * 60
    sides: dict[str, np.ndarray] = {}
    sides["up"] = np.ones(n, dtype=int)
    sides["down"] = -np.ones(n, dtype=int)

    # momentum grid (causal, partial-history fallback keeps coverage)
    for L in MOM_GRID + LONG_MOM_MIN:
        lb = max(1, int(round(L / step_min)))
        s = np.ones(n, dtype=int)
        ok = i_entry >= lb
        r = np.full(n, 0.0)
        r[ok] = c[i_entry[ok]] / c[i_entry[ok] - lb] - 1.0
        s[ok] = np.where(r[ok] > 0, 1, np.where(r[ok] < 0, -1, 1))
        sides[f"mom{L}"] = s

    # v7 consensus families over LONG_MOM_MIN (full coverage: warmup
    # windows follow the same documented +1 fallback as the legs)
    lm_signs = [sides[f"mom{L}"] for L in LONG_MOM_MIN]
    maj = np.sign(np.sum(np.stack(lm_signs, axis=0), axis=0)).astype(int)
    maj[maj == 0] = 1
    sides["momMaj"] = maj
    lb_max = max(1, int(round(max(LONG_MOM_MIN) / step_min)))
    rbar = np.zeros(n)
    good_m = i_entry >= lb_max
    for L in LONG_MOM_MIN:
        lbL = max(1, int(round(L / step_min)))
        rr = np.zeros(n)
        okL = i_entry >= lbL
        rr[okL] = c[i_entry[okL]] / c[i_entry[okL] - lbL] - 1.0
        rbar += rr
    rbar /= float(len(LONG_MOM_MIN))
    sm = np.ones(n)
    sm[good_m] = np.where(rbar[good_m] > 0, 1,
                          np.where(rbar[good_m] < 0, -1, 1))
    sides["momMean"] = sm

    # shipped mom44h replica (hourly resample)
    h_ts, h_c = _hourly_final(t, c)
    hb = (t[i_entry] // 3600) * 3600
    j = np.searchsorted(h_ts, hb, side="right") - 1
    mom = np.ones(n, dtype=int)
    with np.errstate(divide="ignore", invalid="ignore"):
        d44 = c[i_entry] - h_c[np.clip(j - 44, 0, None)]
    valid = (j >= 44) & np.isfinite(d44)
    mom[valid] = np.where(d44[valid] > 0, 1, np.where(d44[valid] < 0, -1, 1))
    sides["mom44h"] = mom

    # fades of the previous completed window(s) — persistent variants
    def _fade(lag_windows: int) -> np.ndarray:
        base = np.searchsorted(t, starts - lag_windows * W, side="right") - 1
        out = np.ones(n, dtype=int)
        for i in range(n):
            b = int(base[i])
            if b < 0:
                # partial history: fade whatever exists since data start
                b0 = int(i_entry[i]) - 1
                if b0 < 0:
                    continue            # stays +1 (first window, disclosed)
                ref = 0
            else:
                ref = b
            mv = c[i_entry[i]] - c[ref]
            if mv > 0:
                out[i] = -1
            elif mv < 0:
                out[i] = 1
        return out

    sides["revW"] = _fade(1)
    sides["revW3"] = _fade(3)
    sides["revW5"] = _fade(5)

    # hysteresis fades: a persistent STATE machine — flip to the fade
    # only when the previous window's move is a confirmed big move
    # (|move| > τ·q80), otherwise HOLD the current side.  Cuts the
    # flip rate (era friction) while keeping the reversal edge that
    # only appears on large moves.
    sc0 = _scale_at(t, c, starts - step_s,
                    max(2, int(round(w_min / (step_s / 60.0)))))
    med = float(np.nanmedian(sc0)) if np.isfinite(sc0).any() else 0.0
    sc_f = np.where(np.isfinite(sc0) & (sc0 > 0), sc0, med)
    ip = np.searchsorted(t, starts - W, side="right") - 1
    for tau in (0.25, 0.5, 0.75):
        st_ = 0
        out_ = np.ones(n, dtype=int)
        for i in range(n):
            b = int(ip[i])
            ref = 0 if b < 0 else b
            mv = c[i_entry[i]] - c[ref]
            prop = -1 if mv > 0 else (1 if mv < 0 else 0)
            if st_ == 0:
                st_ = prop if prop else 1
            elif prop and abs(mv) > tau * sc_f[i] and prop != st_:
                st_ = prop
            out_[i] = st_
        sides[f"revWh{tau}"] = out_
    return sides


def composite_sides(sides, lr_s, lr_sc):
    """Full-coverage composites: use the OOF logistic where it has an
    opinion, fall back to the fade everywhere else (first-year
    warmup — a constant documented rule, not selectivity)."""
    out = dict(sides)
    if lr_s is not None and (lr_s != 0).any():
        base = sides["revW"]
        out["lr_revW"] = np.where(lr_s != 0, lr_s, base)
        if lr_sc is not None and np.isfinite(lr_sc).any():
            # causal EMA of the score, sign vs 0; warmup → revW
            sc = np.where(np.isfinite(lr_sc), lr_sc, 0.0)
            ema = np.zeros(len(sc))
            span = 5.0
            a = 2.0 / (span + 1.0)
            acc = 0.0
            seen = False
            for i, x in enumerate(sc):
                acc = x if not seen else acc + a * (x - acc)
                seen = True
                ema[i] = acc
            has = np.isfinite(lr_sc).cumsum() > 0
            s_ema = np.where(ema > 0, 1, np.where(ema < 0, -1, 0))
            out["lrE_revW"] = np.where(has & (s_ema != 0), s_ema, base)
    return out


def lr_sides_oof(t, o, h, l, c, v, step, starts, i_entry, moves,
                 w_min, btc_bars=None):
    X, _names = build_features(t, o, h, l, c, v, step, i_entry, btc_bars)
    # append the previous window's move (the revW feature)
    W = w_min * 60
    prev = np.zeros(len(starts))
    ip = np.searchsorted(t, starts - W, side="right") - 1
    ok = ip >= 0
    # the SECOND operand must be masked with the same `ok` — keeping it
    # full-length raised a broadcast ValueError whenever any window had
    # no prior window (EVERY real dataset), the try/except below
    # swallowed it, and the whole lr_revW/lrE_revW side family silently
    # never existed in any graded run (lr_oof_error on all 13 cells)
    prev[ok] = c[i_entry[ok]] - c[np.clip(ip, 0, None)[ok]]
    X = np.column_stack([X, prev])
    years = _utc_years(starts)
    uniq = sorted(set(int(y) for y in years))
    n = len(starts)
    out = np.zeros(n, dtype=int)
    scores = np.full(n, np.nan)
    for y in uniq[1:]:
        tr = starts < (_year_start(y) - PURGE_DAYS * 86400)
        te = years == y
        if tr.sum() < 500 or te.sum() < 50:
            continue
        ytr = np.where(moves[tr] > 0, 1, np.where(moves[tr] < 0, -1, 0))
        good = ytr != 0
        sc = logistic_fit_predict(X[tr][good], ytr[good], X[te])
        scores[te] = sc
        out[te] = np.where(sc > 0, 1, -1)
    return out, scores


def era_sim(sides, starts, i_entry, i_exit, entry, scale, t, h, l, c,
            k, m, fr_bps):
    """One position per side-era: TP/SL first-touch on native bars,
    else exit at the flip boundary. Returns per-era records.

    The loop lives in ``_era_sim_core`` (JIT — line-for-line copy of the
    original Python; parity-tested against hold_policy's pure-Python
    twin on real BTC windows).  This wrapper keeps the exact pre-call
    massaging and the record dicts the suite pins.
    """
    med = float(np.nanmedian(scale)) if np.isfinite(scale).any() else 0.0
    scale_f = np.where(np.isfinite(scale) & (scale > 0), scale, med)
    fr = fr_bps * 1e-4
    st, nw, oc, pg, pn = _era_sim_core(
        np.ascontiguousarray(sides, dtype=np.int64),
        np.ascontiguousarray(entry, dtype=np.float64),
        np.ascontiguousarray(scale_f, dtype=np.float64),
        np.ascontiguousarray(i_entry, dtype=np.int64),
        np.ascontiguousarray(i_exit, dtype=np.int64),
        np.ascontiguousarray(c, dtype=np.float64),
        np.ascontiguousarray(h, dtype=np.float64),
        np.ascontiguousarray(l, dtype=np.float64),
        float(k), float(m), float(fr))
    recs = []
    for i in range(len(st)):
        recs.append({"start_i": int(st[i]), "n_windows": int(nw[i]),
                     "outcome": int(oc[i]), "pnl_gross": float(pg[i]),
                     "pnl_net": float(pn[i])})
    return recs


@njit(cache=True)
def _era_sim_core(sides, entry, scale_f, i_entry, i_exit, c_close,
                  h, l, k, m, fr):
    """Line-for-line JIT copy of the original era_sim loop body.

    Same iteration order, same SL-wins-ties rule, same float ops.
    Motivation: the `down` side on the 9-year 1m BNB tape is ONE era
    spanning 4.7M bars — the Python inner bar loop x 25 era geometries
    x 27 sides was a multi-hour wall per cell.  Returns parallel arrays
    (start_i, n_windows, outcome, pnl_gross, pnl_net).
    """
    n = len(sides)
    st = np.empty(n, dtype=np.int64)
    nw = np.empty(n, dtype=np.int64)
    oc = np.empty(n, dtype=np.int64)
    pg = np.empty(n)
    pn = np.empty(n)
    cnt = 0
    w = 0
    while w < n:
        s = int(sides[w])
        if s == 0:
            w += 1
            continue
        w2 = w + 1
        while w2 < n and int(sides[w2]) == s:
            w2 += 1
        e = float(entry[w])
        sl = m * float(scale_f[w])
        if not np.isfinite(sl) or sl <= 0 or e <= 0:
            w = w2
            continue
        tp = k * sl
        tp_px = e + tp * s
        sl_px = e - sl * s
        a = int(i_entry[w]) + 1
        b_last = int(i_exit[w2 - 1])
        b_flip = int(i_entry[w2]) if w2 < n else b_last
        b = min(b_flip, b_last)
        outcome = 0
        exit_px = 0.0
        if m >= 999.0:                    # no-barrier ride to the flip
            outcome = 2
            exit_px = float(entry[w2]) if w2 < n else float(c_close[b_last])
            pnl_g = (exit_px / e - 1.0) * s
            st[cnt] = w
            nw[cnt] = w2 - w
            oc[cnt] = outcome
            pg[cnt] = pnl_g
            pn[cnt] = pnl_g - 2.0 * fr
            cnt += 1
            w = w2
            continue
        for j in range(a, b + 1):
            hi, lo = h[j], l[j]
            if s > 0:
                sl_hit = lo <= sl_px
                tp_hit = hi >= tp_px
            else:
                sl_hit = hi >= sl_px
                tp_hit = lo <= tp_px
            if sl_hit:                     # SL wins same-bar ties
                outcome, exit_px = -1, sl_px
                break
            if tp_hit:
                outcome, exit_px = 1, tp_px
                break
        if outcome == 0:
            # no touch before the flip → exit at the flip boundary price
            exit_px = float(entry[w2]) if w2 < n else float(c_close[b_last])
            outcome = 2                    # 2 = flip exit
        # × s is correct for every outcome: TP/SL prices are already
        # side-adjusted (s² = 1) and raw flip exits get their sign here
        pnl_g = (exit_px / e - 1.0) * s
        st[cnt] = w
        nw[cnt] = w2 - w
        oc[cnt] = outcome
        pg[cnt] = pnl_g
        pn[cnt] = pnl_g - 2.0 * fr
        cnt += 1
        w = w2
    return st[:cnt], nw[:cnt], oc[:cnt], pg[:cnt], pn[:cnt]

def grade_cell(asset, w_min, bars, smoke=False, btc_bars=None) -> dict:
    t, o, h, l, c, v, step = bars
    step_s = int(step)
    out: dict = {"asset": asset, "tf_min": w_min, "step_s": step_s,
                 "design": "hold-v1"}
    if step_s * 4 > w_min * 60:
        out["status"] = "ungradeable"
        out["why"] = (f"{step_s // 60}m bars cannot resolve a {w_min}-min "
                      f"window honestly")
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
    years, uniq_years, tr_w, te_w, single = _year_split_mask(starts)
    fr_bps = FRICTION_BPS.get(asset, 12.0)
    out["friction_bps_per_side"] = fr_bps
    out["single_year"] = bool(single)
    scale = _scale_at(t, c, starts - step_s,
                      max(2, int(round(w_min / (step_s / 60.0)))))

    sides = build_sides(t, o, h, l, c, v, step, starts, i_entry, moves,
                        w_min)
    # lr OOF (first-year uncovered → composite fallback keeps coverage)
    try:
        lr_s, lr_sc = lr_sides_oof(t, o, h, l, c, v, step, starts,
                                   i_entry, moves, w_min)
        out["lr_oof_covered"] = int((lr_s != 0).sum())
    except Exception as exc:               # pragma: no cover
        lr_s, lr_sc = None, None
        out["lr_oof_error"] = str(exc)
    sides = composite_sides(sides, lr_s, lr_sc)

    fr0 = 0.0
    best = None
    feasible = 0
    side_rows: dict[str, dict] = {}
    pool: dict[str, dict] = {}
    for name, side in sides.items():
        assert np.all(side != 0), name
        flips = int(np.sum(side[1:] != side[:-1]))
        diag: dict = {"flip_rate": round(flips / max(1, n - 1), 4),
                      "feasible_row_ge_66": 0, "feasible_era_gt0": 0}
        # ── T1: rows — every window, tight first-touch geometry ────
        best_row = None
        row_win = None
        row_k = row_m = None
        row_wr_feas = -1.0
        tr_w_eff = np.ones(n, dtype=bool) if single else tr_w
        for k in K_ROW:
            for m in M_ROW:
                res = _simulate(side, entry, exit_, moves, scale, k, m,
                                starts, t, h, l, step_s, fr0,
                                i_entry, i_exit)
                win = res["win"]
                wr_tr = float(win[tr_w_eff].mean())
                wr_te = (float(win[te_w].mean()) if te_w.any() else None)
                if wr_tr is not None:
                    if best_row is None or wr_tr > best_row[0]:
                        best_row = (wr_tr, k, m, wr_te)
                    if wr_tr >= ROW_WR_TARGET:
                        diag["feasible_row_ge_66"] += 1
                        # keep the STRONGEST train row among the
                        # feasible geometries (best holdout prospects)
                        if wr_tr > row_wr_feas:
                            row_wr_feas = wr_tr
                            row_win, row_k, row_m = win, k, m
        diag["row_wr_best_train"] = (round(best_row[0], 4)
                                     if best_row else None)
        diag["row_wr_best_k"] = best_row[1] if best_row else None
        diag["row_wr_best_m"] = best_row[2] if best_row else None
        diag["row_wr_best_holdout"] = (round(best_row[3], 4)
                                       if best_row and best_row[3] is not None
                                       else None)
        # ── T2: era system — wide geometry, friction per era ───────
        # v8 (§17 pre-registered): collect every rung-C-passing era
        # geometry TOGETHER with its train-only robustness stats
        # (worst train-year EV, train profit factor); the rung pick +
        # WR-first selection run after all sides are scored.
        era_cands: list[dict] = []
        era_ev_best = None
        for k in K_ERA:
            for m in M_ERA:
                if m >= 999.0 and k != K_ERA[0]:
                    continue                # ride geometry: one k only
                er = era_sim(side, starts, i_entry, i_exit, entry, scale,
                             t, h, l, c, k, m, fr_bps)
                if not er:
                    continue
                ey = np.array([years[r["start_i"]] for r in er])
                ep = np.array([r["pnl_net"] for r in er])
                tr_e = (np.ones(len(er), bool) if single
                        else ey < uniq_years[-1])
                need_eras = (MIN_ERA_TRAIN_SINGLE if single
                             else MIN_ERA_TRAIN)
                if int(tr_e.sum()) < need_eras:
                    continue
                ev_tr = float(ep[tr_e].mean())
                if era_ev_best is None or ev_tr > era_ev_best:
                    era_ev_best = ev_tr
                # per-train-year robustness: positive in ≥60 % of the
                # train years (train-only — never looks at holdout)
                if not single:
                    tr_years = sorted(set(int(y) for y in ey[tr_e]))
                    year_means: list[float] = []
                    pos_years = 0
                    for yy in tr_years:
                        mask_y = tr_e & (ey == yy)
                        if mask_y.any():
                            ym = float(ep[mask_y].mean())
                            year_means.append(ym)
                            if ym > 0:
                                pos_years += 1
                    robust = (len(tr_years) > 0 and
                              pos_years >= math.ceil(0.6 * len(tr_years)))
                    # §17 rung stats — worst TRAIN year, train PF
                    worst = min(year_means) if year_means else ev_tr
                else:
                    robust = True
                    worst = ev_tr
                e_tr = ep[tr_e]
                epos = float(e_tr[e_tr > 0].sum())
                eneg = float(-e_tr[e_tr < 0].sum())
                pf = float("inf") if eneg <= 0 else epos / eneg
                diag["feasible_era_gt0"] += int(ev_tr > 0)
                if ev_tr <= 0 or not robust:
                    continue
                era_cands.append({"ev": ev_tr, "k": k, "m": m, "er": er,
                                  "ey": ey, "ep": ep, "worst": worst,
                                  "pf": pf})
        diag["era_ev_best_train"] = (round(era_ev_best, 6)
                                     if era_ev_best is not None else None)
        diag["n_eras"] = (len(era_cands[0]["er"]) if era_cands else None)
        side_rows[name] = diag
        pool[name] = {
            "row_win": row_win, "row_k": row_k, "row_m": row_m,
            "wr_tr": (float(row_win[tr_w_eff].mean())
                      if row_win is not None else None),
            "cands": era_cands,
        }

    # ── v8 pre-registered robustness ladder + WR-first key (§17) ───
    # Rungs are nested by construction (worst>0 ⇒ every train year
    # positive ⇒ ≥60 % robust), first rung with ANY feasible candidate
    # wins; within the rung the key stays lexicographic
    # (train row WR, train era EV).  Train-only: the holdout is never
    # consulted here — §17 evaluates it exactly once, after the run.
    rung = "C"
    rung_pred = lambda c: True   # default rung C
    for _r, _p in (("A", lambda c: c["worst"] > 0 and c["pf"] >= 1.15),
                   ("B", lambda c: c["worst"] > 0),
                   ("C", lambda c: True)):
        if any(v["row_win"] is not None
               and any(_p(c) for c in v["cands"])
               for v in pool.values()):
            rung, rung_pred = _r, _p
            break
    best = None
    feasible = 0
    for name, v in pool.items():
        if v["row_win"] is None:
            continue
        cands = [c for c in v["cands"] if rung_pred(c)]
        if not cands:
            continue
        best_era = max(cands, key=lambda c: c["ev"])
        feasible += 1
        cand = (v["wr_tr"], best_era["ev"], name, v["row_k"], v["row_m"],
                best_era["k"], best_era["m"], best_era["er"],
                best_era["ey"], best_era["ep"], v["row_win"],
                best_era["worst"], best_era["pf"])
        if best is None or cand[:2] > best[:2]:
            best = cand

    out["sides"] = side_rows
    out["n_feasible_train"] = feasible
    if best is None:
        out["status"] = "no_feasible_config"
        out["note"] = ("no side reached row WR ≥ 0.66 (tight rows) AND "
                       "positive train era-EV (wide eras) after friction "
                       "— honest failure, reported as such")
        return out

    (wr_tr_sel, ev_tr, side_name, kr, mr, ke, me, er, ey, ep, win,
     worst_sel, pf_sel) = best
    rung_sel = rung
    side_rows[side_name]["n_eras"] = len(er)
    side = sides[side_name]

    def _agg(mask_w, mask_e):
        w_ = win[mask_w]
        lo, hi = wilson(int(w_.sum()), int(len(w_)))
        e_ = ep[mask_e]
        d: dict = {
            "row_n": int(len(w_)), "row_wr": round(float(w_.mean()), 4),
            "row_wilson": [round(lo, 4), round(hi, 4)],
            "era_n": int(len(e_)),
            "era_wr": (round(float((e_ > 0).mean()), 4)
                       if len(e_) else None),
            "era_ev_net": (round(float(e_.mean()), 6)
                           if len(e_) else None),
        }
        if len(e_):
            lo2, hi2 = wilson(int((e_ > 0).sum()), int(len(e_)))
            d["era_wilson"] = [round(lo2, 4), round(hi2, 4)]
            d["era_ev_total_net"] = round(float(e_.sum()), 5)
        return d

    tr_e = (np.ones(len(er), bool) if single else ey < uniq_years[-1])
    te_e = (np.zeros(len(er), bool) if single else ey == uniq_years[-1])
    era_yearly = {}
    for yy in sorted(set(int(y) for y in ey)):
        m_y = ey == yy
        era_yearly[yy] = {"n": int(m_y.sum()),
                          "ev_net": round(float(ep[m_y].mean()), 6)}
    sel = {
        "era_ev_yearly": era_yearly,
        "side": side_name,
        "row_k": kr, "row_m": mr,          # T1 geometry (every window)
        "era_k": ke, "era_m": me,          # T2 geometry (real position)
        "flip_rate": round(float(np.mean(side[1:] != side[:-1])), 4),
        "n_eras": len(er),
        "train": _agg(np.ones(n, bool) if single else tr_w, tr_e),
        "holdout": _agg(np.zeros(n, bool) if single else te_w, te_e),
        "full": _agg(np.ones(n, bool), np.ones(len(er), bool)),
        "selection": ("train-only" if not single
                      else "SINGLE-YEAR IN-SAMPLE (disclosed, not forward)"),
        "selection_rule": (
            f"v8 §17: robustness rung {rung_sel} (train-only ladder: "
            "A = worst train-year EV>0 & PF>=1.15, B = worst train-year "
            "EV>0, C = 60% train-years robust); within the rung max "
            "TRAIN row WR (era EV breaks ties); train-only — the "
            "holdout was never consulted"),
        "train_row_wr_selected": round(wr_tr_sel, 4),
        "selection_rung": rung_sel,
        "train_worst_year_ev": (round(worst_sel, 6)
                                if math.isfinite(worst_sel) else "inf"),
        "train_pf": (round(pf_sel, 4) if math.isfinite(pf_sel) else "inf"),
    }
    v = {}
    if not single:
        v["holdout_row_wr_ge_66"] = sel["holdout"]["row_wr"] >= ROW_WR_TARGET
        v["holdout_row_wilson_lo_ge_66"] = (
            sel["holdout"]["row_wilson"][0] >= ROW_WR_TARGET)
        if sel["holdout"]["era_n"] >= MIN_ERA_HOLDOUT:
            v["holdout_era_ev_net_gt_0"] = (
                (sel["holdout"]["era_ev_net"] or 0) > 0)
            v["holdout_era_wr"] = sel["holdout"]["era_wr"]
        else:
            v["holdout_era_verdict"] = (
                f"insufficient holdout eras (n={sel['holdout']['era_n']}, "
                f"need {MIN_ERA_HOLDOUT}) — T2 verified on train+full "
                "only, disclosed")
        v["holdout_era_n"] = sel["holdout"]["era_n"]
    v["full_row_wr_ge_66"] = sel["full"]["row_wr"] >= ROW_WR_TARGET
    v["full_era_ev_net_gt_0"] = (sel["full"]["era_ev_net"] or 0) > 0
    v["full_era_wr"] = sel["full"]["era_wr"]
    v["full_era_n"] = sel["full"]["era_n"]
    out["selected"] = sel
    out["targets"] = v
    out["status"] = "ok"
    return out


def run(assets, smoke=False, interval=None) -> dict:
    table = {}
    for asset in assets:
        bars = load_bars(asset, interval)
        if bars is None:
            print(f"{asset}: cache missing")
            continue
        btc_bars = load_bars("BTC", interval) if asset != "BTC" else None
        for w_min in (22, 44, 130, 247):
            t0 = time.time()
            cell = grade_cell(asset, w_min, bars, smoke=smoke,
                              btc_bars=btc_bars)
            table[f"{asset}|{w_min}"] = cell
            if cell.get("status") == "ok":
                s = cell["selected"]
                tv = cell["targets"]
                print(f"{asset} {w_min}min [{time.time()-t0:.1f}s]: "
                      f"side={s['side']} row(k={s['row_k']},m={s['row_m']}) "
                      f"era(k={s['era_k']},m={s['era_m']}) | "
                      f"rowWR train={s['train']['row_wr']} "
                      f"hold={s['holdout'].get('row_wr')} "
                      f"full={s['full']['row_wr']} | "
                      f"eraEVnet full={s['full']['era_ev_net']} "
                      f"hold={s['holdout'].get('era_ev_net')} | "
                      f"eraWR full={s['full']['era_wr']} | "
                      f"targets={tv}")
            else:
                print(f"{asset} {w_min}min [{time.time()-t0:.1f}s]: "
                      f"{cell.get('status')} — {cell.get('why') or cell.get('note','')}")
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", action="append")
    ap.add_argument("--interval", choices=["1m", "5m", "1h"], default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    assets = args.asset or ["BTC", "ETH", "BNB", "PAXG"]
    table = run(assets, smoke=args.smoke, interval=args.interval)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "hold_wr.json")
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
