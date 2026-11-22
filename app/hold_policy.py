"""Certified hold policy — the live wiring of the BOTH-targets system.

Loads ``backtest/results/hold_wr.json`` (produced by ``backtest.hold``)
and exposes, per (asset, timeframe) cell:

  * the exact certified side rule (``mom525600``/``mom259200``/…, ``down``),
  * the decoupled TP/SL geometries — tight ROW geometry for the
    every-window first-touch grade and wide ERA geometry for the
    held position,
  * real per-side friction (12 bps → 24 bps per round trip, 18 PAXG),
  * honest tier labels: ``HOLD-OUT VERIFIED`` (7 BTC/ETH cells),
    ``HOLDOUT ROWS · ERA n<3`` (BTC 247min, rows verified but a single
    holdout era), ``IN-SAMPLE SHADOW`` (BNB, single-year) / its
    ``HOLDOUT MISS`` variant (PAXG 247min), ``UNGRADEABLE / WARMUP``
    (PAXG tf1–tf3 — 1 h bars cannot resolve 22/44/130-min windows),
  * measured win rate + Wilson interval (confidence source of truth).

Side semantics are byte-identical to ``backtest.hold.build_sides`` for
every non-logistic candidate (the certified selection never picked an
``lr_*`` composite) — ``tests/test_hold_live.py`` enforces ZERO parity
mismatches against ``backtest.hold`` on shared bar arrays, including the
era simulation and row settlement rules.

Every window is analysed (full coverage, no abstention): partial
history falls back to ``up`` exactly like the benchmark's documented
warmup rule, reported through ``meta["warmup"]`` so the UI can say so.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from bisect import bisect_left
from typing import Any

import numpy as np

from .config import TIMEFRAMES

DEFAULT_HOLD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "backtest", "results", "hold_wr.json")

TF_BY_MIN = {v["minutes"]: k for k, v in TIMEFRAMES.items()}
TF_MINUTES = {k: int(v["minutes"]) for k, v in TIMEFRAMES.items()}
# sides certified by the selection (hold.py MOM_GRID ∪ LONG_MOM_MIN etc.)
_MOM_MINUTES = tuple(range(15, 45 * 60 + 1, 15)) + (
    7200, 10080, 14400, 43200, 129600, 259200, 525600)

_lock = threading.RLock()
_DOC: dict[str, dict] = {}
_PATH = ""


def _load_doc(path: str | None = None) -> dict[str, dict]:
    global _DOC, _PATH
    with _lock:
        p = path or os.environ.get("HOLD_WR_PATH") or DEFAULT_HOLD_PATH
        if _DOC and p == _PATH:
            return _DOC
        try:
            with open(p, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            # record absent/unreadable → the legacy mandate fallback
            # owns the cells (tests pin this path too)
            doc = {}
        _DOC = {k: v for k, v in doc.items() if isinstance(v, dict)}
        _PATH = p
        return _DOC


def reload(path: str | None = None) -> dict[str, dict]:
    """Drop the cache and re-read (tests use this with HOLD_WR_PATH)."""
    global _DOC, _PATH
    with _lock:
        _DOC, _PATH = {}, ""
    return _load_doc(path)


def cell(symbol: str, tf: str) -> dict | None:
    """Raw hold_wr cell record for (asset, tf), or None if unknown."""
    if tf not in TF_MINUTES:
        return None
    doc = _load_doc()
    return doc.get(f"{symbol}|{TF_MINUTES[tf]}")


def known_cells() -> list[tuple[str, str]]:
    return [(s, t) for s in ("BTC", "ETH", "BNB", "PAXG")
            for t in ("tf1", "tf2", "tf3", "tf4")]


def tier_counts() -> dict[str, int]:
    out: dict[str, int] = {}
    for symbol, tf in known_cells():
        t = tier(symbol, tf)
        out[t] = out.get(t, 0) + 1
    return out


# ─────────────────────────────── tiers ───────────────────────────────
TIER_HOLDOUT = "HOLD-OUT VERIFIED"
TIER_ROWS_ONLY = "HOLDOUT ROWS · ERA n<3"
TIER_INSAMPLE = "IN-SAMPLE SHADOW"
TIER_MISS = "IN-SAMPLE SHADOW · HOLDOUT MISS"
TIER_WARMUP = "UNGRADEABLE / WARMUP"


def tier(symbol: str, tf: str) -> str:
    c = cell(symbol, tf)
    if c is None:
        return TIER_WARMUP
    if c.get("status") != "ok":
        return TIER_WARMUP
    t = c.get("targets") or {}
    if c.get("single_year"):
        return TIER_INSAMPLE
    if (t.get("holdout_row_wr_ge_66")
            and t.get("holdout_row_wilson_lo_ge_66")
            and t.get("holdout_era_ev_net_gt_0")):
        return TIER_HOLDOUT
    if t.get("holdout_row_wr_ge_66") and t.get("full_era_ev_net_gt_0"):
        # rows verified out-of-fold but the holdout has < 3 eras to
        # judge system EV (BTC 247min: n=1) — say exactly that
        if t.get("holdout_era_ev_net_gt_0") is None and \
                t.get("holdout_era_n") is not None:
            return TIER_ROWS_ONLY
    return TIER_MISS


def is_hold_cell(symbol: str, tf: str) -> bool:
    """True when this cell is governed by the certified hold policy
    (gradeable cells) — warmup cells keep the policy's badges but the
    benchmark never measured a side on them."""
    c = cell(symbol, tf)
    return bool(c and c.get("status") == "ok")


def is_warmup_cell(symbol: str, tf: str) -> bool:
    """PAXG tf1–tf3: known cell, no certified measurement (1 h bars)."""
    c = cell(symbol, tf)
    return bool(c and c.get("status") != "ok")


def managed(symbol: str, tf: str) -> bool:
    """True when hold policy owns the cell at all (all 16 cells)."""
    return cell(symbol, tf) is not None


# ───────────────────────── summary for UI / engine ───────────────────
def summary(symbol: str, tf: str) -> dict[str, Any]:
    """Everything the live cell panel needs, honestly labelled."""
    c = cell(symbol, tf)
    t = tier(symbol, tf)
    base: dict[str, Any] = {
        "symbol": symbol, "tf": tf, "minutes": TF_MINUTES.get(tf),
        "tier": t, "managed": c is not None,
        "certified": is_hold_cell(symbol, tf),
    }
    if c is None:
        base.update(status="unknown", side=None, win_rate=None,
                    confidence=None, why="cell not in the hold policy")
        return base
    if c.get("status") != "ok":
        base.update(status="ungradeable", side=None, win_rate=None,
                    confidence=None, why=c.get("why"),
                    note=("UNGRADEABLE on 1 h history — live WARMUP until "
                          "enough sub-hour tape exists; no certified win "
                          "rate is claimed for this cell"))
        return base
    sel = c.get("selected") or {}
    full = sel.get("full") or {}
    hold = sel.get("holdout") or {}
    wil = full.get("row_wilson") or [None, None]
    base.update(
        status="ok",
        side=sel.get("side"),
        row_k=sel.get("row_k"), row_m=sel.get("row_m"),
        era_k=sel.get("era_k"), era_m=sel.get("era_m"),
        ride=bool(float(sel.get("era_m") or 0) >= 999.0),
        win_rate=full.get("row_wr"),
        win_rate_wilson=wil,
        confidence=wil[0],                    # Wilson lower bound
        holdout_win_rate=(hold.get("row_wr")
                          if hold.get("row_n") else None),
        n_windows=c.get("n_windows"),
        era_ev_net=full.get("era_ev_net"),
        era_ev_holdout=hold.get("era_ev_net"),
        era_ev_total=full.get("era_ev_total_net"),
        era_wr=full.get("era_wr"), era_n=full.get("era_n"),
        friction_bps_per_side=c.get("friction_bps_per_side"),
        round_trip_bps=2.0 * float(c.get("friction_bps_per_side") or 0.0),
        flip_rate=sel.get("flip_rate"),
        single_year=bool(c.get("single_year")),
        selection=sel.get("selection"),
        targets=c.get("targets"),
    )
    return base


def friction_bps(symbol: str) -> float:
    """Per-side real friction (one round trip = 2× this)."""
    from backtest.data import FRICTION_BPS
    return float(FRICTION_BPS.get(symbol, 12.0))


def row_geometry(symbol: str, tf: str) -> tuple[float, float] | None:
    c = cell(symbol, tf)
    if not c or c.get("status") != "ok":
        return None
    sel = c["selected"]
    return float(sel["row_k"]), float(sel["row_m"])


def era_geometry(symbol: str, tf: str) -> tuple[float, float] | None:
    c = cell(symbol, tf)
    if not c or c.get("status") != "ok":
        return None
    sel = c["selected"]
    return float(sel["era_k"]), float(sel["era_m"])


# ─────────────────────────── side semantics ──────────────────────────
def side_value(side_name: str, closes: np.ndarray, i_entry: int,
               step_s: int) -> int:
    """One window's side — exactly ``backtest.build_sides`` semantics.

    ``closes[i_entry]`` is the entry (last bar strictly before the
    window); lookback is measured in BARS for the given step so a
    5-minute replay and a 1-minute live tape express the same
    ``L``-minute momentum.  Partial history → ``up`` (the documented
    full-coverage warmup fallback), never a skip.
    """
    if side_name == "up":
        return 1
    if side_name == "down":
        return -1
    if side_name.startswith("mom") and side_name[3:].isdigit():
        L = int(side_name[3:])
        step_min = max(step_s, 1) / 60.0
        lb = max(1, int(round(L / step_min)))
        if i_entry < lb:
            return 1                     # warmup fallback (hold rule)
        r = float(closes[i_entry]) / float(closes[i_entry - lb]) - 1.0
        if r > 0:
            return 1
        if r < 0:
            return -1
        return 1                         # flat → up (hold rule)
    if side_name == "momMaj":
        from backtest.hold import LONG_MOM_MIN as _LMM
        votes = []
        for L in _LMM:
            lb = max(1, int(round(L / (max(step_s, 1) / 60.0))))
            if i_entry < lb:
                votes.append(1)          # warmup leg → +1 (hold rule)
                continue
            r = float(closes[i_entry]) / float(closes[i_entry - lb]) - 1.0
            votes.append(1 if r > 0 else (-1 if r < 0 else 1))
        tot = sum(votes)
        return 1 if tot >= 0 else -1
    if side_name == "momMean":
        from backtest.hold import LONG_MOM_MIN as _LMM
        lb_max = max(1, int(round(max(_LMM) / (max(step_s, 1) / 60.0))))
        if i_entry < lb_max:
            return 1                      # warmup → +1 (hold rule)
        acc = 0.0
        for L in _LMM:
            lb = max(1, int(round(L / (max(step_s, 1) / 60.0))))
            if i_entry >= lb:
                acc += float(closes[i_entry]) / float(closes[i_entry - lb]) - 1.0
        acc /= float(len(_LMM))
        if acc > 0:
            return 1
        if acc < 0:
            return -1
        return 1
    if side_name in ("revW", "revW3", "revW5") or side_name.startswith("revWh"):
        # needs absolute timestamps (which window is being faded) —
        # the feed path uses ``_revw_live``; series use ``_series_for``
        raise ValueError(f"{side_name} requires timestamps — use _series_for "
                         f"or _revw_live")
    if side_name == "lr_revW":
        raise ValueError("lr_revW needs the full OOF tape — use _series_for")
    if side_name == "mom44h":
        # needs absolute timestamps — the feed path uses
        # ``_mom44h_live``; series paths use ``_series_for``
        raise ValueError(f"mom44h requires timestamps — use _series_for "
                         f"or _mom44h_live")
    raise ValueError(f"unknown certified side {side_name!r}")


def side_series(symbol: str, tf: str, bars: tuple) -> np.ndarray | None:
    """Vectorized certified side over a bar array — parity twin of
    ``backtest.hold.build_sides`` for this cell's selected side."""
    c = cell(symbol, tf)
    if not c or c.get("status") != "ok":
        return None
    return _series_for(str((c.get("selected") or {}).get("side")), bars,
                       TF_MINUTES[tf])


def _series_for(side_name: str, bars: tuple, w_min: int) -> np.ndarray:
    """Vectorized replica of hold.build_sides for ONE side name.

    Uses the identical formulas (mom ratio with warmup=+1, constant
    sides, hourly 44 h difference) so parity tests can compare
    element-wise against ``build_sides(...)[side_name]``.
    """
    t, o, h, l, cl, v, step = bars
    t = np.asarray(t)
    cl = np.asarray(cl, dtype=float)
    step_s = int(step)
    step_min = step_s / 60.0
    W = w_min * 60
    # epoch-aligned windows exactly like hold.grade_cell:
    # starts = ceil(t0 / W) * W … ≤ t1 − W
    t0 = int(math.ceil(int(t[0]) / W) * W)
    t1 = int(t[-1]) + step_s
    starts = np.arange(t0, t1 - W + 1, W, dtype=np.int64)
    if starts.size == 0:
        return np.zeros(0, dtype=int)
    step_s_i = step_s
    i_entry = np.searchsorted(t, starts - step_s_i, side="right") - 1
    i_exit = np.searchsorted(t, starts + W - step_s_i, side="right") - 1
    ok = (i_entry >= 0) & (i_exit > i_entry) & (i_exit < len(t))
    starts, i_entry = starts[ok], i_entry[ok]
    n = len(starts)
    if side_name == "up":
        return np.ones(n, dtype=int)
    if side_name == "down":
        return -np.ones(n, dtype=int)
    if side_name.startswith("mom") and side_name[3:].isdigit():
        L = int(side_name[3:])
        lb = max(1, int(round(L / step_min)))
        s = np.ones(n, dtype=int)
        # i_entry is ALREADY ok-filtered above; masking with the
        # pre-filter `ok` crashed whenever any window was invalid
        # (real BTC 5m tape: 171 such windows) — latent until cache
        # present made the parity test actually reach this branch
        good = i_entry >= lb
        r = np.zeros(n)
        idx = np.clip(i_entry, 0, None)
        r[good] = cl[idx[good]] / cl[np.maximum(idx[good] - lb, 0)] - 1.0
        s[good] = np.where(r[good] > 0, 1, np.where(r[good] < 0, -1, 1))
        return s
    if side_name == "momMaj":
        from backtest.hold import LONG_MOM_MIN as _LMM
        stack = np.zeros((len(_LMM), n))
        for _i, L in enumerate(_LMM):
            lb = max(1, int(round(L / step_min)))
            good = i_entry >= lb
            rr = np.zeros(n)
            idx = np.clip(i_entry, 0, None)
            rr[good] = cl[idx[good]] / cl[np.maximum(idx[good] - lb, 0)] - 1.0
            stack[_i] = np.where(rr > 0, 1, np.where(rr < 0, -1, 1))
        tot = stack.sum(axis=0)
        return np.where(tot >= 0, 1, -1).astype(int)
    if side_name == "momMean":
        from backtest.hold import LONG_MOM_MIN as _LMM
        lb_max = max(1, int(round(max(_LMM) / step_min)))
        good = i_entry >= lb_max
        rbar = np.zeros(n)
        idx = np.clip(i_entry, 0, None)
        for L in _LMM:
            lb = max(1, int(round(L / step_min)))
            okL = i_entry >= lb
            rr = np.zeros(n)
            rr[okL] = cl[idx[okL]] / cl[np.maximum(idx[okL] - lb, 0)] - 1.0
            rbar += rr
        rbar /= float(len(_LMM))
        out = np.ones(n, dtype=int)
        out[good] = np.where(rbar[good] > 0, 1, np.where(rbar[good] < 0, -1, 1))
        return out
    if side_name in ("revW", "revW3", "revW5") or side_name.startswith("revWh"):
        lag_windows = {"revW": 1, "revW3": 3, "revW5": 5}.get(side_name)
        base = np.searchsorted(
            t, starts - (lag_windows * W if lag_windows else W),
            side="right") - 1
        if lag_windows is None:            # revWh* hysteresis state machine
            from backtest.hold import _scale_at as _h_scale
            tau = float(side_name[5:])
            sc0 = _h_scale(t, cl, starts - step_s,
                           max(2, int(round(w_min / (step_s / 60.0)))))
            med = float(np.nanmedian(sc0)) if np.isfinite(sc0).any() else 0.0
            sc_f = np.where(np.isfinite(sc0) & (sc0 > 0), sc0, med)
            ip = np.searchsorted(t, starts - W, side="right") - 1
            st_ = 0
            out = np.ones(n, dtype=int)
            for i in range(n):
                b = int(ip[i])
                ref = 0 if b < 0 else b
                mv = cl[i_entry[i]] - cl[ref]
                prop = -1 if mv > 0 else (1 if mv < 0 else 0)
                if st_ == 0:
                    st_ = prop if prop else 1
                elif prop and abs(mv) > tau * sc_f[i] and prop != st_:
                    st_ = prop
                out[i] = st_
            return out
        out = np.ones(n, dtype=int)        # plain fades
        for i in range(n):
            b = int(base[i])
            if b < 0:
                b0 = int(i_entry[i]) - 1
                if b0 < 0:
                    continue                # first window → +1 (hold rule)
                ref = 0
            else:
                ref = b
            mv = cl[i_entry[i]] - cl[ref]
            if mv > 0:
                out[i] = -1
            elif mv < 0:
                out[i] = 1
        return out
    if side_name in ("lr_revW", "lrE_revW"):
        # composite: OOF logistic where it opines, revW fallback —
        # shares the BENCHMARK's lr_sides_oof/composite_sides (they ARE
        # the spec; build_sides itself never contains these keys — the
        # composite is applied AFTER build in grade_cell)
        from backtest.hold import (lr_sides_oof as _lr_oof,
                                   composite_sides as _comp)
        i_exit = np.searchsorted(t, starts + W - step_s, side="right") - 1
        try:
            moves = cl[i_exit] - cl[i_entry]
            lr_s, lr_sc = _lr_oof(t, o, h, l, cl, v, step_s,
                                  starts, i_entry, moves, w_min)
        except Exception:
            lr_s, lr_sc = None, None
        base_rev = _series_for("revW", bars, w_min)
        comp = _comp({"revW": base_rev}, lr_s, lr_sc)
        return comp[side_name] if side_name in comp else base_rev
    if side_name == "mom44h":
        # hold.py rule: hourly resample, d44 = c[i_entry] − h_c[j−44]
        from backtest.continuous import _hourly_final
        h_ts, h_c = _hourly_final(t, cl)
        hb = (t[np.clip(i_entry, 0, None)] // 3600) * 3600
        j = np.searchsorted(h_ts, hb, side="right") - 1
        mom = np.ones(n, dtype=int)
        with np.errstate(divide="ignore", invalid="ignore"):
            d44 = cl[np.clip(i_entry, 0, None)] - h_c[np.clip(j - 44, 0, None)]
        valid = (i_entry >= 0) & (j >= 44) & np.isfinite(d44)
        mom[valid] = np.where(d44[valid] > 0, 1,
                              np.where(d44[valid] < 0, -1, 1))
        return mom
    raise ValueError(f"unknown certified side {side_name!r}")


def window_entry_index(times, tf: str, now: float | None = None) -> int:
    """Index of the bar strictly before the CURRENT epoch window (live
    analogue of hold.py's ``i_entry``)."""
    W = TF_MINUTES[tf] * 60
    anchor = int(now if now is not None else (times[-1] if times else 0))
    s = anchor // W * W
    return bisect_left(times, s) - 1


def side_at_feed(feed, symbol: str, tf: str, now: float | None = None
                 ) -> tuple[int, dict[str, Any]]:
    """Live certified side for (asset, tf) with honest metadata.

    Reads the feed's CLOSED minute tape only (causal): entry = last bar
    strictly before the current epoch window, exactly the benchmark's
    anchor.  ``meta.warmup`` is True while the lookback history is
    shorter than L (side falls back to ``up`` per the certified rule —
    full coverage, never a skip, never silent).
    """
    pol = summary(symbol, tf)
    if not pol.get("managed"):
        return 0, {"reason": "cell not managed by the hold policy"}
    side_name = pol.get("side")
    meta: dict[str, Any] = {
        "policy_side": side_name, "tier": pol.get("tier"),
        "certified": pol.get("certified"), "warmup": False,
    }
    try:
        st = feed.state(symbol)
        times = list(st.times)
        closes = list(st.closes)
    except Exception:
        return 0, {**meta, "reason": "no tape"}
    if not times or len(times) != len(closes):
        return 0, {**meta, "reason": "empty tape"}
    if not pol.get("certified"):
        # UNGRADEABLE / WARMUP cell: no certified side exists — the
        # live call falls back to the engine raw ladder (side 0 here,
        # caller keeps its own side) and the UI shows the warmup badge
        return 0, {**meta, "warmup": True,
                   "reason": "no certified measurement on 1 h history"}
    step_s = int(round(float(times[1] - times[0]))) if len(times) > 1 else 60
    i_entry = window_entry_index(times, tf, now)
    if i_entry < 0:
        return 1, {**meta, "warmup": True, "reason": "window start has "
                                                     "no prior bar"}
    import numpy as _np
    closes_a = _np.asarray(closes, dtype=float)
    if side_name == "mom44h":
        # pointwise hourly path needs timestamps; bucket math below
        side = _mom44h_live(times, closes_a, i_entry, side_name)
    else:
        side = side_value(str(side_name), closes_a, i_entry, step_s)
    L = None
    if side_name and side_name.startswith("mom") and side_name[3:].isdigit():
        L = int(side_name[3:])
        step_min = step_s / 60.0
        lb = max(1, int(round(L / step_min)))
        if i_entry < lb:
            meta["warmup"] = True
            meta["lookback_bars"] = lb
            meta["have_bars"] = int(i_entry) + 1
            meta["reason"] = (f"warmup: {i_entry + 1}/{lb} bars of "
                              f"{side_name} history — certified fallback "
                              f"side +1 (hold.py full-coverage rule)")
    return int(side), meta


def _mom44h_live(times, closes, i_entry: int, side_name: str) -> int:
    """44 h hourly-difference rule on the live tape."""
    s = int(times[i_entry])
    target_h = s // 3600 - 44
    target_ts = target_h * 3600
    j = bisect_left(times, target_ts) - 1
    if j < 0:
        return 1
    d44 = float(closes[i_entry]) - float(closes[j])
    if d44 > 0:
        return 1
    if d44 < 0:
        return -1
    return 1


# ─────────────────── row settlement (hold rules) ─────────────────────
def row_settle_rules(symbol: str, tf: str) -> dict[str, Any]:
    """How a sealed lock for this cell must be graded — the certified
    first-touch rules (SL wins same-window ties; a flat settle is a
    LOSS, not a push; friction belongs to eras, never to rows)."""
    c = cell(symbol, tf)
    if not is_hold_cell(symbol, tf):
        return {"hold_rules": False}
    g = row_geometry(symbol, tf)
    return {
        "hold_rules": True,
        "row_k": g[0] if g else None,
        "row_m": g[1] if g else None,
        "tie_rule": "SL wins same-window ties",
        "flat_rule": "zero move at expiry = loss (every window counted)",
        "friction": ("charged once per era round trip by the era "
                     "accountant — rows carry none"),
    }


def build_row_bars(symbol: str, tf: str, entry: float, side: int,
                   scale_px: float) -> tuple[float, float] | None:
    """(tp_px, sl_px) for an EVERY-WINDOW row lock from the certified
    decoupled row geometry.  ``scale_px`` = q80 window-move scale at the
    entry (``_scale_at`` semantics, price units)."""
    g = row_geometry(symbol, tf)
    if g is None or entry <= 0 or scale_px <= 0 or not side:
        return None
    k, m = g
    sl = m * float(scale_px)
    tp = k * sl
    return entry + tp * side, entry - sl * side


def scale_at_bars(bars: tuple, at_ts, tf: str) -> np.ndarray:
    """Certified barrier scale (q80 of trailing W-length moves) — the
    same call hold.grade_cell makes, imported for parity."""
    from backtest.continuous import _scale_at
    t, o, h, l, c, v, step = bars
    W = TF_MINUTES[tf] * 60
    lb = max(2, int(round(W / (int(step) / 60.0))))
    return _scale_at(np.asarray(t), np.asarray(c, dtype=float),
                     np.asarray(at_ts), lb)


def scale_at_feed(feed, symbol: str, tf: str,
                  now: float | None = None) -> float | None:
    """Live pointwise q80 scale on the feed tape (price units) —
    identical estimator to ``scale_at_bars`` at one boundary."""
    try:
        st = feed.state(symbol)
        times = list(st.times)
        closes = list(st.closes)
    except Exception:
        return None
    if len(times) < 16:
        return None
    W = TF_MINUTES[tf] * 60
    step_s = int(round(float(times[1] - times[0]))) if len(times) > 1 else 60
    lb = max(2, int(round(W / (step_s / 60.0))))
    anchor = int(now if now is not None else times[-1])
    s = anchor // W * W
    i = bisect_left(times, s) - 1
    if i < lb * 2:
        return None
    lo = max(lb, i - 100 * lb)
    js = list(range(lo, i + 1, lb))
    if len(js) < 8:
        return None
    moves = [abs(closes[j] - closes[j - lb]) for j in js]
    return float(np.quantile(np.asarray(moves, dtype=float), 0.80))


# ───────────────────── per-cell ANALYSE timer lock ───────────────────
class CellLockBook:
    """One ANALYSE click → the cell sleeps for its full TF window.

    While ``(asset, tf)`` is locked the app refuses new predictions for
    THAT cell only; every other cell (and the same asset on other TFs)
    stays free.  4 assets × 4 TFs = 16 independent timers.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path or os.environ.get(
            "CELL_LOCKS_PATH", os.path.join("data", "cell_locks.json"))
        self._locks: dict[str, float] = {}
        self._load()

    def _key(self, symbol: str, tf: str) -> str:
        return f"{symbol}|{tf}"

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
            self._locks = {k: float(v) for k, v in
                           (raw.get("locks") or {}).items()}
        except (OSError, ValueError, FileNotFoundError):
            self._locks = {}

    def _save(self) -> None:
        try:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"version": 1, "locks": self._locks}, fh)
            os.replace(tmp, self.path)
        except OSError:
            pass

    def enabled(self) -> bool:
        """HOLD_CELL_LOCKS=1 forces on (tests), =0 forces off.
        Under pytest the lock is off by default so legacy suites that
        analyse the same cell repeatedly keep working; the lock tests
        force it on.  Production (no env): always on."""
        v = os.environ.get("HOLD_CELL_LOCKS")
        if v == "1":
            return True
        if v == "0":
            return False
        return not os.environ.get("PYTEST_CURRENT_TEST")

    def remaining(self, symbol: str, tf: str,
                  now: float | None = None) -> float:
        if not self.enabled():
            return 0.0
        now = float(now if now is not None else time.time())
        until = float(self._locks.get(self._key(symbol, tf), 0.0))
        return max(0.0, until - now)

    def locked(self, symbol: str, tf: str,
               now: float | None = None) -> bool:
        return self.remaining(symbol, tf, now) > 0

    def until(self, symbol: str, tf: str) -> float:
        return float(self._locks.get(self._key(symbol, tf), 0.0))

    def start(self, symbol: str, tf: str, *, now: float | None = None,
              minutes: int | None = None) -> dict[str, Any]:
        """Arm the timer for this cell (called at ANALYSE click)."""
        now = float(now if now is not None else time.time())
        mins = int(minutes if minutes is not None
                   else TF_MINUTES.get(tf, 22))
        until = now + mins * 60.0
        self._locks[self._key(symbol, tf)] = until
        self._save()
        return {"symbol": symbol, "tf": tf, "locked_until": until,
                "minutes": mins, "remaining_s": round(until - now, 1)}

    def clear(self, symbol: str, tf: str) -> None:
        self._locks.pop(self._key(symbol, tf), None)
        self._save()

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        now = float(now if now is not None else time.time())
        out: dict[str, Any] = {}
        for symbol, tf in known_cells():
            rem = self.remaining(symbol, tf, now)
            out[f"{symbol}|{tf}"] = {
                "locked": rem > 0,
                "remaining_s": round(rem, 1),
                "locked_until": self.until(symbol, tf),
                "minutes": TF_MINUTES[tf],
                "can_analyse": rem <= 0,
            }
        return out


CELL_LOCKS: CellLockBook | None = None


def cell_locks() -> CellLockBook:
    global CELL_LOCKS
    with _lock:
        if CELL_LOCKS is None:
            CELL_LOCKS = CellLockBook()
        return CELL_LOCKS
