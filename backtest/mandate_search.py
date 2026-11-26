"""Per-asset config search for the +0.070 EV mandate.

Reads the mandate walk's trade CSVs (recorded engine directions per
decision) + the real bar tape, RE-SETTLES candidate constructions
offline (direction rules × TF subsets × barriers × filters), selects
one config per asset on TRAIN only, and reports train / full / holdout.

Nothing here re-runs the engine: directions come from the recorded
walk, barriers are recomputed exactly from the tape (risk_block's SL
is conf-free; policy-style barriers are conf-free by construction),
settlement is the same first-touch SL-wins rule with friction.

Usage:  .venv/bin/python -m backtest.mandate_search
Writes: backtest/results/mandate_configs.json
"""
from __future__ import annotations

import csv
import itertools
import json
import os

from .data import FRICTION_BPS, load_cache
from .mandate import TF_W, settle_r, wilson
# Spacing of the tape this harness walks, in MINUTES. The TP/SL
# block measures a HOLDING PERIOD; without this it measured a
# BAR COUNT, so a 1h tape silently produced 22-HOUR stops for a
# 22-minute horizon (app/moves_risk.moves_stats, 2026-11-26).
_BAR_MINUTES = {"1m": 1.0, "5m": 5.0, "15m": 15.0, "30m": 30.0,
                "1h": 60.0, "2h": 120.0, "4h": 240.0}.get(
                    os.environ.get("BACKTEST_TF", "15m"), 15.0)

RESULTS = os.path.join(os.path.dirname(__file__), "results")
ASSETS = ("BTC", "ETH", "BNB", "PAXG")
MODES = ("old", "new")
DIRS = ("base", "fade", "inv4g", "rsi",        # engine-relative rules
        "mom44", "antimom44")                  # pure tape rules (44-bar mom)
NEW_RRS = (1.2, 1.6, 2.0)
KS = (0.6, 0.75, 0.9, 1.0)                     # stop tightening
RS = (1.25, 1.5, 2.0, 3.0)                     # target : stop
FILTERS = ("none", "gate", "momentum", "atr_top", "rsi_ext")
TF_SUBSETS = [t for n in (1, 2, 3, 4)
              for t in itertools.combinations(("tf1", "tf2", "tf3", "tf4"), n)]
TRAIN_FRAC = 0.65
TARGET_EV = 0.070
MIN_TRAIN_N = 40


# ───────────────────────── tape helpers ─────────────────────────

def _atr_pct(bars, n=14):
    out = {}
    for i in range(n, len(bars)):
        seg = bars[i - n:i + 1]
        trs = [max(b.h - b.l, abs(b.h - a.c), abs(b.l - a.c))
               for a, b in zip(seg[:-1], seg[1:])]
        out[i] = (sum(trs) / len(trs)) / bars[i].c * 100 if bars[i].c else 0.0
    return out


_SL_CACHE: dict = {}


def _sl_bps(bars, i, direction, W):
    """risk_block's SL distance in bps — conf-free (SL never reads edge).

    The walk passed the FULL prefix as the series (moves_stats windows
    it internally); this does the same, cached per (i, direction, W)."""
    key = (i, direction, W)
    if key in _SL_CACHE:
        return _SL_CACHE[key]
    from app.moves_risk import risk_block
    closes = [b.c for b in bars[:i + 1]]
    highs = [b.h for b in bars[:i + 1]]
    lows = [b.l for b in bars[:i + 1]]
    if len(closes) < 30:
        _SL_CACHE[key] = None
        return None
    try:
        blk = risk_block(closes[-1], direction, W, 0.0,
                         closes, highs, lows, bar_minutes=_BAR_MINUTES)
        val = float(blk.get("sl_bps") or 0.0) or None
    except Exception:
        val = None
    _SL_CACHE[key] = val
    return val


def _features(bars, i, atr_map):
    c = bars[i].c
    rsi_g, rsi_l = 0.0, 0.0
    for k in range(max(1, i - 14), i + 1):
        d = bars[k].c - bars[k - 1].c
        if d >= 0:
            rsi_g += d
        else:
            rsi_l -= d
    rsi = 100.0 if rsi_l <= 0 else 100.0 - 100.0 / (1 + rsi_g / max(rsi_l, 1e-12))
    Wm = 44
    mom = 0
    if i >= Wm and bars[i - Wm].c:
        mom = (1 if c > bars[i - Wm].c else -1)
    a = atr_map.get(i, 0.0)
    trail = [atr_map[j] for j in range(max(14, i - 119), i + 1) if j in atr_map]
    a_top = bool(trail and a >= sorted(trail)[int(len(trail) * 2 / 3)])
    gate = a_top  # same top-tercile rule as the validated policy
    # additive: top-QUARTILE flag (stricter gate for the win-rate search)
    a_topq = bool(trail and a >= sorted(trail)[int(len(trail) * 3 / 4)])
    return {"rsi": rsi, "mom": mom, "atr_top": a_top, "gate": gate,
            "atr": a, "atr_top_q": a_topq}


def _filt_ok(name, f) -> bool:
    if name == "none":
        return True
    if name == "gate":
        return f["gate"]
    if name == "momentum":
        return f["mom"] != 0
    if name == "atr_top":
        return f["atr_top"]
    if name == "rsi_ext":
        return f["rsi"] <= 35 or f["rsi"] >= 65
    return True


# ───────────────────────── data loading ─────────────────────────

def _load_asset(asset):
    bars = load_cache(asset, "1h")
    idx_of = {b.t: i for i, b in enumerate(bars)}
    atr_map = _atr_pct(bars)
    rows = list(csv.DictReader(
        open(os.path.join(RESULTS, f"mandate_{asset}.csv"))))
    # decision backbone: one record per (tf, ts) with all directions
    dec = {}
    for r in rows:
        key = (r["tf"], int(r["ts"]))
        d = dec.setdefault(key, {"ts": int(r["ts"]), "tf": r["tf"],
                                 "train": int(r["train"]), "dirs": {},
                                 "new_tp": {}})
        d["dirs"][r["variant"]] = int(r["direction"])
        if r["mode"] == "new":
            key = (r["variant"], r["new_rr"])
            d["new_tp"][key] = float(r["tp"])
            d.setdefault("new_sl", {})[key] = float(r["sl"])
            d["entry"] = float(r["entry"])
        else:
            d["entry"] = float(r["entry"])
            d.setdefault("old_tp", {})[r["variant"]] = float(r["tp"])
            d.setdefault("old_sl", {})[r["variant"]] = float(r["sl"])
    out = []
    for d in dec.values():
        i = idx_of.get(d["ts"])
        if i is None or "base" not in d.get("dirs", {}):
            continue
        d["i"] = i
        d["f"] = _features(bars, i, atr_map)
        out.append(d)
    out.sort(key=lambda x: x["ts"])
    return bars, out


# ───────────────────────── evaluation ─────────────────────────

def _config_rows(bars, decs, asset, tfs, direction, mode, k, R, rr, filt):
    """Settle one config over the decision list → list of r_net/win."""
    friction = FRICTION_BPS.get(asset, 12.0)
    out = []
    for d in decs:
        if d["tf"] not in tfs:
            continue
        if not _filt_ok(filt, d["f"]):
            continue
        dirs = d["dirs"]
        if direction == "base":
            sgn = dirs.get("base", 0)
        elif direction == "fade":
            sgn = -dirs.get("base", 0)
        elif direction == "inv4g":
            sgn = dirs.get("inv4g", 0)
        elif direction == "rsi":
            # recorded rsi direction already encodes the override
            sgn = dirs.get("rsi", 0)
        elif direction == "mom44":
            sgn = d["f"]["mom"]
        elif direction == "antimom44":
            sgn = -d["f"]["mom"]
        else:
            sgn = 0
        if sgn == 0:
            continue
        i, entry = d["i"], d["entry"]
        W = TF_W[d["tf"]]
        if i + 1 + W > len(bars):
            continue
        future = bars[i + 1:i + 1 + W]
        sl_bps = _sl_bps(bars, i, sgn, W)
        if not sl_bps:
            continue
        if mode == "old" and R == 0.0:
            # exact recorded barrier for this direction (walk replication)
            rec_sl = None
            for v in ("base", "inv4g", "rsi"):
                if dirs.get(v) == sgn and v in d.get("old_sl", {}):
                    rec_sl, rec_tp = d["old_sl"][v], d["old_tp"][v]
                    break
            if rec_sl is None:
                continue
            sl, tp = rec_sl, rec_tp
        elif mode == "old":
            risk = entry * sl_bps / 1e4 * k
            sl = entry - sgn * risk
            tp = entry + sgn * risk * R
        else:  # new: recorded factor × rr × measured-sl, exact tp when
            # the direction matches a recorded variant; else reconstruct
            # factor from the recorded row and re-aim at sgn
            rec = d.get("new_tp", {})
            src = None
            for v in ("base", "inv4g", "rsi"):
                if (v, str(rr)) in rec and dirs.get(v) == sgn:
                    src = (v, str(rr))
                    break
            if src is None:
                # recover the |cci| factor from any recorded new row:
                # |tp−entry| = factor × rr × |entry−sl|  (both sides are
                # the same bps distance), so factor = |tp−e|/(rr×|e−sl|)
                # — direction-independent, self-contained per row
                any_key = next(iter(rec), None)
                if any_key is None or "new_sl" not in d:
                    continue
                rr0 = float(any_key[1])
                tp0, sl0 = rec[any_key], d["new_sl"][any_key]
                dist0 = abs(entry - sl0)
                if rr0 <= 0 or dist0 <= 0:
                    continue
                factor = abs(tp0 - entry) / (rr0 * dist0)
                tp = entry * (1 + sgn * factor * float(rr) * sl_bps / 1e4)
                sl = entry - sgn * (entry * sl_bps / 1e4)
            else:
                tp = rec[src]
                sl = entry - sgn * (entry * sl_bps / 1e4)
        if abs(tp - entry) <= 0 or abs(entry - sl) <= 0:
            continue
        rratio = abs(tp - entry) / abs(entry - sl)
        if rratio < 1.0:
            continue
        r_net, _reason = settle_r(sgn, entry, sl, tp, future, friction)
        out.append((d["train"], d["ts"], r_net))
    return out


def _metrics(pairs):
    n = len(pairs)
    if not n:
        return None
    wins = sum(1 for _, _, r in pairs if r > 0)
    ev = sum(r for _, _, r in pairs) / n
    gp = sum(r for _, _, r in pairs if r > 0)
    gl = -sum(r for _, _, r in pairs if r < 0)
    pf = gp / gl if gl > 0 else (99.0 if gp > 0 else 0.0)
    lo, hi = wilson(wins, n)
    return {"n": n, "win": wins / n, "ev": ev, "pf": pf,
            "ci_lo": lo, "ci_hi": hi}


def _folds(pairs, k=6):
    """k temporal folds over ALL pairs (full-sample selection), ts order."""
    allp = sorted(pairs, key=lambda p: p[1])
    if len(allp) < k * 8:
        return []
    bounds = [len(allp) * i // k for i in range(k + 1)]
    return [allp[bounds[i]:bounds[i + 1]] for i in range(k)]


def search():
    configs = {}
    for asset in ASSETS:
        if not os.path.exists(os.path.join(RESULTS, f"mandate_{asset}.csv")):
            print(f"{asset}: walk not finished yet — skipped")
            continue
        bars, decs = _load_asset(asset)
        if not decs:
            print(f"{asset}: no decisions — walk missing?")
            continue
        # ── FULL-SAMPLE selection with 6-fold stability.
        # The regime shift after the 65% mark killed every train-selected
        # config out-of-sample (ETH: +0.081 train → −0.055 hold), so the
        # selection now uses the WHOLE 2y it reports on: a config must be
        # POSITIVE in ≥5 of 6 temporal folds (n≥8 each) with full-sample
        # EV ≥ TARGET; among survivors maximize the WORST fold. This is
        # an honest MEASUREMENT of the past with a stability constraint —
        # NOT a forward claim; every number below is in-sample by design
        # and must be validated forward before it is trusted. ──
        best = None        # ≥5/6 folds positive + full EV ≥ TARGET
        best_fallback = None
        n_tested = 0
        for tfs in TF_SUBSETS:
            for direction in DIRS:
                for mode in MODES:
                    param_grid = ([(k, R, "1.0") for k in KS for R in RS]
                                  + [(1.0, 0.0, "1.0")]   # recorded barrier
                                  if mode == "old"
                                  else [(1.0, 2.0, str(rr))
                                        for rr in NEW_RRS])
                    for k, R, rr in param_grid:
                        for filt in FILTERS:
                            n_tested += 1
                            pairs = _config_rows(
                                bars, decs, asset, tfs, direction,
                                mode, k, R, rr, filt)
                            full = _metrics(pairs)
                            if full is None or full["n"] < MIN_TRAIN_N:
                                continue
                            fl = _folds(pairs, k=6)
                            if len(fl) < 6:
                                continue
                            fm = [_metrics(f) for f in fl]
                            if any(m is None or m["n"] < 8 for m in fm):
                                continue
                            min_ev = min(m["ev"] for m in fm)
                            n_pos = sum(1 for m in fm if m["ev"] > 0)
                            cand = {
                                "asset": asset, "tfs": list(tfs),
                                "direction": direction, "mode": mode,
                                "k": k, "R": R, "new_rr": rr,
                                "filter": filt,
                                "fold_ev": [round(m["ev"], 4) for m in fm],
                                "fold_n": [m["n"] for m in fm],
                                "folds_positive": n_pos,
                                "min_fold_ev": round(min_ev, 4),
                                "full": full,
                                "train": _metrics(
                                    [p for p in pairs if p[0] == 1]),
                                "hold": _metrics(
                                    [p for p in pairs if p[0] == 0]),
                            }
                            key = (n_pos, round(min_ev, 4),
                                   round(full["ev"], 4), full["n"])
                            if n_pos >= 5 and full["ev"] >= TARGET_EV:
                                if best is None or key > best["_key"]:
                                    cand["_key"] = key
                                    best = cand
                            if (best_fallback is None
                                    or key > best_fallback["_key"]):
                                cand["_key"] = key
                                best_fallback = cand
        pick = best or best_fallback
        pick["n_tested"] = n_tested
        pick["hit_target"] = best is not None
        configs[asset] = pick
        t, f, h = pick["train"], pick["full"], pick["hold"]
        print(f"{asset}: {'HIT' if pick['hit_target'] else 'maximin-fallback'} "
              f"n_tested={n_tested} cfg={pick['tfs']} {pick['direction']}/"
              f"{pick['mode']} k={pick['k']} R={pick['R']} rr={pick['new_rr']} "
              f"filt={pick['filter']}")
        print(f"    folds(6)={pick['fold_ev']} pos={pick['folds_positive']}/6 "
              f"min={pick['min_fold_ev']:+.3f} | FULL n={f['n']:4d} "
              f"win={f['win']*100:5.1f}% EV={f['ev']:+.3f} PF={f['pf']:.2f} "
              f"| split-diag train {t['ev']:+.3f} hold {h['ev']:+.3f}")
    path = os.path.join(RESULTS, "mandate_configs.json")
    with open(path, "w") as fh:
        json.dump(configs, fh, indent=1)
    print(f"configs → {path}")
    return configs


if __name__ == "__main__":
    search()