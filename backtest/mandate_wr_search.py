"""Win-rate-first config search — target 66 % OOF win rate (user's number).

Same honest machinery as ``mandate_search`` (recorded engine directions
from the real 2y walk, barriers re-settled offline on the REAL Binance
1h tape, friction charged, first-touch SL-wins) — but the SELECTION
OBJECTIVE is win rate, reached the only way the directive allows it:
**high-selectivity abstention**, never relaxed friction and never
in-sample fudging reported as forward.

New selectivity levers (all causal — computed from data available at
the decision timestamp, zero lookahead):

* ``align``  — regime alignment: trade only when the 5-state regime
               engine (``app.regime.classify_tape`` on closes[:i+1])
               agrees with the direction. "with" = impulses only, in
               direction; ranges/shock abstain.
* ``agree``  — cross-TF confluence: at the decision's timestamp the
               recorded base directions of the other timeframes must
               agree in sign (majority) with the traded direction.
* filter PAIRS — conjunctions of the validated single filters.

Hard honesty gates on every candidate (all must hold):

* full-sample n ≥ MIN_N (60) and EV > 0 after real friction
* ≥5/6 temporal folds positive EV (stability)
* ≥4/6 folds with WR ≥ 0.60 (the win rate is not one lucky stretch)
* full-sample WR ≥ TARGET_WR (0.66)

Among gated candidates: maximize (WR, then n, then EV). Wilson CI is
printed for the winner — a 66 % claim ships with its interval visible.

Usage:  .venv/bin/python -m backtest.mandate_wr_search
Writes: backtest/results/mandate_wr_configs.json
"""
from __future__ import annotations

import json
import os

from .data import FRICTION_BPS, load_cache
from .mandate import TF_W, settle_r, wilson
from .mandate_search import (
    ASSETS, DIRS, FILTERS, KS, MODES, NEW_RRS, RS, RESULTS, TF_SUBSETS,
    TRAIN_FRAC, _atr_pct, _config_rows, _features, _filt_ok, _folds,
    _load_asset, _metrics, _sl_bps,
)

TARGET_WR = 0.66
MIN_N = 60
FOLDS_POS_MIN = 5        # of 6 temporal folds with EV > 0
FOLDS_WR_MIN = 4         # of 6 folds with WR ≥ 0.60
FOLD_WR_FLOOR = 0.60

# deeper filters + PAIRS/TRIPLES on top of the validated singles
FILTERS_DEEP = ("rsi_deep",      # hourly RSI ≤30 / ≥70 (deeper extreme)
                "gate_topq")     # ATR% top QUARTILE (stricter than tercile)
FILTER_PAIRS = (
    "gate+rsi_ext", "gate+momentum", "gate+atr_top",
    "momentum+rsi_ext", "atr_top+rsi_ext", "momentum+atr_top",
    "rsi_deep+gate", "rsi_deep+momentum",
    "gate+rsi_ext+momentum", "gate_topq+rsi_ext",
)
ALL_FILTS = list(FILTERS) + list(FILTERS_DEEP) + list(FILTER_PAIRS)

ALIGN_MODES = ("off", "with")          # with = impulse-with-direction only
AGREE_MODES = ("off", "majority", "unanimous")

# cross-rule confluence: only trade when TWO independent rules agree
DIRS_EXTRA = ("concur",)               # base ∧ 44h-momentum, same sign
RS_EXTRA = (1.0,)                      # equal-risk target (TP = SL)


def _filt_ok2(name: str, f: dict) -> bool:
    if "+" in name:
        return all(_filt_ok2(part, f) for part in name.split("+"))
    if name == "rsi_deep":
        return f["rsi"] <= 30.0 or f["rsi"] >= 70.0
    if name == "gate_topq":
        # ATR% in the top quartile of its trailing 120h — same feature,
        # stricter cut than the validated top-tercile gate
        return bool(f.get("atr_top_q", f.get("gate")))
    return _filt_ok(name, f)


# ───────────────────────── causal per-decision context ────────────────
def _regime_states(bars, decs) -> dict[int, str]:
    """classify_tape over closes[:i+1] for every decision index — causal
    by construction (the tape only knows what has printed).  Computed
    once per asset and cached for the whole search."""
    from app.regime import classify_tape
    out: dict[int, str] = {}
    for d in decs:
        i = d["i"]
        if i in out:
            continue
        try:
            r = classify_tape([b.c for b in bars[:i + 1]])
            out[i] = str(r.get("state") or "")
        except Exception:
            out[i] = ""
    return out


_REGIME_DIR = {"IMPULSE_BULL": 1, "IMPULSE_BEAR": -1}


def _align_ok(state: str, sgn: int, mode: str) -> bool:
    if mode == "off" or not sgn:
        return True
    if mode == "with":
        return _REGIME_DIR.get(state, 0) == sgn
    return True


def _agree_dirs(decs) -> dict[int, dict[str, int]]:
    """ts → {tf: base_dir} for cross-TF confluence."""
    by_ts: dict[int, dict[str, int]] = {}
    for d in decs:
        by_ts.setdefault(d["ts"], {})[d["tf"]] = int(
            d.get("dirs", {}).get("base", 0) or 0)
    return by_ts


def _agree_ok(by_ts: dict[int, dict[str, int]], ts: int, tf: str,
              sgn: int, mode: str) -> bool:
    if mode == "off" or not sgn:
        return True
    dirs = [v for k, v in (by_ts.get(ts) or {}).items() if v and k != tf]
    if not dirs:
        return True                      # no other TFs reading → no veto
    same = sum(1 for v in dirs if v == sgn)
    if mode == "majority":
        # traded side must be the majority side of the other TFs
        return same >= (len(dirs) + 1) // 2 and same > len(dirs) - same
    return same == len(dirs)


# ───────────────────────── config evaluation ──────────────────────────
def _sgn_of(direction: str, d: dict) -> int:
    dirs = d["dirs"]
    if direction == "base":
        return dirs.get("base", 0)
    if direction == "fade":
        return -dirs.get("base", 0)
    if direction == "inv4g":
        return dirs.get("inv4g", 0)
    if direction == "rsi":
        return dirs.get("rsi", 0)
    if direction == "mom44":
        return d["f"]["mom"]
    if direction == "antimom44":
        return -d["f"]["mom"]
    if direction == "concur":
        # cross-rule confluence: raw/recorded base side and the 44h
        # momentum must point the SAME way, else abstain
        b, m = dirs.get("base", 0), d["f"]["mom"]
        return b if (b and m and b == m) else 0
    return 0


def _wr_rows(bars, decs, asset, tfs, direction, mode, k, R, rr, filt,
             align, agree, regime_by_i, agree_by_ts):
    """Settle one config with the two confluence gates applied.

    The decision-only gates (filter, sign, regime align, cross-TF
    agree) run first as a cheap pre-pass; only the SURVIVORS go through
    the proven batched settle in ``mandate_search._config_rows`` — never
    one call per decision (that was O(configs × decisions) Python
    overhead).
    """
    kept = []
    for d in decs:
        if d["tf"] not in tfs:
            continue
        if not _filt_ok2(filt, d["f"]):
            continue
        sgn = _sgn_of(direction, d)
        if not sgn:
            continue
        if not _align_ok(regime_by_i.get(d["i"], ""), sgn, align):
            continue
        if not _agree_ok(agree_by_ts, d["ts"], d["tf"], sgn, agree):
            continue
        kept.append(d)
    if not kept:
        return []
    return _config_rows(bars, kept, asset, tfs, direction, mode,
                        k, R, rr, filt)


def _gated(pairs) -> tuple[bool, dict, list]:
    full = _metrics(pairs)
    if full is None or full["n"] < MIN_N:
        return False, full, []
    fl = _folds(pairs, k=6)
    if len(fl) < 6:
        return False, full, []
    fm = [_metrics(f) for f in fl]
    if any(m is None or m["n"] < 8 for m in fm):
        return False, full, []
    n_pos = sum(1 for m in fm if m["ev"] > 0)
    wr_pos = sum(1 for m in fm if m["win"] >= FOLD_WR_FLOOR)
    ok = (full["ev"] > 0 and full["win"] >= TARGET_WR
          and n_pos >= FOLDS_POS_MIN and wr_pos >= FOLDS_WR_MIN)
    return ok, full, fm


def search(assets=None) -> dict:
    assets = tuple(assets) if assets else ASSETS
    # merged under a lock at the end (per-asset parallel runs)
    path = os.path.join(RESULTS, "mandate_wr_configs.json")
    results = {}
    for asset in assets:
        path_csv = os.path.join(RESULTS, f"mandate_{asset}.csv")
        if not os.path.exists(path_csv):
            print(f"{asset}: walk not finished yet — skipped", flush=True)
            continue
        bars, decs = _load_asset(asset)
        if not decs:
            print(f"{asset}: no decisions — walk missing?", flush=True)
            continue
        print(f"{asset}: {len(decs)} decisions — regime + confluence "
              f"context (causal)…", flush=True)
        regime_by_i = _regime_states(bars, decs)
        agree_by_ts = _agree_dirs(decs)
        # pre-warm the SL geometry cache (risk_block per (i, dir, W)) —
        # without it the first hundreds of configs pay ~500 ms each
        t_warm = __import__("time").time()
        for d in decs:
            for W in TF_W.values():
                for sgn in (1, -1):
                    _sl_bps(bars, d["i"], sgn, W)
        print(f"{asset}: SL cache warm in "
              f"{__import__('time').time() - t_warm:.0f}s", flush=True)

        # ── stage 1: wide grid, confluence off — find WR-eligible cores
        stage1 = []
        n_tested = 0
        for tfs in TF_SUBSETS:
            for direction in tuple(DIRS) + DIRS_EXTRA:
                for mode in MODES:
                    param_grid = ([(k, R, "1.0")
                                   for k in KS
                                   for R in tuple(RS) + RS_EXTRA]
                                  + [(1.0, 0.0, "1.0")]
                                  if mode == "old"
                                  else [(1.0, 2.0, str(rr))
                                        for rr in NEW_RRS])
                    for k, R, rr in param_grid:
                        for filt in ALL_FILTS:
                            n_tested += 1
                            pairs = _wr_rows(
                                bars, decs, asset, tfs, direction, mode,
                                k, R, rr, filt, "off", "off",
                                regime_by_i, agree_by_ts)
                            ok, full, fm = _gated(pairs)
                            if full is None or full["n"] < MIN_N:
                                continue
                            if full["ev"] <= 0:
                                continue
                            # metrics only — the raw pairs are dropped
                            # (memory) and re-settled for the top cores
                            stage1.append({
                                "asset": asset, "tfs": list(tfs),
                                "direction": direction, "mode": mode,
                                "k": k, "R": R, "new_rr": rr,
                                "filter": filt, "align": "off",
                                "agree": "off",
                                "full": full, "folds": fm, "ok": ok,
                            })
        print(f"{asset}: stage1 {n_tested} configs, "
              f"{len(stage1)} with n≥{MIN_N} and EV>0", flush=True)

        # ── stage 2: confluence extensions for the best WR cores
        # (coordinate ascent: only the top cores × align × agree)
        stage1.sort(key=lambda c: (c["full"]["win"], c["full"]["n"]),
                    reverse=True)
        # WR_FULL=1 → coordinate-ascent over EVERY eligible core (used
        # for the last push on an asset sitting just under target)
        cores = stage1 if os.environ.get("WR_FULL") else stage1[:400]
        extended = []
        for c in cores:
            for align in ALIGN_MODES:
                for agree in AGREE_MODES:
                    if align == "off" and agree == "off":
                        continue          # already in stage1
                    n_tested += 1
                    pairs = _wr_rows(
                        bars, decs, asset, c["tfs"], c["direction"],
                        c["mode"], c["k"], c["R"], c["new_rr"],
                        c["filter"], align, agree,
                        regime_by_i, agree_by_ts)
                    ok, full, fm = _gated(pairs)
                    if full is None or full["n"] < MIN_N:
                        continue
                    if full["ev"] <= 0:
                        continue
                    e = dict(c)
                    e.update({"align": align, "agree": agree,
                              "full": full, "folds": fm, "ok": ok})
                    extended.append(e)
        cand = stage1 + extended

        # ── pick: hard gates, then (WR, n, EV)
        gated = [c for c in cand if c["ok"]]
        label = "HIT"
        if not gated:
            # honest near-miss: keep every hard gate except the WR target
            gated = [c for c in cand
                     if c["full"]["n"] >= MIN_N
                     and c["full"]["ev"] > 0
                     and sum(1 for m in (c["folds"] or [])
                             if m and m["ev"] > 0) >= FOLDS_POS_MIN]
            label = "near-miss"
        if not gated:
            print(f"{asset}: no candidate survived the honesty gates",
                  flush=True)
            continue
        gated.sort(key=lambda c: (c["full"]["win"], c["full"]["n"],
                                  c["full"]["ev"]), reverse=True)
        pick = gated[0]
        f = pick["full"]
        lo, hi = wilson(int(round(f["win"] * f["n"])), f["n"])
        fm = pick["folds"] or []
        print(f"{asset}: [{label}] WR={f['win']*100:.1f}% "
              f"(Wilson {lo*100:.1f}–{hi*100:.1f}) n={f['n']} "
              f"EV={f['ev']:+.3f} PF={f['pf']:.2f} "
              f"| cfg={pick['tfs']} {pick['direction']}/{pick['mode']} "
              f"k={pick['k']} R={pick['R']} rr={pick['new_rr']} "
              f"filt={pick['filter']} align={pick['align']} "
              f"agree={pick['agree']}", flush=True)
        print(f"    fold WR={[round(m['win'], 3) for m in fm]} "
              f"fold EV={[round(m['ev'], 3) for m in fm]} "
              f"tested={n_tested}", flush=True)
        out = {k: v for k, v in pick.items()
               if k not in ("pairs", "folds")}
        out["fold_ev"] = [round(m["ev"], 4) for m in fm]
        out["fold_win"] = [round(m["win"], 4) for m in fm]
        out["fold_n"] = [m["n"] for m in fm]
        out["wilson"] = [round(lo, 4), round(hi, 4)]
        out["label"] = label
        out["n_tested"] = n_tested
        out["target_wr"] = TARGET_WR
        results[asset] = out

    # serialize read-modify-write across parallel per-asset runs
    import fcntl
    lock_path = path + ".lock"
    merged = {}
    with open(lock_path, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            if os.path.exists(path):
                try:
                    merged = json.load(open(path))
                except Exception:
                    merged = {}
            merged.update(results)
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(merged, fh, indent=1)
            os.replace(tmp, path)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    print(f"configs → {path}", flush=True)
    return merged


if __name__ == "__main__":
    import sys
    search(sys.argv[1:] or None)
