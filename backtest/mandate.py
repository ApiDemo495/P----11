"""2-year accuracy mandate: per asset × TF × old/new model, REAL data.

Answers "what is the honest accuracy / win rate of each (asset, tf,
system_mode) cell on the last two years of real market data, with the
best defensible tuning" — measured, not claimed:

  - Data: Binance Vision 1h klines (GitHub mirror, zero gaps verified):
      BTC/ETH/BNB 2024-11-01 → 2026-09-30 (16,776 bars),
      PAXG        2025-03-27 → 2026-09-30 (13,262 bars).
    Bars are 1h; TF windows are W bars = 22/44/130/247 (the same
    convention as per_tf_real.py: "TF1" = the 22-bar horizon).
  - One engine pass per decision (`AppPredictor.pack`) yields ALL four
    TFs — the full app engine (37 formulas, CCI fusion + direction
    ladder, four brains, physics/emotion stacks), learning frozen
    (record=False). Order-book/news features are absent here exactly
    as in any kline backtest — never pretended otherwise.
  - system_mode OLD  = measured TP/SL ladder (risk_block), the live
    settlement contract.  k=1.0 for the raw engine, k=0.75 tightening
    for the policy variants (their validated form).
  - system_mode NEW  = the engine's horizon forecast target
    (σ-scaled expected move tilted by |CCI|/0.08 — engine.py:6566),
    measured SL kept — the live NEW rule, replayed bar-by-bar.
  - Issuance gates replicated: direction ≠ 0 and ratio ≥ 1.0
    (below_ratio_floor → watch-only, engine rule). First-touch TP/SL,
    SL wins same-bar ties, friction charged (12 bps BTC/ETH/BNB,
    18 bps PAXG).
  - Direction variants per cell (selected on TRAIN = first 65%,
    reported on HOLDOUT = last 35% — no holdout peeking):
      base  engine direction (the app as shipped)
      inv4g inverse TF4 + ATR top-tercile gate (validated policy)
      rsi   RSI(14) ≤30/≥70 mean-reversion override else engine
  - Overlapping windows for tf3/tf4 (step 22 < W) — n overstates
    independence there; Wilson 95% CIs are printed so the uncertainty
    is visible.

Usage:
  .venv/bin/python -m backtest.mandate --asset BTC --smoke   # quick check
  .venv/bin/python -m backtest.mandate --asset BTC           # full walk
  .venv/bin/python -m backtest.mandate --summary             # merge CSVs
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time

from .data import FRICTION_BPS, load_cache
from .predictor import AppPredictor
from .slice import DataSlice
# Spacing of the tape this harness walks, in MINUTES. The TP/SL
# block measures a HOLDING PERIOD; without this it measured a
# BAR COUNT, so a 1h tape silently produced 22-HOUR stops for a
# 22-minute horizon (app/moves_risk.moves_stats, 2026-11-26).
_BAR_MINUTES = {"1m": 1.0, "5m": 5.0, "15m": 15.0, "30m": 30.0,
                "1h": 60.0, "2h": 120.0, "4h": 240.0}.get(
                    os.environ.get("BACKTEST_TF", "15m"), 15.0)

TF_W = {"tf1": 22, "tf2": 44, "tf3": 130, "tf4": 247}
STEP = 22
WARMUP = 3500
TRAIN_FRAC = 0.65
K_POLICY = 0.75            # validated/tp4 barrier tightening
RESULTS = os.path.join(os.path.dirname(__file__), "results")

VARIANTS = ("base", "inv4g", "rsi")
NEW_RRS = (1.2, 1.6, 2.0)   # NEW-mode target candidates (tuned → engine)


# ───────────────────────────── helpers ─────────────────────────────

def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def _rsi(closes: list[float], n: int = 14) -> float:
    if len(closes) < n + 1:
        return 50.0
    gains, losses = [], []
    for a, b in zip(closes[-n - 1:-1], closes[-n:]):
        d = b - a
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag, al = sum(gains) / n, sum(losses) / n
    if al <= 0:
        return 100.0
    return 100.0 - 100.0 / (1 + ag / al)


def _atr_pct(bars, n: int = 14) -> list[float]:
    out = []
    for i in range(len(bars)):
        if i < n:
            continue
        seg = bars[i - n:i + 1]
        trs = [max(b.h - b.l, abs(b.h - a.c), abs(b.l - a.c))
               for a, b in zip(seg[:-1], seg[1:])]
        out.append((sum(trs) / len(trs)) / bars[i].c * 100 if bars[i].c else 0.0)
    return out


def gate_ok(bars) -> bool:
    """ATR% top-tercile of its own trailing 120 bars (validated policy)."""
    s = _atr_pct(bars)[-120:]
    if len(s) < 30:
        return False
    thr = sorted(s)[int(len(s) * (2 / 3))]
    return s[-1] >= thr


def direction_variants(pack: dict, closes: list[float], gated: bool) -> dict[str, int]:
    """base / inverse-TF4(gated) / RSI-override directions for one decision."""
    tf4 = int((pack.get("tf4") or {}).get("direction") or 0)
    out = {}
    for tf in TF_W:
        base = int((pack.get(tf) or {}).get("direction") or 0)
        out[f"base|{tf}"] = base
        inv = -tf4
        out[f"inv4g|{tf}"] = inv if (gated and inv != 0) else 0
        r = _rsi(closes)
        if r <= 30:
            out[f"rsi|{tf}"] = 1
        elif r >= 70:
            out[f"rsi|{tf}"] = -1
        else:
            out[f"rsi|{tf}"] = base
    return out


def settle_r(direction: int, entry: float, sl: float, tp: float,
             future, friction_bps: float) -> tuple[float, str]:
    """First-touch TP/SL over `future` bars; SL wins ties. Returns (r_net, reason)."""
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0, "SKIP"
    s = 1 if direction >= 0 else -1
    exit_price, reason = future[-1].c, "TIMEOUT"
    for b in future:
        hit_sl = b.l <= sl if s > 0 else b.h >= sl
        hit_tp = b.h >= tp if s > 0 else b.l <= tp
        if hit_sl:
            exit_price, reason = sl, "SL"
            break
        if hit_tp:
            exit_price, reason = tp, "TP"
            break
    r_gross = s * (exit_price - entry) / risk
    cost = entry * friction_bps / 1e4
    return r_gross - cost / risk, reason


# ───────────────────────────── walk ─────────────────────────────

def walk_asset(asset: str, smoke: bool = False) -> str:
    bars = load_cache(asset, "1h")
    if len(bars) < WARMUP + TF_W["tf4"] + 10:
        raise SystemExit(f"{asset}: only {len(bars)} bars — run "
                         "scripts/fetch_binance_mirror.py first")
    # gap check (honesty: a gapped tape lies about first-touch)
    for a, b in zip(bars, bars[1:]):
        if b.t - a.t != 3600:
            raise SystemExit(f"{asset}: gap {a.t}->{b.t} — refetch")

    # the mandate rules (app/mandate_cfg) are DEFINED against the raw
    # engine — the walk must keep measuring raw, or re-deriving rules
    # from post-rule directions would be circular
    os.environ["MANDATE_RAW"] = "1"
    pred = AppPredictor()
    ds = DataSlice(bars)
    friction = FRICTION_BPS.get(asset, 12.0)
    rows: list[dict] = []
    t = WARMUP
    end = len(bars) - TF_W["tf4"] - 1
    if smoke:
        end = min(end, t + 60)
    t0 = time.time()
    n_pack = 0
    skipped_ratio = 0
    while t < end:
        view = ds.at(t)
        try:
            pack = pred.pack(view, asset)
        except Exception as e:                       # pragma: no cover
            print(f"  [{asset}] pack failed at t={t}: {e!r}")
            t += STEP
            continue
        n_pack += 1
        entry = bars[t].c
        closes = list(view.closes)
        gated = gate_ok(bars[:t + 1])
        dirs = direction_variants(pack, closes, gated)
        from app.moves_risk import risk_block
        for tf, W in TF_W.items():
            res = pack.get(tf) or {}
            conf = float(res.get("confidence") or 0.0)
            atr = float(res.get("atr") or 0.0)
            cci_adj = float(res.get("cci_adjusted") or res.get("cci") or 0.0)
            future = bars[t + 1:t + 1 + W]
            if len(future) < W:
                continue
            for variant in VARIANTS:
                d = dirs.get(f"{variant}|{tf}", 0)
                if d == 0:
                    continue
                blk = risk_block(entry, d, W, conf,
                                 closes, list(view.highs), list(view.lows),
                                 atr=atr, bar_minutes=_BAR_MINUTES)
                if blk.get("sl") is None or blk.get("tp") is None:
                    continue
                # ── OLD: measured ladder (policy variants at k=0.75) ──
                if variant == "base":
                    sl_o, tp_o = float(blk["sl"]), float(blk["tp"])
                else:
                    risk0 = abs(entry - float(blk["sl"]))
                    sl_o = entry - d * risk0 * K_POLICY
                    tp_o = entry + d * 2.0 * risk0 * K_POLICY
                # ── NEW: signal-tilted target in TAIL units (the fix —
                # the old σ-mean target could never clear the q80 stop),
                # measured SL kept, rr candidates tuned on train ──
                factor = 0.35 + 0.65 * min(1.0, abs(cci_adj) / 0.08)
                sl_n = float(blk["sl"])
                sl_bps_n = abs(entry - sl_n) / entry * 1e4
                candidates = [("old", "1.0", tp_o, sl_o)] + [
                    ("new", f"{rr_cand}",
                     entry * (1 + d * factor * rr_cand * sl_bps_n / 1e4),
                     sl_n) for rr_cand in NEW_RRS]
                for mode, new_rr, tp, sl in candidates:
                    if sl == entry:
                        continue
                    rr = abs(tp - entry) / abs(entry - sl)
                    if rr < 1.0:                    # below_ratio_floor
                        skipped_ratio += 1
                        continue
                    r_net, reason = settle_r(d, entry, sl, tp, future, friction)
                    rows.append({
                        "asset": asset, "tf": tf, "mode": mode,
                        "variant": variant, "new_rr": new_rr,
                        "train": None,   # patched below (time split)
                        "ts": bars[t].t,
                        "direction": d, "entry": round(entry, 6),
                        "tp": round(tp, 6), "sl": round(sl, 6),
                        "rr": round(rr, 3), "r_net": round(r_net, 4),
                        "reason": reason, "win": int(r_net > 0),
                    })
        t += STEP
        if n_pack % 50 == 0 and n_pack:
            rate = n_pack / max(1e-9, time.time() - t0)
            eta = ((end - t) / STEP) / max(rate, 1e-9)
            print(f"  [{asset}] {n_pack} passes · "
                  f"~{eta:,.0f}s left · {len(rows)} rows", flush=True)
        if smoke and n_pack >= 3:
            break

    # train/holdout split by TIME (first 65% of decisions = train)
    if rows:
        times = sorted({r["ts"] for r in rows})
        cut = times[int(len(times) * TRAIN_FRAC)] if len(times) > 4 else times[-1]
        for r in rows:
            r["train"] = int(r["ts"] < cut)
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"mandate_{asset}{'_smoke' if smoke else ''}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[{asset}] {n_pack} engine passes · {len(rows)} rows · "
          f"ratio-floor skips {skipped_ratio} → {path}")
    return path


# ───────────────────────────── summary ─────────────────────────────

def _metrics(rows: list[dict]) -> dict:
    n = len(rows)
    wins = sum(r["win"] for r in rows)
    ev = sum(r["r_net"] for r in rows) / n if n else 0.0
    gp = sum(r["r_net"] for r in rows if r["r_net"] > 0)
    gl = -sum(r["r_net"] for r in rows if r["r_net"] < 0)
    pf = (gp / gl) if gl > 0 else float("inf") if gp > 0 else 0.0
    lo, hi = wilson(wins, n)
    return {"n": n, "win": wins / n if n else 0.0, "ev": ev,
            "pf": pf, "ci_lo": lo, "ci_hi": hi}


def summary() -> None:
    paths = [os.path.join(RESULTS, f) for f in os.listdir(RESULTS)
             if f.startswith("mandate_") and f.endswith(".csv")
             and "smoke" not in f and "summary" not in f]
    if not paths:
        raise SystemExit("no mandate_*.csv — run a walk first")
    all_rows = []
    for p in paths:
        with open(p) as f:
            all_rows.extend(csv.DictReader(f))
    for r in all_rows:
        r["train"] = int(r["train"])
        r["win"] = int(r["win"])
        r["r_net"] = float(r["r_net"])

    print("=" * 108)
    print("2-YEAR MANDATE — per (asset, tf, mode): variant (+ new_rr for "
          "NEW) chosen on TRAIN, reported on HOLDOUT")
    print("=" * 108)
    hdr = (f"{'asset':5} {'tf':4} {'mode':4} {'sel':8} "
           f"{'TRAIN n':>7} {'win%':>6} {'EV':>7} {'PF':>5} | "
           f"{'HOLD n':>6} {'win%':>6} {'EV':>7} {'PF':>5} {'95% CI':>13}")
    print(hdr)
    print("-" * len(hdr))
    out_rows = []
    cells = sorted({(r["asset"], r["tf"], r["mode"]) for r in all_rows})
    for asset, tf, mode in cells:
        cell = [r for r in all_rows
                if r["asset"] == asset and r["tf"] == tf and r["mode"] == mode]
        tr = [r for r in cell if r["train"] == 1]
        ho = [r for r in cell if r["train"] == 0]
        # candidate configs: (variant, new_rr); selection on TRAIN only
        # by EV, tie-break win%, then n — holdout is never consulted
        cands = ([(v, str(rr)) for v in VARIANTS for rr in NEW_RRS]
                 if mode == "new" else [(v, "1.0") for v in VARIANTS])
        best, best_key = None, None
        for v, rr in cands:
            vr = [r for r in tr if r["variant"] == v and r["new_rr"] == rr]
            if len(vr) < 10:
                continue
            m = _metrics(vr)
            key = (round(m["ev"], 4), round(m["win"], 4), m["n"])
            if best_key is None or key > best_key:
                best, best_key = (v, rr), key
        if best is None:
            best = ("base", "1.0")
        bv, brr = best
        trm = _metrics([r for r in tr
                        if r["variant"] == bv and r["new_rr"] == brr])
        hom = _metrics([r for r in ho
                        if r["variant"] == bv and r["new_rr"] == brr])
        sel = bv if mode == "old" else f"{bv}:rr{brr}"
        ci = f"[{hom['ci_lo']:.2f},{hom['ci_hi']:.2f}]"
        print(f"{asset:5} {tf:4} {mode:4} {sel:8} "
              f"{trm['n']:7d} {trm['win']*100:5.1f}% {trm['ev']:+7.3f} {trm['pf']:5.2f} | "
              f"{hom['n']:6d} {hom['win']*100:5.1f}% {hom['ev']:+7.3f} "
              f"{hom['pf']:5.2f} {ci:>13}")
        out_rows.append({"asset": asset, "tf": tf, "mode": mode,
                         "selected": sel, "variant": bv, "new_rr": brr,
                         "train_n": trm["n"], "train_win": round(trm["win"], 4),
                         "train_ev": round(trm["ev"], 4),
                         "hold_n": hom["n"], "hold_win": round(hom["win"], 4),
                         "hold_ev": round(hom["ev"], 4),
                         "hold_pf": round(hom["pf"], 4),
                         "hold_ci_lo": round(hom["ci_lo"], 4),
                         "hold_ci_hi": round(hom["ci_hi"], 4)})
    # pooled headline: the SELECTED config of every cell, by mode
    print("-" * len(hdr))
    for mode in ("old", "new"):
        sel_rows = []
        for o in out_rows:
            if o["mode"] != mode:
                continue
            key_r = o["new_rr"] if mode == "new" else "1.0"
            sel_rows.extend(
                r for r in all_rows
                if r["asset"] == o["asset"] and r["tf"] == o["tf"]
                and r["mode"] == mode and r["variant"] == o["variant"]
                and r["new_rr"] == key_r and r["train"] == 0)
        m = _metrics(sel_rows)
        ci = f"[{m['ci_lo']:.2f},{m['ci_hi']:.2f}]"
        print(f"{'ALL':5} {'':4} {mode:4} {'pooled':8} "
              f"{'':7} {'':6} {'':7} {'':5} | "
              f"{m['n']:6d} {m['win']*100:5.1f}% {m['ev']:+7.3f} "
              f"{m['pf']:5.2f} {ci:>13}   ← HOLDOUT, selected configs")
    out = os.path.join(RESULTS, "mandate_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print("-" * len(hdr))
    print(f"summary → {out}")
    print("NOTE: tf3/tf4 windows overlap (step 22 < W) — their n overstates "
          "independence; CIs above are Wilson on those n.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", choices=["BTC", "ETH", "BNB", "PAXG"])
    ap.add_argument("--smoke", action="store_true",
                    help="3 engine passes only — harness validation")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args(argv)
    if a.summary:
        summary()
        return 0
    assets = [a.asset] if a.asset else ["BTC", "ETH", "BNB", "PAXG"]
    for s in assets:
        walk_asset(s, smoke=a.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())