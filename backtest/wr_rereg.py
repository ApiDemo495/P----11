"""Re-grade the SHIPPED hold cells at the resolution they were measured on.

The certified artifact (backtest/results/hold_wr.json) selects (side,
row_k, row_m) on the 5-minute tape.  Re-running ``backtest.hold
.grade_cell`` on fresh 5m tape reproduces that selection one level up
(``selected.sides.*.row_wr_best_*``) and then reports, for the SHIPPED
geometry, the number that actually matters:

    real win rate  = P(gross window pnl > full round trip)

plus the train/holdout split and the era (one-position-per-side-run)
read, which is the structure the live app trades.

Usage:  .venv/bin/python -m backtest.wr_rereg
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.continuous import _scale_at, load_bars   # noqa: E402
from backtest.data import FRICTION_BPS                # noqa: E402
from backtest.hold import (K_ROW, M_ROW, build_sides,  # noqa: E402
                           era_sim, grade_cell)

OUT = os.path.join("backtest", "results", "wr_rereg_5m.json")


def shown(asset, w_min, side_name, pol):
    """Shipped-geometry numbers on the finest tape available."""
    bars = load_bars(asset)                 # 5m if present, else 1h
    if bars is None:
        return None
    t, o, h, l, c, v, step = bars
    if int(step) * 4 > w_min * 60:
        return {"cell": f"{asset}|{w_min}", "status": "ungradeable",
                "why": f"{step // 60}m bars cannot resolve {w_min}-min"}
    W = w_min * 60
    t0 = int(np.ceil(int(t[0]) / W) * W)
    starts = np.arange(t0, int(t[-1]) + int(step) - W + 1, W, dtype=np.int64)
    i_entry = np.searchsorted(t, starts - step, side="right") - 1
    i_exit = np.searchsorted(t, starts + W - step, side="right") - 1
    ok = (i_entry >= 0) & (i_exit > i_entry) & (i_exit < len(t))
    starts, i_entry, i_exit = starts[ok], i_entry[ok], i_exit[ok]
    if len(starts) < 50:
        return {"cell": f"{asset}|{w_min}", "status": "too_few", "n": len(starts)}
    sides = build_sides(t, o, h, l, c, v, step, starts, i_entry, None, w_min)
    side = sides[side_name]
    entry = c[i_entry]
    scale = _scale_at(t, c, starts - step, max(2, int(round(W / (step / 60.0)))))
    good = np.isfinite(scale) & (scale > 0)
    side, entry, scale = side[good], entry[good], scale[good]
    ie, ix = i_entry[good], i_exit[good]
    n = len(side)

    fr = FRICTION_BPS[asset]
    rt = 2.0 * fr * 1e-4
    kr, mr = float(pol["row_k"]), float(pol["row_m"])
    ke, me = float(pol["era_k"]), float(pol["era_m"])
    m_row, k_row = mr, kr

    # ── ROW geometry: first-touch inside each window (every window)
    from backtest.continuous import _simulate
    exit_ = c[ix]
    moves = exit_ - entry
    res = _simulate(side, entry, exit_, moves, scale, k_row, m_row,
                    starts[good], t, h, l, step, 0.0, ie, ix)
    pnl, win = res["pnl"], res["win"]
    cuts = np.searchsorted(starts[good], starts[good][0])
    years = np.array([int(__import__("datetime").datetime.utcfromtimestamp(
        int(s)).year) for s in starts[good]])
    uniq = sorted(set(years.tolist()))
    tr = years < uniq[-1] if len(uniq) > 1 else np.ones(n, bool)
    ho = ~tr
    net_win = pnl > rt

    def wfrac(mask):
        return round(float(net_win[mask].mean()), 4) if mask.any() else None

    # ── ERA structure: one position per side run (what live holds)
    era = era_sim(side, starts[good], ie, ix, entry, scale, t, h, l, c,
                  (ke if 0 < me < 999 else k_row), (me if 0 < me < 999 else m_row),
                  fr)
    era_pnl = np.array([r["pnl_net"] for r in era])
    era_startw = np.array([r["start_i"] for r in era], dtype=int)

    risk_row = (m_row * scale) / entry
    out = {
        "cell": f"{asset}|{w_min}", "status": "ok",
        "tape": f"{int(step)//60}m bars", "n_windows": int(n),
        "side": side_name, "n_years": len(uniq),
        "row_geometry": {"k": kr, "m": mr},
        "era_geometry": {"k": ke, "m": me,
                         "guarded": bool(0 < me < 999)},
        "row_sl_bps_median": round(float(np.median(risk_row)) * 1e4, 1),
        "row_tp_bps_median": round(float(np.median(
            k_row * m_row * scale / entry)) * 1e4, 1),
        "rt_bps": round(rt * 1e4, 3),
        "gross_row_wr": round(float(win.mean()), 4),
        "real_row_wr": round(float(net_win.mean()), 4),
        "train_real_row_wr": wfrac(tr),
        "holdout_real_row_wr": wfrac(ho),
        "mean_net_bps_per_window": round(float((pnl - rt).mean()) * 1e4, 2),
        "era_n": int(len(era_pnl)),
        "era_win_rate": (round(float((era_pnl > 0).mean()), 4)
                         if len(era_pnl) else None),
        "era_mean_net_bps": (round(float(era_pnl.mean()) * 1e4, 2)
                             if len(era_pnl) else None),
        "era_pf": (round(float(era_pnl[era_pnl > 0].sum()
                               / -era_pnl[era_pnl < 0].sum()), 3)
                   if len(era_pnl) and (era_pnl < 0).any() else None),
        "era_avg_hold_windows": (round(float(np.mean(
            [r["n_windows"] for r in era])), 1) if era else None),
    }
    if len(era_pnl) > 8:
        cut = int(len(era_pnl) * 0.65)
        out["era_train_mean_net_bps"] = round(float(era_pnl[:cut].mean()) * 1e4, 2)
        out["era_holdout_mean_net_bps"] = round(
            float(era_pnl[cut:].mean()) * 1e4, 2)
        out["era_holdout_n"] = int(len(era_pnl) - cut)
        out["era_holdout_wr"] = round(float((era_pnl[cut:] > 0).mean()), 4)
    return out


def main():
    art = json.load(open(os.path.join("backtest", "results", "hold_wr.json")))
    rows = []
    for key in sorted(art):
        if key == "meta":
            continue
        c = art[key]
        if c.get("status") != "ok":
            continue
        asset, tfm = key.split("|")
        r = shown(asset, int(tfm), c["selected"]["side"], c["selected"])
        if r is None:
            print(f"{key}: no tape")
            continue
        rows.append(r)
        if r.get("status") != "ok":
            print(f"{key:9s} {r['status']}: {r.get('why','')}")
            continue
        print(f"\n=== {r['cell']:9s} {r['tape']}  side={r['side']} "
              f"n={r['n_windows']}  RT={r['rt_bps']}bps  "
              f"SL {r['row_sl_bps_median']}bps / TP {r['row_tp_bps_median']}bps")
        print(f"  row : gross {r['gross_row_wr']*100:5.1f}%  "
              f"REAL {r['real_row_wr']*100:5.1f}%  "
              f"(train {r['train_real_row_wr']}, holdout {r['holdout_real_row_wr']})"
              f"  net {r['mean_net_bps_per_window']:+.1f}bps/window")
        print(f"  era : n={r['era_n']}  WR {((r['era_win_rate'] or 0)*100):5.1f}%"
              f"  mean {r['era_mean_net_bps']}bps  PF {r['era_pf']}"
              f"  avg hold {r['era_avg_hold_windows']}w"
              + (f"  | holdout n={r.get('era_holdout_n')} "
                 f"mean {r.get('era_holdout_mean_net_bps')}bps"
                 if r.get("era_holdout_n") else ""))
    json.dump(rows, open(OUT, "w"), indent=1)
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
