"""INDEPENDENT, out-of-sample audit of the certified hold policy.

Answers three questions with real Binance 1h tape only:

1. What does the geometry the app ACTUALLY trades (row k/m, or the
   era geometry when guarded) pay after the FULL round trip?
2. Can a different barrier geometry raise the REAL win rate
   P(net > 0), selected on TRAIN and judged on HOLDOUT?
3. Is any of it distinguishable from the no-edge baseline? (A barrier
   rule cannot create edge; it can only reshape the win/loss
   distribution. If directional accuracy is ~50 %, every geometry
   lands on EV ≈ 0 minus costs.)

Causality: side from bars strictly before the window opens (identical
to backtest.hold.build_sides / app.hold_policy.side_value); barrier
scale = trailing q80 window-move; first touch inside the window; SL
wins same-bar ties (the app's pessimistic convention); friction = the
asset's full-stack round trip.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.continuous import _scale_at                      # noqa: E402
from backtest.data import FRICTION_BPS, load_cache             # noqa: E402
from backtest.hold import build_sides                          # noqa: E402

ART = os.path.join("backtest", "results", "hold_wr.json")
OUT = os.path.join("backtest", "results", "wr_audit_independent.json")
TRAIN_FRAC = 0.65
MIN_N = 40          # 247-min cells only have ~97 windows on 2y of 1h tape


def _arrays(asset):
    bars = load_cache(asset, "1h")
    return {k: np.array([getattr(b, k) for b in bars],
                        dtype=np.int64 if k == "t" else float)
            for k in ("t", "o", "h", "l", "c", "v")}


def simulate(side, entry, sl_px, tp_px, i_entry, i_exit, h, l, c):
    """First-touch inside the window, SL wins ties. Returns price return."""
    n = len(side)
    pnl = np.zeros(n)
    touch = np.zeros(n, dtype=int)              # -1 sl, +1 tp, 0 neither
    for i in range(n):
        s = int(side[i])
        a, b = int(i_entry[i]) + 1, int(i_exit[i])
        e = float(entry[i])
        slp = e - float(sl_px[i]) * s
        tpp = e + float(tp_px[i]) * s
        out, px = 0, None
        for j in range(a, b + 1):
            hi, lo = float(h[j]), float(l[j])
            if s > 0:
                sl_hit, tp_hit = lo <= slp, hi >= tpp
            else:
                sl_hit, tp_hit = hi >= slp, lo <= tpp
            if sl_hit:
                out, px = -1, slp
                break
            if tp_hit:
                out, px = 1, tpp
                break
        if out == 0:
            px = float(c[int(i_exit[i])])
        touch[i] = out
        pnl[i] = (px / e - 1.0) * s
    return pnl, touch


def stats(pnl, sl_frac, rt):
    """Metrics indistinguishable from fabrication-free arithmetic."""
    net = pnl - rt
    r_net = net / np.maximum(sl_frac, 1e-12)
    return {
        "n": int(len(pnl)),
        "gross_wr": round(float((pnl > 0).mean()), 4),
        "net_wr": round(float((net > 0).mean()), 4),
        "mean_net_bps": round(float(net.mean()) * 1e4, 2),
        "ev_net_R": round(float(r_net.mean()), 4),
        "pf_net": (round(float(net[net > 0].sum() / -net[net < 0].sum()), 3)
                   if (net < 0).any() else None),
        "tp_share": (round(float((pnl > 0).mean()), 4)),
    }


def cell(asset, w_min, side_name, pol):
    A = _arrays(asset)
    t, o, h, l, c, v = (A[k] for k in ("t", "o", "h", "l", "c", "v"))
    W, step = w_min * 3600, 3600
    t0 = int(np.ceil(t[0] / W) * W)
    starts = np.arange(t0, int(t[-1]) + step - W + 1, W, dtype=np.int64)
    i_entry = np.searchsorted(t, starts - step, side="right") - 1
    i_exit = np.searchsorted(t, starts + W - step, side="right") - 1
    ok = (i_entry >= 0) & (i_exit > i_entry) & (i_exit < len(t))
    starts, i_entry, i_exit = starts[ok], i_entry[ok], i_exit[ok]
    if len(starts) < 2 * MIN_N:
        return None
    sides = build_sides(t, o, h, l, c, v, step, starts, i_entry, None, w_min)
    side = sides[side_name]
    entry = c[i_entry]
    scale = _scale_at(t, c, starts - step, max(2, w_min))
    good = np.isfinite(scale) & (scale > 0)
    if good.sum() < 2 * MIN_N:
        return None
    side, entry, scale = side[good], entry[good], scale[good]
    ie, ix = i_entry[good], i_exit[good]

    rt = 2.0 * FRICTION_BPS[asset] * 1e-4
    n = len(side)
    cut = int(n * TRAIN_FRAC)
    tr = np.zeros(n, dtype=bool)
    tr[:cut] = True
    ho = ~tr

    kr, mr = float(pol["row_k"]), float(pol["row_m"])
    ke, me = float(pol["era_k"]), float(pol["era_m"])
    m_live = me if 0 < me < 999.0 else mr

    out = {
        "cell": f"{asset}|{w_min}", "asset": asset, "w_min": w_min,
        "side": side_name, "n": int(n), "n_train": int(cut),
        "n_holdout": int(n - cut), "rt_bps": round(rt * 1e4, 3),
        "certified_gross": (pol.get("full") or {}).get("row_wr"),
        "certified_net_field": (pol.get("full") or {}).get("row_wr_net"),
        "live_geometry": {"k": kr, "m": m_live,
                          "guarded_era": bool(0 < me < 999.0)},
    }

    # ── 1. what the app actually trades, whole sample
    sl_live, tp_live = m_live * scale, kr * m_live * scale
    p_live, _ = simulate(side, entry, sl_live, tp_live, ie, ix, h, l, c)
    risk_live = sl_live / entry
    out["traded_geometry"] = {
        "sl_bps_median": round(float(np.median(risk_live)) * 1e4, 1),
        "tp_bps_median": round(float(np.median(tp_live / entry)) * 1e4, 1),
        **stats(p_live, risk_live, rt),
    }
    # directional skill of the certified side (the thing that bounds EV)
    mv = (c[ix] - c[ie]) / entry
    out["side_accuracy"] = {
        "gross_sign_accuracy": round(float(((mv * side) > 0).mean()), 4),
        "holdout_sign_accuracy": round(
            float(((mv[ho] * side[ho]) > 0).mean()), 4),
    }

    # ── 2. geometry candidate, selected on TRAIN only
    grid = []
    for k in (0.15, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0):
        for m in (0.45, 0.55, 0.75, 1.0, 1.5, 2.0, 3.0):
            sl_px, tp_px = m * scale, k * m * scale
            p, _ = simulate(side, entry, sl_px, tp_px, ie, ix, h, l, c)
            risk = sl_px / entry
            s_tr = stats(p[tr], risk[tr], rt)
            s_ho = stats(p[ho], risk[ho], rt)
            grid.append({
                "k": k, "m": m,
                "tp_bps": round(float(np.median(tp_px / entry)) * 1e4, 1),
                "sl_bps": round(float(np.median(risk)) * 1e4, 1),
                # is a "win" actually a win? TP must clear the round trip
                "tp_over_rt": round(float(np.median(tp_px / entry)) * 1e4
                                    / (rt * 1e4), 2),
                "train_net_wr": s_tr["net_wr"],
                "train_ev_net_R": s_tr["ev_net_R"],
                "holdout_net_wr": s_ho["net_wr"],
                "holdout_ev_net_R": s_ho["ev_net_R"],
                "holdout_pf_net": s_ho["pf_net"],
            })
    # selection: highest train net win rate subject to positive train EV
    elig = [g for g in grid if g["train_ev_net_R"] > 0]
    pick = None
    if elig:
        pick = max(elig, key=lambda g: (g["train_net_wr"], g["train_ev_net_R"]))
    ev_best = max(grid, key=lambda g: g["train_ev_net_R"])
    out["selected_on_train"] = pick
    out["max_train_ev"] = ev_best
    out["grid"] = grid
    out["ev_span_R"] = round(max(g["train_ev_net_R"] for g in grid)
                             - min(g["train_ev_net_R"] for g in grid), 4)
    return out


def main():
    art = json.load(open(ART))
    rows = []
    for key in sorted(art):
        if key == "meta":
            continue
        c = art[key]
        if c.get("status") != "ok":
            continue
        asset, tfmin = key.split("|")
        if not os.path.exists(f"backtest/cache/{asset}_1h.csv"):
            continue
        r = cell(asset, int(tfmin), c["selected"]["side"], c["selected"])
        if r is None:
            print(f"{key}: cache too short")
            continue
        rows.append(r)
        t, a = r["traded_geometry"], r["side_accuracy"]
        print(f"\n=== {r['cell']:9s} side={r['side']:11s} n={r['n']} "
              f"(train {r['n_train']}/holdout {r['n_holdout']}) RT={r['rt_bps']}bps")
        print(f"  certified promise   : gross {((r['certified_gross'] or 0)*100):5.1f}%"
              f"   shipped row_wr_net field {r['certified_net_field']}")
        print(f"  side sign accuracy  : all {a['gross_sign_accuracy']*100:5.1f}%"
              f"  holdout {a['holdout_sign_accuracy']*100:5.1f}%")
        print(f"  TRADED k={r['live_geometry']['k']} m={r['live_geometry']['m']}"
              f"  SL {t['sl_bps_median']:.0f}bps TP {t['tp_bps_median']:.1f}bps")
        print(f"    gross win {t['gross_wr']*100:5.1f}%  NET win {t['net_wr']*100:5.1f}%"
              f"  mean NET {t['mean_net_bps']:+7.2f}bps  EV {t['ev_net_R']:+.4f}R")
        if r["selected_on_train"]:
            g = r["selected_on_train"]
            print(f"  train-selected k={g['k']} m={g['m']} (TP {g['tp_bps']:.0f}bps"
                  f" = {g['tp_over_rt']}x RT)")
            print(f"    TRAIN net WR {g['train_net_wr']*100:5.1f}%  EV {g['train_ev_net_R']:+.4f}R")
            print(f"    HOLDOUT net WR {g['holdout_net_wr']*100:5.1f}%"
                  f"  EV {g['holdout_ev_net_R']:+.4f}R  PF {g['holdout_pf_net']}")
        print(f"  EV range over the whole geometry grid: {r['ev_span_R']:+.4f}R"
              f" (a barrier rule reshapes outcomes; it cannot create edge)")
    json.dump(rows, open(OUT, "w"), indent=1)
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
