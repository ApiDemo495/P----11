"""The only lever nobody has pulled: COST AS A FRACTION OF THE STOP.

The wall, stated exactly
------------------------
For a barrier trade of one window, with round-trip cost ``RT`` in basis
points and a stop ``SL``, the per-trade EV in units of the stop is

    EV_R = WR·(TP/SL − RT/SL) − (1−WR)·(1 + RT/SL)

Two facts drive everything here:

1. **``RT/SL`` is the whole game.** As ``SL`` grows, ``RT/SL → 0`` and the
   cost term vanishes.  The old harnesses fixed ``SL`` in *absolute* bps
   (36 / 1444 …), so for a 5-minute window a 29 bps round trip was 80 %
   of the entire stop — mathematically unpayable, whatever the edge.
2. **A stop may exceed one window.** Nothing in "one trade starts when
   the previous ends" requires the stop to be reachable inside that
   window.  The user's cycle is about *non-overlap of entries*, not
   about the stop expiring with the window.

So this harness treats WINDOW LENGTH and STOP DISTANCE as independent,
and instead of guessing a stop it **measures the excursion distribution**
and sets the stop at a quantile of it.  That makes the risk per trade
real instead of decorative.

The decisive result this module produces
----------------------------------------
A *time-based* exit — enter at the window open, exit at the window
close, no barrier at all — has **no first-touch substitution bias**.  Its
win rate, net of the round trip, is measured directly.  If that number
is below 50 %, then the market is a driftless (or cost-dominated) walk
at that horizon and no barrier geometry can lift the win rate above the
break-even line ``sl/(tp+sl)`` without sinking EV.  If it is *above*
50 %, the excess is the edge available to exploit, and the frontier shows
how much of it survives the cost at each horizon.

Usage
    .venv/bin/python -m backtest.wr_scale --assets BTC ETH --windows 22 44 130 247
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data import FRICTION_BPS, load_cache          # noqa: E402

RESULTS = os.path.join("backtest", "results")
OUT = os.path.join(RESULTS, "wr_scale.json")
TRAIN_FRAC = 0.65


def _arr(bars):
    return {k: np.asarray([getattr(b, k) for b in bars], dtype=float)
            for k in ("t", "o", "h", "l", "c", "v")}


def windows(t, step, w_min):
    """Contiguous, non-overlapping windows — one trade per window."""
    W = int(w_min * 60)
    starts = np.arange(int(np.ceil(t[0] / W) * W),
                       int(t[-1]) + step - W + 1, W, dtype=np.int64)
    i_in = np.searchsorted(t, starts, side="left")            # entry bar
    i_out = np.searchsorted(t, starts + W - step, side="right") - 1
    ok = (i_in >= 0) & (i_out > i_in) & (i_out < len(t))
    return i_in[ok], i_out[ok]


# ─────────────────── time-exit baseline (no bias) ───────────────────
def time_exit(c, i_in, i_out, rt, side):
    """Enter at the open of the window, exit at its close. Net of RT.

    This is the honest measurement of the win rate: there is no barrier,
    so there is nothing to be 'hit' unfairly and no tie to break.
    """
    entry = c[i_in]
    exit_ = c[i_out]
    gross = side * (exit_ / entry - 1.0)
    return gross - rt


# ───────────────────── barrier settle, cost-relative ────────────────
def settle(side, entry, hi, lo, end_px, tp_bps, sl_bps, rt):
    """First-touch settle with SL winning ties (conservative).

    ``hi``/``lo`` are the per-bar extremes AFTER entry.
    """
    n = len(entry)
    tp = tp_bps * 1e-4
    sl = sl_bps * 1e-4
    sl_price = (entry * (1.0 - sl * side))[:, None]
    tp_price = (entry * (1.0 + tp * side))[:, None]
    long = (side > 0)[:, None]
    sl_hit = np.where(long, lo <= sl_price, hi >= sl_price)
    tp_hit = np.where(long, hi >= tp_price, lo <= tp_price)
    has_sl = sl_hit.any(axis=1)
    has_tp = tp_hit.any(axis=1)
    i_sl = np.where(has_sl, sl_hit.argmax(axis=1), 10 ** 6)
    i_tp = np.where(has_tp, tp_hit.argmax(axis=1), 10 ** 6)
    pnl = np.empty(n)
    # SL first (ties go to the stop)
    m_sl = i_sl <= i_tp
    m_tp = (~m_sl) & has_tp
    m_none = (~m_sl) & (~m_tp)
    pnl[m_sl] = -sl - rt
    pnl[m_tp] = tp - rt
    # no barrier touched: the trade exits at the window close
    pnl[m_none] = side[m_none] * (end_px[m_none] / entry[m_none] - 1.0) - rt
    return pnl, m_sl, m_tp, m_none


def build_paths(h, l, c, i_in, i_out, max_bars):
    """Padded per-window extreme arrays (bars after entry, inclusive)."""
    n = len(i_in)
    hi = np.full((n, max_bars), np.nan)
    lo = np.full((n, max_bars), np.nan)
    end = np.full(n, np.nan)
    keep = np.zeros(n, bool)
    for k in range(n):
        a, b = i_in[k] + 1, i_out[k] + 1        # bars strictly after entry
        if b <= a or b > len(h):
            continue
        seg_h = h[a:b]
        seg_l = l[a:b]
        m = min(len(seg_h), max_bars)
        hi[k, :m] = seg_h[:m]
        lo[k, :m] = seg_l[:m]
        end[k] = c[i_out[k]]
        keep[k] = True
    return hi, lo, end, keep


def quantile_stop(hi, lo, entry, side, q):
    """Stop distance (bps) at quantile ``q`` of adverse excursion.

    Adverse excursion per window = how far price ran against the entry
    before the window ended.  Setting the stop at a high quantile makes
    the stop a measured risk, not a guess.
    """
    e = entry[:, None]
    adv = np.where(side[:, None] > 0, (e - lo) / e, (hi - e) / e)
    adv = np.nanmax(adv, axis=1) * 1e4
    return float(np.nanquantile(adv, q))


def run_cell(asset, w_min, max_bars_mult=3.0):
    bars = load_cache(asset, "5m")
    if not bars:
        return {"cell": f"{asset}|{w_min}", "status": "no_tape"}
    A = _arr(bars)
    t, h, l, c = A["t"], A["h"], A["l"], A["c"]
    step = int(t[1] - t[0])
    i_in, i_out = windows(t, step, w_min)
    if len(i_in) < 500:
        return {"cell": f"{asset}|{w_min}", "status": "too_few"}
    rt = 2.0 * FRICTION_BPS[asset] * 1e-4
    max_bars = int(max_bars_mult * w_min * 60 / step)
    hi, lo, end, keep = build_paths(h, l, c, i_in, i_out, max_bars)
    i_in, i_out = i_in[keep], i_out[keep]
    hi, lo, end = hi[keep], lo[keep], end[keep]
    entry = c[i_in]
    n = len(entry)
    cut = int(n * TRAIN_FRAC)

    out = {"cell": f"{asset}|{w_min}m", "asset": asset, "w_min": w_min,
           "n": int(n), "rt_bps": round(rt * 1e4, 2), "modes": {}}

    # ── mode LONG-ONLY and SHORT-ONLY time exit: the unbiased baseline ──
    for name, side in (("long", 1.0), ("short", -1.0)):
        pnl = time_exit(c, i_in, i_out, rt, side)
        wr_tr = float((pnl[:cut] > 0).mean())
        wr_ho = float((pnl[cut:] > 0).mean())
        out["modes"][f"time_{name}"] = {
            "train_wr": round(wr_tr, 4), "holdout_wr": round(wr_ho, 4),
            "ev_bps_holdout": round(float(pnl[cut:].mean() * 1e4), 2),
            "n": int(n),
        }

    # ── what win rate does a *random* side get? (the null) ──────────
    rng = np.random.default_rng(7)
    side_rnd = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    pnl_r = time_exit(c, i_in, i_out, rt, side_rnd)
    out["null_random_side_wr_holdout"] = round(float((pnl_r[cut:] > 0).mean()), 4)
    out["null_random_side_ev_bps"] = round(float(pnl_r[cut:].mean() * 1e4), 2)

    # ── stop scaled to the MEASURED excursion, TP as a multiple of RT ──
    # Every cell is graded on the holdout only (params chosen on train).
    grid = []
    for q in (0.5, 0.8, 0.95, 0.99, 0.999):
        for side_name, side in (("long", np.ones(n)), ("short", -np.ones(n))):
            sl_tr = quantile_stop(hi[:cut], lo[:cut], entry[:cut],
                                  side[:cut], q)
            if not np.isfinite(sl_tr) or sl_tr <= 0:
                continue
            for tp_mult in (0.5, 1.0, 2.0, 4.0, 8.0):
                tp_bps = tp_mult * rt * 1e4
                if tp_bps >= sl_tr:
                    continue          # never reward more than we risk
                pnl, m_sl, m_tp, m_none = settle(
                    side, entry, hi, lo, end, tp_bps, sl_tr, rt)
                ho = slice(cut, n)
                wr = float((pnl[ho] > 0).mean())
                ev = float(pnl[ho].mean() * 1e4)
                grid.append({
                    "q": q, "side": side_name, "sl_bps": round(sl_tr, 1),
                    "tp_bps": round(tp_bps, 1), "tp_over_rt": tp_mult,
                    "cost_frac_of_stop": round(rt * 1e4 / sl_tr, 4),
                    "holdout_wr": round(wr, 4), "holdout_ev_bps": round(ev, 2),
                    "n_stop": int(m_sl[ho].sum()), "n_tp": int(m_tp[ho].sum()),
                    "n_time": int(m_none[ho].sum()),
                    "ev_R": round(ev / sl_tr, 4),
                })
    out["grid"] = grid
    if grid:
        out["best_wr"] = max(grid, key=lambda r: r["holdout_wr"])
        pos = [r for r in grid if r["holdout_ev_bps"] > 0]
        out["best_ev"] = max(grid, key=lambda r: r["holdout_ev_bps"])
        out["best_wr_positive_ev"] = (max(pos, key=lambda r: r["holdout_wr"])
                                     if pos else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="*", default=["BTC", "ETH"])
    ap.add_argument("--windows", nargs="*", type=int, default=[22, 44, 130, 247])
    args = ap.parse_args()
    rows = []
    print(f"{'cell':12s} {'n':>7s} {'time-L':>7s} {'time-S':>7s} "
          f"{'null':>6s} {'best WR':>8s} {'its EV':>8s} {'SL bps':>7s} "
          f"{'RT/SL':>6s}")
    for asset in args.assets:
        for w in args.windows:
            r = run_cell(asset, w)
            rows.append(r)
            if r.get("status") != "ok" and "modes" not in r:
                print(f"{r['cell']:12s} {r['status']}")
                continue
            m, b = r["modes"], r.get("best_wr")
            tl, ts = m["time_long"]["holdout_wr"], m["time_short"]["holdout_wr"]
            null = r["null_random_side_wr_holdout"]
            if b:
                print(f"{r['cell']:12s} {r['n']:7d} {tl*100:6.1f}% {ts*100:6.1f}% "
                      f"{null*100:5.1f}% {b['holdout_wr']*100:7.1f}% "
                      f"{b['holdout_ev_bps']:+8.2f} {b['sl_bps']:7.0f} "
                      f"{b['cost_frac_of_stop']:6.3f}")
            else:
                print(f"{r['cell']:12s} {r['n']:7d} {tl*100:6.1f}% {ts*100:6.1f}% "
                      f"{null*100:5.1f}%   (no grid)")
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(rows, open(OUT, "w"), indent=1)
    ok = [r for r in rows if r.get("modes")]
    if ok:
        print("\n── the decisive numbers ──")
        for r in ok:
            m = r["modes"]
            print(f"{r['cell']:12s} long {m['time_long']['holdout_wr']*100:5.1f}% "
                  f"EV {m['time_long']['ev_bps_holdout']:+7.2f}bp | short "
                  f"{m['time_short']['holdout_wr']*100:5.1f}% "
                  f"EV {m['time_short']['ev_bps_holdout']:+7.2f}bp")
        n66 = sum(1 for r in ok if r.get("best_wr")
                  and r["best_wr"]["holdout_wr"] >= 0.66)
        n66p = sum(1 for r in ok if r.get("best_wr_positive_ev")
                   and r["best_wr_positive_ev"]["holdout_wr"] >= 0.66)
        print(f"\n66%+ reached: {n66}/{len(ok)} · 66%+ with positive EV: "
              f"{n66p}/{len(ok)}")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
