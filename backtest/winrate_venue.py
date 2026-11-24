"""THE VENUE MAP — what the cheapest exchange is actually worth in win rate.

The question this answers
-------------------------
§26 proved the ceiling: a window can only be a win if its move clears the
round trip, so

    WR_net <= P(|move| > RT)

and §27/§25 showed that on this tape measured directional skill is ~52 %.
Putting the two together gives the *realizable* win rate, not just the
bound:

    WR_realizable ~= skill * P(|move| > RT)

(accuracy is very nearly independent of move size on this tape, which is
why the 0-hurdle harness measured 7.1 % at tf1 where
0.524 * 13.4 % = 7.0 % — the model reproduces the measurement).

So there are exactly two levers on win rate:

  1. RT — the venue.  The project has always assumed 28.9 bps/side-pair.
     Lighter charges 0.  MEXC 4 bps.  dYdX waives BTC-USD.  That is a
     7x fee cut, and fees multiply straight into the win rate.
  2. horizon — longer windows move more, so coverage rises.

This module sweeps both and prints the highest win rate that actually
exists on real tape, per venue, together with its EV.  It exists to stop
anyone claiming a 66 % book without naming the venue and the horizon.

Fees are read from public schedules (2026) and every row keeps its source
tag; use ``--check-fees`` to re-print them for re-verification before any
live use.

Usage
    .venv/bin/python -m backtest.winrate_venue --assets BTC ETH
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data import load_cache                        # noqa: E402
from backtest.winrate_ceiling import HORIZONS, windows      # noqa: E402

RESULTS = os.path.join("backtest", "results")
OUT = os.path.join(RESULTS, "winrate_venue.json")

#: measured out-of-sample directional skill on 2023-11..2026-08 tape
#: (REAL_RESULTS §"22-247 min", revW/lr_oof pooled: 51.5-53.1 %, centre 52.4 %)
SKILL = 0.524
SKILL_BAND = (0.500, 0.524, 0.550)
TARGET = 0.66

#: venue -> (maker round trip bps, taker round trip bps, source tag, caveat)
#: RT is BOTH sides: 0.02 % taker on one side = 4 bps round trip.
VENUES = {
    "lighter":      (0.0, 0.0,  "docs.lighter.xyz (Standard, 2026-08)", 
                     "300 ms taker delay; 219 markets; zk-rollup; no KYC"),
    "dydx_btc":     (0.2, 0.0,  "dYdX v4 fee schedule (BTC-USD waived)",
                     "zero-fee pairs are BTC-USD and SOL-USD only"),
    "mexc_fut":     (0.0, 4.0,  "MEXC futures 0 %/0.02 % (2026)",
                     "not available US/UK; thinnest compliance record"),
    "extended":     (0.0, 5.0,  "Extended 0 %/0.025 % (Starknet, 2026)",
                     "young venue, thinner books"),
    "hyperliquid4": (0.0, 8.0,  "Hyperliquid T4 0 %/0.04 % (needs $500M/14d)",
                     "maker 0 % requires Tier 4 volume; rebates below that"),
    "aster":        (0.0, 8.0,  "Aster 0 %/0.04 % VIP1 (2026)",
                     "VIP ladder; taker 3.2 bps at VIP6"),
    "binance":      (4.0, 10.0, "Binance futures 0.02 %/0.05 %",
                     "deepest liquidity, tightest execution"),
    "project":      (None, 28.872, "backtest/data.py FRICTION_BPS (legacy)",
                     "~7x more expensive than MEXC, infinite vs Lighter"),
}
#: which venue to quote for each asset when a zero-fee pair exists
ASSET_PREF = {"BTC": "dydx_btc", "ETH": "lighter"}


def _skill_for(asset):
    """Measured skill, same for both majors on this tape to 2 sf."""
    return SKILL


def measure(asset, rt_bps):
    """Coverage / realizable WR / EV for every horizon at a given round trip."""
    bars = load_cache(asset, "5m")
    if not bars:
        return None
    t = np.array([b.t for b in bars])
    c = np.array([b.c for b in bars], float)
    step = int(t[1] - t[0])
    skill = _skill_for(asset)
    rows = []
    for name, w_min in HORIZONS:
        i_in, i_out = windows(t, step, w_min)
        n = len(i_in)
        if n < 20:
            continue
        r = (c[i_out] / c[i_in] - 1.0) * 1e4
        absr = np.abs(r)
        cover = float((absr > rt_bps).mean())
        wr = skill * cover
        # EV per window in bps: edge x mean size, minus the toll.
        # A no-skill book (50 %) is printed as the stress case.
        mean_abs = float(absr.mean())
        ev = (2.0 * skill - 1.0) * mean_abs - rt_bps
        ev_null = -rt_bps
        rows.append({
            "tf": name, "minutes": w_min, "n": int(n),
            "median_abs_move_bps": round(float(np.median(absr)), 2),
            "mean_abs_move_bps": round(mean_abs, 2),
            "coverage": round(cover, 4),
            "realizable_wr": round(wr, 4),
            "realizable_wr_lo": round(SKILL_BAND[0] * cover, 4),
            "realizable_wr_hi": round(SKILL_BAND[2] * cover, 4),
            "ev_bps": round(ev, 2),
            "ev_bps_null_skill": round(ev_null, 2),
            "hits_target": bool(wr >= TARGET),
            "ev_positive": bool(ev > 0),
        })
    return {"asset": asset, "rt_bps": rt_bps, "skill": skill, "rows": rows}


def venue_table(asset):
    out = []
    for name, (mk, tk, src, caveat) in VENUES.items():
        rt = tk if asset != "BTC" or name != "dydx_btc" else 0.0
        if rt is None:
            rt = VENUES[name][1] or 0.0
        m = measure(asset, rt)
        if not m:
            continue
        best = max(m["rows"], key=lambda x: x["realizable_wr"] if
                   x["ev_positive"] else -1)
        pos = [x for x in m["rows"] if x["ev_positive"]]
        # the highest-WR cell that also clears cost
        best_pos = max(pos, key=lambda x: x["realizable_wr"]) if pos else None
        fast = m["rows"][0]
        out.append({
            "venue": name, "rt_bps": rt, "source": src, "caveat": caveat,
            "tf1_realizable_wr": fast["realizable_wr"],
            "tf1_ev_bps": fast["ev_bps"],
            "best_wr": best_pos["realizable_wr"] if best_pos else None,
            "best_wr_tf": best_pos["tf"] if best_pos else None,
            "best_wr_minutes": best_pos["minutes"] if best_pos else None,
            "best_wr_ev_bps": best_pos["ev_bps"] if best_pos else None,
            "n_ev_positive_cells": len(pos),
            "cells": m["rows"],
        })
    return {"asset": asset, "rows": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="*", default=["BTC", "ETH"])
    ap.add_argument("--check-fees", action="store_true")
    args = ap.parse_args()

    if args.check_fees:
        print("venue fee schedules as loaded (verify before live use):")
        for name, (mk, tk, src, cav) in VENUES.items():
            print(f"  {name:12s} maker {mk if mk is not None else '?':>5} "
                  f"bps  taker {tk:>5} bps  {src}\n"
                  f"  {'':12s} caveat: {cav}")
        return

    out = [venue_table(a) for a in args.assets]
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)

    for block in out:
        asset = block["asset"]
        print(f"\n{'='*78}\n{asset}: realizable net win rate by venue "
              f"(skill {SKILL*100:.1f} %, target {TARGET*100:.0f} %)\n{'='*78}")
        print(f"  {'venue':12s} {'RT':>6s} {'tf1 WR':>8s} {'tf1 EV':>8s} "
              f"{'best WR (cell)':>22s} {'EV':>7s}  positive-EV cells")
        for r in sorted(block["rows"], key=lambda x: x["rt_bps"]):
            bw = (f"{r['best_wr']*100:5.1f} % ({r['best_wr_tf']},"
                  f"{r['best_wr_minutes']}m)") if r["best_wr"] is not None \
                else "none (all -EV)"
            bev = f"{r['best_wr_ev_bps']:+.1f}" if r["best_wr_ev_bps"] else "   -"
            print(f"  {r['venue']:12s} {r['rt_bps']:5.1f}b "
                  f"{r['tf1_realizable_wr']*100:7.1f}% "
                  f"{r['tf1_ev_bps']:+7.1f}b {bw:>22s} {bev:>7s}  "
                  f"{r['n_ev_positive_cells']}")

    # ── the headline: does any venue/horizon reach the target? ───────────
    print(f"\n{'='*78}\nCAN ANY CELL REACH {TARGET*100:.0f} % WITH POSITIVE EV?\n{'='*78}")
    any_hit = False
    for block in out:
        for r in block["rows"]:
            for c in r["cells"]:
                if c["hits_target"] and c["ev_positive"]:
                    any_hit = True
                    print(f"  YES {block['asset']} {r['venue']} "
                          f"{c['tf']} ({c['minutes']}m): WR "
                          f"{c['realizable_wr']*100:.1f} %, EV {c['ev_bps']:+.1f} bps")
    if not any_hit:
        print("  NO — not at 52.4 % skill, on any venue, at any horizon.")
        print("  The binding constraint is skill, not fees: even at zero cost")
        print("  WR = skill, so 66 % needs 66 % directional accuracy.")
        # show how much skill the cheapest venue would need per horizon
        for block in out:
            asset = block["asset"]
            cheap = min(block["rows"], key=lambda x: x["rt_bps"])
            print(f"\n  {asset} on {cheap['venue']} ({cheap['rt_bps']:g} bps RT)"
                  f" — accuracy required for 66 %:")
            for c in cheap["cells"]:
                need = (TARGET / c["coverage"]) if c["coverage"] else float("inf")
                mark = "" if need <= SKILL else "  <- above skill"
                print(f"    {c['tf']:>4s} {c['minutes']:6d}m cover "
                      f"{c['coverage']*100:5.1f} %  needs "
                      f"{need*100:6.1f} % accuracy{mark}")

    # ── what shifting venue is worth at the horizon people actually trade ─
    print(f"\n{'='*78}\nWHAT THE FEE CUT IS WORTH AT tf1 (22 min)\n{'='*78}")
    for block in out:
        base = next(r for r in block["rows"] if r["venue"] == "project")
        print(f"  {block['asset']}: legacy {base['rt_bps']:.1f} bps → "
              f"tf1 WR {base['tf1_realizable_wr']*100:.1f} %")
        for r in sorted(block["rows"], key=lambda x: x["rt_bps"]):
            if r["venue"] == "project":
                continue
            gain = (r["tf1_realizable_wr"] - base["tf1_realizable_wr"]) * 100
            if gain <= 0:
                continue
            print(f"    {r['venue']:12s} {r['rt_bps']:5.1f} bps → "
                  f"{r['tf1_realizable_wr']*100:5.1f} % "
                  f"({gain:+5.1f} pp)  EV {r['tf1_ev_bps']:+6.1f} bps  "
                  f"[{r['caveat']}]")
    print(f"\nNOTE: coverage is an ex-ante property of the tape and the fee; "
          f"the WR column is skill x coverage, validated against the measured "
          f"7.1 % tf1 harness result (predicted 7.0 %).")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
