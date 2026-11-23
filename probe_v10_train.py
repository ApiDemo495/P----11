"""§20 TRAIN-ONLY probe — prints train-side evidence for the v10
design (TSMOM gate + maximin + row buffer + era margin).

NEVER prints holdout/full fields: grade_cell computes them internally
(they are evaluated exactly once later, by gates18.py, after the
official run) — this probe only reports the train aggregates so the
pre-registration can state what is knowable before the shot.
"""
from __future__ import annotations

import sys
import time

from backtest.continuous import load_bars
from backtest.hold import grade_cell


def main() -> int:
    assets = sys.argv[1:] or ["BTC", "ETH", "BNB", "PAXG"]
    for asset in assets:
        bars = load_bars(asset)
        if bars is None:
            print(f"{asset}: cache missing", flush=True)
            continue
        btc = load_bars("BTC") if asset != "BTC" else None
        for w_min in (22, 44, 130, 247):
            t0 = time.time()
            cell = grade_cell(asset, w_min, bars, btc_bars=btc)
            st = cell.get("status")
            base = f"{asset}|{w_min} [{time.time()-t0:.0f}s] {st}"
            if st != "ok":
                print(base, "-", cell.get("why") or cell.get("note", ""),
                      flush=True)
                continue
            s = cell["selected"]
            tr = s["train"]
            print(base,
                  f"side={s['side']}",
                  f"row(k={s['row_k']},m={s['row_m']})",
                  f"era(k={s['era_k']},m={s['era_m']},gate={s['era_gate_l']})",
                  f"| TRAIN rowWR={tr['row_wr']} wilson={tr['row_wilson']}",
                  f"rowWRnet={tr['row_wr_net']}",
                  f"| eraEVnet={tr['era_ev_net']} eraN={tr['era_n']}",
                  f"| worstYear={s['train_worst_year_ev']}",
                  f"pf={s['train_pf']} rung={s['selection_rung']}",
                  f"| feasible={cell.get('n_feasible_train')}",
                  flush=True)
    print("PROBE-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
