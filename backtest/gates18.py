"""§18 ship gates — evaluated ONCE, after the single v9 run.

Reads the shipped backtest/results/hold_wr.json and prints, per §18.3:

  (i)  every gradeable cell: full gross row WR >= 0.66 AND guarded
       era EV > 0 under the FULL Binance stack
  (ii) mean full row_wr_net over gradeable cells > the pre-registered
       v8 reference (backtest/results/v8_net_reference.json)
  (iii) >= 4 cells HOLD-OUT VERIFIED (train-selected side + guarded
       era both positive on the holdout under the real stack)

Nothing here selects anything — selection happened on train inside
hold.grade_cell; this only reads the artifact.  Exit 0 = all gates
pass (v9 ships), exit 1 = v8 stays (honest fallback).
"""
from __future__ import annotations

import json
import os
import sys

RESULTS = os.path.join(os.path.dirname(__file__), "results")
MIN_VERIFIED = 4                    # §18.3(iii) — set from train probes


def main() -> int:
    path = os.path.join(RESULTS, "hold_wr.json")
    art = json.load(open(path))
    ref = json.load(open(os.path.join(RESULTS, "v8_net_reference.json")))

    cells = {k: v for k, v in art.items()
             if k != "meta" and isinstance(v, dict)}
    gradeable, gate1_ok = [], True
    print("=== §18 gate (i): real-trade viability (FULL period) ===")
    for key in sorted(cells):
        c = cells[key]
        if c.get("status") not in ("ok", "no_feasible_config"):
            continue
        if c.get("status") == "no_feasible_config":
            print(f"  {key:10s} no_feasible (honest no-trade)")
            continue
        gradeable.append(key)
        sel = c.get("selected") or {}
        full = sel.get("full") or {}
        row = full.get("row_wr")
        ev = full.get("era_ev_net")
        rt = c.get("round_trip_bps")
        ok = bool(row is not None and row >= 0.66
                  and ev is not None and ev > 0)
        gate1_ok = gate1_ok and ok
        print(f"  {key:10s} row_full={row} era_full_ev={ev} "
              f"net_wr={full.get('row_wr_net')} RT={rt} "
              f"-> {'PASS' if ok else 'FAIL'}")

    nets = []
    for key in gradeable:
        sel = cells[key].get("selected") or {}
        n = (sel.get("full") or {}).get("row_wr_net")
        if n is not None:
            nets.append(float(n))
    mean_net = sum(nets) / len(nets) if nets else 0.0
    print(f"\n=== §18 gate (ii): mean row_wr_net = {mean_net:.4f} "
          f"> v8 ref {ref['mean_row_wr_net']:.4f} ?")
    gate2_ok = bool(nets) and mean_net > ref["mean_row_wr_net"]
    print(f"  -> {'PASS' if gate2_ok else 'FAIL'}")

    # (iii) HOLD-OUT VERIFIED = the shipped hold_policy tier rule
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from app import hold_policy as hp
    tiers = {}
    for key in gradeable:
        asset, tfn = key.split("|")
        tf = {"22": "tf1", "44": "tf2", "130": "tf3",
              "247": "tf4"}[tfn]
        t = hp.tier(asset, tf)
        tiers[key] = t
        print(f"  {key:10s} tier = {t}")
    verified = sum(1 for t in tiers.values() if t == hp.TIER_HOLDOUT)
    print(f"\n=== §18 gate (iii): HOLD-OUT VERIFIED = {verified} "
          f"(need >= {MIN_VERIFIED})")
    gate3_ok = verified >= MIN_VERIFIED
    print(f"  -> {'PASS' if gate3_ok else 'FAIL'}")

    all_ok = gate1_ok and gate2_ok and gate3_ok
    print(f"\n=== §18 V9 SHIP = {'PASS' if all_ok else 'FAIL'} ===")
    print(f"  (i) {gate1_ok}  (ii) {gate2_ok}  (iii) {gate3_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
