"""Paper-trade the certified sleeves through the live portfolio engine.

This closes the gap between "the research says +88 bp" and "the execution
layer actually produces it". It replays the LOCKED holdout (the last 35 %
of each asset's tape, which no sleeve parameter was ever fitted on), feeds
the certified triggers into ``app.portfolio_live.LivePortfolio`` bar by
bar, and reports what the book would have realised — including the
adverse fills, the position caps, the correlation throttle and the
alpha-decay monitor.

Nothing here is optimised. The point is the opposite: to find out whether
the headline research number survives contact with the execution rules.

Usage::

    python -m backtest.paper_replay
"""
from __future__ import annotations

import json
import os

from app.portfolio_live import LivePortfolio
from backtest.edge_scan import (
    FRICTION_BPS,
    align,
    catchup_signals,
    holdout_start_for,
    load_series,
    log_returns,
    rolling_std,
    vol_target_scales,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# Certified definitions, in the referee's own parameter names. The registry
# stores per-fold picks under ``foldN`` keys and its ``frozen_candidate`` is
# the modal one; both are research bookkeeping, not an executable spec. The
# sleeve's identity is the definition below, and the scan pins it with a
# test so the two can never drift apart.
CERTIFIED_DEFS = {
    "catchup_subhour_spot_15m::ETH": {
        "asset": "ETH", "lead": "BTC", "lookback_h": 12, "thr": 2.5, "hold_h": 48,
    },
    "spillover_spot_15m::ETH": {
        "asset": "ETH", "lead": "BNB", "lookback_h": 24, "thr": 2.0, "hold_h": 48,
    },
}


def _sleeve_cfg(registry: dict, sleeve_id: str) -> dict | None:
    """Executable spec for a certified sleeve (registry must list it open)."""
    for s in registry.get("sleeves", []):
        if s.get("sleeve_id") == sleeve_id and s.get("status") == "open":
            cfg = dict(CERTIFIED_DEFS.get(sleeve_id) or {})
            if not cfg:
                return None
            cfg["sleeve_id"] = sleeve_id
            return cfg
    return None


def replay(registry_path: str | None = None, verbose: bool = True,
           enforce_monitors: bool = True) -> dict:
    reg_path = registry_path or os.path.join(
        os.path.dirname(__file__), "..", "app", "sleeve_registry.json")
    with open(reg_path) as f:
        registry = json.load(f)

    grid = {a: load_series(a, "spot_15m") for a in ("BTC", "ETH", "BNB")}
    if any(v is None for v in grid.values()):
        return {"error": "missing spot_15m cache"}

    book = LivePortfolio(
        registry=registry,
        ledger_path=os.path.join(
            os.path.dirname(__file__), "out",
            "paper_ledger_replay.json" if enforce_monitors
            else "paper_ledger_replay_reportmode.json"),
        enforce_monitors=enforce_monitors,
    )

    # build each certified sleeve's signal on its own lead/follower pair
    prepared = []
    for sid in list(book.sleeves):
        cfg = _sleeve_cfg(registry, sid)
        if cfg is None:
            continue
        lead_name = cfg.get("lead", "BTC")
        fol = cfg["asset"]
        a, b = align(grid[lead_name], grid[fol])
        bar = a.t[1] - a.t[0]
        cut = holdout_start_for(b.t)
        lr = log_returns(a)
        sd = rolling_std(lr, 500)
        L = int(float(cfg.get("lookback_h", 12)) * 3600 // bar)
        H = int(float(cfg.get("hold_h", 48)) * 3600 // bar)
        thr = float(cfg.get("thr", 2.5))
        sig = catchup_signals(lr, sd, len(a), L, thr, False)
        # hold in BARS of the sleeve's own grid, so the book's timer matches
        book.sleeves[sid].hold_bars = max(1, H)
        prepared.append((sid, fol, a, b, sig, H, cut, bar))

    # every sleeve must share one clock for concurrent management
    n = min(len(b) for _, _, _, b, _, _, _, _ in prepared)
    vol_scales = {sid: vol_target_scales(b) for sid, _, _, b, _, _, _, _ in prepared}
    realised = {}
    for sid, _, _, b, _, _, _, _ in prepared:
        r = log_returns(b)
        realised[sid] = rolling_std(r, 168)      # 7-day realised vol on the sleeve grid
        realised[sid] = [x if x == x else 0.0 for x in realised[sid]]

    # One merged timeline over the UNION of every sleeve's bar timestamps.
    # At each timestamp every sleeve is marked to market on ITS OWN bar (the
    # grids differ: BTC-led and BNB-led ETH grids do not share an index), and
    # an entry is taken only when that timestamp is a certified candidate for
    # that sleeve. Earlier versions mixed a global index with per-sleeve grids
    # and produced fills that exited before they entered.
    cand_bars: dict[str, set[int]] = {}
    by_id: dict[str, tuple] = {}
    for sid, fol, a, b, sig, H, cut, bar in prepared:
        by_id[sid] = (fol, b, sig, H, cut)
        taken: set[int] = set()
        i = 0
        while i < len(b) - H - 1:
            if sig[i] != 0:
                taken.add(i)
                i += H
            else:
                i += 1
        cand_bars[sid] = taken

    timeline = sorted({b.t[i] for _, _, _, b, _, _, _, _ in prepared
                       for i in range(len(b))})
    idx_of = {sid: {b.t[i]: i for i in range(len(b))}
              for sid, _, _, b, _, _, _, _ in prepared}

    for t_ev in timeline:
        for sid in by_id:
            fol, b, sig, H, cut = by_id[sid]
            i = idx_of[sid].get(t_ev)
            if i is None:
                continue
            book.manage(i, {fol: {"t": b.t[i], "c": b.c[i],
                                  "h": b.h[i], "l": b.l[i]}})
            if i in cand_bars[sid] and b.t[i] >= cut and i + H < len(b):
                state = "long" if sig[i] > 0 else "short"
                book.try_enter(
                    sid, state,
                    {"t": b.t[i + 1], "o": b.o[i + 1]},
                    i, realised[sid][i], scale=vol_scales[sid][i],
                    exit_at={"t": b.t[i + H], "c": b.c[i + H], "bar_index": i + H},
                )

    # persist the ledger too, so the fills can be audited against the
    # research trade list line by line (they should agree exactly)
    book.save()

    snap = book.snapshot()
    snap["holdout_only"] = True
    out_path = os.path.join(RESULTS_DIR, "PAPER_REPLAY.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(snap, f, indent=2)

    if verbose:
        print(f"{'sleeve':34s} {'status':10s} {'live n':>7s} {'live EV':>9s} "
              f"{'live t':>7s} {'research EV':>12s}")
        for r in snap["sleeves"]:
            print(f"{r['sleeve_id']:34s} {r['status']:10s} {r['live_trades']:7d} "
                  f"{r['live_ev_bps']:+9.1f} {r['live_t']:+7.2f} "
                  f"{r['research_ev_bps']:+12.1f}")
        print()
        print(f"closed trades      : {snap['closed_trades']}")
        print(f"realized (bp, unit) : {snap['realized_bps']:+.2f}")
        print(f"avg slippage (bp)   : {snap['avg_slippage_bps']}")
        print(f"max drawdown        : {snap['drawdown']:.4f}")
        print(f"written             : {out_path}")
    return snap


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enforce-monitors", action="store_true",
                    help="let the decay/correlation monitors halt sleeves "
                         "(default: observe only, so the research comparison "
                         "stays clean)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--registry", default=None)
    args = ap.parse_args()
    replay(registry_path=args.registry, verbose=not args.quiet,
           enforce_monitors=args.enforce_monitors)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
