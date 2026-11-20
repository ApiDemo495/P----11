"""Volatility-scaled triple barriers, with an explicit friction floor.

Why this module exists: the lab (`backtest/results/ALPHA_LAB.md`) scored
the same signals against the SAME trade geometry the app ships — a fixed
2:1 target/stop in units of realised σ — and against multiples chosen on
each train fold.  Two findings drove the design here:

  1. Barrier geometry matters more than the entry side once friction is
     charged.  At 2 h, a 1 σ excursion is 15–25 bp on BTC/ETH, so a 12 bp
     round trip eats half of a 1 σ target.  Any barrier closer than
     `FRICTION_MULT ×` friction is a guaranteed loser no matter how good
     the direction call is.
  2. The fitted multiples drift with regime (k_pt 1.0–2.5), so the app
     should not hard-code a single pair forever; it should scale by the
     measured σ of the lane and enforce the friction floor.

Live use is deliberately conservative: `vol_scaled()` returns the barrier
distances, the multiples used, and `floor_ok`.  The level contract keeps
its own measured-move ladder (one rung per horizon, frozen at the lock);
this block only (a) supplies the σ anchors, (b) refuses a geometry that
cannot clear friction, and (c) is published as telemetry so the UI can
show *why* a window is watch-only.
"""
from __future__ import annotations

import math

# per-timeframe multiples (k_pt, k_sl) in units of the lane's realised σ.
# Defaults are the medians of the walk-forward fits, widened slightly for
# the longer horizons where σ is larger and the relative cost of the stop
# is smaller.
MULTIPLES = {
    "TF1": (1.5, 1.0),
    "TF2": (1.7, 1.0),
    "TF3": (2.0, 1.1),
    "TF4": (2.2, 1.2),
}
FRICTION_MULT = 3.0      # a TP thinner than 3× friction is not a trade
MAX_SIGMA_MULT = 4.0     # never ask for more than 4 σ in a single window


def friction_floor_bps(friction_bps: float = 12.0) -> float:
    """Minimum TP distance that can survive the round trip."""
    return max(6.0, float(friction_bps) * FRICTION_MULT)


def vol_scaled(sigma_bps: float | None, tf: str = "TF1",
               friction_bps: float = 12.0) -> dict:
    """Barrier distances from the lane's realised σ.

    Returns `{ok, tp_bps, sl_bps, k_pt, k_sl, sigma_bps, floor_bps,
    reason}`.  `ok=False` means the geometry cannot clear friction — the
    caller may still display the numbers, but must not present them as
    tradable.
    """
    k_pt, k_sl = MULTIPLES.get(str(tf).upper(), MULTIPLES["TF1"])
    floor = friction_floor_bps(friction_bps)
    try:
        sig = float(sigma_bps) if sigma_bps is not None else 0.0
    except (TypeError, ValueError):
        sig = 0.0
    if not math.isfinite(sig) or sig <= 0:
        return {"ok": False, "tp_bps": None, "sl_bps": None, "k_pt": k_pt,
                "k_sl": k_sl, "sigma_bps": None, "floor_bps": round(floor, 2),
                "reason": "no realised σ yet"}
    sig = min(sig, 400.0)
    tp = k_pt * sig
    sl = k_sl * sig
    if tp > MAX_SIGMA_MULT * sig:      # belt and braces on the cap
        tp = MAX_SIGMA_MULT * sig
    ok = tp >= floor
    return {
        "ok": ok,
        "tp_bps": round(tp, 2),
        "sl_bps": round(sl, 2),
        "ratio": round(tp / sl, 2) if sl > 0 else None,
        "k_pt": k_pt, "k_sl": k_sl,
        "sigma_bps": round(sig, 2),
        "floor_bps": round(floor, 2),
        "reason": ("clears friction" if ok else
                   f"target {tp:.1f}bp < friction floor {floor:.1f}bp — "
                   f"watch-only, the round trip eats it"),
    }


def levels(entry: float, direction: int, sigma_bps: float, tf: str = "TF1",
           friction_bps: float = 12.0) -> dict:
    """Concrete TP/SL prices around `entry` for `direction`."""
    block = vol_scaled(sigma_bps, tf=tf, friction_bps=friction_bps)
    try:
        e = float(entry)
        d = 1 if int(direction) > 0 else (-1 if int(direction) < 0 else 0)
    except (TypeError, ValueError):
        return {**block, "entry": None, "tp": None, "sl": None}
    if not (e > 0) or d == 0 or block["tp_bps"] is None:
        return {**block, "entry": e if e > 0 else None, "tp": None, "sl": None}
    tp = e * (1 + d * block["tp_bps"] / 1e4)
    sl = e * (1 - d * block["sl_bps"] / 1e4)
    return {**block, "entry": e, "tp": round(tp, 6), "sl": round(sl, 6)}
