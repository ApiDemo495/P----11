"""Dynamic horizon / friction-clearing geometry.

The deep walk-forward (`backtest/results/ALPHA_LAB_DEEP.md`) made one thing
unambiguous: the *direction* module can be right and the lane can still be
untradable, because a window whose typical excursion σ is smaller than a
few multiples of the round trip cannot pay for itself at ANY hit rate.

Measured on real sub-hour data (2026-01 → 08, BTC/ETH/BNB, 12 bp round trip,
adaptive friction from Kyle's λ):

    TF1 (≈20 min)   σ ≈ 22 bp   friction ≈ 15 bp   tradable   1 %
    TF2 (≈45 min)   σ ≈ 34 bp   friction ≈ 15 bp   tradable  13 %
    TF3 (≈135 min)  σ ≈ 60 bp   friction ≈ 17 bp   tradable  64 %
    TF4 (≈240 min)  σ ≈ 81 bp   friction ≈ 23 bp   tradable  75 %

So the honest architecture is: *choose the horizon where the move clears
the cost*, and tell the user when it does not, instead of dressing a
20-minute coin flip as a signal.

This module is the live implementation of that rule. It takes the lane's
realised σ and the measured friction and decides:

    "tradable"      σ ≥ 3 × friction          → normal issuance
    "marginal"      2 × friction ≤ σ < 3 ×    → watch only, say why
    "untradable"    σ < 2 × friction          → do not present as a trade
"""
from __future__ import annotations

import math

GATE_MULT = 3.0        # σ must clear 3× the round trip
MARGINAL_MULT = 2.0
# §18.1 REAL round trip per asset (commission 10 + slippage +
# half-Roll spread, both sides) — the tradability gate charges the
# actual Binance cost, not a legacy 12/18 placeholder.
ASSET_RT_BPS = {"BTC": 28.872, "ETH": 29.808, "BNB": 30.963,
                "PAXG": 37.400}
DEFAULT_FRICTION_BPS = 28.872          # BTC RT — legacy fallback only


def asset_rt(symbol: str) -> float:
    """The asset's full round trip (bps) — §18.1 cost stack."""
    try:
        return float(ASSET_RT_BPS.get(symbol, DEFAULT_FRICTION_BPS))
    except (TypeError, ValueError):
        return DEFAULT_FRICTION_BPS


def classify(sigma_bps: float | None, friction_bps: float | None = None,
             gate_mult: float = GATE_MULT) -> dict:
    """Return the lane's tradability state with the arithmetic shown."""
    try:
        sig = float(sigma_bps) if sigma_bps is not None else float("nan")
    except (TypeError, ValueError):
        sig = float("nan")
    try:
        fr = float(friction_bps) if friction_bps is not None else DEFAULT_FRICTION_BPS
    except (TypeError, ValueError):
        fr = DEFAULT_FRICTION_BPS
    if not math.isfinite(fr) or fr <= 0:
        fr = DEFAULT_FRICTION_BPS
    if not math.isfinite(sig) or sig <= 0:
        return {"state": "unknown", "sigma_bps": None, "friction_bps": rounded(fr),
                "ratio": None, "required_bps": rounded(gate_mult * fr),
                "note": "no realised σ yet"}
    ratio = sig / fr
    if ratio >= gate_mult:
        state, note = "tradable", (f"σ {sig:.1f}bp clears {gate_mult:g}× friction "
                                   f"({gate_mult * fr:.1f}bp)")
    elif ratio >= MARGINAL_MULT:
        state, note = "marginal", (f"σ {sig:.1f}bp is only {ratio:.1f}× friction — "
                                   f"watch only, the round trip eats the edge")
    else:
        state, note = "untradable", (f"σ {sig:.1f}bp < {MARGINAL_MULT:g}× friction "
                                     f"({MARGINAL_MULT * fr:.1f}bp): this horizon "
                                     f"cannot pay for itself")
    return {"state": state, "sigma_bps": rounded(sig),
            "friction_bps": rounded(fr), "ratio": rounded(ratio),
            "required_bps": rounded(gate_mult * fr), "note": note}


def rounded(v: float, n: int = 2) -> float:
    return round(float(v), n)


# Measured viability table (deep walk-forward, 12 bp floor). Used by the UI
# to label a timeframe before any live σ is available, and as a prior for
# the "which horizons should I trust" explanation.
MEASURED = {
    "TF1": {"sigma_bps": 22.5, "friction_bps": 14.9, "tradable_share": 0.01},
    "TF2": {"sigma_bps": 33.8, "friction_bps": 14.9, "tradable_share": 0.13},
    "TF3": {"sigma_bps": 59.8, "friction_bps": 17.4, "tradable_share": 0.64},
    "TF4": {"sigma_bps": 81.1, "friction_bps": 23.3, "tradable_share": 0.75},
}


def prior(tf: str) -> dict:
    """The measured prior for a timeframe (BTC, 2026-01..08)."""
    m = MEASURED.get(str(tf).upper())
    if not m:
        return {"state": "unknown", "note": "unmeasured timeframe"}
    c = classify(m["sigma_bps"], m["friction_bps"])
    return {**c, "measured_tradable_share": m["tradable_share"],
            "source": "deep walk-forward 2026-01..08"}
