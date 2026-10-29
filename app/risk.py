"""Risk system — Formula 28 (Adaptive Risk Calibration, ARC) and Formula 29
(Dynamic Level Placement, DLP), exact Part-2 forms.

F28 ARC (risk ratio — a separate formula from the CCI sum):

    ARC = clamp( 1.5 * (1 + |CCI|)
                   * 1/(1 + sigma_current / sigma_baseline)
                   * (0.8 + 0.4 * |brain_consensus|),  1, 3 )

    ARC < 1.3            -> 1:1
    1.3 <= ARC < 2.0     -> 3:2
    ARC >= 2.0           -> 3:1
    confidence < 0.35    -> NO TRADE (ratio None)

F29 DLP:

    TP = price + dir * ATR * R_ratio * Phi_tp
    Phi_tp = 1 + 0.3 * FBPS_alignment_tp + 0.2 * min(1, DRHE)
    SL = price - dir * ATR * 1 * Phi_sl          (SL always = 1 ATR * Phi_sl)
    Phi_sl = 1 + 0.3 * FBPS_alignment_sl - 0.2 * OFTI

FBPS alignment thresholds (spec leaves the quantification open):
    TP side: boundary farther than 1.05x raw -> +1 (unobstructed);
             0.5x..1.05x -> 0; closer than 0.5x -> -0.5 (pulls TP inward)
    SL side: boundary in [1.0, 1.5]x raw beyond SL -> +1 (fresher protection);
             boundary between price and SL -> -0.3; else -> 0
OFTI is the Part-3 seam (app.orderflow); cyc.sl_tighten (SAFE protocol x0.85)
is applied on top of Phi_sl.
"""
from __future__ import annotations

import math

from .config import CONFIDENCE_FLOOR

_EPS = 1e-12


def arc_ratio(
    cci_value: float,
    sigma_current: float,
    sigma_baseline: float,
    brain_consensus_strength: float,
    confidence: float | None = None,
) -> tuple[float, float | None]:
    """Formula 28 — returns (ARC value clamped to [1,3], TP:SL ratio)."""
    if confidence is not None and confidence < CONFIDENCE_FLOOR:
        return 1.0, None                      # NO TRADE
    certainty_mult = 1.0 + abs(cci_value)
    vol_state = 1.0 / (1.0 + sigma_current / max(sigma_baseline, _EPS))
    brain_agreement = 0.8 + 0.4 * max(0.0, min(1.0, brain_consensus_strength))
    arc = 1.5 * certainty_mult * vol_state * brain_agreement
    arc = max(1.0, min(3.0, arc))
    if arc < 1.3:
        ratio = 1.0                           # 1:1
    elif arc < 2.0:
        ratio = 1.5                           # 3:2
    else:
        ratio = 3.0                           # 3:1
    return arc, ratio


def dynamic_level_placement(
    price: float,
    direction: int,
    ratio: float,
    atr: float,
    boundaries: tuple[list[float], list[float]],
    drhe: float,
    ofti_value: float,
    tighten: float = 1.0,
) -> tuple[float, float, float, float]:
    """Formula 29 — returns (tp_price, sl_price, tp_dist, sl_dist).

    `boundaries` = (below, above) fractal-boundary level lists from
    fractal_boundary_levels(); the nearest level on each relevant side is
    used for the FBPS alignment terms.
    """
    if direction == 0 or atr <= 0:
        direction = direction or 1
        atr = max(atr, price * 1e-4 if price else 1e-4)
    below, above = boundaries

    def _nearest(cands: list[float]) -> float | None:
        if not cands:
            return None
        return min(cands, key=lambda x: abs(x - price))

    # ---- TP side ---------------------------------------------------------
    raw_tp = atr * ratio
    tp_boundary = _nearest(above if direction > 0 else below)
    if tp_boundary is None:
        fbps_tp = 1.0                          # farther than window extreme
    else:
        d = abs(tp_boundary - price)
        if d > 1.05 * raw_tp:
            fbps_tp = 1.0
        elif d >= 0.5 * raw_tp:
            fbps_tp = 0.0
        else:
            fbps_tp = -0.5
    phi_tp = 1.0 + 0.3 * fbps_tp + 0.2 * min(1.0, max(0.0, drhe))
    tp_dist = atr * ratio * phi_tp

    # ---- SL side ---------------------------------------------------------
    raw_sl = atr * 1.0
    sl_boundary = _nearest(below if direction > 0 else above)
    if sl_boundary is None:
        fbps_sl = 0.0
    else:
        d = abs(sl_boundary - price)
        if d < raw_sl:                         # boundary between price and SL
            fbps_sl = -0.3
        elif d <= 1.5 * raw_sl:                # fresh barrier just beyond SL
            fbps_sl = 1.0
        else:
            fbps_sl = 0.0
    phi_sl = 1.0 + 0.3 * fbps_sl - 0.2 * max(0.0, min(1.0, ofti_value))
    sl_dist = atr * 1.0 * phi_sl * tighten

    tp = price + direction * tp_dist
    sl = price - direction * sl_dist
    if sl <= 0:                                # never emit a non-positive SL
        sl = price * 0.5 if price else 0.0
        sl_dist = abs(price - sl)
    return tp, sl, tp_dist, sl_dist
