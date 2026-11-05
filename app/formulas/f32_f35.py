"""Order-book & liquidation formulas (F32–F35) — added on explicit
user authorization to extend the formula set.

All inputs are MEASURED live signals (DepthBook / forceOrder); all
outputs are bounded to [-1, 1]; every cold-data zero falls through
zero_reason() to the EVENT_REASON texts in trace.py ("no live book /
no live forceOrder prints") — cause-6 honesty, never a silent zero.
"""
from __future__ import annotations

from ..book import DepthBook  # noqa: F401   (docs/typing only)

_BK_MIN = 10           # ring samples before a book formula may speak
_LIQ_MIN = 1           # one real forceOrder print is enough to move it


def _book(ctx) -> dict:
    return getattr(ctx, "book", None) or {}


def _mean_last(ring, n: int) -> float | None:
    if not ring or len(ring) < n:
        return None
    vals = list(ring)[-n:]
    return sum(vals) / len(vals)


def f32_dbi(ctx) -> float:
    """F32 DBI — order-book depth imbalance (top-20), mean over the
    available window: (Σbid − Σask)/(Σbid + Σask) ∈ [−1, +1].
    >0 = bid side heavier (resting support), <0 = ask side heavier."""
    v = _mean_last(_book(ctx).get("dbi_ring"), _BK_MIN)
    return 0.0 if v is None else max(-1.0, min(1.0, v))


def f33_mpd(ctx) -> float:
    """F33 MPD — inventory-weighted micro-price position inside the
    spread: 2·((P_micro − bid)/(ask − bid) − 0.5). >0 = flow pressure
    toward the ask (buyers lifting), <0 = toward the bid."""
    v = _mean_last(_book(ctx).get("dev_ring"), _BK_MIN)
    return 0.0 if v is None else max(-1.0, min(1.0, v))


def f34_cwa(ctx) -> float:
    """F34 CWA — cancellation wavefront elasticity Λ =
    ln(V_cancel_ask / V_cancel_bid) over the trailing second,
    normalised by ±3 → [-1, +1]. >0 = makers yanking Asks faster
    (overhead supply pulled = bullish vacuum), <0 = Bids yanked."""
    ring = _book(ctx).get("lambda_ring") or []
    if len(ring) < _BK_MIN:
        return 0.0
    lam = ring[-1]
    return max(-1.0, min(1.0, lam / 3.0))


def f35_lnp(ctx) -> float:
    """F35 LNP — liquidation net pressure: session-decayed share of
    BUY-side vs SELL-side forced-flow mass (real @forceOrder prints).
    >0 = shorts being liquidated upward (buy-side forced flow),
    <0 = longs flushed downward."""
    v = getattr(ctx, "liq_flow", None)
    if v is None or abs(v) < 1e-9:
        return 0.0
    return max(-1.0, min(1.0, float(v)))
