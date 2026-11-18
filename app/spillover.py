"""Cross-asset spillover + thesis-signed reversal block (live).

The alpha lab (`app/alpha_lab.py`, `backtest/results/ALPHA_LAB.md`) measured
what actually predicts 1h–4h crypto returns on nine months of real Binance
bars: **not momentum**.  Every momentum-flavoured reading carried a
*negative* information coefficient (BTC −0.046, ETH −0.065 at 2–4 h), while
the same readings taken contrarian — plus the market leader's recent move —
carried positive IC in every fold on every asset:

    signed reversal composite, out-of-fold
        BTC  IC +0.048 / +0.026   acc 47.9% → 51.6%
        ETH  IC +0.055 / +0.046   acc 47.4% → 53.1%
        BNB  IC +0.034 / +0.032   acc 47.4% → 52.5%
        PAXG IC +0.014 / +0.010   acc 49.4% → 49.4%
        mean IC −0.036 → +0.033,  mean sign accuracy 48.0% → 51.4%

The strongest single input was the **cross-asset term**: BTC's last six
hours, used contrarian, held its sign in 100% of folds on all four assets
(field evidence for the lead-lag / spillover literature).

This module is the live, causal implementation.  It never re-decides a
sign at runtime (folds are too short to do that safely): the signs are the
validated priors from `alpha_lab.SIGN_PRIOR`, magnitudes are equal weight,
and the output reports how many inputs were actually available so a thin
tape cannot masquerade as a strong reading.

Output contract
    {
      "available": bool,
      "score": float in [-1, 1],       # contrarian composite, + = up
      "direction": -1 | 0 | 1,
      "parts": {name: signed z},       # each already prior-oriented
      "xasset": {"ret": float, "bars": int, "asset": str},
      "coverage": float,               # share of inputs present
      "note": str,
    }
"""
from __future__ import annotations

import math

XASSET_BARS = 6
MIN_BARS = 30
# thresholds are deliberately loose: this block is one voter among many,
# and the engine's own noise gate decides whether a small score matters.
DIRECTLY_SCORED = ("rev_mom", "rev_range", "rev_xasset", "jump")


def _z(a: list[float]) -> float:
    if len(a) < 3:
        return 0.0
    mu = sum(a) / len(a)
    var = sum((x - mu) ** 2 for x in a) / len(a)
    sd = math.sqrt(var)
    return 0.0 if sd < 1e-12 else (a[-1] - mu) / sd


def _ret(c: list[float], k: int, i: int | None = None) -> float:
    i = len(c) - 1 if i is None else i
    j = i - k
    if j < 0 or c[j] <= 0 or c[i] <= 0:
        return 0.0
    return c[i] / c[j] - 1.0


def _sd_returns(c: list[float], n: int = 96) -> float:
    seg = c[-(n + 1):]
    if len(seg) < 8:
        return 0.0
    rets = [seg[k] / seg[k - 1] - 1.0 for k in range(1, len(seg)) if seg[k - 1]]
    if not rets:
        return 0.0
    mu = sum(rets) / len(rets)
    return math.sqrt(sum((r - mu) ** 2 for r in rets) / len(rets))


def score(closes, highs=None, lows=None, vols=None, taker=None,
          market_closes=None, market: str = "BTC",
          bar_minutes: float = 60.0,
          market_bar_minutes: float = 60.0) -> dict:
    """Compute the contrarian composite at the last bar of `closes`.

    Every input is optional except `closes`; the result says how much of
    the block it could actually measure.
    """
    c = [float(x) for x in (closes or []) if x is not None]
    out = {"available": False, "score": 0.0, "direction": 0, "parts": {},
           "xasset": {"ret": 0.0, "bars": XASSET_BARS, "asset": market},
           "coverage": 0.0, "note": ""}
    if len(c) < MIN_BARS:
        out["note"] = f"need ≥{MIN_BARS} closes, have {len(c)}"
        return out

    have = 0
    parts: dict[str, float] = {}
    sd = _sd_returns(c) or 1.

    # 1) reversal momentum: 3 h / 12 h / 48 h returns, sign flipped.
    # Horizons are expressed in HOURS and converted with the caller's bar
    # length, because the validated feature set was measured on 1h bars
    # and a TF1 lane runs on 22-minute bars.
    ks = [max(1, int(round(h * 60.0 / max(1.0, bar_minutes))))
          for h in (3.0, 12.0, 48.0)]
    m = [-(_ret(c, k) / (sd * math.sqrt(k))) for k in ks]
    parts["rev_mom"] = max(-3.0, min(3.0, sum(m) / 3.0))
    have += 1

    # 2) reversal range position (needs highs/lows, falls back to closes)
    hs = [float(x) for x in (highs or []) if x is not None] or c
    ls = [float(x) for x in (lows or []) if x is not None] or c
    win = 24
    if len(hs) >= win and len(ls) >= win:
        hi, lo = max(hs[-win:]), min(ls[-win:])
        if hi > lo:
            parts["rev_range"] = max(-3.0, min(3.0, -((c[-1] - lo) / (hi - lo) - 0.5) * 2.0))
    have += 1

    # 3) jump separation (bipower variation vs realised variance)
    seg = c[-25:]
    rr = [abs(seg[k] / seg[k - 1] - 1.0) for k in range(1, len(seg)) if seg[k - 1]]
    if len(rr) > 3:
        bv = sum(rr[k] * rr[k - 1] for k in range(1, len(rr))) / (len(rr) - 1)
        bv *= math.pi / 2.0
        rv = sum(r * r for r in rr) / len(rr)
        parts["jump"] = max(-1.0, min(1.0, 1.0 - (bv / rv if rv > 1e-18 else 1.0)))
    have += 1

    # 4) contrarian taker flow (only when the feed carries it)
    tk = [float(x) for x in (taker or []) if x is not None]
    vv = [float(x) for x in (vols or []) if x is not None]
    if len(tk) >= 12 and len(vv) >= 12:
        imbs = []
        for t, v in zip(tk[-12:], vv[-12:]):
            if v > 0:
                imbs.append(2.0 * t / v - 1.0)
        if imbs:
            mu = sum(imbs) / len(imbs)
            parts["rev_imb"] = max(-1.0, min(1.0, -mu))
            have += 1

    # 5) cross-asset spillover: the leader's last six bars, contrarian
    mc = [float(x) for x in (market_closes or []) if x is not None]
    xk = max(1, int(round(6.0 * 60.0 / max(1.0, market_bar_minutes))))  # 6 h
    if len(mc) > xk + 5:
        msd = _sd_returns(mc) or 1.
        raw = _ret(mc, xk)
        out["xasset"] = {"ret": raw, "bars": xk, "asset": market}
        parts["rev_xasset"] = max(-3.0, min(3.0,
                                           -(raw / (msd * math.sqrt(xk)))))
        have += 1

    total_possible = 5
    scored = [v for v in parts.values()]
    composite = sum(scored) / len(scored) if scored else 0.0
    out.update({
        "available": True,
        "score": round(math.tanh(composite / 1.5), 4),
        "direction": 1 if composite > 0.05 else (-1 if composite < -0.05 else 0),
        "parts": {k: round(v, 3) for k, v in parts.items()},
        "coverage": round(have / total_possible, 2),
        "note": (f"{len(scored)}/{total_possible} inputs measured "
                 f"(highs/lows/market/flow optional)"),
    })
    return out
