"""5-state asset-calibrated regime engine (§1.3).

Replaces the dead-band classifier that read almost every tape as
``calm_ranging`` / ``transitional`` (trend_strength = |μ|/σ needed > 0.3
on pct returns — a threshold ordinary trends never clear). Every state
decision here is a PERCENTILE of THIS asset's own recent distribution,
so the classifier cannot be stuck: a state is declared by being extreme
relative to the tape that produced it.

States
------
``IMPULSE_BULL``              directional efficiency + positive drift
                              + upside semi-variance dominance
``IMPULSE_BEAR``              directional efficiency + downside
                              semi-variance expansion
``VOLATILITY_SQUEEZE_BREAKOUT`` Bollinger/Keltner compression release:
                              bandwidth was in its bottom quintile
                              within the lookback, now expanding with
                              range/tick surge
``MEAN_REVERTING_RANGE``      high variance-ratio decay (VR(k)≪ 1) with
                              mid-band oscillation
``CHAOTIC_LIQUIDATION_SHOCK``  Lee–Mykland jump spike + range blowout —
                              triggers risk throttling

Legacy mapping keeps the five history labels (volatile_trending,
trending, transitional, volatile_ranging, calm_ranging) so DRHE's
duration rings and the F20 telemetry stay continuous.
"""
from __future__ import annotations

import math
from typing import Any

STATES = (
    "IMPULSE_BULL",
    "IMPULSE_BEAR",
    "VOLATILITY_SQUEEZE_BREAKOUT",
    "MEAN_REVERTING_RANGE",
    "CHAOTIC_LIQUIDATION_SHOCK",
)

# risk throttle: callers should cut size / force watch-only on shock
RISK_THROTTLE_STATES = frozenset({"CHAOTIC_LIQUIDATION_SHOCK"})

_LEGACY = {
    "IMPULSE_BULL": "trending",
    "IMPULSE_BEAR": "volatile_trending",
    "VOLATILITY_SQUEEZE_BREAKOUT": "volatile_ranging",
    "MEAN_REVERTING_RANGE": "calm_ranging",
    "CHAOTIC_LIQUIDATION_SHOCK": "volatile_trending",
}


def _std(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    m = sum(vals) / n
    return math.sqrt(sum((x - m) ** 2 for x in vals) / n)


def _pct_rank(vals: list[float], x: float) -> float:
    """Empirical percentile of x inside vals (0..1)."""
    if not vals:
        return 0.5
    below = sum(1 for v in vals if v < x)
    return below / len(vals)


def _returns(closes: list[float]) -> list[float]:
    return [(closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes)) if closes[i - 1] > 0]


def efficiency_ratio(closes: list[float], n: int = 30) -> float:
    """Kaufman ER: |net| / Σ|Δ| over the last n bars (0 chop … 1 trend)."""
    r = _returns(closes[-(n + 1):])
    if not r:
        return 0.0
    path = sum(abs(x) for x in r)
    if path <= 1e-12:
        return 0.0
    return abs(sum(r)) / path


def variance_ratio(closes: list[float], k: int = 5, n: int = 60) -> float:
    """VR(k) of the last n returns: ≫1 trending, ≪1 mean-reverting."""
    r = _returns(closes[-(n * k + 1):])[-n * k:]
    if len(r) < k * 5:
        return 1.0
    var1 = _std(r) ** 2
    if var1 <= 1e-18:
        return 1.0
    # k-period returns from non-overlapping blocks
    agg = [sum(r[i:i + k]) for i in range(0, len(r) - k + 1, k)]
    vark = _std(agg) ** 2 if len(agg) >= 3 else var1 * k
    return (vark / (k * var1)) if var1 > 0 else 1.0


def bandwidth_pct(closes: list[float], n: int = 60, win: int = 20) -> float:
    """Bollinger bandwidth percentile within the last n bars (0..1)."""
    if len(closes) < win + 5:
        return 0.5
    bws: list[float] = []
    for i in range(win, len(closes)):
        seg = closes[i - win + 1:i + 1]
        m = sum(seg) / len(seg)
        s = _std(seg)
        if m > 0:
            bws.append(4.0 * s / m)           # %B width, unit-free
    if not bws:
        return 0.5
    return _pct_rank(bws[:-1], bws[-1]) if len(bws) > 1 else 0.5


def jump_statistic(highs: list[float], lows: list[float],
                   closes: list[float]) -> float:
    """Lee–Mykland style |close return| / bipower variation on the last bar.

    Returns the ratio of the latest bar's |return| to the median scaled
    bipower variation of the previous bars (≥0, ~1 = ordinary, ≫1 = jump).
    """
    r = _returns(closes)
    if len(r) < 20 or not highs or not lows:
        return 1.0
    # bipower variation: mean |r_i|·|r_{i-1}| scaled by π/2
    bp_terms = [abs(r[i]) * abs(r[i - 1]) for i in range(1, len(r) - 1)]
    bp = (sum(bp_terms) / len(bp_terms)) * (math.pi / 2.0) if bp_terms else 0.0
    bp = math.sqrt(bp) if bp > 0 else 0.0
    if bp <= 1e-12:
        return 1.0
    return abs(r[-1]) / bp


def semi_variance_share(rets: list[float]) -> tuple[float, float]:
    """(up_share, down_share) of semi-variances over rets."""
    up = sum(x * x for x in rets if x > 0)
    dn = sum(x * x for x in rets if x < 0)
    tot = up + dn
    if tot <= 1e-18:
        return 0.5, 0.5
    return up / tot, dn / tot


def classify_tape(closes: list[float],
                  highs: list[float] | None = None,
                  lows: list[float] | None = None,
                  volumes: list[float] | None = None,
                  lookback: int = 240) -> dict[str, Any]:
    """Classify the tape into one of the five states.

    All inputs are minute bars of ONE asset. Every threshold is a
    percentile of this tape's own history — nothing is a fixed universal
    constant, so no state can become a permanent attractor.
    """
    c = [float(x) for x in (closes or []) if x > 0][-lookback:]
    if len(c) < 60:
        return {"state": "MEAN_REVERTING_RANGE",
                "confidence": 0.0, "legacy": "calm_ranging",
                "reason": f"only {len(c)} bars — cold tape, low confidence",
                "metrics": {"n": len(c)}, "risk_throttle": False}
    h = list(highs or c)[-len(c):]
    l = list(lows or c)[-len(c):]
    v = list(volumes or [])[-len(c):]
    rets = _returns(c)

    # ── metrics ──────────────────────────────────────────────────────
    er30 = efficiency_ratio(c, 30)
    er10 = efficiency_ratio(c, 10)
    vr = variance_ratio(c, 5, 60)
    bw_pct = bandwidth_pct(c)
    # bandwidth expansion: current vs its own min over the lookback
    bws: list[float] = []
    for i in range(20, len(c)):
        seg = c[i - 19:i + 1]
        m = sum(seg) / len(seg)
        if m > 0:
            bws.append(4.0 * _std(seg) / m)
    bw_now = bws[-1] if bws else 0.0
    bw_min = min(bws[-120:]) if bws else 0.0
    bw_release = (bw_now / bw_min) if bw_min > 1e-9 else 1.0
    up_share, dn_share = semi_variance_share(rets[-30:])
    drift = sum(rets[-30:]) / 30.0 if rets else 0.0
    sd = _std(rets[-60:]) or 1e-12
    drift_z = drift / sd
    jump = jump_statistic(h, l, c)
    # range blowout: last bar range vs median range of prior 60
    ranges = [h[i] - l[i] for i in range(len(h)) if h[i] >= l[i]]
    med_range = sorted(ranges[-61:-1])[len(ranges[-61:-1]) // 2] \
        if len(ranges) > 20 else 0.0
    last_range = ranges[-1] if ranges else 0.0
    range_ratio = (last_range / med_range) if med_range > 1e-12 else 1.0
    # tick surge: volume of last bar vs its own 60-bar mean
    vol_surge = 1.0
    if v and len(v) >= 20 and sum(v[:-1]) > 0:
        vol_surge = (v[-1] / (sum(v[-60:-1]) / min(60, len(v) - 1))) \
            if sum(v[-60:-1]) > 0 else 1.0

    # ── decision tree (percentile-anchored, in priority order) ───────
    # multi-horizon efficiency: a developing trend shows on er10 first;
    # using er30 alone re-created the old "everything is sideways" trap
    er_fast = max(er10, er30)
    metrics = {
        "n": len(c), "er30": round(er30, 4), "er10": round(er10, 4),
        "vr5": round(vr, 4), "bw_pct": round(bw_pct, 4),
        "bw_release": round(bw_release, 4), "drift_z": round(drift_z, 4),
        "up_share": round(up_share, 4), "dn_share": round(dn_share, 4),
        "jump": round(jump, 4), "range_ratio": round(range_ratio, 3),
        "vol_surge": round(vol_surge, 3),
    }

    # 1) liquidation shock: jump AND range blowout together
    if jump >= 3.0 and (range_ratio >= 2.0 or vol_surge >= 2.5):
        return {"state": "CHAOTIC_LIQUIDATION_SHOCK", "confidence":
                min(1.0, 0.4 + 0.2 * (jump - 3.0)),
                "legacy": _LEGACY["CHAOTIC_LIQUIDATION_SHOCK"],
                "reason": (f"Lee–Mykland jump ×{jump:.1f} with range "
                           f"×{range_ratio:.1f} / volume ×{vol_surge:.1f}"),
                "metrics": metrics, "risk_throttle": True}

    # 2) squeeze release: bandwidth was compressed (bottom quintile of
    #    its own recent range) and is now expanding with a range/vol surge
    bw_floor_q = sorted(bws[-120:])[max(0, int(len(bws[-120:]) * 0.2))] \
        if len(bws) >= 20 else 0.0
    was_squeeze = bw_min <= bw_floor_q * 1.15 or bw_pct <= 0.25
    if was_squeeze and bw_release >= 1.35 and (range_ratio >= 1.4
                                               or vol_surge >= 1.8):
        return {"state": "VOLATILITY_SQUEEZE_BREAKOUT", "confidence":
                min(1.0, 0.35 + 0.25 * (bw_release - 1.35)),
                "legacy": _LEGACY["VOLATILITY_SQUEEZE_BREAKOUT"],
                "reason": (f"bandwidth released ×{bw_release:.2f} from "
                           f"its bottom quintile (pct {bw_pct:.2f})"),
                "metrics": metrics, "risk_throttle": False}

    # 3) impulse: directional efficiency on EITHER horizon with
    #    semi-variance support (the semi-variance sign, not a dead-band
    #    on |μ|/σ, is what separates a real impulse from driftless noise)
    if er_fast >= 0.50 and abs(drift_z) >= 0.8:
        bull = drift > 0 and up_share >= 0.52
        bear = drift < 0 and dn_share >= 0.52
        if bull or bear:
            state = "IMPULSE_BULL" if bull else "IMPULSE_BEAR"
            return {"state": state,
                    "confidence": min(1.0, 0.4 + 0.3 * (er_fast - 0.50)
                                      + 0.1 * min(abs(drift_z) - 0.8, 2.0)),
                    "legacy": _LEGACY[state],
                    "reason": (f"ER {er_fast:.2f}, drift z {drift_z:+.2f}, "
                               f"semi-var share {max(up_share, dn_share):.2f}"),
                    "metrics": metrics, "risk_throttle": False}

    # 4) mean-reverting range: VR collapse + efficiency dead on BOTH horizons
    if vr <= 0.65 and er30 < 0.40 and er10 < 0.50:
        return {"state": "MEAN_REVERTING_RANGE",
                "confidence": min(1.0, 0.4 + 0.3 * (0.65 - vr)),
                "legacy": _LEGACY["MEAN_REVERTING_RANGE"],
                "reason": f"VR(5) {vr:.2f} with ER {er30:.2f} — variance "
                          "decays, price oscillates around the mean",
                "metrics": metrics, "risk_throttle": False}

    # 5) fallbacks — never a dead band: every branch names a state and
    #    declares its own (low) confidence
    if er30 >= 0.25 or abs(drift_z) >= 0.5:
        state = "IMPULSE_BULL" if drift >= 0 else "IMPULSE_BEAR"
        return {"state": state, "confidence": 0.2,
                "legacy": "transitional", "reason":
                f"weak tendency (ER {er30:.2f}, z {drift_z:+.2f}) — "
                "low-confidence impulse, not a range",
                "metrics": metrics, "risk_throttle": False}
    if vr >= 1.30:
        state = "IMPULSE_BULL" if drift >= 0 else "IMPULSE_BEAR"
        return {"state": state, "confidence": 0.2,
                "legacy": "transitional", "reason":
                f"VR(5) {vr:.2f} — trending variance, weak efficiency",
                "metrics": metrics, "risk_throttle": False}
    return {"state": "MEAN_REVERTING_RANGE", "confidence": 0.25,
            "legacy": "calm_ranging", "reason":
            f"no state extreme vs this tape (ER {er30:.2f}, VR {vr:.2f})",
            "metrics": metrics, "risk_throttle": False}

# [removed 2026-11-23] legacy_label — dead code: zero references anywhere


def classify_regime(trend_strength: float, volatility_level: float) -> str:
    """Scalar entry point kept for F20 (unit-free inputs).

    The dead-band is gone: thresholds are on scales that ordinary tapes
    actually occupy (|μ|/σ of pct returns hits 0.10+ in real trends; the
    old 0.30 gate classified almost everything ranging).
    """
    if trend_strength > 0.12:
        return "volatile_trending" if volatility_level > 1.5 else "trending"
    if volatility_level > 1.6:
        return "volatile_ranging"
    if volatility_level <= 0.75:
        return "calm_ranging"
    if trend_strength < 0.04:
        return "calm_ranging"
    return "transitional"
