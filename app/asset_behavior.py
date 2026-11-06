"""asset_behavior.py — each asset predicts DIFFERENTLY, by its nature.

Profiles alone (floor/TP-SL scaling) did not change the prediction
itself. This module tempers the signed score + confidence BEFORE the
issue decision, using measured tape regime (trend efficiency,
volatility ratio, stretch from the mean):

  BTC   benchmark: no tempering — the reference behaviour.
  PAXG  gold trends and gaps: AMPLIFY when the tape is actually
        trending (trend efficiency high), DAMP hard in chop — gold
        chop is head-fake territory.
  ETH   unpredictable: in a volatility blow-off (recent vol >> base)
        the score is damped and confidence cut — ETH wicks punish
        conviction; the direction needs a bigger score to speak
        (higher noise gate in ASSET_PROFILES).
  BNB   steady / mean-reverting: when price is stretched far from
        its 30-bar mean, pull the score TOWARD reversion (against
        the stretch) proportional to the stretch — BNB snaps back.

Every adjustment is reported ({factor, action, reason, regime}) so
the UI can show exactly what the asset's nature did to the signal.
"""

from __future__ import annotations

import math


def _std(vals):
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


def regime(closes: list[float]) -> dict:
    """Measured tape regime from the last ~60 closes."""
    c = [float(x) for x in closes if x > 0]
    if len(c) < 35:
        return {"trend_eff": 0.0, "vol_ratio": 1.0, "stretch_z": 0.0,
                "n": len(c)}
    rets = [(c[i] - c[i - 1]) / c[i - 1] for i in range(1, len(c))]
    last30 = rets[-30:]
    path = sum(abs(x) for x in last30) or 1e-12
    trend_eff = abs(sum(last30)) / path            # 0 chop … 1 clean trend
    vol_ratio = (_std(rets[-10:]) / _std(rets[-60:])) if _std(rets[-60:]) else 1.0
    m30 = sum(c[-30:]) / 30.0
    s30 = _std(c[-30:]) or 1e-12
    stretch_z = (c[-1] - m30) / s30
    return {"trend_eff": round(trend_eff, 3),
            "vol_ratio": round(vol_ratio, 3),
            "stretch_z": round(stretch_z, 3), "n": len(c)}


def temper(symbol: str, score: float, confidence: float,
           closes: list[float]) -> dict:
    """Returns adjusted score/confidence + a human explanation."""
    r = regime(closes)
    s, conf = float(score), float(confidence)
    factor, action = 1.0, "reference behaviour (no tempering)"
    if symbol == "PAXG":
        if r["trend_eff"] >= 0.45:
            factor = 1.12
            action = f"gold trending (eff {r['trend_eff']}) — score amplified"
        elif r["trend_eff"] <= 0.25 and r["n"] >= 35:
            factor = 0.82
            action = f"gold chopping (eff {r['trend_eff']}) — score damped"
    elif symbol == "ETH":
        if r["vol_ratio"] >= 1.35:
            factor = 0.78
            action = (f"ETH vol blow-off (×{r['vol_ratio']}) — score + "
                      "confidence damped, wicks punish conviction")
    elif symbol == "BNB":
        if abs(r["stretch_z"]) >= 1.2 and r["n"] >= 35:
            pull = 0.22 * math.tanh(abs(r["stretch_z"]) / 2.0)
            s = s - math.copysign(pull, r["stretch_z"]) \
                if s else -math.copysign(pull, r["stretch_z"])
            action = (f"BNB stretched {r['stretch_z']:+.1f}σ from mean — "
                      f"reversion pull {pull:.2f} applied against the stretch")
            return {"score": round(s, 5),
                    "confidence": round(max(0.0, min(1.0, conf)), 5),
                    "factor": None, "action": action, "regime": r}
    s2 = s * factor
    conf2 = conf * factor if factor < 1.0 else \
        min(1.0, conf * (1.0 + (factor - 1.0) * 0.5))
    return {"score": round(s2, 5),
            "confidence": round(max(0.0, min(1.0, conf2)), 5),
            "factor": factor, "action": action, "regime": r}
