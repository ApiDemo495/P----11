"""Formula 21 — Composite Confidence Index (CCI).

Master aggregation.  The 28 weighted components (config.WEIGHT_IDS):

  * signals F1..F19, F22..F27, F31           (Bgn-adjusted, [-1,+1])
  * brain consensus                           (id 97, Bgn-adjusted)
  * CERC composite emotion flow               (id 98, Bgn-adjusted)

Excluded from the sum by design: F20/DRHE enters as the confidence
multiplier min(1, DRHE); F28 is a risk formula; F29 outputs price levels;
F30 validates the CCI after it exists; F21 is this aggregator.

    omega_f(t) = omega_f(t-1) * (1 + eta * reward_f(t)),  eta = 0.02
    reward_f = +1 if sign(F_f at previous GRC) == sign(price move since then)
               -1 otherwise
    renormalized to sum 1, each weight clamped to [0.005, 0.15]

    CCI = sum_f omega_f * F_f_adjusted
    double-check: if |CCI_reversed| > 0.6 * |CCI| ->
                  CCI_adjusted = CCI * (1 - 0.5 * |CCI_rev| / |CCI|)
    Confidence_base = |CCI_adjusted| * min(1, DRHE)
    Confidence_final = pipeline (engine.final stage):
        x min(1, EQ_B2)                       experience-quality multiplier
        + 0.1 * P(dominant state) if sign matches state direction
        - RTP * 0.2 when RTP > 0.6
        - |PCV| * 0.3 when PCV < 0
    issued only when Confidence_final > 0.35
"""
from __future__ import annotations

import json
import os

from .config import (
    CONFIDENCE_FLOOR,
    DOUBLE_CHECK_PENALTY,
    DOUBLE_CHECK_RATIO,
    LEARNING_RATE_ETA,
    STATE_FILE,
    TF_ORDER,
    WEIGHT_IDS,
    WEIGHT_MAX,
    WEIGHT_MIN,
)


class CCIEngine:
    def __init__(self) -> None:
        # weights[(symbol, tf)][component_id] -> omega
        self.weights: dict[tuple[str, str], dict[int, float]] = {}
        # signals[(symbol, tf)][component_id] -> last GRC's adjusted value
        self.prev_signals: dict[tuple[str, str], dict[int, float]] = {}
        self.prev_price: dict[str, float] = {}
        self._loaded = False

    # -- weights ------------------------------------------------------------
    def _key(self, symbol: str, tf: str) -> tuple[str, str]:
        return (symbol, tf)

    def get_weights(self, symbol: str, tf: str) -> dict[int, float]:
        key = self._key(symbol, tf)
        w = self.weights.get(key)
        if w is None:
            w = {fid: 1.0 / len(WEIGHT_IDS) for fid in WEIGHT_IDS}
            self.weights[key] = w
        return w

    def update_weights(self, symbol: str, tf: str, current_price: float) -> None:
        """Called exactly once per GRC.  Uses the previous GRC's signals and
        the actual price move over the GRC interval (4 minutes)."""
        key = self._key(symbol, tf)
        prev = self.prev_signals.get(key)
        prev_price = self.prev_price.get(symbol)
        if not prev or prev_price is None:
            return
        move = current_price - prev_price
        if move == 0:
            return
        w = self.get_weights(symbol, tf)
        for fid in list(w.keys()):
            f_prev = prev.get(fid, 0.0)
            reward = 1.0 if _sign(f_prev) == _sign(move) else -1.0
            w[fid] = w[fid] * (1.0 + LEARNING_RATE_ETA * reward)
            w[fid] = max(WEIGHT_MIN, min(WEIGHT_MAX, w[fid]))
        total = sum(w.values())
        if total > 0:
            for fid in w:
                w[fid] /= total

    def store_signals(self, symbol: str, tf: str, adjusted: dict[int, float]) -> None:
        self.prev_signals[self._key(symbol, tf)] = dict(adjusted)

    def store_price(self, symbol: str, price: float) -> None:
        self.prev_price[symbol] = price

    # -- aggregation --------------------------------------------------------
    def compute(
        self,
        symbol: str,
        tf: str,
        adjusted: dict[int, float],
        drhe: float,
    ) -> dict:
        w = self.get_weights(symbol, tf)
        cci = sum(w.get(fid, 0.0) * val for fid, val in adjusted.items())

        # double-check: minority disagreement penalty
        cci_reversed = sum(
            w.get(fid, 0.0) * val
            for fid, val in adjusted.items()
            if _sign(val) != _sign(cci) and _sign(val) != 0
        )
        cci_adjusted = cci
        if abs(cci) > 1e-9 and abs(cci_reversed) > DOUBLE_CHECK_RATIO * abs(cci):
            cci_adjusted = cci * (
                1.0 - DOUBLE_CHECK_PENALTY * abs(cci_reversed) / abs(cci)
            )

        confidence = abs(cci_adjusted) * min(1.0, max(0.0, drhe))
        issued = confidence > CONFIDENCE_FLOOR
        return {
            "cci": cci,
            "cci_adjusted": cci_adjusted,
            "cci_reversed": cci_reversed,
            "confidence": confidence,
            "direction": _sign(cci_adjusted),
            "issued": issued,
            "drhe": drhe,
        }

    # -- persistence --------------------------------------------------------
    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(STATE_FILE) or ".", exist_ok=True)
            payload = {
                "weights": {
                    f"{s}|{tf}": {str(k): v for k, v in w.items()}
                    for (s, tf), w in self.weights.items()
                }
            }
            with open(STATE_FILE, "w") as fh:
                json.dump(payload, fh)
        except OSError:
            pass

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            with open(STATE_FILE) as fh:
                payload = json.load(fh)
            for key_s, w in payload.get("weights", {}).items():
                symbol, tf = key_s.split("|", 1)
                if tf in TF_ORDER:
                    loaded = {int(k): float(v) for k, v in w.items()}
                    if set(loaded) != set(WEIGHT_IDS) or not loaded:
                        # part-1 state file (19 components) or corrupt entry:
                        # restart this series at the uniform prior
                        loaded = {fid: 1.0 / len(WEIGHT_IDS) for fid in WEIGHT_IDS}
                    # defensive clamp + renormalize on load
                    for fid in loaded:
                        loaded[fid] = max(WEIGHT_MIN, min(WEIGHT_MAX, loaded[fid]))
                    total = sum(loaded.values())
                    if total > 0:
                        for fid in loaded:
                            loaded[fid] /= total
                    self.weights[(symbol, tf)] = loaded
        except (OSError, ValueError, KeyError):
            pass


def finalize_confidence(
    pack: dict,
    *,
    experience_quality: float,
    p_dominant: float,
    state_aligned: bool,
    rtp: float,
    pcv: float,
) -> dict:
    """The Part-2 confidence pipeline, applied in spec order after the base
    confidence |CCI_adj| * min(1, DRHE):

        1. x (0.7 + 0.3 * EQ)                Brain-2 experience multiplier
        2. + 0.1 * P(dominant state)         Brain-3, only if sign(CCI)
                                             matches the state direction
        3. - RTP * 0.2 when RTP > 0.6        E9 trauma reduction
        4. - |PCV| * 0.3 when PCV < 0        F30 conflicting-signals reduction
        5. clamp >= 0, issue when > 0.35
    """
    conf = max(0.0, pack["confidence"])
    conf *= 0.7 + 0.3 * max(0.0, min(1.0, experience_quality))
    if state_aligned and p_dominant > 0.0:
        conf += 0.1 * p_dominant
    if rtp > 0.6:
        conf -= rtp * 0.2
    if pcv < 0.0:
        conf -= abs(pcv) * 0.3
    conf = max(0.0, conf)
    return {
        "confidence": conf,
        "issued": conf > CONFIDENCE_FLOOR,
        "pcv": pcv,
        "conflicting": pcv < 0.0,
    }


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0
