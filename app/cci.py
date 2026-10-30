"""Formula 21 — Composite Confidence Index (CCI).

Master aggregation.  The 31 weighted components (config.WEIGHT_IDS):

  * signals F1..F19, F22..F27, F31           (Bgn-adjusted, [-1,+1])
  * brain consensus                           (id 97, Bgn-adjusted)
  * CERC composite emotion flow               (id 98, Bgn-adjusted)
  * physics UFP (P22)                         (id 99, Bgn-adjusted, x TME)
  * News Impact Aggregate                     (id 100, Bgn-adjusted)
  * HF1+HF2+HF3 combined (initial weight 0.08)(id 101, Bgn-adjusted)

Excluded from the sum by design: F20/DRHE enters as the confidence
multiplier min(1, DRHE); F28 is a risk formula; F29 outputs price levels;
F30 validates the CCI after it exists; F21 is this aggregator.

    omega_f(t) = omega_f(t-1) * (1 + eta * reward_f(t)),  eta = 0.02
    reward_f = +1 if sign(F_f at previous GRC) == sign(price move since then)
               -1 otherwise
    renormalized to sum 1, each weight clamped to [0.005, 0.15]

    CCI = sum_f omega_f * F_f_adjusted        (id 99 scaled by TME, P9)
    double-check: if |CCI_reversed| > 0.6 * |CCI| ->
                  CCI_adjusted = CCI * (1 - 0.5 * |CCI_rev| / |CCI|)
    Confidence_base = |CCI_adjusted| * min(1, DRHE)
    Confidence_final = pipeline (engine.final stage):
        x (0.7 + 0.3 * EQ)                    experience-quality multiplier
        + 0.1 * P(dominant state) if sign matches state direction
        - RTP * 0.2 when RTP > 0.6
        - |PCV| * 0.3 when PCV < 0
        x (0.5 + 0.5 * WFCP)                  physics: wave-function collapse
        x (1 - DRPT)                          physics: trend decoherence
        x (1 - GSPI)                          physics: gauge non-invariance
    issued only when Confidence_final > 0.35
"""
from __future__ import annotations

import json
import os

from .config import (
    BRAIN_COMPONENT_ID,
    CONFIDENCE_FLOOR,
    CORE_SIGNAL_IDS,
    DOUBLE_CHECK_PENALTY,
    DOUBLE_CHECK_RATIO,
    EMOTION_COMPONENT_ID,
    ETA_MICRO,
    HEDGE_COMPONENT_ID,
    HEDGE_INITIAL_WEIGHT,
    LEARNING_RATE_ETA,
    NEWS_COMPONENT_ID,
    PHYSICS_COMPONENT_ID,
    STATE_FILE,
    SUBSYSTEM_SHARES_INIT,
    TF_ORDER,
    WEIGHT_IDS,
    WEIGHT_MAX,
    WEIGHT_MIN,
)


def uniform_prior() -> dict[int, float]:
    """Section 8: hedge/dark-pool combined starts at exactly 0.08; the
    remaining components share (1 - 0.08) equally."""
    n = len(WEIGHT_IDS)
    w = {fid: (1.0 - HEDGE_INITIAL_WEIGHT) / (n - 1) for fid in WEIGHT_IDS}
    w[HEDGE_COMPONENT_ID] = HEDGE_INITIAL_WEIGHT
    return w


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
            w = uniform_prior()
            self.weights[key] = w
        return w

    def update_weights(self, symbol: str, tf: str, current_price: float,
                       eta: float | None = None) -> None:
        """Micro-learning (Section 10): called exactly once per GRC using
        the previous GRC's signals and the actual price move."""
        rate = ETA_MICRO if eta is None else eta
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
            w[fid] = w[fid] * (1.0 + rate * reward)
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
    @staticmethod
    def effective_weights(w: dict[int, float],
                          shares: dict[str, float] | None) -> dict[int, float]:
        """Section 14 Level 1+2: core signals share `shares['core']`
        (split ∝ their learned ω); components 97..101 take their
        subsystem's share directly."""
        if not shares:
            return w
        sig = {f: w.get(f, 0.0) for f in CORE_SIGNAL_IDS if f in w}
        tot = sum(sig.values())
        eff: dict[int, float] = {}
        for f, v in sig.items():
            eff[f] = shares["core"] * (v / tot if tot > 0 else 1.0 / len(sig or [1]))
        eff[BRAIN_COMPONENT_ID] = shares["brain"]
        eff[EMOTION_COMPONENT_ID] = shares["emotion"]
        eff[PHYSICS_COMPONENT_ID] = shares["physics"]
        eff[NEWS_COMPONENT_ID] = shares["news"]
        eff[HEDGE_COMPONENT_ID] = shares["hf"]
        return eff

    def compute(
        self,
        symbol: str,
        tf: str,
        adjusted: dict[int, float],
        drhe: float,
        *,
        tme: float = 1.0,
        shares: dict[str, float] | None = None,
        raw_weights: dict[int, float] | None = None,
    ) -> dict:
        raw_w = raw_weights or self.get_weights(symbol, tf)
        # two-tier weighting (Section 14) applies when the engine supplies
        # the live subsystem shares; plain calls keep the flat ω semantics
        w = self.effective_weights(raw_w, shares) if shares else raw_w
        # P9: the physics component's effective weight is base * TME
        scaled = {
            fid: (val * tme if fid == PHYSICS_COMPONENT_ID else val)
            for fid, val in adjusted.items()
        }
        cci = sum(w.get(fid, 0.0) * val for fid, val in scaled.items())

        # double-check: minority disagreement penalty
        cci_reversed = sum(
            w.get(fid, 0.0) * val
            for fid, val in scaled.items()
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
            "weights_used": {str(k): round(v, 5) for k, v in w.items()},
        }

    # -- Section 16 mandatory double-check ---------------------------------
    @staticmethod
    def double_check(direction: int, confidence: float,
                     votes: dict[str, int]) -> dict:
        """Compare CCI / PCV / agent / brain assessments.

        3-4 agree  -> confirmed, confidence × 1.05
        2 agree    -> stands, × 0.85, MIXED SIGNALS
        <=1 agrees -> majority direction wins; no majority -> NO TRADE
        """
        others = [v for v in votes.values() if v]
        agree = sum(1 for v in others if v == direction)
        voices = 1 + len(others)
        if voices >= 3 and agree <= 0:
            # CCI alone against everyone: take the majority of the others
            pos = sum(1 for v in others if v > 0)
            neg = len(others) - pos
            if pos > neg:
                return {"direction": 1, "confidence": confidence,
                        "issued": False, "label": "overridden",
                        "agree": agree, "voices": voices}
            if neg > pos:
                return {"direction": -1, "confidence": confidence,
                        "issued": False, "label": "overridden",
                        "agree": agree, "voices": voices}
            return {"direction": direction, "confidence": 0.0,
                    "issued": False, "label": "no_trade",
                    "agree": agree, "voices": voices}
        if agree >= 2:
            return {"direction": direction,
                    "confidence": confidence * 1.05,
                    "issued": True, "label": "confirmed",
                    "agree": agree, "voices": voices}
        if agree == 1 and len(others) >= 2:
            return {"direction": direction,
                    "confidence": confidence * 0.85,
                    "issued": True, "label": "mixed",
                    "agree": agree, "voices": voices}
        return {"direction": direction, "confidence": confidence,
                "issued": True, "label": "standalone",
                "agree": agree, "voices": voices}

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
                        # part-1/2 state file (28 components) or corrupt
                        # entry: restart this series at the spec prior
                        loaded = uniform_prior()
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
    wfcp: float = 1.0,
    drpt: float = 0.0,
    gspi: float = 0.0,
) -> dict:
    """The confidence pipeline, applied in spec order after the base
    confidence |CCI_adj| * min(1, DRHE):

        1. x (0.7 + 0.3 * EQ)                Brain-2 experience multiplier
        2. + 0.1 * P(dominant state)         Brain-3, only if sign(CCI)
                                             matches the state direction
        3. - RTP * 0.2 when RTP > 0.6        E9 trauma reduction
        4. - |PCV| * 0.3 when PCV < 0        F30 conflicting-signals reduction
        5. x (0.5 + 0.5 * WFCP)              P2  wave-function collapse
        6. x (1 - DRPT)                      P6  trend decoherence
        7. x (1 - GSPI)                      P19 gauge non-invariance
        8. clamp >= 0, issue when > 0.35
    Neutral physics defaults (WFCP=1, DRPT=0, GSPI=0) leave steps 1-4 exact.
    """
    conf = max(0.0, pack["confidence"])
    conf *= 0.7 + 0.3 * max(0.0, min(1.0, experience_quality))
    if state_aligned and p_dominant > 0.0:
        conf += 0.1 * p_dominant
    if rtp > 0.6:
        conf -= rtp * 0.2
    if pcv < 0.0:
        conf -= abs(pcv) * 0.3
    # Section 6 physics modifiers (P2/P6/P19)
    conf *= 0.5 + 0.5 * max(0.0, min(1.0, wfcp))
    conf *= 1.0 - max(0.0, min(1.0, drpt))
    conf *= 1.0 - max(0.0, min(1.0, gspi))
    conf = max(0.0, conf)
    return {
        "confidence": conf,
        "issued": conf > CONFIDENCE_FLOOR,
        "pcv": pcv,
        "conflicting": pcv < 0.0,
        "wfcp": wfcp,
        "drpt": drpt,
        "gspi": gspi,
    }


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0
