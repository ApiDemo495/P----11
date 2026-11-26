"""Formula 21 — Composite Confidence Index (CCI).

Master aggregation.  The 35 weighted components (config.WEIGHT_IDS):

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
import math
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
    TRACER_COMPONENT_ID,
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
        elif any(fid not in w for fid in WEIGHT_IDS):
            # state saved before F32..F35 existed: extend with the
            # uniform prior value for the missing signals, then keep the
            # total mass at1 (the system's own normalisation rule)
            prior = uniform_prior()
            for fid in WEIGHT_IDS:
                if fid not in w:
                    w[fid] = prior[fid]
            total = sum(w.values()) or 1.0
            for fid in w:
                w[fid] = w[fid] / total
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
        move_sign = _sign(move)
        for fid in list(w.keys()):
            f_prev = prev.get(fid, 0.0)
            signal_sign = _sign(f_prev)
            if not signal_sign:
                continue                 # abstention is not a scored miss
            reward = 1.0 if signal_sign == move_sign else -1.0
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
        eff[TRACER_COMPONENT_ID] = shares.get("tracer", 0.0)
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
        from .rlsys import damp_all
        # RL error-finder: broken components contribute with a learned
        # damp factor (default1.0 = untouched legacy behaviour)
        raw_w = damp_all(raw_weights or self.get_weights(symbol, tf))
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
            # Bug fixed 2026-11-26: this was
            #   cci * (1 - PENALTY * |rev| / |cci|)
            # which turns NEGATIVE once |rev| > |cci| / PENALTY (0.5). A
            # "penalty" then INVERTED the sign of the fused signal and grew
            # its magnitude — measured: cci = -0.120 (net bearish) produced
            # adjusted = +0.130, and `direction` reads `cci_adjusted`, so the
            # app issued a BUY off a net-bearish vote. A second case went
            # from -0.050 to +0.400, i.e. 8x larger after being "penalised".
            #
            # A disagreement penalty may only ever REDUCE the magnitude, and
            # must never cross zero. Clamping the factor into [0, 1] keeps
            # the intended meaning: heavy disagreement shrinks the call
            # toward no-call (factor 0), it does not reverse it.
            shrink = DOUBLE_CHECK_PENALTY * abs(cci_reversed) / abs(cci)
            cci_adjusted = cci * max(0.0, min(1.0, 1.0 - shrink))

        confidence = abs(cci_adjusted) * min(1.0, max(0.0, drhe))
        # strength = agreement purity: |net| / gross weighted mass.
        # 1.0 = every weighted vote pointed the same way; ~0 = the
        # voters cancelled out. Unlike |CCI| (a weighted MEAN whose
        # scale depends on vote magnitudes, landing at 0.001-0.03),
        # this spreads honestly over [0,1].
        _gross = sum(abs(w.get(fid, 0.0) * val)
                     for fid, val in scaled.items())
        strength = (abs(cci_adjusted) / _gross) if _gross > 1e-12 else 0.0
        issued = confidence > CONFIDENCE_FLOOR
        # Bug fixed 2026-11-26: `direction` was `_sign(cci_adjusted)`, but the
        # engine OVERWRITES `direction` afterwards with the mandate rule
        # (fade / inv4g / mom44 / hold-policy side) and never rewrites
        # `cci_adjusted`. The dashboard then printed a BULLISH banner beside
        # a NEGATIVE "CCI ADJ" in the same viewport (user screenshot, §40).
        # `direction_source` now records which rule produced the side so the
        # UI can label the divergence instead of looking self-contradictory.
        return {
            "cci": cci,
            "cci_adjusted": cci_adjusted,
            "direction_source": "cci",
            "cci_reversed": cci_reversed,
            "confidence": confidence,
            "strength": round(max(0.0, min(1.0, strength)), 4),
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
    # Bug fixed 2026-11-26: `pack["confidence"]` raised KeyError whenever the
    # caller passed a pack without that key (e.g. a raw CCI dict carrying only
    # `cci_adjusted`/`drhe`), and a NaN there propagated to the caller as a
    # NaN confidence that the UI renders as a blank or "—" mid-panel. A
    # missing or non-finite input is "no confidence", not a crash and not a
    # silent NaN.
    raw_conf = pack.get("confidence")
    if raw_conf is None:
        raw_conf = pack.get("cci_adjusted", 0.0)
    try:
        conf = float(raw_conf)
    except (TypeError, ValueError):
        conf = 0.0
    if not math.isfinite(conf):
        return {"confidence": 0.0, "issued": False, "penalty_capped": False,
                "penalty": 0.0, "pcv": pcv, "conflicting": False,
                "wfcp": wfcp, "drpt": drpt, "gspi": gspi, "invalid": True}
    conf = max(0.0, conf)
    conf *= 0.7 + 0.3 * max(0.0, min(1.0, experience_quality))
    if state_aligned and p_dominant > 0.0:
        conf += 0.1 * p_dominant
    # zeros cause-1: additive penalties used to clip the score to an
    # EXACT 0 (a silent zero). Cap the TOTAL penalty at 90% of the
    # available confidence — normal cases keep their exact arithmetic
    # (0.5-0.14, 0.5-0.15 in the pinned tests); the collapse case keeps
    # an honest residue with penalty_capped=True for the UI.
    avail = conf
    penalty = 0.0
    if rtp > 0.6:
        penalty += rtp * 0.2
    if pcv < 0.0:
        penalty += abs(pcv) * 0.3
    conf = avail - min(penalty, 0.9 * avail)
    penalty_capped = penalty > 0.9 * avail + 1e-12
    # Section 6 physics modifiers (P2/P6/P19)
    conf *= 0.5 + 0.5 * max(0.0, min(1.0, wfcp))
    conf *= 1.0 - max(0.0, min(1.0, drpt))
    conf *= 1.0 - max(0.0, min(1.0, gspi))
    conf = max(0.0, conf)
    return {
        "confidence": conf,
        "issued": conf > CONFIDENCE_FLOOR,
        "penalty_capped": penalty_capped,
        "penalty": round(penalty, 6),
        "pcv": pcv,
        "conflicting": pcv < 0.0,
        "wfcp": wfcp,
        "drpt": drpt,
        "gspi": gspi,
    }


#: |CCI_adjusted| that counts as ONE noise unit (F30's measured floor).
#: A weighted mean of [-1,+1] votes lands at 0.02–0.08 in a normal tape —
#: using that raw number as a probability is why the old chain could
#: never reach the 0.35 issuance floor (0.0218 × 0.87 × 0.44 × 0.36 ≈
#: 0.0003 forever).
NOISE_FLOOR = 0.020
#: tanh gain: |CCI| = 1 noise unit → base ≈ 0.47, 3 units → ≈ 0.92
BASE_GAIN = 0.45
#: bound on the TOTAL modulation so no stack of dampers can zero a signal
MOD_MIN, MOD_MAX = 0.72, 1.12
CONF_MIN, CONF_MAX = 0.02, 0.98


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def calibrate_confidence(
    pack: dict,
    *,
    experience_quality: float = 0.5,
    p_dominant: float = 0.0,
    state_aligned: bool = False,
    rtp: float = 0.0,
    pcv: float = 0.0,
    wfcp: float = 1.0,
    drpt: float = 0.0,
    gspi: float = 0.0,
    agreement: float | None = None,
    noise_floor: float = NOISE_FLOOR,
) -> dict:
    """Calibrated confidence — the ONE number every view displays.

    The old chain multiplied ``|CCI_adj|`` (0.02–0.08) by six fractions in
    sequence (×0.87 ×0.44 ×0.36 …), so the product was ~0.0002 and no
    genuine setup could ever clear the 0.35 floor: RECHECK always said
    NO TRADE and CALLS & TARGETS printed conf 0.000.

    This version, in the same *documented* order:

      1. SNR mapping     snr = |CCI_adj| / noise_floor
                         base = tanh(BASE_GAIN · snr) · min(1, DRHE)
                         → a decisive signal starts in [0.25, 0.75]
      2. × (0.90 + 0.10·EQ)          brain-2 experience       [−10 %, 0]
      3. + 0.05·base·P(dominant)     brain-3, only if aligned [+5 % at most]
      4. × (1 − 0.12·ramp(RTP))      E9 trauma                [−12 %, 0]
         × (1 − 0.10·|PCV|)          F30 conflict             [−10 %, 0]
      5. × (0.92 + 0.08·WFCP)        P2  collapse             [−8 %, 0]
      6. × (1 − 0.08·DRPT)           P6  decoherence          [−8 %, 0]
      7. × (1 − 0.06·GSPI)           P19 gauge                [−6 %, 0]
      8. clip(modulator, 0.72, 1.12) then clip(conf, 0.02, 0.98)

    Every modifier is BOUNDED, so this is a calibration, not a product of
    independent probabilities.  The returned ``steps`` list is exactly
    what the CONFIDENCE CHAIN panel renders — no second formula.
    """
    cci_adj = float(pack.get("cci_adjusted") or 0.0)
    drhe = _clamp01(float(pack.get("drhe", 1.0) or 1.0))
    floor = max(float(noise_floor or NOISE_FLOOR), 1e-4)
    # Bug fixed 2026-11-26: a non-finite CCI (NaN/inf from any upstream
    # division or over/underflow) used to propagate into `snr` and then into
    # tanh, producing confidence 0.98 - the HIGHEST possible - because
    # math.tanh(inf) == 1.0 and `max(0.02, min(0.98, nan))` returns 0.98
    # (min/max silently discard NaN). So a broken calculation was displayed
    # as maximum conviction, which is the worst possible failure mode for a
    # prediction panel: it looks like the most confident call the system can
    # make. A failed computation must yield NO confidence, not all of it.
    if not math.isfinite(cci_adj):
        return {"confidence": 0.0, "base": 0.0, "cci_adjusted": None,
                "snr": 0.0, "modulator": 0.0,
                "steps": [("input", "non-finite CCI — confidence withheld")],
                "invalid": True}
    snr = abs(cci_adj) / floor
    base = math.tanh(BASE_GAIN * snr) * min(1.0, drhe)

    eq_f = 0.90 + 0.10 * _clamp01(experience_quality)
    p_dom = _clamp01(p_dominant)
    align_bonus = 0.05 * base * p_dom if (state_aligned and p_dom > 0) else 0.0
    rtp_f = 1.0 - 0.12 * _clamp01((rtp - 0.6) / 0.4)
    pcv_f = 1.0 - 0.10 * _clamp01(abs(pcv))
    wfcp_f = 0.92 + 0.08 * _clamp01(wfcp)
    drpt_f = 1.0 - 0.08 * _clamp01(drpt)
    gspi_f = 1.0 - 0.06 * _clamp01(gspi)

    modulator = eq_f * rtp_f * pcv_f * wfcp_f * drpt_f * gspi_f
    modulator = max(MOD_MIN, min(MOD_MAX, modulator))
    conf = base * modulator + align_bonus
    if agreement is not None:
        # bounded ±5 % voter-agreement band (never the direction)
        band = 0.05 * (2.0 * _clamp01(agreement) - 1.0)
        conf *= 1.0 + band
    conf = max(CONF_MIN, min(CONF_MAX, conf))
    return {
        "confidence": round(conf, 4),
        "confidence_exact": conf,
        "base_confidence": round(base, 4),
        "snr": round(snr, 3),
        "noise_floor": round(floor, 4),
        "modulator": round(modulator, 4),
        "modulator_bounded": True,
        "issued": conf > CONFIDENCE_FLOOR,
        "aligned": bool(state_aligned),
        "wfcp": wfcp, "drpt": drpt, "gspi": gspi, "rtp": rtp, "pcv": pcv,
        # exactly what the CHAIN panel prints, step by step
        "steps": [
            {"step": 1, "label": "|CCI_ADJ| × MIN(1, DRHE) — SNR → tanh",
             "value": round(base, 5),
             "detail": f"|{cci_adj:+.4f}| / floor {floor:.3f} = snr "
                       f"{snr:.2f} → tanh({BASE_GAIN}·snr)·DRHE "
                       f"{drhe:.2f}"},
            {"step": 2, "label": "×(0.90 + 0.10 · EQ) [brain B2]",
             "value": round(eq_f, 4), "detail": f"EQ {experience_quality:.2f}"},
            {"step": 3, "label": "+0.05 · base · P(dominant) [brain B3, aligned]",
             "value": round(align_bonus, 5),
             "detail": f"P(dom) {p_dom:.2f} · aligned {bool(state_aligned)}"},
            {"step": 4, "label": "×(1 − 0.12·ramp RTP) × (1 − 0.10·|PCV|)",
             "value": round(rtp_f * pcv_f, 4),
             "detail": f"RTP {rtp:.3f} · PCV {pcv:+.3f}"},
            {"step": 5, "label": "×(0.92 + 0.08 · WFCP) [P2]",
             "value": round(wfcp_f, 4), "detail": f"WFCP {wfcp:.3f}"},
            {"step": 6, "label": "×(1 − 0.08 · DRPT) [P6]",
             "value": round(drpt_f, 4), "detail": f"DRPT {drpt:.3f}"},
            {"step": 7, "label": "×(1 − 0.06 · GSPI) [P19]",
             "value": round(gspi_f, 4), "detail": f"GSPI {gspi:.3f}"},
            {"step": 8, "label": "clip(base·modulator, 0.02, 0.98)",
             "value": round(conf, 5),
             "detail": f"base {base:.4f} × modulator {modulator:.4f}"
                       + (f" + align {align_bonus:.4f}" if align_bonus else "")},
        ],
    }


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0
