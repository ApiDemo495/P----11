"""Section 5 — EMOTIONAL STATE, rebuilt as a measurement.

Why this file was rewritten
---------------------------
The previous version kept 21 numbered channels and *named* each one after a
published construct (Russell valence, Ekman fear, Kahneman loss pressure …),
but most of the numbers under those names were proxies that did not measure
the construct:

* **the dominant channel was a saturated one.** Several channels used
  ``sigmoid(4·x)`` / ``tanh(6·x)`` gains that pin to 0.999 or 0.001 on an
  ordinary window, so "dominant" was decided by which formula saturated
  first, not by which state was actually strongest;
* **three channels were arithmetic that measured nothing in particular.**
  "somatic marker" started at a constant 0.50 and only moved after five
  settled outcomes; "breadth" needed a peer cross-section that a single
  asset rarely has; "news" reported 0.50 when no headline was scored — a
  number that looks like a measurement and is not;
* **the Fear & Greed index was renormalised, not composed.** Only 50 % of
  alternative.me's published weights were computed, and the printed
  ``index`` was that 50 % rescaled to 0-100 — which reads as the real index;
* **dominance had no independent source.** It was a function of the same
  drift and volatility that produced valence and arousal, so the third PAD
  axis carried no information beyond the first two;
* **the reward rule was blind to magnitude** — a 3σ move and a 0.2σ move
  taught the same amount.

What it measures now
--------------------
Every channel is a stated statistic over the tape, each one carries its own
neutral and its own citation, and every abstention is declared with a
machine-readable cause instead of a middle-of-the-road number:

======================  ============================================="""
from __future__ import annotations

import math
import time

from .config import HIST_EMOTION_LAG
from .formulas._helpers import sign, std_pop
from .models import FormulaContext

# ── learning rule constants (Section 10) ────────────────────────────────
ETA = 0.015               # η_emotion floor
W_CLAMP_LO = 0.005        # per-channel share bounds
W_CLAMP_HI = 0.30
ETA_COLD = 0.08           # cold-start rate — a fresh brain learns FAST
COLD_HALFLIFE = 150
ACC_SMOOTH = 0.90
EMO_IDS = list(range(1, 22))          # channel ids (22 = the CERC composite)
CERC_WEIGHT_ID = 22

# ── literature constants ────────────────────────────────────────────────
LOSS_LAMBDA = 2.25        # Tversky & Kahneman (1992) median λ
EKMAN_SIGMA = 0.55        # prototype kernel width in (v,a,d) ∈ [-1,1]³
FG_WEIGHTS = {"volatility": 0.25, "momentum_volume": 0.25,
              "social": 0.15, "surveys": 0.15, "dominance": 0.10,
              "trends": 0.10}                     # alternative.me weights
# alternative.me's "dominance" is BTC *market-cap share*, which needs cap
# data this app does not have; our breadth channel is a different statistic
# and is therefore reported as its own reading, never substituted into the
# dominance slot. Only these two components are actually measured.
FG_MEASURED = ("volatility", "momentum_volume")
FG_MISSING = {
    "dominance": ("BTC market-cap share needs a cap feed this app does not "
                  "have — our breadth channel is a different statistic and "
                  "is reported separately, not substituted"),
    "social": "no social-media firehose connected — not estimated",
    "surveys": "no investor survey feed connected — not estimated",
    "trends": "no search-trends feed connected — not estimated",
}

# Ekman (1992) prototypes in (valence, arousal, dominance) ∈ [-1,1]³.
# Positions follow Russell (1980) octants (excitement 45°, distress 135°,
# displeasure 180°, depression 225°) plus Mehrabian & Russell (1974) for the
# dominance axis (anger dominates, fear submits).
PROTOTYPES = {
    "fear":      (-0.70, 0.80, -0.60),
    "anger":     (-0.50, 0.70, 0.70),
    "disgust":   (-0.60, 0.20, 0.40),
    "sadness":   (-0.70, -0.50, -0.60),
    "surprise":  (0.00, 0.90, 0.00),
    "enjoyment": (0.80, 0.30, 0.50),
}

#: alternative.me Crypto Fear & Greed published component weights.
FG_WEIGHTS = {"volatility": 0.25, "momentum_volume": 0.25,
              "social": 0.15, "surveys": 0.15, "dominance": 0.10,
              "trends": 0.10}
#: the components this app can actually measure (the rest are declared)
FG_MEASURED = ("volatility", "momentum_volume")
FG_MISSING = {
    "dominance": ("BTC market-cap share needs a cap feed this app does not "
                  "have — our breadth channel is a different statistic and "
                  "is reported separately, never substituted"),
    "social": "no social-media firehose connected — not estimated",
    "surveys": "no investor survey feed connected — not estimated",
    "trends": "no search-trends feed connected — not estimated",
}
#: measured weight of the published index (volatility + momentum/volume)
FG_COVERAGE = round(sum(FG_WEIGHTS[k] for k in FG_MEASURED), 4)

#: A channel only VOTES on direction when it is at least this far from its
#: own neutral (in neutral-relative units). Below the band it is still
#: printed as a reading, but it cannot add to the agreement sum: a channel
#: reading 0.0005 against a 0.0 neutral is noise wearing a direction.
#: This is the same discipline the CCI applies to its own deadband.
MOVE_BAND = 0.10

#: Ekman (1992) prototypes in (valence, arousal, dominance) ∈ [-1,1]³.
#: Positions follow Russell (1980) octants (excitement 45°, distress 135°,
#: displeasure 180°, depression 225°) plus Mehrabian & Russell (1974) for the
#: dominance axis (anger dominates, fear submits).
PROTOTYPES = {
    "fear":      (-0.70, 0.80, -0.60),
    "anger":     (-0.50, 0.70, 0.70),
    "disgust":   (-0.60, 0.20, 0.40),
    "sadness":   (-0.70, -0.50, -0.60),
    "surprise":  (0.00, 0.90, 0.00),
    "enjoyment": (0.80, 0.30, 0.50),
}
EKMAN_DIR = {"fear": -1, "anger": -1, "disgust": -1, "sadness": -1,
             "surprise": 0, "enjoyment": 1}
_EK_IDS = {"fear": 6, "anger": 7, "disgust": 8, "sadness": 9,
           "surprise": 10, "enjoyment": 11}

#: id → (key, printed name, citation, neutral value)
#: E1–E21 rewritten from first principles (§3 directive): every channel
#: is a stated microstructure statistic — event channels sit at neutral
#: 0.0 (absence = "not happening"), state channels at 0.5 (bipolar).
CHANNELS = {
    # ── E1–E5  behavioral panic, capitulation & tail stress ──────────
    1: ("panic_velocity", "Fear & capitulation velocity",
        "downside semi-variance acceleration × volume climax × lower-band "
        "excursion", 0.0),
    2: ("underwater_pain", "Cohort underwater pain (loss aversion)",
        "VWAP-anchor cost basis vs spot, ATR-normalized · Kahneman & "
        "Tversky 1979", 0.5),
    3: ("tail_hedge", "Tail-hedge & liquidity withdrawal",
        "downside kurtosis jump + range-spread expansion (GK estimator)",
        0.0),
    4: ("liquidation_cascade", "Liquidation cascade pressure",
        "directional acceleration × tick-intensity spike × wick-rejection "
        "asymmetry", 0.0),
    5: ("startle_shock", "Startle shock (Lee–Mykland)",
        "|return| / bipower variation — exogenous shock impulse", 0.0),
    # ── E6–E11  smart-money vs retail crowd divergence ───────────────
    6: ("session_greed", "Intraday session greed",
        "session return & volume excess vs same-UTC-hour baseline", 0.5),
    7: ("wyckoff_absorption", "Wyckoff smart-money absorption",
        "(volume_z − range_z) signed by close-location value", 0.5),
    8: ("vol_compression", "Volatility compression stress",
        "percentile of RV15 / RV240 — coiled-spring tension", 0.5),
    9: ("carry_pressure", "Basis & funding-proxy carry pressure",
        "cumulative signed drift + pre-funding premium proxy", 0.5),
    10: ("whale_skew", "Whale vs retail trade-size skew",
         "avg quote volume per trade, recent vs baseline", 0.5),
    11: ("herd_sync", "Herd synchronization (cross-asset)",
         "rolling BTC/ETH/BNB correlation spike", 0.5),
    # ── E12–E17  exhaustion, traps & mean-reversion psychology ───────
    12: ("chop_fatigue", "Range chop fatigue",
         "Kaufman efficiency-ratio decay (net/path)", 0.0),
    13: ("breakout_trap", "Failed-breakout bull/bear trap",
         "N-bar extreme pierce on declining volume, snap-back in 1–3 bars",
         0.0),
    14: ("news_confirmation", "News-price confirmation vs divergence",
         "NIV polarity × realized post-headline follow-through", 0.5),
    15: ("breakout_kinetic", "Breakout kinetic energy",
         "volume acceleration × true-range expansion × flow delta out of "
         "compression", 0.0),
    16: ("headline_half_life", "Headline shock half-life",
         "exponentially decayed volatility impulse after high-tier news",
         0.0),
    17: ("level_gravity", "Psychological level & anchored-VWAP gravity",
         "distance & velocity toward round numbers and session VWAP", 0.5),
    # ── E18–E21  rebuilt core channels ───────────────────────────────
    18: ("chase_fatigue", "Trend-chase fatigue",
         "RSI/stochastic overextension × ΔP per unit volume decay",
         0.5),
    19: ("smart_divergence", "Smart-money divergence (OBV/MFI vs price)",
         "multi-bar slope divergence: money-flow vs price", 0.5),
    20: ("news_fomo", "News FOMO overreaction",
         "extreme news sentiment × >2.2σ short-term extension", 0.0),
    21: ("climax_streak", "Climax exhaustion streak",
         "signed consecutive candle streak × relative-volume blow-off "
         "(Vol > 2.5 × MA)", 0.0),
}
CHANNEL_KEYS = {i: CHANNELS[i][0] for i in EMO_IDS}

MODEL = {
    "name": "behavioral microstructure E1–E22 (panic · smart-money · "
            "exhaustion · meta) + derived core-affect/Ekman display",
    "citations": [
        "Russell, J. A. (1980). A circumplex model of affect. JPSP 39(6).",
        "Ekman, P. (1992). An argument for basic emotions. Cognition & "
        "Emotion 6(3-4).",
        "Kahneman, D., & Tversky, A. (1979). Prospect theory. Econometrica "
        "47(2); Tversky & Kahneman (1992) λ=2.25, J. Risk & Uncertainty 5.",
        "Lee, S. S. W., & Mykland, P. (2008). Jumps in financial markets: "
        "a new nonparametric test and jump dynamics. JFE 88(2).",
        "Barberis, N., Greenwood, R., & Shleifer, A. (2015). X-CAPM. "
        "J. Financial Economics 115(1).",
        "Christie, W., & Huang, R. (1995). Following the pied piper: herding "
        "in equity returns. Financial Analysts Journal 51(3).",
        "Frijda, N. (1986). The Emotions (action tendencies).",
        "Wyckoff, R. (1910). Studies in tape reading (absorption "
        "accumulation/distribution).",
        "alternative.me Crypto Fear & Greed Index methodology (volatility "
        "25 %, momentum/volume 25 %, social 15 %, surveys 15 %, dominance "
        "10 %, trends 10 %).",
    ],
    # the UI measures every channel against ITS OWN neutral — event
    # channels sit at 0.0, state channels at 0.5 — so the neutral ships
    # with the name and the citation
    "channels": {str(i): {"name": CHANNELS[i][1], "cite": CHANNELS[i][2],
                          "key": CHANNELS[i][0], "neutral": CHANNELS[i][3]}
                 for i in EMO_IDS},
    "fear_greed_measured_weight": FG_COVERAGE,
    "fear_greed_missing": dict(FG_MISSING),
    "invariant": ("every channel is a stated statistic with its own neutral; "
                  "an abstention is declared, never imputed"),
}


# ══════════════════════════════════════════════════════════════════════
# small numeric helpers
# ══════════════════════════════════════════════════════════════════════
def sig(x: float) -> float:
    """Logistic — kept for the reward/weight code and external callers."""
    x = float(x)
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 60.0)))
    z = math.exp(max(x, -60.0))
    return z / (1.0 + z)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _finite(*vals) -> bool:
    for v in vals:
        try:
            if not math.isfinite(float(v)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def graded(x: float, gain: float = 1.0, lo: float = 0.02,
           hi: float = 0.98) -> float:
    """Map a signed score to [lo, hi] with a *graded* centre.

    ``0.5 + 0.5·tanh(gain·x)`` is monotone but asymptotically flat: at
    ``gain·x = 4`` it prints 0.9993 and at 6 it prints 0.99999. A channel
    that prints 0.999 is claiming certainty the tape does not contain, and
    worse, several of them tie at the top so "dominant" becomes arbitrary.

    This version keeps the sign and the ordering, then rescales into
    ``[lo, hi]`` so the extreme is *approachable but never reached* — the
    same tape that read 0.9999 now reads 0.978, and two different strengths
    stay distinguishable.
    """
    t = math.tanh(max(-20.0, min(20.0, float(gain) * float(x))))
    return lo + (hi - lo) * (0.5 + 0.5 * t)


def _compress(z: float, scale: float = 1.6) -> float:
    """asinh compression: linear near zero, logarithmic in the tails.

    A 2σ move and a 10σ move must not both print "extreme", or the axis
    stops carrying information.
    """
    return math.asinh(max(-1e6, min(1e6, float(z))) / max(1e-9, scale))


def _lag_value(ring) -> float | None:
    """GRC-lagging comparator: the value from HIST_EMOTION_LAG GRCs ago."""
    if len(ring) == 0:
        return None
    if len(ring) >= HIST_EMOTION_LAG:
        return ring.values()[0]
    return ring.last()


def _neg_frac(closes: list[float], n: int) -> float:
    if len(closes) < 2:
        return 0.0
    tail = closes[-min(n + 1, len(closes)):]
    steps = [(tail[i] - tail[i - 1]) / tail[i - 1]
             for i in range(1, len(tail)) if tail[i - 1]]
    if not steps:
        return 0.0
    return sum(1 for r in steps if r < 0) / len(steps)


def _atr5(ohlc: dict | None, closes: list[float]) -> float:
    if ohlc:
        h = ohlc.get("h") or ohlc.get("highs") or []
        l = ohlc.get("l") or ohlc.get("lows") or []
        c = ohlc.get("c") or ohlc.get("closes") or []
        n = min(len(h), len(l), len(c))
        if n >= 2:
            trs = []
            for i in range(max(1, n - 5), n):
                pc = c[i - 1]
                trs.append(max(h[i] - l[i], abs(h[i] - pc), abs(l[i] - pc)))
            if trs:
                return max(1e-9, sum(trs) / len(trs))
    if len(closes) < 6:
        return 1e-9
    trs = [abs(closes[i] - closes[i - 1]) for i in range(-5, 0)]
    return max(1e-9, sum(trs) / len(trs))


def _causes(zero_reasons: dict) -> dict:
    """Cause bucket per abstention (error / both sides equal / input / …)."""
    from .formulas.trace import classify_zero
    return {k: classify_zero(v) for k, v in zero_reasons.items()}


def _semi_deviations(rets: list[float]) -> tuple[float, float]:
    """Upside / downside semi-deviation (realised, not assumed symmetric)."""
    if not rets:
        return 0.0, 0.0
    ups = [r for r in rets if r > 0]
    dns = [-r for r in rets if r < 0]
    up = math.sqrt(sum(r * r for r in ups) / len(rets))
    dn = math.sqrt(sum(r * r for r in dns) / len(rets))
    return up, dn


def _run_persistence(rets: list[float], side: int) -> float:
    """Longest same-signed run as a share of the window (dominance input).

    Mehrabian & Russell's dominance axis is *felt control*. On a tape, the
    defensible control proxy is not how far price moved (that is valence) but
    how consistently it moved one way — a persistent drip is control, a
    violent whipsaw that ends at the same price is not.
    """
    if not rets or side == 0:
        return 0.0
    best = cur = 0
    for r in rets:
        if sign(r) == side and r != 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best / len(rets)


def _z(x: float, ref: list[float]) -> float:
    """z-score of x against a reference series (0 when no spread)."""
    if not ref:
        return 0.0
    m = sum(ref) / len(ref)
    s = std_pop(ref)
    if s <= 1e-12:
        return 0.0
    return (float(x) - m) / s


def _pct_rank(vals: list[float], x: float) -> float:
    if not vals:
        return 0.5
    return sum(1 for v in vals if v < x) / len(vals)


def _ev(z: float, dead: float = 0.35, gain: float = 1.0) -> float:
    """Event-channel mapper (neutral 0.0): a reading at or below the
    deadband prints the 0.02 floor — 'not happening' — and rises
    smoothly above it, so ordinary tape variation keeps the channel
    alive (sd > 0.08) without ever faking an event."""
    x = max(0.0, float(z) - dead)
    return _clamp(0.02 + 0.96 * math.tanh(gain * x), 0.02, 0.98)


def ekman_scores(valence: float, arousal: float,
                 dominance: float) -> dict[str, dict]:
    """Similarity of the current affect to Ekman's basic-emotion prototypes.

    valence/arousal/dominance arrive in [0,1]; prototypes live in [-1,1]³.
    """
    v = 2.0 * float(valence) - 1.0
    a = 2.0 * float(arousal) - 1.0
    d = 2.0 * float(dominance) - 1.0
    out: dict[str, dict] = {}
    for name, (pv, pa, pd_) in PROTOTYPES.items():
        dist = math.sqrt((v - pv) ** 2 + (a - pa) ** 2 + (d - pd_) ** 2)
        intensity = math.exp(-(dist ** 2) / (2.0 * EKMAN_SIGMA ** 2))
        out[name] = {"intensity": round(_clamp(intensity, 0.0, 0.99), 4),
                     "distance": round(dist, 4),
                     "prototype": {"valence": pv, "arousal": pa,
                                   "dominance": pd_}}
    return out


def fear_greed_index(*, vol_stress: float, momentum: float,
                     herding: float = 0.5) -> dict:
    """Crypto Fear & Greed (alternative.me) over the components we measure.

    All inputs are [0,1] channel values; 1 = greed-side of that component.
    Social / surveys / trends / dominance are NOT estimated. Two numbers are
    printed and they answer different questions:

    * ``index`` — the composite renormalised over the measured 50 % of the
      published weight (this is what a dashboard can honestly show);
    * ``published_index`` — the same two components multiplied by their
      *published* weights and added, i.e. the index as alternative.me would
      score it if the three unmeasured components sat at neutral 50. It is
      always lower-magnitude than ``index`` and therefore never overstates.
    ``partial`` is True whenever less than 100 % of the weight is measured,
    and ``coverage`` prints the fraction, so the number cannot be mistaken
    for the full index.
    """
    comps = {
        "volatility": _clamp(1.0 - float(vol_stress), 0.0, 1.0) * 100.0,
        "momentum_volume": _clamp(float(momentum), 0.0, 1.0) * 100.0,
    }
    wsum = sum(FG_WEIGHTS[k] for k in comps) or 1.0
    index = sum(FG_WEIGHTS[k] * v for k, v in comps.items()) / wsum
    # index as the published weights would score it, unmeasured → neutral 50
    published = sum(FG_WEIGHTS[k] * comps.get(k, 50.0)
                    for k in FG_WEIGHTS)
    label = ("Extreme fear" if index <= 24 else "Fear" if index <= 44 else
             "Neutral" if index <= 55 else "Greed" if index <= 75 else
             "Extreme greed")
    return {"index": round(index, 1), "label": label,
            "published_index": round(published, 1),
            "partial": wsum < 0.999,
            "coverage": round(wsum, 4),
            "components": {k: round(v, 1) for k, v in comps.items()},
            "weights": {k: FG_WEIGHTS[k] for k in comps},
            "measured_weight": round(wsum, 4),
            "unmeasured": {k: FG_MISSING[k] for k in FG_MISSING},
            "note": ("index renormalised over the measured "
                     f"{round(100 * wsum)} % of the published weighting; "
                     "published_index assumes the unmeasured components sit "
                     "at neutral. Neither is presented as the full index"),
            "herding_damping": round(0.5 + 0.5 * _clamp(herding, 0.0, 1.0), 3)}


def action_tendency(*, fear: float, anger: float, sadness: float,
                    enjoyment: float, pa: float, na: float,
                    dominance: float, capitulation: float,
                    trend_dir: int = 0) -> dict:
    """Frijda (1986): what the state is ready to DO.

    Returns the readiness, its urge in [0,1] and the market direction that
    readiness implies (approach = buy, avoid = sell, withdraw = stand aside).
    """
    rules = [
        ("avoid", fear, -1,
         "fear + low control → reduce exposure"),
        ("confront", anger, (-int(trend_dir) if trend_dir else -1),
         "anger + high control → fight the move (fade it)"),
        ("withdraw", capitulation, 0,
         "capitulation → stand aside, do not catch the knife"),
        ("disengage", sadness, 0,
         "sadness/low arousal → no participation"),
        ("approach", enjoyment * (0.5 + 0.5 * pa), +1,
         "enjoyment + positive affect → add exposure"),
    ]
    weight = {"avoid": 1.0, "confront": 0.75, "withdraw": 0.9,
              "disengage": 0.6, "approach": 1.0}
    for name, _, _, _ in rules:
        pass
    scored = [(name, _clamp(v, 0.0, 1.0) * weight[name], direction, note)
              for name, v, direction, note in rules]
    name, urge, direction, note = max(scored, key=lambda t: t[1])
    if urge < 0.25:
        return {"name": "monitor", "urge": round(urge, 4), "direction": 0,
                "note": "no tendency strong enough to act on — watch only",
                "runner_up": name}
    return {"name": name, "urge": round(urge, 4), "direction": direction,
            "note": note,
            "runner_up": max((s for s in scored if s[0] != name),
                             key=lambda t: t[1])[0]}


# ══════════════════════════════════════════════════════════════════════
# the system
# ══════════════════════════════════════════════════════════════════════
def _cerc_vectors(emos: dict, dirs: dict, w: list) -> dict:
    """E22 — the agreement sum decomposed into the three telemetry
    vectors the directive names: Retail Herd, Smart-Money Flow, and the
    Crowd-vs-Smart split (high-conviction contrarian warning)."""

    def _vec(ids, flip: bool) -> float:
        num = den = 0.0
        for eid in ids:
            if eid not in emos or eid not in CHANNELS:
                continue
            neutral = CHANNELS[eid][3]
            lean = (-dirs.get(eid, 0)) if flip else dirs.get(eid, 0)
            dev = (emos[eid] - neutral) * lean
            wt = float(w[eid - 1]) if w and eid - 1 < len(w) else 1.0
            num += dev * wt
            den += wt * (1.0 - neutral)
        return _clamp(num / den, -1.0, 1.0) if den > 0 else 0.0

    # retail channels vote FADE in CERC (buy-panic, fade-chase), so the
    # crowd's own lean is the flipped direction
    retail = _vec((18, 20, 1, 11, 21), flip=True)
    # smart channels already carry the professional's side
    smart = _vec((7, 10, 19, 9), flip=False)
    split = _clamp(max(0.0, retail) * max(0.0, -smart) * 2.5, 0.0, 1.0)
    return {
        "retail_herd": round(retail, 4),
        "smart_money": round(smart, 4),
        "crowd_smart_split": round(split, 4),
        "note": ("crowd_smart_split near 1.0 = retail chasing while smart "
                 "money distributes (high-conviction contrarian warning); "
                 "near 0.0 = same side or both quiet"),
    }


class EmotionSystem:
    """Computes the affect state (see module docstring) + learns weights."""

    def __init__(self) -> None:
        self.weights: dict[tuple[str, str], list[float]] = {}
        self.acc: dict = {}        # (sym, tf) -> 21 EWMA accuracies
        self.n_updates: dict = {}  # (sym, tf) -> reward-call count
        # dominant-channel tracking: previous pass + since-when, so the
        # DOMINANT NOW header is measured server-side (survives reloads)
        self._prev_emos: dict = {}
        self._dom: dict = {}       # (sym, tf) -> (channel id, since ts)
        # somatic marker (Damasio): learned from this system's own outcomes
        self._soma: dict = {}

    @property
    def ready(self) -> bool:
        return True

    def initialize(self) -> None:
        """No external dependencies (history rings bootstrap themselves)."""
        return

    def reset_weights(self) -> None:
        self.weights.clear()

    # ------------------------------------------------------------------
    def _weights(self, symbol: str, tf: str) -> list[float]:
        key = (symbol, tf)
        w = self.weights.get(key)
        if w is None:
            w = [1.0 / 21.0] * 21
            self.weights[key] = w
        return w

    def _accuracy(self, symbol: str, tf: str) -> list[float]:
        """EWMA predictive accuracy per channel (0.5 = uninformative)."""
        key = (symbol, tf)
        a = self.acc.get(key)
        if a is None:
            a = [0.5] * 21
            self.acc[key] = a
        return a

    # ------------------------------------------------------------------
    def compute(
        self,
        ctx: FormulaContext,
        market: dict,
        brain_res: dict,
        news_agg: dict,
        prev_cerc: float | None,
    ) -> dict:
        cl = list(ctx.P or [])                      # timeframe window closes
        mc = list(market.get("closes") or [])       # long window (>= 60 mins)
        if not mc:
            mc = list(cl)
        # a slice with no prices at all must not raise: report an empty,
        # fully-abstained state instead (the engine treats it as "warmup")
        if not mc:
            return self._empty_state(ctx, prev_cerc)
        open_ = market.get("session_open") or mc[0]
        live = market.get("live") or (cl[-1] if cl else mc[-1])
        if not _finite(open_, live) or open_ <= 0 or live <= 0:
            return self._empty_state(ctx, prev_cerc)

        # ---- measured tape features (never the signal — G-3) -----------
        rets = [(mc[i] - mc[i - 1]) / mc[i - 1] for i in range(1, len(mc))
                if mc[i - 1]]
        # The drift t-statistic must divide a return by the standard error OF
        # THAT SAME WINDOW. The old code divided the return over the long
        # market window (up to 600 bars) by ``σ·√(W)`` where W is the
        # timeframe window (22 bars) — so a 200-bar random walk with zero
        # drift produced a "significicant" drift and a flat tape read
        # valence 0.69 / "Extreme greed". Both terms now come from the
        # timeframe window, and the long window only supplies the σ estimate.
        if len(cl) >= 2 and cl[0]:
            n_w = max(2, len(cl) - 1)
            r_w = cl[-1] / cl[0] - 1.0
            session_ret = (live / open_ - 1.0) if open_ else 0.0
        else:
            n_w = max(2, len(mc) - 1)
            r_w = session_ret = (live / open_ - 1.0) if open_ else 0.0
        s1 = std_pop(rets[-10:]) if len(rets) >= 2 else 0.0
        s30 = std_pop(rets[-min(30, len(rets)):]) if len(rets) >= 2 else 0.0
        vols = market.get("volumes") or [0.0] * len(mc)
        v5 = sum(vols[-5:]) / min(5, len(vols)) if vols else 0.0
        v30 = sum(vols[-30:]) / min(30, len(vols)) if vols else 0.0
        hist = ctx.hist
        zero_reasons: dict[str, str] = {}
        emos: dict[int, float] = {}
        dirs: dict[int, int] = {}

        # drift t-stat of the window: the single most defensible "how strong
        # is the current move" measure (return / its own standard error)
        se = s30 * math.sqrt(n_w) if (s30 > 0 and n_w > 0) else 0.0
        z_drift = (r_w / se) if se > 1e-12 else 0.0
        z_drift = _clamp(z_drift, -8.0, 8.0)
        move_dir = sign(r_w)

        # ═══ E1–E21 behavioral microstructure channels (§3 rewrite) ═══
        # Every channel is a stated statistic over THIS tape; event
        # channels (neutral 0.0) print the 0.02 floor when the event is
        # absent, state channels (neutral 0.5) are bipolar.
        ohlc = market.get("ohlc") or {}
        highs = list(ohlc.get("highs") or [])
        lows = list(ohlc.get("lows") or [])
        if len(highs) != len(mc):
            highs = list(mc)
        if len(lows) != len(mc):
            lows = list(mc)
        times = list(market.get("times") or ctx.T or [])
        atr = _atr5(ohlc, mc)
        n_b = len(mc)
        # volume stats + climax (shared)
        v_mean30 = v30 if v30 > 0 else (sum(vols) / len(vols) if vols else 0.0)
        vol_z = _z(v5, vols[-min(60, len(vols)):]) if vols else 0.0
        climax = (v5 / v_mean30) if v_mean30 > 0 else 1.0
        vol_ratio = (s1 / max(s30, 1e-12)) if s30 > 0 else 1.0
        d1 = mc[-1] - mc[-6] if n_b > 6 else 0.0
        d2 = mc[-6] - mc[-11] if n_b > 11 else 0.0
        ma30 = (sum(mc[-min(30, n_b):]) / min(30, n_b)) if n_b else 0.0
        persist = _run_persistence(rets[-min(30, len(rets)):], move_dir)
        # range / wick helpers aligned to mc
        ranges = [max(highs[i] - lows[i], 0.0) for i in range(n_b)]
        clv = ((mc[-1] - lows[-1]) / (highs[-1] - lows[-1]) - 0.5) \
            if (highs[-1] > lows[-1]) else 0.0

        # ── E1 FEAR & CAPITULATION VELOCITY (event) ────────────────────
        # downside semi-variance acceleration × volume climax × lower-band
        # excursion depth
        _dn_now = sum(r * r for r in rets[-10:] if r < 0) / max(1, len(rets[-10:]))
        _dn_prev = (sum(r * r for r in rets[-30:-10] if r < 0)
                    / max(1, len(rets[-30:-10]))) if len(rets) >= 30 else _dn_now
        _sv_accel = _dn_now / _dn_prev if _dn_prev > 1e-18 else 1.0
        _lb = None
        if n_b >= 20:
            _m20 = sum(mc[-20:]) / 20.0
            _s20 = std_pop(mc[-20:])
            _lb = _m20 - 2.0 * _s20
        _exc = max(0.0, (_lb - live) / atr) if (_lb is not None and atr > 0) else 0.0
        _e1 = (max(0.0, _sv_accel - 1.0) * 1.1
               + max(0.0, climax - 1.0) * 1.4 + _exc * 1.6)
        emos[1] = _ev(_e1, dead=0.25, gain=1.1)
        dirs[1] = 1                       # contrarian: fear lows buy

        # ── E2 COHORT UNDERWATER PAIN (state) ─────────────────────────
        # volume-weighted average entry over the rolling window vs spot,
        # normalized by ATR: buyers underwater → forced liquidation pain
        _pv = sum(mc[i] * vols[i] for i in range(n_b)
                  if i < len(vols) and vols[i] > 0)
        _vv = sum(vols[i] for i in range(n_b)
                  if i < len(vols) and vols[i] > 0)
        _vwap = (_pv / _vv) if _vv > 0 else (ma30 or live)
        _pain = ((_vwap - live) / atr) if atr > 0 else 0.0
        emos[2] = graded(_pain, gain=0.55)
        dirs[2] = -1                      # pain = bearish pressure

        # ── E3 TAIL-HEDGE & LIQUIDITY WITHDRAWAL (event) ──────────────
        # realized downside kurtosis jump + range-spread expansion (GK)
        _r60 = rets[-min(60, len(rets)):]
        _kurt = 0.0
        if len(_r60) >= 20 and s30 > 0:
            _m = sum(_r60) / len(_r60)
            _m4 = sum((x - _m) ** 4 for x in _r60) / len(_r60)
            _kurt = (_m4 / (s30 ** 4) - 3.0) if s30 > 1e-12 else 0.0
        _gk = [(highs[i] - lows[i]) / mc[i] for i in range(n_b)
               if mc[i] > 0 and i < len(ranges)]
        _spread_z = _z(_gk[-1], _gk[-min(60, len(_gk)):]) if _gk else 0.0
        emos[3] = _ev(max(0.0, _kurt) * 0.8 + max(0.0, _spread_z) * 1.1,
                      dead=0.4, gain=1.0)
        dirs[3] = -1                      # hedging = risk-off

        # ── E4 LIQUIDATION CASCADE PRESSURE (event) ───────────────────
        # consecutive directional acceleration × tick-intensity spike ×
        # wick-rejection asymmetry (downside cascade dominates)
        _streak = 0
        _accel = 0.0
        for i in range(len(rets) - 1, max(0, len(rets) - 8), -1):
            if sign(rets[i]) == move_dir and rets[i] != 0:
                _streak += 1
                if i > 0 and abs(rets[i]) > abs(rets[i - 1]):
                    _accel += 1.0
            else:
                break
        _wick = 0.0
        if highs[-1] > lows[-1]:
            _body = mc[-1] - (mc[-2] if n_b > 1 else mc[-1])
            _rng = highs[-1] - lows[-1]
            _wick = max(0.0, -(_body / _rng)) if move_dir <= 0 else \
                max(0.0, (_body / _rng) if _body < 0 else 0.0)
        _e4 = _streak * 0.35 + _accel * 0.45 + max(0.0, vol_z) * 0.8 + _wick * 1.2
        emos[4] = _ev(_e4, dead=0.8, gain=0.8)
        dirs[4] = 1                       # contrarian: cascade lows bounce

        # ── E5 STARTLE SHOCK (Lee–Mykland, event) ─────────────────────
        _bp_terms = [abs(rets[i]) * abs(rets[i - 1])
                     for i in range(1, max(1, len(rets) - 1))]
        _bp = (sum(_bp_terms) / len(_bp_terms) * (math.pi / 2.0)) \
            if _bp_terms else 0.0
        _bp = math.sqrt(_bp) if _bp > 0 else 0.0
        _lm = (abs(rets[-1]) / _bp) if (_bp > 1e-12 and rets) else 1.0
        emos[5] = _ev(_lm - 1.0, dead=0.5, gain=1.2)
        dirs[5] = 0                       # shock magnitude, not a side

        # ── E6 INTRADAY SESSION GREED (state) ─────────────────────────
        # session return & volume excess vs the same-UTC-hour seasonal
        # baseline of this tape (falls back to the tape's own baseline)
        _hour_rets: dict[int, list[float]] = {}
        for i in range(1, min(n_b, len(times))):
            if times[i - 1] and mc[i - 1]:
                _h = ((times[i - 1] % 86400) // 3600)
                _hour_rets.setdefault(_h, []).append(
                    (mc[i] - mc[i - 1]) / mc[i - 1])
        _cur_h = ((times[-1] % 86400) // 3600) if times else 12
        _season = _hour_rets.get(_cur_h) or []
        _sret_z = _z(session_ret, _season) if len(_season) >= 10 else \
            _z(session_ret, rets[-min(120, len(rets)):])
        _svols_z = _z(sum(vols[-min(60, len(vols)):]) / max(1, min(60, len(vols))),
                      vols[-min(240, len(vols)):]) if vols else 0.0
        emos[6] = graded(0.55 * _sret_z + 0.45 * _svols_z, gain=1.0)
        dirs[6] = sign(_sret_z) if abs(_sret_z) > 0.2 else 0

        # ── E7 WYCKOFF SMART-MONEY ABSORPTION (state) ─────────────────
        # high relative volume inside a compressed range, signed by CLV
        _rng5 = sum(ranges[-min(5, n_b):]) / min(5, n_b) if n_b else 0.0
        _rng30 = sum(ranges[-min(30, n_b):]) / min(30, n_b) if n_b else 0.0
        _vz = _z(v5, vols[-min(60, len(vols)):]) if vols else 0.0
        _rz = _z(_rng5, ranges[-min(60, n_b):]) if n_b else 0.0
        _absorb = (_vz - _rz) * 0.6 + clv * 2.0
        emos[7] = graded(_absorb, gain=1.1)
        dirs[7] = sign(_absorb) if abs(_absorb) > 0.2 else 0

        # ── E8 VOLATILITY COMPRESSION STRESS (state) ──────────────────
        # percentile of RV15/RV240: tension builds as short-horizon vol
        # compresses under the long-horizon baseline (coiled spring)
        _rv15 = std_pop(rets[-min(15, len(rets)):]) if len(rets) >= 5 else 0.0
        _rv240 = std_pop(rets[-min(240, len(rets)):]) if len(rets) >= 20 else s30
        _ratio = (_rv15 / _rv240) if _rv240 > 1e-12 else 1.0
        emos[8] = graded(-math.log(max(_ratio, 1e-6)), gain=1.0)
        dirs[8] = 0                       # tension, not a side

        # ── E9 BASIS & FUNDING-PROXY CARRY PRESSURE (state) ───────────
        # cumulative signed return drift + pre-funding premium proxy
        _drift24 = sum(rets[-min(24, len(rets)):]) if rets else 0.0
        _prem = ((live - ma30) / ma30) if ma30 > 0 else 0.0
        _carry = _z(_drift24, rets[-min(240, len(rets)):]) * 0.6 \
            + (_prem / max(s30, 1e-9)) * 0.4 if rets else 0.0
        emos[9] = graded(_clamp(_carry, -4.0, 4.0), gain=0.9)
        dirs[9] = sign(_carry) if abs(_carry) > 0.25 else 0

        # ── E10 WHALE vs RETAIL TRADE-SIZE SKEW (state) ───────────────
        sizes = list(market.get("trade_sizes") or [])
        if len(sizes) >= 20:
            _half = len(sizes) // 2
            _recent = sum(sizes[-_half:]) / _half
            _base = sum(sizes[:_half]) / max(1, _half)
            _skew = math.log(max(_recent, 1e-12) / max(_base, 1e-12))
            emos[10] = graded(_skew * 1.4, gain=1.0)
            dirs[10] = sign(_skew * move_dir) if abs(_skew) > 0.15 else 0
        else:
            emos[10] = CHANNELS[10][3]
            dirs[10] = 0
            zero_reasons["10"] = ("no per-trade size ring on this tape — "
                                  "whale skew cannot be measured")

        # ── E11 HERD SYNCHRONIZATION (state) ──────────────────────────
        peer = market.get("peer_returns") or []
        rows = [row for row in peer[-30:] if isinstance(row, (list, tuple))
                and len(row) >= 2]
        disp = None
        herd_corr = market.get("herd_corr")
        if rows:
            sds = [std_pop([float(x) for x in row]) for row in rows]
            disp = sum(sds) / len(sds)
        frac_same = None
        if rows:
            _last_row = [float(x) for x in rows[-1]]
            _same = sum(1 for x in _last_row
                        if x != 0 and sign(x) == move_dir)
            frac_same = _same / len(_last_row) if _last_row else None
        if herd_corr is not None:
            emos[11] = graded((float(herd_corr) - 0.5) * 2.0, gain=1.0)
            dirs[11] = 0
        elif disp is not None and s30 > 1e-12:
            # CSAD collapse: dispersion below own vol → everyone one boat
            emos[11] = graded(1.0 - (disp / s30), gain=1.0)
            dirs[11] = 0
        else:
            emos[11] = CHANNELS[11][3]
            dirs[11] = 0
            zero_reasons["11"] = ("no peer cross-section in this window — "
                                  "herd synchronization cannot be measured")

        # ── E12 RANGE CHOP FATIGUE (event) ────────────────────────────
        # Kaufman ER decay: efficiency falling + level low = fatigue
        def _er(win):
            rr = rets[-win:]
            if not rr:
                return 0.0
            path = sum(abs(x) for x in rr)
            return abs(sum(rr)) / path if path > 1e-12 else 0.0
        _er_now = _er(30)
        _er_prev = _er(60) if len(rets) >= 60 else _er_now
        _fatigue = max(0.0, 0.45 - _er_now) * 2.2 + max(0.0, _er_prev - _er_now) * 3.0
        emos[12] = _ev(_fatigue, dead=0.1, gain=1.0)
        dirs[12] = 0                      # suppressor, not a side

        # ── E13 FAILED-BREAKOUT BULL/BEAR TRAP (event) ────────────────
        # pierce of the prior N-bar extreme on declining volume, snapping
        # back inside the range within 1–3 bars
        _trap = 0.0
        _trap_side = 0
        if n_b >= 26:
            for _back in (1, 2, 3):
                _i = n_b - _back
                if _i < 24:
                    break
                _hi20 = max(highs[_i - 20:_i])
                _lo20 = min(lows[_i - 20:_i])
                _c = mc[_i]
                _prev_c = mc[_i - 1]
                _vbar = vols[_i] if _i < len(vols) else 0.0
                _vref = sum(vols[_i - 10:_i]) / 10 if _i >= 10 and vols else 0.0
                if _c > _prev_c and highs[_i] > _hi20 and _c < _hi20 \
                        and _vref > 0 and _vbar < _vref:
                    _trap = 1.0 / _back
                    _trap_side = 1
                    break
                if _c < _prev_c and lows[_i] < _lo20 and _c > _lo20 \
                        and _vref > 0 and _vbar < _vref:
                    _trap = 1.0 / _back
                    _trap_side = -1
                    break
        if _trap > 0:
            emos[13] = _ev(2.0 * _trap, dead=0.05, gain=1.4)
            dirs[13] = -_trap_side        # fade the failed break
        else:
            emos[13] = CHANNELS[13][3]
            dirs[13] = 0
            zero_reasons["13"] = ("no range-pierce attempt in this window — "
                                  "failed-breakout trap not measurable")

        # ── E14 NEWS-PRICE CONFIRMATION vs DIVERGENCE (state) ─────────
        pol = float(news_agg.get("polarity", 0) or 0) if news_agg else 0.0
        count = int(news_agg.get("count", 0) or 0) if news_agg else 0
        if count:
            _news_ts = float(market.get("news_last_ts") or 0.0)
            _follow = 0.0
            if _news_ts and times:
                _idx = next((i for i, t in enumerate(times)
                             if t >= _news_ts), None)
                if _idx is not None and _idx < n_b - 1:
                    _follow = (mc[-1] - mc[_idx]) / mc[_idx] if mc[_idx] else 0.0
                else:
                    _follow = r_w
            else:
                _follow = r_w
            _ft_z = _follow / max(s30 * math.sqrt(max(n_w, 2)), 1e-9)
            _conf = pol * _clamp(_ft_z, -3.0, 3.0)
            emos[14] = graded(_conf, gain=0.9)
            dirs[14] = sign(_conf) if abs(_conf) > 0.15 else 0
        else:
            emos[14] = CHANNELS[14][3]
            dirs[14] = 0
            zero_reasons["14"] = ("no scored headlines this window — "
                                  "news-price confirmation abstains")

        # ── E15 BREAKOUT KINETIC ENERGY (event) ───────────────────────
        # volume acceleration × range expansion × flow delta out of
        # compression
        _tension = max(0.0, -(math.log(max(_ratio, 1e-6))))   # from E8
        _vol_accel = _z(v5, vols[-min(60, len(vols)):]) if vols else 0.0
        _range_exp = _z(_rng5, ranges[-min(60, n_b):]) if n_b else 0.0
        _flow = float(market.get("ofti") or 0.0)
        _kin = (max(0.0, _tension) * 0.8 + max(0.0, _vol_accel) * 0.7
                + max(0.0, _range_exp) * 0.9 + abs(_flow) * 0.6)
        emos[15] = _ev(_kin, dead=0.55, gain=1.0)
        dirs[15] = sign(_flow) if abs(_flow) > 0.2 else move_dir

        # ── E16 HEADLINE SHOCK HALF-LIFE (event) ──────────────────────
        if count:
            _news_ts = float(market.get("news_last_ts") or 0.0)
            if _news_ts and times:
                _age_min = max(0.0, (times[-1] - _news_ts) / 60.0)
            else:
                _age_min = 15.0   # headline inside the window, age unstamped
            _tier = float(market.get("news_tier") or 1.0)
            _impulse = math.exp(-_age_min / 45.0) \
                * max(0.0, vol_ratio - 0.85) * _tier \
                * max(abs(pol), 0.3)
            emos[16] = _ev(_impulse * 2.4, dead=0.05, gain=1.2)
        else:
            emos[16] = CHANNELS[16][3]
            zero_reasons["16"] = ("no scored headlines this window — "
                                  "headline half-life abstains")
        dirs[16] = 0                       # decay magnitude, not a side

        # ── E17 PSYCHOLOGICAL LEVEL & VWAP GRAVITY (state) ────────────
        # nearest round-number liquidity pool; gravity rises as spot
        # approaches it, direction points AT the level
        _step = 10.0 ** math.floor(math.log10(max(live, 1e-9)))
        if _step > 0:
            _nearest = round(live / _step) * _step
            if abs(_nearest - live) < 1e-12:
                _nearest = _nearest + _step
            _dist_atr = (live - _nearest) / atr if atr > 0 else 0.0
            _closeness = _clamp(1.0 - abs(_dist_atr) / 3.0, 0.0, 1.0)
            emos[17] = graded(2.2 * (_closeness - 0.4), gain=1.0)
            dirs[17] = (1 if _dist_atr < 0 else -1) if abs(_dist_atr) < 2.0 else 0
        else:
            emos[17] = CHANNELS[17][3]
            dirs[17] = 0

        # ── E18 TREND-CHASE FATIGUE (state, rewritten) ────────────────
        # RSI/stochastic overextension × diminishing ΔP per unit volume
        _wins = cl if len(cl) >= 15 else mc
        _g14 = _l14 = 0.0
        for _i in range(max(1, len(_wins) - 14), len(_wins)):
            _d = _wins[_i] - _wins[_i - 1]
            if _d >= 0:
                _g14 += _d
            else:
                _l14 -= _d
        _rsi = (100.0 - 100.0 / (1.0 + _g14 / _l14)) if _l14 > 0 else 100.0
        if _l14 <= 0 and _g14 <= 0:
            _rsi = 50.0
        _w14 = _wins[-min(14, len(_wins)):]
        _hh, _ll = max(_w14), min(_w14)
        _stoch = ((_wins[-1] - _ll) / (_hh - _ll) - 0.5) * 2.0 \
            if _hh > _ll else 0.0
        _over = (_rsi - 50.0) / 50.0 + _stoch * 0.5
        _dp_now = abs(mc[-1] - mc[-6]) if n_b > 6 else 0.0
        _v_now = sum(vols[-5:]) if vols else 0.0
        _dp_prev = abs(mc[-6] - mc[-11]) if n_b > 11 else 0.0
        _v_prev = sum(vols[-11:-5]) if len(vols) >= 11 else 0.0
        _eff_now = (_dp_now / _v_now) if _v_now > 0 else 0.0
        _eff_prev = (_dp_prev / _v_prev) if _v_prev > 0 else 0.0
        _decay = (1.0 - _eff_now / _eff_prev) if _eff_prev > 1e-12 else 0.0
        _chase = _over * (1.0 + 0.8 * _clamp(_decay, 0.0, 2.0))
        emos[18] = graded(_clamp(_chase, -3.0, 3.0), gain=1.1)
        dirs[18] = -sign(_over) if abs(_over) > 0.25 else 0

        # ── E19 SMART-MONEY DIVERGENCE (OBV/MFI vs price) ─────────────
        _obv = 0.0
        _obv_r: list[float] = []
        for _i in range(max(1, n_b - 20), n_b):
            _obv += (1 if mc[_i] >= mc[_i - 1] else -1) * (
                vols[_i] if _i < len(vols) else 0.0)
            _obv_r.append(_obv)
        _px_r = mc[-min(20, n_b):]
        if len(_obv_r) >= 6 and len(_px_r) >= 6 and std_pop(_obv_r) > 0 \
                and std_pop(_px_r) > 0:
            _zobv = _z(_obv_r[-1], _obv_r)
            _zpx = _z(_px_r[-1], _px_r)
            _div = _zobv - _zpx
            emos[19] = graded(_div, gain=1.0)
            dirs[19] = sign(_div) if abs(_div) > 0.3 else 0
        else:
            emos[19] = CHANNELS[19][3]
            dirs[19] = 0
            zero_reasons["19"] = ("insufficient volume series for OBV slope "
                                  "divergence")

        # ── E20 NEWS FOMO OVERREACTION (event) ────────────────────────
        if count:
            _ext = abs(r_w) / max(s30 * math.sqrt(max(n_w, 2)), 1e-9)
            _fomo = max(0.0, _ext - 2.2) * (0.5 + abs(pol))
            emos[20] = _ev(_fomo, dead=0.05, gain=1.1)
            dirs[20] = -sign(pol * move_dir) if pol and move_dir else 0
        else:
            emos[20] = CHANNELS[20][3]
            dirs[20] = 0
            zero_reasons["20"] = ("no scored headlines this window — "
                                  "news FOMO abstains")

        # ── E21 CLIMAX EXHAUSTION STREAK (event) ──────────────────────
        _cstreak = 0
        _cside = 0
        for _i in range(n_b - 1, 0, -1):
            _s = 1 if mc[_i] > mc[_i - 1] else (-1 if mc[_i] < mc[_i - 1] else 0)
            if _s == 0:
                break
            if _cside == 0:
                _cside = _s
            elif _s != _cside:
                break
            _cstreak += 1
        _blow = (v5 / v_mean30) if v_mean30 > 0 else 1.0
        # streak length carries the reading; a volume blow-off amplifies
        # it (the 2.5× MA definition) but a long streak without one still
        # reads as exhaustion-in-progress rather than staying flat
        _climax_raw = max(0.0, _cstreak - 3) * (0.35 + 0.5 * max(0.0, _blow - 1.0))
        emos[21] = _ev(_climax_raw * 1.6, dead=0.15, gain=1.1)
        dirs[21] = -_cside if _cstreak >= 4 else 0
        tedr_val = emos[21]
        rtp_val = emos[1]

        # derived scalars the panels read (NOT channels): affect
        # proxies for the Ekman/Russell display, FG inputs, PANAS
        v_aff = graded(_compress(z_drift, 1.5), gain=1.0)     # valence 0..1
        a_aff = graded(math.log(max(vol_ratio, 1e-6)), gain=1.4)
        d_aff = graded(0.5 * (persist - 0.5) + 0.25
                       * _clamp(z_drift / 3.0, -1, 1), gain=1.6)
        up, dn = _semi_deviations(rets[-min(60, len(rets)):])
        _tot = up + dn
        _up_share = (up / _tot) if _tot > 1e-12 else 0.5
        pa = graded(3.2 * (_up_share - 0.5) + (a_aff - 0.5) * 1.2, gain=0.55)
        na = graded(3.2 * ((1.0 - _up_share) - 0.5) + (a_aff - 0.5) * 1.2,
                    gain=0.55)
        vol_stress = a_aff                    # FG volatility component
        momentum_fg = v_aff                   # FG momentum/volume component
        z5 = (d1 / (s30 * math.sqrt(5.0) * live)) if (
            s30 > 1e-12 and live > 0 and len(rets) >= 5) else 0.0
        n_upd = int(self.n_updates.get((ctx.symbol, ctx.tf), 0))

        # abstained channels never carry a direction — and neither does a
        # channel that is still sitting inside its OWN noise band. Without
        # this the CERC sum counted "fear 0.0005" as a full bearish vote,
        # which is how a 21-channel crowd ended up 82 % "directional" on a
        # tape where four channels had no input at all.
        for eid in EMO_IDS:
            if str(eid) in zero_reasons:
                dirs[eid] = 0
                continue
            neutral = CHANNELS[eid][3]
            span = (1.0 - neutral) if neutral == 0.0 else 0.5
            if abs(emos[eid] - neutral) / span < MOVE_BAND:
                dirs[eid] = 0

        # clamp/round once, then use the PRINTED values for everything below
        emos = {eid: round(_clamp(float(emos[eid]), 0.0, 1.0), 5)
                for eid in EMO_IDS}

        # ---- composite (CERC = weighted agreement sum) -----------------
        w = self._weights(ctx.symbol, ctx.tf)
        cerc = 0.0
        num = den = mass = 0.0
        bulls = bears = 0
        for i, eid in enumerate(EMO_IDS):
            term = w[i] * emos[eid] * dirs[eid]
            if eid == 21:
                # the exhaustion channel gates its own participation
                term *= (1.0 - 0.5 * emos[21])
            cerc += term
            dev = abs(emos[eid] - CHANNELS[eid][3])
            wi_ei = w[i] * dev
            if wi_ei <= 0:
                continue
            mass += wi_ei
            if dirs[eid]:
                # signed by the same deviation the weight is built from, so
                # the ratio stays inside [−1, +1] (it used to reach 1.11)
                num += w[i] * dev * dirs[eid] * (
                    (1.0 - 0.5 * emos[21]) if eid == 21 else 1.0)
                den += wi_ei
                if dirs[eid] > 0:
                    bulls += 1
                else:
                    bears += 1
        net_emotion = (num / den) if den > 0 else 0.0
        agreement = (den / mass) if mass > 0 else 0.0

        # ---- derived model outputs -------------------------------------
        # derived affect (Russell/Ekman DISPLAY): the new channels are
        # behavioral, so the classic (v, a, d) axes are recomputed from
        # the tape aggregates rather than borrowed from channel ids
        ek = ekman_scores(v_aff, a_aff, d_aff)
        prim = max(ek, key=lambda k: ek[k]["intensity"])
        runner = max((k for k in ek if k != prim),
                     key=lambda k: ek[k]["intensity"])
        fg = fear_greed_index(vol_stress=vol_stress, momentum=momentum_fg,
                              herding=emos[11])
        # herd reading is its own internal reading (peer cross-section),
        # NOT the published BTC-dominance component
        fg["breadth_component"] = round(float(emos[11]), 4)
        # core affect and the index answer different questions (fast price
        # affect vs volatility/momentum survey). When they disagree, say WHY
        # instead of letting the panel look self-contradictory.
        _v_raw = 2.0 * v_aff - 1.0
        if _v_raw > 0.3 and fg["index"] < 45:
            fg["vs_affect"] = (
                f"affect is positive (valence {_v_raw:+.2f}) while the index "
                f"reads {fg['label'].lower()} {fg['index']:.0f}: momentum is "
                f"{fg['components']['momentum_volume']:.0f}/100 (below its "
                f"mean) and volatility {fg['components']['volatility']:.0f}/100 "
                f"— the drift is up but the tape is not confirming it")
        elif _v_raw < -0.3 and fg["index"] > 55:
            fg["vs_affect"] = (
                f"affect is negative (valence {_v_raw:+.2f}) while the index "
                f"reads {fg['label'].lower()} {fg['index']:.0f}: momentum "
                f"{fg['components']['momentum_volume']:.0f}/100 is still above "
                f"its mean — greed in a falling drift")
        else:
            fg["vs_affect"] = None
        act = action_tendency(fear=ek["fear"]["intensity"],
                              anger=ek["anger"]["intensity"],
                              sadness=ek["sadness"]["intensity"],
                              enjoyment=ek["enjoyment"]["intensity"],
                              pa=pa, na=na, dominance=d_aff,
                              capitulation=emos[1], trend_dir=move_dir)

        lag_rings = {
            # key names kept for the history rings / UI legend; each now
            # carries the redefined raw input
            "dpi": z5,
            "octo": math.log(max(vol_ratio, 1e-6)),
            "pidx": persist,
            "nsp": emos[20],
        }

        # ---- dominant channel (ranked by distance from ITS OWN neutral) --
        import time as _time
        _key = (ctx.symbol, ctx.tf)
        _prev = self._prev_emos.get(_key) or {}

        def _dev_of(i: int) -> float:
            neutral = CHANNELS[i][3]
            span = (1.0 - neutral) if neutral == 0.0 else 0.5
            return abs(float(emos[i]) - neutral) / span

        # Who may be crowned "dominant"?
        #
        # An abstained channel has no reading at all. Beyond that, the two
        # channel families are not symmetric: an EVENT channel (neutral 0.0,
        # e.g. trend exhaustion) sitting at 0.04 is saying "this event is not
        # happening" — its distance from neutral is large precisely because
        # it is switched off. The old rule crowned that as the dominant
        # emotion, i.e. it announced the *absence* of exhaustion as the
        # market's mood. A state channel (neutral 0.5, e.g. arousal) is
        # meaningful on BOTH sides, so it stays eligible either way.
        _read = [i for i in EMO_IDS if str(i) not in zero_reasons] or \
            list(EMO_IDS)
        _live = [i for i in _read
                 if CHANNELS[i][3] != 0.0 or emos[i] > MOVE_BAND]
        _eligible = _live or _read
        _ranked = sorted(
            ((_dev_of(i),
              abs(float(emos[i]) - float(_prev.get(i, emos[i]))),
              float(w[i - 1]), i)
             for i in _eligible),
            # tie-break order matters: the ROUNDED deviation groups near-ties
            # (so a 0.0002 wobble does not re-crown every pass), but the raw
            # deviation must still compare first — otherwise a channel with
            # dev 0.9998 could be crowned while another sat at exactly 1.0
            # and the "runner-up" outranked the winner on the same card.
            key=lambda t: (round(t[0], 3), t[0], t[1], t[2]), reverse=True)
        _dev, _dlt, _wv, _dom_id = _ranked[0]
        _tied = sum(1 for t in _ranked
                    if t[0] >= _dev - 0.01 and t[3] != _dom_id)
        _now = time.time()
        _cur = self._dom.get(_key)
        if _cur is None or _cur[0] != _dom_id:
            self._dom[_key] = (_dom_id, _now)
            _held = 0.0
        else:
            _held = max(0.0, _now - _cur[1])
        self._prev_emos[_key] = dict(emos)
        _neutral = CHANNELS[_dom_id][3]
        dominant = {
            # which side of its neutral the crowned channel sits on: a state
            # channel reading 0.04 is "very low", not "very high"
            "side": ("high" if emos[_dom_id] > _neutral else
                     "low" if emos[_dom_id] < _neutral else "neutral"),
            "directional": bool(int(dirs.get(_dom_id) or 0)),
            "id": int(_dom_id),
            "name": CHANNELS[_dom_id][1],
            "key": CHANNELS[_dom_id][0],
            "cite": CHANNELS[_dom_id][2],
            "value": round(float(emos[_dom_id]), 5),
            "intensity": round(_dev, 4),          # 0 = neutral, 1 = extreme
            "moved": round(_dlt, 5),
            "held_s": round(_held, 1),
            "tied": int(_tied),
            "runner_up": ({"id": int(_ranked[1][3]),
                           "name": CHANNELS[_ranked[1][3]][1],
                           "intensity": round(_ranked[1][0], 4)}
                          if len(_ranked) > 1 else None),
        }

        for eid in EMO_IDS:
            if (float(emos[eid]) == CHANNELS[eid][3]
                    and str(eid) not in zero_reasons):
                zero_reasons[str(eid)] = (
                    f"{CHANNELS[eid][0]} exactly at neutral "
                    f"{CHANNELS[eid][3]:.2f} — no deviation this window")

        return {
            "model": dict(MODEL),
            "dominant": dominant,
            "emotions": emos,
            "zero_reasons": zero_reasons,
            "zero_causes": _causes(zero_reasons),
            "directions": dirs,
            "cerc": round(_clamp(cerc, -1.0, 1.0), 5),
            "net_emotion": round(_clamp(net_emotion, -1.0, 1.0), 5),
            "agreement": round(agreement, 5),
            "breadth": {"bull": bulls, "bear": bears,
                        "flat": 21 - bulls - bears},
            "rtp": round(rtp_val, 5),
            "tedr": round(tedr_val, 5),
            "prev_cerc": prev_cerc,
            "lag_values": lag_rings,
            # ---- the real model, printed -------------------------------
            "core_affect": {"valence": round(2.0 * v_aff - 1.0, 4),
                            "arousal": round(a_aff, 4),
                            "note": ("Russell (1980) circumplex: valence "
                                     "−1…+1 × arousal 0…1")},
            "pad": {"pleasure": round(2.0 * v_aff - 1.0, 4),
                    "arousal": round(a_aff, 4),
                    "dominance": round(d_aff, 4),
                    "note": "Mehrabian & Russell (1974)"},
            "panas": {"positive": round(pa, 4),
                      "negative": round(na, 4),
                      "balance": round(pa - na, 4),
                      "note": ("Watson et al. (1988): PA and NA are separate "
                               "activated dimensions (upside/downside semi-"
                               "deviation), not complements")},
            "ekman": ek,
            "primary_emotion": {
                "name": prim, "intensity": ek[prim]["intensity"],
                "runner_up": {"name": runner,
                              "intensity": ek[runner]["intensity"]},
                "note": ("Ekman (1992) prototype nearest the measured "
                         "(valence, arousal, dominance)"),
            },
            "fear_greed": fg,
            "action_tendency": act,
            "risk": {
                "loss_lambda": LOSS_LAMBDA,
                "loss_pressure": round(emos[2], 4),
                "somatic_marker": (round(float(self._soma.get(
                                       (ctx.symbol, ctx.tf), 0.5)), 4)
                                   if n_upd >= 3 else None),
                "somatic_updates": n_upd,
                "risk_appetite": round(pa - na, 4),
                "note": ("prospect theory λ from Tversky & Kahneman (1992); "
                         "somatic marker learned from settled outcomes and "
                         "null until it has evidence"),
            },
            "features": {
                "samples": len(mc),
                "vol_z": round(math.log(max(vol_ratio, 1e-6)), 4),
                "climax_ratio": round(climax, 4),
                "violence_gate": round(_clamp(0.5 * max(0.0, climax - 1.0)
                                              + 0.5 * max(0.0, vol_ratio - 1.0),
                                              0.0, 1.0), 4),
                "vol_expansion": round(vol_ratio, 4),
                "window_streak": int(round(persist * min(30, len(rets)))),
                "r30": round(session_ret, 6),
                "window_return": round(r_w, 6),
                "pdi_session": round(d_aff, 4),
                "trend_stretch": round((live / ma30 - 1.0) if ma30 else 0.0, 6),
                "habituation": round(max(0.0, 1.0 - abs(d1) / abs(d2))
                                     if abs(d2) > 1e-12 else 0.0, 4),
                "extrapolation_z": round(z5, 4),
                "drift_z": round(z_drift, 4),
                "csad_dispersion": (round(disp, 8) if disp is not None else None),
                "breadth_same": (round(frac_same, 4)
                                 if frac_same is not None else None),
                "serotonin": round(float((brain_res.get("hormones_avg") or {})
                                         .get("serotonin", 0.5)), 4),
            },
            # retail vs professional reads: the channels that literally
            # describe each crowd, signed by each channel's own direction
            "retail": _crowd(emos, dirs, (18, 20, 1, 11, 21),
                             ("FOMO-chasing (chase fatigue + news euphoria)",
                              "panic-selling (capitulation velocity)",
                              "retail quiet")),
            "hedge": _crowd(emos, dirs, (2, 3, 9, 17, 5),
                            ("hedges pressing (underwater pain + tail hedge)",
                             "risk appetite returning",
                             "hedges quiet")),
            # E22 — the CERC behavioral composite, decomposed: three
            # telemetry vectors the panel prints beside the agreement sum
            "cerc_vectors": _cerc_vectors(emos, dirs, w),
        }

    # ------------------------------------------------------------------
    def _empty_state(self, ctx: FormulaContext, prev_cerc) -> dict:
        """A tape with no prices: every channel abstains, honestly."""
        neutral = {eid: CHANNELS[eid][3] for eid in EMO_IDS}
        zr = {str(eid): "no price tape yet — affect cannot be measured"
              for eid in EMO_IDS}
        return {
            "model": dict(MODEL),
            "dominant": None,
            "emotions": neutral,
            "zero_reasons": zr,
            "zero_causes": _causes(zr),
            "directions": {eid: 0 for eid in EMO_IDS},
            "cerc": 0.0, "net_emotion": 0.0, "agreement": 0.0,
            "breadth": {"bull": 0, "bear": 0, "flat": 21},
            "rtp": 0.5, "tedr": 0.5, "prev_cerc": prev_cerc,
            "lag_values": {"dpi": 0.0, "octo": 0.0, "pidx": 0.0, "nsp": 0.5},
            "core_affect": {"valence": 0.0, "arousal": 0.0,
                            "note": "no tape"},
            "pad": {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0,
                    "note": "no tape"},
            "panas": {"positive": 0.0, "negative": 0.0, "balance": 0.0,
                      "note": "no tape"},
            "ekman": {},
            "primary_emotion": None,
            "fear_greed": None,
            "action_tendency": {"name": "monitor", "urge": 0.0,
                                "direction": 0,
                                "note": "no tape to react to",
                                "runner_up": None},
            "risk": {"loss_lambda": LOSS_LAMBDA, "loss_pressure": 0.5,
                     "somatic_marker": None, "somatic_updates": 0,
                     "risk_appetite": 0.0,
                     "note": "no tape"},
            "features": {"samples": 0},
            "retail": {"score": 0.0, "label": "retail quiet",
                       "channels": []},
            "hedge": {"score": 0.0, "label": "hedges quiet", "channels": []},
        }

    # ------------------------------------------------------------------
    def reward_update(self, symbol: str, tf: str, emos: dict[int, float],
                      dirs: dict[int, int], cerc_before: float,
                      move: float, eta: float | None = None,
                      scale: float | None = None) -> None:
        """GRC-cadence auto-learning for the channel weights + somatic marker.

        Accuracy-driven upgrades (user directive: weightage follows
        predictive accuracy):
          * COLD-START eta — rate decays from ETA_COLD to the floor with a
            150-update half-life, so the first few hundred cycles actually
            differentiate channels instead of freezing at uniform.
          * MAGNITUDE reward — a big verified move teaches more than noise:
            reward = ±clip(|move|/scale, 0.3, 1.5).
          * SOMATIC MARKER (Damasio 1994) — the marker is the EWMA of whether
            the composite called this move correctly, so risk appetite is
            learned from outcomes rather than asserted.
        """
        move_dir = sign(move)
        if move_dir == 0:
            return
        key = (symbol, tf)
        # somatic marker first: it is about the *system's* recent success
        prev = float(self._soma.get(key, 0.5))
        called_right = 1.0 if (sign(cerc_before) == move_dir
                               and abs(cerc_before) > 1e-9) else 0.0
        self._soma[key] = _clamp(0.7 * prev + 0.3 * called_right, 0.0, 1.0)

        w = self._weights(symbol, tf)
        acc = self._accuracy(symbol, tf)
        active = []
        for i, eid in enumerate(EMO_IDS):
            try:
                value = float(emos.get(eid, 0.5))
                direction = int(dirs.get(eid, 0) or 0)
            except (TypeError, ValueError, OverflowError):
                continue
            if direction == 0 or not math.isfinite(value):
                continue                    # abstention is not a miss
            dev = abs(value - CHANNELS[eid][3])
            if dev <= 1e-12:
                continue
            term = w[i] * dev * direction
            if eid == 21:
                term *= (1.0 - 0.5 * dev)
            if abs(term) > 1e-12:
                active.append((i, term))
        if not active:
            return
        base = ETA if eta is None else eta
        n = self.n_updates.get(key, 0)
        rate = max(base, ETA_COLD * 0.5 ** (n / COLD_HALFLIFE))
        self.n_updates[key] = n + 1
        mag = 1.0
        if scale and move != 0.0:
            mag = _clamp(abs(move) / scale, 0.3, 1.5)
        for i, term in active:
            correct = sign(term) == move_dir
            reward = mag if correct else -mag
            acc[i] = ACC_SMOOTH * acc[i] + (1 - ACC_SMOOTH) * (
                1.0 if correct else 0.0)
            # same multiplicative eta rule as the CCI weight learning:
            # omega(t) = omega(t-1) * (1 + eta * reward), then clamp
            w[i] = _clamp(w[i] * (1.0 + rate * reward), W_CLAMP_LO, W_CLAMP_HI)
        total = sum(w)
        if total > 0:
            for i in range(21):
                w[i] /= total

    # ------------------------------------------------------------------
    def status(self) -> dict:
        out: dict[str, list] = {}
        for (sym, tf), w in self.weights.items():
            out[f"{sym}:{tf}"] = [round(x, 4) for x in w]
        return out

    def accuracy(self) -> dict:
        out: dict[str, list] = {}
        for (sym, tf), a in self.acc.items():
            out[f"{sym}:{tf}"] = [round(x, 4) for x in a]
        return out


def _crowd(emos, dirs, ids, labels):
    """Signed read of a crowd from a channel subset, each channel weighted by
    how far it sits from its own neutral and scaled to [−1, +1]."""
    vals = []
    for i in ids:
        neutral = CHANNELS[i][3]
        v = float(emos.get(i, neutral))
        span = (1.0 - neutral) if neutral == 0.0 else 0.5
        vals.append(_clamp((v - neutral) / span, -1.0, 1.0)
                    * int(dirs.get(i, 0) or 0))
    score = sum(vals) / len(vals) if vals else 0.0
    label = (labels[0] if score > 0.15 else
             labels[1] if score < -0.15 else labels[2])
    return {"score": round(score, 4), "label": label,
            "channels": list(ids)}


# ═══════════════════════════════════════════════════════════════════════
# §E2 — DIAGNOSIS-DRIVEN AFFECT EXTENSIONS (2026-11-23)
# ═══════════════════════════════════════════════════════════════════════
# The original 22 emotion cells describe the CROWD's affect.  Grading the
# whole stack on 3 years of tape showed what was missing: nothing measured
# the *trader's* affect — the emotional states that actually cost money.
# These four channels are computed from the same tape and are deliberately
# about discipline rather than sentiment:
#
#   FOMO_PRESSURE   — chasing a move that has already happened
#   PANIC_URGENCY   — the felt need to act NOW (costs the round trip)
#   REVENGE_BIAS    — sizing up after a loss
#   COMPLACENCY_DRIFT — small wins breeding oversized positions
#
# Each is [0,1] and each maps to an action limit, so the emotional state
# changes POSITION SIZE AND HOLDING PERIOD, not just a dashboard colour.

def fomo_pressure(closes: list[float], vol_bps: float,
                  window: int = 22) -> dict:
    """How much of the recent move is already behind us (chase risk).

    High when the last `window` bars moved sharply AND the move is late in
    its own path (most of the net move happened recently).  Chasing a
    spent move is the single most expensive emotion in trading: it buys
    the top of the move and pays the round trip for the privilege.
    """
    if len(closes) < max(4, window) or window < 3:
        return {"value": 0.0,
                "reason": f"needs {max(4, window)} bars, had {len(closes)}"}
    seg = closes[-(window + 1):]
    net = seg[-1] / seg[0] - 1.0
    net_bps = abs(net) * 1e4
    if vol_bps <= 0:
        vol_bps = 1.0
    z = net_bps / max(vol_bps, 1e-9)
    # late-ness: share of the total path travelled in the final third
    path = sum(abs(seg[i + 1] - seg[i]) for i in range(len(seg) - 1))
    _start = max(0, len(seg) - 1 - max(1, window // 3))
    third = sum(abs(seg[i + 1] - seg[i]) for i in range(_start, len(seg) - 1))
    late = (third / path) if path > 1e-12 else 0.0
    v = _clamp(min(1.0, z / 3.0) * (0.4 + 0.6 * late), 0.0, 1.0)
    return {"value": round(v, 4), "z": round(z, 3), "late_share": round(late, 3),
            "reading": ("chasing — move is spent" if v > 0.6
                        else "moderate" if v > 0.3 else "calm")}


def panic_urgency(returns: list[float], cost_bps: float) -> dict:
    """The felt need to act immediately, which is always a cost.

    Measured as the ratio of the current absolute move to the round trip:
    when a window's move is smaller than the cost of capturing it, acting
    NOW is pure loss.  High urgency should REDUCE size, never raise it.
    """
    if not returns:
        return {"value": 0.0, "reason": "no returns"}
    move_bps = abs(returns[-1]) * 1e4
    if cost_bps <= 0:
        cost_bps = 30.0
    ratio = move_bps / cost_bps
    v = _clamp(1.0 / (1.0 + max(ratio, 1e-6)), 0.0, 1.0)
    return {"value": round(v, 4), "move_bps": round(move_bps, 2),
            "cost_bps": round(cost_bps, 2), "ratio": round(ratio, 3),
            "reading": ("act now would be pure cost" if v > 0.6
                        else "move clears cost" if v < 0.3 else "marginal")}


def revenge_bias(losses: list[float], sizes: list[float]) -> dict:
    """Sizing up after losses — the classic account killer.

    Compares the average size following a loss with the average size
    following a win.  A ratio above 1.0 means the book is getting bigger
    exactly when it has just been proven wrong.
    """
    if len(losses) < 4 or len(sizes) < 4:
        return {"value": 0.0, "reason": "not enough graded trades"}
    n = min(len(losses), len(sizes))
    after_loss = [sizes[i] for i in range(1, n) if losses[i - 1] < 0]
    after_win = [sizes[i] for i in range(1, n) if losses[i - 1] >= 0]
    if not after_loss:
        return {"value": 0.0, "reason": "no losses to grade"}
    m_l = sum(after_loss) / len(after_loss)
    if not after_win:
        # Every graded trade followed a loss — there is no win-side baseline
        # to compare against, so the honest statement is "unmeasurable"
        # rather than a fabricated ratio.  Report the level so the caller
        # can still see it, and score 0 (no claim).
        return {"value": 0.0, "size_after_loss": round(m_l, 4),
                "reason": "no wins in the ledger — baseline unmeasurable"}
    m_w = sum(after_win) / len(after_win)
    if m_w <= 1e-12:
        return {"value": 1.0, "reason": "sizes after wins are zero"}
    ratio = m_l / m_w
    v = _clamp((ratio - 1.0) / 2.0, 0.0, 1.0)
    return {"value": round(v, 4), "size_after_loss": round(m_l, 4),
            "size_after_win": round(m_w, 4), "ratio": round(ratio, 3),
            "reading": ("sizing up after losses" if v > 0.3 else "balanced")}


def complacency_drift(win_streak: int, size_mult: float) -> dict:
    """Small wins breeding oversized positions.

    A long win streak is the most dangerous state in a wide-stop book,
    because the next trade is the one that pays for all of them.
    """
    streak = max(0, int(win_streak))
    v = _clamp((min(streak, 12) / 12.0) * (0.5 + 0.5 * min(1.0, size_mult)),
               0.0, 1.0)
    return {"value": round(v, 4), "win_streak": streak,
            "size_mult": round(size_mult, 3),
            "reading": ("complacent — cut size" if v > 0.6 else "normal")}


def discipline_overlay(*, fomo: dict, panic: dict, revenge: dict,
                       complacent: dict) -> dict:
    """Turn the four channels into ONE position-size multiplier.

    This is the point of the extension: the emotional state has to change
    the trade, or it is decoration.  The multiplier multiplies (never
    adds) so it can only ever reduce risk — no emotional reading is
    allowed to INCREASE position size.
    """
    mult = 1.0
    reasons: list[str] = []
    if fomo["value"] > 0.6:
        mult *= 0.5
        reasons.append("FOMO: move already spent, half size")
    if panic["value"] > 0.6:
        mult *= 0.5
        reasons.append("panic: move smaller than cost, half size")
    if revenge["value"] > 0.3:
        mult *= 0.5
        reasons.append("revenge: sizing up after a loss, half size")
    if complacent["value"] > 0.6:
        mult *= 0.6
        reasons.append("complacency: long win streak, cut 40 %")
    mult = _clamp(mult, 0.05, 1.0)
    return {"size_mult": round(mult, 4),
            "block_new": bool(mult <= 0.25),
            "reasons": reasons,
            "note": ("emotional readings may only REDUCE size; nothing here "
                     "can raise it"),
            "channels": {"fomo": fomo["value"], "panic": panic["value"],
                         "revenge": revenge["value"],
                         "complacency": complacent["value"]}}
