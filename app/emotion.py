"""Section 5 — EMOTIONAL STATE, rebuilt on published affect models.

The previous version was 22 invented channels (E1 "FII", E2 "GSL", E5
"octopamine reflex", …) whose coefficients came from a specification, not
from the literature. It produced numbers, but they did not correspond to any
recognised emotion, the "dominant" channel was regularly a channel reading
0.00005, and no reader could say what the panel meant.

This module measures a market's emotional state with models that are
actually used to measure emotion:

  * **Core affect** — Russell (1980) circumplex: valence (pleasure) and
    arousal (activation) are the two axes; every affect term is a position in
    that plane (excitement 45°, distress 135°, depression 225°, …).
  * **PAD** — Mehrabian & Russell (1974): pleasure, arousal, dominance. The
    third axis is what separates fear (low control) from anger (high
    control); without it those two are indistinguishable.
  * **PANAS** — Watson, Clark & Tellegen (1988): positive and negative affect
    are *separate, activated* dimensions (a 45° rotation of the circumplex),
    so a violent tape can raise both at once. We therefore compute PA and NA
    from upside/downside semi-deviation plus activation, never as 1 − x.
  * **Basic emotions** — Ekman (1992): fear, anger, disgust, sadness,
    surprise, enjoyment, scored as Gaussian similarity to prototype positions
    in (valence, arousal, dominance) space.
  * **Loss aversion** — Kahneman & Tversky (1979) prospect theory, with the
    empirically estimated λ = 2.25 from Tversky & Kahneman (1992): losses
    weigh about twice as much as equal gains.
  * **Somatic marker** — Damasio (1994) / Bechara et al. (1994): prior
    outcomes leave a decaying bodily "marker" that biases risk-taking, so the
    panel's risk appetite is learned from this system's own settled results.
  * **Herding** — Christie & Huang (1995): cross-sectional absolute deviation
    (CSAD) of peer returns collapses when the crowd moves as one.
  * **Extrapolation / chase** — Barberis, Greenwood & Shleifer (2015): the
    crowd over-weights recent returns, which is what a late chase looks like.
  * **Market sentiment index** — the Crypto Fear & Greed Index
    (alternative.me) methodology: volatility 25 %, market momentum/volume
    25 %, social media 15 %, surveys 15 %, bitcoin dominance 10 %, trends
    10 %. We implement the components this system can actually measure
    (volatility, momentum/volume) and *say* that dominance, social,
    surveys and search-trends are unavailable rather than inventing them —
    the index is renormalised over the measured weight and the coverage is
    printed with the number.
  * **Action tendency** — Frijda (1986): an emotion is a readiness to act
    (approach / avoid / confront / withdraw / disengage / monitor), which is
    what the trading layer consumes.

The 21 numbered channels kept their numbering because the CCI weights, the
reward-learning rule and the UI grid are keyed by it; each id now names a
construct from the list above (see `CHANNELS`), and every channel carries its
citation into the payload so the panel can print what it is showing.
"""
from __future__ import annotations

import math

from .config import HIST_EMOTION_LAG
from .models import FormulaContext
from .formulas._helpers import sign, std_pop

ETA = 0.015               # Section 10: η_emotion — floor (see COLD rate below)
W_CLAMP_LO = 0.005        # widened: predictive channels can dominate
W_CLAMP_HI = 0.30
ETA_COLD = 0.08           # cold-start rate — a fresh brain learns FAST
COLD_HALFLIFE = 150
ACC_SMOOTH = 0.90
EMO_IDS = list(range(1, 22))          # channel ids (22 stays the composite)
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

# id → (key, printed name, citation, neutral value)
CHANNELS = {
    1: ("valence", "Valence (pleasure)", "Russell 1980", 0.5),
    2: ("arousal", "Arousal (activation)", "Russell 1980", 0.5),
    3: ("dominance", "Dominance (control)", "Mehrabian & Russell 1974", 0.5),
    4: ("pa", "Positive affect (PA)", "Watson, Clark & Tellegen 1988", 0.5),
    5: ("na", "Negative affect (NA)", "Watson, Clark & Tellegen 1988", 0.5),
    6: ("fear", "Fear", "Ekman 1992", 0.0),
    7: ("anger", "Anger", "Ekman 1992", 0.0),
    8: ("disgust", "Disgust", "Ekman 1992", 0.0),
    9: ("sadness", "Sadness", "Ekman 1992", 0.0),
    10: ("surprise", "Surprise", "Ekman 1992", 0.0),
    11: ("enjoyment", "Enjoyment", "Ekman 1992", 0.0),
    12: ("loss_pressure", "Loss-aversion pressure", "Kahneman & Tversky 1979",
         0.5),
    13: ("somatic", "Somatic marker (risk memory)", "Damasio 1994", 0.5),
    14: ("herding", "Herding (CSAD)", "Christie & Huang 1995", 0.5),
    15: ("capitulation", "Capitulation", "Frijda 1986 · F&G volatility", 0.5),
    16: ("extrapolation", "Extrapolation / chase",
         "Barberis, Greenwood & Shleifer 2015", 0.5),
    17: ("vol_stress", "Volatility stress", "CBOE VIX concept · F&G", 0.5),
    18: ("momentum", "Momentum vs mean", "alternative.me momentum/volume", 0.5),
    19: ("breadth", "Market breadth", "alternative.me dominance", 0.5),
    20: ("news", "News sentiment (external NLP)", "measured news pipe", 0.5),
    21: ("exhaustion", "Trend exhaustion", "Frijda 1986 · action tendency",
         0.5),
}
CHANNEL_KEYS = {i: CHANNELS[i][0] for i in EMO_IDS}

MODEL = {
    "name": "core-affect (Russell) + PAD + PANAS + Ekman prototypes + "
            "prospect theory + fear&greed components",
    "citations": [
        "Russell, J. A. (1980). A circumplex model of affect. JPSP 39(6).",
        "Watson, D., Clark, L. A., & Tellegen, A. (1988). PANAS. JPSP 54(6).",
        "Mehrabian, A., & Russell, J. A. (1974). An Approach to "
        "Environmental Psychology (PAD).",
        "Ekman, P. (1992). An argument for basic emotions. Cognition & "
        "Emotion 6(3-4).",
        "Kahneman, D., & Tversky, A. (1979). Prospect theory. Econometrica "
        "47(2); Tversky & Kahneman (1992) λ=2.25, J. Risk & Uncertainty 5.",
        "Damasio, A. (1994). Descartes' Error; Bechara et al. (1994) "
        "somatic-marker hypothesis, Cognition 50.",
        "Christie, W., & Huang, R. (1995). Following the pied piper: herding "
        "in equity returns. Financial Analysts Journal 51(3).",
        "Barberis, N., Greenwood, R., & Shleifer, A. (2015). X-CAPM. "
        "J. Financial Economics 115(1).",
        "Frijda, N. (1986). The Emotions (action tendencies).",
        "alternative.me Crypto Fear & Greed Index methodology (volatility "
        "25 %, momentum/volume 25 %, social 15 %, surveys 15 %, dominance "
        "10 %, trends 10 %).",
    ],
    # the UI measures every channel against ITS OWN neutral — Ekman basics
    # sit at 0.0, the state channels at 0.5 — so the neutral ships with the
    # name and the citation
    "channels": {str(i): {"name": CHANNELS[i][1], "cite": CHANNELS[i][2],
                          "key": CHANNELS[i][0], "neutral": CHANNELS[i][3]}
                 for i in EMO_IDS},
    "fear_greed_measured_weight": round(sum(FG_WEIGHTS[k]
                                            for k in FG_MEASURED), 4),
    "fear_greed_missing": dict(FG_MISSING),
}


def sig(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _finite(*vals: float) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(float(v))
               for v in vals)


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
        h, l, c = ohlc.get("h") or [], ohlc.get("l") or [], ohlc.get("c") or []
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
        out[name] = {"intensity": round(_clamp(intensity, 0.0, 1.0), 4),
                     "distance": round(dist, 4),
                     "prototype": {"valence": pv, "arousal": pa,
                                   "dominance": pd_}}
    return out


def fear_greed_index(*, vol_stress: float, momentum: float,
                     herding: float = 0.5) -> dict:
    """Crypto Fear & Greed (alternative.me) over the components we measure.

    All inputs are [0,1] channel values; 1 = greed-side of that component.
    Social / surveys / trends are NOT estimated — the index is renormalised
    over the measured 60 % of the published weight and says so.
    """
    comps = {
        "volatility": _clamp(1.0 - float(vol_stress), 0.0, 1.0) * 100.0,
        "momentum_volume": _clamp(float(momentum), 0.0, 1.0) * 100.0,
    }
    wsum = sum(FG_WEIGHTS[k] for k in comps) or 1.0
    index = sum(FG_WEIGHTS[k] * v for k, v in comps.items()) / wsum
    label = ("Extreme fear" if index <= 24 else "Fear" if index <= 44 else
             "Neutral" if index <= 55 else "Greed" if index <= 75 else
             "Extreme greed")
    return {"index": round(index, 1), "label": label,
            "components": {k: round(v, 1) for k, v in comps.items()},
            "weights": {k: FG_WEIGHTS[k] for k in comps},
            "measured_weight": round(wsum, 4),
            "unmeasured": {k: FG_MISSING[k] for k in FG_MISSING},
            "note": ("index renormalised over the measured components; the "
                     "unmeasured published weighting (dominance, social, "
                     "surveys, trends) is not faked"),
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


class EmotionSystem:
    """Computes the affect state (see module docstring) + learns channel weights."""

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
        cl = ctx.P                                  # timeframe window closes
        mc = market["closes"]                       # long window (>= 60+ mins)
        open_ = market.get("session_open") or mc[0]
        live = market.get("live", cl[-1])

        # ---- measured tape features (never the signal — G-3) -----------
        rets = [(mc[i] - mc[i - 1]) / mc[i - 1] for i in range(1, len(mc))
                if mc[i - 1]]
        n_w = max(2, len(cl) - 1)
        r_w = (live / open_ - 1.0) if open_ else 0.0
        s1 = std_pop(rets[-10:]) if len(rets) >= 2 else 0.0
        s30 = std_pop(rets[-min(30, len(rets)):]) if len(rets) >= 2 else 0.0
        vols = market.get("volumes", [0.0] * len(mc))
        v5 = sum(vols[-5:]) / min(5, len(vols)) if vols else 0.0
        v10 = sum(vols[-10:]) / min(10, len(vols)) if vols else 0.0
        v30 = sum(vols[-30:]) / min(30, len(vols)) if vols else 0.0
        vall = sum(vols) / len(vols) if vols else 0.0
        hist = ctx.hist
        zero_reasons: dict[str, str] = {}
        emos: dict[int, float] = {}
        dirs: dict[int, int] = {}

        # drift t-stat of the window: the single most defensible "how strong
        # is the current move" measure (return / its own standard error)
        se = s30 * math.sqrt(n_w) if (s30 > 0 and n_w > 0) else 0.0
        z_drift = (r_w / se) if se > 1e-12 else 0.0
        z_drift = _clamp(z_drift, -8.0, 8.0)

        # ---- 1 VALENCE (Russell 1980, pleasure–displeasure) -------------
        # asinh-compressed drift: a two-sigma move and a ten-sigma one must
        # not both print 0.99, or the axis stops carrying information
        _v_raw = math.tanh(math.asinh(z_drift) / 1.5)
        valence = 0.5 * (1.0 + _v_raw)
        emos[1] = valence
        dirs[1] = sign(r_w)

        # ---- 2 AROUSAL (Russell 1980, activation) -----------------------
        vol_ratio = (s1 / s30) if s30 > 1e-12 else 1.0
        climax = (v5 / v30) if v30 > 0 else 1.0
        arousal = sig(1.6 * math.log(max(vol_ratio, 1e-6))
                      + 0.35 * abs(z_drift))
        emos[2] = _clamp(arousal, 0.0, 1.0)
        dirs[2] = 0                                   # magnitude, not a side

        # ---- 3 DOMINANCE (Mehrabian & Russell 1974, felt control) -------
        moves = [(cl[i] - cl[i - 1]) / cl[i - 1]
                 for i in range(1, len(cl)) if cl[i - 1]]
        aligned = sum(1 for m in moves if sign(m) == sign(r_w) and m != 0)
        persist = (aligned / len(moves)) if moves else 0.5
        # graded, not rail-to-rail: gain 5 pinned this axis at 0.99/0.01 on
        # any trending window, which then won every "dominant" crown by
        # saturation instead of by information
        dominance = _clamp(0.5 + 0.5 * math.tanh(
            2.4 * (persist - 0.5) + 0.30 * abs(z_drift)), 0.0, 1.0)
        emos[3] = _clamp(dominance, 0.0, 1.0)
        dirs[3] = sign(r_w) if abs(z_drift) > 0.3 else 0

        # ---- 4/5 PANAS (Watson, Clark & Tellegen 1988) ------------------
        up, dn = _semi_deviations(rets[-min(60, len(rets)):])
        tot = up + dn
        up_share = (up / tot) if tot > 1e-12 else 0.5
        act = (emos[2] - 0.5) * 1.2          # both PA and NA are ACTIVATED
        pa = sig(3.2 * (up_share - 0.5) + act)
        na = sig(3.2 * ((1.0 - up_share) - 0.5) + act)
        emos[4] = _clamp(pa, 0.0, 1.0)
        emos[5] = _clamp(na, 0.0, 1.0)
        dirs[4] = 1
        dirs[5] = -1

        # ---- 6..11 EKMAN (1992) basic emotions via prototypes -----------
        eks = ekman_scores(emos[1], emos[2], emos[3])
        _ek_ids = {"fear": 6, "anger": 7, "disgust": 8, "sadness": 9,
                   "surprise": 10, "enjoyment": 11}
        _ek_dir = {"fear": -1, "anger": -1, "disgust": -1, "sadness": -1,
                   "surprise": 0, "enjoyment": 1}
        for name, eid in _ek_ids.items():
            emos[eid] = float(eks[name]["intensity"])
            dirs[eid] = _ek_dir[name]
            if eks[name]["distance"] > 2.2:
                zero_reasons[str(eid)] = (
                    f"{name} prototype {eks[name]['distance']:.2f} away in "
                    f"(v,a,d) — state is not this emotion")

        # ---- 12 LOSS-AVERSION PRESSURE (prospect theory, λ=2.25) --------
        peak = max(mc[-min(60, len(mc)):]) if mc else live
        drawdown = ((peak - live) / peak) if peak > 0 else 0.0
        dd_z = (drawdown / (s30 * math.sqrt(n_w))) if (s30 > 0) else 0.0
        # tanh, not sigmoid: at 1.1·λ·dd_z the channel pinned to exactly
        # 1.000 on any ≥2σ pullback, which made "dominant" ties trivial
        # 0.22·λ keeps a 1σ pullback mid-scale and a 3σ one high-but-graded;
        # the earlier 1.1·λ pinned this channel at 0.998 on ordinary
        # pullbacks, so it won every "dominant" crown by saturation
        emos[12] = _clamp(0.5 + 0.5 * math.tanh(0.22 * (LOSS_LAMBDA * dd_z)
                                               - 0.15), 0.0, 1.0)
        dirs[12] = -1

        # ---- 13 SOMATIC MARKER (Damasio 1994) --------------------------
        _key = (ctx.symbol, ctx.tf)
        soma = float(self._soma.get(_key, 0.5))
        emos[13] = _clamp(soma, 0.0, 1.0)
        dirs[13] = 1 if soma > 0.5 else (-1 if soma < 0.5 else 0)
        if self.n_updates.get(_key, 0) < 5:
            zero_reasons["13"] = ("no settled outcomes yet — marker still "
                                  f"at neutral 0.50 ({self.n_updates.get(_key, 0)} updates)")

        # ---- 14 HERDING (Christie & Huang 1995 CSAD) -------------------
        peer = market.get("peer_returns") or []
        disp = None
        rows = [row for row in peer[-10:] if isinstance(row, (list, tuple))
                and len(row) >= 2]
        if rows:
            sds = [std_pop([float(x) for x in row]) for row in rows]
            disp = sum(sds) / len(sds)
        if disp is not None and s30 > 1e-12:
            emos[14] = sig(-4.0 * ((disp / s30) - 1.0))
        else:
            emos[14] = 0.5
            zero_reasons["14"] = ("no peer cross-section in this window — "
                                  "CSAD cannot be measured")
        dirs[14] = 0                                  # amplifier, not a side

        # ---- 15 CAPITULATION (panic selling; rtp alias) ----------------
        emos[15] = sig(3.0 * (emos[5] - 0.5) + 3.0 * (0.5 - emos[3])
                       + 1.2 * (climax - 1.0))
        dirs[15] = 1                                  # contrarian: fear lows
        rtp_val = emos[15]

        # ---- 16 EXTRAPOLATION / CHASE (Barberis et al. 2015) -----------
        d1 = mc[-1] - mc[-6] if len(mc) > 6 else 0.0
        d2 = mc[-6] - mc[-11] if len(mc) > 11 else 0.0
        sd5 = (std_pop(rets[-5:]) * math.sqrt(5.0)) if len(rets) >= 5 else 0.0
        z5 = (d1 / (sd5 * live)) if (sd5 > 1e-12 and live > 0) else 0.0
        # asinh-compressed: on a monotone tape sd5→0 and a plain tanh pinned
        # this channel at exactly 1.000, which then "won" the dominant crown
        emos[16] = _clamp(0.5 + 0.5 * math.tanh(0.8 * math.asinh(z5)),
                          0.0, 1.0)
        dirs[16] = sign(d1) if emos[16] > 0.55 else 0

        # ---- 17 VOLATILITY STRESS (VIX concept, F&G volatility) --------
        emos[17] = sig(2.5 * math.log(max(vol_ratio, 1e-6)))
        dirs[17] = -1

        # ---- 18 MOMENTUM vs MEAN (F&G momentum/volume) -----------------
        ma30 = (sum(mc[-min(30, len(mc)):]) / min(30, len(mc))) if mc else live
        # log gap / (σ·√n): the old `(live − ma30) / (σ√n · live)` divided by
        # the price a second time, shrinking every reading ~100×, so the
        # momentum channel read ~0.35 on a roaring uptrend
        mom_z = ((math.log(live / ma30) / (s30 * math.sqrt(min(30, len(mc)))))
                 if (s30 > 0 and live > 0 and ma30 > 0) else 0.0)
        emos[18] = sig(1.4 * _clamp(mom_z, -4.0, 4.0))
        dirs[18] = sign(mom_z) if abs(mom_z) > 0.2 else 0

        # ---- 19 MARKET BREADTH (F&G dominance) -------------------------
        frac_same = None
        if rows:
            last = rows[-1]
            same = sum(1 for x in last if sign(float(x)) == sign(r_w)
                       and float(x) != 0)
            frac_same = same / len(last) if last else None
        if frac_same is None:
            emos[19] = 0.5
            zero_reasons["19"] = ("no peer breadth in this window — "
                                  "cross-section absent")
        else:
            emos[19] = sig(4.0 * (frac_same - 0.5))
        dirs[19] = sign(r_w) if frac_same is not None else 0

        # ---- 20 NEWS SENTIMENT (external NLP proxy, labelled) ----------
        pol = float(news_agg.get("polarity", 0) or 0)
        count = int(news_agg.get("count", 0) or 0)
        if count:
            emos[20] = sig(4.0 * pol)
            dirs[20] = 1 if pol > 1e-6 else (-1 if pol < -1e-6 else 0)
        else:
            emos[20] = 0.5
            dirs[20] = 0
            zero_reasons["20"] = "no scored headlines this window — abstains"

        # ---- 21 TREND EXHAUSTION (Frijda action tendency input) --------
        denom = abs(d2) + abs(d1)
        decay = ((abs(d2) - abs(d1)) / denom) if denom > 1e-12 else 0.0
        emos[21] = sig(4.0 * decay)
        dirs[21] = -sign(r_w) if emos[21] > 0.6 else 0
        tedr_val = emos[21]

        # ---- composite (same mechanics as before, real channels) -------
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
        ek = ekman_scores(emos[1], emos[2], emos[3])
        prim = max(ek, key=lambda k: ek[k]["intensity"])
        runner = max((k for k in ek if k != prim),
                     key=lambda k: ek[k]["intensity"])
        fg = fear_greed_index(vol_stress=emos[17], momentum=emos[18],
                              herding=emos[14])
        # breadth is its own internal reading (peer cross-section), NOT the
        # published BTC-dominance component — reported beside the index
        fg["breadth_component"] = round(float(emos[19]), 4)
        # core affect and the index answer different questions (fast price
        # affect vs volatility/momentum survey). When they disagree, say WHY
        # instead of letting the panel look self-contradictory.
        _v_raw = 2.0 * emos[1] - 1.0
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
        act = action_tendency(fear=emos[6], anger=emos[7],
                              sadness=emos[9], enjoyment=emos[11],
                              pa=emos[4], na=emos[5], dominance=emos[3],
                              capitulation=emos[15], trend_dir=sign(r_w))

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
        _prev = self._prev_emos.get(_key) or {}

        def _dev_of(i: int) -> float:
            neutral = CHANNELS[i][3]
            span = (1.0 - neutral) if neutral == 0.0 else 0.5
            return abs(float(emos[i]) - neutral) / span

        _ranked = sorted(
            ((_dev_of(i),
              abs(float(emos[i]) - float(_prev.get(i, emos[i]))),
              float(w[i - 1]), i)
             for i in EMO_IDS),
            # tie-break order matters: the ROUNDED deviation groups near-ties
            # (so a 0.0002 wobble does not re-crown every pass), but the raw
            # deviation must still compare first — otherwise a channel with
            # dev 0.9998 could be crowned while another sat at exactly 1.0
            # and the "runner-up" outranked the winner on the same card.
            key=lambda t: (round(t[0], 3), t[0], t[1], t[2]), reverse=True)
        _dev, _dlt, _wv, _dom_id = _ranked[0]
        _tied = sum(1 for t in _ranked
                    if t[0] >= _dev - 0.01 and t[3] != _dom_id)
        _now = _time.time()
        _cur = self._dom.get(_key)
        if _cur is None or _cur[0] != _dom_id:
            self._dom[_key] = (_dom_id, _now)
            _held = 0.0
        else:
            _held = _now - _cur[1]
        self._prev_emos[_key] = dict(emos)
        dominant = {
            "id": int(_dom_id),
            "name": CHANNELS[_dom_id][1],
            "key": CHANNELS[_dom_id][0],
            "cite": CHANNELS[_dom_id][2],
            "value": round(float(emos[_dom_id]), 5),
            "intensity": round(_dev, 4),          # 0 = neutral, 1 = extreme
            "moved": round(_dlt, 5),
            "held_s": round(_held, 1),
            "tied": int(_tied),
            "runner_up": {"id": int(_ranked[1][3]),
                          "name": CHANNELS[_ranked[1][3]][1],
                          "intensity": round(_ranked[1][0], 4)},
        }

        for eid in EMO_IDS:
            if float(emos[eid]) == CHANNELS[eid][3] and str(eid) not in zero_reasons:
                zero_reasons[str(eid)] = (
                    f"{CHANNELS[eid][0]} exactly at neutral "
                    f"{CHANNELS[eid][3]:.2f} — no deviation this window")

        return {
            "model": dict(MODEL),
            "dominant": dominant,
            "emotions": {k: round(v, 5) for k, v in emos.items()},
            "zero_reasons": zero_reasons,
            "zero_causes": _causes(zero_reasons),
            "directions": dirs,
            "cerc": round(cerc, 5),
            "net_emotion": round(net_emotion, 5),
            "agreement": round(agreement, 5),
            "breadth": {"bull": bulls, "bear": bears,
                        "flat": 21 - bulls - bears},
            "rtp": round(rtp_val, 5),
            "tedr": round(tedr_val, 5),
            "prev_cerc": prev_cerc,
            "lag_values": lag_rings,
            # ---- the real model, printed -------------------------------
            "core_affect": {"valence": round(2.0 * emos[1] - 1.0, 4),
                            "arousal": round(emos[2], 4),
                            "note": ("Russell (1980) circumplex: valence "
                                     "−1…+1 × arousal 0…1")},
            "pad": {"pleasure": round(2.0 * emos[1] - 1.0, 4),
                    "arousal": round(emos[2], 4),
                    "dominance": round(emos[3], 4),
                    "note": "Mehrabian & Russell (1974)"},
            "panas": {"positive": round(emos[4], 4),
                      "negative": round(emos[5], 4),
                      "balance": round(emos[4] - emos[5], 4),
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
                "loss_pressure": round(emos[12], 4),
                "somatic_marker": round(emos[13], 4),
                "risk_appetite": round(emos[4] - emos[5], 4),
                "note": ("prospect theory λ from Tversky & Kahneman (1992); "
                         "somatic marker learned from settled outcomes"),
            },
            "features": {
                "samples": len(mc),
                "vol_z": round(math.log(max(vol_ratio, 1e-6)), 4),
                "climax_ratio": round(climax, 4),
                "violence_gate": round(_clamp(0.5 * max(0.0, climax - 1.0)
                                              + 0.5 * max(0.0, vol_ratio - 1.0),
                                              0.0, 1.0), 4),
                "vol_expansion": round(vol_ratio, 4),
                "window_streak": int(aligned),
                "r30": round(r_w, 6),
                "pdi_session": round(emos[3], 4),
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
            "retail": _crowd(emos, dirs, (16, 15, 20, 19, 11),
                             ("FOMO-chasing (chase + news euphoria)",
                              "panic-selling (capitulation)",
                              "retail quiet")),
            "hedge": _crowd(emos, dirs, (12, 13, 3, 17, 21),
                            ("hedges pressing (loss pressure + stress)",
                             "risk appetite returning (somatic marker)",
                             "hedges quiet")),
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
            acc[i] = ACC_SMOOTH * acc[i] + (1 - ACC_SMOOTH) * (1.0 if correct else 0.0)
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
    label = labels[0] if score > 0.15 else labels[1] if score < -0.15 \
        else labels[2]
    return {"score": round(score, 4), "label": label,
            "channels": list(ids)}
