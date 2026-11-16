"""Section 5 — EMOTIONAL FLOW (E1..E22).

Twenty-two dimensions with exact coefficient sets from the specification:

  E1  FII   Fear Intensity Index          E12 DRD   Daily Range Deviation
  E2  GSL   Guardian-of-Loss Syndrome     E13 ERI   Extreme Reversal Index
  E3  FPR   Fear Put Ratio                E14 OFS   Optimism/Fear Sentiment
  E4  PCP   Panic Coefficient (rising)    E15 IAF   Intent/Action Fraction
  E5  CD    Octopamine Reflex (original)  E16 LAI   Lead/Announce Impulse
  E6  ESI   Endorphin Session Intensity   E17 CDP   Crowd-Directional Pull
  E7  DPI   Dopamine (smart money)        E18 HTI   Honeymoon Trend Inversion
  E8  ATR   Volume-ATR Emotion            E19 SMDM  Smart-Money Divergence
  E9  RTP   Roll-over Trauma/Panic        E20 NMS   News/Momentum Surge
  E10 ABS   Absorption by smart money     E21 TEDR  Trend Exhaustion Decay
  E11 HMC   Herd/Mirror Co-movement       E22 CERC  Composite (weighted sum)

Every E1..E21 output is a sigmoid/fraction in [0,1]; each contributes to CERC
as  w_e * E_e * direction_e  with the TEDR participation multiplier
(1 - 0.5 * E21) on the TEDR term.  E22 CERC weights start at 1/21 and learn
with the same eta=0.02 clamp/renormalize reward rule as the CCI weights, and
CERC feeds the CCI as weighted component id 98.
"""
from __future__ import annotations

import math

from .config import HIST_EMOTION_LAG
from .models import FormulaContext
from .formulas._helpers import pearson, sign, std_pop

ETA = 0.015               # Section 10: η_emotion — floor (see COLD rate below)
W_CLAMP_LO = 0.005        # widened: predictive channels can dominate
W_CLAMP_HI = 0.30
ETA_COLD = 0.08           # cold-start rate — a fresh brain learns FAST
COLD_HALFLIFE = 150
ACC_SMOOTH = 0.90
EMO_IDS = list(range(1, 22))          # CERC is E22 = the composite itself
CERC_WEIGHT_ID = 22


def sig(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ══════════════════════════════════════════════════════════════════════
# Estimator helpers (guide §2): every channel is squash()'d into
# [-1, +1], so nothing can sit in a dead zone and nothing can saturate
# at a hard 0.00 / 1.00 cliff.
# ══════════════════════════════════════════════════════════════════════
def squash(x: float, scale: float = 1.5) -> float:
    """tanh(x/scale) — continuous, differentiable, bounded in (-1, 1)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(x):
        return 0.0
    return math.tanh(x / (scale if scale else 1.0))


def _ema(xs, span: int) -> float:
    seq = list(xs)[-max(2 * span, span):] if xs else []
    if not seq:
        return 0.0
    a = 2.0 / (span + 1.0)
    out = float(seq[0])
    for v in seq[1:]:
        out = a * float(v) + (1.0 - a) * out
    return out


def _rsi(xs, period: int = 14) -> float:
    seq = list(xs)[-(period + 1):] if xs else []
    if len(seq) < 3:
        return 50.0
    gains = losses = 0.0
    for i in range(1, len(seq)):
        d = float(seq[i]) - float(seq[i - 1])
        if d > 0:
            gains += d
        else:
            losses -= d
    if gains + losses <= 1e-12:
        return 50.0
    return 100.0 * gains / (gains + losses)


def _ad_impulse(hi, lo, cls, vols, k: int = 8) -> float:
    """Volume-weighted close-location value (accumulation/distribution)."""
    n = min(len(hi), len(lo), len(cls), len(vols), max(1, k))
    if n <= 0:
        return 0.0
    num = den = 0.0
    for h_, l_, c_, v_ in zip(hi[-n:], lo[-n:], cls[-n:], vols[-n:]):
        rng = float(h_) - float(l_)
        if rng <= 1e-12:
            continue
        num += float(v_) * ((float(c_) - float(l_)) - (float(h_) - float(c_))) / rng
        den += float(v_)
    return (num / den) if den > 1e-12 else 0.0


def _gk_vol(hi, lo, cls, k: int) -> float:
    """Garman-Klass realised volatility (Parkinson term, log-space)."""
    n = min(len(hi), len(lo), len(cls), max(2, k))
    if n < 3:
        return 0.0
    vals = []
    for h_, l_, c_ in zip(hi[-n:], lo[-n:], cls[-n:]):
        if float(h_) <= 0 or float(l_) <= 0 or float(c_) <= 0 or h_ <= l_:
            continue
        vals.append(max(0.0, 0.5 * math.log(float(h_) / float(l_)) ** 2))
    return math.sqrt(sum(vals) / len(vals)) if vals else 0.0


def _wick_pressure(hi, lo, cls, k: int = 10) -> float:
    """Upper-wick share of the bar range over k bars: 1 = heavy rejection."""
    n = min(len(hi), len(lo), len(cls), max(1, k))
    if n <= 0:
        return 0.0
    tot = 0.0
    used = 0
    for h_, l_, c_ in zip(hi[-n:], lo[-n:], cls[-n:]):
        rng = float(h_) - float(l_)
        if rng <= 1e-12:
            continue
        tot += (float(h_) - max(float(c_), float(l_))) / rng
        used += 1
    return max(0.0, min(1.0, tot / used)) if used else 0.0


def _mean_range(hi, lo, cls, k: int) -> float:
    """Mean relative bar range (high-low)/close over the last k bars."""
    n = min(len(hi), len(lo), len(cls), max(1, k))
    if n <= 0:
        return 0.0
    vals = [(float(hi[-i]) - float(lo[-i])) / float(cls[-i])
            for i in range(1, n + 1) if float(cls[-i]) > 0]
    return sum(vals) / len(vals) if vals else 0.0


def _vwap(cls, vols, k: int = 20) -> float:
    n = min(len(cls), len(vols), max(1, k))
    if n <= 0:
        return float(cls[-1]) if cls else 0.0
    c, v = list(cls)[-n:], list(vols)[-n:]
    den = sum(float(x) for x in v)
    if den <= 1e-12:
        return float(c[-1])
    return sum(float(a) * float(b) for a, b in zip(c, v)) / den


def _crowd(emos, dirs, ids, labels):
    """Signed crowd read from a channel subset: mean of
    (intensity x direction), intensity = |v-0.5|x2. Labels are the
    (positive, negative, flat) human readings of that crowd."""
    vals = []
    for i in ids:
        # RTP (E9) is a magnitude in [0, 1] with zero meaning no trauma;
        # unlike the sigmoid channels its neutral point is 0, not 0.5.
        neutral = 0.0 if i == 9 else 0.5
        v = float(emos.get(i, neutral))
        d = int(dirs.get(i, 0) or 0)
        scale = 1.0 if i == 9 else 0.5
        vals.append((abs(v - neutral) / scale) * d)
    score = sum(vals) / len(vals) if vals else 0.0
    label = labels[0] if score > 0.15 else labels[1] if score < -0.15 \
        else labels[2]
    return {"score": round(score, 4), "label": label,
            "channels": list(ids)}


def _lag_value(ring) -> float | None:
    """GRC-lagging comparator: the value from HIST_EMOTION_LAG GRCs ago
    (oldest ring entry once full; the most recent entry during bootstrap)."""
    if len(ring) == 0:
        return None
    if len(ring) >= HIST_EMOTION_LAG:
        return ring.values()[0]
    return ring.last()


def _neg_frac(closes: list[float], n: int) -> float:
    seg = closes[-n:]
    if len(seg) < 2:
        return 0.0
    rets = [(seg[i] - seg[i - 1]) / seg[i - 1] for i in range(1, len(seg))]
    if not rets:
        return 0.0
    return sum(1 for r in rets if r < 0) / len(rets)


def _atr5(ohlc: dict | None, closes: list[float]) -> float:
    """ATR_5: average true range over the last 5 minutes."""
    if ohlc and len(ohlc.get("highs", [])) >= 6:
        h, l, c = ohlc["highs"], ohlc["lows"], ohlc["closes"]
        trs = []
        for i in range(-5, 0):
            pc = c[i - 1]
            trs.append(max(h[i] - l[i], abs(h[i] - pc), abs(l[i] - pc)))
        return max(1e-9, sum(trs) / 5)
    if len(closes) < 6:
        return 1e-9
    trs = [abs(closes[i] - closes[i - 1]) for i in range(-5, 0)]
    return max(1e-9, sum(trs) / 5)



def _causes(zero_reasons: dict) -> dict:
    """Cause bucket per zero (error / both sides equal / input / …)."""
    from .formulas.trace import classify_zero
    return {k: classify_zero(v) for k, v in zero_reasons.items()}

class EmotionSystem:
    """Holds the 21 CERC weights per (symbol, timeframe) and computes E1..E22."""

    def __init__(self) -> None:
        self.weights: dict[tuple[str, str], list[float]] = {}
        self.acc: dict = {}        # (sym, tf) -> 21 EWMA accuracies
        self.n_updates: dict = {}  # (sym, tf) -> reward-call count
        # dominant-channel tracking: previous pass + since-when, so the
        # DOMINANT NOW header is measured server-side (survives reloads)
        self._prev_emos: dict = {}
        self._dom: dict = {}       # (sym, tf) -> (channel id, since ts)

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
        """EWMA predictive accuracy per CERC channel (0.5 = uninformative)."""
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

        # TPDI fallback: direction = window displacement sign
        trend_w = sign(cl[-1] - cl[0]) if cl[0] else 0
        trend10 = sign(mc[-1] - mc[-11]) if len(mc) > 11 else trend_w
        r30 = (mc[-1] / mc[-31] - 1.0) if len(mc) > 31 else (
            (cl[-1] / cl[0] - 1.0) if cl[0] else 0.0
        )
        rets = [(mc[i] - mc[i - 1]) / mc[i - 1] for i in range(1, len(mc))]
        s1 = std_pop(rets[-10:]) if len(rets) >= 2 else 0.0
        s30 = std_pop(rets[-min(30, len(rets)):]) if len(rets) >= 2 else 0.0
        vols = market.get("volumes", [0.0] * len(mc))
        v10 = sum(vols[-10:]) / min(10, len(vols)) if vols else 0.0
        v30 = sum(vols[-30:]) / min(30, len(vols)) if vols else 0.0
        v5 = sum(vols[-5:]) / min(5, len(vols)) if vols else 0.0
        vall = sum(vols) / len(vols) if vols else 0.0
        hist = ctx.hist
        emos: dict[int, float] = {}
        dirs: dict[int, int] = {}

        # ══════════════════════════════════════════════════════════════
        # FIRST-PRINCIPLES ESTIMATORS (guide §2)
        #
        # Every channel is a CONTINUOUS signed market variable in
        # [-1, +1] — negative = bearish fear / stress / distribution,
        # positive = bullish greed / FOMO / accumulation — produced by
        # squash(x, scale) = tanh(x/scale).  No discrete triggers, no
        # lag comparators that flat-line, no dead-zone 0.00 locks.
        #
        # The UI keeps its 0..1 dial scale (0.5 = neutral):
        #     emos[i] = 0.5 + 0.5 * signed[i]
        # so the banner, the E1–E22 grid and CERC all quote one array.
        # ══════════════════════════════════════════════════════════════
        signed: dict[int, float] = {}
        ohlc = market.get("ohlc") or {}
        Hi = [float(x) for x in (ohlc.get("highs") or [])]
        Lo = [float(x) for x in (ohlc.get("lows") or [])]
        degenerate = (not Hi or not Lo
                      or sum(max(0.0, h_ - l_) for h_, l_ in zip(Hi, Lo)) <= 0.0)
        if degenerate or len(Hi) < len(mc) or len(Lo) < len(mc):
            # kline proxy: bar range from the close-to-close move. Without
            # this a tape whose high == low == close (a synthetic feed, or a
            # quiet print) made E7/E8/E9 identically 0.000 — the dead zeros
            # the audit is about.
            Hi = [max(mc[i - 1] if i else mc[i], mc[i]) for i in range(len(mc))]
            Lo = [min(mc[i - 1] if i else mc[i], mc[i]) for i in range(len(mc))]
        try:
            atr_v = float(market.get("atr") or 0.0)
        except (TypeError, ValueError):
            atr_v = 0.0
        if not (atr_v > 0):
            atr_v = _atr5(ohlc, mc)
        px = live if live > 0 else (mc[-1] if mc else 0.0)
        atr_pct = (atr_v / px) if px > 0 else 0.0

        def col(name: str, default: float = 0.0) -> float:
            try:
                v = float(market.get(name))
            except (TypeError, ValueError):
                return default
            return v if math.isfinite(v) else default

        def ret_at(k: int) -> float:
            return (mc[-1] / mc[-1 - k] - 1.0) if len(mc) > k and mc[-1 - k] else 0.0

        # order-flow / micro seams — kline proxies when the book is silent
        dbi = _clamp(col("dbi", col("ofti", 0.0)), -1.0, 1.0)
        mpd = _clamp(col("mpd", col("cmp_divergence", 0.0)), -1.0, 1.0)
        news_pol = 0.0
        try:
            news_pol = _clamp(float(news_agg.get("polarity") or 0.0), -1.0, 1.0)
        except (TypeError, ValueError):
            news_pol = 0.0
        news_count = 0
        try:
            news_count = int(news_agg.get("count") or 0)
        except (TypeError, ValueError):
            news_count = 0

        seg = [mc[i] / mc[i - 1] - 1.0 for i in range(max(1, len(mc) - 20), len(mc))
               if mc[i - 1]]

        # ---- E1  FEAR — downside vs upside realised semivariance --------
        up_var = sum(max(r, 0.0) ** 2 for r in seg) / len(seg) if seg else 0.0
        dn_var = sum(min(r, 0.0) ** 2 for r in seg) / len(seg) if seg else 0.0
        signed[1] = squash((up_var - dn_var) / (up_var + dn_var + 1e-12), 0.4)

        # ---- E2  LOSS AVERSION — position in the trailing 24-bar range --
        win = mc[-24:] if len(mc) >= 24 else mc
        lo24, hi24 = (min(win), max(win)) if win else (px, px)
        pos24 = 2.0 * (px - lo24) / (hi24 - lo24 + 1e-9) - 1.0
        signed[2] = squash(pos24, 0.8)

        # ---- E3  HEDGE DEMAND — depth imbalance less spread expansion ---
        sp20 = _mean_range(Hi, Lo, mc, 20)
        sp60 = _mean_range(Hi, Lo, mc, 60)
        spread_z = (sp20 / sp60 - 1.0) if sp60 > 1e-12 else 0.0
        signed[3] = squash(dbi - 0.3 * spread_z, 0.8)

        # ---- E4  PANIC — 3-bar return over the 3-bar ATR ----------------
        signed[4] = squash(ret_at(3) / (atr_pct * math.sqrt(3.0) + 1e-9), 1.2)

        # ---- E5  STARTLE REFLEX — true-range shock, signed by the bar ---
        tr = (Hi[-1] - Lo[-1]) if Hi else 0.0
        bar_dir = sign(mc[-1] - (mc[-2] if len(mc) > 1 else mc[-1]))
        signed[5] = squash(bar_dir * ((tr / (atr_v + 1e-9)) - 1.0), 1.0)

        # ---- E6  SESSION GREED — 60-bar drift over 60-bar volatility ----
        c60 = mc[-61] if len(mc) > 60 else (mc[0] if mc else px)
        r60 = [mc[i] / mc[i - 1] - 1.0
               for i in range(max(1, len(mc) - 60), len(mc)) if mc[i - 1]]
        sd60 = (std_pop(r60) * math.sqrt(60.0)) if len(r60) > 2 else 0.0
        signed[6] = squash((px - c60) / (px * sd60 + 1e-9), 1.5)

        # ---- E7  SMART-MONEY FLOW — 8-bar volume-weighted CLV -----------
        signed[7] = squash(_ad_impulse(Hi, Lo, mc, vols, 8), 0.5)

        # ---- E8  VOLATILITY STRESS — Garman-Klass expansion, signed -----
        gk10, gk60 = _gk_vol(Hi, Lo, mc, 10), _gk_vol(Hi, Lo, mc, 60)
        gk_ratio = (gk10 / gk60 - 1.0) if gk60 > 1e-12 else 0.0
        signed[8] = squash((sign(ret_at(5)) or sign(trend_w)) * gk_ratio, 0.6)

        # ---- E9  ROLLOVER PANIC — upper-wick rejection pressure ---------
        pain = _wick_pressure(Hi, Lo, mc, 10)          # 0..1, 1 = heavy top
        signed[9] = -pain

        # ---- E10 ABSORPTION — heavy volume, low displacement ------------
        v_sma20 = (sum(vols[-20:]) / len(vols[-20:])) if vols else 0.0
        vol_z = ((vols[-1] / v_sma20 - 1.0) if v_sma20 > 0 else 0.0)
        disp = abs(ret_at(1)) / (atr_pct + 1e-9)
        sgn_mpd = sign(mpd) or (-sign(ret_at(1)))
        signed[10] = squash(vol_z * sgn_mpd / (disp + 0.5), 1.5)

        # ---- E11 HERDING — signed run length + lag-1 autocorrelation ----
        fwd = (seg[-1] > 0) if seg else False
        run = 0
        for r in reversed(seg):
            if r != 0 and (r > 0) == fwd:
                run += 1
            else:
                break
        ac1 = pearson(seg[:-1], seg[1:]) if len(seg) > 3 else 0.0
        signed[11] = squash(0.25 * (1 if fwd else -1) * min(run, 8) if run else 0.0
                            + 1.5 * ac1, 1.0)

        # ---- E12 RANGE FATIGUE — Kaufman efficiency ratio ---------------
        eff = 0.0
        if len(mc) > 12 and mc[-13]:
            path = sum(abs(mc[i] - mc[i - 1])
                       for i in range(len(mc) - 12, len(mc)))
            eff = (mc[-1] - mc[-13]) / (path + 1e-9)
        signed[12] = squash(eff, 0.5)

        # ---- E13 REVERSAL FEAR — mean-reversion tension vs the VWAP -----
        vwap20 = _vwap(mc, vols, 20)
        signed[13] = squash(-(px - vwap20) / (2.0 * atr_v + 1e-9), 1.0)

        # ---- E14 SENTIMENT — fast/slow EMA spread -----------------------
        signed[14] = squash((_ema(mc, 5) - _ema(mc, 20)) / (atr_v + 1e-9), 1.2)

        # ---- E15 INTENT TO ACT — signed volume surge --------------------
        if vols:
            signed[15] = squash(bar_dir * (vols[-1] / (v_sma20 + 1e-9) - 1.0), 1.0)
        else:
            signed[15] = 0.0

        # ---- E16 ANNOUNCEMENT SHOCK — news velocity × tick-rate surge ---
        tick_ratio = _clamp(col("tick_rate", 1.0)
                            / (col("tick_rate_baseline", 1.0) + 1e-9), 0.5, 3.0)
        signed[16] = squash(news_pol * max(0.5, tick_ratio) * 2.0, 0.5)

        # ---- E17 CROWD PULL — second-order price acceleration -----------
        accel3 = 0.0
        if len(mc) > 6:
            accel3 = ((mc[-1] - mc[-4]) - (mc[-4] - mc[-7])) / (atr_v + 1e-9)
        signed[17] = squash(accel3, 1.0)

        # ---- E18 TREND-CHASE FATIGUE — late leg vs early leg ------------
        r12, late3 = ret_at(12), ret_at(3)
        signed[18] = squash(late3 / (abs(r12) + atr_pct + 1e-9)
                            - 0.25 * (sign(r12) or sign(trend_w)), 0.8)

        # ---- E19 SMART-MONEY DIVERGENCE — book vs crowd -----------------
        signed[19] = squash(0.5 * (dbi + mpd) - 0.5 * signed[11], 0.8)

        # ---- E20 NEWS FOMO — asymmetric news polarity impulse -----------
        signed[20] = squash(news_pol * 2.0, 0.7)

        # ---- E21 TREND EXHAUSTION — centred RSI-14, volume-attenuated ---
        rsi_c = (_rsi(mc, 14) - 50.0) / 50.0
        v_atten = _clamp((_ema(vols, 5) / (_ema(vols, 20) + 1e-9)) if vols else 1.0,
                         0.5, 1.5)
        signed[21] = squash(rsi_c * v_atten, 1.0)

        # ---- map the estimators onto the UI dial scale ------------------
        for i in range(1, 22):
            s = _clamp(signed.get(i, 0.0), -1.0, 1.0)
            emos[i] = 0.5 + 0.5 * s
            dirs[i] = 0 if abs(s) < 1e-4 else (1 if s > 0 else -1)
        # E9 keeps its 0-BASED trauma magnitude (0 = no pain) because RTP,
        # FDT and the risk layer consume it as a probability-like scalar
        emos[9] = _clamp(pain, 0.0, 1.0)
        dirs[9] = 1 if pain > 0.5 else -1
        # E8 is a magnitude channel (volatility stress), never a side
        dirs[8] = 0

        # ---- E22 CERC: precision-weighted signed composite --------------
        # guide §2: cerc = sum(w_i * E_i) / sum(w_i) over the SIGNED
        # estimators. Publishing it as the weighted mean (not a 0..1 dial
        # average) is what makes the banner text agree with the dials:
        # "bears in control" now really means cerc < 0.
        w = self._weights(ctx.symbol, ctx.tf)
        wsum = sum(w) or 1.0
        cerc = 0.0
        for i, eid in enumerate(EMO_IDS):
            term = w[i] * _clamp(signed.get(eid, 0.0), -1.0, 1.0)
            if eid == 21:
                term *= (1.0 - 0.5 * emos[21])   # TEDR participation multiplier
            cerc += term
        cerc = _clamp(cerc / wsum, -1.0, 1.0)

        # ---- shared feature bundle for the lag rings / UI trace ---------
        streak = int(run)
        lo30 = min(mc[-30:]) if len(mc) >= 30 else (min(mc) if mc else px)
        hi30 = max(mc[-30:]) if len(mc) >= 30 else (max(mc) if mc else px)
        idx_now = _clamp((px - lo30) / (hi30 - lo30), 0.0, 1.0) if hi30 > lo30 else 0.5
        climax = (v5 / v30) if v30 > 0 else 1.0
        vol_expand = (s1 / s30) if s30 > 0 else 1.0
        violence = max(0.0, min(1.0, 0.5 * max(0.0, climax - 1.0)
                                + 0.5 * max(0.0, vol_expand - 1.0)))
        pdi_sess = idx_now
        serotonin_brain = brain_res.get("hormones_avg", {}).get("serotonin", 0.5)
        dpi_raw = signed.get(7, 0.0)
        octo_raw = signed.get(5, 0.0)
        volchg = (vols[-1] / v10) if (vols and v10 > 0) else 1.0
        pol = 1 if news_pol > 0 else (-1 if news_pol < 0 else 0)
        count = news_count

        # GRC-lag rings (written by engine when record=True)
        lag_rings = {
            "dpi": dpi_raw,
            "octo": octo_raw * (1.0 + max(0.0, volchg - 1.0)),
            "pidx": idx_now,
            "nsp": emos[17] * (1 if pol >= 0 else -1),
        }
        zero_reasons: dict[str, str] = {}
        if emos.get(5, 1.0) == 0.0:
            zero_reasons["5"] = "no octopamine spike — reflex not fired this window"
        if emos.get(9, 0.5) == 0.0:
            zero_reasons["9"] = "no down closes in the last 10 minutes (all-up tape)"
        if emos.get(18, 1.0) == 0.0:
            zero_reasons["18"] = "lag comparator flat — index unchanged vs the lag ring"
        # EVERY zero needs a reason (zeros rule): channels that computed
        # to exactly 0 with no specific trigger get an honest generic one
        for _eid in range(1, 23):
            if float(emos.get(_eid, 0.0)) == 0.0 \
                    and str(_eid) not in zero_reasons:
                zero_reasons[str(_eid)] = (
                    "computed — channel exactly 0.0 this window "
                    "(no trigger, both sides balanced)")
        # ---- DOMINANT channel (server-side, honest) ---------------------
        # The old UI picked max(raw value): channels are [0,1] with 0.5
        # neutral, and several sig() channels saturate at 1.00 in calm
        # tapes — the FIRST saturated channel in index order (E2) then
        # won every tie, forever ("Greed is always top"). Rank instead
        # by DEVIATION FROM NEUTRAL |v-0.5|*2, break (near-)ties by
        # who MOVED most since the previous pass, and track held-time
        # here so it survives page reloads.
        import time as _time
        _key = (ctx.symbol, ctx.tf)
        _prev = self._prev_emos.get(_key) or {}
        _w = self._weights(ctx.symbol, ctx.tf)
        # E9 (roll-over panic) is a 0-BASED magnitude: 0 = no pain, so
        # its neutral is 0, not 0.5. Ranking it by |v-0.5| made it
        # "100% extreme" in every up-tape for days (user screenshot).
        _NEUTRAL = {9: 0.0}

        def _signed(i):
            """Signed, neutral-relative position in [-1, +1] — THE number
            the E1–E22 dials print. Audit §5.2: the banner ranked by
            |v − 0.5| while the dials printed (v − 0.5)·2, so a channel
            sitting at a dead 0.0 scored a full 100 % "extreme" and won
            the crown with nothing on the tape (E18 100 % vs +0.00)."""
            _n = _NEUTRAL.get(i, 0.5)
            _v = float(emos.get(i, _n))
            if _n == 0.0:
                return -_v                      # 0-based magnitude → bearish
            return 2.0 * (_v - _n)

        def _dev_of(i):
            return abs(_signed(i))

        # A DEAD ZERO is an absent measurement, not an extreme: if the
        # channel computed to exactly 0.0 (and carries a zero-reason) it
        # is neutral for ranking purposes and can never be "dominant".
        _live = [i for i in range(1, 22) if i in emos
                 and not (float(emos.get(i, 0.0)) == 0.0
                          and str(i) in zero_reasons)]
        if len(_live) < 2:
            _live = [i for i in range(1, 22) if i in emos] or list(range(1, 22))
        _ranked = sorted(
            ((_dev_of(i),
              abs(float(emos.get(i, 0.5))
                  - float(_prev.get(i, emos.get(i, 0.5)))),
              float(_w[i - 1]), i)
             for i in _live),
            key=lambda t: (round(t[0], 3), t[1], t[2]), reverse=True)
        _dev, _dlt, _wv, _dom_id = _ranked[0]
        # when several channels are pegged at the same extreme and
        # none moved, crowning one is arbitrary — SAY so (tied count)
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
            "value": round(float(emos[_dom_id]), 5),
            # the signed dial reading of the crowned channel — the banner
            # and the E-card now quote the SAME number (audit §5.2)
            "signed": round(_signed(_dom_id), 4),
            "intensity": round(_dev, 4),          # 0 = neutral, 1 = extreme
            "moved": round(_dlt, 5),
            "held_s": round(_held, 1),
            "tied": int(_tied),                   # others at same extreme
            "runner_up": {"id": int(_ranked[1][3]),
                          "intensity": round(_ranked[1][0], 4)},
        }
        return {
            "dominant": dominant,
            "emotions": {k: round(v, 5) for k, v in emos.items()},
            # the signed dial values the UI must render, so "DOMINANT NOW"
            # and the E1–E22 grid are guaranteed to agree (audit §5.2)
            "signed": {i: round(_signed(i), 5) for i in sorted(emos)},
            "neutral": {i: _NEUTRAL.get(i, 0.5) for i in sorted(emos)},
            "zero_reasons": zero_reasons,
            "zero_causes": _causes(zero_reasons),
            "directions": dirs,
            # E22 CERC — the signed net mood and its dial equivalent.
            # `emotions` stays E1..E21 (the composite is not one of its
            # own inputs); the E22 card reads `cerc` / `cerc_dial`.
            "cerc": round(cerc, 5),
            "cerc_dial": round(_clamp(0.5 + 0.5 * cerc, 0.0, 1.0), 5),
            "rtp": emos[9],
            "tedr": emos[21],
            "prev_cerc": prev_cerc,
            "lag_values": lag_rings,
            # advice §5.1 feature trace — every emotion is measured from
            # these tape features, never derived from the signal (G-3)
            "features": {
                "samples": len(mc),
                "vol_z": round((v10 - v30) / v30, 4) if v30 else 0.0,
                "climax_ratio": round(climax, 4),
                "violence_gate": round(violence, 4),
                "vol_expansion": round(vol_expand, 4),
                "window_streak": int(streak),
                "r30": round(r30, 6),
                "pdi_session": round(pdi_sess, 4),
                "serotonin": round(serotonin_brain, 4),
            },
            # RETAIL vs HEDGE/SMART-MONEY emotion — measured from the
            # channels that actually describe each crowd (herd/crowd/
            # news-surge vs smart-money/absorption/divergence), signed
            # by each channel's own direction. Genuine market reads,
            # not one blended mood.
            "retail": _crowd(emos, dirs, (11, 16, 17, 20, 6),
                             ("FOMO-chasing", "panic-selling", "retail quiet")),
            "hedge": _crowd(emos, dirs, (7, 10, 19, 2, 9),
                            ("hedges pressing (defensive bids)",
                             "smart money distributing",
                             "hedges quiet")),
        }

    # ------------------------------------------------------------------
    def reward_update(self, symbol: str, tf: str, emos: dict[int, float],
                      dirs: dict[int, int], cerc_before: float,
                      move: float, eta: float | None = None,
                      scale: float | None = None) -> None:
        """GRC-cadence auto-learning for the 21 CERC weights (eta rule).

        Accuracy-driven upgrades (user directive: weightage follows
        predictive accuracy):
          * COLD-START eta — rate decays from ETA_COLD to the floor with a
            150-update half-life, so the first few hundred cycles actually
            differentiate channels instead of freezing at uniform.
          * MAGNITUDE reward — a big verified move teaches more than noise:
            reward = ±clip(|move|/scale, 0.3, 1.5).
          * per-channel EWMA ACCURACY exposed to the UI/API.
        """
        move_dir = sign(move)
        if move_dir == 0:
            return
        w = self._weights(symbol, tf)
        acc = self._accuracy(symbol, tf)
        active = []
        for i, eid in enumerate(EMO_IDS):
            try:
                value = float(emos.get(eid, 0.0))
                direction = int(dirs.get(eid, 0) or 0)
            except (TypeError, ValueError, OverflowError):
                continue
            if (direction == 0 or not math.isfinite(value)
                    or abs(value) <= 1e-12):
                continue                    # abstention is not a miss
            term = w[i] * value * direction
            if eid == 21:
                term *= (1.0 - 0.5 * value)
            if abs(term) > 1e-12:
                active.append((i, term))
        if not active:
            return
        base = ETA if eta is None else eta
        key = (symbol, tf)
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
        _ = cerc_before

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
