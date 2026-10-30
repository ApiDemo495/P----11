"""Section 6 — The 22 Physics Formulas (Market Quantum State Framework).

The market is modeled as a quantum system: the current price is a
superposition of possible future price levels with complex probability
amplitudes; prediction collapses the superposition.  P1..P21 implement the
individual physics formulas; P22 (UFP) unifies them into one [-1, +1]
signal that enters the CCI as component 99, weighted by base * TME (P9).

Documented interpretations where the spec is internally ambiguous or
underspecified (README "Part-3 semantics notes"):

  * P1  Confidence_quantum = max|c|^2 / (1/N) implemented exactly as the
        formula states (range [1, N]); the prose sentence describing the
        un-scaled max|c|^2 cannot hold simultaneously.
  * P1  LVD_contribution(P_n) = clip(1 - 2 * local_volume_ratio, -1, +1)
        (vacuum -> +1 boost, half-average -> 0, wall -> -1 suppression);
        the spec fixes only the boost/suppress direction.
  * P4  QTPB with no barrier in a direction = 1 (classical path, no
        tunneling needed); the directional UFP signal is Q_up - Q_down.
  * P8  mu_chem read as V_eff(P_VWAP) (potential units) — the literal
        price-unit comparison makes f(P_n) identically 1 and FDPD ~ 0.
  * P9  hot-minute news test uses the per-minute arrival score sum
        (a single item's score caps at 0.25 < 0.5).
  * P11 tau_ref = 60 minutes (same reference minute P1 uses); beta is
        averaged over the doubling pairs available in the window.
  * P12 order parameter uses the SIGNED sum of sign(r) so the output can
        carry direction (the formula's |sum| has none, the prose needs it).
  * P14 F_normalization cancels pi^2 h_eff / 240 -> tanh(-sign / d^4).
  * P15 beta = (v_leader - v_target)/c_market clipped into (-1, 1)
        before the Doppler factor (domain guard).
  * P18 Vietoris-Rips Betti numbers via ripser at epsilon = 2 * sigma;
        sigma_beta1 == 0 -> the (b1 - mean) term is reported as 0.
  * P19 gauge-invariance compares the SAME top-5 estimator on original
        vs transformed prices (raw signals, renormalized top-5 weights).
  * P22 directional formulas enter as P x TME (P9 clause); the five
        non-directional confidence-type outputs (WFCP, DRPT, TME, GSPI,
        QDPC) are linearly mapped [0,1] -> [-1,1]; coupling = max(0, corr)
        with 1.0 before 50 GRCs of history exist.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from typing import Any, Callable

import numpy as np

from .config import (
    CONFIDENCE_FLOOR,
    LEARNING_RATE_ETA,
    PHYSICS_ALPHA_COUNT,
    PHYSICS_COUPLING_WINDOW,
    PHYSICS_NAMES,
    PHYSICS_STATE_FILE,
    PIPP_MAX_STEPS,
    PIPP_PATHS,
    PROFILE_BINS,
    QDPC_K_B,
    QTPB_INTEGRAL_POINTS,
    QUANTUM_LEVELS,
    SSBD_NEIGHBORS,
    SYMBOL_ORDER,
    TAU_REF_MINUTES,
    TME_COLD_VOLUME_MULT,
    TME_HOT_VOLUME_MULT,
    TME_NEWS_GATE,
    TME_NEWS_PRICE_GATE,
    C_MARKET_LEAD_WINDOW,
    C_MARKET_WINDOW,
    HIST_T_BASELINE,
    HIST_TPTI,
    TF_ORDER,
    UFP_ALPHA_IDS,
    WEIGHT_MAX,
    WEIGHT_MIN,
)
from .formulas._helpers import sign as _sign
from .formulas.__init__ import compute_all
from .models import FormulaContext

EPS = 1e-12
_LN2 = math.log(2.0)

# non-directional confidence-type outputs mapped into UFP as 2x - 1
_NON_DIRECTIONAL = {2, 6, 9, 19, 21}

try:                                    # ripser powers P18's persistent homology
    from ripser import ripser as _ripser
except Exception:                       # pragma: no cover - guarded fallback
    _ripser = None


# ===========================================================================
# shared scene quantities
# ===========================================================================
def _tanh(x: float) -> float:
    if x > 20.0:
        return 1.0
    if x < -20.0:
        return -1.0
    return math.tanh(x)


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def volume_profile(P: list[float], V: list[float],
                   bins: int = PROFILE_BINS) -> dict[str, Any]:
    """100-bin volume profile over the window price range (P13's bins; the
    same density rho feeds P1's potential, P4's barrier, P7's condensate and
    P10's action)."""
    n = min(len(P), len(V))
    if n < 5:
        return {"lo": 0.0, "hi": 1.0, "vol": [], "mid": [], "density": [],
                "avg": 0.0}
    P, V = P[-n:], V[-n:]
    lo, hi = min(P), max(P)
    if hi <= lo:
        hi = lo + max(abs(lo) * 1e-6, 1e-8)
    width = (hi - lo) / bins
    vol = [0.0] * bins
    for p, v in zip(P, V):
        idx = min(bins - 1, max(0, int((p - lo) / width)))
        vol[idx] += v
    mid = [lo + (i + 0.5) * width for i in range(bins)]
    total = sum(vol) or 1.0
    # density = probability mass per price unit (rho integrates to 1)
    density = [(v / total) / width for v in vol]
    return {
        "lo": lo, "hi": hi, "width": width, "vol": vol, "mid": mid,
        "density": density, "avg": (sum(vol) / bins),
    }


def rho_at(profile: dict[str, Any], price: float) -> float:
    """Volume density at a price level (guard: never exactly 0 for logs)."""
    bins = profile.get("vol") or []
    if not bins:
        return EPS
    lo, hi, width = profile["lo"], profile["hi"], profile["width"]
    if price < lo or price > hi:
        return EPS
    idx = min(len(bins) - 1, max(0, int((price - lo) / width)))
    return max(profile["density"][idx], EPS)


def lvd_contribution(profile: dict[str, Any], price: float,
                     atr: float) -> float:
    """P1 Liquidity Correction input: +1 near a vacuum, 0 at half-average
    local volume, -1 at/above a full wall (documented linear ramp)."""
    bins = profile.get("vol") or []
    if not bins or profile["avg"] <= 0 or atr <= 0:
        return 0.0
    # local volume within 0.25 ATR of the level
    lo, hi, width = profile["lo"], profile["hi"], profile["width"]
    span = max(width, 0.25 * atr)
    local = 0.0
    covered = 0.0
    for i, mid in enumerate(profile["mid"]):
        if abs(mid - price) <= span:
            local += bins[i]
            covered += width
    if covered <= 0:
        return 0.0
    # ratio: local density vs the profile-average density
    ratio = (local / covered) / (profile["avg"] / width) if profile["avg"] else 1.0
    return max(-1.0, min(1.0, 1.0 - 2.0 * ratio))


class Scene:
    """Everything the 21 formulas share for one (symbol, timeframe)."""

    def __init__(self, symbol: str, tf: str, ctx: FormulaContext, *,
                 atr: float, raws: dict[int, float],
                 cci_ref: float,
                 long_closes: dict[str, list[float]],
                 news_minutes: dict[int, float],
                 t_market_prev: float | None) -> None:
        self.symbol = symbol
        self.tf = tf
        self.ctx = ctx
        self.tau = float(ctx.W)
        P = list(ctx.P)
        V = list(ctx.V)
        self.P = P
        self.V = V
        self.price = P[-1] if P else 0.0
        self.atr = atr if atr and atr > 0 else EPS
        self.rets = [
            (P[i] - P[i - 1]) / P[i - 1] for i in range(1, len(P))
            if abs(P[i - 1]) > EPS
        ]
        self.sigma_r = _std(self.rets)
        self.mu_r = _mean(self.rets)
        v_avg = _mean([abs(r) for r in self.rets])
        self.v_avg_speed = max(v_avg, EPS)
        self.m_series = []
        vmean = _mean(V) if V else 0.0
        for v in V:
            self.m_series.append((v / vmean) if vmean > 0 else 1.0)
        self.m_eff = self.m_series[-1] if self.m_series else 1.0
        self.sigma_q = self.atr * math.sqrt(max(self.tau, 1.0) / 60.0)
        self.h_eff = max(2.0 * math.pi * self.sigma_q * self.v_avg_speed, EPS)
        self.profile = volume_profile(P, V)
        self.raws = raws
        self.cci_ref = cci_ref
        self.long_closes = long_closes
        self.news_minutes = news_minutes          # minute epoch -> |sum score|
        self.T = list(ctx.T)
        # market temperature T = sigma_returns^2 * m_eff  (P7 / P12 / P17 / P21)
        self.t_market = (self.sigma_r ** 2) * self.m_eff
        self.t_market_prev = t_market_prev

    # -- volume-space helpers ---------------------------------------------
    def v_eff(self, price: float) -> float:
        """Effective potential V_eff = -ln rho(P): high volume = well."""
        return -math.log(rho_at(self.profile, price))

    def t_vwap(self) -> float:
        tot = sum(self.V)
        if tot <= 0:
            return self.price
        return sum(p * v for p, v in zip(self.P, self.V)) / tot

    def c_market(self, window: int) -> float:
        """max |1-minute return| over the last `window` minutes across the
        four assets (P15: 200 min; P16/P17: 500 min)."""
        best = 0.0
        for sym in SYMBOL_ORDER:
            cl = self.long_closes.get(sym) or []
            seg = cl[-(window + 1):]
            for i in range(1, len(seg)):
                if abs(seg[i - 1]) > EPS:
                    best = max(best, abs((seg[i] - seg[i - 1]) / seg[i - 1]))
        return max(best, EPS)


def _aligned_closes(ctx: FormulaContext) -> dict[str, list[float]]:
    """Window-aligned close series for all four assets (self + ffill peers)."""
    out: dict[str, list[float]] = {ctx.symbol: list(ctx.P)}
    tmap = list(ctx.T)
    for sym, (times, prices) in (ctx.others or {}).items():
        if sym in out:
            continue
        pmap = dict(zip(times, prices))
        series: list[float] = []
        last = None
        for ts in tmap:
            if ts in pmap:
                last = pmap[ts]
            if last is None:
                prior = [pmap[t] for t in times if t <= ts]
                last = prior[-1] if prior else None
            series.append(last if last is not None else 0.0)
        out[sym] = series
    return out


def _pearson(x: list[float], y: list[float]) -> float:
    n = min(len(x), len(y))
    if n < 3:
        return 0.0
    x, y = x[-n:], y[-n:]
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx <= EPS or dy <= EPS:
        return 0.0
    return max(-1.0, min(1.0, num / (dx * dy)))


# ===========================================================================
# P1 — Price State Superposition Amplitude (PSSA)
# ===========================================================================
def p01_pssa(sc: Scene) -> tuple[float, dict]:
    N = QUANTUM_LEVELS                        # 21, odd, centered on price
    dP = sc.atr / N
    sig = max(sc.sigma_q, EPS)
    n_center = (N + 1) / 2.0
    levels = [sc.price + (n - n_center) * dP for n in range(1, N + 1)]

    amps: list[float] = []
    for Pn in levels:
        env = math.exp(-((Pn - sc.price) ** 2) / (2.0 * sig * sig))
        kappa = 0.5 * lvd_contribution(sc.profile, Pn, sc.atr)
        amps.append(env * (1.0 + kappa))
    norm = math.sqrt(sum(a * a for a in amps)) or EPS

    # phase: phi_n = (2 pi / h_eff) * integral of the path Lagrangian
    phase: list[float] = []
    v_eff = [-math.log(rho_at(sc.profile, Pn)) for Pn in levels]
    for idx, Pn in enumerate(levels):
        lag = 0.0
        for i in range(1, len(sc.P)):
            v = (sc.P[i] - sc.P[i - 1]) / sc.P[i - 1] if sc.P[i - 1] else 0.0
            m = sc.m_series[i] if i < len(sc.m_series) else 1.0
            lag += 0.5 * m * v * v - v_eff[idx]
        phase.append((2.0 * math.pi / sc.h_eff) * lag)

    probs = [(a / norm) ** 2 for a in amps]            # |c_n|^2, sums to 1
    p_pred = sum(p * Pn for p, Pn in zip(probs, levels))
    conf_q = max(probs) / (1.0 / QUANTUM_LEVELS)       # exact spec formula
    value = _tanh((p_pred - sc.price) / sc.atr * conf_q)
    detail = {
        "levels": [round(x, 4) for x in levels],
        "probs": [round(x, 6) for x in probs],
        "p_predicted": round(p_pred, 4),
        "confidence_q": round(conf_q, 4),
        "sigma_q": round(sc.sigma_q, 4),
        "phases": [round(x % (2 * math.pi), 4) for x in phase],
    }
    return value, detail


# ===========================================================================
# P2 — Wave Function Collapse Probability (WFCP)
# ===========================================================================
def p02_wfcp(sc: Scene) -> tuple[float, dict]:
    N = QUANTUM_LEVELS
    dP = max(sc.atr / N, EPS)
    n_center = (N + 1) / 2.0
    levels = [sc.price + (n - n_center) * dP for n in range(1, N + 1)]
    v_diag = [sc.v_eff(Pn) for Pn in levels]
    # kinetic coupling: T = -h^2/(2 m dP^2) * discrete Laplacian
    a = (sc.h_eff ** 2) / (2.0 * max(sc.m_eff, EPS) * dP * dP)
    H = np.zeros((N, N), dtype=float)
    for n in range(N):
        H[n, n] = v_diag[n] + 2.0 * a
        if n > 0:
            H[n, n - 1] = -a
        if n < N - 1:
            H[n, n + 1] = -a
    # amplitudes from P1 (real envelope is what matters for <Psi|H|Psi>)
    sig = max(sc.sigma_q, EPS)
    amps = []
    for Pn in levels:
        env = math.exp(-((Pn - sc.price) ** 2) / (2.0 * sig * sig))
        kappa = 0.5 * lvd_contribution(sc.profile, Pn, sc.atr)
        amps.append(env * (1.0 + kappa))
    c = np.array(amps, dtype=float)
    nrm = np.linalg.norm(c) or EPS
    c = c / nrm
    e_state = float(c @ H @ c)
    e_ground = float(np.linalg.eigvalsh(H).min())
    gamma = abs(e_state - e_ground) / sc.h_eff
    wfcp = 1.0 - math.exp(-gamma * sc.tau)
    wfcp = max(0.0, min(1.0, wfcp))
    return wfcp, {"gamma": round(gamma, 6), "energy": round(e_state, 6),
                  "ground": round(e_ground, 6), "gap": round(abs(e_state - e_ground), 6)}


# ===========================================================================
# P3 — Heisenberg Price-Volatility Uncertainty (HPVU)
# ===========================================================================
def p03_hpvu(sc: Scene) -> float:
    delta_p = sc.sigma_q
    # rolling 5-minute volatility estimates over the window
    vols5: list[float] = []
    r = sc.rets
    for i in range(4, len(r)):
        vols5.append(_std(r[i - 4:i + 1]))
    delta_sigma = _std(vols5) if len(vols5) >= 2 else 0.0
    bound = sc.h_eff / 2.0
    excess = (delta_p * delta_sigma - bound) / max(bound, EPS)
    direction = 1 if sc.mu_r > 0 else (-1 if sc.mu_r < 0 else 0)
    return _tanh(excess * direction)


# ===========================================================================
# P4 — Quantum Tunneling Probability Through Barriers (QTPB)
# ===========================================================================
def p04_qtpb(sc: Scene) -> tuple[float, dict]:
    from .formulas.f22_f31 import fractal_boundary_levels

    m = max(sc.m_eff, EPS)
    v_cur = sc.rets[-1] if sc.rets else 0.0
    e_current = 0.5 * m * v_cur * v_cur + sc.v_eff(sc.price)

    def tunnel(barrier: float | None) -> float:
        if barrier is None:
            return 1.0                       # no barrier: classical breakout
        dist = barrier - sc.price
        if abs(dist) <= EPS:
            return 1.0
        steps = QTPB_INTEGRAL_POINTS
        integral = 0.0
        forbidden_max = 0.0
        for k in range(1, steps + 1):
            p_mid = sc.price + dist * (k - 0.5) / steps
            v_p = sc.v_eff(p_mid)
            if v_p > e_current:
                forbidden_max = max(forbidden_max, v_p - e_current)
                integral += math.sqrt(2.0 * m * (v_p - e_current)) / sc.h_eff
        if forbidden_max <= 0.0:
            return 1.0                       # classically surmountable
        # 20-point Riemann sum * segment width
        integral *= abs(dist) / steps
        return max(0.0, min(1.0, math.exp(-2.0 * integral)))

    # F24 fractal boundaries: (below, above), nearest-first
    down_levels, up_levels = fractal_boundary_levels(sc.ctx)
    up = up_levels[0] if up_levels else None
    down = down_levels[0] if down_levels else None
    q_up = tunnel(up)
    q_down = tunnel(down)
    value = q_up - q_down
    return value, {"q_up": round(q_up, 4), "q_down": round(q_down, 4),
                   "barrier_up": up, "barrier_down": down,
                   "energy": round(e_current, 6)}


# ===========================================================================
# P5 — Market Entanglement Entropy (MEE)
# ===========================================================================
def p05_mee(sc: Scene) -> tuple[float, dict]:
    aligned = _aligned_closes(sc.ctx)
    order = [s for s in SYMBOL_ORDER if s in aligned]
    if len(order) < 2:
        return 0.0, {"entropy": 0.0, "reduced": 0.5}
    rets: dict[str, list[float]] = {}
    for s in order:
        cl = aligned[s]
        rets[s] = [
            (cl[i] - cl[i - 1]) / cl[i - 1]
            for i in range(1, len(cl)) if abs(cl[i - 1]) > EPS
        ]
    k = min(len(v) for v in rets.values())
    if k < 5:
        return 0.0, {"entropy": 0.0, "reduced": 0.5}
    corr = np.zeros((len(order), len(order)))
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            corr[i, j] = 1.0 if i == j else _pearson(rets[a][-k:], rets[b][-k:])
    denom = float(np.abs(corr).sum()) or EPS
    rho = corr / denom
    tr = float(sum(rho[i, i] for i in range(len(order))))
    ti = order.index(sc.symbol)
    q = rho[ti, ti] / tr if tr > EPS else 0.0
    q = max(0.0, min(1.0, q))

    def h(p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)

    entropy = h(q)                            # range [0, ln 2]
    # entanglement-weighted cross-asset prediction
    summ = 0.0
    for j, s in enumerate(order):
        if s == sc.symbol:
            continue
        r_now = rets[s][-1] if rets[s] else 0.0
        summ += rho[ti, j] * (1 if r_now > 0 else (-1 if r_now < 0 else 0)) * entropy / _LN2
    signal = _tanh(summ)
    return signal, {"entropy": round(entropy, 6), "reduced": round(q, 6),
                    "ln2": round(_LN2, 6)}


# ===========================================================================
# P6 — Decoherence Rate of Price Trends (DRPT)
# ===========================================================================
def p06_drpt(sc: Scene, msrc: float, cthr: float) -> float:
    """DRPT = 1 − exp(−γ·τ) with γ = (1/τ_deco)·(1−msrc)·(1−cthr)/2.

    Input semantics (documented decision, delegated by the user):
      msrc — market coherence LEVEL in [0,1]; the engine passes |F11 MSRC|
             (a coherent bearish trend is just as coherent, so the signed
             value would max out DRPT on strong down-trends).
      cthr — the confidence threshold 0.35 (CONFIDENCE_FLOOR); raw signed
             DVDI made (1−cthr) ≈ 1.76 and pinned DRPT near 1, so
             predictions could never clear the issuance floor. The formula
             itself is unchanged.
    """
    tau_deco = max(sc.tau / 5.0, EPS)
    gamma = (1.0 / tau_deco) * (1.0 - msrc) * (1.0 - cthr) / 2.0
    return max(0.0, min(1.0, 1.0 - math.exp(-gamma * sc.tau)))


# ===========================================================================
# P7 — Bose-Einstein Order Flow Condensation (BEOFC)
# ===========================================================================
def p07_beofc(sc: Scene) -> tuple[float, dict]:
    n_total = sum(sc.V)
    if n_total <= 0 or sc.sigma_r <= 0 or sc.m_eff <= 0:
        return 0.0, {"t_ratio": 1.0, "condensate": sc.price}
    t_market = (sc.sigma_r ** 2) * sc.m_eff
    price_span = max(sc.profile["hi"] - sc.profile["lo"], EPS)
    vol_span = max(max(sc.V) - min(sc.V), EPS)
    v_phase = price_span * vol_span
    t_critical = ((2.0 * math.pi * sc.h_eff ** 2) / sc.m_eff) * \
        ((n_total / v_phase) ** (2.0 / 3.0))
    if t_critical <= 0:
        return 0.0, {"t_ratio": 1.0, "condensate": sc.price}
    t_ratio = t_market / t_critical
    n0_frac = max(0.0, 1.0 - (t_ratio ** 1.5)) if t_ratio >= 0 else 1.0
    # condensate = highest-volume density price level
    cond = max(zip(sc.profile["mid"], sc.profile["vol"]),
               key=lambda kv: kv[1])[0] if sc.profile["vol"] else sc.price
    value = _tanh((n0_frac - t_ratio) * (1 if sc.price > cond else -1))
    return value, {"t_ratio": round(t_ratio, 6), "n0_frac": round(n0_frac, 6),
                   "condensate": round(cond, 4)}


# ===========================================================================
# P8 — Fermi-Dirac Participant Distribution (FDPD)
# ===========================================================================
def p08_fdpd(sc: Scene) -> float:
    N = QUANTUM_LEVELS
    dP = sc.atr / N
    n_center = (N + 1) / 2.0
    t_market = max((sc.sigma_r ** 2) * sc.m_eff, EPS)
    # mu_chem in potential units: V_eff(P_VWAP) (see module docstring)
    mu_chem = sc.v_eff(sc.t_vwap())
    f_above = f_below = 0.0
    for n in range(1, N + 1):
        Pn = sc.price + (n - n_center) * dP
        v_eff = sc.v_eff(Pn)
        x = (v_eff - mu_chem) / t_market
        x = max(-500.0, min(500.0, x))
        f = 1.0 / (math.exp(x) + 1.0)
        if Pn > sc.price:
            f_above += f * (Pn - sc.price)
        elif Pn < sc.price:
            f_below += f * (sc.price - Pn)
    return _tanh((f_above - f_below) / (f_above + f_below + EPS))


# ===========================================================================
# P9 — Thermodynamic Market Efficiency (TME)
# ===========================================================================
def p09_tme(sc: Scene) -> tuple[float, dict]:
    n = len(sc.P)
    if n < 6 or sc.sigma_r <= 0:
        return 0.75, {"hot": 0.0, "cold": 0.0}     # defaults -> 1 - 0.5/2
    avg_v = _mean(sc.V) if sc.V else 0.0
    rets = sc.rets
    hot: list[float] = []
    cold: list[float] = []
    for i in range(n):
        minute = sc.T[i] if i < len(sc.T) else 0
        news_here = abs(sc.news_minutes.get(minute, 0.0)) > TME_NEWS_GATE
        vol_i = sc.V[i] if i < len(sc.V) else 0.0
        r_i = rets[i - 1] if 0 < i <= len(rets) else 0.0
        if news_here or (avg_v > 0 and vol_i > TME_HOT_VOLUME_MULT * avg_v):
            hot.append(r_i)
        elif avg_v > 0 and vol_i < TME_COLD_VOLUME_MULT * avg_v and not news_here:
            cold.append(r_i)
    t_hot = _std(hot) if len(hot) >= 2 else 2.0 * sc.sigma_r
    t_cold = _std(cold) if len(cold) >= 2 else 0.5 * sc.sigma_r
    if t_hot <= EPS:
        t_hot = 2.0 * sc.sigma_r
    tme = 1.0 - t_cold / t_hot
    return max(0.0, min(1.0, tme)), {
        "hot": round(t_hot, 8), "cold": round(t_cold, 8),
        "hot_n": len(hot), "cold_n": len(cold),
    }


# ===========================================================================
# P10 — Path Integral Price Probability (PIPP)
# ===========================================================================
def p10_pipp(sc: Scene, atr: float) -> float:
    tau = int(sc.tau)
    M = max(2, min(PIPP_MAX_STEPS, tau))
    dt = tau / M
    seed_src = f"{sc.symbol}|{sc.tf}|{round(sc.price, 6)}|{sc.T[-1] if sc.T else 0}"
    seed = int(hashlib.sha256(seed_src.encode()).hexdigest()[:12], 16)
    rng = random.Random(seed)
    mu, sig = sc.mu_r, max(sc.sigma_r, EPS)
    m = max(sc.m_eff, EPS)
    h = sc.h_eff
    dP = max(atr / QUANTUM_LEVELS, EPS)
    two_pi = 2.0 * math.pi

    up_target = sc.price + atr
    down_target = sc.price - atr
    num_up = 0j
    num_down = 0j
    den = 0j
    for _ in range(PIPP_PATHS):
        p = sc.price
        action = 0.0
        for _ in range(M):
            p_next = p + mu * dt + sig * math.sqrt(dt) * rng.gauss(0.0, 1.0)
            v = (p_next - p) / dt
            action += (0.5 * m * v * v - sc.v_eff(p)) * dt
            p = p_next
        phase = (action / h) % two_pi
        amp = complex(math.cos(phase), math.sin(phase))
        den += amp
        if abs(p - up_target) < dP:
            num_up += amp
        if abs(p - down_target) < dP:
            num_down += amp
    den_val = abs(den) ** 2
    if den_val <= EPS:
        return 0.0
    p_up = abs(num_up) ** 2 / den_val
    p_down = abs(num_down) ** 2 / den_val
    return _tanh((p_up - p_down) * 10.0)


# ===========================================================================
# P11 — Renormalization Group Volatility Flow (RGVF)
# ===========================================================================
def realized_vol(P: list[float], tau: int) -> float | None:
    if tau < 1 or len(P) <= tau + 1:
        return None
    rs = [(P[i] - P[i - tau]) / P[i - tau]
          for i in range(tau, len(P)) if abs(P[i - tau]) > EPS]
    if len(rs) < 3:
        return None
    return _std(rs)


def p11_rgvf(sc: Scene) -> tuple[float, dict]:
    scales = [1, 5, 22, 130]
    sigmas: dict[int, float] = {}
    for tau in scales:
        v = realized_vol(sc.P, tau)
        if v is not None and v > 0:
            sigmas[tau] = v
    betas: list[float] = []
    for tau, s in sigmas.items():
        s2 = realized_vol(sc.P, 2 * tau)
        if s2 is not None and s2 > 0:
            betas.append((s2 - s) / (s * math.log(2.0)))
    beta = _mean(betas) if betas else 0.0
    if len(sigmas) >= 2:
        taus = sorted(sigmas)
        xs = [math.log(t) for t in taus]
        ys = [sigmas[t] for t in taus]
        mx, my = _mean(xs), _mean(ys)
        den = sum((x - mx) ** 2 for x in xs)
        slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
                 if den > EPS else 0.0)
        sigma_trend = 1 if slope > 0 else (-1 if slope < 0 else 0)
    else:
        sigma_trend = 0
    value = _tanh(beta * math.log(sc.tau / TAU_REF_MINUTES) * sigma_trend)
    return value, {"beta": round(beta, 6), "sigma_trend": sigma_trend,
                   "scales": sorted(sigmas)}


# ===========================================================================
# P12 — Spontaneous Symmetry Breaking Detector (SSBD_physics)
# ===========================================================================
def p12_ssbd(sc: Scene) -> float:
    r = sc.rets
    if len(r) < 10 or sc.sigma_r <= 0 or sc.m_eff <= 0:
        return 0.0
    # lag-1 coupling J = |corr(r(t), r(t-1))|
    j = abs(_pearson(r[:-1], r[1:]))
    t_market = (sc.sigma_r ** 2) * sc.m_eff
    t_crit = j * SSBD_NEIGHBORS / math.log(2.0)
    if t_crit <= EPS:
        return 0.0
    order_param = sum(1 if x > 0 else (-1 if x < 0 else 0) for x in r) / len(r)
    return _tanh((t_market - t_crit) / t_crit * order_param)


# ===========================================================================
# P13 — Gravitational Lensing of Price by Large Orders (GLPLO)
# ===========================================================================
def p13_glplo(sc: Scene) -> tuple[float, dict]:
    prof = sc.profile
    if not prof["vol"] or prof["avg"] <= 0:
        return 0.0, {"massive": 0}
    forces: list[tuple[float, float]] = []     # (force, sign)
    massive = 0
    for mid, v in zip(prof["mid"], prof["vol"]):
        if v <= 3.0 * prof["avg"]:
            continue
        massive += 1
        m_j = v / prof["avg"]
        d_j = max(abs(mid - sc.price) / sc.atr, 1e-6)
        forces.append((m_j / (d_j * d_j),
                       1 if mid > sc.price else (-1 if mid < sc.price else 0)))
    g_norm = sum(f for f, _ in forces)
    if g_norm <= EPS:
        return 0.0, {"massive": massive}
    value = _tanh(sum(f * s for f, s in forces) / g_norm)
    return value, {"massive": massive, "g_norm": round(g_norm, 4)}


# ===========================================================================
# P14 — Casimir Effect of Liquidity Gaps (CELG)
# ===========================================================================
def p14_celg(sc: Scene) -> tuple[float, dict]:
    prof = sc.profile
    vols = prof.get("vol") or []
    if not vols or prof["avg"] <= 0:
        return 0.0, {"gaps": 0}
    width = prof["width"]
    # gaps: consecutive bins below 20% of average volume
    gaps: list[dict] = []
    i = 0
    while i < len(vols):
        if vols[i] < 0.20 * prof["avg"]:
            j = i
            while j < len(vols) and vols[j] < 0.20 * prof["avg"]:
                j += 1
            w_price = (j - i) * width
            if w_price >= 0.5 * sc.atr:
                lo = prof["lo"] + i * width
                gaps.append({"center": lo + w_price / 2.0, "width": w_price})
            i = j
        else:
            i += 1
    if len(gaps) < 2:
        return 0.0, {"gaps": len(gaps)}
    # closest pair within 2 ATR of each other
    best: tuple[float, float, float] | None = None     # dist, c1, c2
    for a in range(len(gaps)):
        for b in range(a + 1, len(gaps)):
            d = abs(gaps[a]["center"] - gaps[b]["center"])
            if d <= 2.0 * sc.atr and (best is None or d < best[0]):
                best = (d, gaps[a]["center"], gaps[b]["center"])
    if best is None:
        return 0.0, {"gaps": len(gaps)}
    d_gap = max(best[0] / sc.atr, EPS)
    mid_gap = (best[1] + best[2]) / 2.0
    # F_normalization = pi^2 h_eff / 240 cancels the force constant
    if sc.h_eff <= EPS:
        return 0.0, {"gaps": len(gaps)}
    force = -math.pi ** 2 * sc.h_eff / (240.0 * d_gap ** 4)
    f_norm = math.pi ** 2 * sc.h_eff / 240.0
    direction = 1 if mid_gap > sc.price else (-1 if mid_gap < sc.price else 0)
    value = _tanh(force / f_norm * direction)
    return value, {"gaps": len(gaps), "d_gap": round(d_gap, 4),
                   "mid_gap": round(mid_gap, 4)}


# ===========================================================================
# P15 — Doppler Shift of Momentum Signals (DSMS)
# ===========================================================================
def _pmtl_lead_scores(long_closes: dict[str, list[float]],
                      window: int) -> str:
    """PMTL (Formula 19) lead metric: for each asset, the best cross-
    correlation of its current returns against the other assets' returns
    at positive lags (others confirm later -> this asset leads)."""
    best_sym = next(iter(long_closes), SYMBOL_ORDER[0])
    best_score = -1e18
    for sym, cl in long_closes.items():
        seg = cl[-(window + 1):]
        if len(seg) < 8:
            continue
        r_self = [(seg[i] - seg[i - 1]) / seg[i - 1]
                  for i in range(1, len(seg)) if abs(seg[i - 1]) > EPS]
        others = [o for o in SYMBOL_ORDER if o != sym and o in long_closes]
        comp: list[float] = []
        for i in range(1, len(seg)):
            vals = []
            for o in others:
                oseg = long_closes[o][-(window + 1):]
                if i < len(oseg) and abs(oseg[i - 1]) > EPS:
                    vals.append((oseg[i] - oseg[i - 1]) / oseg[i - 1])
            comp.append(sum(vals) / len(vals) if vals else 0.0)
        n = min(len(r_self), len(comp))
        if n < 6:
            continue
        rs, rc = r_self[-n:], comp[-n:]
        best_ccf = 0.0
        for lag in range(1, min(max(1, window // 4), n - 1) + 1):
            num = sum(rc[i + lag - 1] * rs[i] for i in range(n - lag))
            den = math.sqrt(sum(v * v for v in rs[:n - lag]) *
                            sum(v * v for v in rc[:n - lag]))
            if den > EPS:
                best_ccf = max(best_ccf, num / den)
        score = best_ccf
        if score > best_score:
            best_score = score
            best_sym = sym
    return best_sym


def p15_dsms(sc: Scene, atr: float) -> tuple[float, dict]:
    if len(sc.rets) < 5:
        return 0.0, {"blueshift": False}
    back = max(1, sc.ctx.W // 4)
    p_back = sc.P[-(back + 1)] if len(sc.P) > back else sc.P[0]
    momentum_raw = ((sc.price - p_back) /
                    (max(back, 1) * max(sc.sigma_r, EPS)))
    c_market = sc.c_market(C_MARKET_LEAD_WINDOW)
    leader = _pmtl_lead_scores(sc.long_closes, C_MARKET_LEAD_WINDOW)

    def last_ret(sym: str) -> float:
        cl = sc.long_closes.get(sym) or []
        if len(cl) < 2 or abs(cl[-2]) <= EPS:
            return 0.0
        return (cl[-1] - cl[-2]) / cl[-2]

    v_leader = last_ret(leader)
    v_target = last_ret(sc.symbol)
    beta = (v_leader - v_target) / c_market
    beta = max(-0.999, min(0.999, beta))
    doppler = math.sqrt((1.0 + beta) / (1.0 - beta))
    value = _tanh(momentum_raw * doppler)
    return value, {"momentum_raw": round(momentum_raw, 4),
                   "leader": leader, "beta": round(beta, 6),
                   "doppler": round(doppler, 4),
                   "blueshift": beta > 0}


# ===========================================================================
# P16 — Relativistic Time Dilation of Market Perception (RTDMP)
# ===========================================================================
def p16_rtdmp(sc: Scene) -> tuple[float, dict]:
    v_market = abs(sc.rets[-1]) if sc.rets else 0.0
    c_market = sc.c_market(C_MARKET_WINDOW)
    ratio = min(v_market / c_market, 1.0 - 1e-12)
    gamma = 1.0 / math.sqrt(max(1.0 - ratio * ratio, 1e-12))
    tau_eff = sc.tau / gamma
    direction = 1 if sc.cci_ref > 0 else (-1 if sc.cci_ref < 0 else 0)
    value = _tanh((tau_eff / sc.tau - 0.5) * 2.0 * direction)
    return value, {"gamma": round(gamma, 6), "tau_eff": round(tau_eff, 4),
                   "v_market": v_market, "c_market": c_market}


# ===========================================================================
# P17 — Black Body Information Radiation (BBIR)
# ===========================================================================
def p17_bbir(sc: Scene, t_baseline: float) -> tuple[float, dict]:
    nu_high = 1.0 / 5.0
    nu_low = 1.0 / 130.0
    t = max(sc.t_market, EPS)

    def log_i(nu: float) -> float:
        # ln(2 h nu^3 / c^2) - ln(expm1(h nu / T)); the prefactor cancels
        # in the I_high/I_low ratio, the exponential part does not.
        x = sc.h_eff * nu / (QDPC_K_B * t)
        if x > 700.0:
            log_den = x                      # ln(expm1(x)) ~ x for large x
        else:
            log_den = math.log(math.expm1(x) if x > 1e-12 else max(x, EPS))
        return 3.0 * math.log(nu) - log_den

    delta = log_i(nu_high) - log_i(nu_low)
    ratio = math.tanh(delta / 2.0)           # (I_h - I_l) / (I_h + I_l)
    if t_baseline is None or t_baseline <= 0:
        return 0.0, {"t_market": t, "ratio": round(ratio, 6)}
    hot = 1 if sc.t_market > t_baseline else (-1 if sc.t_market < t_baseline else 0)
    value = _tanh(ratio * hot)
    return value, {"t_market": t, "t_baseline": t_baseline,
                   "ratio": round(ratio, 6), "hot": hot}


# ===========================================================================
# P18 — Topological Phase Transition Index (TPTI)
# ===========================================================================
def betti_numbers(P: list[float]) -> tuple[int, int]:
    """(beta0, beta1) of the return-pair point cloud under the Vietoris-
    Rips filtration at epsilon = 2 sigma (handled by ripser)."""
    if _ripser is None or len(P) < 6:        # pragma: no cover - fallback
        return 1, 0
    rets = [(P[i] - P[i - 1]) / P[i - 1] for i in range(1, len(P))
            if abs(P[i - 1]) > EPS]
    if len(rets) < 6:
        return 1, 0
    pts = np.array([[rets[i], rets[i - 1]] for i in range(1, len(rets))])
    sigma = float(np.std(rets))
    eps = 2.0 * sigma
    if eps <= 1e-12:
        return 1, 0
    res = _ripser(pts, thresh=eps, maxdim=1)
    dgm0, dgm1 = res["dgms"][0], res["dgms"][1]

    def covering(dgm: np.ndarray) -> int:
        if len(dgm) == 0:
            return 0
        return int(np.sum((dgm[:, 0] <= eps) &
                          ((np.isinf(dgm[:, 1])) | (dgm[:, 1] > eps))))

    return covering(dgm0), covering(dgm1)


def p18_tpti(sc: Scene, beta1_ring: list[float],
             beta0_prev: float | None) -> tuple[float, dict, tuple[int, int]]:
    beta0, beta1 = betti_numbers(sc.P)
    if beta0_prev is None:
        delta_b0 = 0.0
    else:
        delta_b0 = beta0 - beta0_prev
    b1s = list(beta1_ring)
    if len(b1s) >= 5:
        mean_b1 = _mean(b1s)
        sigma_b1 = _std(b1s)
        if sigma_b1 <= EPS:
            z = 0.0                          # degenerate history: documented guard
        else:
            z = (beta1 - mean_b1) / sigma_b1
    else:
        z = 0.0
    direction = 1 if delta_b0 > 0 else (-1 if delta_b0 < 0 else 0)
    value = _tanh(z * direction)
    return value, {"beta0": beta0, "beta1": beta1,
                   "delta_beta0": delta_b0, "z": round(z, 4)}, (beta0, beta1)


# ===========================================================================
# P19 — Gauge Symmetry Price Invariance (GSPI)
# ===========================================================================
def simplified_cci(raws: dict[int, float],
                   weights: dict[int, float], top5: list[int]) -> float:
    """Top-5 weighted simplified CCI over RAW signals (same estimator on
    both sides of every gauge comparison)."""
    wsum = sum(weights.get(f, 0.0) for f in top5)
    if wsum <= EPS:
        return 0.0
    return sum(weights.get(f, 0.0) * raws.get(f, 0.0) for f in top5) / wsum


def gauge_transform(ctx: FormulaContext, mode: str,
                    numeraire: list[float] | None) -> FormulaContext | None:
    """Gauge 1 shift / 2 scale / 3 numeraire on the price series (and OHLC)."""
    if mode == "numeraire":
        if not numeraire or len(numeraire) != len(ctx.P):
            return None
        div = [x if abs(x) > EPS else 1.0 for x in numeraire]
        P = [p / d for p, d in zip(ctx.P, div)]
        O = [o / d for o, d in zip(ctx.O, div)] if ctx.O else []
        H = [h / d for h, d in zip(ctx.H, div)] if ctx.H else []
        L = [l / d for l, d in zip(ctx.L, div)] if ctx.L else []
    elif mode == "shift":
        base = ctx.P[0] if ctx.P else 0.0
        P = [p - base for p in ctx.P]
        O = [o - base for o in ctx.O] if ctx.O else []
        H = [h - base for h in ctx.H] if ctx.H else []
        L = [l - base for l in ctx.L] if ctx.L else []
    elif mode == "scale":
        base = ctx.P[-1] if ctx.P and ctx.P[-1] else 1.0
        P = [p / base for p in ctx.P]
        O = [o / base for o in ctx.O] if ctx.O else []
        H = [h / base for h in ctx.H] if ctx.H else []
        L = [l / base for l in ctx.L] if ctx.L else []
    else:
        return None
    return FormulaContext(
        symbol=ctx.symbol, tf=ctx.tf, W=ctx.W, P=P, V=list(ctx.V),
        T=list(ctx.T), hist=ctx.hist, others=ctx.others,
        intermediates={}, O=O, H=H, L=L,
    )


def p19_gspi(sc: Scene, raws: dict[int, float],
             weights: dict[int, float],
             recompute: Callable[[FormulaContext], dict[int, float]]) -> tuple[float, dict]:
    top5 = sorted(weights, key=lambda f: -weights[f])[:5]
    original = simplified_cci(raws, weights, top5)
    # numeraire series: BTC -> ETH, everyone else -> BTC
    numeraire = None
    aligned = _aligned_closes(sc.ctx)
    num_sym = "ETH" if sc.symbol == "BTC" else "BTC"
    numeraire = aligned.get(num_sym)

    diffs: list[float] = []
    for mode in ("shift", "scale", "numeraire"):
        gctx = gauge_transform(sc.ctx, mode, numeraire)
        if gctx is None:
            diffs.append(0.0)
            continue
        try:
            graw = recompute(gctx)
        except Exception:
            graw = {f: 0.0 for f in raws}
        gauge = simplified_cci(graw, weights, top5)
        diffs.append(abs(original - gauge) / 2.0)
    gspi = max(0.0, min(1.0, sum(diffs) / 3.0))
    return gspi, {"top5": top5, "diffs": [round(d, 6) for d in diffs],
                  "original": round(original, 6)}


# ===========================================================================
# P20 — Phonon Dispersion of Order Book Vibrations (PDOBV)
# ===========================================================================
def p20_pdobv(sc: Scene) -> tuple[float, dict]:
    r = sc.rets
    n = len(r)
    if n < 12:
        return 0.0, {"acoustic": 0.0, "optical": 0.0}
    x = np.array(r) - np.mean(r)
    spec = np.abs(np.fft.rfft(x)) ** 2
    k_max = len(spec) - 1
    if k_max < 4:
        return 0.0, {"acoustic": 0.0, "optical": 0.0}
    k_cutoff = max(1, n // 6)
    k_cutoff = min(k_cutoff, k_max - 1)
    # acoustic branch: k-weighted mean frequency of the low-k modes
    ks = np.arange(1, k_cutoff + 1)
    p_low = spec[1:k_cutoff + 1]
    denom = float(p_low.sum())
    omega_ac = float((p_low * ks).sum() / denom) if denom > EPS else 0.0
    # optical branch: mean power of the high-k modes
    p_high = spec[k_cutoff + 1:k_max + 1]
    count = len(p_high)
    omega_op = float(p_high.sum() / count) if count > 0 else 0.0
    k_peak = int(np.argmax(spec[1:k_max + 1])) + 1
    if 1 <= k_peak <= k_max - 1:
        v_group = float(spec[k_peak + 1] - spec[k_peak - 1]) / 2.0
    else:
        v_group = 0.0
    value = _tanh((omega_ac - omega_op) / (omega_ac + omega_op + EPS) *
                  (1 if v_group > 0 else (-1 if v_group < 0 else 0)))
    return value, {"acoustic": round(omega_ac, 4),
                   "optical": round(omega_op, 6),
                   "v_group": round(v_group, 6), "k_peak": k_peak}


# ===========================================================================
# P21 — Quantum Decoherence of Prediction Confidence (QDPC)
# ===========================================================================
def qdpc_tau(h_eff: float, t_market: float, gamma_env: float) -> float:
    """tau = h_eff / (2 k_B T gamma_env)  (k_B = 1)."""
    denom = 2.0 * QDPC_K_B * max(t_market, EPS) * max(gamma_env, EPS)
    return h_eff / denom


def qdpc(age_minutes: float, h_eff: float, t_market: float,
         gamma_env: float) -> float:
    """QDPC(t, age) = exp(-age / tau); age 0 -> 1."""
    if age_minutes <= 0:
        return 1.0
    tau = qdpc_tau(h_eff, t_market, gamma_env)
    if tau <= EPS:
        return 0.0
    return math.exp(-age_minutes / tau)


def p21_qdpc(sc: Scene) -> tuple[float, float]:
    """Returns (value at full horizon for UFP input, gamma_env).

    News events are counted over the LAST 30 minutes only (per spec) from
    the arrival map, gated at the 0.3 significance threshold."""
    vol_avg = _mean(sc.V) if sc.V else 0.0
    events = 0
    for i in range(1, len(sc.P)):
        r = (sc.P[i] - sc.P[i - 1]) / sc.P[i - 1] if sc.P[i - 1] else 0.0
        if abs(r) > 2.0 * sc.sigma_r and sc.sigma_r > 0:
            events += 1
        if vol_avg > 0 and i < len(sc.V) and sc.V[i] > 2.0 * vol_avg:
            events += 1
    cutoff = (sc.T[-1] - 30 * 60) if sc.T else -1
    for minute, score in sc.news_minutes.items():
        if minute >= cutoff and abs(score) > TME_NEWS_PRICE_GATE:
            events += 1
    gamma_env = events / 30.0
    horizon = qdpc(sc.tau, sc.h_eff, sc.t_market, gamma_env)
    return horizon, gamma_env


# ===========================================================================
# PhysicsSystem — P1..P21 evaluation + P22 UFP (alpha weights, coupling)
# ===========================================================================
class PhysicsSystem:
    """Owns the physics layer: alpha weights (eta rule, persisted), the
    coupling/correlation rings, TPTI/T baselines and the UFP aggregate."""

    def __init__(self) -> None:
        self.ready = False
        # alpha[(symbol, tf)][p] -> weight, init 1/21
        self.alpha: dict[tuple[str, str], dict[int, float]] = {}
        self._loaded = False

    # -- lifecycle ----------------------------------------------------------
    def initialize(self) -> None:
        self.ready = True
        self.load()

    def _weights(self, symbol: str, tf: str) -> dict[int, float]:
        key = (symbol, tf)
        w = self.alpha.get(key)
        if w is None:
            w = {p: 1.0 / PHYSICS_ALPHA_COUNT for p in UFP_ALPHA_IDS}
            self.alpha[key] = w
        return w

    def reward_update(self, symbol: str, tf: str,
                      prev_values: dict[int, float], move: float) -> None:
        """Same multiplicative rule as the CCI weights (P22): alpha *=
        (1 + eta * reward), clamp, renormalize."""
        if move == 0.0 or not prev_values:
            return
        w = self._weights(symbol, tf)
        for p in list(w.keys()):
            v_prev = prev_values.get(p, 0.0)
            reward = 1.0 if _sign(v_prev) == _sign(move) else -1.0
            w[p] = w[p] * (1.0 + LEARNING_RATE_ETA * reward)
            w[p] = max(WEIGHT_MIN, min(WEIGHT_MAX, w[p]))
        total = sum(w.values())
        if total > 0:
            for p in w:
                w[p] /= total

    # -- main evaluation ----------------------------------------------------
    def evaluate(self, symbol: str, tf: str, ctx: FormulaContext, *,
                 atr: float, raws: dict[int, float], cci_ref: float,
                 long_closes: dict[str, list[float]],
                 news_minutes: dict[int, float],
                 weights: dict[int, float],
                 recompute: Callable[[FormulaContext], dict[int, float]],
                 hist: Any,
                 record: bool,
                 move: float = 0.0,
                 ) -> dict[str, Any]:
        """Run P1..P21 for one (symbol, timeframe) and build the P22 UFP."""
        t_ring = hist.ring(hist.tf_key(symbol, tf, "t_mkt"), HIST_T_BASELINE)
        t_prev = t_ring[-1] if len(t_ring) else None
        sc = Scene(symbol, tf, ctx, atr=atr, raws=raws, cci_ref=cci_ref,
                   long_closes=long_closes, news_minutes=news_minutes,
                   t_market_prev=t_prev)

        values: dict[int, float] = {}
        details: dict[str, Any] = {}

        v, d = p01_pssa(sc); values[1] = v; details["p1"] = d
        v, d = p02_wfcp(sc); values[2] = v; details["p2"] = d
        values[3] = p03_hpvu(sc)
        v, d = p04_qtpb(sc); values[4] = v; details["p4"] = d
        v, d = p05_mee(sc); values[5] = v; details["p5"] = d
        # P6 input semantics (see p06_drpt docstring): coherence LEVEL
        # |MSRC| + the 0.35 confidence threshold as CTHR.
        values[6] = p06_drpt(sc, abs(raws.get(11, 0.0)), CONFIDENCE_FLOOR)
        v, d = p07_beofc(sc); values[7] = v; details["p7"] = d
        values[8] = p08_fdpd(sc)
        v, d = p09_tme(sc); tme = v; details["p9"] = d
        values[9] = tme
        values[10] = p10_pipp(sc, atr)
        v, d = p11_rgvf(sc); values[11] = v; details["p11"] = d
        values[12] = p12_ssbd(sc)
        v, d = p13_glplo(sc); values[13] = v; details["p13"] = d
        v, d = p14_celg(sc); values[14] = v; details["p14"] = d
        v, d = p15_dsms(sc, atr); values[15] = v; details["p15"] = d
        v, d = p16_rtdmp(sc); values[16] = v; details["p16"] = d
        v, d = p17_bbir(sc, t_prev if t_prev is not None else sc.t_market)
        values[17] = v; details["p17"] = d
        b0_ring = hist.ring(hist.tf_key(symbol, tf, "beta0"), HIST_TPTI)
        b1_ring = hist.ring(hist.tf_key(symbol, tf, "beta1"), HIST_TPTI)
        v, d, b_pair = p18_tpti(sc, list(b1_ring), b0_ring[-1] if len(b0_ring) else None)
        values[18] = v; details["p18"] = d
        v, d = p19_gspi(sc, raws, weights, recompute); values[19] = v; details["p19"] = d
        v, d = p20_pdobv(sc); values[20] = v; details["p20"] = d
        v, gamma_env = p21_qdpc(sc); values[21] = v
        details["p21"] = {"gamma_env": round(gamma_env, 6),
                          "h_eff": round(sc.h_eff, 6),
                          "tau": qdpc_tau(sc.h_eff, sc.t_market, gamma_env)}

        # --- confidence modifiers the spec assigns to physics -------------
        wfcp = values[2]
        drpt = values[6]
        gspi = values[19]

        # --- P22 UFP -------------------------------------------------------
        ufp = self.build_ufp(symbol, tf, values, tme, hist)

        out: dict[str, Any] = {
            "values": {p: round(values[p], 6) for p in values},
            "ufp": ufp,
            "tme": round(tme, 6),
            "wfcp": round(wfcp, 6),
            "drpt": round(drpt, 6),
            "gspi": round(gspi, 6),
            "details": details,
            "h_eff": sc.h_eff,
            "t_market": sc.t_market,
        }

        if record:
            # rings: coupling series, TPTI baseline, temperature baseline,
            # sparklines (100 GRCs) and accuracy rings (50 GRCs)
            for p in UFP_ALPHA_IDS:
                hist.ring(hist.tf_key(symbol, tf, f"phys{p}"),
                          PHYSICS_COUPLING_WINDOW).append(values[p])
                hist.ring(hist.tf_key(symbol, tf, f"pspark{p}"), 100).append(values[p])
                if move != 0.0:
                    hist.ring(hist.tf_key(symbol, tf, f"pacc{p}"), 50).append(
                        1.0 if _sign(values[p]) == _sign(move) else -1.0
                    )
            hist.ring(hist.tf_key(symbol, tf, "physcci"),
                      PHYSICS_COUPLING_WINDOW).append(cci_ref)
            hist.ring(hist.tf_key(symbol, tf, "pspark22"), 100).append(ufp["value"])
            hist.ring(hist.tf_key(symbol, tf, "t_mkt"),
                      HIST_T_BASELINE).append(sc.t_market)
            hist.ring(hist.tf_key(symbol, tf, "beta0"), HIST_TPTI).append(b_pair[0])
            hist.ring(hist.tf_key(symbol, tf, "beta1"), HIST_TPTI).append(b_pair[1])
        return out

    def build_ufp(self, symbol: str, tf: str, values: dict[int, float],
                  tme: float, hist: Any) -> dict:
        """UFP = sum_p alpha_p * P_p * coupling_p, P adjusted by TME for the
        directional formulas; normalized to [-1, +1] by construction."""
        alpha = self._weights(symbol, tf)
        cci_ring = list(hist.ring(hist.tf_key(symbol, tf, "physcci"),
                                  PHYSICS_COUPLING_WINDOW))
        total = 0.0
        couplings: dict[int, float] = {}
        terms: dict[int, float] = {}
        for p in UFP_ALPHA_IDS:
            raw = values.get(p, 0.0)
            if p in _NON_DIRECTIONAL:
                adj = 2.0 * raw - 1.0
            else:
                adj = raw * tme
            ring = list(hist.ring(hist.tf_key(symbol, tf, f"phys{p}"),
                                  PHYSICS_COUPLING_WINDOW))
            n = min(len(ring), len(cci_ring))
            if n >= 3:
                c = _pearson(ring[-n:], cci_ring[-n:])
                coupling = max(0.0, c)        # disagreement -> no coupling
            else:
                coupling = 1.0                # pre-history neutral
            couplings[p] = coupling
            term = alpha.get(p, 0.0) * adj * coupling
            terms[p] = term
            total += term
        value = max(-1.0, min(1.0, total))
        # Report alpha rounded to 6 decimals (spec storage precision) while
        # keeping the reported sum EXACTLY 1.0: absorb the rounding residual
        # into the largest entry.
        rep_alpha = {p: round(alpha.get(p, 0.0), 6) for p in UFP_ALPHA_IDS}
        resid = round(1.0 - round(sum(rep_alpha.values()), 6), 6)
        if resid:
            big = max(rep_alpha, key=rep_alpha.get)
            rep_alpha[big] = round(rep_alpha[big] + resid, 6)
        return {
            "value": round(value, 6),
            "alpha": rep_alpha,
            "couplings": {p: round(couplings[p], 5) for p in couplings},
            "terms": {p: round(terms[p], 6) for p in terms},
        }

    # -- persistence --------------------------------------------------------
    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(PHYSICS_STATE_FILE) or ".", exist_ok=True)
            payload = {
                "alpha": {
                    f"{s}|{tf}": {str(p): w for p, w in row.items()}
                    for (s, tf), row in self.alpha.items()
                }
            }
            with open(PHYSICS_STATE_FILE, "w") as fh:
                json.dump(payload, fh)
        except OSError:
            pass

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            with open(PHYSICS_STATE_FILE) as fh:
                payload = json.load(fh)
            for key_s, row in payload.get("alpha", {}).items():
                symbol, tf = key_s.split("|", 1)
                if tf not in TF_ORDER:
                    continue
                loaded = {int(p): float(w) for p, w in row.items()}
                if set(loaded) != set(UFP_ALPHA_IDS) or not loaded:
                    loaded = {p: 1.0 / PHYSICS_ALPHA_COUNT for p in UFP_ALPHA_IDS}
                for p in loaded:
                    loaded[p] = max(WEIGHT_MIN, min(WEIGHT_MAX, loaded[p]))
                total = sum(loaded.values())
                if total > 0:
                    for p in loaded:
                        loaded[p] /= total
                self.alpha[(symbol, tf)] = loaded
        except (OSError, ValueError, KeyError):
            pass

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "formulas": PHYSICS_ALPHA_COUNT + 1,
            "alpha_series": len(self.alpha),
            "ufp_component": "99",
        }
