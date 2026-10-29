"""Core Prediction Formulas 11-20 (exact transcription of the specification).

Formula 20 (DRHE) is intentionally NOT bounded to [-1, +1] — it outputs
T_remaining / tau_timeframe and acts as a confidence multiplier.
"""
from __future__ import annotations

import math

from ._helpers import (
    abs_changes,
    clamp,
    ema,
    linear_slope,
    mean,
    median,
    pct_returns,
    safe_tanh,
    sign,
    std_pop,
)
from ..config import HIST_SA, HIST_STD_R

EPS = 1e-12


# =============================================================================
# FORMULA 11: Multi-Scale Return Coherence (MSRC)
# =============================================================================
def f11_msrc(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 14:
        return 0.0
    P = ctx.P[-(W_eff + 1):]
    granularities = [1, 2, 5, 8, 13]
    mu: dict[int, float] = {}
    for g in granularities:
        if g >= len(P):
            return 0.0
        rs = [(P[i] - P[i - g]) / P[i - g] for i in range(g, len(P))
              if abs(P[i - g]) > EPS]
        if not rs:
            return 0.0
        mu[g] = mean(rs)

    total = 0.0
    pairs = 0
    for a_i in range(len(granularities)):
        for b_i in range(a_i + 1, len(granularities)):
            ma = mu[granularities[a_i]]
            mb = mu[granularities[b_i]]
            lo = min(abs(ma), abs(mb))
            hi = max(abs(ma), abs(mb))
            total += sign(ma) * sign(mb) * (lo / (hi + EPS))
            pairs += 1
    if pairs == 0:
        return 0.0
    return clamp(total / pairs, -1.0, 1.0)


# =============================================================================
# FORMULA 12: Recursive Mid-Price Equilibrium Tracker (RMET)
# =============================================================================
def f12_rmet(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 8:
        return 0.0
    P = ctx.P[-(W_eff + 1):]
    V = ctx.V[-(W_eff + 1):]
    W = ctx.W  # base_alpha uses the timeframe duration per spec
    base_alpha = 2.0 / (W + 1)

    # E(0): VWAP of the first W minutes of data (clamped to available samples)
    first = P[:min(len(P), W)]
    first_v = V[:len(first)]
    v_sum = sum(first_v)
    e_prev = (sum(p * v for p, v in zip(first, first_v)) / v_sum) if v_sum > EPS else first[0]

    equilibria: list[float] = [e_prev]
    for i in range(1, len(P)):
        dev = P[i] - e_prev
        # rolling average volume per minute over the window ending at i
        start = max(0, i - W + 1)
        seg_v = V[start:i + 1]
        avg_vol = mean(seg_v)
        v_up = V[i] if P[i] > P[i - 1] else 0.0
        v_dn = V[i] if P[i] < P[i - 1] else 0.0
        r_up = (v_up / avg_vol) if avg_vol > EPS else 0.0
        r_dn = (v_dn / avg_vol) if avg_vol > EPS else 0.0
        a_up = clamp(base_alpha * (1 + r_up), 0.0, 1.0)   # numeric guard (alpha <= 1)
        a_dn = clamp(base_alpha * (1 + r_dn), 0.0, 1.0)
        e_new = e_prev + a_up * max(0.0, dev) + a_dn * min(0.0, dev)
        equilibria.append(e_new)
        e_prev = e_new

    e_now = equilibria[-1]

    # asymmetric bands over the last W minutes
    pts = list(zip(P[-W:], equilibria[-W:]))
    above = [(p - e) ** 2 for p, e in pts if p > e]
    below = [(p - e) ** 2 for p, e in pts if p < e]
    sigma_above = math.sqrt(mean(above)) if above else 0.0
    sigma_below = math.sqrt(mean(below)) if below else 0.0
    u_band = e_now + 2 * sigma_above
    l_band = e_now - 2 * sigma_below

    return safe_tanh((P[-1] - e_now) / (u_band - l_band + EPS))


# =============================================================================
# FORMULA 13: Directional Inertia Quantifier (DIQ)
# =============================================================================
def f13_diq(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 8:
        return 0.0
    P = ctx.P[-(W_eff + 1):]
    V = ctx.V[-(W_eff + 1):]
    r = pct_returns(P)
    n = len(r)

    half = W_eff // 2
    dir_ = sign(P[-1] - P[-1 - min(half, len(P) - 1)])
    if dir_ == 0:
        return 0.0

    mass = 0.0
    vel_num = 0.0
    vel_den = 0.0
    counter = 0.0
    # i = minutes ago, most recent return at i = 0
    for i in range(n):
        idx = n - 1 - i
        decay = math.exp(-0.05 * i)
        v = V[idx + 1] if idx + 1 < len(V) else V[idx]
        rv = r[idx]
        s = sign(rv)
        if s == dir_:
            mass += v * decay
            vel_num += abs(rv) * decay
            vel_den += decay
        else:
            counter += v * abs(rv) * decay

    v_dir = dir_ * (vel_num / vel_den) if vel_den > EPS else 0.0
    counter += EPS
    return safe_tanh(mass * v_dir / counter)


# =============================================================================
# FORMULA 14: Temporal Concentration Index (TCI)
# =============================================================================
def f14_tci(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 10:
        return 0.0
    r = pct_returns(ctx.P[-(W_eff + 1):])
    if len(r) < 8:
        return 0.0
    abs_r = [abs(v) for v in r]
    med = median(abs_r)
    if med <= EPS:
        return 0.0

    sig = [(i, r[i]) for i in range(len(r)) if abs_r[i] > 1.5 * med]
    n_sig = len(sig)
    if n_sig == 0:
        return 0.0

    B = max(4, W_eff // 10)
    counts = [0] * B
    bin_returns: list[list[float]] = [[] for _ in range(B)]
    for i, rv in sig:
        b = min(B - 1, int(i / len(r) * B))
        counts[b] += 1
        bin_returns[b].append(rv)

    h_sig = 0.0
    for c in counts:
        if c > 0:
            p = c / n_sig
            h_sig -= p * math.log2(p + EPS)
    h_max = math.log2(B)
    if h_max <= EPS:
        return 0.0

    best_bin = max(range(B), key=lambda b: counts[b])
    r_conc = mean(bin_returns[best_bin])
    return clamp((h_max - h_sig) / h_max, 0.0, 1.0) * sign(r_conc)


# =============================================================================
# FORMULA 15: Asymmetric Recovery Speed Index (ARSI)
# =============================================================================
def f15_arsi(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 12:
        return 0.0
    P = ctx.P[-(W_eff + 1):]
    changes = abs_changes(P)
    atr = mean([abs(c) for c in changes])
    if atr <= EPS:
        return 0.0
    n = len(P)
    last = n - 1
    threshold = 0.5 * atr

    local_max = [i for i in range(1, n - 1) if P[i] > P[i - 1] and P[i] > P[i + 1]]
    local_min = [i for i in range(1, n - 1) if P[i] < P[i - 1] and P[i] < P[i + 1]]

    drops: list[tuple[float, float]] = []   # (speed, recency_weight)
    for mx in local_max:
        nxt = next((m for m in local_min if m > mx), None)
        if nxt is None:
            continue
        drop_size = P[mx] - P[nxt]
        if drop_size <= threshold:
            continue
        target = P[nxt] + 0.5 * drop_size
        recovery_time = None
        for k in range(nxt + 1, n):
            if P[k] >= target:
                recovery_time = k - nxt
                break
        if recovery_time is None:
            recovery_time = max(1, last - nxt)      # penalize: remaining minutes
        speed = 0.5 * drop_size / (recovery_time + 1)
        drops.append((speed, math.exp(-0.1 * (last - nxt))))

    gains: list[tuple[float, float]] = []
    for mn in local_min:
        nxt = next((m for m in local_max if m > mn), None)
        if nxt is None:
            continue
        gain_size = P[nxt] - P[mn]
        if gain_size <= threshold:
            continue
        target = P[nxt] - 0.5 * gain_size
        giveback_time = None
        for k in range(nxt + 1, n):
            if P[k] <= target:
                giveback_time = k - nxt
                break
        if giveback_time is None:
            giveback_time = max(1, last - nxt)
        speed = 0.5 * gain_size / (giveback_time + 1)
        gains.append((speed, math.exp(-0.1 * (last - nxt))))

    def wavg(items: list[tuple[float, float]]) -> float:
        if not items:
            return 0.0
        wsum = sum(w for _, w in items)
        if wsum <= EPS:
            return 0.0
        return sum(s * w for s, w in items) / wsum

    r_drop = wavg(drops)
    r_gain = wavg(gains)
    return safe_tanh((r_drop - r_gain) / (r_drop + r_gain + EPS))


# =============================================================================
# FORMULA 16: Information Surprise Accumulator (ISA)
# =============================================================================
def f16_isa(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 16:
        return 0.0
    P = ctx.P[-(W_eff + 1):]
    last = len(P) - 1
    r = pct_returns(P)
    recent_start = max(11, last - max(1, W_eff // 4))

    sa = 0.0
    surprise_sum = 0.0
    for k in range(recent_start, last + 1):
        mu_local = (P[k - 1] - P[k - 4]) / 3.0
        p_pred = P[k - 1] + mu_local
        # last 10 returns ENDING at k-1 (known at prediction time): r[j] is the
        # change P[j] -> P[j+1], so the latest usable index is k-2
        window_r = r[max(0, k - 11):k - 1]
        if len(window_r) < 3:
            continue
        sigma_local = std_pop(window_r) * P[k - 1]
        sigma_local = max(sigma_local, EPS)
        surprise = (P[k] - p_pred) ** 2 / (sigma_local ** 2)
        sa += surprise * math.exp(-0.1 * (last - k))
        surprise_sum += P[k] - p_pred
    ctx.intermediates["sa"] = sa

    sa_hist = ctx.hist.ring(ctx.hist.tf_key(ctx.symbol, ctx.tf, "sa"), HIST_SA)
    sa_max = max(sa_hist.vmax(), sa, EPS)
    d_surprise = sign(surprise_sum)
    if d_surprise == 0:
        return 0.0
    return safe_tanh(sa / sa_max * d_surprise)


# =============================================================================
# FORMULA 17: Volume Acceleration Divergence (VAD)
# =============================================================================
def f17_vad(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 22:
        return 0.0
    P = ctx.P[-(W_eff + 1):]
    V = ctx.V[-(W_eff + 1):]
    d_p = [P[i] - P[i - 1] for i in range(1, len(P))]
    d_v = [V[i] - V[i - 1] for i in range(1, len(V))]

    v_p = ema(d_p, 10)
    a_p = ema([v_p[i] - v_p[i - 1] for i in range(1, len(v_p))], 10)
    v_v = ema(d_v, 10)
    a_v = ema([v_v[i] - v_v[i - 1] for i in range(1, len(v_v))], 10)
    if not a_p or not a_v:
        return 0.0

    v_p_now = v_p[-1]
    a_p_now = a_p[-1]
    a_v_now = a_v[-1]
    return safe_tanh(
        (a_v_now - a_p_now) / (abs(a_v_now) + abs(a_p_now) + EPS) * sign(v_p_now)
    )


# =============================================================================
# FORMULA 18: Consecutive Candle Magnitude Decay Rate (CCMDR)
# =============================================================================
def current_streak(ctx) -> tuple[int, int, float]:
    """Return (streak_sign, N, lambda_current) for the live directional streak.

    Shared by the formula itself and by the engine, which commits streak
    outcomes to the auto-learning history strictly once per GRC.
    """
    W_eff = ctx.W_eff
    if W_eff < 8:
        return 0, 0, 0.0
    r = pct_returns(ctx.P[-(W_eff + 1):])
    if not r:
        return 0, 0, 0.0
    streak_sign = sign(r[-1])
    if streak_sign == 0:
        return 0, 0, 0.0
    magnitudes: list[float] = []
    for v in reversed(r):
        if sign(v) != streak_sign:
            break
        magnitudes.append(abs(v))
    magnitudes.reverse()
    N = len(magnitudes)
    if N < 3:
        return streak_sign, N, 0.0
    b_est = min(magnitudes) * 0.5
    xs = [float(i) for i in range(1, N + 1)]
    ys = [math.log(max(c - b_est, 1e-12)) for c in magnitudes]
    slope = linear_slope(xs, ys)
    return streak_sign, N, -slope            # c_i = A * exp(-lam * i) + B


def f18_ccmdr(ctx) -> float:
    streak_sign, N, lam_current = current_streak(ctx)
    if N < 3:
        return 0.0
    lam_critical = ctx.hist.lambda_critical(ctx.symbol, N)
    if lam_critical is None:
        lam_critical = 0.15                  # early-session default per spec
    return safe_tanh((lam_current - lam_critical) / (abs(lam_critical) + EPS))


# =============================================================================
# FORMULA 19: Pairwise Momentum Transfer Lag (PMTL)
# =============================================================================
def f19_pmtl(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 8 or not ctx.others:
        return 0.0
    P = ctx.P[-(W_eff + 1):]
    T = ctx.T[-(W_eff + 1):]
    r_t = pct_returns(P)
    n = len(r_t)
    l_max = max(1, W_eff // 4)

    # backward-indexed returns: rt[i] = return at t - i
    rt = list(reversed(r_t))

    num_total = 0.0
    w_total = 0.0
    for sym, (times, prices) in ctx.others.items():
        # forward-fill `other` prices onto the target's minute grid
        other_map: dict[int, float] = dict(zip(times, prices))
        series: list[float] = []
        last_val = None
        for ts in T:
            if ts in other_map:
                last_val = other_map[ts]
            if last_val is None:             # try backwards for the first point
                prior = [other_map[t] for t in times if t <= ts]
                last_val = prior[-1] if prior else None
            if last_val is None:
                break
            series.append(last_val)
        if len(series) != len(T) or len(series) < 4:
            continue
        ra_fwd = pct_returns(series)
        ra = list(reversed(ra_fwd))          # ra[i] = other's return at t - i
        if len(ra) != n:
            continue

        denom = math.sqrt(
            sum(v * v for v in ra[:n]) * sum(v * v for v in rt[:n])
        )
        if denom < EPS:
            continue

        best_lag, best_ccf = 1, 0.0
        for lag in range(1, min(l_max, n - 1) + 1):
            ccf_num = sum(ra[i + lag] * rt[i] for i in range(0, n - lag))
            ccf = ccf_num / denom
            if abs(ccf) > abs(best_ccf):
                best_lag, best_ccf = lag, ccf

        lag_sign = sign(best_ccf)
        confidence = abs(best_ccf)
        w = 1.0 / (best_lag + 1)
        r_a_lagged = ra[best_lag]            # other's return `optimal_lag` ago
        num_total += lag_sign * confidence * r_a_lagged * w
        w_total += w

    if w_total <= EPS:
        return 0.0
    return safe_tanh(num_total / w_total)


# =============================================================================
# FORMULA 20: Dynamic Regime Half-Life Estimator (DRHE)  — NOT bounded to [-1,1]
# =============================================================================
def classify_regime(trend_strength: float, volatility_level: float) -> str:
    if trend_strength > 0.3:
        return "volatile_trending" if volatility_level > 1.5 else "trending"
    if volatility_level > 1.5:
        return "volatile_ranging"
    if volatility_level <= 0.8:
        return "calm_ranging"
    return "transitional"


def _weibull_k_mle(durations: list[float]) -> float:
    """Newton iteration for the Weibull shape MLE given in the spec."""
    if len(durations) < 3:
        return 1.0
    n = len(durations)
    logs = [math.log(max(d, 1e-9)) for d in durations]
    mean_log = sum(logs) / n
    k = 1.0
    for _ in range(40):
        pow_d = [d ** k for d in durations]
        s_pow = sum(pow_d)
        if s_pow <= 0 or s_pow != s_pow:
            break
        weighted = sum(p * lg for p, lg in zip(pow_d, logs))
        g = weighted / s_pow - 1.0 / k - mean_log
        # derivative (analytic, close Newton step)
        d_weighted = sum(p * lg * lg for p, lg in zip(pow_d, logs))
        d_g = (d_weighted / s_pow - (weighted / s_pow) ** 2) + 1.0 / (k * k)
        if abs(d_g) < 1e-12:
            break
        step = g / d_g
        k_new = clamp(k - step, 0.1, 10.0)
        if abs(k_new - k) < 1e-6:
            k = k_new
            break
        k = k_new
    return k


def f20_drhe(ctx) -> float:
    """Returns T_remaining / tau_timeframe — range 0..5+ (multiplier, not signal)."""
    W_eff = ctx.W_eff
    if W_eff < 12:
        return 1.0
    P = ctx.P[-(W_eff + 1):]
    r = pct_returns(P)
    sd = std_pop(r)
    mu = mean(r)
    ctx.intermediates["std_r"] = sd
    trend_strength = abs(mu) / sd if sd > EPS else 0.0

    std_hist = ctx.hist.ring(ctx.hist.tf_key(ctx.symbol, ctx.tf, "std_r"), HIST_STD_R)
    vol_baseline = std_hist.median()
    volatility_level = (sd / vol_baseline) if vol_baseline > EPS else 1.0

    regime = classify_regime(trend_strength, volatility_level)
    from ..history import _regime_to_float
    ctx.intermediates["regime_label"] = _regime_to_float(regime)

    # regime age from GRC history (labels recorded strictly at GRC cadence)
    age_grcs, hist_len = ctx.hist.regime_ages(ctx.symbol, ctx.tf)
    regime_age_minutes = age_grcs * 4     # each GRC step is 4 minutes

    durations = [
        float(d)
        for d in ctx.hist.regime_durations.get((ctx.symbol, ctx.tf, regime), [])
    ]
    tau = float(ctx.W)
    if len(durations) < 3:
        return 1.0                        # insufficient history -> neutral support

    d_avg = mean(durations)
    k = _weibull_k_mle(durations)
    # scale lambda from MLE: lambda = ((1/n) * sum d^k)^(1/k)
    lam = (sum(d ** k for d in durations) / len(durations)) ** (1.0 / k)
    if lam <= EPS:
        return 1.0

    s_age = math.exp(-((regime_age_minutes / lam) ** k))
    # practical approximation prescribed by the spec (valid for k in [0.5, 3])
    t_remaining = max(1.0, d_avg * s_age * (d_avg - regime_age_minutes * (1.0 - s_age)))
    return max(0.0, t_remaining / tau)
