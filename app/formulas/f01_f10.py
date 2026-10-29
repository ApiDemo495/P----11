"""Core Prediction Formulas 1-10.

Every formula is an exact transcription of the specification.  All formulas
output normalized values in [-1, +1] unless noted.  `ctx` is a FormulaContext
holding minute-resolution P/V/T series (last P = live price) and the GRC
history store for rolling baselines.
"""
from __future__ import annotations

import math

from ._helpers import (
    abs_changes,
    clamp,
    excess_kurtosis,
    linear_slope,
    mean,
    pct_returns,
    safe_tanh,
    sign,
    skewness,
    std_pop,
)
from ..config import (
    HIST_CWP,
    HIST_GAMMA_MAX,
    HIST_IDIV,
    HIST_IRECENT_MAX,
    HIST_SS,
    HIST_TAU,
)

EPS = 1e-12


# =============================================================================
# FORMULA 1: Recursive Price Identity Collapse (RPIC)
# =============================================================================
def f01_rpic(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 15:
        return 0.0
    P = ctx.P[-(W_eff + 1):]
    L = max(3, min(W_eff, ctx.W) // 5)          # L = floor(W/5)
    n_pts = len(P)
    if n_pts < L + 1:
        return 0.0

    # Path Identity Vectors for every (overlapping) sub-window of length L
    pivs: list[list[float]] = []
    for j in range(0, n_pts - L + 1):
        sub = P[j:j + L]
        net = sub[-1] - sub[0]
        diffs = [sub[i + 1] - sub[i] for i in range(L - 1)]
        dir_changes = sum(
            1 for i in range(len(diffs) - 1)
            if diffs[i] != 0 and diffs[i + 1] != 0 and sign(diffs[i]) != sign(diffs[i + 1])
        )
        # time of max/min within sub-window normalized to sub-window length
        t_max = max(range(L), key=lambda i: sub[i])
        t_min = min(range(L), key=lambda i: sub[i])
        path_eff = sum(abs(d) for d in diffs) / (abs(net) + EPS)
        pivs.append([
            net / (sub[0] if abs(sub[0]) > EPS else EPS),      # [1] net return
            max(sub) - min(sub),                               # [2] range traversed
            dir_changes / L,                                   # [3] reversal frequency
            t_max / L,                                         # [4] peak position
            t_min / L,                                         # [5] trough position
            path_eff,                                          # [6] path efficiency
        ])

    n_piv = len(pivs)
    if n_piv < 2:
        return 0.0

    # Gamma(t): average pairwise Euclidean distance between all PIVs
    total = 0.0
    for i in range(n_piv):
        a = pivs[i]
        for j in range(i + 1, n_piv):
            b = pivs[j]
            total += math.sqrt(sum((a[d] - b[d]) ** 2 for d in range(6)))
    gamma = 2.0 * total / (n_piv * (n_piv - 1))
    ctx.intermediates["gamma"] = gamma

    # Gamma_max: maximum Gamma over the last 500 GRC cycles (append happens later)
    gamma_hist = ctx.hist.ring(ctx.hist.tf_key(ctx.symbol, ctx.tf, "gamma"), HIST_GAMMA_MAX)
    g_max = gamma_hist.vmax()
    ratio = gamma / g_max if g_max > EPS else 1.0
    ratio = clamp(ratio, 0.0, 1.0)

    # D_collapse: sign of recent-quarter net returns x (1 - Gamma/Gamma_max)
    quarter_start = int((n_pts - 1) * 0.75)
    recent = pivs[quarter_start:] if quarter_start < n_piv else pivs
    d_bias = sign(sum(p[0] for p in recent))
    d_collapse = d_bias * (1.0 - ratio)

    return safe_tanh(ratio * d_collapse)


# =============================================================================
# FORMULA 2: Cumulative Displacement Inversion Frequency (CDIF)
# =============================================================================
def f02_cdif(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 16:
        return 0.0
    P = ctx.P[-(W_eff + 1):]
    r = pct_returns(P)                       # length W_eff
    n = len(r)

    # Cumulative displacement series
    cd: list[float] = []
    acc = 0.0
    for v in r:
        acc += v
        cd.append(acc)

    diffs = [cd[k] - cd[k - 1] for k in range(1, len(cd))]     # == r[1:]
    threshold = 0.3 * std_pop(diffs)

    # inversion at k (index into cd): sign of increments flips and magnitude > threshold
    inversions: list[int] = [0] * len(cd)                      # 1 at inversion points
    for k in range(2, len(cd)):
        d_now = cd[k] - cd[k - 1]
        d_prev = cd[k - 1] - cd[k - 2]
        if sign(d_now) != sign(d_prev) and abs(d_now) > threshold:
            inversions[k] = 1

    # per-quarter buckets across the entire window
    q = max(1, n // 4)
    quarters = [
        sum(inversions[0:q]),
        sum(inversions[q:2 * q]),
        sum(inversions[2 * q:3 * q]),
        sum(inversions[3 * q:]),
    ]
    i_recent = quarters[3]
    ctx.intermediates["irecent"] = float(i_recent)
    i_baseline = (quarters[0] + quarters[1] + quarters[2]) / 3.0
    sigma_i = std_pop([float(x) for x in quarters])
    if sigma_i < 1e-9:
        sigma_i = 1.0

    # Inversion momentum direction
    cd_t = cd[-1]
    i_hist = ctx.hist.ring(ctx.hist.tf_key(ctx.symbol, ctx.tf, "irecent"), HIST_IRECENT_MAX)
    i_max = i_hist.vmax()
    phi = sign(cd_t) * (1.0 - i_recent / (i_max + 1.0))

    return safe_tanh((i_recent - i_baseline) / sigma_i * phi)


# =============================================================================
# FORMULA 3: Proportional Exhaustion Mapping (PEM)
# =============================================================================
def f03_pem(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 10:
        return 0.0
    P = ctx.P[-(W_eff + 1):]
    r = pct_returns(P)
    t_last = len(P) - 1

    def runs(up: bool) -> list[float]:
        out: list[float] = []
        i = 0
        while i < len(r):
            in_run = (r[i] >= 0) if up else (r[i] <= 0)
            if not in_run:
                i += 1
                continue
            j = i
            while j + 1 < len(r):
                ok = (r[j + 1] >= 0) if up else (r[j + 1] <= 0)
                if not ok:
                    break
                j += 1
            # run covers returns i..j, price points i..j+1
            p_start = P[i]
            distance = (P[j + 1] - P[i]) if up else (P[i] - P[j + 1])
            energy = sum(v * v for v in r[i:j + 1])
            eff = distance / (energy * p_start + EPS)
            weight = math.exp(-0.1 * (t_last - (j + 1)))
            out.append((eff, weight))
            i = j + 2
        return out

    def weighted(effs: list[tuple[float, float]]) -> float:
        if not effs:
            return 0.0
        wsum = sum(w for _, w in effs)
        if wsum <= EPS:
            return 0.0
        return sum(e * w for e, w in effs) / wsum

    eta_up = weighted(runs(up=True))
    eta_down = weighted(runs(up=False))
    direction = sign(P[-1] - P[0])
    return safe_tanh((eta_up - eta_down) * direction)


# =============================================================================
# FORMULA 4: Stochastic Arrival Time Differential (SATD)
# =============================================================================
def f04_satd(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 5:
        return 0.0
    P = ctx.P[-(W_eff + 1):]
    changes = abs_changes(P)                          # r_abs per minute
    atr = sum(abs(c) for c in changes) / len(changes)  # ATR_custom
    mu = (P[-1] - P[0]) / W_eff
    sigma = std_pop(changes)
    eps2 = max(sigma / (10.0 * W_eff), 1e-15)

    bull_target = P[-1] + atr
    bear_target = P[-1] - atr

    e_bull = abs(bull_target - P[-1]) / max(mu, eps2)
    e_bear = abs(bear_target - P[-1]) / max(-mu, eps2)

    return safe_tanh((e_bear - e_bull) / (e_bear + e_bull + EPS))


# =============================================================================
# FORMULA 5: Interval Return Kurtosis Gradient (IRKG)
# =============================================================================
_FIB_SCALES = [1, 2, 3, 5, 8, 13, 21]


def f05_irkg(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 10:
        return 0.0
    P = ctx.P[-(W_eff + 1):]

    scales: list[int] = []
    kappas: list[float] = []
    returns_at: dict[int, list[float]] = {}
    for s in _FIB_SCALES:
        if s > W_eff - 1:
            continue
        rs = [(P[i] - P[i - s]) / P[i - s] for i in range(s, len(P))
              if abs(P[i - s]) > EPS]
        if len(rs) < 6:
            continue
        scales.append(s)
        kappas.append(excess_kurtosis(rs))
        returns_at[s] = rs

    if len(scales) < 2:
        return 0.0

    gradient = linear_slope([float(s) for s in scales], kappas)

    # s_target: the Fibonacci scale closest to floor(timeframe_duration / 10)
    target_w = max(1, ctx.W // 10)
    s_target = min(scales, key=lambda s: abs(s - target_w))
    skew_t = skewness(returns_at[s_target])

    return safe_tanh(gradient * sign(skew_t))


# =============================================================================
# FORMULA 6: Self-Similarity Breakpoint Detection (SSBD)
# =============================================================================
def f06_ssbd(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 20:
        return 0.0
    P = ctx.P[-(W_eff + 1):]
    r = pct_returns(P)
    M = 5
    seg_len = len(r) // M
    if seg_len < 3:
        return 0.0

    whole = [mean(r), std_pop(r), skewness(r),
             sum(1 for v in r if v > 0) / len(r)]

    fingerprints: list[list[float]] = []
    for m in range(M):
        seg = r[m * seg_len:(m + 1) * seg_len] if m < M - 1 else r[m * seg_len:]
        if len(seg) < 3:
            seg = r[-max(3, len(r) // M):]
        fingerprints.append([
            mean(seg),
            std_pop(seg),
            skewness(seg),
            sum(1 for v in seg if v > 0) / len(seg),
        ])

    devs: list[list[float]] = []
    for fp in fingerprints:
        normed = []
        for k in range(4):
            base = whole[k] if abs(whole[k]) > EPS else EPS
            normed.append(clamp(fp[k] / base, -50.0, 50.0))
        devs.append(normed)

    dists = [math.sqrt(sum((v - 1.0) ** 2 for v in d)) for d in devs]
    ss_current = 1.0 / (1.0 + mean(dists))
    ctx.intermediates["ss"] = ss_current

    ss_hist = ctx.hist.ring(ctx.hist.tf_key(ctx.symbol, ctx.tf, "ss"), HIST_SS)
    if len(ss_hist) < 2:
        return 0.0
    ss_baseline = ss_hist.mean()
    sigma_ss = ss_hist.std()
    if sigma_ss < 1e-9:
        sigma_ss = 1e-9

    # most anomalous segment
    m_star = max(range(M), key=lambda i: dists[i])
    d_break = sign(fingerprints[m_star][0])

    return safe_tanh((ss_baseline - ss_current) / sigma_ss * d_break)


# =============================================================================
# FORMULA 7: Weighted Return Polarity Persistence (WRPP)
# =============================================================================
def f07_wrpp(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 8:
        return 0.0
    P = ctx.P[-(W_eff + 1):]
    r = pct_returns(P)
    delta = max(3, W_eff // 8)

    def cwp(end_idx: int) -> float:
        if end_idx < 0 or end_idx >= len(r):
            return 0.0
        s = sign(r[end_idx])
        if s == 0:
            return 0.0
        # walk backwards while the sign persists
        L = 0
        i = end_idx
        while i >= 0 and sign(r[i]) == s:
            L += 1
            i -= 1
        if L == 0:
            return 0.0
        acc = 0.0
        for k in range(L):
            val = abs(r[end_idx - k])
            weight = (L - k) / L          # most recent minute gets weight 1.0
            acc += val * weight
        return s * acc

    cur = cwp(len(r) - 1)
    prev = cwp(len(r) - 1 - delta) if len(r) - 1 - delta >= 0 else 0.0
    ctx.intermediates["cwp"] = cur

    sigma_hist = ctx.hist.ring(ctx.hist.tf_key(ctx.symbol, ctx.tf, "cwp"), HIST_CWP)
    if len(sigma_hist) < 5:
        return 0.0
    sigma = sigma_hist.std()
    if sigma < 1e-12:
        return 0.0
    return safe_tanh((cur - prev) / sigma)


# =============================================================================
# FORMULA 8: Relative Tick Density Imbalance (RTDI)
# =============================================================================
def f08_rtdi(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 5:
        return 0.0
    P = ctx.P[-(W_eff + 1):]
    V = ctx.V[-(W_eff + 1):]

    pv = sum(p * v for p, v in zip(P, V))
    vv = sum(V)
    if abs(vv) < EPS:
        vwap = mean(P)
    else:
        vwap = pv / vv

    above_total = above_ticks = 0
    below_total = below_ticks = 0
    for i in range(1, len(P)):
        changed = abs(P[i] - P[i - 1]) > 0
        if P[i] > vwap:
            above_total += 1
            above_ticks += 1 if changed else 0
        elif P[i] < vwap:
            below_total += 1
            below_ticks += 1 if changed else 0

    rho_above = (above_ticks / above_total) if above_total else 0.0
    rho_below = (below_ticks / below_total) if below_total else 0.0

    return safe_tanh((rho_above - rho_below) / (rho_above + rho_below + EPS))


# =============================================================================
# FORMULA 9: Cascading Lag Decay Function (CLDF)
# =============================================================================
def f09_cldf(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 12:
        return 0.0
    P = ctx.P[-(W_eff + 1):]
    r = pct_returns(P)
    n = len(r)
    r_mean = mean(r)
    denom = sum((v - r_mean) ** 2 for v in r)
    if denom < 1e-30:
        return 0.0

    l_max = max(1, n // 3)
    l_max = min(l_max, n - 2)
    acf: list[float] = []
    for k in range(1, l_max + 1):
        num = 0.0
        for i in range(0, n - k):
            num += (r[i] - r_mean) * (r[i + k] - r_mean)
        acf.append(num / denom)

    if not acf:
        return 0.0
    rho1 = acf[0]
    threshold = 0.5 * abs(rho1)
    tau_half = l_max
    if abs(rho1) > EPS:
        for idx, val in enumerate(acf, start=1):
            if abs(val) < threshold:
                tau_half = idx
                break
    ctx.intermediates["tau"] = float(tau_half)

    tau_hist = ctx.hist.ring(ctx.hist.tf_key(ctx.symbol, ctx.tf, "tau"), HIST_TAU)
    if len(tau_hist) < 5:
        return 0.0
    tau_base = tau_hist.median()
    if tau_base <= 0:
        return 0.0
    return safe_tanh((tau_half - tau_base) / tau_base * sign(rho1))


# =============================================================================
# FORMULA 10: Displacement Volume Divergence Integral (DVDI)
# =============================================================================
def f10_dvdi(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 8:
        return 0.0
    P = ctx.P[-(W_eff + 1):]
    V = ctx.V[-(W_eff + 1):]
    last = len(P) - 1

    p_range = max(P) - min(P)
    cum_v: list[float] = []
    acc = 0.0
    for v in V:
        acc += v
        cum_v.append(acc)
    v_total = cum_v[-1] - cum_v[0]

    i_div = 0.0
    for k in range(1, len(P)):
        np_k = (P[k] - P[0]) / (p_range + EPS)
        nv_k = (cum_v[k] - cum_v[0]) / (v_total + EPS)
        div = np_k - nv_k
        r_sign = sign(P[k] - P[k - 1])
        recency = 1.0 / (1.0 + (last - k) / max(1, W_eff))
        i_div += div * r_sign * recency
    ctx.intermediates["idiv"] = i_div

    idiv_hist = ctx.hist.ring(ctx.hist.tf_key(ctx.symbol, ctx.tf, "idiv"), HIST_IDIV)
    i_max = idiv_hist.vmax()
    i_max = max(i_max, abs(i_div), EPS)
    return safe_tanh(i_div / i_max)
