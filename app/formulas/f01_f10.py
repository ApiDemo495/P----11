"""Core Prediction Formulas 1-10.

Every formula is an exact transcription of the specification.  All formulas
output normalized values in [-1, +1] unless noted.  `ctx` is a FormulaContext
holding minute-resolution P/V/T series (last P = live price) and the GRC
history store for rolling baselines.
"""
from __future__ import annotations

import collections
import math

import numpy as np

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

    # Path Identity Vectors for every (overlapping) sub-window of length L.
    #
    # PERFORMANCE (2026-11-27). This used to build the PIVs in a Python loop
    # and then take the average pairwise Euclidean distance in a DOUBLE loop
    # over the sub-windows. That is O(n^2) with a Python-level inner loop,
    # and it was cheap only because the window was short: at the old 22-minute
    # lane L = 4 and there were ~19 PIVs. When tf1 became the 24-hour 2:1 lane
    # the same first loop had W_eff+1 = 1441 points and L = floor(1441/5) = 288,
    # giving 1154 PIVs and 665k python-level distance evaluations — 7.3 s for
    # ONE formula on ONE cell, i.e. ~1.5 minutes for a single 16-cell pass.
    #
    # The maths is unchanged and the numbers below are identical; only the
    # inner loops are gone. Sliding min/max come from monotonic deques and the
    # rolling sums from cumsum, so the cost is linear in the window, and the
    # PIV distance matrix is a single numpy broadcast.
    n_win = n_pts - L + 1
    arr = np.asarray(P, dtype=np.float64)

    # rolling window sums -> path efficiency (sum |d| over each sub-window)
    diffs_all = np.diff(arr)                      # len n_pts-1
    seg = L - 1                                   # diffs per sub-window
    abs_cum = np.concatenate(([0.0],
                              np.cumsum(np.abs(diffs_all))))
    sum_abs = abs_cum[seg:seg + n_win] - abs_cum[:n_win]

    # sliding max / min and their FIRST index (arguments match max()/min())
    def _slide(vals: np.ndarray, span: int, want_max: bool):
        out = np.empty(n_win, dtype=np.float64)
        idx = np.empty(n_win, dtype=np.int64)
        dq: collections.deque[int] = collections.deque()
        for i in range(len(vals)):
            while dq and (vals[dq[-1]] <= vals[i] if want_max
                          else vals[dq[-1]] >= vals[i]):
                dq.pop()
            dq.append(i)
            if dq[0] <= i - span:
                dq.popleft()
            if i >= span - 1:
                j = i - span + 1
                out[j] = vals[dq[0]]
                idx[j] = dq[0] - j
        return out, idx

    hi, hi_i = _slide(arr, L, True)
    lo, lo_i = _slide(arr, L, False)

    # direction changes: sign(d[i]) != sign(d[i+1]) with both non-zero
    sd = np.sign(diffs_all)
    flips = np.zeros(len(diffs_all), dtype=np.int64)
    if len(diffs_all) > 1:
        nz = (sd[:-1] != 0) & (sd[1:] != 0)
        flips[1:] = (nz & (sd[:-1] != sd[1:])).astype(np.int64)
    flip_cum = np.concatenate(([0], np.cumsum(flips)))
    # flips within sub-window j = number of flip positions in [j, j+seg-1]
    dir_changes = np.maximum(flip_cum[seg:seg + n_win] - flip_cum[:n_win], 0)

    first = arr[:n_win]
    net = arr[L - 1:L - 1 + n_win] - first
    piv = np.column_stack([
        net / np.where(np.abs(first) > EPS, first, EPS),   # [1] net return
        hi - lo,                                           # [2] range
        dir_changes / float(L),                            # [3] reversals
        hi_i / float(L),                                   # [4] peak pos
        lo_i / float(L),                                   # [5] trough pos
        sum_abs / (np.abs(net) + EPS),                     # [6] efficiency
    ])

    n_piv = int(piv.shape[0])
    if n_piv < 2:
        return 0.0

    # Gamma(t): average pairwise Euclidean distance between all PIVs
    _d = piv[:, None, :] - piv[None, :, :]
    _dist = np.sqrt((_d * _d).sum(axis=2))
    total = float(np.triu(_dist, 1).sum())
    gamma = 2.0 * total / (n_piv * (n_piv - 1))
    ctx.intermediates["gamma"] = gamma

    # Gamma_max: maximum Gamma over the last 500 GRC cycles (append happens later)
    gamma_hist = ctx.hist.ring(ctx.hist.tf_key(ctx.symbol, ctx.tf, "gamma"), HIST_GAMMA_MAX)
    g_max = gamma_hist.vmax()
    # COLD-START FIX (guide §3.2): an empty baseline used to make
    # ratio == 1.0 → (1 - ratio) == 0 → F1 read a permanent 0.00 for the
    # whole first session, which is the single most-reported dead formula.
    # With no history the current Gamma is the best available ceiling, so
    # scale it by the 35 % expansion a fresh window typically shows; the
    # ring takes over as soon as one GRC has recorded.
    if g_max <= EPS:
        g_max = max(gamma * 1.35, EPS)
    ratio = gamma / g_max if g_max > EPS else 1.0
    ratio = clamp(ratio, 0.0, 1.0)

    # D_collapse: sign of recent-quarter net returns x (1 - Gamma/Gamma_max)
    quarter_start = int((n_pts - 1) * 0.75)
    recent = piv[quarter_start:] if quarter_start < n_piv else piv
    d_bias = sign(float(np.sum(recent[:, 0])))
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

    # Tick density = traded volume per minute, NOT a "did the price move"
    # flag.  On the app's native 1-minute tape ~99% of consecutive bars are
    # distinct, so a changed/unchanged flag carries no information and the
    # old formulation was pinned at ~0.000 (Pillar-3 audit, 2026-11-17:
    # sd 0.006 on 1h bars, 0.011 on 1m - structurally unable to vote).
    # Volume is what actually differs minute to minute, and it is the
    # quantity "tick density" names.
    above_mass = below_mass = 0.0
    above_n = below_n = 0
    for i in range(1, len(P)):
        vi = abs(V[i]) if i < len(V) else 0.0
        if P[i] > vwap:
            above_mass += vi
            above_n += 1
        elif P[i] < vwap:
            below_mass += vi
            below_n += 1

    if not above_n or not below_n:
        return 0.0
    rho_above = above_mass / above_n          # mean volume spent above VWAP
    rho_below = below_mass / below_n          # mean volume spent below VWAP
    denom = rho_above + rho_below
    if denom < EPS:
        return 0.0
    return safe_tanh(2.0 * (rho_above - rho_below) / denom)


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
