"""Core Prediction Formulas 22-27 and 31 (exact transcription of part 2).

Formula 28 (ARC) and Formula 29 (DLP) live in app/risk.py — they are risk
formulas operating on price levels, not [-1,1] signals.  Formula 30 (PCV) is
implemented as f30_pcv() below because it requires the CCI as its input and
therefore runs after aggregation (validator layer, like the DRHE multiplier).
"""
from __future__ import annotations

import cmath
import math

from ._helpers import ema, linear_slope, mean, median, pct_returns, safe_tanh, sign, std_pop
from ..config import HIST_R_ANOMALY, HIST_VOL_CLUSTERS

EPS = 1e-12


# =============================================================================
# FORMULA 22: Bid-Ask Phantom Pressure Reconstructor (BAPR)
# =============================================================================
def f22_bapr(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 10:
        return 0.0
    P, V = ctx.P[-(W_eff + 1):], ctx.V[-(W_eff + 1):]
    O, H, L = (x[-(W_eff + 1):] for x in ctx.candles())
    n = len(P)
    ema_v = ema(V, 20)
    last = n - 1

    phi_bid = 0.0
    phi_ask = 0.0
    for i in range(n):
        rng = H[i] - L[i]
        price_position = (P[i] - L[i]) / (rng + EPS)
        vol_intensity = V[i] / ema_v[i] if ema_v[i] > EPS else 0.0
        recency = math.exp(-0.03 * (last - i))

        close_open = P[i] - O[i]
        bid_c = (
            price_position * vol_intensity
            * (1.0 + max(0.0, close_open) / (rng + EPS))
        )
        ask_c = (
            (1.0 - price_position) * vol_intensity
            * (1.0 + max(0.0, -close_open) / (rng + EPS))
        )
        phi_bid += bid_c * recency
        phi_ask += ask_c * recency

    return safe_tanh((phi_bid - phi_ask) / (phi_bid + phi_ask + EPS))


# =============================================================================
# FORMULA 23: Sequential Absorption Rate Differential (SARD)
# =============================================================================
def f23_sard(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 8:
        return 0.0
    P, V = ctx.P[-(W_eff + 1):], ctx.V[-(W_eff + 1):]
    r = pct_returns(P)
    last = len(P) - 1

    def waves(up: bool) -> list[tuple[float, float]]:
        """(absorption, recency_weight) per wave."""
        out: list[tuple[float, float]] = []
        i = 0
        while i < len(r):
            in_wave = (r[i] >= 0) if up else (r[i] <= 0)
            if not in_wave:
                i += 1
                continue
            j = i
            while j + 1 < len(r):
                ok = (r[j + 1] >= 0) if up else (r[j + 1] <= 0)
                if not ok:
                    break
                j += 1
            distance = P[j + 1] - P[i] if up else P[i] - P[j + 1]
            wave_vol = sum(V[i:j + 2])          # minutes i..j+1 of the wave
            absorption = wave_vol / (abs(distance) + EPS)
            wave_age = last - (j + 1)           # minutes since wave end
            out.append((absorption, math.exp(-0.1 * wave_age)))
            i = j + 2
        return out

    def efficiency(ws: list[tuple[float, float]]) -> float:
        if not ws:
            return 0.0
        wsum = sum(w for _, w in ws)
        if wsum <= EPS:
            return 0.0
        weighted_abs = sum(a * w for a, w in ws) / wsum
        return 1.0 / (weighted_abs + EPS)

    ae_buy = efficiency(waves(up=True))
    ae_sell = efficiency(waves(up=False))
    return safe_tanh((ae_buy - ae_sell) / max(ae_buy, ae_sell) + EPS * 0)


# =============================================================================
# FORMULA 24: Fractal Boundary Proximity Score (FBPS)
# =============================================================================
def fractal_boundary_levels(ctx) -> tuple[list[float], list[float]]:
    """Levels with exceptionally rough (low-Hölder) price behavior.

    Returns (boundaries_below, boundaries_above) relative to P(t), sorted by
    proximity.  Shared with Formula 29 (Dynamic Level Placement).
    """
    W_eff = ctx.W_eff
    if W_eff < 10:
        return [], []
    P = ctx.P[-(W_eff + 1):]
    p_min, p_max = min(P), max(P)
    if p_max <= p_min:
        return [], []
    grid_n = 100
    spacing = (p_max - p_min) / (grid_n - 1)
    levels = [p_min + k * spacing for k in range(grid_n)]

    # local Hölder exponent per level (delta = 2,3,5; delta=1 is singular)
    h_values: list[float] = []
    for L_ in levels:
        visits = [i for i, p in enumerate(P) if abs(p - L_) < spacing / 2]
        hs: list[float] = []
        for v in visits:
            acc = []
            for delta in (2, 3, 5):
                if v + delta >= len(P):
                    continue
                d = abs(P[v + delta] - P[v])
                if d < 1e-18:
                    continue
                acc.append(math.log(d) / math.log(delta))
            if acc:
                hs.append(mean(acc))
        h_values.append(mean(hs) if hs else float("nan"))

    valid = [(lv, hv) for lv, hv in zip(levels, h_values) if hv == hv]
    if len(valid) < 10:
        return [], []
    hs_sorted = sorted(hv for _, hv in valid)
    p20 = hs_sorted[max(0, int(0.2 * (len(hs_sorted) - 1)))]
    boundary_levels = [lv for lv, hv in valid if hv < p20]

    price = P[-1]
    below = sorted((lv for lv in boundary_levels if lv < price),
                   key=lambda x: price - x)
    above = sorted((lv for lv in boundary_levels if lv > price),
                   key=lambda x: x - price)
    return below, above


def f24_fbps(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 10:
        return 0.0
    P = ctx.P[-(W_eff + 1):]
    changes = [P[i] - P[i - 1] for i in range(1, len(P))]
    atr = mean([abs(c) for c in changes]) if changes else 0.0
    if atr <= EPS:
        return 0.0

    price = P[-1]
    below, above = fractal_boundary_levels(ctx)
    p_min, p_max = min(P), max(P)
    # natural fallbacks: window extremes act as outermost boundaries
    d_upper = ((above[0] - price) / atr) if above else max(0.0, (p_max - price) / atr)
    d_lower = ((price - below[0]) / atr) if below else max(0.0, (price - p_min) / atr)
    if not above:
        d_upper = d_upper if d_upper > 0 else 1.0
    if not below:
        d_lower = d_lower if d_lower > 0 else 1.0

    return safe_tanh((d_upper - d_lower) / (d_upper + d_lower + EPS))


# =============================================================================
# FORMULA 25: Temporal Momentum Density Mapping (TMDM)
# =============================================================================
def f25_tmdm(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 12:
        return 0.0
    P = ctx.P[-(W_eff + 1):]
    r = pct_returns(P)
    q_count = max(1, W_eff // 3)

    momenta: list[float] = []
    for q in range(q_count):
        start = q
        end = start + 3
        if end >= len(P):
            break
        seg_r = r[start:end]                  # 3 returns inside the sub-window
        sigma_local = std_pop(seg_r)
        if sigma_local < EPS:
            sigma_local = EPS
        momentum_q = (P[end] - P[start]) / (3.0 * sigma_local)
        momenta.append(momentum_q)

    Q = len(momenta)
    if Q < 3:
        return 0.0

    xs = [float(q + 1) for q in range(Q)]
    b = linear_slope(xs, momenta)
    a = mean(momenta) - b * mean(xs)
    horizon_steps = ctx.W // 3
    mu_projected = a + b * (Q + horizon_steps)

    fit = [a + b * x for x in xs]
    resid = [momenta[i] - fit[i] for i in range(Q)]
    sigma_momentum = std_pop(resid)
    if sigma_momentum < EPS:
        if abs(mu_projected) < EPS:
            return 0.0
        # perfectly linear momentum trend: saturate with the projected sign
        return safe_tanh(math.copysign(50.0, mu_projected))
    return safe_tanh(mu_projected / sigma_momentum)


# =============================================================================
# FORMULA 26: Inverse Frequency Dominance Score (IFDS)
# =============================================================================
def f26_ifds(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 12:
        return 0.0
    P = ctx.P[-(W_eff + 1):]
    r = pct_returns(P)
    W = len(r)
    if W < 8:
        return 0.0

    half = W // 2
    if half < 2:
        return 0.0
    # DFT power for k = 1..floor(W/2)
    powers: dict[int, float] = {}
    re_im: dict[int, complex] = {}
    for k in range(1, half + 1):
        acc = 0j
        for n_i, rv in enumerate(r):
            acc += rv * cmath.exp(-2j * math.pi * k * n_i / W)
        powers[k] = (abs(acc) ** 2) / W
        re_im[k] = acc

    low_k = list(range(1, max(1, W // 6) + 1))
    high_k = list(range(W // 3, half + 1))
    e_low = mean([powers[k] for k in low_k]) if low_k else 0.0
    e_high = mean([powers[k] for k in high_k]) if high_k else 0.0

    k_dom = max(range(1, half + 1), key=lambda k: powers[k])
    # evaluate the dominant component at the most recent time point
    phase = re_im[k_dom] * cmath.exp(2j * math.pi * k_dom * (W - 1) / W)
    sign_a = 1.0 if phase.real > 0 else (-1.0 if phase.real < 0 else 0.0)

    return safe_tanh((e_low - e_high) / (e_low + e_high + EPS) * sign_a)


# =============================================================================
# FORMULA 27: Volatility Clustering Decay Predictor (VCDP)
# =============================================================================
def analyze_vol_clusters(ctx) -> dict:
    """Cluster analysis over the window (used by F27 and the GRC commit)."""
    W_eff = ctx.W_eff
    if W_eff < 10:
        return {"in_cluster": False, "run": 0, "gap": 0, "ended": [],
                "cluster_mean_r": 0.0}
    P = ctx.P[-(W_eff + 1):]
    r = pct_returns(P)
    T = ctx.T[-(W_eff + 1):]
    abs_r = [abs(x) for x in r]
    med = median(abs_r[-50:]) if len(abs_r) >= 20 else median(abs_r)
    if med <= 0:
        med = median(abs_r) or EPS
    thresh = 1.5 * med
    flags = [a > thresh for a in abs_r]

    clusters: list[tuple[int, int]] = []     # (start_idx, end_idx) inclusive in r
    i = 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(flags) and flags[j + 1]:
            j += 1
        clusters.append((i, j))
        i = j + 2

    # trailing run status
    in_cluster = bool(flags) and flags[-1]
    run = 0
    if in_cluster:
        k = len(flags) - 1
        while k >= 0 and flags[k]:
            run += 1
            k -= 1

    gap = 0
    last_end_ts = None
    if clusters:
        last_start, last_end = clusters[-1]
        last_end_ts = T[last_end + 1] if last_end + 1 < len(T) else T[-1]
        if not in_cluster:
            gap = len(flags) - 1 - last_end     # minutes since cluster end

    ended: list[tuple[int, float, float]] = []  # (end_ts, duration_min, mean_r)
    for (s, e) in clusters[:-1] if in_cluster else clusters:
        end_ts = T[e + 1] if e + 1 < len(T) else T[-1]
        duration = (e - s + 2)                   # minutes covered by the wave
        mean_r = mean(r[s:e + 1])
        ended.append((end_ts, float(duration), mean_r))

    cluster_mean_r = 0.0
    if clusters:
        s, e = clusters[-1]
        cluster_mean_r = mean(r[s:e + 1])

    return {
        "in_cluster": in_cluster,
        "run": run,
        "gap": gap,
        "ended": ended,
        "cluster_mean_r": cluster_mean_r,
        "last_end_ts": last_end_ts or 0,
    }


def f27_vcdp(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 12:
        return 0.0
    tau_tf = float(ctx.W)
    info = analyze_vol_clusters(ctx)
    hist = ctx.hist

    if info["in_cluster"]:
        durations = hist.vol_cluster_durations.get(ctx.symbol, [])
        age = float(info["run"])
        if len(durations) >= 3:
            n = len(durations)
            lam = n / sum(durations)              # exponential MLE
            mean_dur = sum(durations) / n
            tau_remaining = max(
                1.0,
                (1.0 / lam) * math.exp(-lam * age / mean_dur) * mean_dur,
            )
        else:
            tau_remaining = max(1.0, 0.5 * tau_tf)   # bootstrap: neutral
        trend_sign = sign(info["cluster_mean_r"])
    else:
        gaps = hist.vol_cluster_gaps.get(ctx.symbol, [])
        durations = hist.vol_cluster_durations.get(ctx.symbol, [])
        if len(gaps) >= 3 and durations:
            lam_gap = len(gaps) / sum(gaps)
            mean_dur = sum(durations) / len(durations)
            p_new = 1.0 - math.exp(-lam_gap * tau_tf)
            tau_remaining = p_new * mean_dur
        else:
            tau_remaining = 0.5 * tau_tf             # bootstrap: neutral
        # most recent cluster direction
        r = pct_returns(ctx.P[-(W_eff + 1):])
        trend_sign = sign(info["cluster_mean_r"]) if info["cluster_mean_r"] else 0

    if trend_sign == 0:
        return 0.0
    ratio = tau_remaining / tau_tf
    return safe_tanh((ratio - 0.5) * 2.0 * trend_sign)


# =============================================================================
# FORMULA 31: Causal Loop Retrodiction Engine (CLRE)
# =============================================================================
def _ar_fit(series: list[float], p: int) -> tuple[float, list[float]]:
    """Least-squares AR(p) with a tiny ridge for stability: returns (c, a)."""
    n = len(series)
    if n <= p + 1:
        return 0.0, [0.0] * p
    # normal equations for [c, a_1..a_p]
    dim = p + 1
    # build X'X and X'y
    xtx = [[0.0] * dim for _ in range(dim)]
    xty = [0.0] * dim
    for s in range(p, n):
        row = [1.0] + [series[s - l] for l in range(1, p + 1)]
        y = series[s]
        for a_i in range(dim):
            for b_i in range(dim):
                xtx[a_i][b_i] += row[a_i] * row[b_i]
            xty[a_i] += row[a_i] * y
    for i in range(dim):
        xtx[i][i] += 1e-8 * max(1.0, xtx[i][i])
    # Gaussian elimination
    for col in range(dim):
        piv = max(range(col, dim), key=lambda rr: abs(xtx[rr][col]))
        if abs(xtx[piv][col]) < 1e-18:
            continue
        xtx[col], xtx[piv] = xtx[piv], xtx[col]
        xty[col], xty[piv] = xty[piv], xty[col]
        for rr in range(col + 1, dim):
            f = xtx[rr][col] / xtx[col][col]
            for cc in range(col, dim):
                xtx[rr][cc] -= f * xtx[col][cc]
            xty[rr] -= f * xty[col]
    sol = [0.0] * dim
    for rr in range(dim - 1, -1, -1):
        if abs(xtx[rr][rr]) < 1e-18:
            continue
        acc = xty[rr]
        for cc in range(rr + 1, dim):
            acc -= xtx[rr][cc] * sol[cc]
        sol[rr] = acc / xtx[rr][rr]
    return sol[0], sol[1:]


def _ar_iterate(history: list[float], c: float, a: list[float], steps: int) -> list[float]:
    """Iterate the AR model forward `steps` times from the tail of `history`."""
    buf = list(history)
    p = len(a)
    out: list[float] = []
    for _ in range(steps):
        nxt = c + sum(a[l - 1] * buf[-l] for l in range(1, p + 1))
        buf.append(nxt)
        out.append(nxt)
    return out


def _retro_pair(P: list[float], p: int) -> tuple[list[float], list[float]] | None:
    """Retrodict the first half of P from the structure of the second half.

    Returns (actual, retrodicted) aligned in ascending time for the period
    [t-W .. t-W/2], or None when the data is too short.
    """
    half = max(p + 2, len(P) // 2)
    if half >= len(P) or len(P) - half < 5:
        return None
    retro_src = P[-half:]                       # second half, oldest..newest
    r_rev = list(reversed(retro_src))           # newest..oldest (reversed time)
    c_r, a_r = _ar_fit(r_rev, p)

    # iterate the reversed model far enough to cover reversed indices
    # half .. len(P)-1  (which map to original indices len(P)-1-half .. 0)
    steps = len(P) - p
    preds = _ar_iterate(r_rev[:p], c_r, a_r, steps)   # preds[k] -> rev idx p+k

    segment: list[tuple[int, float]] = []
    for k in range(half - p, len(P) - p):
        q = p + k                                # reversed index
        orig = len(P) - 1 - q
        if orig < 0:
            break
        segment.append((orig, preds[k]))
    if len(segment) < 5:
        return None
    segment.reverse()                            # ascending original time
    act = [P[orig] for orig, _ in segment]
    ret = [val for _, val in segment]
    return act, ret


def f31_clre(ctx) -> float:
    W_eff = ctx.W_eff
    if W_eff < 20:
        return 0.0
    P = ctx.P[-(W_eff + 1):]
    p = min(10, max(3, W_eff // 5))
    if len(P) <= p + 2:
        return 0.0

    # Step 1: forward prediction over the timeframe horizon
    c_f, a_f = _ar_fit(P, p)
    forward = _ar_iterate(P, c_f, a_f, max(1, ctx.W))
    p_forward = forward[-1]
    p_now = P[-1]

    # Step 2+3: retrodiction anomaly
    pair = _retro_pair(P, p)
    if pair is None:
        return 0.0
    act, ret = pair
    r_anomaly = sum(((a_ - b_) / (a_ + EPS)) ** 2 for a_, b_ in zip(act, ret))
    r_anomaly /= (W_eff / 2.0)                  # 1/(W/2) normalization per spec

    hist_ring = ctx.hist.ring(
        ctx.hist.tf_key(ctx.symbol, ctx.tf, "ranom"), HIST_R_ANOMALY
    )
    if len(hist_ring) < 5:
        return 0.0                              # bootstrap: neutral
    r_base = hist_ring.mean()
    sigma_r = hist_ring.std()
    if sigma_r < 1e-12:
        sigma_r = 1e-12

    d_retro = sign(p_forward - p_now) * sign(mean([b_ - a_ for a_, b_ in zip(act, ret)]))
    if d_retro == 0:
        return 0.0
    return safe_tanh((r_anomaly - r_base) / sigma_r * d_retro)


def clre_anomaly(ctx) -> float:
    """The R_anomaly intermediate for GRC history recording (F31)."""
    W_eff = ctx.W_eff
    if W_eff < 20:
        return 0.0
    P = ctx.P[-(W_eff + 1):]
    p = min(10, max(3, W_eff // 5))
    pair = _retro_pair(P, p)
    if pair is None:
        return 0.0
    act, ret = pair
    r_anomaly = sum(((a_ - b_) / (a_ + EPS)) ** 2 for a_, b_ in zip(act, ret))
    return r_anomaly / (W_eff / 2.0)


# =============================================================================
# FORMULA 30: Prediction Convergence Validator (PCV)
#   Independent Permutation-Entropy directional prediction compared with the
#   CCI.  Runs AFTER aggregation (it needs the CCI as input) and its negative
#   case reduces confidence by |PCV| * 0.3 with a "CONFLICTING SIGNALS" flag.
# =============================================================================
def f30_pcv(prices: list[float], cci: float, m: int = 4) -> dict:
    if len(prices) < m + 2:
        return {"pcv": 0.0, "direction": 0.0, "agreement": 1.0,
                "conflict": False}
    # ordinal patterns of order m: pattern[i] = rank of i-th element
    freq = [0] * 24
    total = 0
    for s in range(0, len(prices) - m + 1):
        window = prices[s:s + m]
        order = sorted(range(m), key=lambda i: (-window[i], i))
        ranks = [0] * m
        for rank_pos, idx in enumerate(order):
            ranks[idx] = m - rank_pos           # highest value -> rank m
        freq[_perm_rank(tuple(ranks))] += 1
        total += 1
    if total == 0:
        return {"pcv": 0.0, "direction": 0.0, "agreement": 1.0,
                "conflict": False}

    dir_sum = 0.0
    for j in range(24):
        if freq[j] == 0:
            continue
        ranks = _perm_unrank(j, m)
        dir_score = (ranks[m - 1] - ranks[0]) / (m - 1)
        dir_sum += (freq[j] / total) * dir_score

    cci_sign = sign(cci)
    dir_sign = sign(dir_sum)
    alignment = float(dir_sign * cci_sign)
    agreement = 1.0 - abs(dir_sum - cci) / 2.0
    pcv = agreement * alignment
    return {
        "pcv": pcv,
        "direction": dir_sum,
        "agreement": agreement,
        "conflict": pcv < 0,
    }


def _perm_rank(ranks: tuple[int, ...]) -> int:
    """Canonical lexicographic index of a permutation of 1..m."""
    m = len(ranks)
    avail = list(range(1, m + 1))
    idx = 0
    for i, r in enumerate(ranks):
        pos = avail.index(r)
        idx = idx * (m - i) + pos
        avail.pop(pos)
    return min(idx, 23)


def _perm_unrank(idx: int, m: int) -> list[int]:
    avail = list(range(1, m + 1))
    out: list[int] = []
    for i in range(m):
        divisor = m - i - 1
        base = factorial(divisor) if divisor > 0 else 1
        pos = idx // base
        idx %= base
        out.append(avail.pop(min(pos, len(avail) - 1)))
    return out


def factorial(n: int) -> int:
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r
