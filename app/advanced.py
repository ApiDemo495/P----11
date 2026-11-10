"""Advanced mathematics layer (user-approved 2026-11-09, items 3-7,
9, 11-16, 20-21 of the research questionnaire).

Every module here was individually approved by the user after a
plain-language explanation. Design rules honoured throughout:

  * NOTHING is removed or silenced — every existing voter keeps its
    vote; these modules ADD votes / soft multipliers on top.
  * Soft gates only SCALE confidence/size in [0.45, 1.0]; they can
    never veto a side and never create one.
  * Causal only: every function reads bars up to 'now' — no future
    bar, no centred windows, no laggy symmetric smoothers.
  * Thin data => small weight, printed reason — never a fake number.

Modules (approval order):
  bocpd      Bayesian online changepoint (run-length posterior) —
             upgrades the _learners display clock into a voter:
             trend of the CURRENT regime segment; soft gate.
  fracdiff   fractional differentiation (1-B)^d feed + its voter.
  entropy    permutation entropy + Lempel-Ziv of signs + sign
             Markov voter; predictability soft gate.
  ou_hurst   Ornstein-Uhlenbeck half-life + DFA Hurst regime;
             fade/trend voter by regime; soft gate in the dead zone.
  kalman     level+velocity state-space filter; velocity voter.
  jumps      bipower variation / Lee-Mykland jump split; the
             post-jump retrace voter.
  pairs_ou   cointegration spread vs BTC with OU half-life voter.
  wavelet    MODWT (Haar, a-trous) scale votes.
  har_rv     HAR realised-vol forecast (30m / 4h / 24h intraday
             horizons, user-specified) — sizing layer, no direction.
  evt        POT/GPD tail quantiles — sizing dampener + honest
             catastrophic-stop scale, never a direction voter.
  rmt        random-matrix market mode (Marchenko-Pastur) — risk
             readout + capped residual voter (N=4 => modest).
  rqa        recurrence determinism — CONFIDENCE MULTIPLIER ONLY
             (user directive), never a signed vote.
  signature  truncated path signature (levels 1-2) — STRICT floor
             weight until >=500 walk-forward graded samples prove
             it (user directive); grading lives in the engine.
  conformal  adaptive conformal band state (ACI) — the engine owns
             the per-(asset,TF) state; helpers here update/read it.

`advanced_block()` is the single entry point the engine calls.
"""
from __future__ import annotations

import math

EPS = 1e-12


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _rets(P: list[float]) -> list[float]:
    return [(P[i] - P[i - 1]) / P[i - 1]
            for i in range(1, len(P)) if P[i - 1]]


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _tanh(x: float) -> float:
    return math.tanh(max(-8.0, min(8.0, x)))


# ---------------------------------------------------------------- fracdiff
def fracdiff_weights(d: float, n: int, thresh: float = 1e-4) -> list[float]:
    w = [1.0]
    for k in range(1, min(n, 64)):       # fixed-width window (de Prado)
        wk = -w[-1] * (d - k + 1) / k
        if abs(wk) < thresh:
            break
        w.append(wk)
    return w


def fracdiff_series(P: list[float], d: float = 0.4) -> list[float]:
    """Stationary-with-memory transform of log price."""
    if len(P) < 30:
        return []
    lp = [math.log(p) if p > 0 else 0.0 for p in P]
    w = fracdiff_weights(d, len(lp))
    out = []
    for i in range(len(w) - 1, len(lp)):
        out.append(sum(w[k] * lp[i - k] for k in range(len(w))))
    return out


def fracdiff_vote(P: list[float]) -> dict:
    fd = fracdiff_series(P)
    if len(fd) < 20:
        return {"vote": 0.0, "why": "thin"}
    seg = fd[-22:]
    slope = (seg[-1] - seg[0]) / (len(seg) - 1)
    sd = _std([seg[i] - seg[i - 1] for i in range(1, len(seg))]) or EPS
    return {"vote": round(_tanh(slope / sd * 2.0), 4),
            "d": 0.4, "n": len(fd)}


# ---------------------------------------------------------------- kalman
def kalman_vote(P: list[float]) -> dict:
    """Level+velocity Kalman filter; gain adapts via innovation
    variance ratio (q/r estimated online from the tape itself)."""
    if len(P) < 12:
        return {"vote": 0.0, "why": "thin"}
    lp = [math.log(p) if p > 0 else 0.0 for p in P[-(min(len(P), 300)):]]
    lvl, vel = lp[0], 0.0
    p11, p12, p22 = 1e-4, 0.0, 1e-4
    r = max(1e-10, _std([lp[i] - lp[i - 1] for i in range(1, len(lp))]) ** 2)
    q = r * 0.02
    innov_sd = 0.0
    for x in lp[1:]:
        # predict
        lvl_p, vel_p = lvl + vel, vel
        p11p = p11 + 2 * p12 + p22 + q
        p12p = p12 + p22
        p22p = p22 + q * 0.1
        innov = x - lvl_p
        s = p11p + r
        k1, k2 = p11p / s, p12p / s
        lvl, vel = lvl_p + k1 * innov, vel_p + k2 * innov
        p11, p12, p22 = (1 - k1) * p11p, (1 - k1) * p12p, p22p - k2 * p12p
        innov_sd = 0.95 * innov_sd + 0.05 * abs(innov)
    sd = (innov_sd or EPS)
    return {"vote": round(_tanh(vel / (sd + EPS) * 1.5), 4),
            "vel_bps_bar": round(vel * 1e4, 3)}


# ---------------------------------------------------------------- bocpd
def bocpd_vote(P: list[float], hazard: float = 1.0 / 60.0) -> dict:
    """Bayesian online changepoint detection (Adams & MacKay 2007)
    with the conjugate Normal-Inverse-Gamma posterior — unknown
    mean AND variance per segment, Student-t predictive. The vote
    is the posterior-weighted segment drift; the soft gate tightens
    when the changepoint probability (run length ~ 0) is high."""
    r = _rets(P[-170:])
    if len(r) < 25:
        return {"vote": 0.0, "gate": 1.0, "why": "thin"}
    MAXRL = 120
    # NIG prior: mu0=0, kappa0=1, alpha0=2, beta0 from global var
    var0 = (_std(r) ** 2) or 1e-8
    # per-run-length posterior params (lists, index = run length)
    mu = [0.0]; kap = [1.0]; alp = [2.0]; bet = [var0]
    post = [1.0]
    p_cp = 0.0
    exp_drift = 0.0
    for x in r:
        # Student-t predictive for each run length
        new_post = [0.0] * (min(len(post) + 1, MAXRL))
        nmu = [0.0] * len(new_post); nkap = [1.0] * len(new_post)
        nalp = [2.0] * len(new_post); nbet = [var0] * len(new_post)
        # changepoint: all mass -> run length 0 under the prior
        cp_mass = sum(post) * hazard
        new_post[0] = cp_mass
        nmu[0], nkap[0], nalp[0], nbet[0] = x, 2.0, 2.5, var0 + 0.0
        growth_total = 0.0
        for rl in range(len(post)):
            if post[rl] <= 0:
                continue
            nu = 2 * alp[rl]
            scale2 = bet[rl] * (kap[rl] + 1) / (alp[rl] * kap[rl])
            # Student-t log pdf
            z = (x - mu[rl]) ** 2 / (nu * scale2)
            logp = (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
                    - 0.5 * math.log(nu * math.pi * scale2)
                    - (nu + 1) / 2 * math.log1p(z))
            w = post[rl] * (1 - hazard) * math.exp(max(-30.0, logp))
            nrl = rl + 1
            if nrl >= MAXRL:
                nrl = MAXRL - 1
            new_post[nrl] += w
            growth_total += w
            # NIG update with this observation
            k_old = kap[rl]
            nkap[nrl] = k_old + 1
            nmu[nrl] = (k_old * mu[rl] + x) / (k_old + 1)
            nalp[nrl] = alp[rl] + 0.5
            nbet[nrl] = bet[rl] + (k_old * (x - mu[rl]) ** 2) / (2 * (k_old + 1))
        tot = sum(new_post) or EPS
        post = [v / tot for v in new_post]
        mu, kap, alp, bet = nmu, nkap, nalp, nbet
        p_cp = post[0] if post else 0.0
        exp_drift = sum(post[rl] * mu[rl] for rl in range(len(post)))
    sd = _std(r) or EPS
    vote = _tanh(exp_drift / (sd + EPS) * 8.0)
    gate = 1.0 - 0.5 * min(1.0, p_cp * 4.0)
    best_rl = max(range(len(post)), key=lambda k: post[k])
    return {"vote": round(vote, 4), "gate": round(gate, 3),
            "run_len": int(best_rl), "p_break": round(float(p_cp), 4)}


# ---------------------------------------------------------------- entropy
def _perm_entropy(r: list[float], m: int = 3) -> float:
    if len(r) < m + 5:
        return 1.0
    from itertools import permutations
    idx = {p: i for i, p in enumerate(permutations(range(m)))}
    counts = [0] * len(idx)
    for i in range(len(r) - m + 1):
        w = r[i:i + m]
        order = tuple(sorted(range(m), key=lambda j: w[j]))
        counts[idx[order]] += 1
    tot = sum(counts) or 1
    H = -sum((c / tot) * math.log(c / tot) for c in counts if c)
    return H / math.log(len(idx))


def _lz_complexity(signs: list[int]) -> float:
    s = "".join("1" if x > 0 else "0" for x in signs if x)
    n = len(s)
    if n < 10:
        return 1.0
    seen = set()
    i, c = 0, 0
    while i < n:
        j = i + 1
        while j <= n and s[i:j] in seen:
            j += 1
        seen.add(s[i:j] if j <= n else s[i:])
        c += 1
        i = j if j <= n else n
    return c * math.log2(n) / n if n else 1.0    # ~1 = incompressible


def entropy_block(P: list[float]) -> dict:
    """Predictability meter + sign-Markov voter. The voter: order-2
    Markov chain over return signs estimated inside the window —
    P(next = majority continuation) mapped to a vote. Entropy/LZ
    scale the SOFT GATE, never the side."""
    r = _rets(P[-140:])
    if len(r) < 40:
        return {"vote": 0.0, "gate": 1.0, "why": "thin"}
    signs = [_sign(x) for x in r]
    H = _perm_entropy(r)
    lz = _lz_complexity(signs)
    predictability = max(0.0, min(1.0, 1.0 - (H * 0.6 + lz * 0.4)))
    # order-2 sign Markov
    trans: dict[tuple, list[int]] = {}
    for i in range(2, len(signs)):
        trans.setdefault((signs[i - 2], signs[i - 1]), []).append(signs[i])
    hist = trans.get((signs[-2], signs[-1]), [])
    vote = 0.0
    if len(hist) >= 4:
        vote = _tanh((sum(hist) / len(hist)) * 1.6) * (0.35 + 0.65 * predictability)
    gate = 0.55 + 0.45 * predictability
    return {"vote": round(vote, 4), "gate": round(gate, 3),
            "perm_entropy": round(H, 3), "lz": round(lz, 3),
            "predictability": round(predictability, 3)}


# ---------------------------------------------------------------- ou/hurst
def _dfa_hurst(r: list[float]) -> float:
    n = len(r)
    if n < 64:
        return 0.5
    y = []
    m = sum(r) / n
    acc = 0.0
    for x in r:
        acc += x - m
        y.append(acc)
    scales = [s for s in (8, 16, 32, 64) if s <= n // 2]
    if len(scales) < 2:
        return 0.5
    Fs = []
    for s in scales:
        vals = []
        for st in range(0, n - s + 1, s):
            seg = y[st:st + s]
            # linear detrend
            xs = list(range(s))
            mx = (s - 1) / 2.0
            my = sum(seg) / s
            den = sum((x - mx) ** 2 for x in xs) or EPS
            slope = sum((xs[i] - mx) * (seg[i] - my) for i in range(s)) / den
            inter = my - slope * mx
            vals.append((sum((seg[i] - (slope * xs[i] + inter)) ** 2
                             for i in range(s)) / s) ** 0.5)
        Fs.append(sum(vals) / len(vals) if vals else EPS)
    # H = slope of log F vs log scale
    lx = [math.log(s) for s in scales]
    ly = [math.log(max(f, EPS)) for f in Fs]
    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)
    den = sum((x - mx) ** 2 for x in lx) or EPS
    H = sum((lx[i] - mx) * (ly[i] - my) for i in range(len(lx))) / den
    return max(0.05, min(0.95, H))


def ou_hurst_block(P: list[float], W: int) -> dict:
    r = _rets(P)
    if len(r) < 70:
        return {"vote": 0.0, "gate": 1.0, "why": "thin"}
    lp = [math.log(p) if p > 0 else 0.0 for p in P[-(min(len(P), 260)):]]
    # OU fit via AR(1) on log price: x_t = a + b x_{t-1}
    xs, ys = lp[:-1], lp[1:]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs) or EPS
    b = sum((xs[i] - mx) * (ys[i] - my) for i in range(len(xs))) / den
    b = min(0.999, max(0.90, b))
    kappa = -math.log(b)
    half_life = math.log(2) / kappa if kappa > EPS else 9999.0
    theta = my - b * mx
    mu = theta / (1 - b) if (1 - b) > EPS else lp[-1]
    sd_lp = _std([lp[i] - lp[i - 1] for i in range(1, len(lp))]) or EPS
    z = (lp[-1] - mu) / (sd_lp * (max(1.0, half_life) ** 0.5) + EPS)
    H = _dfa_hurst(r[-160:])
    if H < 0.45:
        regime = "reverting"
        vote = -_tanh(z * 1.4) * min(1.0, W / max(half_life, 1.0))
        gate = 1.0
    elif H > 0.55:
        regime = "trending"
        vote = _tanh((lp[-1] - lp[-min(22, len(lp) - 1) - 1])
                     / (sd_lp * 10 + EPS))
        gate = 1.0
    else:
        regime = "dead-zone"
        vote = -_tanh(z) * 0.4
        gate = 0.6
    return {"vote": round(vote, 4), "gate": gate, "hurst": round(H, 3),
            "half_life_bars": round(half_life, 1), "z": round(z, 3),
            "regime": regime}


# ---------------------------------------------------------------- jumps
def jumps_vote(P: list[float]) -> dict:
    r = _rets(P[-130:])
    if len(r) < 40:
        return {"vote": 0.0, "why": "thin"}
    # bipower variation (robust to jumps) vs realised variance
    absr = [abs(x) for x in r]
    bv = (math.pi / 2) * sum(absr[i] * absr[i - 1]
                             for i in range(1, len(absr))) / len(r)
    rv = sum(x * x for x in r) / len(r)
    jump_share = max(0.0, (rv - bv) / rv) if rv > 0 else 0.0
    # Lee-Mykland style: last significant jump within 8 bars
    vote = 0.0
    found = ""
    for back in range(0, 8):
        i = len(r) - 1 - back
        if i < 5:
            break
        seg = absr[max(1, i - 20):i]
        # local bipower variation: jump-robust local sigma
        local = ((math.pi / 2) * sum(seg[k] * seg[k - 1]
                 for k in range(1, len(seg))) / max(1, len(seg))) ** 0.5 \
            if len(seg) > 3 else (absr[i - 1] if i else EPS)
        stat = abs(r[i]) / (local + EPS)
        if stat > 3.0:
            # jumps partially retrace, scaled by recency decay
            vote = -_sign(r[i]) * min(1.0, (stat - 3.0) / 3.0 + 0.4) \
                * (0.85 ** back)
            found = f"jump {back}b ago z={stat:.1f}"
            break
    return {"vote": round(vote, 4), "jump_share": round(jump_share, 3),
            "why": found}


# ---------------------------------------------------------------- pairs OU
def pairs_vote(symbol: str, P: list[float],
               others: dict[str, list[float]]) -> dict:
    ref = others.get("BTC")
    if symbol == "BTC" or not ref or len(P) < 60 or len(ref) < 60:
        return {"vote": 0.0, "why": "no ref"}
    n = min(len(P), len(ref), 240)
    x = [math.log(v) for v in ref[-n:] if v > 0]
    y = [math.log(v) for v in P[-n:] if v > 0]
    n = min(len(x), len(y))
    if n < 60:
        return {"vote": 0.0, "why": "thin"}
    x, y = x[-n:], y[-n:]
    mx = sum(x) / n
    my = sum(y) / n
    den = sum((v - mx) ** 2 for v in x) or EPS
    beta = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / den
    spread = [y[i] - beta * x[i] for i in range(n)]
    # OU half-life of the spread
    sx, sy = spread[:-1], spread[1:]
    msx = sum(sx) / len(sx)
    msy = sum(sy) / len(sy)
    den2 = sum((v - msx) ** 2 for v in sx) or EPS
    b = sum((sx[i] - msx) * (sy[i] - msy) for i in range(len(sx))) / den2
    if b <= 0 or b >= 0.999:
        return {"vote": 0.0, "why": "spread not stationary"}
    hl = -math.log(2) / math.log(b)
    ms = sum(spread) / n
    sds = _std(spread) or EPS
    z = (spread[-1] - ms) / sds
    vote = -_tanh(z / 1.6) * min(1.0, 130.0 / max(hl, 1.0))
    return {"vote": round(vote, 4), "z": round(z, 2),
            "half_life_bars": round(hl, 1), "beta": round(beta, 3)}


# ---------------------------------------------------------------- wavelet
def wavelet_vote(P: list[float]) -> dict:
    """A-trous Haar MODWT on log price: at each scale (2/4/8/16 bars)
    the approximation (smoothed price) slope over the last scale
    bars votes; coarse scales = slower trend, fine = fast trend.
    Agreement across scales stacks, conflict shrinks — printed."""
    if len(P) < 40:
        return {"vote": 0.0, "why": "thin"}
    cur = [math.log(v) if v > 0 else 0.0 for v in P[-140:]]
    sd = _std(_rets(P[-140:])) or EPS
    votes = []
    step = 1
    for level in range(4):
        nxt = []
        for i in range(len(cur)):
            j = i - step
            prev = cur[j] if j >= 0 else cur[0]
            nxt.append((cur[i] + prev) / 2.0)
        k = min(len(nxt) - 1, max(2, step * 2))
        slope = (nxt[-1] - nxt[-1 - k]) / k
        votes.append(slope / (sd + EPS))
        cur = nxt
        step *= 2
    num = sum(v / (i + 1) for i, v in enumerate(votes))
    den = sum(1.0 / (i + 1) for i in range(len(votes)))
    return {"vote": round(_tanh((num / den) * 1.6), 4),
            "scales": [round(float(v), 2) for v in votes]}


# ---------------------------------------------------------------- HAR-RV
def har_rv(Pfull: list[float]) -> dict:
    """Intraday HAR: RV over 30m / 4h / 24h lags forecasting the
    next-window per-bar sigma (bars are 1m on the live tape)."""
    r = _rets(Pfull[-1500:])
    if len(r) < 200:
        return {"sigma_bar": None, "why": "thin"}

    def rv(k):
        seg = r[-k:] if len(r) >= k else r
        return (sum(x * x for x in seg) / max(1, len(seg))) ** 0.5
    s30, s240, s1440 = rv(30), rv(240), rv(1440)
    sig = 0.45 * s30 + 0.35 * s240 + 0.20 * s1440
    calm = rv(1440) or sig or EPS
    size = max(0.45, min(1.0, (calm / (sig + EPS)) ** 0.5)) if sig else 1.0
    return {"sigma_bar": round(sig, 6),
            "rv30": round(s30, 6), "rv4h": round(s240, 6),
            "rv24h": round(s1440, 6), "size_mult": round(size, 3)}


# ---------------------------------------------------------------- EVT tail
def evt_tail(Pfull: list[float]) -> dict:
    """POT/GPD on |returns|: honest 1-in-100 window-move quantile +
    sizing dampener. Hill-style shape estimate on exceedances."""
    r = [abs(x) for x in _rets(Pfull[-1500:])]
    if len(r) < 300:
        return {"q99_bps": None, "why": "thin"}
    srt = sorted(r)
    thr = srt[int(0.95 * len(srt))]
    exc = [x - thr for x in r if x > thr]
    if len(exc) < 25:
        return {"q99_bps": None, "why": "thin tail"}
    # GPD fit by probability-weighted moments (Hosking & Wallis):
    # b0 = E[X], b1 = E[X * F(X)] over exceedances (ascending order)
    exc_s = sorted(exc)
    m = len(exc_s)
    if m < 2:
        return {"q99_bps": None, "why": "thin tail"}
    b0 = sum(exc_s) / m
    b1 = sum((i / (m - 1)) * exc_s[i] for i in range(m)) / m
    if b1 <= EPS or b0 <= EPS:
        return {"q99_bps": None, "why": "degenerate tail"}
    ratio = b0 / b1
    if abs(2.0 - ratio) < 1e-6:
        return {"q99_bps": None, "why": "degenerate tail"}
    k_hw = (4.0 - ratio) / (2.0 - ratio)     # Hosking shape (=-xi)
    xi = max(-0.2, min(0.6, -k_hw))          # EVT shape (fat tail > 0)
    beta = max(EPS, b0 * (1.0 + xi))
    # POT return level: P(X > thr + y) = (Nexc/n)(1 + xi y/beta)^(-1/xi)
    n_all, p_target = len(r), 0.01
    if abs(xi) > 1e-6:
        q = thr + (beta / xi) * (((len(exc) / n_all) / p_target) ** xi - 1)
    else:
        q = thr + beta * math.log((len(exc) / n_all) / p_target)
    median_r = srt[len(srt) // 2] or EPS
    size = max(0.4, min(1.0, (3.0 * median_r) / (q + EPS)))
    return {"q99_bps": round(q * 1e4, 1), "xi": round(xi, 3),
            "thr_bps": round(thr * 1e4, 2), "n_exc": len(exc),
            "size_mult": round(size, 3),
            "cat_stop_bps": round(q * 1e4 * 1.25, 1)}


# ---------------------------------------------------------------- RMT mode
def rmt_block(series: dict[str, list[float]]) -> dict:
    """Market mode from the 4-asset correlation matrix (equal-time
    returns). Marchenko-Pastur edge for N=4, T bars."""
    syms = [s for s in ("BTC", "ETH", "BNB", "PAXG") if s in series]
    if len(syms) < 3:
        return {"mode_share": None, "why": "thin"}
    rs = {}
    T = None
    for s in syms:
        r = _rets(series[s][-200:])
        rs[s] = r
        T = len(r) if T is None else min(T, len(r))
    if not T or T < 40:
        return {"mode_share": None, "why": "thin"}
    import numpy as _np
    M = _np.array([rs[s][-T:] for s in syms], dtype=float)
    M = (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + EPS)
    C = (M @ M.T) / T
    ev = sorted(_np.linalg.eigvalsh(C).tolist(), reverse=True)
    lam_max = ev[0]
    share = lam_max / (sum(ev) + EPS)
    mp_edge = (1 + (len(syms) / T) ** 0.5) ** 2
    # market mode = first principal component direction (all-positive
    # loadings for a true market mode)
    mode_ret = M.mean(axis=0)
    out = {"mode_share": round(float(share), 3),
           "lambda_max": round(float(lam_max), 3),
           "mp_edge": round(float(mp_edge), 3),
           "residual_vote": {}}
    for i, s in enumerate(syms):
        resid = M[i] - mode_ret * (float(_np.corrcoef(M[i], mode_ret)[0, 1])
                                   if T > 3 else 0.0)
        tail = resid[-22:]
        out["residual_vote"][s] = round(
            _tanh(float(tail.mean()) * 6.0), 4)
    return out


# ---------------------------------------------------------------- RQA
def rqa_multiplier(P: list[float]) -> dict:
    """Recurrence determinism of delay-embedded returns (m=3, tau=1).
    CONFIDENCE MULTIPLIER ONLY — no signed vote (user directive)."""
    r = _rets(P[-110:])
    if len(r) < 60:
        return {"mult": 1.0, "why": "thin"}
    m, tau = 3, 1
    pts = [tuple(r[i:i + m * tau:tau]) for i in range(len(r) - m * tau)]
    n = len(pts)
    if n < 40:
        return {"mult": 1.0, "why": "thin"}
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(max(abs(pts[i][k] - pts[j][k]) for k in range(m)))
    dists.sort()
    eps_r = dists[int(0.10 * len(dists))] or EPS    # 10% recurrence rate
    rec = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and max(abs(pts[i][k] - pts[j][k])
                              for k in range(m)) <= eps_r:
                rec[i][j] = 1
    # determinism: share of recurrent points in diagonals >= 3
    det_pts = 0
    tot = 0
    for i in range(n):
        run = 0
        for j in range(n):
            tot += rec[i][j]
            if rec[i][j]:
                run += 1
            else:
                if run >= 3:
                    det_pts += run
                run = 0
        if run >= 3:
            det_pts += run
    det = det_pts / tot if tot else 0.0
    mult = 0.75 + 0.4 * det
    return {"mult": round(mult, 3), "determinism": round(det, 3)}


# ---------------------------------------------------------------- signature
def signature_vote(P: list[float], V: list[float]) -> dict:
    """Truncated signature (levels 1-2) of the (price, volume) path:
    displacement + Levy area. STRICT floor weight until the engine
    has >=500 walk-forward graded samples for it (user directive)."""
    n = min(len(P), len(V), 90)
    if n < 25:
        return {"vote": 0.0, "why": "thin"}
    p = [math.log(v) if v > 0 else 0.0 for v in P[-n:]]
    v = list(V[-n:])
    mv = sum(v) / len(v) or 1.0
    v = [x / mv for x in v]
    dp = [p[i] - p[i - 1] for i in range(1, n)]
    dv = [v[i] - v[i - 1] for i in range(1, n)]
    disp = p[-1] - p[0]
    # Levy area of the (price, volume) path: 0.5 * sum(p dv - v dp)
    area = 0.0
    for i in range(len(dp)):
        area += 0.5 * (p[i] * dv[i] - v[i] * dp[i])
    sd = _std(dp) or EPS
    vote = _tanh(disp / (sd * (n ** 0.5)) + area * 0.002)
    return {"vote": round(vote, 4), "disp": round(disp, 5),
            "area": round(area, 3), "weight_cap": 0.15}


# ---------------------------------------------------------------- conformal
def conformal_update(state: dict, abs_ret: float, alpha: float = 0.2,
                     gamma: float = 0.02) -> dict:
    """ACI (Gibbs-Candes): adapt the band half-width q so the
    realised exceedance rate tracks alpha (80% band => alpha .2)."""
    q = float(state.get("q", 0.0))
    if q <= 0:
        q = max(abs_ret, 1e-4)
    exceeded = 1.0 if abs_ret > q else 0.0
    q = q * (1.0 + gamma * (exceeded - alpha) * 4.0)
    q = max(1e-5, min(0.25, q))
    return {"q": q, "n": int(state.get("n", 0)) + 1,
            "hits": int(state.get("hits", 0)) + (0 if exceeded else 1)}


# ---------------------------------------------------------------- composite
MODULE_WEIGHTS = {
    "kalman": 1.00, "fracdiff": 1.00, "ou_hurst": 1.00,
    "wavelet": 0.90, "jumps": 0.80, "pairs_ou": 0.80,
    "bocpd": 0.70, "entropy": 0.70, "rmt_residual": 0.35,
    "signature": 0.15,          # floor until validated (engine raises)
}


def advanced_block(symbol: str, tf: str, ctx, others: dict | None = None,
                   full_closes: list[float] | None = None,
                   sig_weight: float = 0.15,
                   rmt: dict | None = None,
                   weight_scales: dict | None = None) -> dict:
    """Single engine entry point. Returns votes, soft gates and
    sizing layers; every field safe to print in the UI/API."""
    P = list(ctx.P or [])
    V = list(ctx.V or [])
    W = int(getattr(ctx, "W", 22) or 22)
    others = others or {}
    # price-series modules run on the full tape (up to ~1500 bars of
    # 1m history), not just the W+1 window slice — fracdiff/Hurst/OU
    # on 23 bars would be numerology, and they say so ('thin') when
    # the tape itself is short
    tape = list(full_closes or P)[-280:]
    mods: dict[str, dict] = {}
    mods["kalman"] = kalman_vote(tape)
    mods["fracdiff"] = fracdiff_vote(tape)
    mods["bocpd"] = bocpd_vote(tape)
    mods["entropy"] = entropy_block(tape)
    mods["ou_hurst"] = ou_hurst_block(tape, W)
    mods["jumps"] = jumps_vote(tape)
    mods["pairs_ou"] = pairs_vote(symbol, tape, others)
    mods["wavelet"] = wavelet_vote(tape)
    mods["signature"] = signature_vote(P, V)
    rmt_res = 0.0
    if rmt and rmt.get("residual_vote"):
        rmt_res = float(rmt["residual_vote"].get(symbol, 0.0) or 0.0)
    mods["rmt_residual"] = {"vote": round(rmt_res, 4)}

    weights = dict(MODULE_WEIGHTS)
    weights["signature"] = min(float(sig_weight), 0.15) \
        if sig_weight <= 0.15 else min(float(sig_weight), 0.6)
    # walk-forward grade-driven scaling (engine supplies, from the
    # adv_* accuracy rings): proven modules earn up to 1.5x, weak
    # ones fall to 0.5x — the floor keeps every voter voting
    for _k, _sc in (weight_scales or {}).items():
        if _k in weights and _k != "signature":
            weights[_k] = weights[_k] * max(0.5, min(1.5, float(_sc)))
    num = den = 0.0
    signs: dict[str, int] = {}
    for name, blk in mods.items():
        v = float(blk.get("vote", 0.0) or 0.0)
        w = weights.get(name, 0.5)
        num += w * v
        den += w
        signs[name] = _sign(v)
    vote = num / den if den else 0.0

    gate = 1.0
    for name in ("entropy", "ou_hurst", "bocpd"):
        gate *= float(mods[name].get("gate", 1.0) or 1.0)
    gate = max(0.45, min(1.0, gate))

    full = full_closes or P
    har = har_rv(full)
    evt = evt_tail(full)
    rqa = rqa_multiplier(tape)
    size_mult = 1.0
    for blk in (har, evt):
        sm = blk.get("size_mult")
        if sm:
            size_mult *= float(sm)
    if rmt and rmt.get("mode_share"):
        # market mode dominant => fake diversification => dampen
        size_mult *= max(0.6, 1.15 - float(rmt["mode_share"]))
    size_mult = max(0.25, min(1.0, size_mult))

    return {
        "vote": round(vote, 4),
        "signs": signs,
        "modules": mods,
        "soft_gate": round(gate, 3),
        "rqa_mult": float(rqa.get("mult", 1.0)),
        "rqa": rqa,
        "har_rv": har,
        "evt": evt,
        "rmt": rmt,
        "size_mult": round(size_mult, 3),
        "weights": weights,
    }
