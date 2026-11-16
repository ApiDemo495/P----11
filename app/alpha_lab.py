"""Alpha lab — the measurement rig behind every fusion upgrade.

This module exists so that a change to the live pipeline can be judged on
real Binance history instead of intuition.  It is deliberately free of
the app's runtime (no feed, no engine, no UI): it takes bars, builds
strictly causal features, labels them with the triple barrier, and scores
them with purged walk-forward validation.

Grounded in the literature this repo cites (see docs/ALPHA_RESEARCH.md):

  * order-flow imbalance / taker imbalance predicts crypto returns over
    hours, with the *persistent* component carrying the signal
    (arXiv 2607.09426 "quarter-hour effect"; arXiv 2602.00776 crypto
    microstructure SHAP study);
  * meta-labeling + triple barrier + purged & embargoed walk-forward
    (Lopez de Prado, AFML ch. 3/4/7);
  * probability calibration scored with Brier / ECE and a calibration
    slope, fitted on out-of-fold predictions only.

Everything here is pure NumPy + stdlib so it can be imported by the
service (to fit live calibrators) and by the CLI backtest alike.

Units: returns are fractional; "bp" means basis points of notional.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field

import numpy as np

# ─────────────────────────────────────────────────────────────────────
# bars
# ─────────────────────────────────────────────────────────────────────


@dataclass
class Bars:
    """OHLCV + the order-flow columns Binance publishes per bar."""

    t: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    quote_v: np.ndarray = field(default_factory=lambda: np.zeros(0))
    trades: np.ndarray = field(default_factory=lambda: np.zeros(0))
    taker_buy_v: np.ndarray = field(default_factory=lambda: np.zeros(0))
    taker_buy_qv: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def has_flow(self) -> bool:
        return self.taker_buy_v.size == self.t.size and bool(
            np.any(self.taker_buy_v > 0))


def load_csv(path: str) -> Bars:
    """Read a cache file written by scripts/fetch_binance_mirror.py."""
    cols: dict[str, list[float]] = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            for k, val in row.items():
                if k is None or val in (None, ""):
                    continue
                cols.setdefault(k, []).append(float(val))
    def arr(k: str) -> np.ndarray:
        return np.asarray(cols.get(k, []), dtype=float)
    return Bars(
        t=arr("t"), o=arr("o"), h=arr("h"), l=arr("l"), c=arr("c"),
        v=arr("v"), quote_v=arr("quote_v"), trades=arr("trades"),
        taker_buy_v=arr("taker_buy_v"), taker_buy_qv=arr("taker_buy_qv"))


def resample(bars: Bars, step: int) -> Bars:
    """Aggregate `step` consecutive bars (time-ordered, no gaps assumed)."""
    if step <= 1 or len(bars) < step:
        return bars
    n = len(bars) // step
    idx = np.arange(n) * step
    sl = lambda a, fn: (np.asarray([fn(a[i:i + step]) for i in idx])
                        if a.size == len(bars) else np.zeros(0))
    out = Bars(
        t=bars.t[idx],
        o=bars.o[idx],
        h=sl(bars.h, np.max),
        l=sl(bars.l, np.min),
        c=bars.c[idx + step - 1],
        v=sl(bars.v, np.sum),
        quote_v=sl(bars.quote_v, np.sum),
        trades=sl(bars.trades, np.sum),
        taker_buy_v=sl(bars.taker_buy_v, np.sum),
        taker_buy_qv=sl(bars.taker_buy_qv, np.sum))
    return out


# ─────────────────────────────────────────────────────────────────────
# small numeric helpers (no scipy dependency at runtime)
# ─────────────────────────────────────────────────────────────────────


def _safe(x: float) -> float:
    return float(x) if math.isfinite(x) else 0.0


def _z(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Standardise `x` using the mean/std of `ref` (the train fold)."""
    sd = float(np.std(ref)) or 1e-9
    return (x - float(np.mean(ref))) / sd


def zscore(a: np.ndarray) -> np.ndarray:
    mu = float(np.nanmean(a)) if a.size else 0.0
    sd = float(np.nanstd(a)) if a.size else 0.0
    return (a - mu) / sd if sd > 1e-12 else np.zeros_like(a)


def rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    n = a.size
    if n == 0:
        return a
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    # average ties
    sa = a[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sa[j] == sa[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + j + 1) / 2.0
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 8 or y.size != x.size:
        return 0.0
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return _safe(float((rx * ry).sum()) / d) if d > 1e-12 else 0.0


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 8 or y.size != x.size:
        return 0.0
    xm, ym = x - x.mean(), y - y.mean()
    d = math.sqrt(float((xm * xm).sum()) * float((ym * ym).sum()))
    return _safe(float((xm * ym).sum()) / d) if d > 1e-12 else 0.0


def brier(p: np.ndarray, y: np.ndarray) -> float:
    if p.size == 0 or p.size != y.size:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error (equal-width bins)."""
    if p.size < 20:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total, n = 0.0, p.size
    for k in range(bins):
        m = (p >= edges[k]) & (p < edges[k + 1] if k < bins - 1
                               else p <= edges[k + 1])
        if not np.any(m):
            continue
        total += abs(float(p[m].mean()) - float(y[m].mean())) * float(m.sum())
    return total / n


def calib_slope(p: np.ndarray, y: np.ndarray) -> float:
    """Slope of the reliability line: y ~ a + b·p on logit pairs.

    b = 1 perfectly calibrated, b < 1 overconfident (the common case:
    stated 0.62 realises 0.54), b > 1 underconfident.
    """
    if p.size < 30 or y.size != p.size:
        return float("nan")
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    lp = np.log(pc / (1 - pc))
    if float(np.std(lp)) < 1e-9:
        return float("nan")
    return pearson(lp, y.astype(float))


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    pc = np.clip(p, eps, 1 - eps)
    return np.log(pc / (1 - pc))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                 iters: int = 40) -> tuple[np.ndarray, float]:
    """IRLS logistic regression (adds an intercept). Deterministic."""
    n, k = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(k + 1)
    for _ in range(iters):
        p = sigmoid(Xb @ w)
        W = np.clip(p * (1 - p), 1e-6, None)
        g = Xb.T @ (p - y) + l2 * np.r_[0.0, w[1:]]
        H = (Xb * W[:, None]).T @ Xb + l2 * np.eye(k + 1)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return w[1:], float(w[0])


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float | None = None,
              standardise: bool = True) -> tuple[np.ndarray, float, dict]:
    """Ridge regression with an intercept. Returns (beta, intercept, meta).

    Adaptive shrinkage: `lam` defaults to n·1e-3, chosen so collinear
    indicators (momentum vs range position vs taker imbalance, which all
    read the same price path to a degree) share weight instead of
    double-counting it. Closed form, deterministic, no sklearn.
    """
    n, k = X.shape
    if n < 20 or k == 0:
        return np.zeros(k), 0.0, {"lam": 0.0, "n": n}
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (X - mu) / sd if standardise else X
    ym = float(y.mean())
    yc = y - ym
    lam = float(n * 1e-3) if lam is None else float(lam)
    A = Z.T @ Z + lam * np.eye(k)
    try:
        b = np.linalg.solve(A, Z.T @ yc)
    except np.linalg.LinAlgError:
        b = np.linalg.lstsq(A, Z.T @ yc, rcond=None)[0]
    # effective degrees of freedom (ridge hat-trace)
    try:
        tr = float(np.trace(np.linalg.solve(A, Z.T @ Z)))
    except np.linalg.LinAlgError:
        tr = float(k)
    return b / sd, ym - float((mu / sd) @ b), {
        "lam": lam, "n": n, "edof": tr, "scale": sd.tolist()}


def isotonic_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators. Returns (knots_x, knots_y) increasing."""
    if x.size == 0:
        return np.zeros(0), np.zeros(0)
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order].astype(float)
    w = np.ones_like(ys)
    vy, vw, vx = [], [], []
    for i in range(ys.size):
        cy, cw, cx = ys[i], w[i], [xs[i]]
        while vy and cy < vy[-1]:
            py = vy.pop(); pw = vw.pop(); px = vx.pop()
            cy = (cy * cw + py * pw) / (cw + pw)
            cw += pw
            cx = px + cx
        vy.append(cy); vw.append(cw); vx.append(cx)
    ky = np.asarray(vy, dtype=float)
    kx = np.asarray([float(np.mean(v)) for v in vx], dtype=float)
    return kx, ky


def isotonic_apply(kx: np.ndarray, ky: np.ndarray, x: np.ndarray) -> np.ndarray:
    if kx.size == 0:
        return np.full_like(x, 0.5, dtype=float)
    return np.interp(np.clip(x, kx[0], kx[-1]), kx, ky)


# ─────────────────────────────────────────────────────────────────────
# features — every value at index i uses bars[:i+1] only
# ─────────────────────────────────────────────────────────────────────

FEATURE_NAMES = (
    "mom_z",        # multi-horizon momentum z
    "eff_ratio",    # trend efficiency (1 clean trend, 0 chop)
    "vol_regime",   # short/long realised vol ratio - 1
    "jump_ratio",   # 1 - BV/RV  (jump share of variance)
    "taker_imb",    # EWMA taker imbalance in [-1,1]
    "taker_persist",  # sign-persistence of taker imbalance
    "trade_size",   # large-print asymmetry (signed)
    "vwap_dev",     # close vs volume-weighted price, in σ
    "liquidity",    # quote volume vs its median (liquidity vacuum < 1)
    "range_pos",    # position inside the rolling range, centred
    "season_sin",   # intraday clock phase (quarter-hour effect family)
    "season_cos",
    "xasset",       # market leader's recent return, contrarian
)

TA_FEATURES = ("mom_z", "eff_ratio", "range_pos", "vol_regime", "season_sin",
               "season_cos")
# Thesis sign prior — the direction each feature is EXPECTED to carry at
# 1h–4h crypto horizons, from the walk-forward audit in
# backtest/results/ALPHA_LAB.md (short-horizon reversal + jump separation
# + contrarian taker flow; NOT momentum). +1 means "keep the sign",
# -1 means "the validated relationship is reversed". Every live use is
# scored by how often the sign held out-of-sample (see sign_stability).
SIGN_PRIOR = {
    "mom_z": -1.0,        # 3/12/48-bar momentum → reversal
    "eff_ratio": -1.0,    # a clean trend this hour tends to give back
    "range_pos": -1.0,    # top of the range → mean revert
    "vol_regime": -1.0,   # vol expansion → fade
    "jump_ratio": +1.0,   # jump-dominated variance → continuation
    "taker_imb": -1.0,    # aggressive buying → contrarian
    "taker_persist": -1.0,
    "trade_size": +1.0,   # large prints confirm the bar's direction
    "vwap_dev": -1.0,     # stretched off VWAP → revert
    "liquidity": -1.0,    # volume spike → fade
    "season_sin": +1.0,
    "season_cos": +1.0,
    "xasset": -1.0,       # market leader's move → contrarian (validated)
}
# the cross-asset spillover feature: the market leader's recent return,
# used contrarian for every asset (validated on all four).
XASSET_SIGN = -1.0
XASSET_BARS = 6
FLOW_FEATURES = ("taker_imb", "taker_persist", "trade_size", "vwap_dev",
                 "liquidity", "jump_ratio")


def _ewma(a: np.ndarray, halflife: float) -> float:
    if a.size == 0:
        return 0.0
    lam = 0.5 ** (1.0 / max(1e-9, halflife))
    w = lam ** np.arange(a.size)[::-1]
    w = w / w.sum()
    return float((a * w).sum())


def build_features(bars: Bars, need: int = 200,
                   market: Bars | None = None) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return (features [n, 12], forward-ready index list, bar index list).

    Rows start at `need` and stop 1 bar before the end so a horizon label
    can still be attached by the caller.
    """
    c = bars.c
    r = np.zeros_like(c)
    r[1:] = np.diff(c) / np.maximum(c[:-1], 1e-12)
    tb = bars.taker_buy_v if bars.has_flow else np.zeros_like(c)
    vol = np.maximum(bars.v, 1e-12)
    tr = np.maximum(bars.trades, 1e-12)
    qv = bars.quote_v if bars.quote_v.size == c.size else np.zeros_like(c)
    rows, idxs = [], []
    hours = None
    if bars.t.size == c.size:
        hours = (bars.t // 3600) % 24
    for i in range(need, len(bars) - 1):
        w24 = r[i - 23:i + 1]
        sd24 = float(np.std(w24)) or 1e-9
        # momentum: 3 / 12 / 48 bar returns scaled by their own vol
        m3 = _safe((c[i] / c[i - 3] - 1) / (sd24 * math.sqrt(3)))
        m12 = _safe((c[i] / c[i - 12] - 1) / (sd24 * math.sqrt(12)))
        m48 = _safe((c[i] / c[i - 48] - 1) / (sd24 * math.sqrt(48)))
        mom_z = (m3 + m12 + m48) / 3.0
        # efficiency ratio over 24 bars
        seg = c[i - 24:i + 1]
        net = abs(seg[-1] - seg[0])
        path = float(np.abs(np.diff(seg)).sum()) or 1e-12
        eff = net / path
        # vol regime
        sd96 = float(np.std(r[i - 95:i + 1])) or 1e-9
        vol_reg = sd24 / sd96 - 1.0
        # bipower variation vs realised variance (jump separation)
        rr = np.abs(r[i - 23:i + 1])
        bv = float(np.mean(rr[1:] * rr[:-1])) * (math.pi / 2.0) if rr.size > 2 else 0.0
        rv = float(np.mean(w24 ** 2))
        jump = 1.0 - (bv / rv if rv > 1e-18 else 1.0)
        jump = float(np.clip(jump, -1.0, 1.0))
        # taker imbalance
        imb_hist = (2.0 * tb[i - 23:i + 1] / vol[i - 23:i + 1] - 1.0
                    if bars.has_flow else np.zeros(24))
        ti = _ewma(imb_hist, 6.0)
        persist = float(np.mean(np.sign(imb_hist[-12:])))
        # large-print asymmetry: avg trade size now vs 96-bar median,
        # signed by the bar's own direction (who is trading big)
        tsz = float(np.mean(tr[i - 5:i + 1]))
        tsz_med = float(np.median(tr[max(0, i - 96):i + 1])) or 1e-12
        size_rel = math.log(max(tsz, 1e-9) / tsz_med)
        trade_size = _safe(math.tanh(size_rel) * math.copysign(1.0, r[i] or 1.0))
        # volume-weighted price deviation
        if qv[i] > 0 and vol[i] > 0:
            vwap = qv[i] / vol[i]
            vwap_dev = _safe((c[i] / vwap - 1.0) / sd24)
        else:
            vwap_dev = 0.0
        # liquidity vacuum
        qmed = float(np.median(qv[max(0, i - 96):i + 1])) or 1e-12
        liq = _safe(math.log(max(qv[i], 1e-9) / qmed))
        # range position
        hi = float(np.max(bars.h[i - 23:i + 1]))
        lo = float(np.min(bars.l[i - 23:i + 1]))
        rng = hi - lo
        rpos = _safe(((c[i] - lo) / rng - 0.5) * 2.0) if rng > 1e-12 else 0.0
        if hours is not None:
            frac = (bars.t[i] % 86400) / 86400.0
            s_sin, s_cos = math.sin(2 * math.pi * frac), math.cos(2 * math.pi * frac)
        else:
            s_sin = s_cos = 0.0
        if market is not None and market.t.size:
            k = int(np.searchsorted(market.t, bars.t[i], side="right")) - 1
            k = max(XASSET_BARS, min(k, market.t.size - 1))
            xa = _safe((market.c[k] / market.c[max(0, k - XASSET_BARS)] - 1.0)
                       / (sd96 * math.sqrt(XASSET_BARS)))
        else:
            xa = 0.0
        rows.append([mom_z, eff, vol_reg, jump, ti, persist, trade_size,
                     vwap_dev, liq, rpos, s_sin, s_cos, xa])
        idxs.append(i)
    X = np.asarray(rows, dtype=float)
    # Winsorise at FIXED bounds. (Full-sample percentiles would leak the
    # future into a past row — the causality test in tests/test_alpha_lab
    # pins this: truncating the tape must not change an earlier row.)
    if X.size:
        X = np.clip(X, -6.0, 6.0)
    return X, np.asarray(idxs, dtype=int), idxs


# ─────────────────────────────────────────────────────────────────────
# triple barrier
# ─────────────────────────────────────────────────────────────────────


def triple_barrier(bars: Bars, i: int, horizon: int,
                   pt_frac: float, sl_frac: float) -> dict:
    """First-touch settlement. SL wins ties (conservative)."""
    entry = float(bars.c[i])
    up, dn = entry * (1 + pt_frac), entry * (1 - sl_frac)
    end = min(i + horizon, len(bars) - 1)
    for j in range(i + 1, end + 1):
        if bars.l[j] <= dn:
            return {"label": -1, "ret": -sl_frac, "bars": j - i,
                    "barrier": "sl"}
        if bars.h[j] >= up:
            return {"label": 1, "ret": pt_frac, "bars": j - i,
                    "barrier": "tp"}
    px = float(bars.c[end])
    return {"label": int(np.sign(px / entry - 1.0)), "ret": px / entry - 1.0,
            "bars": end - i, "barrier": "time"}


def vol_sigma(bars: Bars, i: int, horizon: int, lookback: int = 200) -> float:
    """Realised vol scaled to `horizon` bars, from bars[:i+1] only."""
    a = max(1, i - lookback)
    seg = bars.c[a:i + 1]
    if seg.size < 10:
        return 0.0
    r = np.diff(seg) / np.maximum(seg[:-1], 1e-12)
    sd = float(np.std(r))
    return sd * math.sqrt(max(1, horizon))


# ─────────────────────────────────────────────────────────────────────
# purged walk-forward
# ─────────────────────────────────────────────────────────────────────


def purged_folds(n: int, folds: int = 4, embargo: int = 0,
                 min_train: float = 0.3) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward-only splits with an embargo gap at each seam."""
    out = []
    start = int(n * min_train)
    if start < 30:
        return out
    edges = np.linspace(start, n, folds + 1).astype(int)
    for k in range(folds):
        te0, te1 = int(edges[k]), int(edges[k + 1])
        tr1 = max(0, te0 - embargo)
        if te1 - te0 < 10:
            continue
        out.append((np.arange(0, tr1), np.arange(te0, te1)))
    return out


# ─────────────────────────────────────────────────────────────────────
# the evaluation itself
# ─────────────────────────────────────────────────────────────────────


def evaluate_symbol(bars: Bars, horizon: int, friction_bps: float = 12.0,
                    floor: float = 0.35, market: Bars | None = None) -> dict:
    """Before vs after on one asset/horizon, purged walk-forward.

    BEFORE  the app's current tilt: an equal-weight technical composite
            (momentum + efficiency + range position) with the signs as
            published, confidence = |composite|/3, fixed 2:1 vol barriers.
    AFTER   the thesis-signed composite: every feature multiplied by its
            validated sign prior (short-horizon reversal, jump
            continuation, contrarian taker flow, cross-asset spillover),
            z-scored on the train fold, equal-weighted (nothing to
            over-fit), with a meta-label success filter, isotonic
            confidence calibration and vol-scaled barrier multiples
            chosen on the train fold.

    All reported numbers are out-of-fold: each fold trains on its own
    past only and an embargo of `horizon` bars sits at every seam.
    """
    X, idx, _ = build_features(bars, market=market)
    if X.shape[0] < 300:
        return {"n": int(X.shape[0]), "error": "not enough bars"}
    names = list(FEATURE_NAMES)
    col = {n: k for k, n in enumerate(names)}
    ta_cols = [col[n] for n in TA_FEATURES]
    prior = np.asarray([SIGN_PRIOR.get(n, 1.0) for n in names])

    fwd = np.zeros(idx.size)
    for k, i in enumerate(idx):
        j = min(int(i) + horizon, len(bars) - 1)
        fwd[k] = bars.c[j] / bars.c[int(i)] - 1.0
    sig = np.asarray([vol_sigma(bars, int(i), horizon) for i in idx])
    y_up = (fwd > 0).astype(float)

    folds = purged_folds(X.shape[0], folds=5, embargo=horizon)
    if not folds:
        return {"n": int(X.shape[0]), "error": "not enough folds"}

    oos = {"before": [], "after": [], "p_corr": [], "y": [], "fwd": [],
           "sig": [], "idx": [], "barriers": []}
    sign_hold: dict[str, list[int]] = {n: [] for n in names}
    feat_ic: dict[str, list[float]] = {n: [] for n in names}
    for tr, te in folds:
        if tr.size < 150 or te.size < 30:
            continue
        Z = np.column_stack([_z(X[:, c], X[tr, c]) for c in range(X.shape[1])])
        # ── BEFORE: published momentum tilt, equal weight ──
        before = Z[:, ta_cols].mean(axis=1)
        # ── AFTER: thesis-signed composite, equal weight ──
        S = Z * prior
        # ── prior-anchored IC weighting ──
        # magnitude from the TRAIN fold's IC, direction from the prior.
        # A feature whose train IC contradicts its prior gets zero weight
        # rather than a flipped sign (folds are too short to re-decide a
        # sign without flapping).
        wts = np.zeros(len(names))
        for c, nm in enumerate(names):
            ic_tr = spearman(X[tr, c], fwd[tr])
            agree = ic_tr * prior[c]
            wts[c] = prior[c] * max(0.0, agree)
            sign_hold[nm].append(1 if agree > 0 else 0)
            feat_ic[nm].append(ic_tr)
        if np.abs(wts).sum() < 1e-9:
            wts = prior.copy()
        wnorm = wts / np.abs(wts).sum()
        after = Z @ wnorm

        # meta-label: P(the scored side is right | flow + regime + |score|)
        mX = np.column_stack([S[:, [col[n] for n in FLOW_FEATURES]],
                              Z[:, [col["eff_ratio"], col["vol_regime"],
                                    col["jump_ratio"]]], np.abs(after)])
        ok_tr = ((np.sign(after[tr]) * np.sign(fwd[tr])) > 0).astype(float)
        mw, mb = fit_logistic(mX[tr], ok_tr, l2=6.0)
        p_tr = sigmoid(mX[tr] @ mw + mb)
        cut = max(60, int(tr.size * 0.7))
        # calibrate CONFIDENCE (|score|) → P(correct); the card prints a
        # confidence, so that is the quantity that must be honest.
        kx, ky = isotonic_fit(np.abs(after[tr])[:cut], ok_tr[:cut])
        p_te = isotonic_apply(kx, ky, np.abs(after[te]))

        # barriers chosen on the train fold (EV, friction charged)
        best = (2.0, 1.0, -1e18)
        for kpt in (1.0, 1.5, 2.0, 2.5):
            for ksl in (0.75, 1.0, 1.5):
                ev = 0.0
                for j in tr[:250]:
                    tbj = triple_barrier(bars, int(idx[j]), horizon,
                                         max(1e-5, kpt * sig[j]),
                                         max(1e-5, ksl * sig[j]))
                    d = np.sign(after[j]) or 1.0
                    win = (d * tbj["label"]) > 0
                    ev += (abs(tbj["ret"]) if win else -abs(tbj["ret"]))
                if ev > best[2]:
                    best = (kpt, ksl, ev)
        oos["before"].extend(before[te].tolist())
        oos["after"].extend(after[te].tolist())
        oos["p_corr"].extend(p_te.tolist())
        oos["y"].extend(y_up[te].tolist())
        oos["fwd"].extend(fwd[te].tolist())
        oos["sig"].extend(sig[te].tolist())
        oos["idx"].extend(idx[te].tolist())
        oos["barriers"].append((best[0], best[1]))

    if not oos["y"]:
        return {"n": int(X.shape[0]), "error": "no out-of-fold rows"}

    bf = np.asarray(oos["before"]); af = np.asarray(oos["after"])
    p_corr = np.asarray(oos["p_corr"]); y = np.asarray(oos["y"])
    fw = np.asarray(oos["fwd"]); sg = np.asarray(oos["sig"])
    ix = np.asarray(oos["idx"])
    up = y > 0.5
    fric = friction_bps / 1e4

    def sign_acc(score, mask=None):
        m = np.ones(y.size, bool) if mask is None else mask
        return float(np.mean((score[m] > 0) == up[m])) if np.any(m) else float("nan")

    def p_up(score, pc=None):
        base = np.clip(0.5 + 0.5 * np.tanh(score / 2.0), 0.02, 0.98)
        if pc is None:
            return base
        p = np.clip(pc, 0.02, 0.98)
        return np.where(score > 0, p, 1.0 - p)

    def ev_bp(score, kpt, ksl, mask=None, cost=True):
        m = np.ones(y.size, bool) if mask is None else mask
        ii = np.nonzero(m)[0][:600]
        if ii.size == 0:
            return float("nan")
        tot = 0.0
        for j in ii:
            d = 1.0 if score[j] > 0 else -1.0
            tbj = triple_barrier(bars, int(ix[j]), horizon,
                                 max(1e-5, kpt * sg[j]), max(1e-5, ksl * sg[j]))
            win = (d * tbj["label"]) > 0
            tot += (abs(tbj["ret"]) if win else -abs(tbj["ret"]))
        bp = tot / ii.size * 1e4
        return bp - (friction_bps if cost else 0.0)

    ks = oos["barriers"] or [(2.0, 1.0)]
    kpt_f = float(np.median([k[0] for k in ks]))
    ksl_f = float(np.median([k[1] for k in ks]))
    conf_b = np.abs(bf) / 3.0
    conf_a = np.abs(af) / 3.0
    hi = conf_a >= floor
    gate = hi & (p_corr >= 0.52)

    return {
        "n": int(y.size),
        "horizon": horizon,
        "base_rate": float(np.mean(y)),
        "before": {
            "sign_acc": sign_acc(bf), "ic": spearman(bf, fw),
            "brier": brier(p_up(bf), y), "slope": calib_slope(p_up(bf), y),
            "ev_bp": ev_bp(bf, 2.0, 1.0, cost=True),
            "ev_bp_nocost": ev_bp(bf, 2.0, 1.0, cost=False),
        },
        "before_hi": {
            "n": int(hi.sum()), "sign_acc": sign_acc(bf, hi),
            "ev_bp": ev_bp(bf, 2.0, 1.0, hi),
        },
        "after": {
            "sign_acc": sign_acc(af), "ic": spearman(af, fw),
            "brier": brier(p_up(af, p_corr), y), "slope": calib_slope(p_up(af, p_corr), y),
            "p_corr_mean": float(np.mean(p_corr)),
            "ev_bp": ev_bp(af, kpt_f, ksl_f, cost=True),
            "ev_bp_nocost": ev_bp(af, kpt_f, ksl_f, cost=False),
        },
        "after_gated": {
            "n": int(gate.sum()), "sign_acc": sign_acc(af, gate),
            "coverage": float(np.mean(gate)),
            "ev_bp": ev_bp(af, kpt_f, ksl_f, gate),
        },
        "calib": {
            "raw_brier": brier(p_up(af), y),
            "iso_brier": brier(p_up(af, p_corr), y),
            "iso_ece": ece(p_up(af, p_corr), y),
            "iso_slope": calib_slope(p_up(af, p_corr), y),
            "base_rate": float(np.mean(y)),
            "baseline_brier": float(np.mean(y) * (1 - np.mean(y))),
        },
        "barriers": {"k_pt": kpt_f, "k_sl": ksl_f},
        "sign_stability": {n: (float(np.mean(v)) if v else 0.0)
                           for n, v in sign_hold.items()},
        "feature_ic": {n: (float(np.mean(v)) if v else 0.0)
                       for n, v in feat_ic.items()},
        "ic_ta": float(np.mean([abs(v) for v in feat_ic_get(feat_ic, TA_FEATURES)])),
        "ic_flow": float(np.mean([abs(v) for v in feat_ic_get(feat_ic, FLOW_FEATURES)])),
    }


def feat_ic_get(feat_ic: dict[str, list[float]], names) -> list[float]:
    out = []
    for n in names:
        v = feat_ic.get(n) or []
        if v:
            out.append(float(np.mean(v)))
    return out
