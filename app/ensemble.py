"""Sign-constrained, sparsity-pruned IC ensemble (walk-forward).

Replaces equal-weight averaging with a fitted meta-learner that respects
what the audit found:

  * many features are **noise** at a given horizon → an L1-style screen
    zeroes anything whose train-fold IC is inside the noise band
    `1.5/√n`, so the ensemble does not average noise into the signal;
  * the **sign** of an effect is not stable enough to be re-learned every
    fold (that is how you get a model that flips long/short weekly) → the
    sign comes from the validated prior and the fit only decides
    *magnitude*;
  * features are strongly collinear (momentum ≈ range position ≈ taker
    flow) → ridge shrinkage on standardised features handles the
    covariance instead of double-counting it.

`fit()` returns weights that can be applied as `Z @ w`. Everything is
deterministic NumPy; no sklearn, no random seeds.
"""
from __future__ import annotations

import math

import numpy as np


def noise_band(n: int, k: float = 1.5) -> float:
    """|IC| below this is statistically indistinguishable from zero."""
    return float(k / math.sqrt(max(8, n)))


def ic_weights(X: np.ndarray, y: np.ndarray, prior_sign: np.ndarray,
               lam: float | None = None, band: float | None = None,
               n_eff: int | None = None) -> dict:
    """Prior-signed, noise-screened, ridge-shrunk weights.

    w_c = prior_c · max(0, prior_c·IC_c − band) shrunk by the ridge
    solution's direction, so a feature contributes only when it both
    agrees with theory and clears the noise band.
    """
    n, k = X.shape
    if n < 20 or k == 0:
        return {"w": np.zeros(k), "ic": np.zeros(k), "kept": [],
                "band": 0.0, "lam": 0.0}
    band = noise_band(n_eff or n) if band is None else float(band)
    ic = np.zeros(k)
    for c in range(k):
        col = X[:, c]
        if float(np.std(col)) <= 1e-12:
            continue
        yc = y - y.mean()
        cc = col - col.mean()
        den = math.sqrt(float((cc * cc).sum()) * float((yc * yc).sum()))
        if den > 1e-18:
            ic[c] = float((cc * yc).sum() / den)
    agree = ic * prior_sign
    mag = np.maximum(0.0, agree - band)
    w = prior_sign * mag
    # ridge shrinkage on the surviving columns (collinearity control)
    keep = np.nonzero(np.abs(w) > 0)[0]
    if keep.size == 0:
        return {"w": np.zeros(k), "ic": ic, "kept": [], "band": band,
                "lam": 0.0}
    Z = X[:, keep]
    sd = Z.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Zs = (Z - Z.mean(axis=0)) / sd
    lam = float(n) * 1e-3 if lam is None else float(lam)
    A = Zs.T @ Zs + lam * np.eye(keep.size)
    try:
        beta = np.linalg.solve(A, Zs.T @ (y - y.mean()))
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(A, Zs.T @ (y - y.mean()), rcond=None)[0]
    # keep the prior's sign on every surviving coefficient
    beta = np.sign(w[keep]) * np.abs(beta)
    w_out = np.zeros(k)
    w_out[keep] = beta / sd
    scale = float(np.abs(w_out).sum())
    if scale > 0:
        w_out = w_out / scale
    return {"w": w_out, "ic": ic, "kept": [int(c) for c in keep],
            "band": band, "lam": lam,
            "sign_agree": float(np.mean(agree[np.abs(ic) > band] > 0))
            if np.any(np.abs(ic) > band) else float("nan")}


def score(Z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Apply ensemble weights to standardised features."""
    if Z.size == 0 or w.size == 0:
        return np.zeros(Z.shape[0] if Z.ndim else 0)
    return Z @ w


def zscore_train(X: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Standardise every column with TRAIN-fold statistics only."""
    mu = X[train_idx].mean(axis=0)
    sd = X[train_idx].std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd


# ───────────────────────── cost-aware meta-label ─────────────────────────


def cost_labels(fwd: np.ndarray, side: np.ndarray, hurdle_bps: np.ndarray
                ) -> np.ndarray:
    """1 when the trade's NET move clears its own hurdle, else 0.

    This is the label the cost-aware filter is trained on: not "was the
    direction right", but "did the move pay for the round trip".
    """
    net = side * fwd * 1e4 - hurdle_bps
    return (net > 0).astype(float)


def net_edge_bps(fwd: np.ndarray, side: np.ndarray,
                 hurdle_bps: np.ndarray) -> np.ndarray:
    return side * fwd * 1e4 - hurdle_bps


def fit_platt(features: np.ndarray, y: np.ndarray, l2: float = 4.0,
              iters: int = 40) -> tuple[np.ndarray, float]:
    """Logistic regression (IRLS) — used for the cost-aware filter."""
    n, k = features.shape
    if n < 40 or k == 0:
        return np.zeros(k), 0.0
    Xb = np.hstack([np.ones((n, 1)), features])
    w = np.zeros(k + 1)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -40, 40)))
        W = np.clip(p * (1 - p), 1e-6, None)
        g = Xb.T @ (p - y) + l2 * np.r_[0.0, w[1:]]
        H = (Xb * W[:, None]).T @ Xb + l2 * np.eye(k + 1)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w -= step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return w[1:], float(w[0])


def apply_platt(features: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    if w.size == 0:
        return np.full(features.shape[0], 0.5)
    return 1.0 / (1.0 + np.exp(-np.clip(features @ w + b, -40, 40)))
