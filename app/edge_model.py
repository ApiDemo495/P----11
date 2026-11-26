"""Live consumer of the walk-forward-validated directional edge model.

What this is
------------
`scripts/train_edge_model.py` fits a gradient-boosted classifier on causal
price features (momentum at nine horizons, mean-reversion z-scores, intrabar
close location, range position, volume pressure, acceleration, volatility
ratio, hour-of-day, day-of-week) to answer ONE question: at a stop of
4 x ATR(14) and a target of 2 x that stop over a 336-bar horizon, does the
next window go up or down?

Why it exists
-------------
`backtest/edge_model.py` measures it out-of-sample, walk-forward with purge:
BTCUSDT **39.9 %** at 2:1 (CI 38.8–41.1, n=7168), BNBUSDT **39.5 %**
(38.3–40.6), ETHUSDT 34.6 % — against a 33.33 % break-even. That is +0.20 R
per trade, the first POSITIVE expectancy this repo has measured at 2:1.

The same audit also settled what does NOT work: a model trained on the app's
own 44 formula outputs scores 36.2 % at 2:1, the generic features score
36.2 %, and a SHUFFLED copy of the physics values scores 37.1 %. The
formula stack carries no measurable directional signal, which is why this
module reads price directly.

Honesty rules baked in
----------------------
* The artifact is OPTIONAL. With no file the app is byte-identical to
  before; `available()` is False and nothing changes.
* `prob_up` is the model's own probability. `side` needs only a coin-flip
  threshold (0.5) and `confidence` is |p-0.5|, so a 0.42 is a SELL with
  modest conviction, not a withheld window.
* This NEVER abstains, and it never overrides the app's own risk gates: it
  is one more vote in `_direction_ladder`, published with its probability so
  a reader can see exactly what moved the side.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

ARTIFACT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "models", "edge_btc.joblib")

_CACHE: dict[str, Any] = {"loaded": False, "model": None, "features": None,
                          "meta": {}}


def _load() -> dict[str, Any]:
    if _CACHE["loaded"]:
        return _CACHE
    _CACHE["loaded"] = True
    try:
        import joblib
        if not os.path.exists(ARTIFACT):
            return _CACHE
        blob = joblib.load(ARTIFACT)
        _CACHE["model"] = blob["model"]
        _CACHE["features"] = list(blob["features"])
        _CACHE["meta"] = {k: blob.get(k) for k in
                          ("horizon", "stop_atr", "trained_on", "bars")}
    except Exception:                       # a broken artifact is "absent"
        _CACHE["model"] = None
    return _CACHE


def available() -> bool:
    return _load()["model"] is not None


def meta() -> dict[str, Any]:
    return dict(_load()["meta"])


# ─────────────────────── features, identical to training ─────────────────
def _rmean(x, w):
    cs = np.cumsum(np.insert(x, 0, 0.0))
    o = np.full(len(x), np.nan)
    o[w:] = (cs[w + 1:] - cs[1:-w]) / w
    return o


def _rstd(x, w):
    cs = np.cumsum(np.insert(x, 0, 0.0))
    cs2 = np.cumsum(np.insert(x * x, 0, 0.0))
    o = np.full(len(x), np.nan)
    m = (cs[w + 1:] - cs[1:-w]) / w
    v = (cs2[w + 1:] - cs2[1:-w]) / w - m * m
    o[w:] = np.sqrt(np.maximum(v, 0.0))
    return o


def build_features(closes, highs, lows, volumes, times):
    """The EXACT training transform. Kept here (not imported from backtest/)
    so the live app never depends on the research tree at runtime, and a
    unit test pins the two implementations together."""
    import math
    c = np.asarray(closes, dtype=float)
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows, dtype=float)
    v = np.asarray(volumes, dtype=float)
    t = np.asarray(times, dtype=np.int64)
    n = len(c)
    if n < 400:
        return None
    lr = np.concatenate([[0.0], np.diff(np.log(np.maximum(c, 1e-12)))])
    F = {}
    for w in (3, 6, 12, 24, 48, 72, 120, 168, 336):
        sd = _rstd(lr, w) * math.sqrt(w)
        F[f"mom{w}"] = np.where(sd > 0,
                                _rmean(lr, w) * w / np.maximum(sd, 1e-12), 0.0)
        F[f"vol{w}"] = _rstd(lr, w)
    for w in (12, 24, 48, 96, 240):
        m, sd = _rmean(c, w), _rstd(c, w)
        F[f"rev{w}"] = np.where(sd > 0, (c - m) / np.maximum(sd, 1e-12), 0.0)
    rng = np.maximum(h - l, 1e-12)
    clv = (2.0 * c - h - l) / rng
    F["clv"] = clv
    for w in (6, 12, 24):
        F[f"clv{w}"] = _rmean(clv, w)
    for w in (24, 96):
        hi_w = np.array([np.max(h[max(0, i - w):i + 1]) for i in range(n)])
        lo_w = np.array([np.min(l[max(0, i - w):i + 1]) for i in range(n)])
        span = np.maximum(hi_w - lo_w, 1e-12)
        F[f"pos{w}"] = (c - lo_w) / span - 0.5
    mv = _rmean(np.maximum(v, 0.0), 48)
    F["volp"] = np.where(mv > 0, v / np.maximum(mv, 1e-12) - 1.0, 0.0)
    F["accel"] = F["mom6"] - F["mom72"]
    F["vov"] = np.where(F["vol24"] > 0,
                        F["vol6"] / np.maximum(F["vol24"], 1e-12), 1.0)
    hod = (t // 3600) % 24
    dow = (t // 86400) % 7
    F["hod_sin"] = np.sin(2 * math.pi * hod / 24)
    F["hod_cos"] = np.cos(2 * math.pi * hod / 24)
    F["dow_sin"] = np.sin(2 * math.pi * dow / 7)
    F["dow_cos"] = np.cos(2 * math.pi * dow / 7)
    names = sorted(F)
    X = np.column_stack([np.nan_to_num(F[k], nan=0.0, posinf=0.0, neginf=0.0)
                         for k in names])
    return np.clip(X, -8.0, 8.0), names


def predict(closes, highs, lows, volumes, times) -> dict[str, Any] | None:
    """The last bar's probability. `None` when the artifact or tape is absent."""
    st = _load()
    if st["model"] is None:
        return None
    try:
        built = build_features(closes, highs, lows, volumes, times)
        if built is None:
            return None
        X, names = built
        want = st["features"]
        col = {n: i for i, n in enumerate(names)}
        Xa = np.column_stack([X[:, col[n]] if n in col else np.zeros(len(X))
                              for n in want])
        p = float(st["model"].predict_proba(Xa[-1:])[:, 1][0])
        return {
            "prob_up": round(p, 4),
            "side": 1 if p >= 0.5 else -1,
            "confidence": round(abs(p - 0.5), 4),
            "horizon_bars": st["meta"].get("horizon"),
            "stop_atr": st["meta"].get("stop_atr"),
            "trained_on": st["meta"].get("trained_on"),
            "source": "walk-forward GBM, 39.9% @ 2:1 out-of-sample (BTC)",
        }
    except Exception:
        return None
