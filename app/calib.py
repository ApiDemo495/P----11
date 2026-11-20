"""Honest confidence: isotonic calibration of stated confidence → P(win).

The complaint this answers: "the card says 0.62 and the win rate is 0.54",
and its cousin "every asset's confidence hovers in the same narrow band".
Stated confidence is a *score* magnitude, not a probability.  Calibration
maps one to the other with the monotone (isotonic / PAV) fit the
calibration literature prescribes, and reports the three numbers that
prove whether the map is honest:

    Brier   mean (p − y)²  — 0.25 is a coin flip, lower is sharper
    ECE     mean |p − observed rate| over probability bins
    slope   reliability slope of y ~ logit(p); 1.0 perfect, <1 overconfident

Design rules (deliberate, from the walk-forward audit):

  * fit ONLY on settled, non-seeded ledger rows that carry a confidence —
    never on the live window being scored;
  * below `MIN_SAMPLES` the calibrator is an identity and says so
    (`source: "insufficient"`), so a fresh install cannot pretend;
  * pooled across assets/TFs when a single lane is thin, but a lane's own
    fit always wins once it has enough rows;
  * every observation is kept (bounded at `MAX_SAMPLES` per lane) so the
    map can be refitted as the regime moves — stale calibration is
    reported, never hidden.

The engine only ever *displays* `p_correct` and, when the map is real,
uses it to gate issuance.  Confidence itself is left to the fusion layer.
"""
from __future__ import annotations

import json
import math
import os
import threading

MIN_SAMPLES = 40
MAX_SAMPLES = 4000
STORE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "calibration.json")


def _brier(p, y) -> float:
    if not p:
        return float("nan")
    return sum((a - b) ** 2 for a, b in zip(p, y)) / len(p)


def _ece(p, y, bins: int = 10) -> float:
    if len(p) < 20:
        return float("nan")
    tot, n = 0.0, len(p)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, v in enumerate(p)
               if (lo <= v < hi) or (b == bins - 1 and v == hi)]
        if not idx:
            continue
        pm = sum(p[i] for i in idx) / len(idx)
        ym = sum(y[i] for i in idx) / len(idx)
        tot += abs(pm - ym) * len(idx)
    return tot / n


def _slope(p, y) -> float:
    """OLS slope of y on logit(p): 1.0 = calibrated, <1 = overconfident."""
    if len(p) < 30:
        return float("nan")
    xs, ys = [], []
    for a, b in zip(p, y):
        a = min(max(a, 1e-4), 1 - 1e-4)
        xs.append(math.log(a / (1 - a)))
        ys.append(b)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 1e-12 else float("nan")


def isotonic_fit(x, y):
    """Pool-adjacent-violators. Returns (kx, ky) non-decreasing in y."""
    if x is None or y is None or len(x) == 0 or len(x) != len(y):
        return [], []
    pairs = sorted((float(p[0]), float(p[1])) for p in zip(x, y))
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    ws = [1.0] * len(ys)
    vy, vw, vx = [], [], []
    for i in range(len(ys)):
        cy, cw, cx = ys[i], ws[i], [xs[i]]
        while vy and cy < vy[-1]:
            py, pw, px = vy.pop(), vw.pop(), vx.pop()
            cy = (cy * cw + py * pw) / (cw + pw)
            cw += pw
            cx = px + cx
        vy.append(cy)
        vw.append(cw)
        vx.append(cx)
    kx = [float(sum(v) / len(v)) for v in vx]
    return kx, [float(v) for v in vy]


def isotonic_apply(kx, ky, x: float) -> float:
    if not kx:
        return float("nan")
    if x <= kx[0]:
        return ky[0]
    if x >= kx[-1]:
        return ky[-1]
    for i in range(1, len(kx)):
        if x <= kx[i]:
            span = kx[i] - kx[i - 1]
            f = 0.0 if span <= 0 else (x - kx[i - 1]) / span
            return ky[i - 1] + f * (ky[i] - ky[i - 1])
    return ky[-1]


class Calibrator:
    """Per (symbol, tf) isotonic calibrators with pooled fallback."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or STORE
        self._lock = threading.RLock()
        self._lanes: dict[str, list[list[float]]] = {}
        self._fits: dict[str, dict] = {}
        self.load()

    # ── persistence ──────────────────────────────────────────────────
    def load(self) -> None:
        try:
            with open(self.path) as f:
                raw = json.load(f)
            lanes = raw.get("lanes") if isinstance(raw, dict) else None
            if isinstance(lanes, dict):
                self._lanes = {str(k): [[float(a), float(b)] for a, b in v][-MAX_SAMPLES:]
                               for k, v in lanes.items() if isinstance(v, list)}
        except Exception:
            self._lanes = {}
        self._refit_all()

    def save(self) -> bool:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as f:
                json.dump({"lanes": self._lanes,
                           "fits": {k: {"n": v["n"], "brier_raw": v["brier_raw"],
                                        "brier_iso": v["brier_iso"],
                                        "ece_iso": v["ece_iso"],
                                        "slope_raw": v["slope_raw"],
                                        "slope_iso": v["slope_iso"]}
                                    for k, v in self._fits.items()}},
                          f)
            return True
        except Exception:
            return False

    # ── fitting ──────────────────────────────────────────────────────
    @staticmethod
    def _key(symbol: str, tf: str) -> str:
        return f"{symbol}|{tf}"

    def _pooled(self, tf: str | None = None) -> list[list[float]]:
        rows: list[list[float]] = []
        for k, v in self._lanes.items():
            if tf and not k.endswith(f"|{tf}"):
                continue
            rows.extend(v)
        return rows

    def _fit(self, rows: list[list[float]]) -> dict | None:
        if len(rows) < MIN_SAMPLES:
            return None
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        kx, ky = isotonic_fit(xs, ys)
        if not kx:
            return None
        p_iso = [isotonic_apply(kx, ky, x) for x in xs]
        # BEFORE = the published confidence taken at face value (that is
        # exactly the number the card prints today); AFTER = isotonic.
        p_raw = [min(max(x, 1e-4), 1 - 1e-4) for x in xs]
        base = float(sum(ys) / len(ys))
        return {
            "kx": kx, "ky": ky, "n": len(rows), "base_rate": base,
            "brier_raw": _brier(p_raw, ys), "brier_iso": _brier(p_iso, ys),
            "ece_iso": _ece(p_iso, ys), "slope_iso": _slope(p_iso, ys),
            "slope_raw": float("nan"),
        }

    def _refit_all(self) -> None:
        with self._lock:
            self._fits = {}
            for key in list(self._lanes):
                f = self._fit(self._lanes[key])
                if f:
                    self._fits[key] = f
            for tf in ("TF1", "TF2", "TF3", "TF4"):
                rows = self._pooled(tf)
                f = self._fit(rows)
                if f:
                    self._fits[f"*|{tf}"] = f
            pooled = self._pooled(None)
            f = self._fit(pooled)
            if f:
                self._fits["*|*"] = f

    # ── public API ───────────────────────────────────────────────────
    def observe(self, symbol: str, tf: str, conf, won) -> None:
        """Add one settled sample. Non-finite confidences are ignored."""
        try:
            c = float(conf)
        except (TypeError, ValueError):
            return
        if not (0.0 < c <= 1.0):
            return
        key = self._key(symbol, tf)
        with self._lock:
            lane = self._lanes.setdefault(key, [])
            lane.append([round(c, 4), 1.0 if won else 0.0])
            if len(lane) > MAX_SAMPLES:
                del lane[:len(lane) - MAX_SAMPLES]
            f = self._fit(lane)
            if f:
                self._fits[key] = f

    def fit_for(self, symbol: str, tf: str) -> tuple[dict | None, str]:
        with self._lock:
            for key, src in ((self._key(symbol, tf), "lane"),
                             (f"*|{tf}", "pooled-tf"),
                             ("*|*", "pooled")):
                f = self._fits.get(key)
                if f:
                    return f, src
        return None, "insufficient"

    def apply(self, symbol: str, tf: str, conf) -> dict:
        """Confidence → P(this call wins). Never raises."""
        try:
            c = float(conf)
        except (TypeError, ValueError):
            return {"p": None, "source": "insufficient", "n": 0}
        f, src = self.fit_for(symbol, tf)
        if not f:
            return {"p": None, "source": src, "n": 0}
        # clamp to the observed range so extrapolation cannot invent a 99%
        p = isotonic_apply(f["kx"], f["ky"], min(max(c, 0.0), 1.0))
        return {"p": round(float(min(max(p, 0.02), 0.98)), 4),
                "source": src, "n": int(f["n"]),
                "base_rate": round(float(f["base_rate"]), 4),
                "brier_iso": round(float(f["brier_iso"]), 4),
                "slope": (None if math.isnan(f["slope_iso"])
                          else round(float(f["slope_iso"]), 3))}

    def telemetry(self, symbol: str | None = None, tf: str | None = None) -> dict:
        if symbol and tf:
            f, src = self.fit_for(symbol, tf)
        else:
            with self._lock:
                f, src = self._fits.get("*|*"), "pooled" if self._fits.get("*|*") else "insufficient"
        if not f:
            return {"source": "insufficient", "n": 0, "min_samples": MIN_SAMPLES,
                    "lanes": len(self._lanes)}
        return {
            "source": src, "n": int(f["n"]),
            "brier_raw": round(float(f["brier_raw"]), 4),
            "brier_iso": round(float(f["brier_iso"]), 4),
            "ece": (None if math.isnan(f["ece_iso"]) else round(float(f["ece_iso"]), 4)),
            "slope": (None if math.isnan(f["slope_iso"]) else round(float(f["slope_iso"]), 3)),
            "base_rate": round(float(f["base_rate"]), 4),
            "min_samples": MIN_SAMPLES, "lanes": len(self._lanes),
        }


_default: Calibrator | None = None


def calibrator() -> Calibrator:
    """Process-wide calibrator (the engine holds this one instance)."""
    global _default
    if _default is None:
        _default = Calibrator()
    return _default
