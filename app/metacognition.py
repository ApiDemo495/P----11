"""§4 — Metacognition: the model auditing itself before it speaks.

Three jobs, in the order the LIMITLESS MASTER ARCHITECTURAL DIRECTIVE
specifies them:

1. **Pillar audit** — the prediction pillars (formula brain, physics,
   order-flow, committee …) each emit a direction and a confidence.
   Dispersion across pillars, cosine similarity of their direction
   vectors, and KL divergence between their probability distributions
   feed one report: when the pillars disagree, the metacognition layer
   RAISES the confidence floor for that cell and can force ABSTAIN.
   Disagreement is not averaged away — it is priced in.

2. **Rolling calibration** — per 16-cell Brier score, ECE (expected
   calibration error) and a Platt scaling fit, updated on settled
   outcomes only.  Adjustments are bounded to ±3 percentage points per
   update so a bad streak cannot swing the panel.

3. **Bounded reweighting** — Exp3-IX bandit weights over the 16 cells
   with L2 shrinkage toward uniform, updated ONLY when a window has
   SETTLED (closed trade / scored prediction).  Live in-flight windows
   never move the weights.

Plus the honesty gate: **nested purged walk-forward with a locked
holdout and Deflated Sharpe Ratio > 0.90** decides which cells are
CONFIRMED.  Unconfirmed cells run SHADOW — visible, scored, no
capital.  In-sample numbers are never reported as forward claims.

No future-bar leakage: every update consumes only outcomes that have
already settled, and the holdout slice is never read during fitting.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Sequence

from .config import SYMBOL_ORDER

# ── the 16 Θ(a,τ) cells (mirrors app.cell_registry ids) ──────────────
CELL_IDS: list[tuple[str, str]] = [
    (sym, tf) for sym in SYMBOL_ORDER
    for tf in ("tf1", "tf2", "tf3", "tf4")
]

#: cells below this DSR run SHADOW (no capital) — directive §4
DSR_CONFIRM = 0.90
#: Platt / floor adjustments per update are capped at ±3 pp (directive)
CALIB_STEP_MAX = 0.03
#: Exp3-IX exploration floor (bounded — weights never collapse to zero)
EXP3_GAMMA_MIN = 0.05
#: L2 shrink strength toward the uniform weight vector
L2_SHRINK = 0.01
#: purged walk-forward defaults
WF_OUTER_TEST = 0.20        # locked holdout fraction (never read in fit)
WF_INNER_FOLDS = 5
WF_PURGE_BARS = 64           # embargo between train and test slices


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation).

    Bug fixed 2026-11-26: `deflated_sharpe` needed a QUANTILE - the z at
    which P(Z < z) = 1 - 1/N - and called `_norm_cdf` instead, which is the
    CDF. For N = 10 .. 6600 that expression returns 0.82 .. 0.84 no matter
    what N is, so the expected-maximum-Sharpe haircut `sr0` was a constant
    ~0.019 and the Deflated Sharpe Ratio printed 1.0000 ("CONFIRMED") for a
    single test and for a 6600-configuration search alike. Multiple-testing
    correction that does not depend on the number of tests is not a
    correction.
    """
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                 + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
            + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
                            + b[4]) * r + 1)


# ══════════════════════════════════════════════════════════════════════
# 1 · pillar audit — dispersion / cosine / KL
# ══════════════════════════════════════════════════════════════════════
@dataclass
class PillarReading:
    """One pillar's vote: direction in [-1, +1] (sign = side, |·| =
    conviction) and a probability p(up) in (0, 1)."""
    name: str
    direction: float
    p_up: float
    weight: float = 1.0


@dataclass
class PillarReport:
    dispersion: float            # weighted std of pillar directions
    cosine: float                # cosine similarity of direction vectors
    kl: float                    # mean KL(pillar ‖ mixture) in nats
    n_pillars: int
    elevated_floor: float        # confidence floor AFTER the audit
    abstain: bool
    reasons: list[str] = field(default_factory=list)
    base_floor: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def audit_pillars(pillars: Sequence[PillarReading],
                  base_floor: float = 0.15,
                  dispersion_gate: float = 0.35,
                  cos_gate: float = 0.25,
                  kl_gate: float = 0.35) -> PillarReport:
    """Dispersion + cosine + KL audit of the pillar crowd.

    High dispersion (pillars point different ways), low cosine
    agreement, or high KL (one pillar's distribution is far from the
    mixture) each RAISE the confidence floor; past all three gates the
    report abstains — the cell does not issue on a divided board.
    """
    reasons: list[str] = []
    live = [p for p in pillars if p.weight > 0]
    if not live:
        return PillarReport(
            dispersion=1.0, cosine=0.0, kl=1.0, n_pillars=0,
            elevated_floor=min(1.0, base_floor + 0.30), abstain=True,
            reasons=["no pillars reading — board empty"], base_floor=base_floor)

    wsum = sum(max(0.0, p.weight) for p in live) or 1.0
    mean_dir = sum(p.direction * p.weight for p in live) / wsum
    var = sum(p.weight * (p.direction - mean_dir) ** 2 for p in live) / wsum
    dispersion = math.sqrt(max(0.0, var))

    # cosine over the 2-vector set (side, conviction) flattened: treat
    # each pillar as a scalar direction and compare against the mean —
    # equivalent to the sign-agreement cosine in 1-D
    dots = sum(p.direction * p.weight for p in live)
    nrm = math.sqrt(sum(p.direction ** 2 * p.weight for p in live)) * \
        math.sqrt(sum(p.weight for p in live))
    cosine = (dots / nrm) if nrm > 1e-12 else 0.0

    # KL(pillar ‖ mixture) with a shared two-outcome mixture
    mix_up = sum(_clamp(p.p_up, 1e-6, 1 - 1e-6) * p.weight for p in live) / wsum
    mix = (1.0 - mix_up, mix_up)
    kl = 0.0
    for p in live:
        q = _clamp(p.p_up, 1e-6, 1 - 1e-6)
        kl += p.weight * (
            q * math.log(q / max(mix[1], 1e-12))
            + (1 - q) * math.log((1 - q) / max(mix[0], 1e-12))
        ) / wsum

    floor = base_floor
    if dispersion > dispersion_gate:
        bump = min(0.25, (dispersion - dispersion_gate) * 0.8)
        floor += bump
        reasons.append(
            f"pillar dispersion {dispersion:.2f} > {dispersion_gate} "
            f"(+{bump:.2f} floor)")
    if cosine < cos_gate:
        bump = min(0.20, (cos_gate - cosine) * 0.6)
        floor += bump
        reasons.append(
            f"cosine agreement {cosine:.2f} < {cos_gate} (+{bump:.2f} floor)")
    if kl > kl_gate:
        bump = min(0.20, (kl - kl_gate) * 0.6)
        floor += bump
        reasons.append(
            f"KL divergence {kl:.2f} > {kl_gate} (+{bump:.2f} floor)")

    floor = _clamp(floor, 0.0, 0.95)
    # abstain when the board is genuinely divided: wide dispersion AND
    # poor agreement — a mildly spread board only pays a higher floor
    abstain = (dispersion > dispersion_gate * 1.5
               and cosine < cos_gate * 0.5) or len(live) < 2 and dispersion > 0.5
    if abstain:
        reasons.append("divided board — ABSTAIN until pillars converge")

    return PillarReport(
        dispersion=round(dispersion, 4), cosine=round(cosine, 4),
        kl=round(kl, 4), n_pillars=len(live),
        elevated_floor=round(floor, 4), abstain=abstain,
        reasons=reasons, base_floor=base_floor)


# ══════════════════════════════════════════════════════════════════════
# 2 · rolling calibration — Brier / ECE / Platt (±3 pp bounded)
# ══════════════════════════════════════════════════════════════════════
def brier(p: float, y: int) -> float:
    return (p - y) ** 2


def ece(confidences: Sequence[float], outcomes: Sequence[int],
        bins: int = 10) -> float:
    """Expected calibration error over equal-width confidence bins."""
    n = len(confidences)
    if n == 0:
        return 0.0
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i in range(n)
               if (confidences[i] >= lo and
                   (confidences[i] < hi or (b == bins - 1 and confidences[i] <= hi)))]
        if not idx:
            continue
        acc = sum(outcomes[i] for i in idx) / len(idx)
        conf = sum(confidences[i] for i in idx) / len(idx)
        tot += (len(idx) / n) * abs(acc - conf)
    return tot


def platt_step(p: float, offset: float, step: float = CALIB_STEP_MAX) -> float:
    """One bounded Platt-style step: nudge the logit by `offset` but
    never move the printed probability more than ±CALIB_STEP_MAX."""
    p = _clamp(p, 1e-4, 1 - 1e-4)
    logit = math.log(p / (1 - p)) + offset
    q = 1.0 / (1.0 + math.exp(-_clamp(logit, -20, 20)))
    q = _clamp(q, p - step, p + step)
    return _clamp(q, 1e-4, 1 - 1e-4)


@dataclass
class CellCalibration:
    brier_sum: float = 0.0
    n: int = 0
    offset: float = 0.0          # Platt logit offset (bounded)
    ece_hist: list[float] = field(default_factory=list)

    @property
    def brier_mean(self) -> float:
        return self.brier_sum / self.n if self.n else 0.0

    @property
    def ece_last(self) -> float:
        return self.ece_hist[-1] if self.ece_hist else 0.0

    def calibrate(self, p: float) -> float:
        return platt_step(p, self.offset)


class CalibrationBook:
    """Per-cell rolling Brier/ECE/Platt — updates on SETTLED outcomes."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cells: dict[tuple[str, str], CellCalibration] = {}
        self._recent: dict[tuple[str, str], tuple[list[float],
                                                  list[int]]] = {}

    def record(self, symbol: str, tf: str, p: float, y: int) -> None:
        """Feed one settled (forecast, outcome) pair."""
        key = (symbol, tf)
        with self._lock:
            cell = self._cells.setdefault(key, CellCalibration())
            cell.brier_sum += brier(_clamp(p, 0.0, 1.0), 1 if y else 0)
            cell.n += 1
            r = self._recent.setdefault(key, ([], []))
            r[0].append(_clamp(p, 0.0, 1.0))
            r[1].append(1 if y else 0)
            if len(r[0]) > 200:
                r[0] = r[0][-200:]
                r[1] = r[1][-200:]
            # bounded Platt step toward the recent empirical frequency
            if len(r[0]) >= 20:
                recent_conf = sum(r[0][-20:]) / 20
                recent_acc = sum(r[1][-20:]) / 20
                gap = _clamp(recent_acc - recent_conf, -CALIB_STEP_MAX,
                             CALIB_STEP_MAX)
                # move offset part-way; calibrate() itself re-clamps ±3pp
                cell.offset = _clamp(cell.offset + 0.5 * gap, -1.5, 1.5)
                cell.ece_hist.append(ece(r[0], r[1]))
                if len(cell.ece_hist) > 50:
                    cell.ece_hist = cell.ece_hist[-50:]

    def calibrate(self, symbol: str, tf: str, p: float) -> float:
        with self._lock:
            cell = self._cells.get((symbol, tf))
            return cell.calibrate(p) if cell else _clamp(p, 1e-4, 1 - 1e-4)

    def stats(self, symbol: str, tf: str) -> dict:
        with self._lock:
            cell = self._cells.get((symbol, tf))
            if not cell:
                return {"brier": None, "ece": None, "n": 0, "offset": 0.0}
            return {"brier": round(cell.brier_mean, 5),
                    "ece": round(cell.ece_last, 5),
                    "n": cell.n, "offset": round(cell.offset, 4)}


# ══════════════════════════════════════════════════════════════════════
# 3 · bounded Exp3-IX + L2 shrink (settled windows only)
# ══════════════════════════════════════════════════════════════════════
class Exp3Ix:
    """Bounded Exp3-IX over the 16 cells with L2 shrink to uniform.

    Weight updates happen ONLY via `update` when a window has settled.
    The exploration floor keeps every cell's weight ≥ EXP3_GAMMA_MIN/K,
    and each step shrinks the weight vector toward uniform (L2) so a
    hot streak cannot monopolise the capital allocation.
    """

    def __init__(self, k: int = 16, gamma: float = 0.10,
                 eta: float = 4.0) -> None:
        self.k = k
        self.gamma = _clamp(gamma, EXP3_GAMMA_MIN, 0.5)
        self.eta = eta
        self._lock = threading.RLock()
        self.w = [1.0 / k] * k
        self._last_update: float = 0.0

    def weights(self) -> list[float]:
        with self._lock:
            return list(self.w)

    def allocate(self) -> list[float]:
        """Exploration-mixed allocation (what actually gets capital)."""
        with self._lock:
            return [(1.0 - self.gamma) * wi + self.gamma / self.k
                    for wi in self.w]

    def update(self, idx: int, reward: float,
               now: float | None = None) -> None:
        """One SETTLED-window update. reward ∈ [0, 1]."""
        if not (0 <= idx < self.k):
            return
        reward = _clamp(reward, 0.0, 1.0)
        now = now if now is not None else time.time()
        with self._lock:
            alloc = self.allocate()
            p_i = max(alloc[idx], 1e-9)
            # Exp3-IX importance-weighted estimate
            est = reward / p_i
            self.w[idx] = self.w[idx] * math.exp(self.eta * self.gamma
                                                 * est / self.k)
            # L2 shrink toward uniform + normalise
            u = 1.0 / self.k
            self.w = [(1 - L2_SHRINK) * wi + L2_SHRINK * u for wi in self.w]
            tot = sum(self.w) or 1.0
            self.w = [wi / tot for wi in self.w]
            # floor: no cell may fall below the exploration minimum
            floor = EXP3_GAMMA_MIN / self.k
            self.w = [max(wi, floor) for wi in self.w]
            tot = sum(self.w) or 1.0
            self.w = [wi / tot for wi in self.w]
            self._last_update = now

    @property
    def last_update(self) -> float:
        return self._last_update


# ══════════════════════════════════════════════════════════════════════
# 4 · nested purged walk-forward + locked holdout + DSR
# ══════════════════════════════════════════════════════════════════════
def sharpe_ratio(returns: Sequence[float]) -> float:
    n = len(returns)
    if n < 2:
        return 0.0
    m = sum(returns) / n
    var = sum((r - m) ** 2 for r in returns) / (n - 1)
    sd = math.sqrt(var)
    if sd <= 1e-12:
        return 0.0
    # per-period SR; scale to annualised only at reporting time
    return m / sd


def deflated_sharpe(observed_sr: float, n_obs: int, n_trials: int,
                    skew: float = 0.0, kurt: float = 3.0) -> float:
    """Bailey–López de Prado Deflated Sharpe Ratio.

    P(SR* < SR_obs) under the max-of-N-trials null, accounting for
    non-normal returns (skew, excess kurtosis).  DSR > 0.90 is the
    CONFIRM gate of the directive.
    """
    if n_obs < 3:
        return 0.0
    # expected max SR under N independent trials (Mertens / Bailey)
    if n_trials <= 1:
        sr0 = 0.0
    else:
        # expected max SR under N trials: SR0 = sqrt(V[SR*])·((1−γ)z₁+γz₂)
        v_sr = 1.0 / max(1.0, n_obs - 1)
        g = 0.5772156649
        z1 = _norm_ppf(1 - 1.0 / n_trials)
        z2 = _norm_ppf(1 - 1.0 / (n_trials * math.e))
        sr0 = math.sqrt(v_sr) * ((1 - g) * z1 + g * z2)
    denom_base = 1.0 - skew * observed_sr + (kurt - 1) / 4.0 * observed_sr ** 2
    if denom_base <= 1e-12 or n_obs < 3:
        return 0.0
    se = math.sqrt(max(denom_base, 1e-12) / max(1.0, n_obs - 1))
    if se <= 1e-12:
        return 0.0
    return _norm_cdf((observed_sr - sr0) / se)


@dataclass
class CellVerdict:
    symbol: str
    tf: str
    status: str                 # "CONFIRMED" | "SHADOW"
    dsr: float
    oof_win_rate: float | None
    oof_net_r: float | None
    oof_pf: float | None
    holdout_net_r: float | None
    n_oof: int
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _fold_slices(n: int, folds: int, purge: int,
                 test_frac: float) -> list[tuple[list[int], list[int]]]:
    """Nested purged folds over indices [0, n).

    Returns (train_idx, test_idx) pairs.  The OUTER locked holdout is
    the last `test_frac` of the series and is sliced off before the
    inner folds are built — callers must never fit on it.
    """
    hold = int(n * test_frac)
    usable = n - hold
    slices: list[tuple[list[int], list[int]]] = []
    step = max(1, usable // (folds + 1))
    for f in range(folds):
        t0 = f * step
        t1 = min(usable, t0 + step)
        if t1 - t0 < 8:
            continue
        # purge: drop `purge` bars before the test window from train
        train = list(range(0, max(0, t0 - purge))) + \
            list(range(t1, usable))
        test = list(range(t0, t1))
        if len(train) >= 16 and len(test) >= 4:
            slices.append((train, test))
    return slices


def nested_purged_walk_forward(
    returns_by_cell: dict[tuple[str, str], list[float]],
    settle_fn: Callable[[tuple[str, str], int], float] | None = None,
    folds: int = WF_INNER_FOLDS,
    purge: int = WF_PURGE_BARS,
    holdout_frac: float = WF_OUTER_TEST,
    n_trials: int | None = None,
) -> dict[tuple[str, str], CellVerdict]:
    """Nested purged WF per cell + locked holdout + DSR gate.

    `returns_by_cell` maps a cell to its per-settled-window return series
    (already purged of lookahead by construction — one entry per
    closed window).  The LAST `holdout_frac` of each series is the
    locked holdout: it is scored but never used to pick anything.
    CONFIRMED requires inner out-of-fold performance that survives the
    DSR > 0.90 gate against `n_trials` multiple testing (default: all
    16 cells × the folds actually evaluated).
    """
    if n_trials is None:
        n_trials = max(16, folds * max(1, len(returns_by_cell)))
    out: dict[tuple[str, str], CellVerdict] = {}
    for (sym, tf), rets in returns_by_cell.items():
        n = len(rets)
        reasons: list[str] = []
        if n < 30:
            out[(sym, tf)] = CellVerdict(
                sym, tf, "SHADOW", 0.0, None, None, None, None, n,
                [f"only {n} settled windows — needs ≥30 before any "
                 f"capital (SHADOW)"])
            continue
        hold = int(n * holdout_frac)
        train_series = rets[:n - hold] if hold > 0 else list(rets)
        holdout = rets[n - hold:] if hold > 0 else []
        # inner nested purged folds on the pre-holdout prefix
        slices = _fold_slices(len(train_series), folds, purge, 0.0)
        oof_rets: list[float] = []
        fold_srs: list[float] = []
        for train_idx, test_idx in slices:
            # inner selection: mean of train (the model choice is fixed
            # upstream; here we verify the slice is profitable at all)
            tr = [train_series[i] for i in train_idx]
            if not tr:
                continue
            te = [train_series[i] for i in test_idx]
            oof_rets.extend(te)
            fold_srs.append(sharpe_ratio(te))
        if len(oof_rets) < 8:
            out[(sym, tf)] = CellVerdict(
                sym, tf, "SHADOW", 0.0, None, None, None,
                None, n, ["inner folds too small after purge (SHADOW)"])
            continue
        wins = sum(1 for r in oof_rets if r > 0)
        gross_w = sum(r for r in oof_rets if r > 0)
        gross_l = -sum(r for r in oof_rets if r < 0)
        pf = (gross_w / gross_l) if gross_l > 1e-12 else (
            99.0 if gross_w > 0 else 0.0)
        oof_net = sum(oof_rets)
        oof_wr = wins / len(oof_rets)
        sr = sharpe_ratio(oof_rets)
        skew = 0.0
        kurt = 3.0
        if len(oof_rets) >= 8:
            m = sum(oof_rets) / len(oof_rets)
            sd = math.sqrt(sum((r - m) ** 2 for r in oof_rets)
                           / len(oof_rets)) or 1e-12
            skew = sum(((r - m) / sd) ** 3 for r in oof_rets) / len(oof_rets)
            kurt = sum(((r - m) / sd) ** 4 for r in oof_rets) / len(oof_rets)
        dsr = deflated_sharpe(sr, len(oof_rets), n_trials, skew, kurt)
        hold_net = sum(holdout) if holdout else None
        # gate: DSR above threshold AND positive OOF net AND PF ≥ 1.0
        if dsr > DSR_CONFIRM and oof_net > 0 and pf >= 1.0:
            status = "CONFIRMED"
        else:
            status = "SHADOW"
            if dsr <= DSR_CONFIRM:
                reasons.append(f"DSR {dsr:.3f} ≤ {DSR_CONFIRM} (SHADOW)")
            if oof_net <= 0:
                reasons.append("non-positive OOF net (SHADOW)")
            if pf < 1.0:
                reasons.append(f"PF {pf:.2f} < 1.0 (SHADOW)")
        if status == "CONFIRMED":
            reasons.append(f"DSR {dsr:.3f} > {DSR_CONFIRM} on purged OOF")
        out[(sym, tf)] = CellVerdict(
            sym, tf, status, round(dsr, 4),
            round(oof_wr, 4) if oof_wr is not None else None,
            round(oof_net, 4), round(pf, 3),
            round(hold_net, 4) if hold_net is not None else None,
            n, reasons)
    return out


# ══════════════════════════════════════════════════════════════════════
# façade
# ══════════════════════════════════════════════════════════════════════
class Metacognition:
    """Singleton wiring: pillar audit → calibration → reweighting →
    CONFIRM/SHADOW gate.  Persisted as JSON under backtest/results/."""

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        self.calib = CalibrationBook()
        self.exp3 = Exp3Ix(k=len(CELL_IDS))
        self.verdicts: dict[tuple[str, str], CellVerdict] = {}
        self.shadow_only: set[tuple[str, str]] = set(CELL_IDS)  # until proven
        self._persist = Path(persist_path) if persist_path else None
        self._load()

    # ── pillar audit ────────────────────────────────────────────────
    def audit(self, pillars: Sequence[PillarReading],
              base_floor: float = 0.15) -> PillarReport:
        return audit_pillars(pillars, base_floor=base_floor)

    def apply_audit(self, symbol: str, tf: str, report: PillarReport,
                    confidence: float) -> tuple[float, bool]:
        """Raise the cell's confidence to the audited floor; return
        (confidence, abstain)."""
        c = max(float(confidence), report.elevated_floor)
        return _clamp(c, 0.0, 1.0), report.abstain

    # ── settled-window updates ─────────────────────────────────────
    def calibrate(self, symbol: str, tf: str, p: float) -> float:
        return self.calib.calibrate(symbol, tf, p)

    def cell_weights(self) -> dict[str, float]:
        alloc = self.exp3.allocate()
        return {f"{s}|{t}": round(alloc[i], 5)
                for i, (s, t) in enumerate(CELL_IDS)
                if i < len(alloc)}

    # ── CONFIRM / SHADOW gate ──────────────────────────────────────
    def evaluate(self, returns_by_cell: dict[tuple[str, str], list[float]]
                 ) -> dict[tuple[str, str], CellVerdict]:
        verdicts = nested_purged_walk_forward(returns_by_cell)
        with self._lock:
            self.verdicts = verdicts
            self.shadow_only = {
                k for k, v in verdicts.items() if v.status != "CONFIRMED"}
            # cells with no verdict yet stay shadow
            for k in CELL_IDS:
                if k not in verdicts:
                    self.shadow_only.add(k)
        self._save()
        return verdicts

    def status(self, symbol: str, tf: str) -> str:
        """CONFIRMED only through the DSR gate — default SHADOW."""
        if (symbol, tf) in self.shadow_only:
            return "SHADOW"
        v = self.verdicts.get((symbol, tf))
        return "CONFIRMED" if v and v.status == "CONFIRMED" else "SHADOW"

    def allows_capital(self, symbol: str, tf: str) -> bool:
        """SHADOW cells run no capital (directive §4)."""
        return self.status(symbol, tf) == "CONFIRMED"

    def _idx(self, symbol: str, tf: str) -> int | None:
        try:
            return CELL_IDS.index((symbol, tf))
        except ValueError:
            return None

    # ── persistence (verdicts + weights only — no raw series) ───────
    def _save(self) -> None:
        if not self._persist:
            return
        try:
            self._persist.parent.mkdir(parents=True, exist_ok=True)
            blob = {
                "saved_at": time.time(),
                "verdicts": {f"{s}|{t}": v.as_dict()
                             for (s, t), v in self.verdicts.items()},
                "weights": self.cell_weights(),
                "shadow_only": sorted(f"{s}|{t}" for s, t in self.shadow_only),
                "calibration": {f"{s}|{t}": self.calib.stats(s, t)
                                for s, t in CELL_IDS},
            }
            self._persist.write_text(json.dumps(blob, indent=1))
        except Exception:
            pass

    def _load(self) -> None:
        if not self._persist or not self._persist.exists():
            return
        try:
            blob = json.loads(self._persist.read_text())
            for k, d in (blob.get("verdicts") or {}).items():
                sym, tf = k.split("|", 1)
                self.verdicts[(sym, tf)] = CellVerdict(**d)
            self.shadow_only = {
                tuple(k.split("|", 1))  # type: ignore[misc]
                for k in (blob.get("shadow_only") or [])
            } or set(CELL_IDS)
        except Exception:
            self.shadow_only = set(CELL_IDS)


#: process-wide singleton (engine + main both talk to this one)
MC = Metacognition(
    persist_path=Path(__file__).resolve().parent.parent
    / "backtest" / "results" / "metacognition.json")
