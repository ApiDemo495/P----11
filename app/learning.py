"""Section 10 — the permanently active auto-learning system.

Three timescales:
  * MICRO  — every GRC (η updates live in cci/emotion/physics/brain; this
             module keeps the accuracy windows, brain influence weights and
             adaptive rates that drive them)
  * MACRO  — every 15 GRCs (probation/bonus, redundancy pruning, Brain-4
             biological loop, Weibull + λ_critical recalibration)
  * DEEP   — every 247 minutes (diagnostic mode, Bgn anchor, HMM retrain,
             Weibull × 5 regimes, subsystem weight re-shares)

Everything is persisted to LEARN_STATE_FILE so learning resumes exactly
where it left off between sessions.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import deque

from .config import (
    ACCURACY_ROLLING,
    DEEP_DIAGNOSTIC_HR,
    DEEP_PERIOD_MINUTES,
    DEEP_RIGHT_BOOST,
    DEEP_WRONG_CUT,
    EMOTION_W_MAX,
    EMOTION_W_MIN,
    EMOTION_COMPONENT_ID,
    ETA_BRAIN,
    ETA_EMOTION,
    ETA_MICRO,
    ETA_PHYSICS,
    HEDGE_COMPONENT_ID,
    HIGH_ALERT_HIGH,
    HIGH_ALERT_LOW,
    HIGH_ALERT_PROBATION,
    BRAIN_COMPONENT_ID,
    LEARN_STATE_FILE,
    MACRO_ACC_WINDOW,
    MACRO_BONUS,
    MACRO_CORR_REDUNDANCY,
    MACRO_CUT,
    MACRO_PERIOD_GRC,
    MACRO_PROBATION,
    NEWS_COMPONENT_ID,
    PHYSICS_W_MAX,
    PHYSICS_W_MIN,
    PHYSICS_COMPONENT_ID,
    SUBSYSTEM_BEARINGS,
    SUBSYSTEM_SHARES_INIT,
    SUBSYSTEM_STEP,
    SYMBOL_ORDER,
    TF_ORDER,
    WEIGHT_MAX,
    WEIGHT_MIN,
    WEIGHT_IDS,
    CORE_SIGNAL_IDS,
)

_EPOCH = deque


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _pearson(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = a[-n:], b[-n:]
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 1e-18 or vb <= 1e-18:
        return 0.0
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return max(-1.0, min(1.0, cov / math.sqrt(va * vb)))


class HMM2:
    """2-state Gaussian HMM with Baum-Welch (Section 10, deep step 6).

    Kept as MSRC's regime-context module; the MSRC formula itself is the
    exact Part-1 transcription and is NOT modified by this trainer.
    """

    def __init__(self) -> None:
        self.pi = [0.5, 0.5]
        self.A = [[0.9, 0.1], [0.1, 0.9]]
        self.mu = [0.0, 0.0]
        self.var = [1e-4, 1e-4]
        self.n_iter = 0
        self.last_loglik = None

    def _norm_pdf(x: float, mu: float, var: float) -> float:
        var = max(var, 1e-12)
        return math.exp(-0.5 * (x - mu) ** 2 / var) / math.sqrt(2 * math.pi * var)

    def fit(self, series: list[float], iters: int = 8) -> float:
        """Baum-Welch; returns the final mean log-likelihood per step."""
        xs = [float(x) for x in series[-247:]]     # full TF4 window
        n = len(xs)
        if n < 12:
            return float("nan")
        # init by split
        srt = sorted(xs)
        self.mu = [srt[n // 4], srt[(3 * n) // 4]]
        self.var = [max(1e-10, _var(xs, self.mu[0])), max(1e-10, _var(xs, self.mu[1]))]
        ll = float("nan")
        for _ in range(iters):
            # E-step (scaled forward-backward)
            alpha = [[0.0, 0.0] for _ in range(n)]
            beta = [[0.0, 0.0] for _ in range(n)]
            c = [0.0] * n
            for i, x in enumerate(xs):
                a_i = [
                    (self.pi[s] if i == 0 else alpha[i - 1][0] * self.A[0][s]
                     + alpha[i - 1][1] * self.A[1][s]) * self._pdf(x, s)
                    for s in (0, 1)
                ]
                c[i] = max(1e-300, a_i[0] + a_i[1])
                alpha[i] = [a_i[0] / c[i], a_i[1] / c[i]]
            beta[n - 1] = [1.0 / c[n - 1], 1.0 / c[n - 1]]
            for i in range(n - 2, -1, -1):
                beta[i] = [
                    sum(self.A[s][j] * self._pdf(xs[i + 1], j) * beta[i + 1][j]
                        for j in (0, 1)) / c[i + 1]
                    for s in (0, 1)
                ]
            # gammas + xi
            gam = []
            for i in range(n):
                g = [alpha[i][s] * beta[i][s] for s in (0, 1)]
                tot = g[0] + g[1] or 1e-300
                gam.append([g[0] / tot, g[1] / tot])
            # M-step
            pi0 = gam[0]
            a_num = [[0.0, 0.0], [0.0, 0.0]]
            a_den = [0.0, 0.0]
            for i in range(n - 1):
                for s in (0, 1):
                    for j in (0, 1):
                        xi = (alpha[i][s] * self.A[s][j]
                              * self._pdf(xs[i + 1], j) * beta[i + 1][j])
                        a_num[s][j] += xi
                    a_den[s] += a_num[s][0] + a_num[s][1]
            for s in (0, 1):
                if a_den[s] > 1e-300:
                    self.A[s] = [a_num[s][j] / a_den[s] for j in (0, 1)]
            for s in (0, 1):
                gw = sum(gam[i][s] for i in range(n))
                if gw > 1e-12:
                    self.mu[s] = sum(gam[i][s] * xs[i] for i in range(n)) / gw
                    self.var[s] = max(
                        1e-12,
                        sum(gam[i][s] * (xs[i] - self.mu[s]) ** 2
                            for i in range(n)) / gw,
                    )
            self.pi = pi0
            ll = sum(math.log(x) for x in c) / n
        self.n_iter += 1
        self.last_loglik = ll
        return ll

    def _pdf(self, x: float, s: int) -> float:
        var = max(self.var[s], 1e-12)
        return math.exp(-0.5 * (x - self.mu[s]) ** 2 / var) / math.sqrt(
            2 * math.pi * var)

    def state_probability(self, series: list[float]) -> float:
        """P(state=1 | last observation) — the 'hot' regime posterior."""
        if not series:
            return 0.5
        p = [self.pi[s] * self._pdf(float(series[-1]), s) for s in (0, 1)]
        tot = p[0] + p[1]
        return p[1] / tot if tot > 0 else 0.5

    def to_dict(self) -> dict:
        return {"pi": self.pi, "A": self.A, "mu": self.mu, "var": self.var,
                "n_iter": self.n_iter, "last_loglik": self.last_loglik}

    def load_dict(self, d: dict) -> None:
        try:
            self.pi = [float(x) for x in d["pi"]]
            self.A = [[float(x) for x in row] for row in d["A"]]
            self.mu = [float(x) for x in d["mu"]]
            self.var = [float(x) for x in d["var"]]
            self.n_iter = int(d.get("n_iter", 0))
            self.last_loglik = d.get("last_loglik")
        except (KeyError, TypeError, ValueError):
            pass


def _var(xs: list[float], mu: float) -> float:
    return sum((x - mu) ** 2 for x in xs) / max(1, len(xs))


class LearningSystem:
    """Owns every Section-10/14 learning state that outlives a single GRC."""

    def __init__(self) -> None:
        # --- adaptive rates (high-alert mode doubles them) ---------------
        self.rates = {"micro": ETA_MICRO, "brain": ETA_BRAIN,
                      "emotion": ETA_EMOTION, "physics": ETA_PHYSICS}
        self.high_alert = False
        self.probation_threshold = MACRO_PROBATION

        # --- accuracy windows -------------------------------------------
        # per-formula all-time correctness (last 500 scored GRCs)
        self.acc500: dict[int, deque] = {f: _EPOCH(maxlen=500)
                                         for f in WEIGHT_IDS}
        # per (formula, tf) and per (formula, asset) last-100 correctness
        self.acc_tf: dict[tuple[int, str], deque] = {}
        self.acc_sym: dict[tuple[int, str], deque] = {}
        # per-timeframe hit-rate windows for the accuracy target
        self.tf_hits: dict[str, deque] = {tf: _EPOCH(maxlen=ACCURACY_ROLLING)
                                          for tf in TF_ORDER}   # 1=tp,0=sl
        self.tf_hr: dict[str, float] = {tf: 0.5 for tf in TF_ORDER}

        # --- macro state -------------------------------------------------
        self.probation: dict[tuple[str, str, int], float] = {}  # until ts
        self.macro_tone = 0.0             # Brain-4 DN controller tone
        self.last_macro_grc = -10 ** 9
        self.macro_count = 0

        # --- deep state --------------------------------------------------
        self.last_deep_ts = 0.0
        self.deep_count = 0
        self.diagnostic_until: dict[tuple[str, str], float] = {}
        self.subsystem_shares: dict[str, float] = dict(SUBSYSTEM_SHARES_INIT)
        # agreement-contribution accumulators (deep steps 8-11)
        self.contrib: dict[str, list[int]] = {k: [0, 0, 0, 0]
                                              for k in
                                              self.subsystem_shares}  # a_ok,a_n,d_ok,d_n
        self.bgn_anchor: dict[str, float] = {}   # deep coherence anchor

        # --- brain influence weights (per pair, renormalized across 4) ---
        self.brain_w: dict[tuple[str, str], list[float]] = {}

        # --- HMMs + Weibull ----------------------------------------------
        self.hmm: dict[tuple[str, str], HMM2] = {}
        self.weibull: dict[str, tuple[float, float]] = {}  # sym|tf|regime

        # --- L3/L4 performance series (filled by engine each GRC) --------
        self.f_returns: dict[tuple[str, str], deque] = {}

        self.grc_count = 0
        self.last_macro_ts = 0.0

    # ==================================================================
    # helpers
    # ==================================================================
    def rates_for(self) -> dict[str, float]:
        return dict(self.rates)

    def _acc_ring(self, ring_map, key, maxlen):
        r = ring_map.get(key)
        if r is None:
            r = _EPOCH(maxlen=maxlen)
            ring_map[key] = r
        return r

    def hmm_for(self, symbol: str, tf: str) -> HMM2:
        key = (symbol, tf)
        if key not in self.hmm:
            self.hmm[key] = HMM2()
        return self.hmm[key]

    def brain_influence(self, symbol: str, tf: str) -> list[float]:
        return self.brain_w.get((symbol, tf), [0.25, 0.25, 0.25, 0.25])

    # ==================================================================
    # MICRO (called by the engine at every GRC)
    # ==================================================================
    def record_micro(self, symbol: str, tf: str, move_sign: int,
                     signals: dict[int, int], subsystems: dict[str, int],
                     brain_sigs: list[int],
                     target_hit: int | None, sl_hit: int | None) -> None:
        """Accuracy bookkeeping for one (asset, tf) pass at a GRC."""
        for fid, s in signals.items():
            if fid in self.acc500 and s:
                self.acc500[fid].append(1 if s == move_sign else 0)
            if fid and s:
                self._acc_ring(self.acc_tf, (fid, tf), ACCURACY_ROLLING).append(
                    1 if s == move_sign else 0)
                self._acc_ring(self.acc_sym, (fid, symbol), ACCURACY_ROLLING).append(
                    1 if s == move_sign else 0)
        # subsystem agreement contribution (feeds deep steps 8-11)
        final_ok = move_sign != 0 and subsystems.get("final", 0) == move_sign
        for name in ("emotion", "physics", "news", "hf", "brain", "core"):
            s = subsystems.get(name, 0)
            if not s:
                continue
            row = self.contrib.setdefault(name, [0, 0, 0, 0])
            if s == subsystems.get("final", 0):
                row[0] += 1 if final_ok else 0
                row[1] += 1
            else:
                row[2] += 1 if final_ok else 0
                row[3] += 1
        # per-brain micro rewards (h_signal is float -> compare signs)
        w = self.brain_w.setdefault((symbol, tf), [0.25] * 4)
        for b, sig in enumerate(brain_sigs):
            if sig:
                sgn = 1 if sig > 0 else (-1 if sig < 0 else 0)
                reward = 1.0 if sgn == move_sign else -1.0
                w[b] = _clamp(w[b] * (1.0 + self.rates["brain"] * reward),
                              0.1, 0.4)
        tot = sum(w)
        if tot > 0:
            for b in range(4):
                w[b] /= tot
        # hit-rate window (targets before SL, per timeframe)
        if target_hit is not None:
            self.tf_hits[tf].append(1)
        elif sl_hit is not None:
            self.tf_hits[tf].append(0)
        self._refresh_tf_hr()
        # return history for the MSRC HMM
        fr = self._acc_ring(self.f_returns, (symbol, tf), 62)

    def feed_return(self, symbol: str, tf: str, ret: float) -> None:
        r = self._acc_ring(self.f_returns, (symbol, tf), 62)
        r.append(float(ret))

    def _refresh_tf_hr(self) -> None:
        for tf in TF_ORDER:
            w = self.tf_hits[tf]
            self.tf_hr[tf] = (sum(w) / len(w)) if w else 0.5

    def update_adaptive_rates(self) -> None:
        """High-alert optimization mode (Section 10 accuracy target)."""
        hrs = list(self.tf_hr.values())
        worst = min(hrs) if hrs else 0.5
        best = min(hrs) if hrs else 0.5
        if not self.high_alert and worst < HIGH_ALERT_LOW:
            self.high_alert = True
        elif self.high_alert and best > HIGH_ALERT_HIGH:
            self.high_alert = False
        if self.high_alert:
            self.rates["micro"] = 0.04
            self.rates["brain"] = 0.02
            self.rates["emotion"] = 0.03
            self.rates["physics"] = 0.03
            self.probation_threshold = HIGH_ALERT_PROBATION
        else:
            self.rates["micro"] = ETA_MICRO
            self.rates["brain"] = ETA_BRAIN
            self.rates["emotion"] = ETA_EMOTION
            self.rates["physics"] = ETA_PHYSICS
            self.probation_threshold = MACRO_PROBATION

    def formula_accuracy(self, fid: int, tf: str | None = None) -> float | None:
        if tf is not None:
            r = self.acc_tf.get((fid, tf))
            return (sum(r) / len(r)) if r else None
        r = self.acc500.get(fid)
        return (sum(r) / len(r)) if r else None

    def performance_multiplier(self, fid: int, axis: str, key: str) -> float:
        """Level-3/4: perf_f_axis / avg_perf_f over the last 100 cycles."""
        ring_map = self.acc_tf if axis == "tf" else self.acc_sym
        r = ring_map.get((fid, key))
        if not r or len(r) < 5:
            return 1.0
        perf = sum(r) / len(r)
        peers = []
        for (f, k), ring in ring_map.items():
            if f == fid and len(ring) >= 5:
                peers.append(sum(ring) / len(ring))
        if not peers:
            return 1.0
        avg = sum(peers) / len(peers)
        if avg <= 1e-9:
            return 1.0
        return _clamp(perf / avg, 0.5, 2.0)

    # ==================================================================
    # MACRO (every 15 GRCs)
    # ==================================================================
    def maybe_macro(self, grc_count: int, engine) -> bool:
        if grc_count - self.last_macro_grc < MACRO_PERIOD_GRC:
            return False
        self.last_macro_grc = grc_count
        self.macro_cycle(engine)
        return True

    def macro_cycle(self, engine) -> None:
        self.macro_count += 1
        now = time.time()
        self.last_macro_ts = now
        for symbol in SYMBOL_ORDER:
            for tf in TF_ORDER:
                w = engine.cci.get_weights(symbol, tf)
                if not w:
                    continue
                # Step 1-3: probation / bonus on the rolling-15 accuracy
                flags = []
                for fid in list(w.keys()):
                    acc = self.formula_accuracy(fid, None)
                    if acc is None:
                        continue
                    # structural review uses this pair's tf window when
                    # available, else the global ring
                    acc_tf = self.formula_accuracy(fid, tf)
                    use = acc_tf if acc_tf is not None else acc
                    flags.append((fid, use))
                    if use < self.probation_threshold:
                        w[fid] = _clamp(w[fid] * 0.5, WEIGHT_MIN, WEIGHT_MAX)
                        self.probation[(symbol, tf, fid)] = now + DEEP_PERIOD_MINUTES * 60
                    elif use > MACRO_BONUS:
                        bonus = MACRO_BONUS * (1.0 + self.macro_tone)
                        w[fid] = _clamp(w[fid] * (1.0 + bonus * 0.25),
                                        WEIGHT_MIN, WEIGHT_MAX)
                # Step 4: redundancy pruning via output correlation
                core = [f for f in CORE_SIGNAL_IDS if f in w]
                for i in range(len(core)):
                    for j in range(i + 1, len(core)):
                        fi, fj = core[i], core[j]
                        ri = engine.hist.ring(
                            engine.hist.tf_key(symbol, tf, f"fspark{fi}"), 15)
                        rj = engine.hist.ring(
                            engine.hist.tf_key(symbol, tf, f"fspark{fj}"), 15)
                        if len(ri) < MACRO_ACC_WINDOW:
                            continue
                        if abs(_pearson(list(ri), list(rj))) > MACRO_CORR_REDUNDANCY:
                            ai = self.formula_accuracy(fi, tf)
                            aj = self.formula_accuracy(fj, tf)
                            if ai is None or aj is None:
                                continue
                            loser = fi if (ai or 0) <= (aj or 0) else fj
                            w[loser] = _clamp(
                                w[loser] * (1.0 - MACRO_CUT), WEIGHT_MIN, WEIGHT_MAX)
                engine.cci.weights[(symbol, tf)] = w
                # Section 14: renormalize the row to sum 1 after weight edits
                _tot = sum(w.values())
                if _tot > 0:
                    for _fid in w:
                        w[_fid] /= _tot
                engine.cci.save()
        # Step 5: Brain-4 biological loop — FSB receives the macro metrics
        metrics = {}
        for fid in CORE_SIGNAL_IDS:
            a = self.formula_accuracy(fid)
            if a is not None:
                metrics[fid] = a
        self.macro_tone = engine.brains.macro_feedback(metrics)
        # Step 6/7: Weibull + λ_critical come from the live stores the
        # formulas already append to each GRC (regime durations / streaks);
        # refresh the persisted snapshots here.
        self._snapshot_weibull(engine)
        self.update_adaptive_rates()

    def _snapshot_weibull(self, engine) -> None:
        try:
            from .formulas.f11_f20 import _weibull_k_mle
        except Exception:
            return
        rd = getattr(engine.hist, "regime_durations", {})
        for (symbol, tf, regime), durs in rd.items():
            if len(durs) >= 8:
                k = _weibull_k_mle([float(d) for d in durs])
                lam = ((sum(float(d) ** k for d in durs) / len(durs))
                       ** (1.0 / max(k, 1e-6)))
                self.weibull[f"{symbol}|{tf}|{regime}"] = (round(k, 5),
                                                           round(lam, 5))

    # ==================================================================
    # DEEP (every 247 minutes)
    # ==================================================================
    def maybe_deep(self, engine) -> bool:
        now = time.time()
        if self.last_deep_ts and now - self.last_deep_ts < DEEP_PERIOD_MINUTES * 60:
            return False
        if not self.last_deep_ts:
            self.last_deep_ts = now       # first boot: anchor at start
            return False
        self.last_deep_ts = now
        self.deep_cycle(engine)
        return True

    def deep_cycle(self, engine) -> None:
        self.deep_count += 1
        # --- Steps 1-4: per-timeframe hit rates + diagnostic mode -------
        for tf in TF_ORDER:
            hr = self.tf_hr.get(tf, 0.5)
            if hr >= DEEP_DIAGNOSTIC_HR:
                self.diagnostic_until.pop(tf, None)
                continue
            # diagnostic: rank formulas by weight×direction inside this tf
            wrong: list[tuple[float, int]] = []
            rescued: list[tuple[float, int]] = []
            for symbol in SYMBOL_ORDER:
                w = engine.cci.get_weights(symbol, tf) or {}
                for fid, wv in w.items():
                    ring = self.acc_tf.get((fid, tf))
                    if not ring:
                        continue
                    # wrong-prediction episodes: scored 0 more often than 1
                    wrongness = 1.0 - (sum(ring) / len(ring))
                    wrong.append((wv * wrongness, fid))
                    rescued.append(((sum(ring) / len(ring)) * wv, fid))
            wrong.sort(reverse=True)
            rescued.sort(reverse=True)
            bad = _unique_top([f for _, f in wrong], 3)
            good = _unique_top([f for _, f in rescued
                                if f not in bad], 3)
            until = now_until()
            for symbol in SYMBOL_ORDER:
                w = engine.cci.get_weights(symbol, tf) or {}
                for fid in bad:
                    if fid in w:
                        w[fid] = _clamp(w[fid] * (1.0 - DEEP_WRONG_CUT),
                                        WEIGHT_MIN, WEIGHT_MAX)
                for fid in good:
                    if fid in w:
                        w[fid] = _clamp(w[fid] * (1.0 + DEEP_RIGHT_BOOST),
                                        WEIGHT_MIN, WEIGHT_MAX)
                engine.cci.weights[(symbol, tf)] = w
                # Section 14: renormalize the row to sum 1 after weight edits
                _tot = sum(w.values())
                if _tot > 0:
                    for _fid in w:
                        w[_fid] /= _tot
                engine.cci.save()
            # brain affinity inhibitory input for the next 247 minutes
            engine.brains.diagnostic_inhibit(tf, bad, until)
            self.diagnostic_until[(tf,)] = until
        # --- Step 5: deep Bgn coherence anchor (247-min RPIC history) ----
        anchor = {}
        for tf in TF_ORDER:
            series = []
            for symbol in SYMBOL_ORDER:
                r = engine.hist.ring(engine.hist.tf_key(symbol, tf, "fspark21"), 62)
                series.append(list(r))
            n = min((len(s) for s in series), default=0)
            row = {}
            for i in range(len(TF_ORDER)):
                for j in range(len(TF_ORDER)):
                    if n >= 5:
                        row[TF_ORDER[j]] = _pearson(series[i][-n:], series[j][-n:])
            if n >= 5:
                anchor[tf] = {k: abs(v) for k, v in row.items()}
        if anchor:
            self.bgn_anchor = anchor
            engine.bgn.force_anchor(anchor)
        # --- Step 6: Baum-Welch retrain of each MSRC HMM ----------------
        for symbol in SYMBOL_ORDER:
            for tf in TF_ORDER:
                ring = self.f_returns.get((symbol, tf))
                if ring and len(ring) >= 12:
                    self.hmm_for(symbol, tf).fit(list(ring))
        # --- Step 7: Weibull snapshot for all five regime types ----------
        self._snapshot_weibull(engine)
        # --- Steps 8-11: subsystem re-shares ---------------------------
        for name in ("emotion", "physics", "news", "hf"):
            row = self.contrib.get(name, [0, 0, 0, 0])
            a = row[0] / row[1] if row[1] else 0.5
            d = row[2] / row[3] if row[3] else 0.5
            contribution = a - d
            step = 0.05 if contribution > 0 else -0.05   # Section 10 ±5%
            cur = self.subsystem_shares.get(name, SUBSYSTEM_SHARES_INIT[name])
            lo, hi = SUBSYSTEM_BEARINGS[name]
            self.subsystem_shares[name] = _clamp(cur * (1.0 + step), lo, hi)
        # Level-1 rule (Section 14): renormalize all six to sum 1
        total = sum(self.subsystem_shares.values())
        if total > 0:
            for k in self.subsystem_shares:
                self.subsystem_shares[k] /= total
        # reset accumulators for the next window
        self.contrib = {k: [0, 0, 0, 0] for k in self.subsystem_shares}
        self.tf_hits = {tf: _EPOCH(maxlen=ACCURACY_ROLLING) for tf in TF_ORDER}
        self.update_adaptive_rates()

    # ==================================================================
    # persistence
    # ==================================================================
    def state_dict(self, engine=None) -> dict:
        core_w = {}
        if engine is not None:
            for (s, tf), w in engine.cci.weights.items():
                core_w[f"{s}|{tf}"] = {str(k): round(v, 6) for k, v in w.items()}
        return {
            "version": 4,
            "saved_at": time.time(),
            "rates": self.rates,
            "high_alert": self.high_alert,
            "probation": self.probation_threshold,
            "subsystem_shares": self.subsystem_shares,
            "macro_tone": self.macro_tone,
            "macro_count": self.macro_count,
            "deep_count": self.deep_count,
            "last_macro_grc": self.last_macro_grc,
            "last_macro_ts": self.last_macro_ts,
            "last_deep_ts": self.last_deep_ts,
            "brain_w": {f"{s}|{tf}": [round(x, 6) for x in w]
                        for (s, tf), w in self.brain_w.items()},
            "tf_hr": self.tf_hr,
            "contrib": self.contrib,
            "bgn_anchor": self.bgn_anchor,
            "weibull": self.weibull,
            "hmm": {f"{s}|{tf}": h.to_dict() for (s, tf), h in self.hmm.items()},
            "core_weights": core_w,
            "brains": (engine.brains.state_dict() if engine is not None else {}),
        }

    def load_state(self, engine=None, emotion=None, physics=None,
                   brains=None) -> None:
        try:
            with open(LEARN_STATE_FILE) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            return
        self.rates = {**self.rates, **d.get("rates", {})}
        self.high_alert = bool(d.get("high_alert", False))
        self.probation_threshold = float(d.get("probation", MACRO_PROBATION))
        shares = d.get("subsystem_shares") or {}
        if set(shares) == set(SUBSYSTEM_SHARES_INIT):
            self.subsystem_shares = {k: float(v) for k, v in shares.items()}
        self.macro_tone = float(d.get("macro_tone", 0.0))
        self.macro_count = int(d.get("macro_count", 0))
        self.deep_count = int(d.get("deep_count", 0))
        self.last_macro_grc = int(d.get("last_macro_grc", -10 ** 9))
        self.last_macro_ts = float(d.get("last_macro_ts", 0.0))
        self.last_deep_ts = float(d.get("last_deep_ts", 0.0))
        for key, w in d.get("brain_w", {}).items():
            s, tf = key.split("|", 1)
            self.brain_w[(s, tf)] = [float(x) for x in w]
        for tf, v in d.get("tf_hr", {}).items():
            if tf in TF_ORDER:
                self.tf_hr[tf] = float(v)
        self.contrib = {k: [int(x) for x in row]
                        for k, row in d.get("contrib", {}).items()}
        self.bgn_anchor = d.get("bgn_anchor", {})
        self.weibull = {k: (float(v[0]), float(v[1]))
                        for k, v in d.get("weibull", {}).items()}
        for key, hd in d.get("hmm", {}).items():
            s, tf = key.split("|", 1)
            self.hmm_for(s, tf).load_dict(hd)
        if engine is not None:
            for key, w in d.get("core_weights", {}).items():
                s, tf = key.split("|", 1)
                if tf not in TF_ORDER:
                    continue
                loaded = {int(k): float(v) for k, v in w.items()}
                if set(loaded) != set(WEIGHT_IDS) or not loaded:
                    continue
                for fid in loaded:
                    loaded[fid] = _clamp(loaded[fid], WEIGHT_MIN, WEIGHT_MAX)
                engine.cci.weights[(s, tf)] = loaded
            if self.bgn_anchor:
                engine.bgn.force_anchor(self.bgn_anchor)
            if brains is None and engine is not None:
                brains = engine.brains
            if brains is not None:
                brains.load_state(d.get("brains", {}))

    def save(self, engine=None) -> None:
        try:
            os.makedirs(os.path.dirname(LEARN_STATE_FILE) or ".", exist_ok=True)
            with open(LEARN_STATE_FILE, "w") as fh:
                json.dump(self.state_dict(engine), fh)
        except OSError:
            pass

    def reset(self, engine) -> None:
        self.__init__()
        engine.cci.weights.clear()
        for symbol in SYMBOL_ORDER:
            for tf in TF_ORDER:
                from .cci import uniform_prior
                engine.cci.weights[(symbol, tf)] = uniform_prior()
        engine.emotions.reset_weights()
        engine.physics.reset_weights()
        try:
            os.remove(LEARN_STATE_FILE)
        except OSError:
            pass
        self.save(engine)


def now_until() -> float:
    return time.time() + DEEP_PERIOD_MINUTES * 60


def _unique_top(items: list[int], n: int) -> list[int]:
    out: list[int] = []
    for x in items:
        if x not in out:
            out.append(x)
        if len(out) >= n:
            break
    return out
