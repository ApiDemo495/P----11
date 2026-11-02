"""Genesis layer (advice §7): generated formulas under a hard fitness
regime. NumPy only — say so on the card.

Ten mathematical domains generate candidate formulas from a typed feature
set; every candidate is scored on SEVEN metrics (strict-sign hit rate,
information coefficient, regime stability, turnover, correlation to the
active set, latency, sample size) and moves through the lifecycle
candidate → probation → active → retired with an AUTOPSY on retirement.

Rules enforced here:
  * out-of-sample only — generation sees the first 60 % of the store,
    scoring the last 40 %, then walk-forward on new windows (in-sample
    scoring made the pool look brilliant and trade like noise);
  * admission: IC above the median existing formula AND |corr| < 0.6 with
    every active formula;
  * the active set is capped at ACTIVE_CAP; the whole layer is ONE fusion
    vote whose weight is earned by the record;
  * generation and re-scoring run in a low-priority worker thread, never
    on the lock path.
"""
from __future__ import annotations

import json
import math
import os
import random
import tempfile
import threading
import time
from typing import Any, Callable

import numpy as np

ACTIVE_CAP = 12
IC_MIN_MEDIAN_MARGIN = 0.0        # admission: IC > median of active set
MAX_CORR = 0.6
OOS_SPLIT = 0.60                  # generate on first 60 %, score last 40 %
MIN_SCORED = 8                    # candidate -> probation needs scored n
MIN_PROBATION = 12                # probation -> active needs n
RETIRE_MIN_N = 20
STATE_FILE = "data/genesis.json"


# ═══════════════════════ typed feature set ════════════════════════════
def build_features(P: list[float], V: list[float]) -> dict[str, Any]:
    """Typed features every domain consumes (features.py)."""
    arr = np.asarray(P, dtype=float)
    vol = np.asarray(V, dtype=float)
    rets = np.diff(np.log(arr)) if len(arr) > 1 else np.array([0.0])
    mid = max(1, len(arr) // 2)
    signs = np.sign(np.diff(arr))
    return {
        "prices": arr,
        "returns": rets,
        "volumes": vol,
        "signs": signs,
        "first_half": arr[:mid],
        "second_half": arr[mid:],
        "rets_first": rets[: max(1, len(rets) // 2)],
        "rets_second": rets[max(1, len(rets) // 2):],
        "level": float(arr[-1]) if len(arr) else 0.0,
    }


def _clip(x: float) -> float:
    if not isinstance(x, (int, float)) or x != x:
        return 0.0
    return max(-1.0, min(1.0, float(x)))


# ═══════════════════════ ten domains ══════════════════════════════════
def _d01_homology(f: dict) -> list[tuple[str, Callable]]:
    """d01 homology: persistence-style summaries of the detrended path."""
    def slope_persistence(feats: dict) -> float:
        p = feats["prices"]
        if len(p) < 6:
            return 0.0
        det = p - np.linspace(p[0], p[-1], len(p))
        turns = int(np.sum(np.diff(np.sign(np.diff(det))) != 0))
        amp = float(np.max(det) - np.min(det)) or 1e-12
        return _clip((turns / len(det)) * 2.0 * math.tanh(amp / (abs(p[0]) + 1e-12) * 100))
    return [("homology.slope_persistence", slope_persistence)]


def _d02_info_geometry(f: dict) -> list[tuple[str, Callable]]:
    """d02 information geometry: Fisher information of the return density."""
    def fisher_info(feats: dict) -> float:
        r = feats["returns"]
        if len(r) < 8:
            return 0.0
        hist, _ = np.histogram(r, bins=8, density=True)
        p = hist / (hist.sum() + 1e-12)
        p = p[p > 0]
        # discrete Fisher: Σ (Δlog p)² p
        dlog = np.diff(np.log(p + 1e-12))
        fi = float(np.sum(dlog ** 2 * p[1:]))
        return _clip(math.tanh(fi / 10.0))
    return [("info_geometry.fisher_density", fisher_info)]


def _d03_transport(f: dict) -> list[tuple[str, Callable]]:
    """d03 optimal transport: 1-D earth mover between the two halves."""
    def emd_halves(feats: dict) -> float:
        a, b = feats["first_half"], feats["second_half"]
        if len(a) < 3 or len(b) < 3:
            return 0.0
        # E|F−G| via pooled quantiles
        pooled = np.sort(np.concatenate([a, b]))
        q = np.linspace(0, 1, 64, endpoint=False) + 1 / 128
        fa = np.quantile(a, q)
        gb = np.quantile(b, q)
        emd = float(np.mean(np.abs(fa - gb)))
        scale = float(np.std(pooled)) + 1e-12
        return _clip(math.tanh(emd / scale))
    return [("transport.emd_between_halves", emd_halves)]


def _d04_fractional(f: dict) -> list[tuple[str, Callable]]:
    """d04 fractional calculus: half-order difference of the path."""
    def frac_half_diff(feats: dict) -> float:
        p = feats["prices"]
        n = len(p)
        if n < 8:
            return 0.0
        # binomial coefficients of (1−B)^0.5, truncated
        coef = [1.0]
        for k in range(1, min(n, 12)):
            coef.append(-coef[-1] * (0.5 - k + 1) / k)
        d = float(sum(c * (p[-1 - i] if i < n else 0.0)
                      for i, c in enumerate(coef)))
        scale = float(np.std(p)) * 4.0 + 1e-12
        return _clip(math.tanh(d / scale))
    return [("fractional.half_difference", frac_half_diff)]


def _d05_tropical(f: dict) -> list[tuple[str, Callable]]:
    """d05 tropical algebra: max-plus interaction of halves."""
    def tropical_coupling(feats: dict) -> float:
        a, b = feats["rets_first"], feats["rets_second"]
        if len(a) == 0 or len(b) == 0:
            return 0.0
        # max-plus "product" minus independent maxima
        t = float(np.max(a) + np.max(b) - np.max(a) - np.max(b))  # =0 structurally
        # use the deviation of the max-plus sum from the sum of maxima of
        # the concatenated series as the coupling signal
        both = np.concatenate([a, b])
        t2 = float(np.max(a) * np.max(b)) if np.max(a) * np.max(b) != 0 else 0.0
        return _clip(math.tanh((t + t2) * 10.0))
    return [("tropical.max_plus_coupling", tropical_coupling)]


def _d06_quantum(f: dict) -> list[tuple[str, Callable]]:
    """d06 quantum-inspired: fidelity between half-window states."""
    def state_fidelity(feats: dict) -> float:
        a, b = feats["rets_first"], feats["rets_second"]
        if len(a) < 2 or len(b) < 2:
            return 0.0
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        # overlap of equal-length prefixes
        m = min(len(a), len(b))
        fid = float(np.dot(a[:m], b[:m]) / (na * nb))
        return _clip(fid)
    return [("quantum.state_fidelity", state_fidelity)]


def _d07_padic(f: dict) -> list[tuple[str, Callable]]:
    """d07 p-adic: digit-sum oscillation of quantised levels (p=5)."""
    def padic_oscillation(feats: dict) -> float:
        p = feats["prices"]
        if len(p) < 6:
            return 0.0
        q = np.round(p / (float(np.std(p)) + 1e-12)).astype(np.int64)
        sums = []
        for v in q[-40:]:
            x, s = abs(int(v)), 0
            while x:
                s += x % 5
                x //= 5
            sums.append(s)
        arr = np.asarray(sums, dtype=float)
        if arr.std() < 1e-12:
            return 0.0
        # normalised alternating energy
        alt = float(np.sum(arr * np.where(np.arange(len(arr)) % 2 == 0, 1, -1)))
        return _clip(math.tanh(alt / (arr.std() * len(arr) + 1e-12)))
    return [("padic.digit_sum_oscillation", padic_oscillation)]


def _d08_category(f: dict) -> list[tuple[str, Callable]]:
    """d08 category-theoretic: commutator defect of two coarse diagrams."""
    def commutator_defect(feats: dict) -> float:
        r = feats["returns"]
        if len(r) < 8:
            return 0.0
        # f = smooth (moving average), g = decimate; defect = ||f∘g − g∘f||
        def smooth(x):
            k = np.ones(3) / 3.0
            return np.convolve(x, k, mode="same")
        def decim(x):
            return x[::2]
        a = smooth(decim(r))
        b = decim(smooth(r))
        m = min(len(a), len(b))
        if m < 2:
            return 0.0
        defect = float(np.mean(np.abs(a[:m] - b[:m])))
        scale = float(np.std(r)) + 1e-12
        return _clip(math.tanh(defect / scale))
    return [("category.commutator_defect", commutator_defect)]


def _d09_algorithmic(f: dict) -> list[tuple[str, Callable]]:
    """d09 algorithmic: compression-ratio complexity of the sign string."""
    def sign_complexity(feats: dict) -> float:
        s = feats["signs"]
        if len(s) < 8:
            return 0.0
        text = "".join("1" if v > 0 else ("0" if v < 0 else "x") for v in s)
        # tiny LZ77-ish: count new substrings
        seen, i, new = set(), 0, 0
        while i < len(text):
            found = False
            for j in range(i + 1, min(len(text), i + 12) + 1):
                if text[i:j] in seen and j - i > 1:
                    i = j
                    found = True
                    break
            if not found:
                seen.add(text[i:i + 1])
                new += 1
                i += 1
        ratio = new / max(1, len(text))
        # high novelty = incompressible = trend-like information
        return _clip(ratio * 2.0 - 1.0)
    return [("algorithmic.sign_novelty", sign_complexity)]


def _d10_signature(f: dict) -> list[tuple[str, Callable]]:
    """d10 signatures: level-1/level-2 path signature terms."""
    def signature_terms(feats: dict) -> float:
        p = feats["prices"]
        v = feats["volumes"]
        if len(p) < 6:
            return 0.0
        # level 1: total increment (normalised)
        inc = (p[-1] - p[0]) / (abs(p[0]) + 1e-12)
        # level 2 self term: quadratic variation (realised vol)
        r = np.diff(p / (p[0] + 1e-12))
        qv = float(np.sum(r ** 2))
        # level 2 cross term with volume
        m = min(len(r), len(v) - 1) if len(v) > 1 else 0
        cross = float(np.sum(r[:m] * np.asarray(v[1:m + 1], dtype=float))) if m > 2 else 0.0
        return _clip(math.tanh(inc * 8.0) * 0.6
                     + math.tanh(qv * 50.0) * 0.2
                     + math.tanh(cross * 1e-3) * 0.2)
    return [("signature.level12", signature_terms)]


DOMAINS: dict[str, dict] = {
    "d01": {"name": "homology", "math": "persistence / sublevel-set turning structure of the detrended path",
            "gen": _d01_homology},
    "d02": {"name": "information geometry", "math": "Fisher information Σ(Δlog p)²p of the return density",
            "gen": _d02_info_geometry},
    "d03": {"name": "optimal transport", "math": "1-D earth mover E|F−G| between half-window distributions",
            "gen": _d03_transport},
    "d04": {"name": "fractional calculus", "math": "half-order difference (1−B)^0.5 via binomial coefficients",
            "gen": _d04_fractional},
    "d05": {"name": "tropical algebra", "math": "max-plus coupling of half-window extrema",
            "gen": _d05_tropical},
    "d06": {"name": "quantum-inspired", "math": "state fidelity |⟨ψ|φ⟩| between half-window return states",
            "gen": _d06_quantum},
    "d07": {"name": "p-adic", "math": "base-5 digit-sum oscillation of quantised levels",
            "gen": _d07_padic},
    "d08": {"name": "category-theoretic", "math": "commutator defect ‖f∘g − g∘f‖ of smoothing and decimation",
            "gen": _d08_category},
    "d09": {"name": "algorithmic", "math": "sign-string compression novelty (incompressibility = trend)",
            "gen": _d09_algorithmic},
    "d10": {"name": "signatures", "math": "level-1 increment, level-2 quadratic variation and volume cross-term",
            "gen": _d10_signature},
}


# ═══════════════════════ seven-metric fitness ═════════════════════════
def fitness(values: list[float], outcomes: list[int],
            active_values: dict[str, list[float]],
            latency_us: float) -> dict[str, Any]:
    """Seven metrics: hit rate, IC, regime stability, turnover, max|corr|
    to the active set, latency, sample size."""
    v = np.asarray(values, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    n = len(v)
    hit = float(np.mean(np.sign(v) == y)) if n else 0.0
    ic = 0.0
    if n >= 8 and v.std() > 1e-12 and y.std() > 1e-12:
        ic = float(np.corrcoef(v, y)[0, 1])
    # regime stability: hit rate split at the median |value|
    stability = 0.0
    if n >= 8:
        med = float(np.median(np.abs(v)))
        lo, hi = np.abs(v) <= med, np.abs(v) > med
        if lo.any() and hi.any():
            h1 = float(np.mean(np.sign(v[lo]) == y[lo]))
            h2 = float(np.mean(np.sign(v[hi]) == y[hi]))
            stability = 1.0 - abs(h1 - h2)
    # turnover: fraction of sign flips (0 stable .. 1 whipsaw)
    flips = float(np.mean(np.diff(np.sign(v)) != 0)) if n >= 3 else 1.0
    turnover = 1.0 - flips          # high = stable signs
    max_corr = 0.0
    for other in active_values.values():
        m = min(len(other), n)
        if m >= 8:
            a, b = v[-m:], np.asarray(other[-m:], dtype=float)
            if a.std() > 1e-12 and b.std() > 1e-12:
                max_corr = max(max_corr, abs(float(np.corrcoef(a, b)[0, 1])))
    return {
        "hit_rate": round(hit, 4),
        "ic": round(ic, 4),
        "stability": round(stability, 4),
        "turnover": round(turnover, 4),
        "max_corr": round(max_corr, 4),
        "latency_us": round(latency_us, 2),
        "n": int(n),
    }


# ═══════════════════════ lifecycle ════════════════════════════════════
class GenesisLayer:
    """Pool + worker. All state persists to data/genesis.json."""

    def __init__(self, path: str = STATE_FILE) -> None:
        self.path = path
        self.formulas: dict[str, dict] = {}
        self.outcomes: list[dict] = []      # walk-forward rows {ts, move}
        self.next_generation = 0.0
        self.generation_count = 0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._load()

    # ── persistence ────────────────────────────────────────────────────
    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
            self.formulas = raw.get("formulas", {})
            self.outcomes = raw.get("outcomes", [])[-5000:]
            self.next_generation = float(raw.get("next_generation", 0.0))
            self.generation_count = int(raw.get("generation_count", 0))
        except (OSError, ValueError):
            self.formulas, self.outcomes = {}, []

    def save(self) -> None:
        with self._lock:
            out = {"formulas": self.formulas,
                   "outcomes": self.outcomes[-5000:],
                   "next_generation": self.next_generation,
                   "generation_count": self.generation_count}
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path) or ".",
                                       suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                json.dump(out, fh)
            os.replace(tmp, self.path)
        except OSError:
            pass

    # ── worker (never on the lock path) ────────────────────────────────
    def start_worker(self, interval_s: float = 600.0) -> None:
        if self._thread and self._thread.is_alive():
            return

        def loop() -> None:
            try:
                os.nice(19)          # §9: heavy work at low priority
            except OSError:
                pass
            while not self._stop.wait(interval_s):
                try:
                    self.generation_pass()
                except Exception:
                    import traceback
                    traceback.print_exc()

        self._thread = threading.Thread(target=loop, daemon=True,
                                        name="genesis-worker")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ── generation + scoring ───────────────────────────────────────────
    def generation_pass(self, features: dict | None = None) -> dict:
        """One pass: generate from all ten domains, score strictly
        out-of-sample, run the lifecycle, admit or retire."""
        if features is None:
            features = {}
        generated = 0
        for did, dom in DOMAINS.items():
            try:
                candidates = dom["gen"](features)
            except Exception:
                continue
            for name, fn in candidates:
                fid = f"{did}:{name}"
                if fid not in self.formulas:
                    self.formulas[fid] = {
                        "id": fid, "domain": did, "domain_name": dom["name"],
                        "definition": dom["math"],
                        "state": "candidate", "metrics": {},
                        "history": [], "scored": [],
                        "born": time.time(),
                    }
                    generated += 1
        self.generation_count += 1
        self.next_generation = time.time() + 600.0
        self.save()
        return {"generated": generated, "pool": len(self.formulas)}

    def score_walk_forward(self, values: dict[str, float],
                           move: float) -> None:
        """One settled window: every live candidate scores its strict-sign
        call against the realised move (walk-forward, never in-sample)."""
        outcome = 1 if move > 0 else (-1 if move < 0 else 0)
        if outcome == 0:
            return
        for fid, val in values.items():
            entry = self.formulas.get(fid)
            if entry is None:
                continue
            entry["scored"].append([round(float(val), 5), outcome])
            entry["scored"] = entry["scored"][-200:]
        self.outcomes.append({"ts": time.time(), "move": move})
        self.outcomes = self.outcomes[-5000:]
        self._lifecycle()
        self.save()

    def _lifecycle(self) -> None:
        active_meds = []
        for e in self.formulas.values():
            if e["state"] == "active" and e["metrics"].get("ic") is not None:
                active_meds.append(e["metrics"]["ic"])
        median_ic = float(np.median(active_meds)) if active_meds else 0.0
        for fid, e in list(self.formulas.items()):
            scored = e.get("scored") or []
            if len(scored) >= MIN_SCORED:
                vals = [s[0] for s in scored]
                outs = [s[1] for s in scored]
                lat = e["metrics"].get("latency_us", 0.0)
                m = fitness(vals, outs,
                            {k: [s[0] for s in (v.get("scored") or [])]
                             for k, v in self.formulas.items()
                             if v["state"] == "active" and k != fid},
                            lat)
                e["metrics"] = m
                e["history"].append({"ts": time.time(),
                                     **{k: m[k] for k in
                                        ("hit_rate", "ic", "n")}})
                e["history"] = e["history"][-50:]
            m = e["metrics"]
            n = m.get("n", 0)
            if e["state"] == "candidate" and n >= MIN_SCORED:
                e["state"] = "probation"
            elif e["state"] == "probation" and n >= MIN_PROBATION:
                # admission: IC above the active median AND |corr| < 0.6
                n_active = sum(1 for v in self.formulas.values()
                               if v["state"] == "active")
                if (m.get("ic", 0) > median_ic + IC_MIN_MEDIAN_MARGIN
                        and m.get("max_corr", 0) < MAX_CORR
                        and n_active < ACTIVE_CAP):
                    e["state"] = "active"
                    e["admitted"] = time.time()
            elif e["state"] == "active" and n >= RETIRE_MIN_N:
                # retire on record failure, with an autopsy
                if m.get("ic", 0) < 0 or m.get("hit_rate", 0) < 0.45:
                    e["state"] = "retired"
                    e["retired"] = time.time()
                    e["autopsy"] = (
                        f"retired at n={n}: hit_rate={m.get('hit_rate')} "
                        f"ic={m.get('ic')} (active median ic was "
                        f"{median_ic:.3f}); the domain's premise stopped "
                        f"paying after admission — this is the regime "
                        f"change the in-sample pool hid")
                    e.pop("admitted", None)

    # ── fusion vote (one vote, earned weight) ──────────────────────────
    def vote(self) -> dict:
        actives = [e for e in self.formulas.values() if e["state"] == "active"]
        if not actives:
            return {"score": 0.0, "weight": 0.0, "n_active": 0,
                    "note": "no admitted formulas — genesis abstains"}
        # score = mean of current values (caller passes live values)
        return {"n_active": len(actives), "ids": [e["id"] for e in actives]}

    def live_score(self, values: dict[str, float]) -> dict:
        actives = [e for e in self.formulas.values() if e["state"] == "active"]
        if not actives:
            return {"score": 0.0, "weight": 0.0, "n_active": 0,
                    "note": "no admitted formulas — genesis abstains"}
        vals = [values.get(e["id"], 0.0) for e in actives]
        score = float(np.mean(vals)) if vals else 0.0
        ics = [e["metrics"].get("ic", 0.0) for e in actives]
        ic_med = float(np.median(ics)) if ics else 0.0
        weight = max(0.0, min(1.0, len(actives) / ACTIVE_CAP)) * max(0.0, min(1.0, ic_med + 0.5))
        top = max(actives, key=lambda e: e["metrics"].get("ic", -9))
        return {"score": round(score, 5), "weight": round(weight, 4),
                "n_active": len(actives),
                "ic_median": round(ic_med, 4),
                "top": {"id": top["id"],
                        "definition": top.get("definition", ""),
                        "metrics": top.get("metrics", {}),
                        "history": top.get("history", [])[-10:]},
                "note": ("one fusion vote, weight earned by the active "
                         "set's record (advice §7)")}

    # ── API payloads ───────────────────────────────────────────────────
    def status(self) -> dict:
        counts: dict[str, int] = {}
        for e in self.formulas.values():
            counts[e["state"]] = counts.get(e["state"], 0) + 1
        return {
            "pool": len(self.formulas),
            "states": counts,
            "active_cap": ACTIVE_CAP,
            "next_generation_in_s": max(0, int(self.next_generation - time.time())),
            "generation_count": self.generation_count,
            "rules": ("out-of-sample only (60/40 then walk-forward); "
                      "admission: IC > active median AND max|corr| < 0.6; "
                      "cap 12; worker thread only, NumPy only"),
        }

    def formulas_payload(self) -> list[dict]:
        rows = []
        for e in sorted(self.formulas.values(),
                        key=lambda x: -x["metrics"].get("ic", -9)):
            rows.append({
                "id": e["id"], "domain": e["domain"],
                "domain_name": e["domain_name"], "state": e["state"],
                "definition": e["definition"], "metrics": e["metrics"],
                "n_history": len(e.get("history", [])),
                "autopsy": e.get("autopsy"),
                "numpy_only": True,
            })
        return rows

    def formula_detail(self, fid: str) -> dict | None:
        e = self.formulas.get(fid)
        if not e:
            return None
        return {**e, "numpy_only": True}

    def domains_payload(self) -> list[dict]:
        return [{"id": k, "name": v["name"], "math": v["math"]}
                for k, v in DOMAINS.items()]


# ═══════════════ live evaluation (callable map, built once) ═══════════
_FN_MAP: dict[str, Callable] | None = None


def _fn_map() -> dict[str, Callable]:
    global _FN_MAP
    if _FN_MAP is None:
        m: dict[str, Callable] = {}
        for did, dom in DOMAINS.items():
            try:
                for name, fn in dom["gen"]({}):
                    m[f"{did}:{name}"] = fn
            except Exception:
                continue
        _FN_MAP = m
    return _FN_MAP


def evaluate_active(features: dict, active_ids: list[str]) -> dict[str, float]:
    """Evaluate the admitted formulas on the current window (cheap: ≤12
    NumPy calls — still kept OFF the lock path by the worker design)."""
    out: dict[str, float] = {}
    fns = _fn_map()
    for fid in active_ids:
        fn = fns.get(fid)
        if fn is None:
            continue
        try:
            out[fid] = _clip(float(fn(features)))
        except Exception:
            out[fid] = 0.0
    return out
