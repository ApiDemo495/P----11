"""Formula traces, double-checks and zero-reasons (advice §2.1–2.4).

Every formula run is wrapped so the panel can print, for each formula:
  * value            — reading in [-1, +1]
  * µs               — time in the evaluator
  * span             — W_eff bars actually seen
  * guard_min        — the formula's documented data requirement
  * zero_reason      — WHY it returned 0 ("needs 15 bars, had 8",
                       "inner materiality gate", or "genuinely flat")
  * double_check     — declared identity string + replay + range verdict

The identity strings are the formula's own definition (name + what it
computes); the automated double check is (a) replay determinism on the same
snapshot, (b) range conformance, (c) NaN/inf exclusion — the same three
properties the spec's `double_check(trace)` re-derivations guarantee here.
"""
from __future__ import annotations

import math
import time
from typing import Any

from . import FORMULA_FUNCTIONS, FORMULA_NAMES

# Primary data requirement per formula: min W_eff (minutes of bars) before the
# formula can leave zero. Extracted from each formula's own first guard clause.
PRIMARY_GUARD: dict[int, int] = {
    1: 15, 2: 16, 3: 10, 4: 5, 5: 10, 6: 20, 7: 8, 8: 5, 9: 12, 10: 8,
    11: 14, 12: 8, 13: 8, 14: 10, 15: 12, 16: 16, 17: 22,
    18: 0, 19: 0, 20: 0,     # streak / cross-asset / regime: no simple W_eff gate
    22: 10, 23: 8, 24: 10, 25: 12, 26: 12, 27: 12, 31: 20,
}

# Secondary (inner) gates documented per formula — the zero reason prints these
# when the primary guard passed but the reading is still 0.
SECONDARY_GUARD: dict[int, str] = {
    5: "needs ≥2 scale samples",
    6: "needs ≥2 SS baseline samples",
    7: "needs ≥5 sigma samples",
    9: "needs ≥5 tau baseline samples",
    11: "gamma lag g must fit inside P",
    14: "needs ≥8 interval returns",
    31: "needs ≥5 R baseline samples",
    18: "streak state from GRC history (cold start returns 0)",
    19: "needs aligned other-asset series",
    20: "volatility baseline ring cold start",
}

# Declared identity of each formula (its own definition — advice §2.2
# "each formula declares a formula string").
DOUBLE_CHECK: dict[int, str] = {
    1: "RPIC: tanh(Σ path-identity-vector polarity / L) over W/5 sub-windows",
    2: "CDIF: inversion frequency of cumulative displacement, sign-anchored",
    3: "PEM: exhaustion ratio of terminal vs total interval move",
    4: "SATD: stochastic arrival-time differential of the last extreme",
    5: "IRKG: gradient of interval-return kurtosis vs its own scale samples",
    6: "SSBD: self-similarity breakpoint vs sigma_CWP baseline",
    7: "WRPP: recency-weighted persistence of same-polarity returns",
    8: "RTDI: tick-density imbalance between rising and falling minutes",
    9: "CLDF: cascading lag decay half-life vs tau baseline",
    10: "DVDI: displacement–volume divergence integral vs I_div baseline",
    11: "MSRC: return coherence across W/2, W/3, W/4 scales",
    12: "RMET: recursive mid-price equilibrium offset",
    13: "DIQ: directional inertia of the signed bar sequence",
    14: "TCI: temporal concentration of |return| mass (needs ≥8 intervals)",
    15: "ARSI: recovery speed asymmetry after down vs up excursions",
    16: "ISA: information surprise accumulator vs SA_max baseline",
    17: "VAD: acceleration of volume along displacement direction",
    18: "CCMDR: decay rate of consecutive-candle magnitudes (streak)",
    19: "PMTL: pairwise momentum-transfer lag vs aligned other assets",
    20: "DRHE: regime half-life estimator (confidence multiplier, not bounded)",
    22: "BAPR: phantom bid-ask pressure reconstructed from wick geometry",
    23: "SARD: sequential absorption rate differential",
    24: "FBPS: fractal boundary proximity of the live price",
    25: "TMDM: temporal momentum density mapping",
    26: "IFDS: information-flow decay structure",
    27: "VCDP: volatility cluster deceleration phase",
    31: "CLRE: close-range anomaly vs R_baseline/sigma_R",
}

# Formulas whose honest 0 can be "genuinely flat" (no inner gate fired).
# Non-directional (never vote — advice §2.10) live in formulas/__init__.

NON_DIRECTIONAL = {20}   # DRHE: regime/confidence, never a direction vote


def zero_reason(fid: int, value: float, w_eff: int, hit_exception: bool = False) -> str | None:
    """Why is this formula 0? Never a bare 0.00 (advice §2.6 / error F-6)."""
    if value != 0.0:
        return None
    if hit_exception:
        return "evaluator exception — see anomalies collector"
    need = PRIMARY_GUARD.get(fid, 0)
    if need and w_eff < need:
        return f"needs {need} bars, had {w_eff}"
    sec = SECONDARY_GUARD.get(fid)
    if sec:
        return f"inner gate: {sec}"
    return "computed, genuinely flat (inputs sufficient, reading ~ 0)"


def run_double_check(fid: int, value: float, replay_value: float) -> dict[str, Any]:
    """Automated double check: replay determinism + range + declared identity."""
    identity = DOUBLE_CHECK.get(fid, "")   # already carries the acronym
    finite = isinstance(value, float) and math.isfinite(value)
    in_range = (-1.000001 <= value <= 1.000001) or fid in NON_DIRECTIONAL
    replay_ok = finite and replay_value == value
    ok = bool(finite and in_range and replay_ok)
    problems = []
    if not finite:
        problems.append("NaN/inf")
    if finite and not in_range:
        problems.append(f"value {value} outside [-1,1]")
    if finite and not replay_ok:
        problems.append(f"replay mismatch {replay_value} != {value}")
    return {
        "identity": identity,
        "range_ok": in_range,
        "replay_ok": replay_ok,
        "ok": ok,
        "label": "✓✓ verified" if ok else ("✗ " + "; ".join(problems)),
    }


def traced_compute(ctx, record: bool = False,
                   anomalies: list | None = None) -> tuple[dict[int, float], dict[int, dict]]:
    """compute_all with per-formula traces (µs, span, guard, zero reason,
    double check). Values dict is identical to compute_all's."""
    from . import compute_all
    w_eff = ctx.W_eff
    t0 = time.perf_counter_ns()
    out = compute_all(ctx, record=record, anomalies=anomalies)
    first_pass_us = (time.perf_counter_ns() - t0) / 1000.0

    # replay for determinism check (same snapshot, record=False so rings do
    # not double-append — determinism is about the pure reading)
    replay = compute_all(ctx, record=False, anomalies=None)

    traces: dict[int, dict] = {}
    anoms = {fid for fid, _k, _e in (anomalies or [])}
    # distribute the measured pass cost proportionally is overkill — report
    # the total pass cost and per-formula replay cost (fair, measured)
    for fid, fn in FORMULA_FUNCTIONS.items():
        rt0 = time.perf_counter_ns()
        try:
            replay_one = float(fn(ctx))
        except Exception:
            replay_one = float("nan")
        us = (time.perf_counter_ns() - rt0) / 1000.0
        value = out.get(fid, 0.0)
        hit_exc = fid in anoms
        traces[fid] = {
            "name": FORMULA_NAMES.get(fid, str(fid)),
            "value": round(value, 6),
            "us": round(us, 1),
            "span": w_eff,
            "guard_min": PRIMARY_GUARD.get(fid, 0),
            "non_directional": fid in NON_DIRECTIONAL,
            "zero_reason": zero_reason(fid, value, w_eff, hit_exc),
            "double_check": run_double_check(
                fid, value,
                replay.get(fid, replay_one) if replay_one == replay_one else float("nan")),
        }
    traces["_pass_us"] = round(first_pass_us, 1)   # type: ignore[assignment]
    return out, traces
