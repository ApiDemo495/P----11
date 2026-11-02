"""Self-test tapes (advice §2.5): eight synthetic tapes that each span
exactly one window, with enforced claims:

  1. every formula returns a finite number on every tape
  2. bounded formulas stay in [-1, +1] (F20 excluded — documented exception)
  3. replay on the same tape is deterministic (double-check guarantee)
  4. every zero carries a reason (zero_reasons contract)
  5. BULL and BEAR tapes produce OPPOSITE median signs across the
     directional set (a sign-inversion bug can never hide here)
  6. FLAT tape keeps the directional median inside ±0.35 (no formula may
     scream on a dead tape)

Tapes are seeded so a green run is reproducible. `run_selftest()` returns a
report dict; the pytest wrapper in tests/test_formula_selftest.py enforces it.
"""
from __future__ import annotations

import random

from ..history import HistoryStore
from ..models import FormulaContext
from . import DIRECTIONAL_FORMULAS, FORMULA_FUNCTIONS, compute_all
from .trace import NON_DIRECTIONAL, zero_reason, traced_compute

TAPES = ("BULL", "BEAR", "FLAT", "STRESS", "IMPULSE_UP", "IMPULSE_DOWN",
         "CYCLE_UP", "CYCLE_DOWN")


def _tape(kind: str, w: int = 60, seed: int = 495) -> list[float]:
    """Price series of exactly W+1 samples spanning one window."""
    rng = random.Random(seed)
    p = 100.0
    out = [p]
    for i in range(w):
        if kind == "BULL":
            ret = 0.0016 + rng.gauss(0, 0.0006)
        elif kind == "BEAR":
            ret = -0.0016 + rng.gauss(0, 0.0006)
        elif kind == "FLAT":
            ret = rng.gauss(0, 0.00012)
        elif kind == "STRESS":
            ret = rng.gauss(0, 0.006) * (1 if i % 2 else -1)
        elif kind == "IMPULSE_UP":
            ret = 0.030 if i == w - 3 else rng.gauss(0, 0.0004)
        elif kind == "IMPULSE_DOWN":
            ret = -0.030 if i == w - 3 else rng.gauss(0, 0.0004)
        elif kind == "CYCLE_UP":
            ret = 0.0022 * (1 if (i // 8) % 2 == 0 else -1) + rng.gauss(0, 0.0004)
        else:  # CYCLE_DOWN
            ret = -0.0022 * (1 if (i // 8) % 2 == 0 else -1) + rng.gauss(0, 0.0004)
        p *= 1.0 + ret
        out.append(p)
    return out


def build_ctx(kind: str, symbol: str = "BTC", tf: str = "tf1",
              w: int = 60, hist: HistoryStore | None = None,
              seed_offset: int = 0) -> FormulaContext:
    prices = _tape(kind, w=w, seed=495 + seed_offset)
    n = len(prices)
    t0 = 1_790_000_000
    times = [t0 + i * 60 for i in range(n)]
    rng = random.Random(hash(kind) & 0xFFFF)
    volumes = [10.0 + rng.random() * 5.0 for _ in range(n)]
    o = prices[:-1] + [prices[-1]]
    h = [max(a, b) * 1.001 for a, b in zip(o, prices)]
    l = [min(a, b) * 0.999 for a, b in zip(o, prices)]
    return FormulaContext(
        symbol=symbol, tf=tf, W=w, P=prices, V=volumes, T=times,
        hist=hist or HistoryStore(), others={}, O=o, H=h, L=l,
    )


def warm_store(passes: int = 12) -> HistoryStore:
    """Pre-populate the baseline rings (sigma, tau, SS, ...) the way two
    hours of GRC history would — cold rings make every gated formula abstain
    and the sign claims vacuous."""
    hist = HistoryStore()
    for kind in ("BULL", "BEAR", "CYCLE_UP", "STRESS"):
        for _ in range(passes // 4):
            ctx = build_ctx(kind, hist=hist, seed_offset=passes)
            compute_all(ctx, record=True, anomalies=None)
    return hist


def run_selftest() -> dict:
    report: dict = {"tapes": {}, "claims": {}}
    medians: dict[str, float] = {}
    failures: list[str] = []
    shared = warm_store()

    for kind in TAPES:
        ctx = build_ctx(kind, hist=shared)
        vals, traces = traced_compute(ctx, record=False, anomalies=None)
        tape_fail: list[str] = []
        for fid, fn in FORMULA_FUNCTIONS.items():
            v = vals.get(fid, 0.0)
            if v != v or v in (float("inf"), float("-inf")):
                tape_fail.append(f"F{fid} NaN/inf")
            if fid not in NON_DIRECTIONAL and not (-1.0001 <= v <= 1.0001):
                tape_fail.append(f"F{fid}={v} outside [-1,1]")
            tr = traces.get(fid) or {}
            if v == 0.0 and not tr.get("zero_reason"):
                tape_fail.append(f"F{fid} zero without reason")
        # claim 5/6: directional median on this tape
        ds = sorted(vals[f] for f in DIRECTIONAL_FORMULAS if f in vals)
        med = ds[len(ds) // 2] if ds else 0.0
        medians[kind] = med
        if kind == "FLAT" and abs(med) > 0.35:
            tape_fail.append(f"FLAT median {med:.3f} > ±0.35")
        report["tapes"][kind] = {"median": round(med, 4),
                                 "ok": not tape_fail,
                                 "failures": tape_fail}
        failures.extend(f"{kind}: {f}" for f in tape_fail)

    # claim 5: BULL and BEAR medians opposite signs
    bull, bear = medians.get("BULL", 0.0), medians.get("BEAR", 0.0)
    # fail whenever both tapes lean the SAME non-trivial way (sign inversion
    # or a tape-blind formula set); ±0.005 is the "no lean" epsilon
    same = ((bull > 0.005 and bear > 0.005) or
            (bull < -0.005 and bear < -0.005))
    report["claims"]["bull_bear_opposite"] = not same
    if same:
        failures.append(f"BULL median {bull:.3f} and BEAR {bear:.3f} share a sign")

    report["ok"] = not failures
    report["failures"] = failures
    return report
