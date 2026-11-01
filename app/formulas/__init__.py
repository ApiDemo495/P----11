"""Formula registry — signal formulas that feed the CCI sum.

  F1..F19, F22..F27, F31 : direct [-1,1] signals (weighted summands)
  F20 (DRHE)             : confidence multiplier only (not a signal)
  F28 (ARC), F29 (DLP)   : risk layer in app/risk.py (price-level outputs)
  F30 (PCV)              : post-CCI validator (needs the CCI as input)
  F21 (CCI)              : the aggregator itself (app.cci)
"""
from __future__ import annotations

from ..config import HIST_CWP, HIST_GAMMA_MAX, HIST_IDIV, HIST_IRECENT_MAX, HIST_R_ANOMALY, HIST_SA, HIST_SS, HIST_STD_R, HIST_TAU
from ..models import FormulaContext
from .f01_f10 import (
    f01_rpic,
    f02_cdif,
    f03_pem,
    f04_satd,
    f05_irkg,
    f06_ssbd,
    f07_wrpp,
    f08_rtdi,
    f09_cldf,
    f10_dvdi,
)
from .f11_f20 import (
    f11_msrc,
    f12_rmet,
    f13_diq,
    f14_tci,
    f15_arsi,
    f16_isa,
    f17_vad,
    f18_ccmdr,
    f19_pmtl,
    f20_drhe,
)
from .f22_f31 import (
    f22_bapr,
    f23_sard,
    f24_fbps,
    f25_tmdm,
    f26_ifds,
    f27_vcdp,
    f31_clre,
)

FORMULA_NAMES: dict[int, str] = {
    1: "RPIC", 2: "CDIF", 3: "PEM", 4: "SATD", 5: "IRKG",
    6: "SSBD", 7: "WRPP", 8: "RTDI", 9: "CLDF", 10: "DVDI",
    11: "MSRC", 12: "RMET", 13: "DIQ", 14: "TCI", 15: "ARSI",
    16: "ISA", 17: "VAD", 18: "CCMDR", 19: "PMTL", 20: "DRHE",
    21: "CCI", 22: "BAPR", 23: "SARD", 24: "FBPS", 25: "TMDM",
    26: "IFDS", 27: "VCDP", 28: "ARC", 29: "DLP", 30: "PCV", 31: "CLRE",
}

FORMULA_FUNCTIONS = {
    1: f01_rpic, 2: f02_cdif, 3: f03_pem, 4: f04_satd, 5: f05_irkg,
    6: f06_ssbd, 7: f07_wrpp, 8: f08_rtdi, 9: f09_cldf, 10: f10_dvdi,
    11: f11_msrc, 12: f12_rmet, 13: f13_diq, 14: f14_tci, 15: f15_arsi,
    16: f16_isa, 17: f17_vad, 18: f18_ccmdr, 19: f19_pmtl, 20: f20_drhe,
    22: f22_bapr, 23: f23_sard, 24: f24_fbps, 25: f25_tmdm,
    26: f26_ifds, 27: f27_vcdp, 31: f31_clre,
}

# Formulas bounded to [-1, +1] (DRHE is the documented exception)
DIRECTIONAL_FORMULAS: list[int] = [f for f in list(FORMULA_FUNCTIONS) if f != 20]

# intermediate key -> (ring name, maxlen) : pre-normalization scalars that feed
# the rolling baselines referenced by the formulas themselves.
_INTERMEDIATE_RINGS: dict[str, tuple[str, int]] = {
    "gamma": ("gamma", HIST_GAMMA_MAX),        # F1  Gamma_max (500 GRCs)
    "irecent": ("irecent", HIST_IRECENT_MAX),  # F2  max inversions (200 GRCs)
    "ss": ("ss", HIST_SS),                     # F6  SS baseline/sigma (100 GRCs)
    "cwp": ("cwp", HIST_CWP),                  # F7  sigma_CWP (200 GRCs)
    "tau": ("tau", HIST_TAU),                  # F9  tau_half baseline (300 GRCs)
    "idiv": ("idiv", HIST_IDIV),               # F10 I_div_max (300 GRCs)
    "sa": ("sa", HIST_SA),                     # F16 SA_max (300 GRCs)
    "std_r": ("std_r", HIST_STD_R),            # F20 volatility baseline (200 GRCs)
    "ranom": ("ranom", HIST_R_ANOMALY),        # F31 R_baseline/sigma_R (200 GRCs)
}


def compute_all(ctx: FormulaContext, record: bool = False,
                anomalies: list | None = None) -> dict[int, float]:
    """Run all registered signal formulas for one (asset, timeframe).

    `record=True` appends GRC-cadence history entries (used only at GRCs);
    event-driven recomputes pass record=False so baselines keep their cadence.

    `anomalies` (optional collector) receives (fid, kind, err) for every
    swallowed exception / NaN — feeds the Ollama formula auditor instead
    of failing silently.
    """
    out: dict[int, float] = {}
    for fid, fn in FORMULA_FUNCTIONS.items():
        try:
            val = float(fn(ctx))
        except Exception as e:
            if anomalies is not None:
                import traceback as _tb
                anomalies.append((fid, "exception",
                                  _tb.format_exc()[-800:] or repr(e)))
            val = 0.0
        if val != val or val in (float("inf"), float("-inf")):   # NaN/inf
            if anomalies is not None:
                anomalies.append((fid, "nan", None))
            val = 0.0
        out[fid] = val

    if record:
        hist = ctx.hist
        for key, (ring_name, maxlen) in _INTERMEDIATE_RINGS.items():
            val = ctx.intermediates.get(key)
            if val is None and key == "ranom":
                from .f22_f31 import clre_anomaly
                try:
                    val = clre_anomaly(ctx)
                except Exception:
                    val = None
            if val is not None and val == val:
                hist.ring(hist.tf_key(ctx.symbol, ctx.tf, ring_name), maxlen).append(val)
        # RPIC outputs feed Bgn coherence (last 50 GRC cycles per timeframe)
        hist.ring(hist.tf_key(ctx.symbol, ctx.tf, "rpic"), HIST_GAMMA_MAX).append(out[1])
        # Formula 20 regime label (drives durations + regime age)
        regime = ctx.intermediates.get("regime_label")
        if regime is not None:
            from ..history import _float_to_regime
            hist.append_regime(ctx.symbol, ctx.tf, _float_to_regime(regime))
        # Formula 18 auto-learning: commit streak outcome at GRC cadence only
        from .f11_f20 import current_streak
        sign_, n_, lam_ = current_streak(ctx)
        hist.commit_streak(ctx.symbol, sign_, n_, lam_)
        # Formula 27 auto-learning: commit finished volatility clusters
        from .f22_f31 import analyze_vol_clusters
        try:
            info = analyze_vol_clusters(ctx)
            hist.commit_vol_cluster(ctx.symbol, info)
        except Exception:
            pass
    return out
