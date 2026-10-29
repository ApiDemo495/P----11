"""Bgn (Brain-Grid-Network) Synchronization System.

The master interconnection layer.  Every signal in the application — formula
outputs, brain outputs, emotion aggregates, physics signals, news impacts —
passes through the same 4x4 synchronization matrix before reaching the CCI
aggregation, preventing cross-timeframe contamination while enabling beneficial
cross-timeframe confirmation.
"""
from __future__ import annotations

import math

from .config import BGN_BETA, BGN_MIN_COHERENCE_POINTS, BGN_RPIC_HISTORY, TF_ORDER, TIMEFRAMES
from .history import HistoryStore
from ._helpers_pearson import pearson_or_default


def base_coupling(tau_i: int, tau_j: int, beta: float = BGN_BETA) -> float:
    """s_ij = exp(-|ln(tau_i/tau_j)|^2 / (2 * beta^2)); diagonal is always 1.0."""
    if tau_i == tau_j:
        return 1.0
    lr = math.log(tau_i / tau_j)
    return math.exp(-(lr * lr) / (2.0 * beta * beta))


# Precomputed base couplings — must match the specification:
#   s12=0.846  s13=0.334  s14=0.131  s23=0.664  s24=0.354  s34=0.867
BASE_COUPLING: dict[tuple[str, str], float] = {}
for _i, _ti in enumerate(TF_ORDER):
    for _j, _tj in enumerate(TF_ORDER):
        BASE_COUPLING[(_ti, _tj)] = base_coupling(
            TIMEFRAMES[_ti]["minutes"], TIMEFRAMES[_tj]["minutes"]
        )


class BgnSynchronizer:
    """Computes and applies the synchronization matrix per asset at every GRC."""

    def __init__(self, hist: HistoryStore):
        self.hist = hist
        self.last_matrix: dict[str, dict[str, float]] = {}

    # -- coherence ----------------------------------------------------------
    def _coherence(self, symbol: str, tf_i: str, tf_j: str) -> float:
        """|Pearson correlation| between RPIC outputs of two timeframes over
        the last 50 GRC cycles.  Bootstrap defaults to 1.0 (full coupling)."""
        if tf_i == tf_j:
            return 1.0
        ki = self.hist.tf_key(symbol, tf_i, "rpic")
        kj = self.hist.tf_key(symbol, tf_j, "rpic")
        ri = self.hist.ring(ki, BGN_RPIC_HISTORY).values()[-BGN_RPIC_HISTORY:]
        rj = self.hist.ring(kj, BGN_RPIC_HISTORY).values()[-BGN_RPIC_HISTORY:]
        if len(ri) < BGN_MIN_COHERENCE_POINTS or len(rj) < BGN_MIN_COHERENCE_POINTS:
            return 1.0
        return pearson_or_default(ri, rj)

    def matrix(self, symbol: str) -> dict[str, dict[str, float]]:
        """Final coupling s_ij = base_coupling_ij * coherence_ij (diagonal = 1.0)."""
        m: dict[str, dict[str, float]] = {}
        for tf_i in TF_ORDER:
            row: dict[str, float] = {}
            for tf_j in TF_ORDER:
                if tf_i == tf_j:
                    row[tf_j] = 1.0
                else:
                    row[tf_j] = BASE_COUPLING[(tf_i, tf_j)] * self._coherence(
                        symbol, tf_i, tf_j
                    )
            m[tf_i] = row
        self.last_matrix[symbol] = m
        return m

    # -- application --------------------------------------------------------
    @staticmethod
    def adjust(matrix: dict[str, dict[str, float]], raw: dict[str, float]) -> dict[str, float]:
        """F_adjusted(TF_i) = sum_j s_ij * F_raw(TF_j) / sum_j s_ij.

        `raw` maps tf_key -> raw formula output.  Returns the same shape.
        """
        out: dict[str, float] = {}
        for tf_i in TF_ORDER:
            row = matrix[tf_i]
            num = 0.0
            den = 0.0
            for tf_j in TF_ORDER:
                s = row.get(tf_j, 0.0)
                if tf_j not in raw:
                    continue
                num += s * raw[tf_j]
                den += s
            out[tf_i] = (num / den) if den > 1e-12 else raw.get(tf_i, 0.0)
        return out

    def adjust_vector(self, symbol: str, raw_by_tf: dict[str, float]) -> dict[str, float]:
        """Convenience: build (or reuse this-GRC) matrix and adjust."""
        matrix = self.last_matrix.get(symbol) or self.matrix(symbol)
        return self.adjust(matrix, raw_by_tf)
