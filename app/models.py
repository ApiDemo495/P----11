"""Shared data structures for the prediction application."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FormulaContext:
    """Immutable per-(asset, timeframe) snapshot handed to every formula.

    P/V/T are minute-resolution series.  The last element of P is always the
    current live price so formulas observe the freshest sample between GRCs;
    V's last element is the current partial-minute volume.
    """

    symbol: str
    tf: str
    W: int                            # timeframe duration in minutes (spec)
    P: list[float]
    V: list[float]
    T: list[int]                      # minute epoch (UTC seconds) per sample
    hist: Any                         # app.history.HistoryStore
    others: dict[str, tuple[list[int], list[float]]] = field(default_factory=dict)
    # others: aligned minute-epoch -> ffill price series of the other assets (F19)
    intermediates: dict[str, float] = field(default_factory=dict)
    # intermediates: pre-normalization scalars (Gamma, SS, I_div, SA, ...) that
    # the engine records into GRC histories exactly once per cycle.
    O: list[float] = field(default_factory=list)   # minute open  (F22 candles)
    H: list[float] = field(default_factory=list)   # minute high
    L: list[float] = field(default_factory=list)   # minute low

    def candles(self) -> tuple[list[float], list[float], list[float]]:
        """OHLC-derived O/H/L aligned to P; falls back to close-only candles."""
        n = len(self.P)
        if len(self.O) == n and len(self.H) == n and len(self.L) == n:
            return self.O, self.H, self.L
        o = [self.P[0]] + self.P[:-1]
        h = [max(self.P[i], o[i]) for i in range(n)]
        lo = [min(self.P[i], o[i]) for i in range(n)]
        return o, h, lo

    @property
    def W_eff(self) -> int:
        """Effective window in minutes given available data."""
        return max(0, len(self.P) - 1)

    @property
    def ready(self) -> bool:
        return len(self.P) >= self.W + 1


@dataclass
class TargetState:
    """One sub-target of a prediction cycle (Tp1..Tp4)."""
    k: int                 # 1..4
    minutes: int           # horizon in minutes (5,10,15,22 ...)
    level: float | None    # projected price level (None until calculated)
    status: str            # pending | locked | hit | invalidated
    direction: int = 0     # +1 / -1 at lock time (hit detection uses this)
    locked_at: float | None = None   # epoch seconds
    locked_cci: float | None = None  # signed CCI at lock time (lock strength ref)

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "minutes": self.minutes,
            "level": self.level,
            "status": self.status,
            "direction": self.direction,
            "locked_at": self.locked_at,
            "locked_cci": self.locked_cci,
        }


@dataclass
class TFCycle:
    """Full state of one timeframe's current prediction cycle."""
    symbol: str
    tf: str
    W: int
    direction: int = 0               # +1 / -1
    confidence: float = 0.0          # Confidence_final
    cci: float = 0.0                 # signed, double-checked CCI
    drhe: float = 1.0
    issued: bool = False             # False -> NO TRADE overlay
    cycle_start: float = 0.0         # epoch seconds
    cycle_end: float = 0.0           # epoch seconds (countdown target)
    targets: list[TargetState] = field(default_factory=list)
    active_k: int | None = None      # sub-target currently being tracked (1..3)
    tp: float | None = None
    sl: float | None = None
    sl_dist: float | None = None     # stored so the SAFE event can tighten by 15%
    tp_dist: float | None = None
    ratio: float | None = None       # 1.0 / 1.5 / 3.0
    sl_tighten: float = 1.0
    safe_events: int = 0
    pending_reset: bool = False      # CLOSE POSITION -> fresh cycle at next GRC
    close_alert: bool = False        # one-shot UI flag
    safe_alert: bool = False         # one-shot UI flag
    warning: bool = False            # initial-analysis 4-minute expiry warning
    refreshed_at: float = 0.0        # last time prediction values changed
    events: list[dict[str, Any]] = field(default_factory=list)

    def target(self, k: int) -> TargetState:
        return self.targets[k - 1]

    def to_dict(self, now: float | None = None) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "tf": self.tf,
            "W": self.W,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "cci": round(self.cci, 4),
            "drhe": round(self.drhe, 4),
            "issued": self.issued,
            "cycle_start": self.cycle_start,
            "cycle_end": self.cycle_end,
            "countdown_s": max(0, int(self.cycle_end - now)) if now else 0,
            "targets": [t.to_dict() for t in self.targets],
            "active_k": self.active_k,
            "tp": self.tp,
            "sl": self.sl,
            "ratio": self.ratio,
            "safe_events": self.safe_events,
            "pending_reset": self.pending_reset,
            "warning": self.warning,
            "refreshed_at": self.refreshed_at,
        }
