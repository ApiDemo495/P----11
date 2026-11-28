"""16 independent (asset × timeframe) parameter spaces (§2).

Every one of the 16 cells — {BTC, ETH, BNB, SOL} × {tf1 24h, tf2 44m,
tf3 130m, tf4 247m} — is a distinct instrument-horizon pair with its own
microstructure, fee-to-volatility ratio and autocorrelation. This
registry keeps Θ_{a,τ} fully isolated:

  * formula weight init ω^(F)_{a,τ}      (F1–F35)
  * emotion channel init ω^(E)_{a,τ}     (E1–E22)
  * physics multipliers ω^(P)_{a,τ}      (P1–P22)
  * brain connectome gains (DB-1..DB-4)
  * confidence floor, min edge hurdle (bps), TP/SL ATR geometry,
    trailing-stop activation, max hold duration

Mutating one cell's parameters NEVER touches another cell — enforced by
tests. Defaults encode the asset personalities and horizon emphasis
from the directive; cells start at those defaults and learn in place
(online updates write only their own slot).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field, asdict
from typing import Any

from .config import SYMBOL_ORDER, TF_ORDER, TIMEFRAMES

# horizon emphasis (§2.4): short horizons live on microstructure, long
# horizons on vol-persistence / macro alignment
_HORIZON = {
    # tf1 is the 24-hour 2:1 lane (it was a 22-minute scalp until
    # 2026-11-27). It now reads on the same macro/vol-persistence axis as
    # tf4, because that is what a 24-hour window is.
    "tf1": {"micro_weight": 0.75, "macro_weight": 1.30, "hold_mult": 5.8},
    "tf2": {"micro_weight": 1.15, "macro_weight": 0.85, "hold_mult": 1.1},
    "tf3": {"micro_weight": 0.90, "macro_weight": 1.15, "hold_mult": 1.5},
    "tf4": {"micro_weight": 0.80, "macro_weight": 1.25, "hold_mult": 2.0},
}

# asset personalities (§2.2)
_ASSET = {
    "BTC": {"edge_floor_bps": 4.0, "atr_sl": 1.0, "atr_tp": 1.5,
            "trailing_atr": 1.2, "conf_floor": 0.15,
            "notes": "macro liquidity benchmark — trend continuation"},
    "ETH": {"edge_floor_bps": 6.0, "atr_sl": 1.15, "atr_tp": 1.6,
            "trailing_atr": 1.4, "conf_floor": 0.18,
            "notes": "high-beta follower — wick/stop-hunt filter"},
    "BNB": {"edge_floor_bps": 5.0, "atr_sl": 0.9, "atr_tp": 1.4,
            "trailing_atr": 1.1, "conf_floor": 0.16,
            "notes": "exchange-ecosystem — step breakouts, lower noise"},
    # SOL replaced PAXG on 2026-11-27. PAXG was removed on evidence (worst
    # measured asset: 44.0 % WR, -768 bps gross over 209 windows). These
    # numbers are a PRIOR — no certified walk exists for SOL yet — sized for
    # a higher-beta asset: a wider stop, a wider target, a higher floor.
    "SOL": {"edge_floor_bps": 9.0, "atr_sl": 1.3, "atr_tp": 1.8,
            "trailing_atr": 1.5, "conf_floor": 0.22,
            "notes": "high-beta L1 — wide ranges, trend legs (PRIOR, unmeasured)"},
}


@dataclass
class CellParams:
    """Θ_{a,τ} — one cell, fully isolated."""
    symbol: str
    tf: str
    # weight-space initials (learned in place by the owning subsystem)
    formula_weight_init: dict[int, float] = field(default_factory=dict)
    emotion_weight_init: dict[int, float] = field(default_factory=dict)
    physics_mult: dict[int, float] = field(default_factory=dict)
    brain_gains: dict[str, float] = field(default_factory=dict)
    # trading geometry
    conf_floor: float = 0.15          # min confidence to ever issue
    edge_floor_bps: float = 5.0       # min expected edge after friction
    atr_sl: float = 1.0               # stop = atr_sl × ATR(W)
    atr_tp: float = 1.5               # target = atr_tp × ATR(W)
    trailing_atr: float = 1.2         # trailing activation at ×ATR
    max_hold_mult: float = 1.0        # × timeframe minutes
    micro_weight: float = 1.0         # horizon feature emphasis
    macro_weight: float = 1.0
    notes: str = ""

    @property
    def max_hold_minutes(self) -> int:
        base = int(TIMEFRAMES.get(self.tf, {}).get("minutes", 22) or 22)
        return int(base * self.max_hold_mult)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(symbol: str, tf: str) -> CellParams:
    a = _ASSET[symbol]
    h = _HORIZON[tf]
    # initial weight spaces: flat with the horizon tilt applied — the
    # learning rules own the evolution; the registry owns ISOLATION
    fw = {i: 1.0 for i in range(1, 36)}
    ew = {i: 1.0 for i in range(1, 23)}
    pw = {i: 1.0 for i in range(1, 23)}
    return CellParams(
        symbol=symbol, tf=tf,
        formula_weight_init=fw, emotion_weight_init=ew, physics_mult=pw,
        brain_gains={"db1": 1.0, "db2": 1.0, "db3": 1.0, "db4": 1.0},
        conf_floor=a["conf_floor"], edge_floor_bps=a["edge_floor_bps"],
        atr_sl=a["atr_sl"], atr_tp=a["atr_tp"],
        trailing_atr=a["trailing_atr"],
        max_hold_mult=h["hold_mult"],
        micro_weight=h["micro_weight"], macro_weight=h["macro_weight"],
        notes=a["notes"],
    )


class CellRegistry:
    """16 isolated parameter spaces with per-cell mutation."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cells: dict[tuple[str, str], CellParams] = {
            (s, tf): _build(s, tf)
            for s in SYMBOL_ORDER for tf in TF_ORDER
        }

    def get(self, symbol: str, tf: str) -> CellParams:
        """Read-only view: returns the live cell (mutations must go
        through ``mutate`` so isolation stays auditable)."""
        with self._lock:
            key = (symbol, tf)
            if key not in self._cells:
                # unknown symbol/tf gets its own fresh cell, never a
                # shared default (no implicit BTC/tf1 fallback)
                self._cells[key] = _build(
                    symbol if symbol in _ASSET else "BTC",
                    tf if tf in TIMEFRAMES else "tf1")
                self._cells[key].symbol = symbol
                self._cells[key].tf = tf
            return self._cells[key]

    def mutate(self, symbol: str, tf: str, **fields: Any) -> CellParams:
        with self._lock:
            cell = self.get(symbol, tf)
            for k, v in fields.items():
                if not hasattr(cell, k):
                    raise AttributeError(f"CellParams has no field {k!r}")
                setattr(cell, k, v)
            return cell

    def snapshot(self) -> dict[str, dict]:
        """All 16 cells as plain dicts (status / API payloads)."""
        with self._lock:
            return {f"{s}|{tf}": self._cells[(s, tf)].to_dict()
                    for s in SYMBOL_ORDER for tf in TF_ORDER
                    if (s, tf) in self._cells}

    def reset(self) -> None:
        """Back to the directive defaults (tests, cold restart)."""
        with self._lock:
            self._cells = {
                (s, tf): _build(s, tf)
                for s in SYMBOL_ORDER for tf in TF_ORDER
            }


# one process-wide registry (the engine owns Θ, not the request)
CELLS = CellRegistry()


def cell_params(symbol: str, tf: str) -> CellParams:
    return CELLS.get(symbol, tf)
