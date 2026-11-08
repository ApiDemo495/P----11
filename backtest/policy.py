"""Validated policy (walk-forward winner, see REAL_RESULTS.md).

What the experiments REMOVED/CHANGED vs the raw app signal:
1. The engine's TF4 (macro timeframe) direction is ANTI-predictive
   on 1h bars — the policy trades its INVERSE.
2. Trades only in high-volatility regimes (current ATR% in the top
   tercile of its own trailing 120 bars): direction signals in
   quiet regimes measured as noise.
3. Barriers tightened to k=0.75 of the q80-MAE stop (TP still
   exactly 2x the stop — the user's frozen 2:1 is kept).
Everything else (fusion, emotions, physics, brain, risk maths) is
unchanged. Evidence level: ONE walk-forward split (train 58.6%
n=29 -> test 58.3% n=48, EV +0.25R both halves). This is a
candidate, not a proof — it runs in the backtest/CLI now and must
confirm on forward data before it touches live direction.
"""
from __future__ import annotations

from .predictor import HORIZON, AppPredictor, Signal
from .slice import SliceView

K = 0.75
GATE_LOOKBACK = 120
GATE_Q = 2 / 3


def atr_pct_series(bars, n: int = 14) -> list[float]:
    out = []
    for i in range(len(bars)):
        if i < n:
            continue
        seg = bars[i - n:i + 1]
        trs = [max(b.h - b.l, abs(b.h - a.c), abs(b.l - a.c))
               for a, b in zip(seg[:-1], seg[1:])]
        out.append((sum(trs) / len(trs)) / bars[i].c * 100 if bars[i].c else 0.0)
    return out


def gate_ok(bars) -> bool:
    s = atr_pct_series(bars)[-GATE_LOOKBACK:]
    if len(s) < 30:
        return False
    thr = sorted(s)[int(len(s) * GATE_Q)]
    return s[-1] >= thr


def policy_direction(pack: dict) -> int:
    return -int((pack.get("tf4") or {}).get("direction") or 0)


class PolicyPredictor(AppPredictor):
    def predict(self, view: SliceView, symbol: str) -> Signal | None:
        pack = self.pack(view, symbol)
        d = policy_direction(pack)
        if d == 0 or not gate_ok(view.bars):
            return None
        from app.moves_risk import risk_block
        entry = view.bars[-1].c
        blk = risk_block(entry, d, HORIZON,
                         float(pack["tf1"].get("confidence") or 0.0),
                         view.closes, view.highs, view.lows)
        if blk.get("sl") is None:
            return None
        risk = abs(entry - float(blk["sl"]))
        return Signal(d, entry, entry - d * risk * K,
                      entry + d * 2 * risk * K,
                      float(pack["tf1"].get("confidence") or 0.0),
                      str(pack["tf1"].get("tier") or "") + "+policy")


# ── TP4 policy (247-bar macro timeframe, last-1-year validated) ──────

H4 = 247
K4 = 0.75
R4 = 2.0        # TP:SL ratio — user bounds [1:1, 4:1], preference 2:1.
                # Ratio grid measured (REAL_RESULTS): win% identical
                # across 1:1..4:1 (TP rarely binds at TP4), EV best at
                # 1.5:1 (+0.172R vs +0.153R at 2:1, within noise) —
                # the user's preferred 2:1 is kept.


def tp4_direction(pack: dict, rsi: float) -> tuple[int, str]:
    """RSI-extreme mean reversion overrides; otherwise the engine's
    own TF4 (macro thesis) direction stands."""
    if rsi <= 30:
        return 1, "rsi_revert"
    if rsi >= 70:
        return -1, "rsi_revert"
    return int((pack.get("tf4") or {}).get("direction") or 0), "tf4"


class TP4PolicyPredictor(AppPredictor):
    """One TP4 trade at a time over a 247-bar horizon. Validated on
    the last 1 year of real 1h data (walk-forward test 61.0% win,
    EV +0.238R, n=41 — see REAL_RESULTS.md). Candidate, not live."""

    def predict(self, view: SliceView, symbol: str) -> Signal | None:
        from .experiment import _rsi
        from app.moves_risk import risk_block
        pack = self.pack(view, symbol)
        d, rule = tp4_direction(pack, _rsi(view.closes))
        if d == 0:
            return None
        entry = view.bars[-1].c
        blk = risk_block(entry, d, H4,
                         float(pack["tf4"].get("confidence") or 0.0),
                         view.closes, view.highs, view.lows)
        if blk.get("sl") is None:
            return None
        risk = abs(entry - float(blk["sl"]))
        return Signal(d, entry, entry - d * risk * K4,
                      entry + d * R4 * risk * K4,
                      float(pack["tf4"].get("confidence") or 0.0),
                      f"tp4:{rule}")


# ── Vote ensemble (TP4): manipulated vote, both-directions validated ──
# Weights = train-fitted (first 55% of the 2025-10..2026-09 year)
# per-component barrier EV, clipped to [-0.3, 0.3]. Only components
# with positive train EV vote; dead components (always zero at tf4:
# F1/F6/F7/F9/F18/F27/C31-35) are removed — they are absent here.
# A trade needs >= 60% of the total vote weight agreeing (TAU) —
# the vote-percentage manipulation the user authorized — or an
# RSI extreme override. Cross-validated BOTH directions:
# fwd test 70.0% (n=20), rev test 62.5% (n=24), pooled OOS 65.9%
# win / EV +0.330R / PF 2.63 (n=44). See REAL_RESULTS.md.
VOTE_WEIGHTS = {"F25": 0.22, "F5": 0.139, "F23": 0.123, "F4": 0.087,
                "F26": 0.083, "F22": 0.068, "F12": 0.047, "F2": 0.036,
                "F19": 0.033, "F17": 0.021, "F14": 0.014, "F11": 0.01,
                "F20": 0.01, "F3": 0.007, "PHYSICS": 0.01}
VOTE_TAU = 0.6


def vote_direction(comp: dict, weights=VOTE_WEIGHTS, tau=VOTE_TAU):
    """(direction, agreement_fraction). 0 = no trade (vote too split)."""
    tot = sum(abs(w) for w in weights.values()) or 1.0
    s = sum(w * (1 if comp.get(n, 0.0) > 0 else
                 -1 if comp.get(n, 0.0) < 0 else 0)
            for n, w in weights.items())
    frac = abs(s) / tot
    return ((1 if s > 0 else -1 if s < 0 else 0) if frac >= tau else 0), frac


class VoteEnsemblePredictor(AppPredictor):
    """TP4 vote ensemble: RSI-extreme override, else the weighted
    component vote at >=60% agreement. Captures the fusion inputs
    via backtest.tp4 wrappers (engine code untouched)."""

    def __init__(self, tf: str = "tf4"):
        super().__init__(tf)
        self._cap = None

    def predict(self, view: SliceView, symbol: str) -> Signal | None:
        from .experiment import _rsi
        from .tp4 import _comp_names, _install_capture
        from app.moves_risk import risk_block
        if self._cap is None:
            self._cap = _install_capture(self)
        pack = self.pack(view, symbol)
        d, rule = tp4_direction(pack, _rsi(view.closes))
        if d == 0:
            comp = _comp_names(self._cap.get(("comp", symbol, "tf4")) or {})
            d, _frac = vote_direction(comp)
            rule = "vote"
            if d == 0:
                return None
        entry = view.bars[-1].c
        blk = risk_block(entry, d, H4,
                         float(pack["tf4"].get("confidence") or 0.0),
                         view.closes, view.highs, view.lows)
        if blk.get("sl") is None:
            return None
        risk = abs(entry - float(blk["sl"]))
        return Signal(d, entry, entry - d * risk * K4,
                      entry + d * R4 * risk * K4,
                      float(pack["tf4"].get("confidence") or 0.0),
                      f"tp4:{rule}")
