"""Validated policy (walk-forward winner, see REAL_RESULTS.md).

How this backtest-only policy differs from the raw app signal
(the app's formulas are ALL intact — every component still
computes and votes in the live fusion; nothing is removed,
per the user's standing rule: change values, never remove):
1. The engine's TF4 (macro timeframe) direction measured
   ANTI-predictive on 1h bars — the policy trades its INVERSE
   (a sign change of the value, not a removal).
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


# ── Changed-vote ensemble (TP4): ALL components vote, weights changed ──
# Second vote attempt, built to the user's corrected rule: nothing
# is removed. All 37 fusion components vote; the 11 measured
# anti-predictive ones (BRAIN, EMOTION, F8, F10, F13, F15, F16,
# F24, HEDGE, NEWS, TRACER) carry NEGATIVE weights — their votes
# are flipped (changed), not dropped. Near-zero voters keep the
# 0.005 floor weight. Weights fitted on the first 55% of the
# 2025-10..2026-09 year (fit logic: backtest/vote_lab.py).
# Walk-forward both directions: fwd 72.2% (n=18), rev 58.1%
# (n=31), pooled OOS 63.3% win / EV +0.300R / PF 2.23 (n=49).
CHANGED_VOTE_WEIGHTS = {
    "BRAIN": -0.043,
    "C31": 0.005,
    "C32": 0.005,
    "C33": 0.005,
    "C34": 0.005,
    "C35": 0.005,
    "EMOTION": -0.045,
    "F1": 0.005,
    "F10": -0.026,
    "F11": 0.01,
    "F12": 0.047,
    "F13": -0.042,
    "F14": 0.014,
    "F15": -0.073,
    "F16": -0.017,
    "F17": 0.021,
    "F18": 0.005,
    "F19": 0.033,
    "F2": 0.036,
    "F20": 0.01,
    "F22": 0.068,
    "F23": 0.123,
    "F24": -0.179,
    "F25": 0.22,
    "F26": 0.083,
    "F27": 0.005,
    "F3": 0.007,
    "F4": 0.087,
    "F5": 0.139,
    "F6": 0.005,
    "F7": 0.005,
    "F8": -0.089,
    "F9": 0.005,
    "HEDGE": -0.181,
    "NEWS": -0.15,
    "PHYSICS": 0.01,
    "TRACER": -0.137
}
CHANGED_VOTE_TAU = 0.5


class ChangedVotePredictor(AppPredictor):
    """TP4 changed-vote: RSI-extreme override, else the all-voters
    weighted vote at >=50% |weight| agreement. Fusion inputs are
    captured via backtest.tp4 wrappers (engine untouched)."""

    def __init__(self, tf: str = "tf4"):
        super().__init__(tf)
        self._cap = None

    def predict(self, view: SliceView, symbol: str) -> Signal | None:
        from .experiment import _rsi
        from .tp4 import _comp_names, _install_capture
        from .vote_lab import changed_vote
        from app.moves_risk import risk_block
        if self._cap is None:
            self._cap = _install_capture(self)
        pack = self.pack(view, symbol)
        d, rule = tp4_direction(pack, _rsi(view.closes))
        if d == 0:
            comp = _comp_names(self._cap.get(("comp", symbol, "tf4")) or {})
            d, _frac = changed_vote(comp, CHANGED_VOTE_WEIGHTS,
                                    CHANGED_VOTE_TAU)
            rule = "changed-vote"
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
