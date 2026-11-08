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
