"""price_action.py — read the CHART the user reads.

The app's signals historically ignored price action entirely. User
report (this branch): "it ignores price action ... sometimes make me
loss." Deep research (2026 sources, recorded in the run report):

* Candle patterns ALONE have no measurable edge: a 373,748-occurrence
  study across 105 categories found NO lift surviving multiple-testing
  correction — crypto engulfing hit ~50%, i.e. the baseline
  (patternfetch.com/pattern-base-rates-study).
* The edge appears with CONTEXT: pin bar at support/resistance 61–64%
  vs 48% random; engulfing at S/R with volume 67–71%; S/R flip holds on
  first retest ~72%; volume-confirmed signals +12pp; strict patterns
  with context 55–60% in liquid pairs (50k-trade study).

So this module is CONTEXT-GATED:

* ``structure`` — swing pivots: higher-highs/higher-lows vs the
  opposite (trend structure read direction ~78% of the time in the
  reviewed material);
* ``level`` — proximity (within 1.5 x ATR) of the nearest recent
  swing: at support = +1 context, at resistance = −1;
* ``trigger`` — STRICT definitions only: a pin bar needs a wick of at
  least 2x the body with the close rejecting back; an engulfing needs
  the second body to fully envelop the first and close beyond it.
  A lone candle without structure/level agreement never reaches a
  strong reading (structure alone caps the score at ±0.35, so it can
  never veto a call by itself);
* ``volume`` — the last closed candle at >= 1.5x its 20-bar mean
  confirms the trigger.

All inputs are CLOSED candles — a forming bar is not evidence.
The read is a voter/consensus input and a strong-opposition veto; it
never flips the certified side and never inflates confidence on its
own (confidence rises only through realised accuracy elsewhere).
"""

from __future__ import annotations

import math
from typing import Iterable

# context-gated scoring weights (documented above)
_W_STRUCTURE = 0.35
_W_TRIGGER_AT_LEVEL = 0.45
_W_VOLUME_CONFIRM = 0.15
_W_TRIGGER_ALIGNED = 0.30
_W_TRIGGER_ALIGNED_VOL = 0.10
_W_TRIGGER_SOLO = 0.15
_DIR_FLOOR = 0.20          # |score| below this reads as no opinion
_STRONG = 0.50             # engine veto threshold on opposing scores


def _f(values: Iterable) -> list[float]:
    out: list[float] = []
    for v in values:
        try:
            x = float(v)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(x):
            out.append(x)
    return out


def _atr(highs: list[float], lows: list[float], closes: list[float],
         n: int = 14) -> float:
    """Mean true range over the last ``n`` closed bars."""
    m = min(n, len(closes) - 1)
    if m < 1:
        return 0.0
    total = 0.0
    for i in range(len(closes) - m, len(closes)):
        prev_close = closes[i - 1]
        h, l = highs[i], lows[i]
        total += max(h - l, abs(h - prev_close), abs(l - prev_close))
    return total / m


def _pivots(highs: list[float], lows: list[float],
            left: int = 5, right: int = 5) -> tuple[list[int], list[int]]:
    """Swing indices: local max highs / local min lows (closed bars)."""
    hi_p: list[int] = []
    lo_p: list[int] = []
    n = len(highs)
    for i in range(left, n - right):
        window_h = highs[i - left:i + right + 1]
        window_l = lows[i - left:i + right + 1]
        if highs[i] >= max(window_h) and highs[i] > max(
                highs[i - left:i], highs[i + 1:i + right + 1]):
            hi_p.append(i)
        if lows[i] <= min(window_l) and lows[i] < min(
                lows[i - left:i], lows[i + 1:i + right + 1]):
            lo_p.append(i)
    return hi_p, lo_p


def _structure(hi_p: list[int], lo_p: list[int], highs: list[float],
               lows: list[float]) -> tuple[int, str]:
    """Last two pivots each side: rising highs + rising lows = +1."""
    if len(hi_p) < 2 or len(lo_p) < 2:
        return 0, "not enough swings yet"
    h2, h1 = highs[hi_p[-1]], highs[hi_p[-2]]
    l2, l1 = lows[lo_p[-1]], lows[lo_p[-2]]
    if h2 > h1 and l2 > l1:
        return (1, "HH/HL structure "
                   f"(highs {h1:.6g}→{h2:.6g}, lows {l1:.6g}→{l2:.6g})")
    if h2 < h1 and l2 < l1:
        return (-1, "LH/LL structure "
                    f"(highs {h1:.6g}→{h2:.6g}, lows {l1:.6g}→{l2:.6g})")
    return 0, "mixed swing structure"


def _nearest_level(price: float, hi_p: list[int], lo_p: list[int],
                   highs: list[float], lows: list[float],
                   atr: float) -> tuple[int, float]:
    """Nearest recent swing: +1 if it sits BELOW price (support), −1 above."""
    band = 1.5 * atr if atr > 0 else None
    best: tuple[float, int] | None = None   # (distance, side)
    for i in list(hi_p)[-6:]:
        level = highs[i]
        d = abs(price - level)
        if band is not None and d > band:
            continue
        side = +1 if level < price else -1  # support below / resistance above
        if best is None or d < best[0]:
            best = (d, side)
    for i in list(lo_p)[-6:]:
        level = lows[i]
        d = abs(price - level)
        if band is not None and d > band:
            continue
        side = +1 if level < price else -1
        if best is None or d < best[0]:
            best = (d, side)
    if best is None:
        return 0, float("inf")
    return best[1], best[0]


def _trigger(o: float, h: float, l: float, c: float,
             po: float, ph: float, pl: float, pc: float) -> tuple[int, str]:
    """Strict pin bar or engulfing on the last CLOSED candle."""
    body = abs(c - o)
    prev_body = abs(pc - po)
    rng = h - l
    # strict engulfing: second body fully envelops the first and
    # closes beyond the first body's far edge
    if prev_body > 0 and body >= prev_body:
        if c > o and pc < po and c >= po and o <= pc:
            return (1, "bullish engulfing "
                       "(body envelops prior body, close beyond)")
        if c < o and pc > po and c <= po and o >= pc:
            return (-1, "bearish engulfing "
                        "(body envelops prior body, close beyond)")
    # strict pin: wick at least 2x the body, close rejects back —
    # body must be a real body (a doji is indecision, not a trigger)
    if body > 0 and rng > 0:
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        mid = (h + l) / 2.0
        if lower_wick >= 2.0 * body and c >= mid:
            return (1, f"bullish pin (lower wick {lower_wick / body:.1f}x "
                       "body, close rejected back)")
        if upper_wick >= 2.0 * body and c <= mid:
            return (-1, f"bearish pin (upper wick {upper_wick / body:.1f}x "
                        "body, close rejected back)")
    return 0, "no strict trigger on the last closed candle"


def _trigger_ohlc(opens: list[float], highs: list[float], lows: list[float],
                  closes: list[float]) -> tuple[int, str]:
    """Strict pin/engulfing using actual feed OHLC, never reconstructed opens."""
    return _trigger(opens[-1], highs[-1], lows[-1], closes[-1],
                    opens[-2], highs[-2], lows[-2], closes[-2])

def read(times: list | None, opens: list, closes: list, highs: list, lows: list,
         volumes: list | None = None, price: float | None = None,
         now: float | None = None) -> dict:
    """Context-gated price-action read from CLOSED candles.

    Returns a neutral dict when the tape is too thin to read — an
    honest "no opinion" is never dressed up as conviction.
    """
    o, c, h, l = _f(opens), _f(closes), _f(highs), _f(lows)
    n = min(len(o), len(c), len(h), len(l))
    neutral = {
        "direction": 0, "score": 0.0, "structure": 0, "structure_note": "",
        "level": 0, "level_atr": None, "trigger": 0, "trigger_note": "",
        "volume_ok": False, "atr": 0.0, "reason": "no price action to read",
        "strong": False, "note": ("context-gated: a lone candle never "
                                  "decides; confidence is never inflated here"),
    }
    if n < 60:
        return neutral
    o, c, h, l = o[-n:], c[-n:], h[-n:], l[-n:]
    # a forming (current-minute) bar is not evidence
    if times is not None and now is not None:
        t = _f(times)[-n:]
        if t:
            minute_now = (float(now) // 60) * 60
            cut = sum(1 for x in t if x < minute_now)
            if cut < 60:
                return neutral
            o, c, h, l = o[:cut], c[:cut], h[:cut], l[:cut]
    if len(c) < 60:
        return neutral
    vols = _f(volumes) if volumes is not None else []
    vols = vols[-n:][:len(c)] if len(vols) >= n else []

    atr = _atr(h, l, c)
    price_now = float(price if price and price > 0 else c[-1])
    hi_p, lo_p = _pivots(h, l)
    structure, structure_note = _structure(hi_p, lo_p, h, l)
    level, level_dist = _nearest_level(price_now, hi_p, lo_p, h, l, atr)
    trig, trig_note = _trigger_ohlc(o, h, l, c)
    volume_ok = False
    if len(vols) >= 21:
        base = sum(vols[-21:-1]) / 20.0
        volume_ok = base > 0 and vols[-1] >= 1.5 * base

    score = _W_STRUCTURE * structure
    if trig and level and trig == level:
        # the researched case: trigger AT the level, in the level's
        # direction (pin at support, engulfing out of resistance)
        score += _W_TRIGGER_AT_LEVEL
        if volume_ok:
            score += _W_VOLUME_CONFIRM
    elif trig and structure and trig == structure:
        score += _W_TRIGGER_ALIGNED
        if volume_ok:
            score += _W_TRIGGER_ALIGNED_VOL
    elif trig:
        score += _W_TRIGGER_SOLO
    score = max(-1.0, min(1.0, score))
    direction = (0 if abs(score) < _DIR_FLOOR
                 else (1 if score > 0 else -1))
    bits = [structure_note, trig_note]
    if level:
        bits.append(("at support" if level > 0 else "at resistance")
                    + (f" ({level_dist / atr:.1f}x ATR)" if atr > 0 else ""))
    if volume_ok:
        bits.append("volume ≥1.5× mean")
    return {
        "direction": direction,
        "score": round(score, 4),
        "structure": structure,
        "structure_note": structure_note,
        "level": level,
        "level_atr": (round(level_dist / atr, 2)
                      if atr > 0 and math.isfinite(level_dist) else None),
        "trigger": trig,
        "trigger_note": trig_note,
        "volume_ok": bool(volume_ok),
        "atr": round(atr, 8),
        "reason": " · ".join(b for b in bits if b),
        "strong": bool(direction and abs(score) >= _STRONG),
        "note": ("context-gated price action — a lone candle never "
                 "decides; this block never inflates confidence"),
    }
