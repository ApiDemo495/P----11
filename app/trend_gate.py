"""Trend-interaction gate — the measured lever on the issued win rate.

What this is
============
A **causal, out-of-fold checked interaction** between the engine's side and
the tape's own slow trend *at the moment the window opens*. It is an
ABSTENTION rule and nothing else: it can withhold a call and say why, and it
can never flip a side, change a level, change a confidence or change a lock.

Why it exists (the measurement)
==============================
``backtest/results/per_tf_full_engine_real.csv`` is the app's own engine
walked over 72 days of real tape on all four assets × all four timeframes
(893 settled windows, every row carrying the realised net bps). On that
record:

===================================  ==========  =========  =================
slice                                n           win rate   95 % Wilson CI
===================================  ==========  =========  =================
every window (what shipped)          893         52.9 %    49.6 – 56.1 %
engine side AGREES with trend        478         46.0 %    41.6 – 50.5 %
engine side OPPOSES the trend        383         60.3 %    55.3 – 65.1 %
trend alone as the side              861         43.2 %    39.9 – 46.5 %
===================================  ==========  =========  =================

The engine's side carries no standalone edge on this tape (52.9 % is inside
the noise), the slow trend is actively wrong as a side (43.2 %, i.e. fading
it would have beaten following it), and the two *interact*: the same engine
call is worth 60.3 % when it is a pullback against the recent trend and
46.0 % when it is a chase with it — a gap of 14.3 pp on 861 windows, both
intervals excluding the other branch's point estimate. Fading the
crowd's most recent impulse is a documented effect in this repo
(``docs/ALPHA_RESEARCH.md`` §12: ``fade_prev`` 51.5–53.9 % vs mom44 48–49.8 %
at exactly these horizons), and this gate is the same effect expressed at the
22-hour sampling interval the engine actually settles on.

Honesty about the evidence
--------------------------
* The split above was found by looking at the pooled record, so the gate is
  reported **in-sample and selection-aware**. It is not presented as a
  forward-proven edge, and it is NOT wired into the certified hold policy or
  the ship gate.
* What is reported alongside it, and what must travel with any claim:
  a 5-fold temporal split (61.1 / 67.8 / 62.3 / 56.0 / 54.6 % — all five
  folds above the 52.9 % baseline), a 65/35 time split on the gated rows
  (train n=248 62.5 %, holdout n=135 56.3 %, CI 47.9 – 64.4), and per-cell
  counts, because most cells have n < 40 and several intervals cross 50 %.
* Where it does NOT work, stated plainly: PAXG (n=77 gated, 48.1 %) and
  ETH tf1/tf2 (n=19/18, 47.4 % / 50.0 %). The rule is applied uniformly to
  all 16 cells rather than switching itself off per cell, because choosing
  the cells from this same sample would be the very selection the rest of
  this docstring disclaims.
* A SECOND rule was tested and REJECTED, and that is recorded here so it is
  not re-invented: stated confidence is not monotone on this record (the
  0.30–0.60 band won 62.6 % while conf >= 0.60 won 49.5 %), and 199 of the
  315 high-confidence windows are exactly the ones that chase the trend.
  "Withhold only high-confidence chases" looked attractive in-sample
  (54.7 % on 77 % coverage) but its 65/35 time split held up at just 51.3 %
  — i.e. within noise of the 52.9 % baseline — with two of five folds below
  baseline. Confidence is therefore NOT used as a filter here; the single
  measured rule stays a single rule.
* Sample sizes are small (n = 383 gated windows pooled, 17–40 per cell).
  A +7.4 pp pooled difference is ≈ 2.9 sigma: strong enough to act on as an
  abstention, not strong enough to promise a forward win rate.

The rule
--------
At a window boundary, with ``L = TREND_LAG_ROWS`` sampled decisions of the
same cell before this one:

    move = close(previous decision) / close(decision L before it) − 1
    move > 0 → the tape's trend is UP, < 0 → DOWN, 0 → no reading

    side opposes the trend  → call stands   (measured 60.3 %)
    side agrees with trend  → call WITHHELD  (measured 46.0 %)
    no reading (warmup)     → call stands   (never invent a reason to block)

The reference series is the engine's own sampled decision prices, which is
what the record was measured on; on the live tape those are the closed
minute closes the engine locks at, so the rule is causal by construction.

Switch
------
``TREND_GATE=off`` in the environment disables it (the engine is then
byte-identical to the pre-gate behaviour). ``app.engine`` reports every gate
decision in the payload, so a withheld call always carries its reason.
"""
from __future__ import annotations

import os
import threading
from collections import deque
from typing import Any

#: how many sampled decisions back the "trend" is measured (the record's
#: 22-hour sampling interval x 2 ≈ a 44-hour close-to-close move)
TREND_LAG_ROWS = 2

#: measured pooled win rates of the two branches (see the module docstring;
#: regenerate with `python -m backtest.trend_gate_audit`)
MEASURED_WR_OPPOSES = 0.603
MEASURED_WR_AGREES = 0.460
MEASURED_N_OPPOSES = 383
MEASURED_N_AGREES = 478
#: every-window reference the gate is compared against
MEASURED_WR_ALL = 0.529

#: minimum sampled closes before a reading exists at all
MIN_ROWS = TREND_LAG_ROWS + 1

_lock = threading.RLock()
_hist: dict[tuple[str, str], deque] = {}


def enabled() -> bool:
    """Environment switch; a deliberately small, documented surface."""
    return os.environ.get("TREND_GATE", "on").strip().lower() not in {
        "0", "off", "false", "no"}


def reset() -> None:
    """Drop every cell's sampled-close history (tests, asset handover)."""
    with _lock:
        _hist.clear()


def observe(symbol: str, tf: str, close: float) -> None:
    """Record one sampled decision close (called once per settled window).

    Causal by construction: the engine calls this at a window boundary with
    the close it is locking at, and the gate only ever reads entries written
    BEFORE the boundary it is asked about.
    """
    try:
        value = float(close)
    except (TypeError, ValueError):
        return
    if not (value > 0):
        return
    with _lock:
        q = _hist.get((symbol, tf))
        if q is None:
            q = _hist[(symbol, tf)] = deque(maxlen=TREND_LAG_ROWS + 1)
        if q and q[-1] == value:
            # The SAME close twice is not a new observation. The ring only
            # holds TREND_LAG_ROWS+1 samples, so accepting repeats let a
            # caller that re-read one tick (a GRC refreshing an open window)
            # flush the real history out and report "trend 0" — the gate then
            # silently stopped withholding on all 16 cells.
            return
        q.append(value)


def trend(symbol: str, tf: str) -> int:
    """Signed slow trend: +1 up, −1 down, 0 = not enough history yet."""
    with _lock:
        q = _hist.get((symbol, tf))
        vals = list(q) if q else []
    if len(vals) < MIN_ROWS:
        return 0
    move = vals[-1] - vals[-TREND_LAG_ROWS]
    if move > 0:
        return 1
    if move < 0:
        return -1
    return 0


def check(symbol: str, tf: str, side: int) -> dict[str, Any]:
    """Should this call stand? Returns a fully explained verdict.

    ``blocked`` is True only when a genuine trend reading exists AND the
    call chases it. Missing history never blocks — an absention has to be
    earned, not defaulted to.
    """
    want = int(side or 0)
    out: dict[str, Any] = {
        "rule": ("fade the trend: a call that OPPOSES the slow trend stands, "
                 "a call that chases it is withheld"),
        "lag_rows": TREND_LAG_ROWS,
        "trend": 0,
        "blocked": False,
        "enabled": enabled(),
        "measured": {"opposes_wr": MEASURED_WR_OPPOSES,
                     "opposes_n": MEASURED_N_OPPOSES,
                     "agrees_wr": MEASURED_WR_AGREES,
                     "agrees_n": MEASURED_N_AGREES},
        "why": "",
    }
    if not out["enabled"]:
        out["why"] = "trend gate disabled (TREND_GATE=off)"
        return out
    if want == 0:
        out["why"] = "no side to check"
        return out
    tr = trend(symbol, tf)
    out["trend"] = tr
    if tr == 0:
        out["why"] = (f"no trend reading yet ({len(_hist.get((symbol, tf), []))}"
                      f"/{MIN_ROWS} sampled decisions) — call stands")
        return out
    out["agrees"] = bool(tr == want)
    if tr == want:
        out["blocked"] = True
        out["why"] = (
            f"chases the {TREND_LAG_ROWS}-decision trend "
            f"({'up' if tr > 0 else 'down'}) — measured "
            f"{100 * MEASURED_WR_AGREES:.1f} % on {MEASURED_N_AGREES} windows "
            f"vs {100 * MEASURED_WR_OPPOSES:.1f} % on "
            f"{MEASURED_N_OPPOSES} against; call withheld")
    else:
        out["why"] = (
            f"opposes the {TREND_LAG_ROWS}-decision trend "
            f"({'up' if tr > 0 else 'down'}) — measured "
            f"{100 * MEASURED_WR_OPPOSES:.1f} % on {MEASURED_N_OPPOSES} windows")
    return out


def stats() -> dict[str, Any]:
    """Cell → sampled closes held, for the diagnosis panel."""
    with _lock:
        return {f"{s}|{t}": len(v) for (s, t), v in _hist.items()}
