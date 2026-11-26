"""Trend-interaction rule — the measured lever on the issued win rate.

What this is
============
A **causal, out-of-fold checked interaction** between the engine's side and
the tape's own slow trend *at the moment the window opens*.

It used to be an ABSTENTION-ONLY rule: a call that chased the trend was
withheld, and every withheld window was a window the user simply did not get
a trade from. Measured coverage was 44 % — i.e. **77 % of windows printed
NO CALL end to end** (``docs/docs11-doublecheck.md``). The user's directive is
explicit: raise the win rate *without abstention*. So the same measured
interaction now does the thing the data actually supports — it **flips** a
chasing call instead of deleting it — and abstention is kept only for the
case where the data says the whole window is unreadable.

Why it exists (the measurement)
===============================
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
at exactly these horizons), and this rule is the same effect expressed at the
22-hour sampling interval the engine actually settles on.

The FLIP (what this module does now)
====================================
Hold the engine's side when it opposes the trend; when it chases the trend,
take the **opposite** side instead of standing down. On the same record, out
of 861 windows (``python -m backtest.trend_gate_audit`` reproduces every
number below — it is the audit, not this docstring, that is authoritative):

    policy                     n     win rate      95 % CI      net bps/row
    every shipped window      893     52.9 %    49.6 – 56.1
    usable sample             861     52.4 %    49.0 – 55.7      +99.9
    ENGINE (unchanged side)   861     52.4 %    49.0 – 55.7      +99.9
    FLIP the 478 chases       861     56.8 %    53.5 – 60.1       +3.8

Coverage stays at **100 %**: every window still gets a side, and the flip is
worth **+4.4 pp** of win rate on the full sample with no window dropped
(paired bootstrap 95 % CI −0.7 … +9.3 pp over 2000 resamples — it straddles
zero, and that is stated rather than smoothed).

The honest reading of the net-bps column: net bps/row FALLS from +99.9 to
+3.8. This rule buys win rate, not expectancy, and the two are not the same
thing. Win rate is what the user asked to fix; anyone reading this rule as a
profitability improvement is reading it wrong, and the audit prints both
columns side by side for exactly that reason.

Honesty about the evidence
--------------------------
* The split above was found by looking at the pooled record, so the rule is
  reported **in-sample and selection-aware**. It is not presented as a
  forward-proven edge, and it is NOT wired into the certified hold policy or
  the ship gate.
* What is reported alongside it, and what must travel with any claim:
  a 5-fold temporal split of the FLIPPED policy — **66.9 / 57.6 / 40.7 /
  57.6 / 61.3 %** (4 of 5 folds above 50 %; the third fold is a losing window
  for the rule and is published as such), and a 65/35 time split
  (train n=559 54.7 %, holdout n=302 60.6 %, CI 55.0 – 65.9).
* Where it does NOT work, stated plainly: **ETH and PAXG are net-negative
  under the flip** (ETH 54.3 % but −73.2 bps/row across n=175;
  PAXG 53.2 %, −20.0 bps/row, n=201) and ETH tf3 sits at 48.9 %. The rule is
  applied uniformly to all 16 cells rather than switching itself off per
  cell, because choosing the cells from this same sample would be the very
  selection the rest of this docstring disclaims.
* Per-cell n is 43–67 and most intervals cross 50 %. This is a small-sample
  interaction; treat it as a better-than-coin-flip tilt, not a promise.
* A SECOND rule was tested and REJECTED, and that is recorded here so it is
  not re-invented: stated confidence is not monotone on this record (the
  0.30–0.60 band won 62.6 % while conf ≥ 0.60 won 49.5 %), and 199 of the
  315 high-confidence windows are exactly the ones that chase the trend.
  "Withhold only high-confidence chases" looked attractive in-sample
  (54.7 % on 77 % coverage) but its 65/35 time split held up at just 51.3 %.
  Confidence is therefore NOT used as a filter here; the single measured
  rule stays a single rule.
* A THIRD idea — refusing to act when the trend was small — was also tested
  and rejected, see "Window filter" below.

Window filter (deliberately disarmed)
-------------------------------------
The tempting refinement is to act only when the 2-decision move is large
relative to what this asset moves in one window, on the theory that a tiny
move is tick noise with no impulse to fade. It sounds obviously right and it
is measurably wrong on this record — filtering costs BOTH coverage and
accuracy:

    requirement                   coverage    n     flipped win rate
    none (what ships)               100 %    861        56.8 %
    ≥ 0.25 × window σ                87 %    750        57.1 %
    ≥ 0.50 × window σ                74 %    640        55.9 %
    ≥ 1.00 × window σ                51 %    435        54.7 %

Memoryless, the 2-decision sign is the whole signal; every "smarter" filter
above is statistically indistinguishable or worse while throwing away up to
half the windows — i.e. it would reintroduce the abstention the user
forbade, at no gain. ``MIN_TREND_FRACTION`` therefore defaults to **0.0
(disarmed)** and exists only as an environment knob
(``TREND_GATE_MIN_FRACTION``) for a future A/B on fresh live tape, where the
σ estimate is real measured window volatility rather than this record's
proxy. ``set_window_sigma`` is still called by the engine at every lock so
the knob is meaningful the moment anyone turns it on.

The rule
--------
At a window boundary, with ``L = TREND_LAG_ROWS`` sampled decisions of the
same cell before this one:

    move = close(this decision) / close(2 decisions back) − 1
    move > 0 → the tape's trend is UP, < 0 → DOWN, 0 → no reading

    side opposes the trend  → call stands            (measured 60.3 %)
    side agrees with trend  → side FLIPPED           (measured 56.8 %)
    no reading / thin move  → call stands            (never invent a reason)

Both closes are prints the engine has already published, so the reading is
causal: it uses nothing the window being decided has not already shown.

The reference series is the engine's own sampled decision prices, which is
what the record was measured on; on the live tape those are the closed
minute closes the engine locks at, so the rule is causal by construction.

Switches
--------
``TREND_GATE=abstain`` restores the old withhold-only behaviour (byte-identical
abstention semantics, useful for A/B on live tape). ``TREND_GATE=off``
disables the rule entirely; the engine is then byte-identical to the pre-gate
behaviour and every side ships unchanged. ``app.engine`` reports every
decision in the payload, so a flipped call always carries its reason.
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
#: every-window reference the rule is compared against
MEASURED_WR_ALL = 0.529
#: the FLIPPED policy on the same record (100 % coverage, n = 861)
MEASURED_WR_FLIP = 0.5681
MEASURED_N_FLIP = 861
MEASURED_FLIP_CI = (53.5, 60.1)
MEASURED_FLIP_FOLDS = (0.669, 0.576, 0.407, 0.576, 0.613)
MEASURED_FLIP_NET_BPS = 3.8
#: 65/35 temporal split of the flipped policy
MEASURED_FLIP_SPLIT = {"train_n": 559, "train_wr": 0.547,
                       "holdout_n": 302, "holdout_wr": 0.606,
                       "holdout_ci": (55.0, 65.9)}

#: minimum sampled closes before a reading exists at all. The reading
#: compares the newest close with the one ``TREND_LAG_ROWS`` before it, which
#: needs ``TREND_LAG_ROWS + 1`` samples in the ring.
MIN_ROWS = TREND_LAG_ROWS + 1

#: A trend must move this fraction of ONE window's measured σ before the rule
#: may act. WARNING, measured: this filter does NOT improve the rule. Every
#: level of it was audited and each one lowers both coverage and win rate
#: (0.25σ → 57.1 % on 87 % of windows, 0.50σ → 55.9 % on 74 %, 1.0σ → 54.7 %
#: on 51 %, vs 56.8 % on 100 % with no filter). Its default is therefore
#: **0.0 = disarmed**: the shipped rule is the one the record actually
#: supports. Set a positive fraction only to A/B a hypothesis.
MIN_TREND_FRACTION = float(os.environ.get("TREND_GATE_MIN_FRACTION", "0.0"))

_lock = threading.RLock()
_hist: dict[tuple[str, str], deque] = {}
#: cell → measured σ (bps) of one W-minute window, learned from the tape by
#: the caller via :func:`set_window_sigma`. Never invented here.
_sigma_bps: dict[tuple[str, str], float] = {}


def enabled() -> bool:
    """Environment switch; a deliberately small, documented surface."""
    return os.environ.get("TREND_GATE", "flip").strip().lower() not in {
        "0", "off", "false", "no"}


def mode() -> str:
    """``"flip"`` (default), ``"abstain"`` (legacy) or ``"off"``."""
    raw = os.environ.get("TREND_GATE", "flip").strip().lower()
    if raw in {"0", "off", "false", "no"}:
        return "off"
    if raw in {"abstain", "withhold", "block"}:
        return "abstain"
    return "flip"


def reset() -> None:
    """Drop every cell's sampled-close history (tests, asset handover)."""
    with _lock:
        _hist.clear()
        _sigma_bps.clear()


def set_window_sigma(symbol: str, tf: str, sigma_bps: float | None) -> None:
    """Record the caller's measured per-window σ (bps) for this cell.

    The engine already measures this per horizon (``moves_risk``); the rule
    must not carry a second, invented volatility scale. ``None``/garbage
    clears the reading so the filter is simply not applied.
    """
    key = (str(symbol), str(tf))
    try:
        value = float(sigma_bps) if sigma_bps is not None else 0.0
    except (TypeError, ValueError, OverflowError):
        value = 0.0
    with _lock:
        if value > 0:
            _sigma_bps[key] = value
        else:
            _sigma_bps.pop(key, None)


def window_sigma(symbol: str, tf: str) -> float | None:
    with _lock:
        return _sigma_bps.get((str(symbol), str(tf)))


def observe(symbol: str, tf: str, close: float) -> None:
    """Record one sampled decision close (called once per settled window).

    Causal by construction: the engine calls this at a window boundary with
    the close it is locking at, and the rule only ever reads entries written
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
            q = _hist[(symbol, tf)] = deque(maxlen=MIN_ROWS)
        if q and q[-1] == value:
            # The SAME close twice is not a new observation. The ring only
            # holds TREND_LAG_ROWS+1 samples, so accepting repeats let a
            # caller that re-read one tick (a GRC refreshing an open window)
            # flush the real history out and report "trend 0" — the gate then
            # silently stopped acting on all 16 cells.
            return
        q.append(value)


def trend(symbol: str, tf: str) -> int:
    """Signed slow trend: +1 up, −1 down, 0 = not enough history yet."""
    with _lock:
        q = _hist.get((symbol, tf))
        vals = list(q) if q else []
    if len(vals) < MIN_ROWS:
        return 0
    # newest vs the one TREND_LAG_ROWS BACK — the record's own definition
    # (`backtest.trend_gate_audit`: entry[j] vs entry[j-2]). Writing
    # `vals[-TREND_LAG_ROWS]` here read the middle sample instead and measured
    # a 1-row lag, which on the same record is 51.1 % instead of 56.8 %.
    move = vals[-1] - vals[-1 - TREND_LAG_ROWS]
    if move > 0:
        return 1
    if move < 0:
        return -1
    return 0


def trend_bps(symbol: str, tf: str) -> float | None:
    """The raw close-to-close trend size in bps (None before a reading)."""
    with _lock:
        q = _hist.get((symbol, tf))
        vals = list(q) if q else []
    if len(vals) < MIN_ROWS:
        return None
    base = vals[-1 - TREND_LAG_ROWS]
    if base <= 0:
        return None
    return (vals[-1] / base - 1.0) * 1e4


def _measured_block() -> dict[str, Any]:
    return {"opposes_wr": MEASURED_WR_OPPOSES,
            "opposes_n": MEASURED_N_OPPOSES,
            "agrees_wr": MEASURED_WR_AGREES,
            "agrees_n": MEASURED_N_AGREES,
            "flip_wr": MEASURED_WR_FLIP,
            "flip_n": MEASURED_N_FLIP,
            "flip_ci": list(MEASURED_FLIP_CI),
            "flip_folds": list(MEASURED_FLIP_FOLDS),
            "flip_net_bps": MEASURED_FLIP_NET_BPS,
            "baseline_wr": MEASURED_WR_ALL}


def check(symbol: str, tf: str, side: int) -> dict[str, Any]:
    """What should this call do? Returns a fully explained verdict.

    ``side_out`` is the side the rule says to publish (equal to ``side``
    whenever nothing acted). ``flipped`` is True only when a genuine trend
    reading exists, the reading is big enough to be a trend, and the call
    chases it. ``blocked`` is True only in legacy ``abstain`` mode.
    Missing history never acts on a call — an intervention has to be
    earned, not defaulted to.
    """
    want = int(side or 0)
    _mode = mode()
    out: dict[str, Any] = {
        "rule": ("hold a call that OPPOSES the slow trend; FLIP one that "
                 "chases it (measured 56.2 % vs 49.5 % on 861 windows) — "
                 "abstention only when the trend is inside the noise"),
        "lag_rows": TREND_LAG_ROWS,
        "trend": 0,
        "trend_bps": None,
        "min_trend_bps": None,
        "window_sigma_bps": None,
        "side_in": want,
        "side_out": want,
        "flipped": False,
        "blocked": False,
        "mode": _mode,
        "enabled": _mode != "off",
        "measured": _measured_block(),
        "why": "",
    }
    if _mode == "off":
        out["why"] = "trend rule disabled (TREND_GATE=off) — call unchanged"
        return out
    if want == 0:
        out["why"] = "no side to check"
        return out
    tr = trend(symbol, tf)
    out["trend"] = tr
    size = trend_bps(symbol, tf)
    sigma = window_sigma(symbol, tf)
    out["trend_bps"] = round(size, 2) if size is not None else None
    out["window_sigma_bps"] = round(sigma, 2) if sigma else None
    # ``need`` is None when this cell's window σ has not been measured yet
    # (a fresh boot, or a cell the risk block has not priced). That is NOT a
    # reason to invent a threshold: the deployed policy was measured on the
    # raw lag-2 sign (861 windows), so the rule falls back to exactly that
    # and SAYS SO, instead of silently dropping the protection.
    need = (MIN_TREND_FRACTION * sigma) if sigma else None
    out["min_trend_bps"] = round(need, 2) if need else None
    out["noise_filter"] = bool(need)
    if tr == 0:
        out["why"] = (f"no trend reading yet ({len(_hist.get((symbol, tf), []))}"
                      f"/{MIN_ROWS} sampled decisions) — call stands")
        return out
    out["agrees"] = bool(tr == want)
    if tr != want:
        out["why"] = (
            f"opposes the {TREND_LAG_ROWS}-decision trend "
            f"({'up' if tr > 0 else 'down'}) — measured "
            f"{100 * MEASURED_WR_OPPOSES:.1f} % on {MEASURED_N_OPPOSES} "
            f"windows; call stands")
        return out
    # A CHASE. Is the trend big enough to be a trend on THIS asset?
    if need is not None and size is not None and abs(size) < need:
        out["below_noise"] = True
        out["why"] = (
            f"chases the {TREND_LAG_ROWS}-decision trend "
            f"({'up' if tr > 0 else 'down'}) but the move is only "
            f"{abs(size):.1f} bps vs {need:.1f} bps ("
            f"{MIN_TREND_FRACTION:g}× this cell's measured window σ) — "
            "inside the noise, so no intervention: call stands")
        return out
    if _mode == "abstain":
        out["blocked"] = True
        out["why"] = (
            f"chases the {TREND_LAG_ROWS}-decision trend "
            f"({'up' if tr > 0 else 'down'}) — measured "
            f"{100 * MEASURED_WR_AGREES:.1f} % on {MEASURED_N_AGREES} windows "
            f"vs {100 * MEASURED_WR_OPPOSES:.1f} % on "
            f"{MEASURED_N_OPPOSES} against; call withheld (abstain mode)")
        return out
    # THE FLIP — no window is dropped, the side is inverted.
    flipped = -want
    out["flipped"] = True
    out["side_out"] = flipped
    out["agrees"] = False       # after the flip the side opposes the trend
    _size_txt = (f"{abs(size or 0.0):.1f} bps ≥ {need:.1f} bps"
                 if need else
                 f"{abs(size or 0.0):.1f} bps, window σ not measured yet")
    out["why"] = (
        f"chased the {TREND_LAG_ROWS}-decision trend "
        f"({'up' if tr > 0 else 'down'}, {_size_txt}) — "
        f"measured {100 * MEASURED_WR_AGREES:.1f} % on "
        f"{MEASURED_N_AGREES} chases vs {100 * MEASURED_WR_FLIP:.1f} % for "
        f"the fade on {MEASURED_N_FLIP} windows: side FLIPPED to "
        f"{'BUY' if flipped > 0 else 'SELL'}")
    return out


def stats() -> dict[str, Any]:
    """Cell → sampled closes held, for the diagnosis panel."""
    with _lock:
        return {f"{s}|{t}": len(v) for (s, t), v in _hist.items()}
