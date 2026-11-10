"""24-hour warm-history seed (user: 'give it the last 24 hours
backtest history so everything works').

On a fresh start the HISTORY tab, the formula leaderboard, the
confidence calibration and the measured win-rates were all empty
— not broken, just unborn: they only fill from live windows, one
22-minute window at a time. This module replays the last 24 hours
of the REAL 1-minute tape the feed already backfilled:

  for every asset x timeframe, walk the non-overlapping windows
  of the past 24 h; at each window start, lock a call with the
  app's own logic on bars up to that moment (all 37 formulas via
  compute_all + the CCI aggregation with its stored weights —
  formulas-only: brains/physics/news need live state and are not
  faked); at the window end, settle it against the realised close
  (strict sign, tie = loss, TP/SL first-touch from the measured
  risk block over the window's own bars).

Everything it writes is REAL historical grading: the settlement
ledger (rows flagged seeded=True, confidence kept), the formula
accuracy rings (acc500 + per-TF rings via record_micro), the
confidence-calibration rings and the confidence percentile pools.
Nothing is invented; windows without a full W bars of prior tape
are skipped, and the whole seed is skipped on a simulated feed —
sim history dressed as a backtest would be a lie.

Runs once per 24 h (stamp file), in a background thread shortly
after boot, only while the ledger is still thin.
"""
from __future__ import annotations

import logging
import time

from .config import SYMBOL_ORDER, TF_ORDER, TIMEFRAMES
from .formulas import compute_all
from .models import FormulaContext

log = logging.getLogger("history_seed")

MIN_PRIOR_BARS = 60          # a call needs real context, not 3 bars


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def seed_history(engine, hours: int = 24) -> dict:
    """Replay the tape; returns a stats dict for the UI/API."""
    feed = engine.feed
    stats = {"windows": 0, "settled": 0, "graded": 0,
             "symbols": {}, "hours": hours,
             "mode": "formulas-only replay of the real backfilled tape"}
    cutoff_bars = hours * 60
    for symbol in SYMBOL_ORDER:
        st = feed.state(symbol)
        closes = list(st.closes or [])
        times = list(st.times or [])
        opens = list(st.opens or closes)
        highs = list(st.highs or closes)
        lows = list(st.lows or closes)
        vols = list(st.volumes or [])
        n = len(closes)
        if n < MIN_PRIOR_BARS + 30:
            continue
        per_tf = {}
        for tf in TF_ORDER:
            W = TIMEFRAMES[tf]["minutes"]
            max_windows = max(1, cutoff_bars // W)
            # window end indices: step W bars back from the last bar
            ends = []
            e = n - 1
            # a seeded call needs a FULL W bars of prior tape
            while e - W >= max(MIN_PRIOR_BARS, W - 1) \
                    and len(ends) < max_windows:
                ends.append(e)
                e -= W
            ends.reverse()
            prev = None        # (signals, cci, conf, dir, entry, tp, sl, start_i)
            settled = graded = 0

            def _adv_signs(start_i, ctx):
                """Advanced-module signs for the call locked at
                start_i — graded with the same walk-forward rule as
                everything else, so modules (incl. the signature
                voter) accumulate honest history from the seed."""
                try:
                    from .advanced import advanced_block, rmt_block
                    others, series = {}, {}
                    for s2 in SYMBOL_ORDER:
                        cl = list(feed.state(s2).closes or [])
                        series[s2] = cl[:start_i + 1]
                        if s2 != symbol:
                            others[s2] = cl[:start_i + 1]
                    blk = advanced_block(
                        symbol, tf, ctx, others=others,
                        full_closes=closes[:start_i + 1],
                        rmt=rmt_block(series))
                    return blk.get("signs") or {}
                except Exception:
                    return {}

            def _settle_prev(p, settle_i):
                """Settle window p at closes[settle_i] (its own end bar):
                strict sign via the ledger, TP/SL first-touch over the
                window's own bars, formula grading + calibration rings."""
                nonlocal settled, graded
                (p_sig, p_cci, p_conf, p_dir, p_entry, p_tp,
                 p_sl, p_start, p_adv) = p
                settle_px = closes[settle_i]
                lvl = None
                if p_tp and p_sl and p_dir:
                    seg_h = highs[p_start + 1:settle_i + 1]
                    seg_l = lows[p_start + 1:settle_i + 1]
                    for h, l in zip(seg_h, seg_l):
                        if p_dir > 0:
                            if l <= p_sl:
                                lvl = "SL first"; break
                            if h >= p_tp:
                                lvl = "TP first"; break
                        else:
                            if h >= p_sl:
                                lvl = "SL first"; break
                            if l <= p_tp:
                                lvl = "TP first"; break
                engine.ledger.settle(
                    symbol, tf, p_entry, settle_px, p_dir,
                    window_end=(times[settle_i]
                                if settle_i < len(times) else None),
                    level_outcome=lvl, seeded=True, conf=p_conf)
                settled += 1
                move = settle_px - p_entry
                ms = _sign(move)
                if ms:
                    engine.learning.record_micro(
                        symbol, tf, ms,
                        {f: _sign(v) for f, v in p_sig.items()},
                        {}, [], None, None)
                    graded += 1
                    for _mod, _s in (p_adv or {}).items():
                        if _s:
                            engine.hist.ring(engine.hist.tf_key(
                                symbol, tf, f"adv_{_mod}"), 600).append(
                                1.0 if _s == ms else 0.0)
                    if (p_adv or {}).get("signature"):
                        engine.hist.ring(engine.hist.tf_key(
                            symbol, tf, "adv_sig"), 600).append(
                            1.0 if p_adv["signature"] == ms else 0.0)
                    # calibration rings (stated conf vs realised)
                    try:
                        engine.hist.ring(engine.hist.tf_key(
                            symbol, tf, "calib_p"), 400).append(
                            round(min(1.0, max(0.0, p_conf)) * 20)
                            / 20.0)
                        engine.hist.ring(engine.hist.tf_key(
                            symbol, tf, "calib_h"), 400).append(
                            1.0 if _sign(p_cci) == ms else 0.0)
                        engine.score_hist.setdefault(
                            (symbol, tf), []).append(p_cci)
                    except Exception:
                        pass

            for end_i in ends:
                start_i = end_i - W
                # live build_ctx feeds exactly W+1 samples ending at the
                # lock bar — mirror that, or tf1/tf2 (22/44 bars) would
                # never clear a 60-bar floor and silently never seed
                lo = max(0, start_i - W)
                seg_c = closes[lo:start_i + 1]
                if len(seg_c) < min(W + 1, 30):
                    prev = None
                    continue
                try:
                    ctx = FormulaContext(
                        symbol=symbol, tf=tf, W=W,
                        P=seg_c, V=vols[lo:start_i + 1],
                        T=times[lo:start_i + 1], hist=engine.hist,
                        O=opens[lo:start_i + 1], H=highs[lo:start_i + 1],
                        L=lows[lo:start_i + 1])
                    vals = compute_all(ctx, record=False)
                    pack = engine.cci.compute(
                        symbol, tf,
                        {f: vals[f] for f in vals},
                        float(vals.get(20, 1.0) or 1.0))
                except Exception:
                    log.exception("seed compute failed %s/%s", symbol, tf)
                    prev = None
                    continue
                direction = _sign(float(pack.get("cci_adjusted") or 0.0))
                conf = float(pack.get("confidence") or 0.0)
                entry = closes[start_i]
                # measured levels for the first-touch outcome
                tp = sl = None
                try:
                    from .moves_risk import risk_block
                    blk = risk_block(
                        entry, direction or 1, W, min(
                            1.0, abs(float(pack.get("cci_adjusted")
                                           or 0.0)) / 0.35),
                        closes[max(0, start_i - 1000):start_i + 1],
                        highs[max(0, start_i - 1000):start_i + 1],
                        lows[max(0, start_i - 1000):start_i + 1],
                        atr=0.0)
                    tp, sl = blk["tp"], blk["sl"]
                except Exception:
                    pass
                # ── settle the PREVIOUS window at this bar ──
                # (windows are contiguous: prev window = bars
                #  p_start+1 .. start_i, settled at closes[start_i])
                if prev is not None:
                    _settle_prev(prev, start_i)
                stats["windows"] += 1
                prev = ({f: float(v) for f, v in vals.items()},
                        float(pack.get("cci_adjusted") or 0.0), conf,
                        direction, entry, tp, sl, start_i,
                        _adv_signs(start_i, ctx))
            # the last window ends at its own end bar (<= last tape bar)
            if prev is not None and ends:
                _settle_prev(prev, ends[-1])
            per_tf[tf] = {"windows": len(ends), "settled": settled,
                          "graded": graded}
            stats["settled"] += settled
            stats["graded"] += graded
        stats["symbols"][symbol] = per_tf
    stats["ts"] = time.time()
    log.info("history seed done: %s", stats)
    return stats
