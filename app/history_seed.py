"""Retry-safe formulas-only replay of verified Binance 1m history.

No simulator data, fabricated minutes, future context or assumed OHLC
ordering is admitted. Incomplete coverage is reported to callers instead of
being presented as a complete 24-hour backtest.
"""
from __future__ import annotations

import logging
import math
import time

from .cci import uniform_prior
from .config import SYMBOL_ORDER, TF_ORDER, TIMEFRAMES
from .feed import _family, _valid_ohlcv
from .formulas import compute_all
from .history import HistoryStore
from .models import FormulaContext

log = logging.getLogger("history_seed")
MIN_PRIOR_BARS = 60


def _sign(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def _verified_tape(state, now: float) -> dict[str, list]:
    """Sorted, deduplicated, closed Binance OHLCV rows; never fill a gap."""
    limit = min(len(state.times), len(state.opens), len(state.highs),
                len(state.lows), len(state.closes), len(state.volumes),
                len(state.sources))
    rows: dict[int, tuple[float, float, float, float, float]] = {}
    duplicates: set[int] = set()
    current_minute = int(now // 60) * 60
    for i in range(limit):
        if _family(state.sources[i]) != "binance":
            continue
        try:
            raw_ts = float(state.times[i])
            o, high, low, close, volume = (
                float(state.opens[i]), float(state.highs[i]),
                float(state.lows[i]), float(state.closes[i]),
                float(state.volumes[i]))
        except (TypeError, ValueError, OverflowError, IndexError):
            continue
        if (not _valid_ohlcv(raw_ts, o, high, low, close, volume)
                or raw_ts >= current_minute):
            continue
        stamp = int(raw_ts)
        if stamp in rows:
            duplicates.add(stamp)
        else:
            rows[stamp] = (o, high, low, close, volume)
    ordered = sorted((stamp, bar) for stamp, bar in rows.items()
                     if stamp not in duplicates)
    return {
        "times": [stamp for stamp, _ in ordered],
        "opens": [bar[0] for _, bar in ordered],
        "highs": [bar[1] for _, bar in ordered],
        "lows": [bar[2] for _, bar in ordered],
        "closes": [bar[3] for _, bar in ordered],
        "volumes": [bar[4] for _, bar in ordered],
    }


def _coverage(tape: dict[str, list], now: float, hours: int = 24) -> dict:
    times = tape["times"]
    current = int(now // 60) * 60
    cutoff24 = current - 24 * 60 * 60
    requested_cutoff = current - max(0, int(hours)) * 3600
    recent24 = [stamp for stamp in times if cutoff24 <= stamp < current]
    requested = [stamp for stamp in times if requested_cutoff <= stamp < current]
    gaps24 = [b - a for a, b in zip(recent24, recent24[1:]) if b - a != 60]
    gaps_requested = [b - a for a, b in zip(requested, requested[1:])
                      if b - a != 60]
    expected = max(0, int(hours) * 60)
    requested_complete = (
        hours > 0 and len(requested) == expected
        and requested[0] == requested_cutoff
        and requested[-1] == current - 60
        and not gaps_requested)
    return {
        "bars": len(times),
        "from": times[0] if times else None,
        "to": times[-1] if times else None,
        "hours_available": round((times[-1] - times[0]) / 3600, 2)
                           if len(times) > 1 else 0.0,
        "last_24h_bars": len(recent24), "expected_24h_bars": 1440,
        "gaps_24h": len(gaps24),
        "largest_gap_minutes": max(gaps24, default=0) // 60,
        "continuous_last_24h": (
            len(recent24) == 1440 and recent24[0] == cutoff24
            and recent24[-1] == current - 60 and not gaps24),
        "requested_hours": int(hours),
        "requested_bars": len(requested),
        "expected_requested_bars": expected,
        "requested_complete": requested_complete,
        **({} if times else {"why": "no verified Binance-family OHLCV bars"}),
    }


def _contiguous(times: list[int], lo: int, hi: int) -> bool:
    return (lo >= 0 and hi < len(times) and lo <= hi
            and all(times[i] - times[i - 1] == 60
                    for i in range(lo + 1, hi + 1)))


def _others_at(tapes: dict[str, dict[str, list]], symbol: str,
               at_ts: int) -> dict[str, tuple[list[int], list[float]]]:
    """Cross-asset context contains only verified observations known then."""
    others = {}
    for other, tape in tapes.items():
        if other == symbol:
            continue
        end = 0
        while end < len(tape["times"]) and tape["times"][end] <= at_ts:
            end += 1
        others[other] = (tape["times"][:end], tape["closes"][:end])
    return others


def _classify_touch(side: int, entry: float, tp: float | None,
                    sl: float | None, highs: list[float], lows: list[float],
                    start_i: int, end_i: int) -> tuple[str, bool, bool]:
    """Classify chronological OHLC touches; same-bar hits are ambiguous."""
    if side not in (-1, 1) or tp is None or sl is None:
        return "unavailable", False, False
    try:
        entry, tp, sl = float(entry), float(tp), float(sl)
    except (TypeError, ValueError, OverflowError):
        return "unavailable", False, False
    if (not all(math.isfinite(value) for value in (entry, tp, sl))
            or entry <= 0 or tp <= 0 or sl <= 0
            or (side > 0 and not tp > entry > sl)
            or (side < 0 and not tp < entry < sl)
            or start_i < 0 or end_i >= min(len(highs), len(lows))
            or end_i <= start_i):
        return "unavailable", False, False
    for i in range(start_i + 1, end_i + 1):
        target = highs[i] >= tp if side > 0 else lows[i] <= tp
        stop = lows[i] <= sl if side > 0 else highs[i] >= sl
        if target and stop:
            return "ambiguous", True, True
        if target:
            return "target", True, False
        if stop:
            return "stop", True, False
    return "neither", True, False


def _record_learning(engine, symbol: str, tf: str, values: dict,
                     pack: dict, settle_price: float, entry: float,
                     touch: str) -> bool:
    move_sign = _sign(settle_price - entry)
    if not move_sign:
        return False
    signals = {int(fid): _sign(float(value)) for fid, value in values.items()
               if _sign(float(value))}
    try:
        engine.learning.record_micro(
            symbol, tf, move_sign, signals, {}, [],
            1 if touch == "target" else None,
            1 if touch == "stop" else None)
        conf = float(pack.get("confidence") or 0.0)
        cci = float(pack.get("cci_adjusted") or 0.0)
        engine.hist.ring(engine.hist.tf_key(symbol, tf, "calib_p"), 400).append(
            round(min(1.0, max(0.0, conf)) * 20) / 20.0)
        engine.hist.ring(engine.hist.tf_key(symbol, tf, "calib_h"), 400).append(
            1.0 if _sign(cci) == move_sign else 0.0)
        engine.score_hist.setdefault((symbol, tf), []).append(cci)
        return True
    except Exception:
        log.exception("seed learning update failed %s/%s", symbol, tf)
        return False


def seed_history(engine, hours: int = 24, *, now: float | None = None) -> dict:
    """Replay complete, contiguous forecast windows from verified Binance bars.

    Rows are settled only after their entire future horizon is present. Stable
    record IDs make both ledger persistence and learning grading retry-safe.
    """
    mode = getattr(engine.feed, "mode", "unknown")
    if mode != "binance":
        return {"state": "skipped-sim" if mode == "sim" else "unavailable",
                "why": "verified Binance-family history is required; simulator bars are not replayed",
                "settled": 0, "new_settled": 0, "graded": 0,
                "symbols": {}, "coverage": {}}
    try:
        hours = max(0, min(168, int(hours)))
    except (TypeError, ValueError, OverflowError):
        hours = 24
    now = time.time() if now is None else float(now)
    if not math.isfinite(now) or now <= 0:
        return {"state": "unavailable", "why": "invalid replay timestamp",
                "settled": 0, "new_settled": 0, "graded": 0,
                "symbols": {}, "coverage": {}}
    current = int(now // 60) * 60
    cutoff = current - hours * 3600
    tapes = {symbol: _verified_tape(engine.feed.state(symbol), now)
             for symbol in SYMBOL_ORDER}
    coverage = {symbol: _coverage(tape, now, hours)
                for symbol, tape in tapes.items()}
    stats = {
        "state": "seeding", "hours": hours,
        "mode": "formulas-only walk-forward replay of verified Binance-family 1m OHLCV",
        "windows": 0, "settled": 0, "new_settled": 0, "graded": 0,
        "duplicates": 0, "target_first": 0, "stop_first": 0,
        "neither": 0, "unavailable": 0, "ambiguous": 0,
        "skipped_gap_windows": 0, "persistence_failures": 0,
        "learning_persisted": None, "coverage": coverage, "symbols": {},
    }
    if not any(tape["times"] for tape in tapes.values()):
        stats.update(state="unavailable",
                     why="no verified Binance-family closed candles")
        return stats

    local_hist = HistoryStore()
    for symbol in SYMBOL_ORDER:
        tape = tapes[symbol]
        times = tape["times"]
        opens, highs, lows = tape["opens"], tape["highs"], tape["lows"]
        closes, volumes = tape["closes"], tape["volumes"]
        per_tf = {}
        for tf in TF_ORDER:
            width = int(TIMEFRAMES[tf]["minutes"])
            expected = hours * 60 // width if width > 0 else 0
            row = {"expected_windows": expected, "available_windows": 0,
                   "windows": 0, "settled": 0, "new_settled": 0,
                   "graded": 0, "duplicates": 0, "target_first": 0,
                   "stop_first": 0, "neither": 0, "unavailable": 0,
                   "ambiguous": 0, "skipped_gaps": 0}
            if not hours or not width:
                per_tf[tf] = row
                continue
            starts = []
            start_i = len(times) - width - 1
            while start_i >= MIN_PRIOR_BARS and len(starts) < expected:
                if times[start_i] >= cutoff and times[start_i + width] < current:
                    starts.append(start_i)
                start_i -= width
            starts.reverse()
            row["available_windows"] = len(starts)

            for start_i in starts:
                end_i = start_i + width
                context_start = start_i - width
                if (context_start < 0 or
                        not _contiguous(times, context_start, end_i)):
                    row["skipped_gaps"] += 1
                    stats["skipped_gap_windows"] += 1
                    continue
                ctx_slice = slice(context_start, start_i + 1)
                ctx = FormulaContext(
                    symbol=symbol, tf=tf, W=width,
                    P=closes[ctx_slice], V=volumes[ctx_slice],
                    T=times[ctx_slice], hist=local_hist,
                    others=_others_at(tapes, symbol, times[start_i]),
                    O=opens[ctx_slice], H=highs[ctx_slice], L=lows[ctx_slice])
                try:
                    values = compute_all(ctx, record=True)
                    pack = engine.cci.compute(
                        symbol, tf, values,
                        float(values.get(20, 1.0) or 1.0),
                        raw_weights=uniform_prior())
                except Exception:
                    log.exception("seed compute failed %s/%s", symbol, tf)
                    row["skipped_gaps"] += 1
                    stats["skipped_gap_windows"] += 1
                    continue
                side = _sign(float(pack.get("cci_adjusted") or 0.0))
                confidence = float(pack.get("confidence") or 0.0)
                entry = closes[start_i]
                tp = sl = None
                if side in (-1, 1):
                    risk_floor = max(0, start_i - 1000)
                    risk_start = start_i
                    while (risk_start > risk_floor
                           and times[risk_start] - times[risk_start - 1] == 60):
                        risk_start -= 1
                    try:
                        from .moves_risk import risk_block
                        block = risk_block(
                            entry, side, width,
                            min(1.0, abs(float(pack.get("cci_adjusted") or 0.0)) / 0.35),
                            closes[risk_start:start_i + 1],
                            highs[risk_start:start_i + 1],
                            lows[risk_start:start_i + 1], atr=0.0)
                        tp, sl = block.get("tp"), block.get("sl")
                    except Exception:
                        log.exception("seed barrier calculation failed %s/%s", symbol, tf)
                touch, barriers, ambiguous = _classify_touch(
                    side, entry, tp, sl, highs, lows, start_i, end_i)
                settle_price = closes[end_i]
                record_id = f"seed-v1:{symbol}:{tf}:{times[start_i]}"
                result = engine.ledger.settle(
                    symbol, tf, entry, settle_price, side,
                    window_end=times[end_i], seeded=True, conf=confidence,
                    record_id=record_id, first_touch=touch,
                    barriers_available=barriers, ambiguous=ambiguous)
                if not result.get("_persisted", True):
                    stats["persistence_failures"] += 1
                    continue
                stats["windows"] += 1
                stats["settled"] += 1
                row["windows"] += 1
                row["settled"] += 1
                touch_key = {"target": "target_first",
                             "stop": "stop_first"}.get(touch, touch)
                row[touch_key] += 1
                stats[touch_key] += 1
                if result.get("_duplicate"):
                    stats["duplicates"] += 1
                    row["duplicates"] += 1
                    continue
                stats["new_settled"] += 1
                row["new_settled"] += 1
                if result.get("result") in ("win", "loss"):
                    stats["graded"] += 1
                    row["graded"] += 1
                    _record_learning(engine, symbol, tf, values, pack,
                                     settle_price, entry, touch)
            per_tf[tf] = row
        stats["symbols"][symbol] = per_tf

    try:
        stats["learning_persisted"] = bool(engine.learning.save(engine))
    except Exception:
        stats["learning_persisted"] = False
        log.exception("seed learning state save failed")
    requested_complete = all(v.get("requested_complete")
                             for v in coverage.values())
    if stats["persistence_failures"] or stats["learning_persisted"] is False:
        stats["state"] = "persistence-failed"
        if stats["persistence_failures"]:
            stats["why"] = "some settlement rows could not be durably saved"
        else:
            stats["why"] = "learning state could not be durably saved"
    elif stats["settled"] == 0:
        stats["state"] = "insufficient-history"
        stats["why"] = "no complete contiguous context and future horizon windows"
    elif not requested_complete or stats["skipped_gap_windows"]:
        stats["state"] = "partial-history"
        stats["why"] = "verified history is partial or contains gaps; missing minutes were not filled"
    else:
        stats["state"] = "done"
    stats["ts"] = time.time()
    log.info("history seed done: %s", stats)
    return stats
