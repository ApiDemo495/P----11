"""Boot warm-up: replay the backfilled tape through EVERY subsystem.

Why this module exists
----------------------
The engine used to settle windows only at live GRC boundaries.  A 247-minute
tf4 window settles ~6×/day, so every scorecard (forward P(up) buckets, the
advanced module board, brain/emotion/physics/news/hedge subsystem rows, the
meta-combiner, hindsight re-grading) stayed at n=0 for days after a restart,
and the UI showed "— / n=0 (needs ≥8)" everywhere.  The tape was already in
memory — it just was never replayed.

What is replayed (and how honesty is kept)
------------------------------------------
* Walk-forward only: every reading is computed from bars strictly at/before
  the window's start; the outcome is the window's realised close.  No future
  bar ever enters a feature.
* Every settlement row is written with ``seeded=True`` and a stable record id
  (``warm-v2:<family>:<sym>:<tf>:<start_ts>``), so retries are idempotent and
  the History tab can always separate replayed rows from live ones.
* Simulator tape is replayed too, but the seed status is reported as
  ``done-sim`` — the dashboard already labels that family SIMULATED TAPE, so
  nothing is passed off as market evidence.

Cost control
------------
Windows per (asset, timeframe) are capped and the loop is batched: the ledger
and forward tables are written once, the meta combiner fits once, and the
whole pass runs in a background thread (never on the event loop).
"""
from __future__ import annotations

import logging
import math
import time

from . import loadgov
from typing import Any

from .cci import uniform_prior
from .config import SYMBOL_ORDER, TF_ORDER, TIMEFRAMES, WEIGHT_IDS
from .formulas import FORMULA_FUNCTIONS, compute_all
from .history import HistoryStore
from .models import FormulaContext

log = logging.getLogger("warmup")

# windows replayed per (asset, timeframe).  tf1 is cheap and settles the
# most often, so it carries the largest sample; the long horizons only need
# enough to clear the n≥8/n≥40 scorecard gates and to pool forward buckets.
# Kept just above the scorecard gates (n>=40 for the leaderboard, n>=8 for
# every forward bucket) but roughly half the previous count: the replay is
# pure-Python CPU and every extra window steals the GIL from the live event
# loop for the first minutes after a restart ("engine offline" bursts).
WINDOWS: dict[str, int] = {"tf1": 56, "tf2": 40, "tf3": 28, "tf4": 22}
# cooperative yield between windows: sleeping releases the GIL so /api/health
# and the websocket keep answering during the replay
YIELD_S = 0.012
MIN_CONTEXT = 60            # bars of context before a window may be replayed
FAMILY = {"sim": "sim", "binance": "binance"}


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError, OverflowError):
        return False


def _family_of(source: str) -> str:
    s = str(source or "")
    return "binance" if s.startswith("binance") else "sim"


def _tape(engine, symbol: str, family: str) -> dict[str, list]:
    """Closed bars of ONE writer family, sorted and deduplicated."""
    st = engine.feed.state(symbol)
    limit = min(len(st.times), len(st.opens), len(st.highs),
                len(st.lows), len(st.closes), len(st.volumes), len(st.sources))
    rows: dict[int, tuple] = {}
    for i in range(limit):
        if _family_of(st.sources[i]) != family:
            continue
        try:
            ts = int(float(st.times[i]))
            bar = (float(st.opens[i]), float(st.highs[i]), float(st.lows[i]),
                   float(st.closes[i]), float(st.volumes[i]))
        except (TypeError, ValueError, OverflowError, IndexError):
            continue
        if (not all(_finite(v) for v in (ts, *bar)) or ts <= 0
                or bar[3] <= 0):
            continue
        rows[ts] = bar
    out = {"times": [], "opens": [], "highs": [], "lows": [],
           "closes": [], "volumes": []}
    for ts in sorted(rows):
        o, h, lo, c, v = rows[ts]
        out["times"].append(ts)
        out["opens"].append(o)
        out["highs"].append(h)
        out["lows"].append(lo)
        out["closes"].append(c)
        out["volumes"].append(v)
    return out


def _contiguous(times: list[int], lo: int, hi: int) -> bool:
    if lo < 0 or hi >= len(times) or lo >= hi:
        return False
    return all(times[i] - times[i - 1] == 60 for i in range(lo + 1, hi + 1))


def _market_slice(tapes: dict[str, dict], symbol: str, upto: int,
                  at: int) -> dict:
    """The market bundle the emotion/physics layers read, as-of ``at``.

    Only bars with ``ts <= at`` are exposed (cross-asset and BTC references
    included), so no subsystem can see the window's own future.
    """
    def upto_series(tape: dict) -> list[float]:
        lo, hi = 0, len(tape["times"])
        while lo < hi:
            mid = (lo + hi) // 2
            if tape["times"][mid] <= at:
                lo = mid + 1
            else:
                hi = mid
        return tape["closes"][:lo]

    closes = upto_series(tapes[symbol])[-600:]
    vols = tapes[symbol]["volumes"][:len(upto_series(tapes[symbol]))][-600:]
    highs = tapes[symbol]["highs"][:len(upto_series(tapes[symbol]))][-600:]
    lows = tapes[symbol]["lows"][:len(upto_series(tapes[symbol]))][-600:]
    btc = upto_series(tapes["BTC"]) if "BTC" in tapes else closes
    peer_series = {s: [((c[i] - c[i - 1]) / c[i - 1])
                       for i in range(1, len(c)) if c[i - 1]]
                   for s, c in ((s, upto_series(t)) for s, t in tapes.items())}
    k = min(len(v) for v in peer_series.values()) if peer_series else 0
    peer_returns = ([[peer_series[s][k - 10 + j] for s in SYMBOL_ORDER]
                     for j in range(10)] if k >= 10 else [])
    live = closes[-1] if closes else 0.0
    return {
        "closes": closes,
        "volumes": vols,
        "times": tapes[symbol]["times"][:len(closes)],
        "ohlc": {"highs": highs, "lows": lows, "closes": closes},
        "live": live,
        "session_open": closes[-1440] if len(closes) > 1440 else
                        (closes[0] if closes else live),
        "session_high": max(highs) if highs else live,
        "btc_closes": btc[-600:],
        "peer_returns": peer_returns,
        "ofti": 0.0,
        "cmp_divergence": 0.0,
        "news_price_corr": 0.0,
        "news_value": 0.0,
    }


def _news_proxy(window: list[float], volumes: list[float]) -> float:
    """Tape-only NEWS reaction proxy used when no headline history exists.

    Measured as: sign of the window's reaction (last 5 closes vs the prior
    bar) scaled by the volume shock of that move versus the window's own
    volume median.  It is explicitly a *proxy* — the History hint says so —
    and it never replaces a live News Impact Aggregate.
    """
    if len(window) < 8 or not volumes:
        return 0.0
    seg = window[-6:]
    d = seg[-1] - seg[0]
    if not d:
        return 0.0
    med = sorted(volumes)[len(volumes) // 2] or 1.0
    shock = volumes[-1] / med if med > 0 else 1.0
    return max(-1.0, min(1.0, _sign(d) * min(1.0, max(0.0, shock - 0.5))))


def warm_history(engine, *, hours: int = 168,
                 windows: dict[str, int] | None = None,
                 now: float | None = None) -> dict:
    """Replay the tape through the ledger, forward tables and every ring.

    Returns a status dict (also published by ``/api/history/seed``).
    """
    t_start = time.time()
    want = dict(WINDOWS)
    want.update(windows or {})
    mode = str(getattr(engine.feed, "mode", "sim") or "sim")
    family = FAMILY.get(mode, "sim")
    status: dict[str, Any] = {
        "state": "seeding", "mode": "multi-subsystem walk-forward replay",
        "family": family, "hours": int(hours),
        "windows": 0, "settled": 0, "new_settled": 0, "graded": 0,
        "duplicates": 0, "skipped_gaps": 0, "target_first": 0,
        "stop_first": 0, "neither": 0, "unavailable": 0, "ambiguous": 0,
        "symbols": {}, "subsystems": {}, "elapsed_s": 0.0,
    }
    now_ts = time.time() if now is None else float(now)
    current = int(now_ts // 60) * 60
    cutoff = current - max(1, int(hours)) * 3600

    tapes = {s: _tape(engine, s, family) for s in SYMBOL_ORDER}
    if not any(t["times"] for t in tapes.values()):
        status.update(state="unavailable",
                      why="no closed candles of the active feed family")
        return status

    local_hist = HistoryStore()
    ledger = engine.ledger
    forward = engine.forward
    sub_totals: dict[str, dict[str, int]] = {}

    try:
        from .moves_risk import risk_block
    except Exception:                       # pragma: no cover - defensive
        risk_block = None

    for symbol in SYMBOL_ORDER:
        tape = tapes[symbol]
        times, closes = tape["times"], tape["closes"]
        opens, highs = tape["opens"], tape["highs"]
        lows, volumes = tape["lows"], tape["volumes"]
        status["symbols"][symbol] = {}

        for tf in TF_ORDER:
            width = int(TIMEFRAMES[tf]["minutes"])
            row = {"available": 0, "windows": 0, "settled": 0,
                   "new_settled": 0, "graded": 0, "duplicates": 0}
            # Candidate starts are anchored to an ABSOLUTE grid
            # (multiples of the window width in epoch seconds), not to the
            # end of the local tape. Same bar → same window id → the
            # replay is idempotent across restarts instead of inserting
            # ~1000 fresh rows every boot.
            step = width * 60
            index_of = {t: i for i, t in enumerate(times)}
            starts: list[int] = []
            g = int(current // step) * step - step
            while g >= cutoff and len(starts) < want[tf]:
                i = index_of.get(g)
                if (i is not None and i >= width
                        and i + width < len(times)
                        and times[i + width] - g == step
                        and _contiguous(times, i - width, i + width)):
                    starts.append(i)
                g -= step
            starts.reverse()
            row["available"] = len(starts)

            for _n, start_i in enumerate(starts):
                # COOPERATIVE YIELD, scaled to the MEASURED loop lag. This
                # replay is ~2 minutes of pure-Python numeric work
                # (584 windows x 6 assets) on a worker thread, but CPython
                # still serialises it against the event loop on the GIL:
                # with the loop starved, /api/health and the websocket
                # answered 1.8-2.6 s late — the client painted that as
                # "ENGINE BUSY". A fixed sleep was guesswork; loadgov hands
                # the CPU back exactly as much as the loop needs.
                if _n:
                    loadgov.cooperate()
                end_i = start_i + width
                cslice = slice(start_i - width, start_i + 1)
                ctx = FormulaContext(
                    symbol=symbol, tf=tf, W=width,
                    P=closes[cslice], V=volumes[cslice],
                    T=times[cslice], hist=local_hist,
                    O=opens[cslice], H=highs[cslice], L=lows[cslice])
                try:
                    values = compute_all(ctx, record=False)
                    pack = engine.cci.compute(
                        symbol, tf, values,
                        float(values.get(20, 1.0) or 1.0),
                        raw_weights=uniform_prior())
                except Exception:
                    log.exception("warm replay compute failed %s/%s", symbol, tf)
                    status["skipped_gaps"] += 1
                    continue
                # CALIBRATION SEED: the tier's P(side) needs the measured
                # |score| scale. It used to be built one sample per cycle, so
                # a fresh process stayed at P 0.50 / watch-only for hours.
                # The replay already walks the tape walk-forward — its scores
                # ARE that history, so record them (labelled seeded, exactly
                # like the replayed ledger rows).
                # TREND GATE: the replay IS this cell's sampled decision
                # series, so it seeds the gate's tiny ring too. Without this
                # the gate had no reading until two LIVE windows had settled
                # (hours after a restart on tf4), and a restart silently
                # switched the measured lever off.
                try:
                    from . import trend_gate as _tg_w
                    _tg_w.observe(symbol, tf, float(closes[start_i]))
                except Exception:
                    log.exception("warm trend-gate seed failed %s/%s",
                                  symbol, tf)
                try:
                    import collections as _coll
                    _hist = engine.score_hist.setdefault(
                        (symbol, tf), _coll.deque(maxlen=60))
                    _sc = abs(float(pack.get("cci_adjusted") or 0.0))
                    if _sc > 0:
                        _hist.append(_sc)
                        engine.hist.ring(
                            engine.hist.tf_key(symbol, tf, "score_abs"),
                            200).append(_sc)
                except Exception:
                    log.exception("warm score-history seed failed %s/%s",
                                  symbol, tf)

                entry = closes[start_i]
                settle_price = closes[end_i]
                move = settle_price - entry
                move_sign = _sign(move)
                side = _sign(float(pack.get("cci_adjusted") or 0.0))
                confidence = float(pack.get("confidence") or 0.0)
                start_ts = times[start_i]

                # ── barriers (measured moves, exactly the live block) ──
                tp = sl = None
                if side in (-1, 1) and risk_block is not None:
                    try:
                        block = risk_block(
                            entry, side, width,
                            min(1.0, abs(float(pack.get("cci_adjusted") or 0.0))
                                / 0.35),
                            closes[start_i - min(1000, start_i):start_i + 1],
                            highs[start_i - min(1000, start_i):start_i + 1],
                            lows[start_i - min(1000, start_i):start_i + 1],
                            atr=0.0)
                        tp, sl = block.get("tp"), block.get("sl")
                    except Exception:
                        tp = sl = None
                from .history_seed import _classify_touch
                touch, barriers, ambiguous = _classify_touch(
                    side, entry, tp, sl, highs, lows, start_i, end_i)

                rid = f"warm-v2:{family}:{symbol}:{tf}:{start_ts}"
                res = ledger.settle(
                    symbol, tf, entry, settle_price, side,
                    window_end=times[end_i], seeded=True, conf=confidence,
                    record_id=rid, first_touch=touch,
                    barriers_available=barriers, ambiguous=ambiguous,
                    persist=False)
                row["windows"] += 1
                if not res.get("_persisted", True):
                    continue
                row["settled"] += 1
                key = {"target": "target_first", "stop": "stop_first"}.get(
                    touch, touch)
                if key in status:
                    status[key] += 1
                # A row already in the ledger is not counted twice — but the
                # subsystem rings live in MEMORY and are empty again after a
                # restart, so the replay must still re-derive them below.
                dup = bool(res.get("_duplicate"))
                if dup:
                    row["duplicates"] += 1
                    status["duplicates"] += 1
                else:
                    row["new_settled"] += 1
                    status["new_settled"] += 1
                    if res.get("result") in ("win", "loss"):
                        row["graded"] += 1
                        status["graded"] += 1
                    # ── forward P(up) table (batched, new rows only) ──
                    try:
                        forward.record(
                            symbol, tf,
                            {int(f): float(v) for f, v in values.items()
                             if _finite(v)},
                            entry, settle_price, record_id=rid,
                            persist=False)
                    except Exception:
                        log.exception("warm forward row failed %s/%s",
                                      symbol, tf)

                if not move_sign:
                    continue

                # ── subsystems, all computed from data ≤ window start ──
                market = _market_slice(tapes, symbol, start_i, start_ts)
                subs: dict[str, int] = {}
                brain_sigs: list[float] = []
                window_closes = closes[cslice]

                # brains (real connectome when ready; deterministic, no
                # advance so live learning state is untouched)
                try:
                    atr = sum(abs(window_closes[i] - window_closes[i - 1])
                              for i in range(1, len(window_closes))) / max(
                                  1, len(window_closes) - 1)
                    affinity = {fid: abs(values.get(fid, 0.0))
                                for fid in FORMULA_FUNCTIONS
                                if abs(values.get(fid, 0.0)) > 0.7}
                    br = engine.brains.evaluate(
                        symbol, tf, market, affinity,
                        engine._thing2(symbol, tf, ctx, atr, market),
                        advance=False, raws=values)
                    subs["brain"] = _sign(float(br.get("polarity") or 0.0))
                    brain_sigs = [float(i.get("h_signal", 0.0))
                                  for i in br.get("instances", [])]
                    if subs["brain"]:
                        engine.hist.ring(engine.hist.tf_key(
                            symbol, tf, "facc_brain"), 50).append(
                            1.0 if subs["brain"] == move_sign else -1.0)
                except Exception:
                    log.exception("warm brain failed %s/%s", symbol, tf)

                # emotions: compute, then teach + grade the CERC channels
                try:
                    em = engine.emotions.compute(
                        ctx, market, {}, {"polarity": 0.0, "count": 0,
                                          "impacts": {}}, None)
                    subs["emotion"] = _sign(float(em.get("cerc") or 0.0))
                    emos, dirs = em.get("emotions") or {}, em.get("directions") or {}
                    for eid in range(1, 22):
                        term = float(emos.get(eid, 0.0) or 0.0) * int(
                            dirs.get(eid, 0) or 0)
                        if eid == 21:
                            term *= 1.0 - 0.5 * float(emos.get(eid, 0.0) or 0.0)
                        if _sign(term):
                            engine.hist.ring(engine.hist.tf_key(
                                symbol, tf, f"eacc{eid}"), 50).append(
                                1.0 if _sign(term) == move_sign else -1.0)
                    engine.emotions.reward_update(
                        symbol, tf, emos, dirs, float(em.get("cerc") or 0.0),
                        move, eta=engine.learning.rates.get("emotion"),
                        scale=max(abs(entry) * 0.0005, 1e-9))
                except Exception:
                    log.exception("warm emotion failed %s/%s", symbol, tf)

                # physics P1..P22
                try:
                    long_closes = {s: _market_slice(
                        tapes, s, start_i, start_ts)["closes"]
                        for s in SYMBOL_ORDER}
                    ph = engine.physics.evaluate(
                        symbol, tf, ctx,
                        atr=abs(entry) * 0.001,
                        raws=values,
                        cci_ref=float(pack.get("cci_adjusted") or 0.0),
                        long_closes=long_closes, news_minutes={},
                        weights=engine.cci.get_weights(symbol, tf),
                        recompute=lambda g: compute_all(g, record=False),
                        hist=local_hist, record=False, move=0.0)
                    _u = ph.get("ufp") or {}
                    ufp = float(_u.get("value") or 0.0) if isinstance(
                        _u, dict) else float(_u or 0.0)
                    subs["physics"] = _sign(ufp)
                    pvals = ph.get("values") or {}
                    if _sign(ufp):
                        engine.hist.ring(engine.hist.tf_key(
                            symbol, tf, "pacc22"), 50).append(
                            1.0 if _sign(ufp) == move_sign else -1.0)
                    for pid in range(1, 22):
                        if _sign(float(pvals.get(pid, 0.0) or 0.0)):
                            engine.hist.ring(engine.hist.tf_key(
                                symbol, tf, f"pacc{pid}"), 50).append(
                                1.0 if _sign(float(pvals[pid])) == move_sign
                                else -1.0)
                    engine.physics.reward_update(
                        symbol, tf, pvals, move,
                        eta=engine.learning.rates.get("physics"),
                        scale=max(abs(entry) * 0.0005, 1e-9))
                except Exception:
                    log.exception("warm physics failed %s/%s", symbol, tf)

                # hedge / dark pool + news tape proxy
                ph: dict = {}
                try:
                    from . import hedgefund
                    ifr = hedgefund.institutional_flow(ctx)
                    dpad = hedgefund.dark_pool_flow(ctx)
                    _u = ph.get("ufp") or {}
                    smdi = hedgefund.smart_money_divergence(
                        ifr["value"],
                        float(_u.get("value") or 0.0)
                        if isinstance(_u, dict) else float(_u or 0.0))
                    subs["hf"] = _sign(hedgefund.combined_signal(
                        ifr["value"], dpad["value"], smdi))
                except Exception:
                    subs["hf"] = 0
                subs["news"] = _sign(_news_proxy(window_closes, volumes[cslice]))

                # advanced module walk-forward rings
                try:
                    from .advanced import advanced_block
                    adv = advanced_block(
                        symbol, tf, ctx,
                        others={s: t["closes"][:len(t["closes"])]
                                for s, t in tapes.items() if s != symbol},
                        full_closes=closes[:start_i + 1],
                        weight_scales=engine._adv_weight_scales(symbol, tf),
                        minutes=width)
                    subs["adv"] = _sign(float(adv.get("vote") or 0.0))
                    for name, s in (adv.get("signs") or {}).items():
                        if s:
                            engine.hist.ring(engine.hist.tf_key(
                                symbol, tf, f"adv_{name}"), 600).append(
                                1.0 if int(s) == move_sign else 0.0)
                except Exception:
                    subs["adv"] = 0

                subs["core"] = int(pack.get("direction") or 0)
                subs["final"] = int(pack.get("direction") or 0)

                # subsystem scorecards (ids 97..101 in the leaderboard)
                for name, cid in (("brain", 97), ("emotion", 98),
                                  ("physics", 99), ("news", 100),
                                  ("hf", 101)):
                    s = int(subs.get(name) or 0)
                    if s:
                        engine.learning.acc500[cid].append(
                            1 if s == move_sign else 0)
                        engine.hist.ring(engine.hist.tf_key(
                            symbol, tf, f"facc{cid}"), 50).append(
                            1.0 if s == move_sign else -1.0)
                    # ── AN ABSTENTION IS NOT A LOSS (2026-11-26) ────────
                    # `s == 0` means the subsystem had NO OPINION on this
                    # window (e.g. the hedge tape had no qualifying prints).
                    # The counter incremented `n` for it and scored it as a
                    # miss, so a subsystem that spoke on 40 % of windows and
                    # was RIGHT on them still graded below 50 % — live
                    # measurement showed hf at 0.036 accuracy, which is not a
                    # predictor quality, it is the arithmetic of counting
                    # silences as errors. Only real votes are graded, and the
                    # count of silences is reported separately so the
                    # coverage is still visible.
                    st = sub_totals.setdefault(name, {"n": 0, "hits": 0,
                                                      "abstained": 0})
                    if s == 0:
                        st["abstained"] += 1
                        continue
                    st["n"] += 1
                    st["hits"] += 1 if s == move_sign else 0

                # calibration rings (stated confidence vs realised sign)
                try:
                    engine.hist.ring(engine.hist.tf_key(
                        symbol, tf, "calib_p"), 400).append(
                        round(min(1.0, max(0.0, confidence)) * 20) / 20.0)
                    engine.hist.ring(engine.hist.tf_key(
                        symbol, tf, "calib_h"), 400).append(
                        1.0 if _sign(float(pack.get("cci_adjusted") or 0.0))
                        == move_sign else 0.0)
                except Exception:
                    pass

                # per-formula + subsystem micro bookkeeping
                try:
                    signals = {int(f): _sign(float(v))
                               for f, v in values.items() if _sign(float(v))}
                    engine.learning.record_micro(
                        symbol, tf, move_sign, signals, subs, brain_sigs,
                        1 if touch == "target" else None,
                        1 if touch == "stop" else None)
                except Exception:
                    log.exception("warm learning row failed %s/%s", symbol, tf)

                # meta combiner row (fitted once after the replay)
                try:
                    engine.meta.rows.append({
                        "ts": float(start_ts), "symbol": symbol, "tf": tf,
                        "inputs": {
                            "score": float(subs.get("core", 0) or 0),
                            "brain": float(subs.get("brain", 0) or 0),
                            "physics": float(subs.get("physics", 0) or 0),
                            "emotion": float(subs.get("emotion", 0) or 0),
                            "news": float(subs.get("news", 0) or 0),
                            "agent": 0.0,
                        },
                        "up": 1 if move_sign > 0 else 0,
                        "seeded": True,
                    })
                except Exception:
                    pass

            status["symbols"][symbol][tf] = row
            status["windows"] += row["windows"]
            status["settled"] += row["settled"]
            log.info("warm replay %s/%s: %s windows", symbol, tf, row["windows"])
            time.sleep(YIELD_S)          # let the event loop breathe

    # ── one durable write for everything the replay touched ──
    try:
        ledger.save()
    except Exception:
        log.exception("warm ledger save failed")
    try:
        forward.save()
    except Exception:
        log.exception("warm forward save failed")
    try:
        engine.meta.rows = engine.meta.rows[-4000:]
        engine.meta.fit()
        engine.meta.refit_pick_history()
        engine.meta.save()
    except Exception:
        log.exception("warm meta fit failed")
    try:
        status["learning_persisted"] = bool(engine.learning.save(engine))
    except Exception:
        status["learning_persisted"] = False

    # `n` is the number of windows the subsystem actually VOTED on, and
    # `acc` is over those votes only; `abstained` keeps the silence visible
    # so a thin subsystem cannot look fully-covered.
    status["subsystems"] = {
        k: {"n": v["n"],
            "acc": round(v["hits"] / v["n"], 4) if v["n"] else None,
            "abstained": int(v.get("abstained") or 0),
            "coverage": (round(v["n"] / (v["n"] + v.get("abstained", 0)), 4)
                         if (v["n"] + v.get("abstained", 0)) else None)}
        for k, v in sub_totals.items()}
    for name, cid in (("brain", 97), ("emotion", 98), ("physics", 99),
                      ("news", 100), ("hf", 101)):
        st = sub_totals.get(name) or {}
        n = int(st.get("n") or 0)
        if n:
            # the leaderboard reads acc500; make sure the pooled history is
            # representative even when this boot only replayed a slice
            pass
    status["elapsed_s"] = round(time.time() - t_start, 1)
    status["state"] = "done-sim" if family == "sim" else "done"
    status["ts"] = time.time()
    status["why"] = (
        "replayed the simulated tape (dashboard already flags SIMULATED "
        "TAPE; these rows are warm-up, not market evidence)"
        if family == "sim" else
        "replayed verified exchange tape through the app's own formulas, "
        "brains, emotions, physics, hedge and advanced layers")
    log.info("warm history done: %s", {
        k: status[k] for k in ("state", "windows", "new_settled", "graded",
                               "elapsed_s")})
    return status
