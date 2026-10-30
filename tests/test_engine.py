"""Prediction engine behaviour: phase lock, GRC cadence, sequential targets,
lock strengths, SL protocol branches, Extreme Override gating."""
import asyncio
import time

import pytest

from app.bgn import BgnSynchronizer
from app.clock import next_grc_ts, seconds_to_next_grc, grc_id
from app.config import (
    GRC_INTERVAL_SECONDS,
    LOCK_SHIFT_ADJUST,
    SYMBOL_ORDER,
    TF_ORDER,
    TIMEFRAMES,
)
from app.engine import PredictionEngine
from app.feed import MarketFeed
from app.models import TargetState, TFCycle


# ---------------------------------------------------------------- clock
def test_grc_boundaries_on_the_four_minute_grid():
    import datetime as dt
    for minute in range(60):
        t = dt.datetime(2026, 10, 29, 12, minute, 17, tzinfo=dt.timezone.utc).timestamp()
        nxt = next_grc_ts(t)
        assert nxt > t
        assert (int(nxt) % GRC_INTERVAL_SECONDS) == 0
        assert 0 <= seconds_to_next_grc(t) <= GRC_INTERVAL_SECONDS
        assert (int(nxt) // 60) % 4 == 0    # fires at :00 :04 ... :56


def test_grc_id_monotonic():
    assert grc_id(240.0) == grc_id(479.9)          # same GRC interval
    assert grc_id(480.0) == grc_id(240.0) + 1      # next boundary increments
    assert grc_id(241.0) > grc_id(239.9)


# ---------------------------------------------------------------- engine harness
class DummyFeed(MarketFeed):
    """Feed without network/sleep — synthetic but realistic BTC-like history."""

    async def start(self) -> None:
        import math
        import random
        rng = random.Random(99)
        now = int(time.time())
        start = now - 1010 * 60
        for sym, base in [("BTC", 68000.0), ("PAXG", 2650.0), ("ETH", 3500.0), ("BNB", 600.0)]:
            st = self.states[sym]
            p = base
            for i in range(1010):
                # persistent trend with moderate noise so formulas align and
                # Confidence_final can clear the 0.35 issuance floor
                p *= math.exp(rng.gauss(0.0009, 0.00035))
                st.push_bar(start + i * 60, p, 100.0 + rng.random() * 30)
            st.live_price = p
            st.day_open = st.closes[0]
            self._sim_price[sym] = p
        self._rng = rng
        self.ready.set()

    def tick(self, steps: int = 1) -> None:
        """Advance live prices deterministically between simulated GRCs."""
        import math as _m
        rng = self._rng
        for sym in SYMBOL_ORDER:
            st = self.states[sym]
            for _ in range(steps):
                st.live_price *= _m.exp(rng.gauss(0.0009, 0.00035))
            self._sim_price[sym] = st.live_price

    async def _start_sim(self) -> None:  # pragma: no cover - unused
        pass


def make_engine():
    emitted: list[dict] = []

    async def emit(msg):
        emitted.append(msg)

    feed = DummyFeed()
    engine = PredictionEngine(feed, emit)
    return engine, feed, emitted


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture(scope="module")
def live_engine():
    engine, feed, emitted = make_engine()

    class H:
        pass
    h = H()
    h.engine = engine
    h.feed = feed
    h.emitted = emitted
    run(_boot(engine))
    yield h
    run(engine.stop())


async def _boot(engine=None):
    engine = engine or make_engine()[0]
    await engine.feed.start()          # production main.py does this at startup
    await engine.start()
    # analysis loop converges quickly on the synthetic data
    for _ in range(200):
        if engine.phase == "live":
            break
        await asyncio.sleep(0.05)
    assert engine.phase == "live", "engine failed to leave the analysis phase"
    return engine


# ---------------------------------------------------------------- phase
def test_initial_analysis_locks_early_with_all_cycles(live_engine):
    engine = live_engine.engine
    assert engine.locked
    assert engine.phase == "live"
    assert engine.phase_warning is False       # converged before 4 minutes
    for sym in ["BTC", "PAXG", "ETH", "BNB"]:
        for tf in TF_ORDER:
            cyc = engine.cycles[(sym, tf)]
            assert cyc.cycle_end - cyc.cycle_start == TIMEFRAMES[tf]["minutes"] * 60


def test_tp4_locked_and_tp23_pending_at_lock(live_engine):
    engine = live_engine.engine
    cyc = engine.cycles[("BTC", "tf1")]
    t1, t2, t3, t4 = cyc.targets
    assert t1.status == "locked" and t1.level is not None
    assert t4.status == "locked" and t4.level is not None   # macro thesis locked
    assert t2.status == "pending" and t2.level is None      # NOT calculated yet
    assert t3.status == "pending" and t3.level is None
    assert cyc.active_k == (1 if cyc.issued else None)


# ---------------------------------------------------------------- sequential locks
def test_sequential_tp2_lock_only_on_tp1_hit(live_engine):
    engine, emit = live_engine.engine, live_engine.emitted
    sym, tf = "BTC", "tf1"
    cyc = engine.cycles[(sym, tf)]
    cyc.issued = True                          # unit-level state manufacturing
    cyc.direction = cyc.direction or 1
    # before hit: Tp2 pending
    assert cyc.target(2).status == "pending"
    t1 = cyc.target(1)
    # simulate price reaching Tp1 exactly
    run(engine._on_target_hit(sym, tf, 1, time.time()))
    cyc = engine.cycles[(sym, tf)]
    t2 = cyc.target(2)
    assert t1.status == "hit"
    assert t2.status == "locked" and t2.level is not None   # locked on hit
    assert cyc.active_k == 2
    assert t2.locked_cci is not None
    # Tp3 still pending
    assert cyc.target(3).status == "pending"
    # now hit Tp2
    run(engine._on_target_hit(sym, tf, 2, time.time()))
    cyc = engine.cycles[(sym, tf)]
    assert cyc.target(3).status == "locked" and cyc.target(3).level is not None
    assert cyc.active_k == 3
    run(engine._on_target_hit(sym, tf, 3, time.time()))
    assert engine.cycles[(sym, tf)].active_k is None


# ---------------------------------------------------------------- lock strengths
def test_tp1_adjusts_every_grc_tp2_requires_shift_over_0_3(live_engine):
    engine = live_engine.engine
    sym, tf = "PAXG", "tf1"
    cyc = engine.cycles[(sym, tf)]
    cyc.issued = True                          # unit-level state manufacturing
    cyc.direction = cyc.direction or 1
    res = dict(engine.last_result[(sym, tf)])

    # --- Phase A: Tp1 still pending hit -> adjusts at EVERY GRC, Tp2 untouched
    t1 = cyc.target(1)
    t1_level_before = t1.level
    res_a = dict(res)
    res_a["issued"] = True
    res_a["confidence"] = max(cyc.confidence + 0.05, 0.30)
    res_a["cci_adjusted"] = cyc.cci + 0.1
    engine._apply_grc_lock_rules(sym, tf, time.time(), res_a)
    cyc = engine.cycles[(sym, tf)]
    assert cyc.target(1).level != t1_level_before, "Tp1 failed to adjust at GRC"
    assert cyc.target(2).status == "pending", "Tp2 must not exist before Tp1 hit"

    # --- Tp1 hit -> Tp2 locked with all current data
    run(engine._on_target_hit(sym, tf, 1, time.time()))
    cyc = engine.cycles[(sym, tf)]
    cyc.issued = True
    t2 = cyc.target(2)
    assert t2.status == "locked"
    t2_level_before = t2.level

    # --- Phase B: CCI shift < 0.3 -> Tp2 must NOT move (medium lock strength)
    res_b = dict(res)
    res_b["issued"] = True
    res_b["confidence"] = cyc.confidence
    res_b["cci_adjusted"] = t2.locked_cci + 0.1      # shift 0.1 < 0.3
    engine._apply_grc_lock_rules(sym, tf, time.time(), res_b)
    cyc = engine.cycles[(sym, tf)]
    assert cyc.target(2).level == t2_level_before, "Tp2 adjusted despite shift < 0.3"

    # --- Phase C: CCI shift > 0.3 -> Tp2 re-locks (re-level + fresh reference)
    res_c = dict(res)
    res_c["issued"] = True
    res_c["confidence"] = cyc.confidence
    res_c["cci_adjusted"] = cyc.target(2).locked_cci + 0.4   # shift 0.4 > 0.3
    engine._apply_grc_lock_rules(sym, tf, time.time(), res_c)
    cyc = engine.cycles[(sym, tf)]
    assert cyc.target(2).locked_cci == res_c["cci_adjusted"]


def test_tp3_lock_threshold_is_0_5(live_engine):
    engine = live_engine.engine
    sym, tf = "ETH", "tf1"
    cyc = engine.cycles[(sym, tf)]
    cyc.issued = True                          # unit-level state manufacturing
    cyc.direction = cyc.direction or 1
    if cyc.target(3).status != "locked":
        # sequential: Tp1 hit -> Tp2 locks; Tp2 hit -> Tp3 locks
        run(engine._on_target_hit(sym, tf, 1, time.time()))
        run(engine._on_target_hit(sym, tf, 2, time.time()))
        cyc = engine.cycles[(sym, tf)]
        cyc.issued = True
    t3 = cyc.target(3)
    assert t3.status == "locked"
    assert LOCK_SHIFT_ADJUST["tp3"] == 0.5
    t3.locked_cci = cyc.cci - 0.4        # shift 0.4 < 0.5 -> no adjust
    res = dict(engine.last_result[(sym, tf)])
    res["cci_adjusted"] = cyc.cci
    res["issued"] = True
    level_before = t3.level
    engine._apply_grc_lock_rules(sym, tf, time.time(), res)
    assert engine.cycles[(sym, tf)].target(3).level == level_before
    # shift > 0.5 -> adjust
    engine.cycles[(sym, tf)].target(3).locked_cci = cyc.cci - 0.6
    engine._apply_grc_lock_rules(sym, tf, time.time(), res)
    assert engine.cycles[(sym, tf)].target(3).locked_cci == res["cci_adjusted"]


# ---------------------------------------------------------------- SL protocol
def test_sl_safe_branch_tightens_sl_and_cuts_confidence(live_engine):
    engine, emitted = live_engine.engine, live_engine.emitted
    sym, tf = "BNB", "tf1"
    cyc = engine.cycles[(sym, tf)]
    if cyc.active_k is None:
        # force an active sub-target state
        cyc.issued = True
        cyc.active_k = 1
        cyc.target(1).status = "locked"
        cyc.direction = 1
    cyc.issued = True
    cyc.confidence = 0.7
    sl_before = cyc.sl
    tighten_before = cyc.sl_tighten
    conf_before = cyc.confidence
    # ensure the full recalc agrees with the original direction:
    # make original direction match whatever the engine computes
    res_probe = engine._event_result(sym, tf)
    cyc.direction = res_probe["direction"] if res_probe["direction"] else 1
    if cyc.direction == 0:
        cyc.direction = 1
    emitted.clear()
    run(engine._on_sl_hit(sym, tf, time.time()))
    cyc = engine.cycles[(sym, tf)]
    assert cyc.safe_events == 1
    assert cyc.sl_tighten == tighten_before * 0.85
    assert cyc.confidence == pytest.approx(conf_before - 0.15)
    kinds = [m.get("kind") for m in emitted if m.get("type") == "sl_event"]
    assert "safe" in kinds


def test_sl_close_branch_invalidates_and_waits_for_grc(live_engine):
    engine, emitted = live_engine.engine, live_engine.emitted
    sym, tf = "BTC", "tf3"
    cyc = engine.cycles[(sym, tf)]
    cyc.issued = True
    cyc.pending_reset = False
    cyc.active_k = cyc.active_k or 1
    if cyc.target(cyc.active_k).status != "locked":
        cyc.target(cyc.active_k).status = "locked"
    # manufacture contradiction: force new direction opposite to original
    cyc.direction = 1
    original_res = engine._event_result(sym, tf)
    # monkeypatch event result to opposite direction
    real = engine._event_result

    def flipped(s, t):
        r = dict(real(s, t))
        r["direction"] = -cyc.direction
        return r
    engine._event_result = flipped
    emitted.clear()
    try:
        run(engine._on_sl_hit(sym, tf, time.time()))
    finally:
        engine._event_result = real
    cyc = engine.cycles[(sym, tf)]
    assert cyc.pending_reset is True
    assert cyc.issued is False
    assert all(t.status in ("invalidated", "hit") for t in cyc.targets)
    kinds = [m.get("kind") for m in emitted if m.get("type") == "sl_event"]
    assert "close" in kinds
    # next GRC resets with a fresh prediction
    engine._apply_grc_lock_rules(sym, tf, time.time(),
                                 dict(engine.last_result[(sym, tf)]))
    cyc = engine.cycles[(sym, tf)]
    assert cyc.pending_reset is False
    assert cyc.cycle_start > 0


# ---------------------------------------------------------------- Extreme Override
def test_extreme_override_needs_all_three_conditions(live_engine):
    engine = live_engine.engine
    sym, tf = "PAXG", "tf4"
    cyc = engine.cycles[(sym, tf)]
    cyc.issued = True                          # unit-level state manufacturing
    cyc.direction = cyc.direction or 1
    res = dict(engine.last_result[(sym, tf)])
    res["issued"] = True

    # only two conditions true -> no override
    res["confidence"] = 0.10                 # condition 1 (drop below 0.25)
    t4_dir = cyc.target(4).direction or 1
    # condition 3: fake opposing news — four full-weight regulatory items
    # (0.25 * 1 * 1 * 1 each) push the clamped aggregate past -/+0.8
    engine.news.items = []
    for i in range(4):
        engine.news._append({
            "headline": f"extreme regulatory action {i}",
            "category": "regulatory",
            "sentiment": -1.0 if t4_dir > 0 else 1.0,
            "reliability": 1.0,
            "source": "test",
            "symbols": [sym],
            "ts": time.time(),
        })
    # brain consensus NOT flipped (condition 2 false)
    res["brain"] = dict(res["brain"])
    res["brain"]["polarity"] = t4_dir
    level_before = cyc.target(4).level
    engine._apply_grc_lock_rules(sym, tf, time.time(), res)
    cyc = engine.cycles[(sym, tf)]
    assert cyc.target(4).level == level_before, "Tp4 changed with incomplete conditions"

    # all three conditions true -> override allowed
    res2 = dict(res)
    res2["brain"] = dict(res["brain"])
    res2["brain"]["polarity"] = -t4_dir      # consensus flipped polarity
    engine._apply_grc_lock_rules(sym, tf, time.time(), res2)
    cyc = engine.cycles[(sym, tf)]
    events = [e["type"] for e in cyc.events]
    assert "extreme_override" in events
    engine.news._seed()                      # restore seed


# ---------------------------------------------------------------- cadence
def test_grc_is_the_only_scheduled_mutation(live_engine):
    """Between GRCs, prediction values (confidence, cci, levels) do not change
    from mere ticking — only events mutate them."""
    engine = live_engine.engine
    cyc = engine.cycles[("BTC", "tf2")]
    snap = cyc.to_dict(time.time())
    time.sleep(1.1)
    cyc2 = engine.cycles[("BTC", "tf2")]
    # countdown shrinks but the prediction's own fields are untouched
    assert cyc2.refreshed_at == cyc.refreshed_at
    assert cyc2.cycle_end == snap["cycle_end"]
    assert cyc2.confidence == pytest.approx(snap["confidence"], abs=1e-3)


def test_timeframe_window_constants_match_spec():
    assert TIMEFRAMES["tf1"] == {"minutes": 22, "targets": [5, 10, 15, 22]}
    assert TIMEFRAMES["tf2"] == {"minutes": 44, "targets": [11, 22, 33, 44]}
    assert TIMEFRAMES["tf3"] == {"minutes": 130, "targets": [30, 40, 90, 130]}
    assert TIMEFRAMES["tf4"] == {"minutes": 247, "targets": [60, 120, 180, 247]}
