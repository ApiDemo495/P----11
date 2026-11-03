"""PredictionEngine — orchestrates the entire application.

Responsibilities
  * Initial Analysis Phase (max 4 min countdown, early lock on convergence,
    warning flag on expiry)
  * GRC orchestration: full recalculation of every formula, brain state,
    emotion calculation, news impact, weightage and prediction across all
    four timeframes and all four assets, exactly at :00 :04 ... :56
  * Prediction lock rules (sequential Tp1->Tp2->Tp3, stable Tp4, lock
    strengths 0.3 / 0.5, Extreme Override conditions)
  * Sub-target hit detection (event-driven locks between GRCs), countdown
    rollovers, manual override
  * SL Hit Protocol: Full Recalculation Event with SAFE / CLOSE branches
  * TP/SL placement through Formula 28/29 call sites
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import deque
from typing import Any, Awaitable, Callable

from .bgn import BgnSynchronizer
from .cci import CCIEngine, finalize_confidence
from .clock import grc_id, next_grc_ts, seconds_to_next_grc
from .config import (
    BRAIN_COMPONENT_ID,
    CONFIDENCE_FLOOR,
    EMOTION_COMPONENT_ID,
    EXTREME_OVERRIDE_CCI,
    EXTREME_OVERRIDE_NEWS,
    HEDGE_COMPONENT_ID,
    HIST_COMPOSITE_EMOTION,
    HIST_EMOTION_LAG,
    HIST_STD_R,
    INITIAL_ANALYSIS_MAX_SECONDS,
    LOCK_SHIFT_ADJUST,
    MIN_BARS_COMPUTE,
    MIN_BARS_FULL_WINDOW,
    NEWS_COMPONENT_ID,
    NEWS_CATEGORIES,
    PHYSICS_COMPONENT_ID,
    PHYSICS_NAMES,
    SL_CONFIDENCE_PENALTY,
    SL_TIGHTEN_FACTOR,
    SYMBOL_ORDER,
    TARGET_SCALE,
    TF_ORDER,
    TIMEFRAMES,
    UFP_ALPHA_IDS,
    WEIGHT_IDS,
)
from .brain import BrainSystem
from .emotion import EmotionSystem
from .feed import MarketFeed
from .formulas import FORMULA_FUNCTIONS, FORMULA_NAMES, compute_all
from .formulas.trace import traced_compute
from .forward import ForwardTables
from .meta import MetaCombiner
from . import genesis as genesis_mod

# §2.10 non-directional components: never a direction vote (confidence /
# regime routing only). 97-101 are the brain/emotion/physics/news/hedge
# subsystem inputs to the CCI; 20 is DRHE the regime gauge.
NON_VOTING_IDS = frozenset({20, 97, 98, 99, 100, 101})


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def np_var(xs):
    if not xs:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def _pearson(a, b):
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = a[:n], b[:n]
    ma, mb = _mean(a), _mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, num / (da * db)))
AGENT_DEADLINE_S = 20.0        # §0.1 agent timeout at the 22-min horizon
from .formulas.f01_f10 import abs_changes as _abs_changes
from .formulas._helpers import mean as _mean, pearson as _pearson, sign as _sign, std_pop as _std_pop
from .formulas.f22_f31 import f30_pcv, fractal_boundary_levels
from .history import HistoryStore
from .models import FormulaContext, TargetState, TFCycle
from .news import NewsSystem
from . import hedgefund, orderflow
from .physics import PhysicsSystem
from .risk import arc_ratio, dynamic_level_placement
from .learning import LearningSystem
from .agents import AgentManager
from .historydb import HistoryDB, empty_record

log = logging.getLogger("engine")

AlertEmitter = Callable[[dict[str, Any]], Awaitable[None]]

# Brain-3 dominant market state -> directional bias of the CCI component
STATE_DIRECTIONS = {
    "accumulation": 1,
    "markup": 1,
    "distribution": -1,
    "markdown": -1,
    "ranging": 0,
}


class PredictionEngine:
    def __init__(
        self,
        feed: MarketFeed,
        emit: AlertEmitter,
    ) -> None:
        self.feed = feed
        self.emit = emit
        self.hist = HistoryStore()
        self.bgn = BgnSynchronizer(self.hist)
        self.cci = CCIEngine()
        self.brains = BrainSystem()
        self.emotions = EmotionSystem()
        self.news = NewsSystem()
        self.physics = PhysicsSystem()

        # --- session phase -------------------------------------------------
        self.phase = "analysis"                       # analysis -> live
        self.phase_start = time.time()
        self.phase_deadline = self.phase_start + INITIAL_ANALYSIS_MAX_SECONDS
        self.phase_warning = False
        self.locked = False
        # last traced formula pass per (symbol, tf) — advice §2.1 traces
        self.last_traces: dict[tuple[str, str], dict] = {}
        # forward expectancy tables + flat watch (advice §2.7 / §2.9)
        self.forward = ForwardTables()
        self.meta = MetaCombiner()          # §8.4 meta-combiner + §8.5 ledger
        # §7 genesis: generated formulas under a hard fitness regime
        self.genesis = genesis_mod.GenesisLayer()
        self._genesis_values: dict[tuple[str, str], dict[str, float]] = {}
        # §11.2 agent call discipline: one call per window + hard deadline
        # (lead ≈ agent_timeout + 1.5s + formula pass, ≤2% of W → 20s at 22m)
        self._agent_cache: dict = {}
        self.agent_deadline_misses = 0     # §12 agents: deadline misses
        self.analyse_job: dict | None = None   # staged ANALYSE flow
        self._analyse_task: asyncio.Task | None = None
        self.started_at = time.time()      # §12 ops: uptime
        self.vault_last_write = 0.0        # §12 ops: vault last write
        self.build_id = ""                 # §12 ops: build id vs remote
        # hourly/daily hindsight reviews (advice §8.3)
        self.hindsight_mult: dict[tuple[str, str, int], float] = {}
        self.hindsight_changes: list[dict] = []
        self._hindsight_last: dict[str, float] = {}   # tf -> last review ts
        self._hindsight_reviewed_n: dict[str, int] = {}
        # fused-score history for the noise floor (advice §8.2 step 1)
        from collections import deque as _dq2
        self.score_hist: dict[tuple[str, str], _dq2] = {}
        from collections import deque as _dq
        self.flat: dict[tuple[str, str, int], _dq] = {}
        self.readiness: dict[str, bool] = {
            "bars": False,
            "formulas": False,
            "brains": False,
            "emotions": False,
            "news": False,
            "physics": False,
            "weights": False,
            "agents": True,        # optional subsystem — never blocks analysis
        }

        # --- cycles --------------------------------------------------------
        self.cycles: dict[tuple[str, str], TFCycle] = {}
        self.last_result: dict[tuple[str, str], dict[str, Any]] = {}
        # display snapshots — updated ONLY at GRC cadence (spec)
        self.display_formulas: dict[tuple[str, str], dict[str, Any]] = {}
        self.display_systems: dict[tuple[str, str], dict[str, Any]] = {}
        self.last_snapshot: dict[str, Any] = {}
        self.last_grc: float = 0.0
        self.grc_count: int = 0

        # --- Part-2 cross-GRC state (rewards/learning inputs) --------------
        self.prev_comp: dict[tuple[str, str], dict[int, float]] = {}
        self.prev_brain_signals: dict[tuple[str, str], list[float]] = {}
        self.prev_emotions: dict[tuple[str, str], tuple[dict, dict]] = {}
        self.prev_cci_dir: dict[tuple[str, str], int] = {}
        self.prev_physics: dict[tuple[str, str], dict[int, float]] = {}
        self.brain_weight_adj: dict[tuple[str, str], dict[int, float]] = {}
        self.news_polarity: deque[float] = deque(maxlen=30)
        self._market_cache: dict[str, dict] = {}
        self._market_cache_ts: float = 0.0

        # --- Part-4 learning / agents / history ---------------------------
        self.learning = LearningSystem()
        self.agents = AgentManager()
        from .auditor import FormulaAuditor
        self.auditor = FormulaAuditor()
        from .ledger import SettlementLedger
        self.ledger = SettlementLedger()
        self.locks: dict[tuple[str, str], dict] = {}   # frozen signals
        self._load_locks()
        self.last_window_ts = time.time()              # watchdog anchor
        self.historydb = HistoryDB()
        self.last_agent_signal: dict[str, Any] = {}
        self.last_grc_signals: dict[tuple[str, str], dict] = {}

        self._tasks: list[asyncio.Task] = []
        self._busy = asyncio.Lock()
        self._stop = asyncio.Event()

    # ==================================================================
    # lifecycle
    # ==================================================================
    async def start(self) -> None:
        self.genesis.start_worker()
        self.cci.load()
        # Section 10: learned weights/brain state resume from the JSON file
        self.learning.load_state(engine=self, brains=self.brains)
        try:
            self.historydb.compress_old()
        except Exception:
            log.exception("history retention pass failed")
        await self._initialize_subsystems()
        self._tasks.append(asyncio.create_task(self._analysis_loop()))
        self._tasks.append(asyncio.create_task(self._tick_loop()))

    async def stop(self) -> None:
        self.genesis.stop()
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self.cci.save()
        self.physics.save()
        self.learning.save(self)

    async def _initialize_subsystems(self) -> None:
        # wait for the feed to deliver history
        try:
            await asyncio.wait_for(self.feed.ready.wait(), timeout=60)
        except asyncio.TimeoutError:
            log.warning("feed not ready within 60s; computing with what exists")
        self.brains.initialize()
        self.emotions.initialize()
        self.news.initialize()
        self.physics.initialize()
        self.readiness["brains"] = self.brains.ready
        self.readiness["emotions"] = self.emotions.ready
        self.readiness["news"] = self.news.ready
        self.readiness["physics"] = self.physics.ready
        # bootstrap warm-up: replay the real backfill through the record
        # path so every baseline ring (sigma/tau/SS/R, emotion lag rings)
        # has its first samples IMMEDIATELY — otherwise the first ~5 GRCs
        # (hours at tf3/tf4) show 0.00 for every gated formula (the user's
        # 'most formulas are 0' report). Same tape, no invented data; the
        # reward path is inert (move == 0), so accuracy rings stay clean.
        await asyncio.get_event_loop().run_in_executor(
            None, self._warmup_baselines)

    def _warmup_baselines(self, passes: int = 6) -> None:
        """Replay SLIDING windows of the real backfill through the record
        path so baseline rings hold VARIETY (a static tape gives std≈0,
        which gated F6/F7/F9/F31 to 0.00 even with 'enough' samples)."""
        from .models import FormulaContext
        for symbol in SYMBOL_ORDER:
            st = self.feed.state(symbol)
            closes, times = list(st.closes), list(st.times)
            opens = list(st.opens) if st.opens else list(closes)
            highs = list(st.highs) if st.highs else list(closes)
            lows = list(st.lows) if st.lows else list(closes)
            volumes = list(st.volumes)
            if len(closes) < 30:
                continue
            for tf in TF_ORDER:
                W = TIMEFRAMES[tf]["minutes"]
                for i in range(passes):
                    end = len(closes) - 1 - i * max(1, W // 4)
                    start = end - W
                    if start < 1:
                        break
                    try:
                        ctx = FormulaContext(
                            symbol=symbol, tf=tf, W=W,
                            P=closes[start:end + 1],
                            V=volumes[start:end + 1],
                            T=times[start:end + 1],
                            hist=self.hist,
                            O=opens[start:end + 1],
                            H=highs[start:end + 1],
                            L=lows[start:end + 1],
                        )
                        compute_all(ctx, record=True)
                    except Exception:
                        log.exception("warm-up %s/%s pass %d failed",
                                      symbol, tf, i)
            # emotion lag rings (pidx/octo/dpi/nsp) are written by the
            # compute_asset record path — two passes so E18's lag comparator
            # exists from the first render instead of returning 0.00
            for _ in range(2):
                try:
                    self.compute_asset(symbol, record=True)
                except Exception:
                    log.exception("warm-up compute failed for %s", symbol)

    # ==================================================================
    # context construction
    # ==================================================================
    def build_ctx(self, symbol: str, tf: str) -> FormulaContext:
        st = self.feed.state(symbol)
        W = TIMEFRAMES[tf]["minutes"]
        now_min = int(time.time() // 60) * 60

        times = list(st.times)
        closes = list(st.closes)
        volumes = list(st.volumes)
        opens = list(st.opens) if st.opens else list(st.closes)
        highs = list(st.highs) if st.highs else list(st.closes)
        lows = list(st.lows) if st.lows else list(st.closes)
        live = st.live_price or (closes[-1] if closes else 0.0)

        if times and times[-1] == now_min:
            closes[-1] = live
            volumes[-1] = max(volumes[-1], st.live_volume)
            highs[-1] = max(highs[-1], st.live_high or live, live)
            lows[-1] = min(lows[-1], st.live_low or live, live)
        else:
            times.append(now_min)
            closes.append(live)
            volumes.append(st.live_volume)
            o_live = st.live_open or (closes[-2] if len(closes) > 1 else live)
            opens.append(o_live)
            highs.append(max(st.live_high or live, live, o_live))
            lows.append(min(st.live_low or live, live, o_live))

        # window slice: exactly the timeframe's duration (W+1 samples)
        sl = slice(-(W + 1), None)
        P, V, T = closes[sl], volumes[sl], times[sl]
        O, H, L = opens[sl], highs[sl], lows[sl]
        # keep live candle fields coherent with the live price
        if len(P) and live:
            H[-1] = max(H[-1], live)
            L[-1] = min(L[-1], live)
        return FormulaContext(
            symbol=symbol,
            tf=tf,
            W=W,
            P=P,
            V=V,
            T=T,
            hist=self.hist,
            others=self.feed.others_aligned(symbol),
            O=O,
            H=H,
            L=L,
        )

    def _atr(self, ctx: FormulaContext) -> float:
        if ctx.W_eff < 1:
            return 0.0
        return _mean([abs(x) for x in _abs_changes(ctx.P)])

    # ==================================================================
    # Part-2 market/brain helpers
    # ==================================================================
    def _build_market(self, symbol: str) -> dict:
        """Long-context market bundle for brains & emotions (30/60-min stats,
        peer co-movement, order-flow seams).  Cached for 60s — event recomputes
        between GRCs share the same minute's context."""
        now = time.time()
        cached = self._market_cache.get(symbol)
        if cached is not None and now - self._market_cache_ts < 60.0:
            return cached

        def snap(sym: str) -> tuple[list, list, list, list, list]:
            st = self.feed.state(sym)
            return (
                list(st.closes), list(st.volumes), list(st.times),
                list(st.highs) if st.highs else list(st.closes),
                list(st.lows) if st.lows else list(st.closes),
            )

        closes, volumes, times, highs, lows = snap(symbol)
        live = self.feed.prices().get(symbol) or (closes[-1] if closes else 0.0)
        st = self.feed.state(symbol)
        session_open = st.day_open or (closes[0] if closes else live)
        session_high = max([x for x in (highs + closes + [live]) if x] or [live])

        # BTC closes for E10 absorption
        btc_closes = snap("BTC")[0] if "BTC" in SYMBOL_ORDER else closes

        # peer 1-minute returns (last 10 minutes) for HMC/CMP
        peer_series: dict[str, list[float]] = {}
        for sym in SYMBOL_ORDER:
            c = snap(sym)[0]
            peer_series[sym] = [
                (c[i] - c[i - 1]) / c[i - 1]
                for i in range(1, len(c))
                if c[i - 1]
            ]
        k = min(len(v) for v in peer_series.values()) if peer_series else 0
        peer_returns = [
            [peer_series[sym][k - 10 + j] for sym in SYMBOL_ORDER]
            for j in range(10)
        ] if k >= 10 else []

        # order-flow seams (Part 3 exact replacements live in app.orderflow)
        ofti_value = orderflow.ofti(closes, volumes, highs, lows)
        cmp_value = orderflow.cmp_divergence(peer_returns)
        price_rets = peer_series.get(symbol, [])
        corr = orderflow.news_price_corr(
            price_rets[-10:], list(self.news_polarity)[-10:]
        )

        bundle = {
            "closes": closes,
            "volumes": volumes,
            "times": times,
            "ohlc": {"highs": highs, "lows": lows, "closes": closes},
            "live": live,
            "session_open": session_open,
            "session_high": session_high,
            "btc_closes": btc_closes,
            "peer_returns": peer_returns,
            "ofti": ofti_value,
            "cmp_divergence": cmp_value,
            "news_price_corr": corr,
        }
        self._market_cache[symbol] = bundle
        self._market_cache_ts = now
        return bundle

    def _thing2(self, symbol: str, tf: str, ctx: FormulaContext,
                atr: float, market: dict) -> dict:
        """Per-(asset, timeframe) Thing 2 inputs for the four brain instances."""
        # B1: last 20 returns, z-score normalized
        seg = ctx.P[-21:]
        rets = [
            (seg[i] - seg[i - 1]) / seg[i - 1]
            for i in range(1, len(seg))
        ] if len(seg) > 1 else []
        rets = rets[-20:]
        mu = _mean(rets) if rets else 0.0
        sd = _std_pop(rets) if rets else 0.0
        norm = [((r - mu) / sd if sd > 1e-12 else 0.0) for r in rets]
        while len(norm) < 20:
            norm.insert(0, 0.0)

        # B2: current target distances (previous placement or ATR defaults)
        cyc = self.cycles.get((symbol, tf))
        tp_dist = cyc.tp_dist if (cyc and cyc.tp_dist) else atr * 3.0
        sl_dist = cyc.sl_dist if (cyc and cyc.sl_dist) else atr * 1.0

        # B3: five market-state summary statistics
        w = ctx.P
        lo, hi = (min(w), max(w)) if w else (0.0, 1.0)
        pdi = (w[-1] - lo) / (hi - lo) if hi > lo else 0.5
        vols = ctx.V
        vm = _mean(vols) if vols else 0.0
        vs = _std_pop(vols) if vols else 0.0
        volz = (vols[-1] - vm) / vs if (vols and vs > 1e-12) else 0.0
        moves = [(w[i] - w[i - 6]) for i in range(6, len(w))] if len(w) > 7 else []
        if len(moves) > 1:
            mm, msd = _mean(moves), _std_pop(moves)
            momz = (moves[-1] - mm) / msd if msd > 1e-12 else 0.0
        else:
            momz = 0.0
        mc = market["closes"]
        r50 = (mc[-1] / mc[-51] - 1.0) if len(mc) > 51 else 0.0
        peer = market.get("peer_returns") or []
        agree_fracs = []
        for row in peer:
            vals = [v for v in row if v != 0.0]
            if len(vals) >= 2:
                same = total = 0
                for a in range(len(vals)):
                    for b in range(a + 1, len(vals)):
                        total += 1
                        if (vals[a] > 0) == (vals[b] > 0):
                            same += 1
                if total:
                    agree_fracs.append(same / total)
        agree = _mean(agree_fracs) if agree_fracs else 0.5
        cc_inputs = [
            pdi,
            math.tanh(volz),
            math.tanh(momz),
            math.tanh(20.0 * r50),
            agree,
        ]

        return {
            "norm_returns20": norm,
            "tp_dist": tp_dist,
            "sl_dist": sl_dist,
            "cc_inputs": cc_inputs,
            "news_value": market.get("news_value", 0.0),
            **self.brains.performance_inputs(),
        }

    def _apply_brain4(self, symbol: str, tf: str) -> None:
        """B4 Auto-Learning Controller: omega *= (1 + 0.01 * (A_DN - 0.5)),
        applied on top of the reward rule, then clamped + renormalized."""
        adj = self.brain_weight_adj.get((symbol, tf))
        if not adj:
            return
        w = self.cci.get_weights(symbol, tf)
        for fid in list(w.keys()):
            a = adj.get(fid, 0.0)
            if a:
                w[fid] = w[fid] * (1.0 + 0.01 * a)
            w[fid] = max(0.005, min(0.15, w[fid]))
        total = sum(w.values())
        if total > 0:
            for fid in w:
                w[fid] /= total

    # ==================================================================
    # full computation pass (formulas -> Bgn -> brains/emotion/news/physics -> CCI)
    # ==================================================================
    def compute_asset(self, symbol: str, record: bool, display: bool = False) -> dict[str, dict]:
        matrix = self.bgn.matrix(symbol)   # coherence from RPIC history
        results: dict[str, dict] = {}

        # 1) Formulas 1..20 raw, per timeframe
        raws: dict[str, dict[int, float]] = {}
        ctxs: dict[str, FormulaContext] = {}
        for tf in TF_ORDER:
            ctx = self.build_ctx(symbol, tf)
            ctxs[tf] = ctx
            _anoms: list = []
            raws[tf], _traces = traced_compute(ctx, record=record,
                                               anomalies=_anoms)
            self.last_traces[(symbol, tf)] = _traces
            if _anoms:
                self.auditor.report_anomalies(symbol, tf, _anoms, raws[tf])
            # §7 genesis: evaluate admitted candidates on this window
            # (features from the same tape — one shared feature build)
            _gfeat = genesis_mod.build_features(ctx.P, ctx.V)
            _gact = [e["id"] for e in self.genesis.formulas.values()
                     if e["state"] == "active"]
            self._genesis_values[(symbol, tf)] = \
                genesis_mod.evaluate_active(_gfeat, _gact)

        # 2) Bgn synchronization of every registered formula across timeframes
        adj: dict[str, dict[int, float]] = {tf: {} for tf in TF_ORDER}
        for fid in FORMULA_FUNCTIONS:
            raw_by_tf = {tf: raws[tf].get(fid, 0.0) for tf in TF_ORDER}
            synced = BgnSynchronizer.adjust(matrix, raw_by_tf)
            for tf in TF_ORDER:
                adj[tf][fid] = synced[tf]

        # 3) long market context (30/60-minute stats, peers, order-flow seams)
        market = self._build_market(symbol)
        self.news.refresh()
        news_agg = self.news.aggregate(symbol)
        market["news_value"] = news_agg["value"]
        market["news_impacts"] = news_agg["impacts"]
        # minute-epoch news arrival map (P9 hot flags / P21 events) + the
        # long close series every physics formula needs for c_market/leader
        news_minutes = self.news.arrival_minutes_last(600)
        long_closes = {sym: list(self.feed.state(sym).closes)
                       for sym in SYMBOL_ORDER}

        price = self.feed.prices().get(symbol) or (
            ctxs["tf1"].P[-1] if ctxs["tf1"].P else 0.0
        )
        # move since the previous GRC — drives every reward rule this pass
        move = 0.0
        if record:
            prev_price = self.cci.prev_price.get(symbol)
            if prev_price:
                move = price - prev_price

        # 4) reward-phase updates (previous pass' signals) BEFORE new reads
        if record:
            for tf in TF_ORDER:
                # Brain-1 pattern valence auto-learn (previous pass act)
                self.brains.update_pattern_valence(symbol, tf, move)
                # CCI component rewards (28 weighted components)
                prev_c = self.prev_comp.get((symbol, tf))
                if prev_c and move != 0:
                    rewards = {
                        fid: (1.0 if _sign(prev_c[fid]) == _sign(move) else -1.0)
                        for fid in WEIGHT_IDS
                        if fid in prev_c
                    }
                    prev_bs = self.prev_brain_signals.get((symbol, tf)) or []
                    brain_rewards = {
                        i: (1.0 if _sign(sig) == _sign(move) else -1.0)
                        for i, sig in enumerate(prev_bs) if sig != 0
                    }
                    self.brains.push_rewards(rewards, brain_rewards)
                    # rolling accuracy rings (last 50 scored GRCs) for the
                    # formula panels — same reward rule as weight learning
                    for fid, r in rewards.items():
                        if fid in FORMULA_FUNCTIONS or fid in (
                                28, 30, 97, 98, 99, 100, 101):
                            ring = self.hist.ring(
                                self.hist.tf_key(symbol, tf, f"facc{fid}"), 50
                            )
                            ring.append(r)
                            # auditor trigger (b): rolling accuracy < 45%
                            self.auditor.check_accuracy(
                                symbol, tf, fid, list(ring))
                # Formula 21 weight learning + Brain-4 secondary adjustment
                # (Section 10 micro-learning with adaptive eta rates)
                self.cci.update_weights(symbol, tf, price,
                                        eta=self.learning.rates["micro"])
                self._apply_brain4(symbol, tf)
                # P22 physics alpha learning (same eta reward rule).
                # scale = 5 bps of price: a move >= 5 bps teaches at full
                # magnitude; smaller moves get a scaled-down lesson.
                _scale = max(price * 0.0005, 1e-9)
                prev_ph = self.prev_physics.get((symbol, tf))
                if prev_ph and move != 0:
                    self.physics.reward_update(symbol, tf, prev_ph, move,
                                               eta=self.learning.rates["physics"],
                                               scale=_scale)
                # CERC emotion-weight rewards (21 channels, eta rule)
                prev_em = self.prev_emotions.get((symbol, tf))
                if prev_em and move != 0:
                    self.emotions.reward_update(
                        symbol, tf, prev_em[0], prev_em[1], 0.0, move,
                        eta=self.learning.rates["emotion"],
                        scale=_scale,
                    )
                # agent scorecard: each cloud agent's realized accuracy
                # ("best of best" auto-routing ranks on this)
                if self.last_agent_signal and self.agents.active_slot:
                    _adir = int(self.last_agent_signal.get("direction") or 0)
                    if _adir != 0:
                        _ok = _adir == _sign(move)
                        self.agents.record_outcome(_ok)
                        # agent re-grade ring (advice §8.3: every system)
                        self.hist.ring(
                            self.hist.tf_key(symbol, tf, "facc_agent"), 50
                        ).append(1.0 if _ok else -1.0)
                # consecutive-correct running ratio (0..1, last 10)
                pdir = self.prev_cci_dir.get((symbol, tf), 0)
                if move != 0 and pdir != 0:
                    self.hist.commit_win_streak(symbol, pdir == _sign(move))
                    # F21 (CCI) panel accuracy: sign of the previous pass'
                    # adjusted CCI vs the realized move
                    self.hist.ring(
                        self.hist.tf_key(symbol, tf, "facc21"), 50
                    ).append(1.0 if pdir == _sign(move) else -1.0)

        # 5) brains (4 instances per asset x timeframe) + emotions
        brain_res: dict[str, dict] = {}
        emotion_res: dict[str, dict] = {}
        for tf in TF_ORDER:
            tf_ctx = ctxs[tf]
            atr_tf = self._atr(tf_ctx)
            affinity = {
                fid: abs(raws[tf].get(fid, 0.0))
                for fid in FORMULA_FUNCTIONS
                if abs(raws[tf].get(fid, 0.0)) > 0.7
            }
            self.brains.set_influence(
                symbol, tf, self.learning.brain_influence(symbol, tf)
            )
            brain_res[tf] = self.brains.evaluate(
                symbol, tf, market, affinity, self._thing2(symbol, tf, tf_ctx, atr_tf, market),
                advance=record,
            )
            if record:
                # B1 valence learns from THIS pass' DB-1 pattern activations
                self.brains._prev_pattern_act[(symbol, tf)] = \
                    brain_res[tf]["instances"][0]["pattern_act"]
            self.brain_weight_adj[(symbol, tf)] = brain_res[tf]["weight_adj"]
            emotion_res[tf] = self.emotions.compute(
                tf_ctx, market, brain_res[tf], news_agg, None
            )

        # 6) Bgn synchronization of brain consensus + CERC across timeframes
        brain_sync = BgnSynchronizer.adjust(
            matrix, {tf: brain_res[tf]["consensus"] for tf in TF_ORDER}
        )
        cerc_sync = BgnSynchronizer.adjust(
            matrix, {tf: emotion_res[tf]["cerc"] for tf in TF_ORDER}
        )

        # 7a) physics layer (P1..P21 + P22 UFP) and hedge/dark-pool layer
        # (HF1..HF3) per timeframe.  Physics references the PRE-physics core
        # CCI (P16 sign, P22 coupling correlation, P19 top-5 gauge check);
        # HF3 needs this pass' brain consensus (retail_flow = -H_signal).
        phys_raw: dict[str, dict] = {}
        hedge_raw: dict[str, float] = {}

        def _recompute(g: FormulaContext) -> dict[int, float]:
            # P19 gauge recomputes: full formula pass on transformed prices,
            # record=False so the GRC-cadence rings stay untouched
            return compute_all(g, record=False)

        for tf in TF_ORDER:
            drhe = adj[tf].get(20, 1.0)
            comp_core: dict[int, float] = {fid: adj[tf][fid] for fid in FORMULA_FUNCTIONS}
            comp_core[BRAIN_COMPONENT_ID] = brain_sync[tf]
            comp_core[EMOTION_COMPONENT_ID] = cerc_sync[tf]
            w = self.cci.get_weights(symbol, tf)
            wsum = sum(w.get(fid, 0.0) for fid in comp_core)
            cci_ref = (
                sum(w.get(fid, 0.0) * v for fid, v in comp_core.items()) / wsum
                if wsum else 0.0
            )
            phys_raw[tf] = self.physics.evaluate(
                symbol, tf, ctxs[tf], atr=self._atr(ctxs[tf]), raws=raws[tf],
                cci_ref=cci_ref, long_closes=long_closes,
                news_minutes=news_minutes, weights=w,
                recompute=_recompute, hist=self.hist,
                record=record, move=move,
            )
            ifr = hedgefund.institutional_flow(ctxs[tf])
            dpad = hedgefund.dark_pool_flow(ctxs[tf])
            smdi = hedgefund.smart_money_divergence(
                ifr["value"], brain_res[tf]["consensus"]
            )
            hedge_raw[tf] = hedgefund.combined_signal(
                ifr["value"], dpad["value"], smdi
            )
            # per-tf hedge detail kept for the display layer
            phys_raw[tf]["hedge"] = {
                "ifr": round(ifr["value"], 6),
                "dpad": round(dpad["value"], 6),
                "smdi": round(smdi, 6),
                "events": ifr["events"],
                "prints": dpad["prints"],
                "combined": round(hedge_raw[tf], 6),
            }

        # 7b) Bgn synchronization of the part-3 signals across timeframes
        physics_sync = BgnSynchronizer.adjust(
            matrix, {tf: phys_raw[tf]["ufp"]["value"] for tf in TF_ORDER}
        )
        hedge_sync = BgnSynchronizer.adjust(matrix, hedge_raw)
        news_sync = BgnSynchronizer.adjust(
            matrix, {tf: news_agg["value"] for tf in TF_ORDER}
        )

        # 7) Formula 21 per timeframe: 31 weighted components + confidence
        for tf in TF_ORDER:
            atr_tf = self._atr(ctxs[tf])        # per-tf ATR (fixes leak from step 5)
            drhe = adj[tf].get(20, 1.0)
            comp: dict[int, float] = {fid: adj[tf][fid] for fid in FORMULA_FUNCTIONS}
            comp[BRAIN_COMPONENT_ID] = brain_sync[tf]
            comp[EMOTION_COMPONENT_ID] = cerc_sync[tf]
            comp[PHYSICS_COMPONENT_ID] = physics_sync[tf]
            comp[NEWS_COMPONENT_ID] = news_sync[tf]
            comp[HEDGE_COMPONENT_ID] = hedge_sync[tf]
            phys = phys_raw[tf]
            # Section 14 Level 3/4: performance-scaled raw weights
            base_w = self.cci.get_weights(symbol, tf)
            raw_mod = {
                fid: base_w.get(fid, 0.0)
                * self.learning.performance_multiplier(fid, "tf", tf)
                * self.learning.performance_multiplier(fid, "sym", symbol)
                * self._hindsight_factor(symbol, tf, fid)
                for fid in WEIGHT_IDS
            }
            pack = self.cci.compute(
                symbol, tf, comp, drhe, tme=phys["tme"],
                shares=self.learning.subsystem_shares,
                raw_weights=raw_mod,
            )

            # F30 PCV validator — runs on the CCI (post-aggregation)
            pcv_detail = f30_pcv(ctxs[tf].P, pack["cci_adjusted"])
            pcv = pcv_detail["pcv"]

            br = brain_res[tf]
            em = emotion_res[tf]
            state_dir = STATE_DIRECTIONS.get(br["dominant"], 0)
            aligned = state_dir != 0 and state_dir == _sign(pack["cci_adjusted"])
            fin = finalize_confidence(
                pack,
                experience_quality=br["experience"],
                p_dominant=br["p_dominant"],
                state_aligned=aligned,
                rtp=em["rtp"],
                pcv=pcv,
                wfcp=phys["wfcp"],
                drpt=phys["drpt"],
                gspi=phys["gspi"],
            )
            # advice §5.1: the emotion fusion weight is a CONFIDENCE
            # MODULATOR of ±15 %, never the direction ("formulas over
            # emotions" — after emotions out-voted the tape). Agreement
            # with the pick lifts conf up to +15 %, disagreement cuts it
            # up to −15 %; the direction itself never moves here.
            _dir0 = _sign(pack["cci_adjusted"])
            _band = 0.15 * float(cerc_sync[tf]) * _dir0
            if _band:
                fin = dict(fin)
                fin["confidence"] = max(0.0, min(
                    1.0, float(fin["confidence"]) * (1.0 + _band)))
                fin["issued"] = fin["confidence"] > CONFIDENCE_FLOOR
                fin["emotion_band"] = round(_band, 4)

            # F28 inputs: volatility state vs baseline (same measure as F20)
            window_rets = [
                (ctxs[tf].P[i] - ctxs[tf].P[i - 1]) / ctxs[tf].P[i - 1]
                for i in range(1, len(ctxs[tf].P))
                if ctxs[tf].P[i - 1]
            ]
            sigma_cur = _std_pop(window_rets) if window_rets else 0.0
            base_ring = self.hist.ring(self.hist.tf_key(symbol, tf, "std_r"), HIST_STD_R)
            sigma_base = base_ring.mean() if len(base_ring) else sigma_cur
            arc, ratio_band = arc_ratio(
                pack["cci_adjusted"], sigma_cur, sigma_base,
                br["strength"], confidence=fin["confidence"],
            )

            brain_consensus = {
                "polarity": br["polarity"],
                "strength": br["strength"],
                "agreement": br["agreement"],
                "consensus": br["consensus"],
                "values": {i["id"]: round(i["h_signal"], 5) for i in br["instances"]},
                "instances": [
                    {k: (round(v, 5) if isinstance(v, float) else v)
                     for k, v in inst.items() if k != "weight_adj"}
                    for inst in br["instances"]
                ],
                "hormones_avg": {k: round(v, 5) for k, v in br["hormones_avg"].items()},
                "dominant": br["dominant"],
                "p_dominant": br["p_dominant"],
                "state_dist": br["state_dist"],
                "experience": round(br["experience"], 5),
            }
            emotion_agg = {
                "aggregate": cerc_sync[tf],
                "polarity": 1 if cerc_sync[tf] > 1e-6 else (
                    -1 if cerc_sync[tf] < -1e-6 else 0
                ),
                "cerc": em["cerc"],
                "emotions": em["emotions"],
                "directions": em["directions"],
                "rtp": em["rtp"],
                "tedr": em["tedr"],
                # advice §5: measured features + print accounting (D-7)
                "features": em.get("features") or {},
                "zero_reasons": em.get("zero_reasons") or {},
                "prints": {"quote_prints": self.feed.quote_prints,
                           "note": "quote prints excluded from volume/side stats"},
            }
            # Section 22: human-readable reasoning, not just numbers
            _top_e = sorted(em["emotions"].items(),
                            key=lambda kv: -abs(kv[1]))[:4]
            _eparts = [
                f"E{eid} {v:+.2f}{'▲' if em['directions'].get(eid, 0) > 0 else '▼' if em['directions'].get(eid, 0) < 0 else '·'}"
                for eid, v in _top_e if abs(v) > 0.01
            ]
            emotion_agg["narrative"] = (
                ("Dominant flows: " + "; ".join(_eparts) + ". ") if _eparts else
                "All 22 emotion channels flat. "
            ) + (
                f"Net CERC {em['cerc']:+.4f} — " + (
                    "bulls in control." if em["cerc"] > 0.02 else
                    "bears in control." if em["cerc"] < -0.02 else
                    "balanced tug-of-war (conflict keeps confidence low)."
                )
            ) + (f" Trauma index RTP {em['rtp']:.2f} > 0.6 — confidence suppressed."
                 if em["rtp"] > 0.6 else "")

            results[tf] = {
                "raw": raws[tf],
                "adj": adj[tf],
                "atr": atr_tf,
                "sigma_cur": sigma_cur,
                "sigma_base": sigma_base,
                "arc": {"value": round(arc, 4), "ratio": ratio_band},
                "cci": pack["cci"],
                "cci_adjusted": pack["cci_adjusted"],
                "confidence": fin["confidence"],
                "base_confidence": pack["confidence"],
                "direction": pack["direction"],
                "issued": fin["issued"],
                "drhe": drhe,
                "pcv": pcv,
                "pcv_detail": pcv_detail,
                "conflicting": fin["conflicting"],
                "brain": brain_consensus,
                "emotion": emotion_agg,
                "physics": {
                    "mechanisms": phys.get("mechanisms") or [],
                    "estimators": phys.get("estimators") or {},
                    "values": phys["values"],
                    "ufp": phys["ufp"]["value"],
                    "tme": phys["tme"],
                    "wfcp": phys["wfcp"],
                    "drpt": phys["drpt"],
                    "gspi": phys["gspi"],
                    "p1": phys["details"].get("p1"),
                    "alpha": {str(k): round(v, 5) for k, v in
                              self.physics._weights(symbol, tf).items()},
                    "accuracy": {str(k): round(v, 4) for k, v in
                                 self.physics._accuracy(symbol, tf).items()},
                    "n_updates": self.physics.n_updates.get((symbol, tf), 0),
                },
                "hedge": phys["hedge"],
                "news_value": news_sync[tf],
                "qdpc_tau": phys["details"]["p21"]["tau"],
                "weight_profile": self._weight_profile(symbol, tf, {
                    "raw": raws[tf], "drhe": drhe,
                    "physics": {"ufp": phys["ufp"]["value"]},
                }),
                "agent": None,           # filled by _apply_agents at GRC
                "double_check": None,
                "lock": dict(self.locks.get((symbol, tf)) or {}),
            }
            if record:
                # reward-rule storage: the 28 weighted components only
                # (DRHE is a multiplier, not a rewarded signal)
                comp_w = {fid: comp[fid] for fid in WEIGHT_IDS}
                self.cci.store_signals(symbol, tf, comp_w)
                self.prev_comp[(symbol, tf)] = comp_w
                self.prev_brain_signals[(symbol, tf)] = [
                    inst["h_signal"] for inst in br["instances"]
                ]
                self.prev_emotions[(symbol, tf)] = (em["emotions"], em["directions"])
                self.prev_cci_dir[(symbol, tf)] = pack["direction"]
                self.prev_physics[(symbol, tf)] = dict(phys["values"])
                # F21 sparkline: the aggregator's own adjusted output
                self.hist.ring(
                    self.hist.tf_key(symbol, tf, "fspark21"), 100
                ).append(pack["cci_adjusted"])
                # GRC-lag rings consumed by the emotion system
                for ring_name, val in em["lag_values"].items():
                    self.hist.ring(
                        self.hist.tf_key(symbol, tf, ring_name), HIST_EMOTION_LAG
                    ).append(val)
                self.hist.ring(
                    self.hist.tf_key(symbol, tf, "compo"), HIST_COMPOSITE_EMOTION
                ).append(em["cerc"])
                # news polarity series for the E20 correlation seam
                self.news_polarity.append(news_agg.get("polarity", 0))
                # --- sparklines (last 100 GRCs) for the formula panels -----
                for f in FORMULA_FUNCTIONS:
                    self.hist.ring(
                        self.hist.tf_key(symbol, tf, f"fspark{f}"), 100
                    ).append(raws[tf].get(f, 0.0))
                self.hist.ring(
                    self.hist.tf_key(symbol, tf, "fspark28"), 100
                ).append(arc)
                self.hist.ring(
                    self.hist.tf_key(symbol, tf, "fspark30"), 100
                ).append(pcv)
                # physics sparklines P1..P21 + P22 (UFP) + accuracies
                pvals = phys["values"]
                for pid in range(1, 22):
                    self.hist.ring(
                        self.hist.tf_key(symbol, tf, f"pspark{pid}"), 100
                    ).append(pvals[pid])
                self.hist.ring(
                    self.hist.tf_key(symbol, tf, "pspark22"), 100
                ).append(phys["ufp"]["value"])
                if move != 0:
                    for pid in range(1, 22):
                        self.hist.ring(
                            self.hist.tf_key(symbol, tf, f"pacc{pid}"), 50
                        ).append(1.0 if _sign(pvals[pid]) == _sign(move)
                                 else -1.0)
                    self.hist.ring(
                        self.hist.tf_key(symbol, tf, "pacc22"), 50
                    ).append(1.0 if _sign(phys["ufp"]["value"]) == _sign(move)
                             else -1.0)
                # emotion sparklines + accuracy (same term as reward_update)
                ew = self.emotions._weights(symbol, tf)
                for i, (eid, ev) in enumerate(em["emotions"].items()):
                    self.hist.ring(
                        self.hist.tf_key(symbol, tf, f"emospark{eid}"), 100
                    ).append(ev)
                    if move != 0 and eid in em["directions"]:
                        if eid <= 21 and i < len(ew):
                            term = ew[i] * ev * em["directions"][eid]
                            if eid == 21:
                                term *= (1.0 - 0.5 * ev)
                        else:
                            term = em["cerc"]
                        self.hist.ring(
                            self.hist.tf_key(symbol, tf, f"eacc{eid}"), 50
                        ).append(1.0 if _sign(term) == _sign(move) else -1.0)
                # E22 (CERC) sparkline — not part of the emotions dict
                self.hist.ring(
                    self.hist.tf_key(symbol, tf, "emospark22"), 100
                ).append(em["cerc"])
                if move != 0:
                    self.hist.ring(
                        self.hist.tf_key(symbol, tf, "eacc22"), 50
                    ).append(1.0 if _sign(em["cerc"]) == _sign(move) else -1.0)
                # --- System Co-relation series (last 50 GRCs) --------------
                for i, inst in enumerate(br["instances"]):
                    self.hist.ring(
                        self.hist.tf_key(symbol, tf, f"corr_db{i}"), 50
                    ).append(inst["h_signal"])
                self.hist.ring(
                    self.hist.tf_key(symbol, tf, "corr_em"), 50
                ).append(cerc_sync[tf])
                self.hist.ring(
                    self.hist.tf_key(symbol, tf, "corr_ph"), 50
                ).append(physics_sync[tf])
                self.hist.ring(
                    self.hist.tf_key(symbol, tf, "corr_cci"), 50
                ).append(pack["cci_adjusted"])
                # display snapshot — GRC cadence only (spec)
                self.display_formulas[(symbol, tf)] = {
                    "raw": {str(f): round(raws[tf].get(f, 0.0), 5)
                            for f in FORMULA_FUNCTIONS},
                    "adj": {str(f): round(adj[tf].get(f, 0.0), 5)
                            for f in FORMULA_FUNCTIONS},
                    "names": {str(f): FORMULA_NAMES[f] for f in FORMULA_FUNCTIONS},
                    "extra": {"28": round(arc, 5), "30": round(pcv, 5)},
                    "extra_names": {"28": "ARC", "29": "DLP", "30": "PCV"},
                }
                # --- WHY this prediction: full CCI transparency (Section 21)
                _comp_names = {
                    **{str(f): FORMULA_NAMES[f] for f in FORMULA_FUNCTIONS},
                    str(BRAIN_COMPONENT_ID): "Drosophila brain consensus",
                    str(EMOTION_COMPONENT_ID): "CERC emotion flow",
                    str(PHYSICS_COMPONENT_ID): "Physics UFP (P22)",
                    str(NEWS_COMPONENT_ID): "News impact aggregate",
                    str(HEDGE_COMPONENT_ID): "HF institutional flow",
                }
                _ew = pack.get("weights") or base_w
                _comps = sorted(
                    ({
                        "id": str(fid),
                        "name": _comp_names.get(str(fid), f"F{fid}"),
                        "value": round(v, 5),
                        "weight": round(_ew.get(fid, 0.0), 5),
                        "contribution": round(_ew.get(fid, 0.0) * v, 5),
                    } for fid, v in comp.items()),
                    key=lambda c: -abs(c["contribution"]),
                )
                _top3 = _comps[:3]
                _agree = sum(1 for c in _top3
                             if _sign(c["value"]) == _sign(pack["cci_adjusted"]))
                _reason = (
                    f"CCI {pack['cci_adjusted']:+.4f} from {len(comp)} weighted "
                    f"components"
                    + ("; top movers: " + ", ".join(
                        f"{c['name']} {c['value']:+.2f}" for c in _top3)
                       if _top3 else "")
                    + (f". Top signals {'reinforce' if _agree >= 2 else 'conflict'} "
                       "the net direction. "
                       if _top3 else ". ")
                    + (f"Trend decoherence P6 ×{fin['drpt']:.2f} — choppy market. "
                       if fin.get("drpt", 0) > 0.1 else
                       "Physics multipliers near neutral. ")
                    + (f"PCV conflict penalty {pcv:+.2f}. " if pcv < 0 else "")
                    + ("Confidence cleared the 0.35 issue floor — signal issued."
                       if fin["issued"] else
                       f"Confidence {fin['confidence']:.3f} below the 0.35 floor — "
                       "correct call is NO TRADE until signals align.")
                )
                # advice §8.2 direction ladder + §8.1 weighted mass —
                # printed with the pick so the WHY panel shows WHICH rung
                # decided the side (never a bare arrow).
                _prev_lock = self.locks.get((symbol, tf)) or {}
                _now_l = time.time()
                _prev_side = (int(_prev_lock.get("side") or 0)
                              if float(_prev_lock.get("window_end") or 0) < _now_l
                              else self.prev_cci_dir.get((symbol, tf), 0))
                _lad, _lad_src, _floor = self._direction_ladder(
                    symbol, tf, float(pack["cci_adjusted"]), _comps,
                    brain_consensus, _prev_side)
                pack["direction"] = _lad
                results[tf]["direction"] = _lad
                results[tf]["ladder"] = {
                    "source": _lad_src, "noise_floor": round(_floor, 5),
                    "score": round(float(pack["cci_adjusted"]), 5),
                    "rung": (_lad_src.split("→")[0][:40] or "fused score"),
                }
                results[tf]["mass"] = self._mass_for_against(_comps, _lad)
                # §8.6 learners: edge guard, temporal twin, BOCPD clock,
                # Hawkes boundary intensity — printed features, never
                # silently absorbed into the score
                results[tf]["learners"] = self._learners(symbol, tf, ctxs[tf])
                try:
                    _gv = self.genesis.live_score(
                        self._genesis_values.get((symbol, tf)) or {})
                    results[tf]["genesis"] = _gv
                except Exception:
                    results[tf]["genesis"] = {"score": 0.0, "weight": 0.0,
                                              "n_active": 0}
                self.prev_cci_dir[(symbol, tf)] = _lad
                self.display_systems[(symbol, tf)] = {
                    "lock": dict(self.locks.get((symbol, tf)) or {}),
                    "ladder": dict(results[tf]["ladder"]),
                    "mass": results[tf]["mass"],
                    "genesis": dict(results[tf].get("genesis") or {}),
                    "learners": dict(results[tf].get("learners") or {}),
                    "brain": brain_consensus,
                    "emotion": {
                        "aggregate": round(cerc_sync[tf], 5),
                        "cerc": em["cerc"],
                        "features": em.get("features") or {},
                        "zero_reasons": em.get("zero_reasons") or {},
                        "emotions": em["emotions"],
                        "directions": em["directions"],
                        "rtp": round(em["rtp"], 4),
                        "tedr": round(em["tedr"], 4),
                        "narrative": emotion_agg["narrative"],
                        "weights": [round(x, 5) for x in self.emotions._weights(symbol, tf)],
                        "accuracy": [round(x, 4) for x in
                                     self.emotions._accuracy(symbol, tf)],
                        "n_updates": self.emotions.n_updates.get((symbol, tf), 0),
                    },
                    "physics": {
                        "mechanisms": phys.get("mechanisms") or [],
                        "estimators": phys.get("estimators") or {},
                        "values": phys["values"],
                        "ufp": round(physics_sync[tf], 5),
                        "alpha": phys["ufp"]["alpha"],
                        "alpha_accuracy": {str(k): round(v, 4) for k, v in
                                           self.physics._accuracy(symbol, tf).items()},
                        "n_updates": self.physics.n_updates.get((symbol, tf), 0),
                        "couplings": phys["ufp"]["couplings"],
                        "tme": round(phys["tme"], 5),
                        "wfcp": round(phys["wfcp"], 5),
                        "drpt": round(phys["drpt"], 5),
                        "gspi": round(phys["gspi"], 5),
                        "p1": phys["details"].get("p1"),
                        "qtpb": phys["details"].get("p4", {}),
                        # every physics formula's detail dict (Term Breakdown)
                        "details": {
                            k: v for k, v in phys["details"].items()
                            if str(k).startswith("p")
                        },
                    },
                    "news": {
                        "value": news_sync[tf],
                        "bull": news_agg["bull"],
                        "bear": news_agg["bear"],
                        "polarity": news_agg["polarity"],
                        "count": news_agg["count"],
                        "impacts": news_agg["impacts"],
                        "source": news_agg["source"],
                        "items": news_agg["items"],
                    },
                    "hedge": phys["hedge"],
                    "drhe": round(drhe, 4),
                    "atr": results[tf]["atr"],
                    "pcv": round(pcv, 5),
                    "conflicting": fin["conflicting"],
                    "arc": round(arc, 4),
                    "arc_ratio": ratio_band,
                    "sigma": {"cur": round(sigma_cur, 8),
                              "base": round(sigma_base, 8)},
                    "confidence": round(fin["confidence"], 4),
                    "base_confidence": round(pack["confidence"], 4),
                    "physics_conf": {
                        "wfcp": round(phys["wfcp"], 5),
                        "drpt": round(phys["drpt"], 5),
                        "gspi": round(phys["gspi"], 5),
                        "qdpc_tau": phys["details"]["p21"]["tau"],
                    },
                    "corelation": self._corelation_dict(symbol, tf),
                    "win_ratio": round(self.hist.win_ratio.get(symbol, 0.0), 3),
                    # WHY-panel payload: components, chain, reasoning
                    "breakdown": {
                        "components": _comps[:14],
                        "n_components": len(_comps),
                        "sum_contributions": round(
                            sum(c["contribution"] for c in _comps), 5),
                        "multipliers": {
                            "drhe": round(drhe, 4),
                            "eq": round(br["experience"], 4),
                            "p_dominant": br["p_dominant"],
                            "dominant": br["dominant"],
                            "state_aligned": bool(aligned),
                            "rtp": round(em["rtp"], 4),
                            "pcv": round(pcv, 4),
                            "wfcp": round(phys["wfcp"], 4),
                            "drpt": round(phys["drpt"], 4),
                            "gspi": round(phys["gspi"], 4),
                        },
                        "base_confidence": round(pack["confidence"], 5),
                        "final_confidence": round(fin["confidence"], 5),
                        "floor": 0.35,
                        "votes": {
                            "pcv": 1 if pcv > 0 else (-1 if pcv < 0 else 0),
                            "brain": br["polarity"],
                            "emotion": 1 if em["cerc"] > 0 else (
                                -1 if em["cerc"] < 0 else 0),
                            "physics": 1 if physics_sync[tf] > 0 else (
                                -1 if physics_sync[tf] < 0 else 0),
                            "news": 1 if news_sync[tf] > 0 else (
                                -1 if news_sync[tf] < 0 else 0),
                            "hedge": 1 if hedge_sync[tf] > 0 else (
                                -1 if hedge_sync[tf] < 0 else 0),
                        },
                    },
                    "reason": _reason,
                    # Part-4 sync: the values the dashboard quadrants read
                    "cci_adjusted": pack["cci_adjusted"],
                    "direction": pack["direction"],
                    "issued": bool(fin["issued"]),
                    "weight_profile": self._weight_profile(symbol, tf, {
                        "raw": raws[tf], "drhe": drhe,
                        "physics": {"ufp": phys["ufp"]["value"]},
                    }),
                    "agent": None,          # patched by _apply_agents
                    "double_check": None,   # patched by _apply_agents
                }
        if record:
            self.cci.store_price(symbol, price)
            self.readiness["formulas"] = True
            self.readiness["weights"] = True
        return results

    # ==================================================================
    # visualization data (Section 9)
    # ==================================================================
    def _corelation_dict(self, symbol: str, tf: str) -> dict:
        """System Co-relation network: pairwise correlation of every major
        subsystem's output over the last 50 GRCs (green = positive,
        red = negative; thickness is drawn client-side from |corr|)."""
        def series(name: str) -> list[float]:
            return list(self.hist.ring(self.hist.tf_key(symbol, tf, name), 50))

        local = {
            "DB-1": series("corr_db0"), "DB-2": series("corr_db1"),
            "DB-3": series("corr_db2"), "DB-4": series("corr_db3"),
            "EMOTION": series("corr_em"), "PHYSICS": series("corr_ph"),
            "CORE": series("corr_cci"),
        }
        edges: list[dict] = []
        keys = list(local)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                xs, ys = local[a], local[b]
                n = min(len(xs), len(ys))
                if n >= 3:
                    corr = _pearson(xs[-n:], ys[-n:])
                    edges.append({"a": a, "b": b, "corr": round(corr, 4)})
        return {
            "nodes": [
                {"id": "DB-1", "kind": "brain"}, {"id": "DB-2", "kind": "brain"},
                {"id": "DB-3", "kind": "brain"}, {"id": "DB-4", "kind": "brain"},
                {"id": "EMOTION", "kind": "emotion"},
                {"id": "PHYSICS", "kind": "physics"},
                {"id": "CORE", "kind": "core"},
                {"id": "AGENT-G", "kind": "agent", "ready": False},
                {"id": "AGENT-L", "kind": "agent", "ready": False},
            ],
            "edges": edges,
        }

    def _direction_ladder(self, symbol: str, tf: str, score: float,
                          comps: list, brain: dict, prev_side: int) -> tuple[int, str, float]:
        """§8.2 direction ladder — always returns a side, and PRINTS which
        rung decided it.  1) fused score beyond the measured noise floor
        max(0.02, 0.5·median|score| of the last 60 windows)
        2) the horizon-matched window formulas' lean (magnitude-weighted,
           tie-risk/non-directional excluded)
        3) the brain's vote if confidence ≥ 25%
        4) the previous window's side
        5) BUY.   HOLD does not exist — the contest forces a side."""
        hist = self.score_hist.setdefault((symbol, tf), __import__("collections").deque(maxlen=60))
        hist.append(abs(score))
        if len(hist) >= 10:
            ordered = sorted(hist)
            med = ordered[len(ordered) // 2]
        else:
            med = 0.0
        floor = max(0.02, 0.5 * med)

        # rung 0 (advice §8.4): the meta-combiner takes the side ONLY with
        # a 40-window strict-sign record ≥ 53 % and ≥ the recipe
        try:
            take, take_msg = self.meta.takeover(tf)
        except Exception:
            take, take_msg = False, "meta unavailable"
        if take:
            def _comp_val(cid: int) -> float:
                for c in comps or []:
                    if str(c.get("id")) == str(cid):
                        return float(c.get("value") or 0.0)
                return 0.0
            mp = self.meta.meta_pick({
                "score": score, "brain": _comp_val(97), "physics": _comp_val(99),
                "emotion": _comp_val(98), "news": _comp_val(100),
                "agent": _comp_val(101)})
            if mp in (1, -1):
                return (mp, f"{take_msg} → side {'UP' if mp > 0 else 'DOWN'}", floor)

        # rung 1: the fused score itself
        if abs(score) > floor:
            d = 1 if score > 0 else -1
            return (d,
                    f"fused score {score:+.4f} leans {'up' if d > 0 else 'down'} — "
                    f"|score| {abs(score):.4f} > noise floor {floor:.4f}",
                    floor)

        # rung 2: window-formula lean (directional formula components only)
        lean = 0.0
        lean_top = []
        for c in comps or []:
            try:
                fid = int(c.get("id"))
            except (TypeError, ValueError):
                continue
            # §2.10: non-directional indicators NEVER vote — DRHE (20) and
            # the subsystem components (brain/emotion/physics/news/hedge)
            # modulate confidence and route regimes only.
            if fid in NON_VOTING_IDS:
                continue
            lean += float(c.get("contribution") or 0.0)
            if abs(float(c.get("contribution") or 0.0)) > 1e-9:
                lean_top.append(f"{c.get('name', fid)} {float(c['contribution']):+.4f}")
        # §7: genesis is ONE fusion vote whose weight is earned by the
        # active set's record — it joins the lean, printed with it
        try:
            gv = self.genesis.vote()
            g_vals = self._genesis_values.get((symbol, tf)) or {}
            gscore = self.genesis.live_score(g_vals)
            g_term = float(gscore.get("score") or 0.0) * float(gscore.get("weight") or 0.0)
            if g_term:
                lean += g_term
                lean_top.append(f"genesis {g_term:+.4f} (w={gscore['weight']}, "
                                f"n_active={gscore['n_active']})")
        except Exception:
            pass
        if abs(lean) > 1e-6:
            d = 1 if lean > 0 else -1
            return (d,
                    f"score {score:+.4f} inside noise floor {floor:.4f} → "
                    f"formula lean {lean:+.4f} ({'; '.join(lean_top[:3])})",
                    floor)

        # rung 3: brain vote at ≥ 25 % confidence
        b_dir = int((brain or {}).get("polarity") or 0)
        b_str = float((brain or {}).get("strength") or 0.0)
        if b_dir != 0 and b_str >= 0.25:
            return (b_dir,
                    f"score and lean both flat (floor {floor:.4f}) → brain "
                    f"vote {'UP' if b_dir > 0 else 'DOWN'} "
                    f"(strength {b_str:.2f} ≥ 0.25)",
                    floor)

        # rung 4: previous window's side
        if prev_side in (1, -1):
            return (prev_side,
                    f"no decisive input (floor {floor:.4f}) → previous "
                    f"window's side {'UP' if prev_side > 0 else 'DOWN'}",
                    floor)

        # rung 5: BUY
        return (1,
                f"all rungs flat (noise floor {floor:.4f}) → BUY "
                f"(contest forces a side — HOLD does not exist)",
                floor)

    @staticmethod
    def _mass_for_against(comps: list, side: int) -> dict:
        """§8.1: print the consensus as weighted mass for and against, plus
        the strongest supporters and opponents (4–6), and the non-directional
        subsystem votes separately (they modulate, never vote)."""
        for_mass = against = 0.0
        supporters: list[dict] = []
        opponents: list[dict] = []
        subsystems: dict[str, float] = {}
        for c in comps or []:
            contribution = float(c.get("contribution") or 0.0)
            try:
                fid = int(c.get("id"))
            except (TypeError, ValueError):
                fid = None
            if fid is None or fid in NON_VOTING_IDS:
                subsystems[c.get("name", "?")] = round(contribution, 5)
                continue
            sgn = 1 if contribution > 0 else -1
            if side == 0 or sgn == side:
                for_mass += abs(contribution)
                supporters.append(c)
            else:
                against += abs(contribution)
                opponents.append(c)
        supporters.sort(key=lambda c: -abs(float(c.get("contribution") or 0)))
        opponents.sort(key=lambda c: -abs(float(c.get("contribution") or 0)))
        pick = lambda lst: [{"name": c.get("name"), "value": c.get("value"),
                             "contribution": c.get("contribution")}
                            for c in lst[:6]]
        return {"for": round(for_mass, 5), "against": round(against, 5),
                "supporters": pick(supporters), "opponents": pick(opponents),
                "subsystem_votes": subsystems}

    # ==================================================================
    # Staged ANALYSE flow: formulas → emotions → physics → news → hedge →
    # AI review (Gemini + local qwen2.5-coder) → final box (signal, 4 TPs,
    # SL). Each stage carries its own timestamps — the "calculation
    # period" the user sees from the moment ANALYSE is pressed. Read-only:
    # locks are NEVER mutated by an analysis (they change only at the
    # window boundary).
    # ==================================================================
    ANALYSE_STAGES = ("formulas", "emotions", "physics", "news", "hedge",
                      "ai_review", "final")

    def start_analyse(self, symbol: str, tf: str = "tf1") -> dict:
        if self.analyse_job and self.analyse_job.get("status") == "running":
            return {"ok": False, "error": "an analysis is already running"}
        if symbol not in SYMBOL_ORDER:
            return {"ok": False, "error": "unknown symbol"}
        self.analyse_job = {
            "symbol": symbol, "tf": tf,
            "started_at": time.time(), "stages": [],
            "status": "running", "result": None,
        }
        self._analyse_task = asyncio.create_task(
            self._run_analyse(self.analyse_job))
        return {"ok": True, "symbol": symbol, "tf": tf,
                "started_at": self.analyse_job["started_at"],
                "stages": list(self.ANALYSE_STAGES)}

    async def _emit_stage(self, job: dict, stage: str, status: str,
                          t0: float, t1: float, summary: str,
                          data: dict | None = None) -> None:
        entry = {"stage": stage, "status": status,
                 "t_start_ms": int(t0 * 1000), "t_end_ms": int(t1 * 1000),
                 "duration_ms": round((t1 - t0) * 1000, 1),
                 "summary": summary}
        if data is not None:
            entry["data"] = data
        job["stages"].append(entry)
        await self.emit({"type": "analyse_stage", "symbol": job["symbol"],
                         "stage": entry})

    async def _run_analyse(self, job: dict) -> None:
        symbol, tf = job["symbol"], job["tf"]
        t_job0 = time.time()
        await self.emit({"type": "analyse_started", "symbol": symbol,
                         "tf": tf, "started_at": job["started_at"],
                         "stages": list(self.ANALYSE_STAGES)})
        store: dict = {}
        try:
            # 1 ── formulas: every value, every timeframe, timed ----------
            t0 = time.time()
            raws, per_tf_ms = {}, {}
            for t in TF_ORDER:
                ctx = self.build_ctx(symbol, t)
                anoms: list = []
                f0 = time.perf_counter()
                raws[t] = compute_all(ctx, record=False, anomalies=anoms)
                per_tf_ms[t] = round((time.perf_counter() - f0) * 1000, 1)
            store["raws"] = raws
            nz = sum(1 for t in TF_ORDER for v in raws[t].values()
                     if abs(v) > 1e-9)
            total = sum(len(v) for v in raws.values())
            top = sorted(
                ((v, t, fid) for t in TF_ORDER for fid, v in raws[t].items()),
                key=lambda r: -abs(r[0]))[:5]
            await self._emit_stage(
                job, "formulas", "done", t0, time.time(),
                f"{nz}/{total} readings non-zero · top: " + ", ".join(
                    f"F{fid}@{t.upper()}={v:+.3f}" for v, t, fid in top),
                {"nonzero": nz, "total": total, "per_tf_ms": per_tf_ms})
            # 2 ── emotions (brain consensus feeds them) -------------------
            t0 = time.time()
            market = self._build_market(symbol)
            news_agg = self.news.aggregate(symbol)
            em_out = {}
            for t in TF_ORDER:
                ctx = self.build_ctx(symbol, t)
                atr = self._atr(ctx)
                thing2 = self._thing2(symbol, t, ctx, atr, market)
                affinity = {fid: abs(v) for fid, v in raws[t].items()
                            if abs(v) > 0.7}
                br = self.brains.evaluate(symbol, t, market, affinity,
                                          thing2, advance=False)
                em = self.emotions.compute(ctx, market, br, news_agg, None)
                dom = max(em["emotions"], key=em["emotions"].get)
                em_out[t] = {"cerc": round(em["cerc"], 4),
                             "top_emotion": f"E{dom}",
                             "brain_state": br["dominant"],
                             "polarity": br["polarity"]}
            store["emotions"] = em_out
            e1 = em_out[tf]
            await self._emit_stage(
                job, "emotions", "done", t0, time.time(),
                f"CERC {e1['cerc']:+.4f} · top {e1['top_emotion']} · "
                f"brain {e1['brain_state']} ({e1['polarity']:+d})",
                em_out)
            # 3 ── physics (P1..P21 + UFP) --------------------------------
            t0 = time.time()
            long_closes = {sym: list(self.feed.state(sym).closes)
                           for sym in SYMBOL_ORDER}
            news_minutes = self.news.arrival_minutes_last(600)
            phys_out = {}
            for t in TF_ORDER:
                ctx = self.build_ctx(symbol, t)
                w = self.cci.get_weights(symbol, t)
                comp_core = {fid: raws[t].get(fid, 0.0)
                             for fid in FORMULA_FUNCTIONS}
                wsum = sum(w.get(fid, 0.0) for fid in comp_core) or 1.0
                cci_ref = sum(w.get(fid, 0.0) * v
                              for fid, v in comp_core.items()) / wsum

                def _recompute(c):
                    return compute_all(c, record=False)
                ph = self.physics.evaluate(
                    symbol, t, ctx, atr=self._atr(ctx), raws=raws[t],
                    cci_ref=cci_ref, long_closes=long_closes,
                    news_minutes=news_minutes, weights=w,
                    recompute=_recompute, hist=self.hist, record=False)
                mech = ph.get("mechanisms") or []
                phys_out[t] = {
                    "ufp": round(ph["ufp"]["value"], 4),
                    "wfcp": round(ph["wfcp"], 4),
                    "drpt": round(ph["drpt"], 4),
                    "gspi": round(ph["gspi"], 4),
                    "mech_active": sum(1 for m in mech if m["active"]),
                    "mech_total": len(mech),
                }
            store["physics"] = phys_out
            pe = phys_out[tf]
            await self._emit_stage(
                job, "physics", "done", t0, time.time(),
                f"UFP {pe['ufp']:+.4f} · {pe['mech_active']}/{pe['mech_total']} "
                f"mechanisms active · WFCP {pe['wfcp']:.3f} DRPT {pe['drpt']:.3f}",
                phys_out)
            # 4 ── news ----------------------------------------------------
            t0 = time.time()
            self.news.refresh()
            agg = self.news.aggregate(symbol)
            news_out = {"value": agg["value"], "niv": agg["niv"],
                        "count": agg["count"],
                        "dropped": agg.get("relevance_dropped", 0),
                        "critical": agg.get("critical", {}).get("active", False),
                        "impacts": agg["impacts"]}
            store["news"] = news_out
            await self._emit_stage(
                job, "news", "done", t0, time.time(),
                f"NIV {news_out['niv']:+.3f} · {news_out['count']} items · "
                f"{news_out['dropped']} dropped by relevance gate"
                + (" · ⚠ CRITICAL" if news_out["critical"] else ""),
                news_out)
            # 5 ── hedge / dark-pool ---------------------------------------
            t0 = time.time()
            hedge_out = {}
            for t in TF_ORDER:
                ctx = self.build_ctx(symbol, t)
                ifr = hedgefund.institutional_flow(ctx)
                dpad = hedgefund.dark_pool_flow(ctx)
                smdi = hedgefund.smart_money_divergence(
                    ifr["value"], store["emotions"][t]["polarity"])
                hedge_out[t] = {
                    "ifr": round(ifr["value"], 4),
                    "dpad": round(dpad["value"], 4),
                    "smdi": round(smdi, 4),
                    "combined": round(hedgefund.combined_signal(
                        ifr["value"], dpad["value"], smdi), 4),
                }
            store["hedge"] = hedge_out
            he = hedge_out[tf]
            await self._emit_stage(
                job, "hedge", "done", t0, time.time(),
                f"combined {he['combined']:+.4f} · IFR {he['ifr']:+.3f} · "
                f"dark-pool {he['dpad']:+.3f} · smart-money {he['smdi']:+.3f}",
                hedge_out)
            # 6 ── AI review: Gemini + local qwen2.5-coder -----------------
            await self._analyse_ai_review(job, store, tf)
            # 7 ── final: authoritative fusion + box ------------------------
            t0 = time.time()
            res_pack = self.compute_asset(symbol, record=False, display=True)
            res = res_pack.get(tf) or {}
            price = self.feed.prices().get(symbol, 0.0)
            atr = float(res.get("atr") or self._atr(self.build_ctx(symbol, tf)))
            direction = int(res.get("direction") or 0)
            conf = float(res.get("confidence") or 0.0)
            minutes = TIMEFRAMES[tf]["targets"]
            tps = []
            for k, mins in enumerate(minutes, start=1):
                lvl = self._project_level(price, direction or 1, conf, atr, mins)
                tps.append({"k": k, "minutes": mins, "level": round(lvl, 6)})
            cyc = self.cycles.get((symbol, tf))
            sl = None
            if cyc is not None and cyc.sl:
                sl = cyc.sl
            elif direction != 0:
                sl = price - direction * atr * 1.0
            # per-asset weightage — every asset runs its OWN weights
            wts = self.cci.get_weights(symbol, tf)
            top_w = sorted(wts.items(), key=lambda kv: -abs(kv[1]))[:8]
            ai = job.get("ai") or {}
            final = {
                "signal": "BUY" if direction > 0 else ("SELL" if direction < 0 else "FLAT"),
                "direction": direction,
                "confidence": round(conf, 4),
                "issued": bool(res.get("issued")),
                "price": price,
                "atr": round(atr, 6),
                "tps": tps,
                "sl": round(sl, 6) if sl else None,
                "ratio": None,
                "weightage": [{"id": str(k), "w": round(v, 5)}
                              for k, v in top_w],
                "weightage_note": ("per-asset, per-timeframe weights — "
                                   "every asset analyses with its own mix"),
                "ladder": res.get("ladder"),
                "ai": ai,
                "elapsed_ms": round((time.time() - t_job0) * 1000, 1),
                "calculation_period": {
                    "started_at": job["started_at"],
                    "finished_at": time.time(),
                    "stages": [{"stage": e["stage"],
                                "at": e["t_start_ms"],
                                "ms": e["duration_ms"]}
                               for e in job["stages"]],
                },
            }
            job["result"] = final
            job["status"] = "done"
            await self._emit_stage(
                job, "final", "done", t0, time.time(),
                f"{final['signal']} · conf {final['confidence']:.3f} · "
                f"TP1 {tps[0]['level']} · SL {final['sl']} · "
                f"total {final['elapsed_ms']:.0f} ms", final)
            await self.emit({"type": "analyse_done", "symbol": symbol,
                             "tf": tf, "result": final})
        except Exception as exc:
            job["status"] = "error"
            job["error"] = repr(exc)
            log.exception("analysis failed for %s", symbol)
            await self.emit({"type": "analyse_done", "symbol": symbol,
                             "tf": tf, "result": None,
                             "error": repr(exc)})

    async def _analyse_ai_review(self, job: dict, store: dict,
                                 tf: str) -> None:
        """Stage 6: Gemini and the local qwen2.5-coder review the assembled
        analysis for errors/mistakes and propose corrections. Keyless →
        skipped with a printed note (agents are always optional)."""
        t0 = time.time()
        symbol = job["symbol"]
        raws = store.get("raws", {}).get(tf, {})
        top = sorted(raws.items(), key=lambda kv: -abs(kv[1]))[:8]
        payload = {
            "symbol": symbol, "tf": tf,
            "formulas_top": {f"F{k}": round(v, 4) for k, v in top},
            "emotions": store.get("emotions", {}).get(tf),
            "physics": store.get("physics", {}).get(tf),
            "news": store.get("news"),
            "hedge": store.get("hedge", {}).get(tf),
        }
        prompt = (
            'You are reviewing an intraday market analysis BEFORE it is '
            'shown to a trader. Find mistakes or contradictions in this '
            'analysis (sign conflicts, stale/absurd numbers, risk that '
            'does not match volatility) and give your own call. '
            'Reply ONLY with JSON: {"errors": ["..."], "fixes": ["..."], '
            '"direction": 1|-1|0, "confidence": 0.0..1.0, '
            '"reason": "one sentence"}. Data:\n'
            + json.dumps(payload, default=str)[:4000]
        )
        reviews: dict[str, dict] = {}
        # Gemini (real API call through the key ring)
        key = (self.agents.settings.get("gemini_key") or "")
        if key:
            try:
                res = await self.agents._call_provider("gemini", key, prompt)
                if res.get("ok"):
                    reviews["gemini"] = self._parse_ai_review(
                        res.get("insight") or res.get("text") or "", res)
                else:
                    reviews["gemini"] = {"ok": False,
                                         "error": str(res.get("reason"))}
            except Exception as e:
                reviews["gemini"] = {"ok": False, "error": repr(e)}
        # local qwen2.5-coder (the engine's own reviewer)
        if getattr(self.agents.local, "available", False):
            try:
                text = await asyncio.get_event_loop().run_in_executor(
                    None, self.agents.local.query, prompt)
                reviews["qwen2.5-coder"] = self._parse_ai_review(text or "", {})
            except Exception as e:
                reviews["qwen2.5-coder"] = {"ok": False, "error": repr(e)}
        t1 = time.time()
        if not reviews:
            await self._emit_stage(
                job, "ai_review", "skipped", t0, t1,
                "no AI configured — deterministic pipeline only "
                "(keyless operation, agents optional)", {})
            job["ai"] = {"available": False,
                         "note": "keyless — brain + formulas + physics only"}
            return
        # an AI correction only counts at confidence ≥ 0.70 (printed, never
        # silent — the deterministic ladder stays visible underneath)
        correction = None
        for name, rv in reviews.items():
            if rv.get("ok") and abs(int(rv.get("direction") or 0)) == 1                     and float(rv.get("confidence") or 0) >= 0.70:
                correction = {"by": name,
                              "direction": int(rv["direction"]),
                              "confidence": float(rv["confidence"]),
                              "reason": rv.get("reason", "")}
                break
        job["ai"] = {"available": True, "reviews": reviews,
                     "correction": correction}
        bits = []
        for name, rv in reviews.items():
            if rv.get("ok"):
                n_err = len(rv.get("errors") or [])
                bits.append(f"{name}: {n_err} issue(s), "
                            f"{'CORRECTS to ' + ('BUY' if rv['direction'] > 0 else 'SELL') if rv is correction else 'concurs'}")
            else:
                bits.append(f"{name}: {rv.get('error', 'failed')}")
        await self._emit_stage(
            job, "ai_review", "done", t0, t1, " · ".join(bits),
            job["ai"])

    @staticmethod
    def _parse_ai_review(text: str, fallback: dict) -> dict:
        out = {"ok": True, "errors": [], "fixes": [],
               "direction": int(fallback.get("direction") or 0),
               "confidence": float(fallback.get("confidence") or 0.0),
               "reason": str(fallback.get("insight") or "")[:300],
               "raw": str(text)[:600]}
        try:
            m = text[text.find("{"):text.rfind("}") + 1]
            data = json.loads(m)
            out["errors"] = [str(x) for x in (data.get("errors") or [])][:6]
            out["fixes"] = [str(x) for x in (data.get("fixes") or [])][:6]
            out["direction"] = int(data.get("direction") or 0)
            out["confidence"] = float(data.get("confidence") or 0.0)
            out["reason"] = str(data.get("reason") or out["reason"])[:300]
        except Exception:
            pass
        if not out["direction"]:
            out["direction"] = int(fallback.get("direction") or 0)
        if not out["confidence"]:
            out["confidence"] = float(fallback.get("confidence") or 0.0)
        return out

    def ops_status(self) -> dict:
        """§12 ops row: degradation, vault last write, watchdog, uptime,
        build id vs remote."""
        if not self.build_id:
            try:
                import subprocess
                self.build_id = subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL, timeout=2).decode().strip()
            except Exception:
                self.build_id = "unknown"
        wd = getattr(self, "_watchdog", {}) or {}
        return {
            "degrade": self.degrade_status(),
            "vault_last_write_ago_s": (round(time.time() - self.vault_last_write, 1)
                                       if self.vault_last_write else None),
            "watchdog": {"alert": bool(wd.get("alert")),
                         "seconds_since_window": wd.get("seconds_since_window")},
            "uptime_s": round(time.time() - self.started_at, 1),
            "build_id": self.build_id,
        }

    def degrade_status(self) -> dict:
        """§9.4 honest degradation levels: every level with its reasons and
        which endpoint still serves what. L5 is explicit shutdown only."""
        reasons: list[str] = []
        level = 0
        if self.phase != "live":
            level = max(level, 3)
            reasons.append(f"analysis warmup (phase={self.phase}) — "
                           "formulas/physics/weights still warming")
        feed = self.feed.stats()
        if feed.get("simulated") or feed.get("label") == "SIMULATED TAPE":
            level = max(level, 1)
            reasons.append("live feed unavailable — simulated tape, "
                           "predictions are NOT market-real")
        if getattr(self, "_watchdog", {}).get("alert"):
            level = max(level, 4)
            reasons.append("watchdog: stalled window detected")
        subs = [k for k, v in self.readiness.items() if not v]
        if subs:
            level = max(level, 2)
            reasons.append(f"subsystems not ready: {', '.join(subs)}")
        if getattr(self, "_manual_stop", False):
            level = 5
            reasons.append("explicit manual shutdown (P0-P21 frozen)")
        levels = {
            0: "full — live feed, all subsystems warm",
            1: "feed degraded — simulated tape, UI flagged",
            2: "subsystem degraded — that path idle, core keeps running",
            3: "analysis warmup — no locks claimed from frozen values",
            4: "watchdog alert — stalled window shown on panel",
            5: "manual shutdown — P0-P21 frozen until restart",
        }
        return {"level": level, "name": levels[level], "reasons": reasons,
                "all_levels": levels,
                "endpoints": {
                    "health": "/api/health (always served, never an "
                              "empty body)",
                    "placeholder": "/placeholder (never-ready page)",
                    "watchdog": "/api/watchdog",
                    "secrets": "/api/secrets/status (no secrets in files)",
                }}

    def _learners(self, symbol: str, tf: str, ctx) -> dict:
        """§8.6: edge guard (flat windows), temporal twin (segmented vs
        single-leg variance), BOCPD run-length clock (hazard 1/14), Hawkes
        boundary intensity of lock flips/vetoes."""
        P = list(ctx.P)
        out: dict[str, Any] = {}
        # edge guard: price sitting at one point
        if len(P) >= 10:
            rng = (max(P[-10:]) - min(P[-10:])) / (abs(P[-1]) + 1e-12)
            out["edge_guard"] = {
                "flat": bool(rng < 0.0004),
                "range_bps": round(rng * 1e4, 2),
                "reason": ("flat window — price sitting at one point; "
                           "the contest is a coin flip with costs"
                           if rng < 0.0004 else
                           f"range {rng * 1e4:.1f} bps over last 10 bars"),
            }
        # temporal twin: one W-leg vs split A+B with the covariance term
        if len(P) >= 12:
            r = [(P[i] - P[i - 1]) / P[i - 1] for i in range(1, len(P))]
            half = max(3, len(r) // 2)
            ra, rb = r[:half], r[half:]
            var_a = float(np_var(ra))
            var_b = float(np_var(rb))
            sa, sb = var_a ** 0.5, var_b ** 0.5
            rho = _pearson(ra, rb) if len(ra) == len(rb) and len(ra) > 2 else 0.0
            var_seg = var_a + var_b + 2 * rho * sa * sb
            var_whole = float(np_var(r))
            out["temporal_twin"] = {
                "var_single_leg": round(var_whole, 10),
                "var_segmented": round(var_seg, 10),
                "rho": round(rho, 4),
                "decay_spread": round(var_whole - var_seg, 10),
                "note": ("var_W = var_A + var_B + 2ρσ_Aσ_B — the spread is "
                         "the volatility-decay signal; only the real "
                         "contest leg ever settles"),
            }
        # BOCPD run-length clock, hazard 1/14 windows
        ring = list(self.hist.ring(self.hist.tf_key(symbol, tf, "bocpd_r"), 64))
        last_move = (P[-1] - P[-2]) / P[-2] if len(P) >= 2 and P[-2] else 0.0
        # hazard h = 1/14: posterior run length updates on surprise
        h = 1.0 / 14.0
        surprise = abs(last_move) / (float(np_var([ (P[i]-P[i-1])/P[i-1] for i in range(1, len(P)) ])) ** 0.5 + 1e-12) if len(P) > 3 else 0.0
        rl = (ring[-1] if ring else 0.0)
        if surprise > 2.5:
            rl = 0.0                      # change point
        else:
            rl = min(64.0, rl + 1.0)
        r2 = self.hist.ring(self.hist.tf_key(symbol, tf, "bocpd_r"), 64)
        r2.append(rl)
        out["bocpd"] = {"run_length_windows": int(rl),
                        "hazard": round(h, 4),
                        "surprise_z": round(surprise, 3),
                        "note": "hazard 1/14 — a jump > 2.5σ resets the clock"}
        # Hawkes boundary intensity of recent lock flips/vetoes
        # (in-memory event list — the hist Ring only appends)
        evs = getattr(self, "_hawkes_events", None)
        if evs is None:
            evs = self._hawkes_events = {}
        key = (symbol, tf)
        now = time.time()
        evs.setdefault(key, []).append(now)
        evs[key] = [t for t in evs[key] if now - t < 3600][-32:]
        tau = 600.0
        lam = sum(math.exp(-(now - t) / tau) for t in evs[key]) / tau
        out["hawkes"] = {"intensity": round(lam, 6),
                         "events_1h": len(evs[key]),
                         "note": "Σ exp(−Δt/τ)/τ over boundary events, τ=600s"}
        return out

    def formula_traces_payload(self, symbol: str, tf: str) -> dict:
        """§2 traces on demand — kept OUT of the broadcast snapshot (the
        600KB payload was re-parsed on every refresh and lagged the UI)."""
        t = self.last_traces.get((symbol, tf)) or {}
        return {str(k): v for k, v in t.items()}

    def _flat_payload(self, cap: int = 8) -> dict:
        """§2.9 flat watch: share of the last 50 locks each formula spent
        inside ±0.02, with the zero reason when it had one. Offenders only."""
        out: dict[str, list] = {}
        for (symbol, tf, fid), dq in self.flat.items():
            if not dq:
                continue
            share = sum(dq) / len(dq)
            if share < 0.5:
                continue
            tr = (self.last_traces.get((symbol, tf)) or {}).get(fid) or {}
            out.setdefault(f"{symbol}|{tf}", []).append({
                "fid": fid, "share": round(share, 2), "n": len(dq),
                "zero_reason": tr.get("zero_reason"),
            })
        for key in out:
            out[key].sort(key=lambda r: -r["share"])
            out[key] = out[key][:cap]
        return out

    def formula_history(self, symbol: str, tf: str) -> dict:
        """Sparklines (100 GRCs) + rolling accuracy (50 GRCs) for every
        core formula, emotion and physics formula of one (asset, timeframe)."""
        from .config import PHYSICS_NAMES as _P_NAMES

        def r(name: str, n: int) -> list[float]:
            return [round(x, 5) for x in
                    self.hist.ring(self.hist.tf_key(symbol, tf, name), n)]

        core_ids = list(range(1, 32))              # F1..F31 panels (31)
        core = {str(f): r(f"fspark{f}", 100) for f in core_ids}
        core_acc = {str(f): r(f"facc{f}", 50)
                    for f in list(FORMULA_FUNCTIONS) + [20, 28, 30]}
        emotion = {str(e): r(f"emospark{e}", 100) for e in range(1, 23)}
        emotion_acc = {str(e): r(f"eacc{e}", 50) for e in range(1, 23)}
        physics = {str(p): r(f"pspark{p}", 100) for p in range(1, 23)}
        physics_acc = {str(p): r(f"pacc{p}", 50) for p in range(1, 23)}
        return {
            "symbol": symbol, "tf": tf,
            "core": core, "core_accuracy": core_acc,
            "emotion": emotion, "emotion_accuracy": emotion_acc,
            "physics": physics, "physics_accuracy": physics_acc,
            "names": {
                "core": {str(f): FORMULA_NAMES[f] for f in core_ids},
                "physics": {str(p): _P_NAMES[p] for p in range(1, 23)},
            },
            "weights": {str(fid): round(self.cci.get_weights(symbol, tf)
                                        .get(fid, 0.0), 5)
                        for fid in WEIGHT_IDS},
            "alpha": ({p: round(w, 5) for p, w in
                       self.physics._weights(symbol, tf).items()}
                      if self.physics.ready else {}),
            "alpha_accuracy": ({str(p): round(v, 4) for p, v in
                                self.physics._accuracy(symbol, tf).items()}
                               if self.physics.ready else {}),
        }

    def brain_wiring(self, symbol: str = "BTC", tf: str = "tf1") -> dict:
        """§3.3 wiring table + polarity/odd-symmetry probes on live inputs."""
        ctx = self.build_ctx(symbol, tf)
        market = self._build_market(symbol)
        thing2 = self._thing2(symbol, tf, ctx, self._atr(ctx), market)
        return self.brains.wiring(symbol, tf, market, thing2)

    def brain_geometry(self) -> dict:
        """Static connectome geometry for the 3D brain visualization."""
        return self.brains.geometry()

    def brain_activations(self, symbol: str, tf: str) -> dict:
        """Per-instance unit activations (uint8 base64) for one pair."""
        return self.brains.activations(symbol, tf)

    # ==================================================================
    # cycle helpers
    # ==================================================================
    def _weight_profile(self, symbol: str, tf: str, res: dict) -> dict:
        """Section 14 Level 5 — sub-target relevance profiles (Tp1 fast:
        SATD/VAD/WRPP; Tp3: MSRC/DIQ; Tp4 slow: CLRE/DRHE/UFP)."""
        raw = res.get("raw") or {}
        return {
            "l3_tf": round(self.learning.performance_multiplier(21, "tf", tf), 4),
            "l4_sym": round(self.learning.performance_multiplier(21, "sym", symbol), 4),
            "l5": {
                "tp1": {"SATD": round(raw.get(1, 0.0), 4),      # F1 SATE
                        "VAD": round(raw.get(2, 0.0), 4),       # F2
                        "WRPP": round(raw.get(3, 0.0), 4)},     # F3
                "tp3": {"MSRC": round(raw.get(11, 0.0), 4),     # F11
                        "DIQ": round(raw.get(15, 0.0), 4)},     # F15
                "tp4": {"CLRE": round(raw.get(19, 0.0), 4),     # F19
                        "DRHE": round(res.get("drhe", 0.0), 4),
                        "UFP": round(res.get("physics", {}).get("ufp", 0.0), 4)},
            },
        }

    async def _apply_agents(self, symbol: str, res_pack: dict[str, dict]) -> None:
        """Section 11: one Gemini call per GRC (+ local model), then the
        CCI multiplier, then the Section-16 mandatory double-check."""
        if symbol != SYMBOL_ORDER[0]:
            # one agent call per GRC (BTC anchor) — the signal is shared
            # across assets with each asset's own CCI/confidence multipliers
            sig = self.last_agent_signal or {}
        else:
            tf1 = res_pack.get("tf1") or {}
            prices = self.feed.prices()
            returns = {}
            for s2 in SYMBOL_ORDER:
                st = self.feed.state(s2)
                cl = st.closes
                if len(cl) >= 11 and cl[-11]:
                    returns[s2] = round(cl[-1] / cl[-11] - 1.0, 6)
            w = self.cci.get_weights(symbol, "tf1")
            top5 = sorted(
                ((fid, v) for fid, v in w.items() if fid <= 31),
                key=lambda kv: -abs(kv[1]))[:5]
            ctx = {
                "prices": prices,
                "returns": returns,
                "cci_dir": tf1.get("direction"),
                "cci_conf": round(tf1.get("confidence", 0.0), 4),
                "top5": [{"formula": int(f), "weight": round(v, 5),
                          "signal": round((tf1.get("adj") or {}).get(int(f), 0.0), 4)}
                         for f, v in top5],
                "brain_consensus": tf1.get("brain", {}).get("polarity"),
                "headline": ((self.news._live_items() or [{}])[0]
                             .get("headline", "")),
                "news_sentiment": round(self.news.aggregate(symbol).get("polarity", 0.0), 4),
                "cerc": tf1.get("emotion", {}).get("cerc"),
            }
            # §11.2: one call per (asset, window) — never two calls for
            # one lock — and a hard deadline BEFORE the boundary; a late
            # answer is discarded and logged, never applied mid-window (I-3)
            _start_w, _end_w = self._grid_window("tf1", time.time())
            _cache_key = (symbol, _end_w)
            cached = self._agent_cache.get("sig") if \
                self._agent_cache.get("key") == _cache_key else None
            if cached is not None:
                sig = dict(cached)
                sig["cached"] = True
            else:
                t_call = time.monotonic()
                try:
                    sig = await asyncio.wait_for(
                        self.agents.query(ctx), timeout=AGENT_DEADLINE_S)
                except asyncio.TimeoutError:
                    late_by = round(time.monotonic() - t_call, 1)
                    self.agent_deadline_misses += 1
                    log.warning("agent late by %.1fs — discarded (deadline %ss)",
                                late_by, AGENT_DEADLINE_S)
                    sig = {"direction": 0, "confidence": 0.0, "weight": 0.0,
                           "source": "none",
                           "insight": "",
                           "warning": f"late by {late_by}s — discarded, "
                                      f"deadline {AGENT_DEADLINE_S}s"}
                except Exception:
                    log.exception("agent query failed")
                    sig = {"direction": 0, "confidence": 0.0, "weight": 0.0,
                           "source": "error", "insight": ""}
                if sig.get("source") not in ("error", "none"):
                    self._agent_cache = {"key": _cache_key, "sig": dict(sig)}
            self.last_agent_signal = sig
        gd = float(sig.get("confidence") or 0.0)
        g_dir = int(sig.get("direction") or 0)
        gw = float(sig.get("weight") or 0.0)
        for tf in TF_ORDER:
            res = res_pack.get(tf)
            if not res:
                continue
            # Section 11: CCI × (1 + w_agent × gd × gc)
            if gw > 0 and gd > 0:
                factor = 1.0 + gw * gd * res.get("base_confidence", 0.0)
                res["cci_adjusted"] = res["cci_adjusted"] * factor
                _lad2, _src2, _fl2 = self._direction_ladder(
                    symbol, tf, float(res["cci_adjusted"]),
                    (res.get("breakdown") or {}).get("components") or [],
                    res.get("brain") or {},
                    self.prev_cci_dir.get((symbol, tf), 0))
                res["direction"] = _lad2
                res["ladder"] = {"source": _src2, "noise_floor": round(_fl2, 5),
                                 "score": round(float(res["cci_adjusted"]), 5)}
            res["agent"] = {
                "direction": g_dir, "confidence": round(gd, 4),
                "weight": round(gw, 5), "source": sig.get("source", "none"),
                "insight": sig.get("insight", ""),
                "warning": sig.get("warning", ""),
            }
            # Section 16: mandatory 4-vote double-check
            votes = {
                "pcv": 1 if res.get("pcv", 0.0) > 0 else (
                    -1 if res.get("pcv", 0.0) < 0 else 0),
                "agent": g_dir,
                "brain": int(res.get("brain", {}).get("polarity", 0)),
            }
            dc = self.cci.double_check(res["direction"], res["confidence"], votes)
            if dc["label"] == "mixed":
                dc["message"] = "MIXED SIGNALS"
            res["double_check"] = dc
            if not dc["issued"]:
                res["issued"] = False
            res["confidence"] = dc["confidence"]
            if dc["label"] == "overridden":
                res["direction"] = dc["direction"]
            self.last_result[(symbol, tf)] = res
            # keep the snapshot display in sync with the post-agent values
            disp = self.display_systems.get((symbol, tf))
            if disp is not None:
                disp["agent"] = res.get("agent")
                if res.get("ladder"):
                    disp["ladder"] = dict(res["ladder"])
                disp["lock"] = dict(self.locks.get((symbol, tf)) or {})
                disp["double_check"] = dc
                disp["confidence"] = round(res.get("confidence", 0.0), 4)
                disp["cci_adjusted"] = res.get("cci_adjusted")
                disp["direction"] = res.get("direction")
                disp["issued"] = bool(res.get("issued"))
        # ---- Groq qualitative synthesis (Prompt 2) ----
        try:
            await self._apply_groq_synthesis(symbol, res_pack)
        except Exception:
            log.exception("groq synthesis failed")
        # ---- tape veto + lock sync (advice 8.1) ----
        self._apply_tape_veto(symbol, res_pack)
        lock = self.locks.get((symbol, "tf1"))
        if lock is not None:
            lock["side"] = int((res_pack.get("tf1") or {}).get("direction") or 0)
            lock["confidence"] = round(float((res_pack.get("tf1") or {}).get("confidence") or 0), 4)
            lock["reason"] = str((res_pack.get("tf1") or {}).get("reason") or "")[:500]
            self._save_locks()

    def _apply_tape_veto(self, symbol: str, res_pack: dict[str, dict]) -> None:
        """Last 2-3 bars contradicting the side flatten a low-conviction
        call (advice 8.1). Printed as tape_veto in the reason."""
        price = self.feed.prices().get(symbol, 0.0)
        if price <= 0:
            return
        for tf in TF_ORDER:
            res = res_pack.get(tf)
            if not res or not res.get("issued") or not res.get("direction"):
                continue
            closes = self.feed.state(symbol).closes[-3:]
            if len(closes) < 3:
                continue
            move = closes[-1] - closes[0]
            threshold = price * 0.0004          # ~4 bps over 3 bars
            opposes = (move > threshold and res["direction"] < 0) or \
                      (move < -threshold and res["direction"] > 0)
            if opposes and float(res.get("confidence") or 0) < 0.62:
                res["direction"] = 0
                res["issued"] = False
                res["tape_veto"] = True
                res["reason"] = (f"TAPE VETO: last 3 bars moved "
                                 f"{move / price * 1e4:+.1f} bps against the call "
                                 f"and confidence < 0.62 — flattened. " +
                                 str(res.get("reason") or ""))[:400]
                disp = self.display_systems.get((symbol, tf))
                if disp is not None:
                    disp["direction"] = 0
                    disp["issued"] = False
                    disp["reason"] = res["reason"]

    # cadence per advice §8.3: 4-hourly for 22/44 min, daily for 130/247
    _HINDSIGHT_CADENCE = {"tf1": 4 * 3600, "tf2": 4 * 3600,
                          "tf3": 24 * 3600, "tf4": 24 * 3600}
    HINDSIGHT_MIN_DECIDED = 8          # advice §8.3
    HINDSIGHT_INVERT_Z = -3.0          # invert only at z ≤ −3 …
    HINDSIGHT_INVERT_MIN_N = 16        # … with ≥ 16 decided windows

    @staticmethod
    def _grade_ring(ring: list[float]) -> dict | None:
        """Hits/misses/silent, bias, z of the signed edge — strict sign."""
        decided = [v for v in ring if v != 0]
        silent = len(ring) - len(decided)
        if not decided:
            return None
        hits = sum(1 for v in decided if v > 0)
        misses = len(decided) - hits
        n = len(decided)
        mean = sum(decided) / n
        var = sum((v - mean) ** 2 for v in decided) / max(1, n - 1)
        std = var ** 0.5
        z = (mean / (std / (n ** 0.5))) if std > 1e-9 else 0.0
        mult = max(0.25, min(1.5, 1.0 + 0.25 * z))
        inverted = z <= -3.0 and n >= 16
        if inverted:
            mult = -max(0.25, min(1.5, 1.0 + 0.25 * abs(z) * 0.5))
        return {"n": n, "hits": hits, "misses": misses, "silent": silent,
                "rate": round(hits / n, 3), "bias": round(mean, 4),
                "z": round(z, 3), "multiplier": round(mult, 3),
                "inverted": inverted}

    def hindsight(self) -> dict:
        """§8.3 re-grade of every formula AND system on the recent windows:
        hits/misses/silent, won rate, bias, z of the signed edge, multiplier
        clip(1 + 0.25·z, 0.25, 1.5); inversion only at z ≤ −3 with ≥ 16
        decided; MIN_DECIDED = 8 — below it the card says 'not enough
        windows', never a percentage (advice §0.5)."""
        per: dict[str, dict] = {}
        for symbol in SYMBOL_ORDER:
            for tf in TF_ORDER:
                for fid in list(WEIGHT_IDS) + ["agent"]:
                    key_name = (f"facc{fid}" if fid != "agent"
                                else "facc_agent")
                    key = self.hist.tf_key(symbol, tf, key_name)
                    ring = list(self.hist.ring(key, 50))
                    label = f"F{fid}" if fid != "agent" else "agent"
                    if len(ring) < self.HINDSIGHT_MIN_DECIDED:
                        entry = {"label": label, "n": len(ring),
                                 "enough": False,
                                 "note": "not enough windows to say "
                                         f"(need {self.HINDSIGHT_MIN_DECIDED})"}
                    else:
                        g = self._grade_ring(ring)
                        if not g:
                            continue
                        entry = {"label": label, "enough": True, **g}
                        entry["changes_note"] = (
                            f"damped {label} ×{g['multiplier']}: "
                            f"{g['hits']}/{g['n']}"
                            + (" · INVERTED" if g["inverted"] else ""))
                    k = f"{symbol}|{tf}|{label}"
                    prev = per.get(k)
                    if prev is None or entry.get("n", 0) >= prev.get("n", 0):
                        per[k] = entry
        top = sorted((e for e in per.values() if e.get("enough")),
                     key=lambda d: d["multiplier"])[:6]
        return {
            "per": per,
            "most_damped": top,
            "min_decided": self.HINDSIGHT_MIN_DECIDED,
            "cadence": {k: f"{v // 3600}h" for k, v in self._HINDSIGHT_CADENCE.items()},
            "last_review": {k: v for k, v in self._hindsight_last.items()},
            "changes": self.hindsight_changes[-8:],
            "rule": ("mult = clip(1 + 0.25·z, 0.25, 1.5) on the strict-sign "
                     "edge; invert only at z ≤ −3 with ≥16 decided; "
                     "MIN_DECIDED = 8; changes apply at the next boundary"),
        }

    def _maybe_review_hindsight(self, symbol: str, tf: str, now: float) -> None:
        """Called at every window boundary: re-grade if the cadence is due,
        publish the multiplier changes, apply at THIS boundary (advice §8.3:
        changes apply at the next boundary and the card prints them)."""
        period = self._HINDSIGHT_CADENCE.get(tf, 24 * 3600)
        last = self._hindsight_last.get(tf, 0.0)
        snap = self.hindsight()                 # cheap: reads rings only
        enough_n = sum(1 for e in snap["per"].values() if e.get("enough"))
        reviewed_n = self._hindsight_reviewed_n.get(tf, 0)
        # due by cadence, OR enough newly-decided entries that waiting out
        # the cadence would leave the card stale (cold-start refill)
        if last and now - last < period and enough_n <= reviewed_n + 4:
            return
        self._hindsight_last[tf] = now
        self._hindsight_reviewed_n[tf] = enough_n
        for key, entry in snap["per"].items():
            if not entry.get("enough"):
                continue
            sym, t, label = key.rsplit("|", 2)
            if label == "agent":
                continue                     # agents have no weight slot
            try:
                fid = int(label[1:])
            except (ValueError, TypeError):
                continue
            old_m = self.hindsight_mult.get((sym, t, fid), 1.0)
            new_m = float(entry["multiplier"])
            if abs(old_m - new_m) > 1e-6:
                self.hindsight_changes.append({
                    "at": now, "symbol": sym, "tf": t, "formula": label,
                    "from": round(old_m, 3), "to": round(new_m, 3),
                    "reason": entry.get("changes_note", ""),
                })
            self.hindsight_mult[(sym, t, fid)] = new_m
        self.hindsight_changes = self.hindsight_changes[-40:]

    def _hindsight_factor(self, symbol: str, tf: str, fid: int) -> float:
        return self.hindsight_mult.get((symbol, tf, fid), 1.0)

    async def _apply_groq_synthesis(self, symbol: str,
                                   res_pack: dict[str, dict]) -> None:
        """Regime classification + bounded weight adjustments via Groq.

        Runs ONLY when a GROQ_API_KEY is configured, and only when the
        double-check came back mixed OR every 3rd GRC — keeps well under
        the free-tier 30 RPM cap. Pre-aggregates the formula summary
        (compact JSON) instead of shipping raw series.
        """
        ag = self.agents
        if not ag.groq_key:
            return
        mixed = any((res_pack.get(tf) or {}).get("double_check", {})
                    .get("label") == "mixed" for tf in TF_ORDER)
        self._groq_n = getattr(self, "_groq_n", 0) + 1
        if not mixed and self._groq_n % 3 != 0:
            return
        tf1 = res_pack.get("tf1") or {}
        bd = tf1.get("breakdown") or {}
        top = sorted((bd.get("components") or []),
                     key=lambda c: -abs(c.get("contribution", 0)))[:10]
        compact = {
            "symbol": symbol,
            "direction": tf1.get("direction"),
            "confidence": round(float(tf1.get("confidence") or 0), 3),
            "cci": round(float(tf1.get("cci_adjusted") or 0), 4),
            "top_components": [{"id": c.get("id"), "contrib": round(c.get("contribution", 0), 5)}
                               for c in top],
            "multipliers": bd.get("multipliers"),
            "cerc": (tf1.get("emotion") or {}).get("cerc"),
            "ufp": (tf1.get("physics") or {}).get("ufp"),
            "brain_polarity": (tf1.get("brain") or {}).get("polarity"),
            "news": tf1.get("news_value"),
            "double_check": [(tf, (res_pack.get(tf) or {}).get("double_check", {}).get("label"))
                             for tf in TF_ORDER],
        }
        syn = await ag.synthesize(compact)
        for tf in TF_ORDER:
            res = res_pack.get(tf)
            if res is None:
                continue
            res["synthesis"] = syn
            disp = self.display_systems.get((symbol, tf))
            if disp is not None:
                disp["synthesis"] = syn
            # bounded application: only high-conviction committee calls
            if (syn.get("ok") and syn.get("confidence_score", 0) >= 0.6
                    and syn.get("weight_adjustments")):
                w = self.cci.get_weights(symbol, tf)
                for k, mult in syn["weight_adjustments"].items():
                    try:
                        fid = int(k)
                    except (TypeError, ValueError):
                        continue
                    if fid in w:
                        w[fid] = max(1e-3, min(0.5, w[fid] * float(mult)))

    def _record_learning(self, symbol: str, res_pack: dict[str, dict],
                         now: float) -> None:
        """Section 10/13 bookkeeping after the lock rules ran."""
        prices = self.feed.prices()
        price = prices.get(symbol, 0.0)
        record = empty_record(now)
        record["prices"] = {s: round(p, 6) for s, p in prices.items()}
        prev_price = self.cci.prev_price.get(symbol)
        move_sign = _sign(price - prev_price) if prev_price else 0
        for tf in TF_ORDER:
            res = res_pack.get(tf)
            if not res:
                continue
            cyc = self.cycles.get((symbol, tf))
            evs = [e for e in (cyc.events if cyc else [])
                   if e.get("ts", 0) > now - 260]
            target_hit = any(e["type"] == "target_hit" for e in evs)
            sl_hit = any(e["type"] == "sl_hit" for e in evs)
            sub = {
                "emotion": 1 if res.get("emotion", {}).get("aggregate", 0) > 0 else -1,
                "physics": 1 if res.get("physics", {}).get("ufp", 0) > 0 else -1,
                "news": 1 if res.get("news_value", 0) > 0 else -1,
                "hf": 1 if res.get("hedge", {}).get("combined", 0) > 0 else -1,
                "brain": int(res.get("brain", {}).get("polarity", 0)),
                "core": int(res.get("direction", 0)),
                "final": int(res.get("direction", 0)),
            }
            signals = {int(f): _sign(v)
                       for f, v in (res.get("adj") or {}).items()}
            brain_sigs = [float(inst.get("h_signal", 0.0))
                          for inst in res.get("brain", {}).get("instances", [])]
            self.learning.record_micro(
                symbol, tf, move_sign, signals, sub, brain_sigs,
                1 if target_hit else None,
                1 if sl_hit else None,
            )
            if res.get("direction") and prev_price:
                self.learning.feed_return(symbol, tf, price / prev_price - 1.0)
            # --- Section 13 per-GRC record ------------------------------
            record["predictions"][tf] = {
                "direction": res.get("direction"),
                "confidence": round(res.get("confidence", 0.0), 5),
                "tp_sl": {
                    "tp": (cyc.tp if cyc else None),
                    "sl": (cyc.sl if cyc else None),
                },
                "lock_status": ({
                    str(getattr(t, "k", i)): getattr(t, "status", "idle")
                    for i, t in enumerate(getattr(cyc, "targets", []) or [])
                } if cyc else {}),
                "issued": bool(cyc.issued) if cyc else False,
            }
            record["outcomes"][tf] = {
                "target_hit": target_hit,
                "sl_hit": sl_hit,
                "pnl_pct": (round((price / cyc.tp - 1.0) * 100, 4)
                            if target_hit and cyc and cyc.direction > 0 and cyc.tp
                            else 0.0),
            }
            record["formula_outputs"][tf] = {
                str(f): round(v, 5) for f, v in (res.get("adj") or {}).items()
            }
            record["brain_states"][tf] = {
                "polarity": res.get("brain", {}).get("polarity"),
                "dominant": res.get("brain", {}).get("dominant"),
                "state_dist": res.get("brain", {}).get("state_dist"),
            }
            record["emotion_composite"] = round(
                (res.get("emotion", {}).get("cerc") or 0.0), 5)
            record["physics_composite"] = round(
                (res.get("physics", {}).get("ufp") or 0.0), 5)
            record["news_impact"] = round(res.get("news_value") or 0.0, 5)
            record["agent_signals"] = {
                "gemini": (res.get("agent") or {}),
                "local": self.agents.status().get("local", {}).get("last") or {},
            }
            w = self.cci.get_weights(symbol, tf)
            record["weights"][tf] = {str(f): round(w.get(f, 0.0), 5)
                                     for f in WEIGHT_IDS}
        try:
            self.historydb.append(record)
        except Exception:
            log.exception("history append failed")

    def _project_level(
        self, price: float, direction: int, confidence: float, atr: float, minutes: int
    ) -> float:
        """Provisional sub-target projection — seam for part 2's exact form."""
        if atr <= 0:
            atr = price * 1e-4
        return price + direction * abs(confidence) * atr * math.sqrt(minutes) * TARGET_SCALE

    def _risk_pack(
        self, symbol: str, tf: str, cyc: TFCycle, res: dict
    ) -> tuple[float | None, float | None, float | None, float | None, float | None]:
        """Formula 28 (ARC) + Formula 29 (DLP) for the active sub-target."""
        price = self.feed.prices().get(symbol, 0.0)
        if not cyc.issued or cyc.active_k is None or price <= 0:
            return None, None, None, None, None
        if cyc.confidence < CONFIDENCE_FLOOR:
            return None, None, None, None, None

        # F28: ratio band from the ARC computed during the pipeline pass
        arc_info = res.get("arc") or {}
        ratio = arc_info.get("ratio")
        if ratio is None:
            # defensive fallback when a hand-built result lacks arc data
            _, ratio = arc_ratio(
                res.get("cci_adjusted", 0.0),
                res.get("sigma_cur", 0.0) or 1.0,
                res.get("sigma_base", 0.0) or 1.0,
                (res.get("brain") or {}).get("strength", 0.0),
                confidence=cyc.confidence,
            )
        if ratio is None:
            return None, None, None, None, None

        # F29: fractal boundaries (shared with F24 FBPS) + seams
        ctx = self.build_ctx(symbol, tf)
        boundaries = fractal_boundary_levels(ctx)
        dlp_details: dict = {}
        tp, sl, tp_dist, sl_dist = dynamic_level_placement(
            price=price,
            direction=cyc.direction,
            ratio=ratio,
            atr=res["atr"],
            boundaries=boundaries,
            drhe=res.get("drhe", 1.0),
            ofti_value=self._build_market(symbol).get("ofti", 0.0),
            tighten=cyc.sl_tighten,
            details=dlp_details,
        )
        # visualization: phi multipliers + F29 sparkline (GRC cadence)
        dsys = self.display_systems.get((symbol, tf))
        if dsys is not None and dlp_details:
            dsys["dlp"] = dict(dlp_details)
            self.hist.ring(
                self.hist.tf_key(symbol, tf, "fspark29"), 100
            ).append(dlp_details.get("phi_tp", 1.0))
        return tp, sl, tp_dist, sl_dist, ratio

    def _apply_risk(self, symbol: str, tf: str, cyc: TFCycle, res: dict) -> None:
        tp, sl, tp_dist, sl_dist, ratio = self._risk_pack(symbol, tf, cyc, res)
        if ratio is None and cyc.issued and cyc.active_k is None:
            return          # all sub-targets completed — keep last placement
        cyc.tp, cyc.sl = tp, sl
        cyc.tp_dist, cyc.sl_dist = tp_dist, sl_dist
        cyc.ratio = ratio

    # ---------------- Signal Lock Protocol + settlement (advice 0.3/0.4)
    def _load_locks(self) -> None:
        try:
            with open("data/locks.json", encoding="utf-8") as fh:
                data = json.load(fh)
            self.locks = {(d["symbol"], d["tf"]): d for d in data.get("locks", [])}
        except (OSError, ValueError, KeyError):
            self.locks = {}

    def _save_locks(self) -> None:
        try:
            import os as _os
            _os.makedirs("data", exist_ok=True)
            with open("data/locks.json", "w", encoding="utf-8") as fh:
                json.dump({"locks": list(self.locks.values())}, fh)
        except OSError:
            pass

    @staticmethod
    def _grid_window(tf: str, now: float) -> tuple[float, float]:
        """Epoch-aligned window (advice 0.1): start = floor(now/W)*W."""
        ws = TIMEFRAMES[tf]["minutes"] * 60
        start = math.floor(now / ws) * ws
        return float(start), float(start + ws)

    def _settle_and_lock(self, symbol: str, tf: str, now: float,
                         res: dict) -> None:
        """Settle the previous window in the ledger, then freeze the new
        signal. Mid-window refashes (same grid window) keep the lock."""
        prev = self.locks.get((symbol, tf))
        start, end = self._grid_window(tf, now)
        if prev is not None:
            same_window = abs(float(prev.get("window_end", 0)) - end) < 1e-6
            if same_window:
                return                      # lock holds for the whole window
            price = self.feed.prices().get(symbol, 0.0)
            self.ledger.settle(symbol, tf,
                               float(prev.get("price") or 0.0),
                               price, int(prev.get("side") or 0),
                               window_end=float(prev.get("window_end") or 0))
            # forward table: settle the PREVIOUS lock's frozen readings
            if prev.get("fv"):
                try:
                    self.forward.record(symbol, tf,
                                        {int(k): float(v)
                                         for k, v in prev["fv"].items()},
                                        float(prev.get("price") or 0.0),
                                        price)
                except Exception:
                    pass
            # §7 genesis: score the admitted set walk-forward on the
            # settled move (never in-sample)
            if prev.get("genesis_values"):
                try:
                    self.genesis.score_walk_forward(
                        prev["genesis_values"],
                        price - float(prev.get("price") or 0.0))
                except Exception:
                    pass
            # meta-combiner + evidence ledger: settle the previous lock's
            # frozen subsystem inputs (advice §8.4/§8.5)
            if prev.get("inputs"):
                try:
                    up = price > float(prev.get("price") or 0.0)
                    self.meta.note_pick(prev["inputs"], up)
                    self.meta.record(symbol, tf, prev["inputs"], up)
                except Exception:
                    pass
        dom = ((res.get("emotion") or {}).get("dominant")
               or (res.get("brain") or {}).get("dominant"))
        # freeze the formula readings with the lock (forward tables +
        # flat watch read these; the lock never repaints them)
        tr = self.last_traces.get((symbol, tf)) or {}
        fv = {str(fid): round(float(t["value"]), 5)
              for fid, t in tr.items() if isinstance(fid, int)}
        inputs = {
            "score": round(float(res.get("cci_adjusted") or 0.0), 5),
            "brain": int(((res.get("brain") or {}).get("polarity")) or 0),
            "physics": round(float(((res.get("physics") or {}).get("ufp"))
                                   if isinstance(res.get("physics"), dict) else 0.0) or 0.0, 5),
            "emotion": round(float((res.get("emotion") or {}).get("aggregate")
                                   if isinstance(res.get("emotion"), dict) else 0.0) or 0.0, 5),
            "news": round(float(res.get("news_value")
                               if "news_value" in res else
                               ((res.get("market") or {}).get("news_value")
                                if isinstance(res.get("market"), dict) else 0.0)) or 0.0, 5),
            "agent": int(((res.get("agent") or {}).get("direction")) or 0),
        }
        for fid, t in tr.items():
            if not isinstance(fid, int) or fid == 20:
                continue                    # 20 is a confidence gauge, not a vote
            dq = self.flat.setdefault((symbol, tf, fid),
                                      __import__("collections").deque(maxlen=50))
            dq.append(bool(abs(float(t.get("value") or 0.0)) <= 0.02))
        self.locks[(symbol, tf)] = {
            "symbol": symbol, "tf": tf,
            "side": int(res.get("direction") or 0),
            "confidence": round(float(res.get("confidence") or 0.0), 4),
            "price": round(self.feed.prices().get(symbol, 0.0), 6),
            "reason": str(res.get("reason") or "")[:500],
            "emotion": str(dom or ""),
            "locked_at": now, "computed_at": now,
            "window_start": start, "window_end": end,
            "relocks": 0, "relock_at": None,
            "fv": fv,
            "inputs": inputs,
            "genesis_values": dict(self._genesis_values.get((symbol, tf)) or {}),
        }
        self._maybe_review_hindsight(symbol, tf, now)   # cadence-gated §8.3
        self._save_locks()

    def fresh_cycle(
        self,
        symbol: str,
        tf: str,
        now: float,
        res: dict,
        reason: str = "grc",
        warning: bool = False,
    ) -> TFCycle:
        W = TIMEFRAMES[tf]["minutes"]
        # completed previous cycle without any SL event -> E2/E9 no-hit entry
        prev_cyc = self.cycles.get((symbol, tf))
        if prev_cyc is not None and prev_cyc.issued:
            if not (prev_cyc.safe_events or prev_cyc.close_alert):
                self.hist.record_no_sl_hit(symbol)
        targets = []
        for k, minutes in enumerate(TIMEFRAMES[tf]["targets"], start=1):
            if k in (1, 4):
                level = self._project_level(
                    res.get("_price", 0.0) or self.feed.prices().get(symbol, 0.0),
                    res["direction"] or 1,
                    res["confidence"],
                    res["atr"],
                    minutes,
                )
                targets.append(
                    TargetState(
                        k=k,
                        minutes=minutes,
                        level=level,
                        status="locked",
                        direction=res["direction"] or 1,
                        locked_at=now,
                        locked_cci=res["cci_adjusted"],
                    )
                )
            else:
                # Tp2/Tp3 are NOT calculated until the previous target is hit
                targets.append(
                    TargetState(k=k, minutes=minutes, level=None, status="pending")
                )
        self._settle_and_lock(symbol, tf, now, res)
        _g_start, _g_end = self._grid_window(tf, now)
        cyc = TFCycle(
            symbol=symbol,
            tf=tf,
            W=W,
            direction=res["direction"],
            confidence=res["confidence"],
            cci=res["cci_adjusted"],
            drhe=res["drhe"],
            issued=res["issued"],
            cycle_start=_g_start,
            cycle_end=_g_end,
            targets=targets,
            active_k=1 if res["issued"] else None,
            warning=warning,
            refreshed_at=now,
            qdpc_tau=res.get("qdpc_tau"),
        )
        cyc.events.append({"ts": now, "type": "cycle_fresh", "reason": reason})
        self.cycles[(symbol, tf)] = cyc
        self._apply_risk(symbol, tf, cyc, res)
        return cyc

    # ==================================================================
    # Initial Analysis Phase
    # ==================================================================
    async def _analysis_loop(self) -> None:
        """Readiness evaluation + prominent countdown broadcast (1 Hz)."""
        last_beat = 0.0
        while not self._stop.is_set() and self.phase == "analysis":
            now = time.time()
            self._update_readiness()
            ready = all(self.readiness.values())
            expired = now >= self.phase_deadline

            if ready or expired:
                self.phase_warning = not ready
                await self._lock_initial(warning=not ready)
                break

            if now - last_beat >= 1.0:
                last_beat = now
                await self.emit({
                    "type": "phase",
                    "phase": "analysis",
                    "remaining_s": max(0, int(self.phase_deadline - now)),
                    "readiness": dict(self.readiness),
                    "warning": False,
                })
            await asyncio.sleep(0.25)

    def _update_readiness(self) -> None:
        bars_ok = all(
            len(self.feed.state(s).closes) >= MIN_BARS_FULL_WINDOW
            for s in SYMBOL_ORDER
        )
        enough_any = all(
            len(self.feed.state(s).closes) >= MIN_BARS_COMPUTE
            for s in SYMBOL_ORDER
        )
        self.readiness["bars"] = bars_ok
        self.readiness["brains"] = self.brains.ready
        self.readiness["emotions"] = self.emotions.ready
        self.readiness["news"] = self.news.ready
        self.readiness["physics"] = self.physics.ready
        if enough_any and not self.readiness["formulas"]:
            # dry-run the entire pipeline so all subsystems report readiness
            try:
                for symbol in SYMBOL_ORDER:
                    self.compute_asset(symbol, record=False, display=False)
                self.readiness["formulas"] = True
                self.readiness["weights"] = True
            except Exception:
                log.exception("analysis dry-run failed")

    async def _lock_initial(self, warning: bool) -> None:
        if self.locked:
            return                      # claim immediately (analysis loop and
                                        # a GRC during phase may race here)
        self.locked = True
        now = time.time()
        for symbol in SYMBOL_ORDER:
            try:
                res_pack = self.compute_asset(symbol, record=True, display=True)
            except Exception:
                log.exception("initial compute failed for %s", symbol)
                res_pack = self.last_result
            for tf in TF_ORDER:
                res = res_pack.get(tf)
                if not res:
                    continue
                self.last_result[(symbol, tf)] = res
                self.fresh_cycle(symbol, tf, now, res, reason="initial_lock", warning=warning)
        self.phase = "live"
        await self.emit({
            "type": "phase",
            "phase": "live",
            "remaining_s": 0,
            "readiness": dict(self.readiness),
            "warning": warning,
            "message": (
                "Initial analysis expired — locked best available prediction"
                if warning else "Initial analysis complete — prediction locked"
            ),
        })
        await self._broadcast_full("initial_lock")

    # ==================================================================
    # GRC — full recalculation at :00 :04 ... :56
    # ==================================================================
    async def on_grc(self, ts: float) -> None:
        async with self._busy:
            now = time.time()
            self.last_grc = ts
            self.grc_count += 1

            if self.phase == "analysis":
                # non-recording pass: the ONE official recording pass for the
                # initial lock happens in _lock_initial (never double-append
                # GRC baselines within a single cycle)
                for symbol in SYMBOL_ORDER:
                    try:
                        res_pack = self.compute_asset(symbol, record=False, display=False)
                        for tf in TF_ORDER:
                            self.last_result[(symbol, tf)] = res_pack[tf]
                    except Exception:
                        log.exception("analysis GRC compute failed")
                self._update_readiness()
                if all(self.readiness.values()):
                    await self._lock_initial(warning=False)
                return

            if not self.locked:
                return

            for symbol in SYMBOL_ORDER:
                try:
                    res_pack = self.compute_asset(symbol, record=True, display=True)
                except Exception:
                    log.exception("GRC compute failed for %s", symbol)
                    continue
                # Section 11 agent signal + multiplier, Section 16 double-check
                try:
                    await self._apply_agents(symbol, res_pack)
                except Exception:
                    log.exception("agent pass failed for %s", symbol)
                for tf in TF_ORDER:
                    res = res_pack[tf]
                    self.last_result[(symbol, tf)] = res
                    self._apply_grc_lock_rules(symbol, tf, now, res)
                # Sections 10/13 learning + history bookkeeping
                try:
                    self._record_learning(symbol, res_pack, now)
                except Exception:
                    log.exception("learning bookkeeping failed for %s", symbol)
            # Section 10 macro (every 15 GRCs) + deep (every 247 min)
            try:
                self.learning.maybe_macro(self.grc_count, self)
                self.learning.maybe_deep(self)
                self.learning.save(self)
            except Exception:
                log.exception("macro/deep learning cycle failed")
            self.cci.save()
            self.physics.save()
            await self._broadcast_full("grc")

    # -- lock rules at GRC --------------------------------------------------
    def _apply_grc_lock_rules(
        self, symbol: str, tf: str, now: float, res: dict
    ) -> None:
        cyc = self.cycles.get((symbol, tf))
        price = self.feed.prices().get(symbol, 0.0)

        if cyc is None:
            self.fresh_cycle(symbol, tf, now, res, reason="grc_first")
            return

        if cyc.pending_reset:
            # CLOSE POSITION branch: the cycle resets with a NEW prediction
            # after the next GRC (this is that GRC)
            self.fresh_cycle(symbol, tf, now, res, reason="post_close_reset")
            return

        # --- Extreme Override Conditions (ALL THREE must be simultaneously true).
        # Evaluated BEFORE the NO-TRADE gate: condition 1 is precisely the
        # confidence collapsing below 0.25.
        t4 = cyc.target(4)
        if t4.status == "locked":
            t4_dir = t4.direction or cyc.direction
            cond_conf = res["confidence"] < EXTREME_OVERRIDE_CCI
            cond_brain = (
                res["brain"]["polarity"] != 0
                and res["brain"]["polarity"] != t4_dir
            )
            cond_news = (
                self.news.impact_against(symbol, t4_dir) > EXTREME_OVERRIDE_NEWS
            )
            if cond_conf and cond_brain and cond_news:
                # Tp4 (macro thesis) changes ONLY here
                t4.direction = res["direction"] or t4_dir
                t4.level = self._project_level(
                    price, t4.direction, res["confidence"], res["atr"], t4.minutes
                )
                t4.locked_at = now
                t4.locked_cci = res["cci_adjusted"]
                cyc.direction = t4.direction
                cyc.events.append({"ts": now, "type": "extreme_override"})

        if not res["issued"]:
            # NO TRADE — insufficient confidence
            cyc.issued = False
            cyc.confidence = res["confidence"]
            cyc.cci = res["cci_adjusted"]
            cyc.drhe = res["drhe"]
            cyc.active_k = None
            cyc.tp = cyc.sl = None
            cyc.ratio = None
            cyc.refreshed_at = now
            return

        if not cyc.issued or cyc.direction == 0:
            # confidence recovered (or first live GRC) -> fresh lock
            self.fresh_cycle(symbol, tf, now, res, reason="confidence_recovered")
            return

        prev_cci = cyc.cci
        new_cci = res["cci_adjusted"]
        shift = abs(new_cci - prev_cci)

        # --- Tp1: lowest lock strength — adjusts at every GRC before being hit
        t1 = cyc.target(1)
        if t1.status == "locked":
            t1.level = self._project_level(
                price, res["direction"] or t1.direction, res["confidence"],
                res["atr"], t1.minutes,
            )
            t1.direction = res["direction"] or t1.direction
            t1.locked_at = now
            t1.locked_cci = new_cci
            cyc.direction = t1.direction      # live direction follows Tp1

        # --- Tp2: medium lock strength — only if CCI shift > 0.3
        t2 = cyc.target(2)
        if t2.status == "locked" and LOCK_SHIFT_ADJUST["tp2"] is not None:
            if abs(new_cci - (t2.locked_cci or 0.0)) > LOCK_SHIFT_ADJUST["tp2"]:
                t2.level = self._project_level(
                    price, cyc.direction, res["confidence"], res["atr"], t2.minutes
                )
                t2.direction = cyc.direction
                t2.locked_at = now
                t2.locked_cci = new_cci

        # --- Tp3: high lock strength — only if CCI shift > 0.5
        t3 = cyc.target(3)
        if t3.status == "locked" and LOCK_SHIFT_ADJUST["tp3"] is not None:
            if abs(new_cci - (t3.locked_cci or 0.0)) > LOCK_SHIFT_ADJUST["tp3"]:
                t3.level = self._project_level(
                    price, cyc.direction, res["confidence"], res["atr"], t3.minutes
                )
                t3.direction = cyc.direction
                t3.locked_at = now
                t3.locked_cci = new_cci

        cyc.confidence = res["confidence"]
        cyc.cci = new_cci
        cyc.drhe = res["drhe"]
        cyc.issued = True
        cyc.refreshed_at = now
        self._apply_risk(symbol, tf, cyc, res)

    # ==================================================================
    # event-driven recomputes (between GRCs — explicitly allowed by the spec)
    # ==================================================================
    def _event_result(self, symbol: str, tf: str) -> dict:
        pack = self.compute_asset(symbol, record=False, display=False)
        return pack[tf]

    # -- sequential target locks -------------------------------------------
    async def _on_target_hit(self, symbol: str, tf: str, k: int, now: float) -> None:
        cyc = self.cycles.get((symbol, tf))
        if cyc is None:
            return
        t = cyc.target(k)
        if t.status != "locked":
            return
        t.status = "hit"
        cyc.events.append({"ts": now, "type": "target_hit", "k": k, "level": t.level})

        if k in (1, 2):
            # "The moment Tp1 is reached, immediately calculate and lock Tp2
            #  using all current data" (same for Tp2 -> Tp3)
            res = self._event_result(symbol, tf)
            self.last_result[(symbol, tf)] = res
            shift = abs(res["cci_adjusted"] - (cyc.cci or 0.0))
            if res["direction"] and shift > (0.3 if k == 1 else 0.5):
                cyc.direction = res["direction"]
            nxt = cyc.target(k + 1)
            nxt.level = self._project_level(
                self.feed.prices().get(symbol, 0.0),
                cyc.direction,
                res["confidence"],
                res["atr"],
                nxt.minutes,
            )
            nxt.direction = cyc.direction
            nxt.status = "locked"
            nxt.locked_at = now
            nxt.locked_cci = res["cci_adjusted"]
            cyc.active_k = k + 1
            cyc.confidence = res["confidence"]
            cyc.cci = res["cci_adjusted"]
            self._apply_risk(symbol, tf, cyc, res)
            await self.emit({
                "type": "event",
                "kind": "target_locked",
                "symbol": symbol,
                "tf": tf,
                "k": k + 1,
                "level": nxt.level,
                "direction": nxt.direction,
                "ts": now,
            })
        elif k == 3:
            cyc.active_k = None       # all sequential sub-targets completed
        # k == 4: macro thesis hit — informational only

    # -- SL Hit Protocol ----------------------------------------------------
    async def _on_sl_hit(self, symbol: str, tf: str, now: float) -> None:
        cyc = self.cycles.get((symbol, tf))
        if cyc is None or not cyc.issued or cyc.active_k is None:
            return
        original_direction = cyc.direction
        # E2/E9 history: this cycle experienced an SL hit (both branches)
        self.hist.record_sl_hit(symbol)
        # FULL RECALCULATION EVENT: every formula across all subsystems,
        # from scratch, using the latest available data
        res = self._event_result(symbol, tf)
        self.last_result[(symbol, tf)] = res
        new_direction = res["direction"]

        if new_direction == original_direction:
            # SAFE — BE CAUTIOUS (amber): prediction stays active,
            # SL tightened by 15%, next sub-target recalculated, confidence -0.15
            cyc.safe_events += 1
            cyc.sl_tighten *= (1.0 - SL_TIGHTEN_FACTOR)
            cyc.confidence = max(0.0, cyc.confidence - SL_CONFIDENCE_PENALTY)
            cyc.issued = cyc.confidence > CONFIDENCE_FLOOR
            cyc.active_k = (
                cyc.active_k if cyc.issued else None
            )
            # next sub-target recalculated with updated parameters
            if cyc.issued and cyc.active_k is not None:
                t = cyc.target(cyc.active_k)
                if t.status == "locked":
                    t.level = self._project_level(
                        self.feed.prices().get(symbol, 0.0),
                        cyc.direction,
                        cyc.confidence,
                        res["atr"],
                        t.minutes,
                    )
                    t.locked_at = now
                    t.locked_cci = res["cci_adjusted"]
            self._apply_risk(symbol, tf, cyc, res)
            cyc.refreshed_at = now
            cyc.safe_alert = True
            await self.emit({
                "type": "sl_event",
                "kind": "safe",
                "symbol": symbol,
                "tf": tf,
                "message": "SAFE — BE CAUTIOUS",
                "confidence": cyc.confidence,
                "sl_tighten_pct": round((1.0 - cyc.sl_tighten) * 100, 1),
                "ts": now,
            })
        else:
            # CLOSE POSITION IMMEDIATELY (red): haptic + alert sound;
            # all remaining sub-targets AND Tp4 invalidated;
            # cycle resets with a new prediction after the next GRC
            for tgt in cyc.targets:
                if tgt.status in ("locked", "pending"):
                    tgt.status = "invalidated"
            cyc.active_k = None
            cyc.issued = False
            cyc.pending_reset = True
            cyc.close_alert = True
            cyc.tp = cyc.sl = None
            cyc.ratio = None
            cyc.refreshed_at = now
            await self.emit({
                "type": "sl_event",
                "kind": "close",
                "symbol": symbol,
                "tf": tf,
                "message": "CLOSE POSITION IMMEDIATELY",
                "ts": now,
            })

    # ==================================================================
    # 1 Hz tick loop: hit detection, countdown rollovers, phase beat
    # ==================================================================
    async def _tick_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(1.0)
            if self.phase != "live" or not self.locked:
                continue
            now = time.time()
            prices = self.feed.prices()
            for symbol in SYMBOL_ORDER:
                price = prices.get(symbol, 0.0)
                if price <= 0:
                    continue
                for tf in TF_ORDER:
                    cyc = self.cycles.get((symbol, tf))
                    if cyc is None or cyc.pending_reset:
                        continue
                    # sub-target / TP / SL hit detection (event-driven locks)
                    if cyc.issued and cyc.active_k is not None:
                        t = cyc.target(cyc.active_k)
                        if t.status == "locked" and t.level is not None:
                            d = t.direction or cyc.direction
                            if (d > 0 and price >= t.level) or (
                                d < 0 and price <= t.level
                            ):
                                await self._on_target_hit(symbol, tf, t.k, now)
                                cyc = self.cycles.get((symbol, tf))
                        if cyc.sl is not None:
                            against = (price <= cyc.sl and cyc.direction > 0) or (
                                price >= cyc.sl and cyc.direction < 0
                            )
                            if against:
                                await self._on_sl_hit(symbol, tf, now)
                                continue
                        if cyc.tp is not None:
                            hit_tp = (price >= cyc.tp and cyc.direction > 0) or (
                                price <= cyc.tp and cyc.direction < 0
                            )
                            if hit_tp and cyc.tp_dist:
                                cyc.events.append(
                                    {"ts": now, "type": "tp_hit", "tp": cyc.tp}
                                )
                                cyc.tp = None      # one-shot marker
                    # countdown reached zero -> immediate fresh cycle
                    if cyc.cycle_end and now >= cyc.cycle_end and not cyc.pending_reset:
                        self.last_window_ts = time.time()
                        res = self._event_result(symbol, tf)
                        self.last_result[(symbol, tf)] = res
                        self.fresh_cycle(symbol, tf, now, res, reason="rollover")
                        await self.emit({
                            "type": "event",
                            "kind": "rollover",
                            "symbol": symbol,
                            "tf": tf,
                            "cycle_end": self.cycles[(symbol, tf)].cycle_end,
                            "ts": now,
                        })

    # ==================================================================
    # manual override (per timeframe)
    # ==================================================================
    async def override(self, symbol: str, tf: str) -> bool:
        if symbol not in SYMBOL_ORDER or tf not in TF_ORDER:
            return False
        if self.phase != "live":
            return False
        async with self._busy:
            now = time.time()
            res = self._event_result(symbol, tf)
            self.last_result[(symbol, tf)] = res
            self.fresh_cycle(symbol, tf, now, res, reason="manual_override")
            await self.emit({
                "type": "event",
                "kind": "override",
                "symbol": symbol,
                "tf": tf,
                "cycle_end": self.cycles[(symbol, tf)].cycle_end,
                "ts": now,
            })
            await self._broadcast_full("override")
        return True

    # ==================================================================
    # snapshots / broadcast
    # ==================================================================
    def build_snapshot(self) -> dict[str, Any]:
        now = time.time()
        assets: dict[str, Any] = {}
        prices = self.feed.prices()
        for symbol in SYMBOL_ORDER:
            st = self.feed.state(symbol)
            base = st.day_open or (st.closes[0] if st.closes else 0.0)
            price = prices.get(symbol, 0.0)
            tfs: dict[str, Any] = {}
            for tf in TF_ORDER:
                cyc = self.cycles.get((symbol, tf))
                tfs[tf] = cyc.to_dict(now) if cyc else None
            assets[symbol] = {
                "price": price,
                "window_change_pct": round(
                    (price / base - 1.0) * 100, 3
                ) if base else 0.0,
                "mode": self.feed.mode,
                "tfs": tfs,
            }
        # one-shot alert flags are consumed by this snapshot (Section 12)
        for cyc in self.cycles.values():
            if cyc is not None:
                cyc.close_alert = False
                cyc.safe_alert = False
        return {
            "type": "snapshot",
            "ts": now,
            "phase": self.phase,
            "grc": {
                "count": self.grc_count,
                "last": self.last_grc,
                "next": next_grc_ts(now),
                "seconds_to_next": round(seconds_to_next_grc(now), 1),
                "id": grc_id(now),
            },
            "analysis": {
                "remaining_s": (
                    max(0, int(self.phase_deadline - now))
                    if self.phase == "analysis" else 0
                ),
                "readiness": dict(self.readiness),
                "warning": self.phase_warning,
            },
            "assets": assets,
            "bgn": {
                tf: {t2: round(v, 4) for t2, v in row.items()}
                for tf, row in (self.bgn.last_matrix.get("BTC") or {}).items()
            },
            "weights": {
                str(f): round(
                    sum(
                        self.cci.get_weights(s, tf).get(f, 0.0)
                        for s in SYMBOL_ORDER for tf in TF_ORDER
                    ) / (len(SYMBOL_ORDER) * len(TF_ORDER)),
                    4,
                )
                for f in WEIGHT_IDS
            },
            "formula_names": {
                f: FORMULA_NAMES[f] for f in FORMULA_FUNCTIONS
            },
            "weight_labels": {
                **{str(f): FORMULA_NAMES[f] for f in FORMULA_FUNCTIONS},
                str(BRAIN_COMPONENT_ID): "BRAIN CONSENSUS",
                str(EMOTION_COMPONENT_ID): "CERC COMPOSITE",
                str(PHYSICS_COMPONENT_ID): "PHYSICS UFP",
                str(NEWS_COMPONENT_ID): "NEWS IMPACT",
                str(HEDGE_COMPONENT_ID): "HEDGE+DARK",
                "20": FORMULA_NAMES[20],
                "28": "ARC (risk)",
                "29": "DLP (risk)",
                "30": "PCV (validator)",
            },
            "physics": self.physics.status(),
            "news_meta": {
                "status": self.news.status(),
                "categories": {
                    name: {"weight": c["weight"], "duration": c["duration"]}
                    for name, c in NEWS_CATEGORIES.items()
                },
            },
            "brains": self.brains.status(),
            "flat_watch": self._flat_payload(),
            # §10.3 status chips: every subsystem's honesty state on top
            "chips": {
                "feed": "simulated" if self.feed.stats().get("simulated")
                        else "live",
                "local_model": "on" if getattr(self.agents.local, "available", False)
                               else "off",
                "agents": (self.agents.status().get("mode")
                           if hasattr(self.agents, "status") else "unknown"),
                "news": "live" if getattr(self.news, "source", "") not in ("", "seed")
                        else "seed",
                "keyless": not any(self.agents.settings.get(k) for k in
                                   ("gemini_key", "backup1_key", "backup2_key",
                                    "backup3_key")),
                "degrade": self.degrade_status()["level"],
                "deadline_misses": self.agent_deadline_misses,
            },
            "meta": self.meta.payload(),
            "formula_values": {
                f"{s}|{tf}": data
                for (s, tf), data in self.display_formulas.items()
            },
            "systems": {
                f"{s}|{tf}": data
                for (s, tf), data in self.display_systems.items()
            },
            # --- Part 4 blocks (Sections 10/11/13/14) ----------------------
            "learning": {
                "rates": dict(self.learning.rates),
                "high_alert": self.learning.high_alert,
                "probation_threshold": self.learning.probation_threshold,
                "subsystem_shares": {
                    k: round(v, 4)
                    for k, v in self.learning.subsystem_shares.items()
                },
                "tf_hr": {tf: round(self.learning.tf_hr.get(tf, 0.5), 4)
                          for tf in TF_ORDER},
                "macro": {"count": self.learning.macro_count,
                          "last_grc": self.learning.last_macro_grc,
                          "tone": round(self.learning.macro_tone, 4)},
                "deep": {"count": self.learning.deep_count,
                         "last_ts": self.learning.last_deep_ts,
                         "minutes_to_next": round(
                             max(0.0, self.learning.last_deep_ts
                                 + 247 * 60 - now) / 60, 1)},
                "brain_influence": {
                    tf: [round(x, 4) for x in
                         self.learning.brain_influence("BTC", tf)]
                    for tf in TF_ORDER
                },
                "formula_accuracy": {
                    str(f): (round(a, 4) if (a := self.learning.formula_accuracy(f)) is not None else None)
                    for f in list(WEIGHT_IDS)[:12]
                },
                "probation_count": len(self.learning.probation),
            },
            "agents": self.agents.status(),
            "feed": self.feed.stats(),
            "ledger": self.ledger.api_payload(SYMBOL_ORDER, TF_ORDER),
            "locks": [dict(v) for v in self.locks.values()],
            "hindsight": self.hindsight(),
            "watchdog": {
                "seconds_since_window": round(time.time() - self.last_window_ts, 1),
                "stalled": (time.time() - self.last_window_ts) >
                            (TIMEFRAMES["tf4"]["minutes"] * 60 + 45),
            },
            "agent_signal": self.last_agent_signal or {},
        }

    async def _broadcast_full(self, reason: str) -> None:
        snap = self.build_snapshot()
        snap["reason"] = reason
        self.last_snapshot = snap
        await self.emit(snap)
