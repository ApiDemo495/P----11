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
    HIST_COMPOSITE_EMOTION,
    HIST_EMOTION_LAG,
    HIST_STD_R,
    INITIAL_ANALYSIS_MAX_SECONDS,
    LOCK_SHIFT_ADJUST,
    MIN_BARS_COMPUTE,
    MIN_BARS_FULL_WINDOW,
    SL_CONFIDENCE_PENALTY,
    SL_TIGHTEN_FACTOR,
    SYMBOL_ORDER,
    TARGET_SCALE,
    TF_ORDER,
    TIMEFRAMES,
    WEIGHT_IDS,
)
from .brain import BrainSystem
from .emotion import EmotionSystem
from .feed import MarketFeed
from .formulas import FORMULA_FUNCTIONS, FORMULA_NAMES, compute_all
from .formulas.f01_f10 import abs_changes as _abs_changes
from .formulas._helpers import mean as _mean, sign as _sign, std_pop as _std_pop
from .formulas.f22_f31 import f30_pcv, fractal_boundary_levels
from .history import HistoryStore
from .models import FormulaContext, TargetState, TFCycle
from .news import NewsSystem
from . import orderflow
from .physics import PhysicsSystem
from .risk import arc_ratio, dynamic_level_placement

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
        self.readiness: dict[str, bool] = {
            "bars": False,
            "formulas": False,
            "brains": False,
            "emotions": False,
            "news": False,
            "weights": False,
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
        self.brain_weight_adj: dict[tuple[str, str], dict[int, float]] = {}
        self.news_polarity: deque[float] = deque(maxlen=30)
        self._market_cache: dict[str, dict] = {}
        self._market_cache_ts: float = 0.0

        self._tasks: list[asyncio.Task] = []
        self._busy = asyncio.Lock()
        self._stop = asyncio.Event()

    # ==================================================================
    # lifecycle
    # ==================================================================
    async def start(self) -> None:
        self.cci.load()
        await self._initialize_subsystems()
        self._tasks.append(asyncio.create_task(self._analysis_loop()))
        self._tasks.append(asyncio.create_task(self._tick_loop()))

    async def stop(self) -> None:
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self.cci.save()

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
            raws[tf] = compute_all(ctx, record=record)

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
                # Formula 21 weight learning + Brain-4 secondary adjustment
                self.cci.update_weights(symbol, tf, price)
                self._apply_brain4(symbol, tf)
                # CERC emotion-weight rewards (21 channels, eta rule)
                prev_em = self.prev_emotions.get((symbol, tf))
                if prev_em and move != 0:
                    self.emotions.reward_update(
                        symbol, tf, prev_em[0], prev_em[1], 0.0, move
                    )
                # consecutive-correct running ratio (0..1, last 10)
                pdir = self.prev_cci_dir.get((symbol, tf), 0)
                if move != 0 and pdir != 0:
                    self.hist.commit_win_streak(symbol, pdir == _sign(move))

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
        physics_raw = {tf: self.physics.aggregate(symbol, tf)["aggregate"] for tf in TF_ORDER}
        physics_sync = BgnSynchronizer.adjust(matrix, physics_raw)

        # 7) Formula 21 per timeframe + Part-2 confidence pipeline
        for tf in TF_ORDER:
            drhe = adj[tf].get(20, 1.0)
            comp: dict[int, float] = {fid: adj[tf][fid] for fid in FORMULA_FUNCTIONS}
            comp[BRAIN_COMPONENT_ID] = brain_sync[tf]
            comp[EMOTION_COMPONENT_ID] = cerc_sync[tf]
            pack = self.cci.compute(symbol, tf, comp, drhe)

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
            )

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
            }

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
                "physics": {"aggregate": physics_sync[tf]},
                "news": news_agg,
            }
            if record:
                self.cci.store_signals(symbol, tf, comp)
                self.prev_comp[(symbol, tf)] = dict(comp)
                self.prev_brain_signals[(symbol, tf)] = [
                    inst["h_signal"] for inst in br["instances"]
                ]
                self.prev_emotions[(symbol, tf)] = (em["emotions"], em["directions"])
                self.prev_cci_dir[(symbol, tf)] = pack["direction"]
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
                # display snapshot — GRC cadence only (spec)
                self.display_formulas[(symbol, tf)] = {
                    "raw": {str(f): round(raws[tf].get(f, 0.0), 5) for f in FORMULA_FUNCTIONS},
                    "adj": {str(f): round(adj[tf].get(f, 0.0), 5) for f in FORMULA_FUNCTIONS},
                    "names": {str(f): FORMULA_NAMES[f] for f in FORMULA_FUNCTIONS},
                }
                self.display_systems[(symbol, tf)] = {
                    "brain": brain_consensus,
                    "emotion": {
                        "aggregate": round(cerc_sync[tf], 5),
                        "cerc": em["cerc"],
                        "emotions": em["emotions"],
                        "directions": em["directions"],
                        "weights": [round(x, 5) for x in self.emotions._weights(symbol, tf)],
                    },
                    "physics": {"aggregate": round(physics_sync[tf], 5)},
                    "news": {
                        "bull": news_agg["bull"],
                        "bear": news_agg["bear"],
                        "polarity": news_agg["polarity"],
                        "items": news_agg["items"],
                    },
                    "drhe": round(drhe, 4),
                    "atr": results[tf]["atr"],
                    "pcv": round(pcv, 5),
                    "arc": round(arc, 4),
                    "win_ratio": round(self.hist.win_ratio.get(symbol, 0.0), 3),
                }
        if record:
            self.cci.store_price(symbol, price)
            self.readiness["formulas"] = True
            self.readiness["weights"] = True
        return results

    # ==================================================================
    # cycle helpers
    # ==================================================================
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
        tp, sl, tp_dist, sl_dist = dynamic_level_placement(
            price=price,
            direction=cyc.direction,
            ratio=ratio,
            atr=res["atr"],
            boundaries=boundaries,
            drhe=res.get("drhe", 1.0),
            ofti_value=self._build_market(symbol).get("ofti", 0.0),
            tighten=cyc.sl_tighten,
        )
        return tp, sl, tp_dist, sl_dist, ratio

    def _apply_risk(self, symbol: str, tf: str, cyc: TFCycle, res: dict) -> None:
        tp, sl, tp_dist, sl_dist, ratio = self._risk_pack(symbol, tf, cyc, res)
        if ratio is None and cyc.issued and cyc.active_k is None:
            return          # all sub-targets completed — keep last placement
        cyc.tp, cyc.sl = tp, sl
        cyc.tp_dist, cyc.sl_dist = tp_dist, sl_dist
        cyc.ratio = ratio

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
        cyc = TFCycle(
            symbol=symbol,
            tf=tf,
            W=W,
            direction=res["direction"],
            confidence=res["confidence"],
            cci=res["cci_adjusted"],
            drhe=res["drhe"],
            issued=res["issued"],
            cycle_start=now,
            cycle_end=now + W * 60,
            targets=targets,
            active_k=1 if res["issued"] else None,
            warning=warning,
            refreshed_at=now,
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
                for tf in TF_ORDER:
                    res = res_pack[tf]
                    self.last_result[(symbol, tf)] = res
                    self._apply_grc_lock_rules(symbol, tf, now, res)
            self.cci.save()
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
            "formula_values": {
                f"{s}|{tf}": data
                for (s, tf), data in self.display_formulas.items()
            },
            "systems": {
                f"{s}|{tf}": data
                for (s, tf), data in self.display_systems.items()
            },
        }

    async def _broadcast_full(self, reason: str) -> None:
        snap = self.build_snapshot()
        snap["reason"] = reason
        self.last_snapshot = snap
        await self.emit(snap)
