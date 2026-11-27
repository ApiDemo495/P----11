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
import os
import tempfile
import threading
import time

import numpy as np
from collections import deque
from typing import Any, Awaitable, Callable

from . import loadgov as _gov
from .bgn import BgnSynchronizer
from .cci import CCIEngine, calibrate_confidence, finalize_confidence
from .clock import grc_id, next_grc_ts, seconds_to_next_grc
from .config import (
    asset_profile,
    BRAIN_COMPONENT_ID,
    CONFIDENCE_FLOOR,
    RECHECK_PASS_SCORE,
    RECHECK_CONF_REACH,
    EMOTION_COMPONENT_ID,
    EXTREME_OVERRIDE_CCI,
    EXTREME_OVERRIDE_NEWS,
    HEDGE_COMPONENT_ID,
    TRACER_COMPONENT_ID,
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
    WEIGHT_IDS,
)
from .contracts import clamp_rung, rung_level   # canonical rung prices
from .brain import BrainSystem
from .rlsys import damp_all, damp_shares
from .silence import guard, report as _silence_report   # counted best-effort
from .emotion import EmotionSystem
from .feed import MarketFeed
from .sleeves import evaluate_sleeves, load_registry as load_sleeve_registry
from .portfolio_live import LivePortfolio
from .formulas import FORMULA_FUNCTIONS, FORMULA_NAMES, compute_all
from .formulas.trace import traced_compute
from .forward import ForwardTables
from .meta import MetaCombiner
from .predlock import PredictionJournal
from . import genesis as genesis_mod

# §2.10 non-directional components: never a direction vote (confidence /
# regime routing only). 97-101 are the brain/emotion/physics/news/hedge
# subsystem inputs to the CCI; 20 is DRHE the regime gauge.
NON_VOTING_IDS = frozenset({20, 97, 98, 99, 100, 101})
# NEW-mode target: signal-tilled multiple of the measured stop (tail
# units). Tuned on the 2y real-data mandate (backtest/mandate.py) —
# candidates {1.2, 1.6, 2.0}, selected on train, validated on holdout.
NEW_TARGET_RR = 2.0
EMERGENCY_RELOCK_COOLDOWN_S = 900.0
EMERGENCY_RELOCK_MIN_MOVE_ATR = 3.0
#: WINDOW FREEZE (audit §1.3). A locked contract is immutable for the
#: duration of its window. The old gate (≥0.5 ATR, 60 s cooldown) fired
#: every 10–30 s — the UI's own ">1/HOUR = BUG (N=8)" banner was right.
#: Only a structural invalidation (>3 ATR away from the locked entry AND
#: at least 15 minutes since the lock) may re-anchor the contract.
RELOCK_MIN_MOVE_ATR = 3.0
RELOCK_MIN_AGE_S = 15 * 60.0
#: Lower bound on the advanced layer's confidence multiplier (soft_gate ×
#: RQA). Both are calibration factors with a documented band; multiplying
#: them unchecked produced a ×0.34 collapse on live tapes (gate 0.45 ×
#: RQA 0.75) AFTER the calibrated confidence had already cleared the floor —
#: every lane printed 0.009 against a 0.366 bar and nothing could issue.
#: 0.70 is deliberately wider than either factor's own lower bound would
#: produce on a normal tape, so the layer still damps a genuinely bad window
#: without being able to silence the board.
_ADV_CONF_FLOOR = 0.60


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
from . import levels as _levels
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


def _recheck_prediction(*, direction: int, conf: float, floor: float,
                        feed_age_s, voters: dict, levels_ok: bool,
                        ratio, lock_dir: int, tfs_computed: int,
                        exp_bps=None, exp_n: int = 0,
                        exp_bps_all=None, exp_n_all: int = 0,
                        exp_sd=None, pa_conflict: bool = False) -> dict:
    """The analysis RECHECK — correctness is hard, opinion is graded.

    It used to be a flat AND of seven checks, so a single soft
    disagreement ("only 2 of 5 voters agreed", "confidence 0.34 vs floor
    0.35") printed NO TRADE even for a perfectly serviceable call. Markets
    are never unanimous and the old gate demanded unanimity, so it never
    fired: the "always says no trade / pessimistic" report.

    What is HARD (never negotiable, these are correctness or user rules):
      * the tape is live (tick <= 20 s old);
      * all four timeframes were computed in this run;
      * conviction is within reach of the floor (conf >= 0.6 x floor —
        better than 40 % short is not a call, it is noise);
      * levels are valid and TP:SL >= 1:1 (below 1:1 is never worth trading);
      * the call does not contradict the window's locked side (the lock
        stands; a flip mid-window is the churn the user reported);
      * measured expectancy is not negative when the ledger has >= 20
        settled windows.

    What is GRADED (partial credit, weighted into a score):
      * subsystem consensus (credit = agreeing/4, capped at 1);
      * confidence above the floor (credit = conf/floor, capped at 1).
    A prediction is produced when no hard rule fails and the weighted score
    clears `RECHECK_PASS_SCORE` (0.55). Every check is listed with its
    weight and credit, so a refusal always says what would have to change
    and by how much instead of a blanket NO TRADE.
    """
    checks: list[dict] = []
    hard_failed: list[str] = []

    def add(name: str, ok: bool, detail: str, weight: float,
            hard: bool = False, credit: float | None = None) -> None:
        c = {"name": name, "ok": bool(ok), "detail": detail,
             "weight": float(weight), "hard": bool(hard)}
        if credit is not None:
            c["credit"] = round(max(0.0, min(1.0, float(credit))), 4)
        checks.append(c)
        if hard and not ok:
            hard_failed.append(name)

    live_ok = feed_age_s is not None and feed_age_s <= 20
    add("live data", live_ok,
        (f"feed age {feed_age_s:.1f}s" if feed_age_s is not None
         else "no ticks yet"), 0.0, hard=True)
    add("all timeframes computed", tfs_computed >= 4,
        f"{tfs_computed}/4 timeframes in this run", 0.0, hard=True)

    agreeing = [k for k, v in voters.items()
                if v and direction and (v > 0) == (direction > 0)]
    n_agree = len(agreeing)
    add("subsystem consensus", direction != 0 and n_agree >= 3,
        f"{n_agree}/{len(voters)} voters agree"
        + (f" ({', '.join(agreeing)})" if agreeing else "")
        if direction else "no direction to agree with",
        0.45, credit=(min(1.0, n_agree / 4.0) if direction else 0.0))
    if pa_conflict:
        add("strong opposing price action", False,
            "closed-bar structure + level + trigger oppose the proposed side; "
            "watch-only until the conflict resolves", 0.0, hard=True)
    else:
        add("strong opposing price action", True,
            "no strong context-gated price-action veto", 0.0, hard=True)

    ratio_ok = bool(levels_ok and ratio is not None
                    and 1.0 <= float(ratio) <= 4.0)
    add("levels valid + ratio ≥ 1:1", ratio_ok,
        (f"ratio {ratio} (band 1:1–4:1, 2:1 preferred)"
         if ratio is not None else "no levels"), 0.0, hard=True)

    conf_credit = (min(1.0, max(0.0, conf) / floor) if floor > 0 else 0.0)
    above_floor = direction != 0 and conf > floor
    # A failed confidence check must NAME the numbers, not just the check:
    # "confidence above floor" alone sent users hunting for which number
    # missed (68bbb922 regression pin).
    _conf_detail = (f"conf {conf:.3f} vs floor {floor:.3f}" if above_floor
                    else f"confidence below floor ({conf:.3f} ≤ {floor:.3f})"
                         f" — gap {floor - conf:.3f}")
    add("confidence above floor", above_floor, _conf_detail,
        0.30, credit=conf_credit if direction != 0 else 0.0)
    reachable = direction != 0 and conf >= RECHECK_CONF_REACH * floor
    add("conviction within reach of the floor", reachable,
        f"conf {conf:.3f} vs minimum {RECHECK_CONF_REACH * floor:.3f} "
        f"(0.6 × floor)", 0.0, hard=True)

    if exp_bps is not None and exp_n >= 20:
        # Refuse only on EVIDENCE. A negative mean over a handful of noisy
        # windows (t ≈ −1) is indistinguishable from zero; treating it as a
        # veto is pessimism dressed up as rigour — the "always NO TRADE"
        # report. A mean that is negative AND significant (t ≤ −2) is a
        # real measured edge against us, and stays hard.
        import math as _m
        _t = None
        if exp_sd and exp_n > 1:
            _t = exp_bps / (float(exp_sd) / _m.sqrt(exp_n))
        _significant = (_t is None) or (_t <= -2.0)
        _all = (f" · all-time {exp_bps_all:+.1f} bp/window over n={exp_n_all}"
                if exp_bps_all is not None and exp_n_all else "")
        _hard = bool(exp_bps <= 0 and _significant)
        add("measured expectancy (settled ledger)", exp_bps > 0,
            f"recent {exp_bps:+.1f} bp/window over n={exp_n} settled"
            + (f" (t={_t:.2f}" + (", significant" if _t <= -2.0 else
                                  ", not significant") + ")"
               if _t is not None else "")
            + _all, 0.20, hard=_hard,
            credit=(1.0 if exp_bps > 0 else
                    max(0.0, min(1.0, 0.5 + 0.25 * (_t or -2.0)))))
    if lock_dir:
        agrees = direction == lock_dir
        add("agrees with locked call", agrees,
            "locked " + ("BUY" if lock_dir > 0 else "SELL")
            + " vs analysis " + ("BUY" if direction > 0 else
                                  "SELL" if direction < 0 else "FLAT"),
            0.0, hard=True)

    graded = [c for c in checks if c["weight"] > 0]
    total_w = sum(c["weight"] for c in graded) or 1.0
    score = sum(c["weight"] * (c.get("credit", 1.0 if c["ok"] else 0.0))
                for c in graded) / total_w
    failed = [c["name"] for c in checks if not c["ok"]]
    passed = (not hard_failed) and direction != 0 and score >= RECHECK_PASS_SCORE
    if hard_failed:
        _det = {c["name"]: c["detail"] for c in checks if c["hard"]}
        verdict = ("WATCH ONLY — hard rule failed: " + "; ".join(
            f"{n} ({_det.get(n, '')})" for n in hard_failed))
    elif direction == 0:
        verdict = "NO CALL — no direction to issue"
    elif passed:
        verdict = (f"RECHECK PASSED — graded score {score:.2f} ≥ "
                   f"{RECHECK_PASS_SCORE:.2f}"
                   + (f" (soft flags: {', '.join(failed)})" if failed
                      else " (every check green)"))
    else:
        verdict = (f"WATCH ONLY — graded score {score:.2f} < "
                   f"{RECHECK_PASS_SCORE:.2f}; weakest: "
                   + ", ".join(c["name"] for c in graded if not c["ok"]))
    return {"passed": bool(passed), "checks": checks, "failed": failed,
            "hard_failed": hard_failed, "score": round(score, 4),
            "threshold": RECHECK_PASS_SCORE,
            "note": ("correctness and user rules are hard; consensus and "
                     "confidence are graded — the market is never unanimous"),
            "verdict": verdict}


def _calibration_delta(audit: dict) -> float:
    """Additive correction for future P(target) forecasts, fitted ONLY
    from graded forecasts: systematically predict 62% but realise 54%
    -> future forecasts shift by the measured gap (clamped +-0.15).
    Needs n>=30 graded windows. SELF-DISABLING: once n>=60 and the
    calibrated Brier score is worse than the raw one, the correction
    turns itself off - calibration must prove it helps or it stops."""
    n = int(audit.get("n") or 0)
    if n < 30:
        return 0.0
    gap = audit["tp_first"] / n - audit["sum_pred"] / n
    if (n >= 60 and audit.get("brier_cal_sum") is not None
            and audit["brier_sum"] and audit["brier_cal_sum"] > audit["brier_sum"]):
        return 0.0
    return round(max(-0.15, min(0.15, gap)), 4)


def _locks_path() -> str:
    """Where the Signal Lock Protocol persists frozen contracts.

    PRED_LOCKS_PATH lets the test suite keep its manufactured locks out of
    the running app's file: engines in one pytest session used to inherit a
    lock written by an earlier test (old price, old levels), and the fresh
    cycle then found TP1 already satisfied — a failure with no code change
    and no relation to the test's subject.
    """
    return os.environ.get("PRED_LOCKS_PATH") or "data/locks.json"


def _lock_emotion_text(dom) -> str:
    """Human text for the LOCKED banner's emotion field.

    `dominant` is a dict (deviation ranking carries the name, the channel
    id, the citation and the tie count). Handing that dict to an f-string
    leaked a raw Python repr into the banner; this is the one place that
    turns it into words, and it never prints a dict.
    """
    if isinstance(dom, dict):
        name = (dom.get("name") or (f"E{dom['id']}" if dom.get("id")
                                    else "unknown emotion"))
        inten = float(dom.get("intensity") or 0.0)
        return f"{name} · {inten:.0%} intensity"
    return str(dom or "")


# ── heavy-pass executor ────────────────────────────────────────────────
# A SEPARATE, small pool for the CPU-bound passes (GRC, analyse, readiness
# dry-run, snapshot build). The first attempt used asyncio.to_thread, which
# runs on the SAME default executor FastAPI uses for every sync route — so
# "off the loop" work simply moved the stall into the request pool: with the
# analytics queue saturated, /api/health and /api/snapshot queued behind it
# (measured: 3.1 s round-trip, lag 6.6 s on a 2-core box). Bounded to the
# core count so jobs cannot pile up either.
import concurrent.futures as _fut
import functools as _ft

_PASS_WORKERS = max(2, min(4, (os.cpu_count() or 2)))
_pass_pool = _fut.ThreadPoolExecutor(max_workers=_PASS_WORKERS,
                                    thread_name_prefix="engine-pass")


async def run_pass(fn, /, *args, **kwargs):
    """Run a CPU-bound pass on the engine's own pool (never the loop)."""
    loop = asyncio.get_running_loop()
    call = _ft.partial(fn, *args, **kwargs)
    return await loop.run_in_executor(_pass_pool, call)


def _sync_block() -> dict:
    """§1.4 snapshot clock: one shared snapshot_ts + the 5 countdowns."""
    try:
        from .sync_clock import CLOCK
        return CLOCK.countdowns()
    except Exception:
        return {}


def _metacog_block() -> dict:
    """§4 metacognition telemetry: per-cell status + Exp3 weights."""
    try:
        from .metacognition import MC, CELL_IDS
        return {
            "statuses": {f"{s}|{t}": MC.status(s, t) for s, t in CELL_IDS},
            "weights": MC.cell_weights(),
        }
    except Exception:
        return {}


def _verdict_rule(*a, **k):
    from .verdict import public_verdict as _pv
    return _pv(*a, **k)


def _committee_block() -> dict:
    """§5 four-agent committee: the last vote each prediction cast."""
    try:
        from .agent_committee import COMMITTEE
        last = COMMITTEE.last
        return {"vote_count": COMMITTEE.vote_count,
                "last": last.as_dict() if last else None}
    except Exception:
        return {"vote_count": 0, "last": None}


def public_verdict(symbol: str, tf: str, direction: int, confidence: float,
                   tier: str | None, issue_note: str | None = None,
                   ) -> tuple[bool, str, str]:
    """Backwards-compatible re-export — the rule lives in app/verdict.py."""
    return _verdict_rule(symbol, tf, direction, confidence, tier, issue_note)


def resolve_verdict(symbol: str, tf: str, direction: int, confidence: float,
                    tier: str | None, issue_note: str | None = None,
                    ) -> tuple[bool, str, str, int]:
    """Same rule, plus the SIDE it resolved to (the trend flip lives there)."""
    from .verdict import resolve as _res
    return _res(symbol, tf, direction, confidence, tier, issue_note)


def calib_should_record(done: dict, key: tuple[str, str], move: float) -> bool:
    """True once per settled window, whichever grading path asks first.

    Two paths grade the same window at the same boundary (the paired
    ``calib_p``/``calib_h`` rings and the lock-learning pass), and both used
    to append — so every window was counted twice in the calibration panel
    and in its Brier score. The wake is identified by the realised move,
    which is frozen for a settled window.
    """
    prev = done.get(key)
    if prev is not None and abs(float(prev) - float(move)) < 1e-12:
        return False
    done[key] = float(move)
    return True


def _trend_evidence(symbol: str, tf: str, side: int) -> dict:
    """The trend gate's own reading, for the card's evidence block."""
    try:
        from . import trend_gate
        return trend_gate.check(symbol, tf, side)
    except Exception:
        return {}


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
        # certified-sleeve snapshot cache: (timestamp, payload) or (0, None)
        self._sleeve_cache: tuple[float, dict[str, Any] | None] = (0.0, None)
        # live paper portfolio: concurrent certified sleeves + risk caps
        self.portfolio = LivePortfolio()
        self._portfolio_bar: int = 0
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
        self._last_auto_analyse: float = 0.0   # AUTO-PREDICT cadence
        self._analyse_task: asyncio.Task | None = None
        # A manual ▶ ANALYSE must never be refused because the 45-second
        # auto pass happens to be running. A different scope is queued and
        # started the moment the running job finishes.
        self._analyse_queue: list[tuple[str, str]] = []
        # §1.1 latest-wins request ids per (symbol, tf)
        self._analyse_reqids: dict[tuple[str, str], str] = {}
        # §1.1 monotonic state version for atomic client-side discard
        self._state_version: int = 0
        # win-rate search confluence: latest finalized direction per
        # (symbol, tf) with its timestamp — the agree gate reads the
        # OTHER timeframes from the most recent completed pass
        self._last_dirs: dict[tuple[str, str], tuple[float, int]] = {}
        # HOLD POLICY: per-cell cursor for the era accountant (window
        # boundary + closed-bar index) — 16 independent (asset, tf)
        # cells, each holding one position per side-era
        self._hold_cursor: dict[tuple[str, str], dict] = {}
        self.started_at = time.time()      # §12 ops: uptime
        self.vault_last_write = 0.0        # §12 ops: vault last write
        self.build_id = ""                 # §12 ops: build id vs remote
        # hourly/daily hindsight reviews (advice §8.3)
        self.hindsight_mult: dict[tuple[str, str, int], float] = {}
        self.hindsight_changes: list[dict] = []
        # §1.1 composite keys: per (symbol, tf) — a tf-only key let
        # BTC's review stamp every asset's cadence, so ETH/BNB/PAXG
        # skipped their own re-grade (the BTC-22m bleed root cause)
        self._hindsight_last: dict[tuple[str, str], float] = {}
        self._hindsight_reviewed_n: dict[tuple[str, str], int] = {}
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
        # last walk-forward edge-model vote (app/edge_model.py); None when no
        # artifact is present, which keeps behaviour byte-identical
        self._edge_model: dict[str, Any] | None = None
        # display snapshots — updated ONLY at GRC cadence (spec)
        self.display_formulas: dict[tuple[str, str], dict[str, Any]] = {}
        self.display_systems: dict[tuple[str, str], dict[str, Any]] = {}
        self.last_snapshot: dict[str, Any] = {}
        self.last_grc: float = 0.0
        self.grc_count: int = 0

        # --- Part-2 cross-GRC state (rewards/learning inputs) --------------
        self.prev_comp: dict[tuple[str, str], dict[int, float]] = {}
        self.prev_brain_signals: dict[tuple[str, str], list[float]] = {}
        self.prev_emotions: dict[tuple[str, str], tuple[dict, dict, float]] = {}
        self.prev_cci_dir: dict[tuple[str, str], int] = {}
        self.prev_physics: dict[tuple[str, str], dict[int, float]] = {}
        self.brain_weight_adj: dict[tuple[str, str], dict[int, float]] = {}
        # brain.md §6: brain vote fusion weight = 0.25 × hindsight
        # multiplier clip(1 + 0.25·z, 0.25, 1.5); 0 when absent
        self._fw_hindsight: dict[tuple[str, str], float] = {}
        self._fw_last_vote: dict[tuple[str, str], float] = {}
        self._last_raws: dict[tuple[str, str], dict] = {}
        self._calib_pending: dict[tuple[str, str], float] = {}
        # last realised move already counted in the calibration rings, so the
        # two grading paths cannot book the same window twice
        self._calib_done: dict[tuple[str, str], float] = {}
        # RL controllers (error-finder + issue bandit) + tracer last value
        from .rlsys import get_rl
        self.rl = get_rl()
        self._tracer_last: dict[tuple[str, str], float] = {}
        # RL deepening: per-(asset,tf) window state / chosen Q-mix /
        # bandit context from the latest compute, frozen into the lock
        self._rl_state: dict[tuple[str, str], dict[str, str]] = {}
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
        # SEALED PREDICTIONS: ANALYSE writes one immutable lock per window,
        # the tick loop grades it, and the settlement feeds the learners.
        # This is the user-visible "prediction lock" (app/predlock.py).
        self.predlocks = PredictionJournal()
        self._last_lock_settle = 0.0
        self._thoughts: dict[str, list[dict]] = {}
        self._load_thoughts()
        # honest confidence: isotonic map confidence → P(win), fed by the
        # ledger's own settled rows (app/calib.py)
        from .calib import calibrator
        self.calib = calibrator()
        # optional Binance execution (disabled unless the user enables
        # + arms it in Settings — see app/trading.py safety doctrine)
        from .trading import Trader
        self.trader = Trader()
        # probability-forecast audit: every window's predicted
        # P(target-before-stop) is remembered and graded against the
        # realised first touch at settlement — the probability system
        # itself gets an accuracy score, not just the direction.
        self._conf_hist: dict = {}
        self._prob_pred: dict = {}
        self._prob_audit = {"n": 0, "sum_pred": 0.0, "tp_first": 0,
                            "brier_sum": 0.0}
        self._load_prob_audit()
        self.locks: dict[tuple[str, str], dict] = {}   # frozen signals
        self._load_locks()
        self.last_window_ts = time.time()              # watchdog anchor
        self.historydb = HistoryDB()
        self.last_agent_signal: dict[str, Any] = {}
        self.last_grc_signals: dict[tuple[str, str], dict] = {}

        self._tasks: list[asyncio.Task] = []
        self._busy = asyncio.Lock()
        # readiness dry-run bookkeeping (see _update_readiness): attempt count,
        # next allowed attempt, and the last measured failure
        self._readiness_tries = 0
        self._readiness_next_at = 0.0
        self._readiness_error = ""
        self.grc_started_at = 0.0
        self.grc_ms: float | None = None
        self.grc_profile: dict[str, float] = {}
        # Probability-Branches replay cache: the empirical first-touch scan
        # is O(history x window) and was 32% of every snapshot build (the
        # single biggest reason /api/snapshot took ~2 s). Keyed on the
        # window's own signature, so a new cycle/bar/price move recomputes.
        self._branch_cache: dict[tuple, tuple] = {}
        self._stop = asyncio.Event()

    # ==================================================================
    # lifecycle
    # ==================================================================
    async def start(self) -> None:
        self.genesis.start_worker()
        self.cci.load()
        # brain.md B-11: never block first prediction — load the cached
        # connectome now or build it in the background (absent until then)
        try:
            # the connectome load is synchronous CPU work on a 202 MB
            # npz — running it inline froze the event loop solid for
            # the whole load (health/WS dead = proxy 502 on Codespaces)
            import asyncio as _aio
            _aio.get_running_loop().create_task(
                _aio.to_thread(self.brains.fw.ensure))
        except Exception:
            log.exception("flywire ensure failed")
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
                        self._attach_flow(ctx)
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
        ctx = FormulaContext(
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
            funding=self._funding_series(symbol),
        )
        self._attach_flow(ctx)
        return ctx

    def _attach_flow(self, ctx) -> None:
        """Live order-book rings + liquidation flow for F32–F35.

        Cause-6 honesty: with no live feed the rings are empty and the
        flow is None → the formulas read0 and trace.py prints the
        'no live book / no live forceOrder' reason."""
        try:
            ctx.book = self.feed.book_view(ctx.symbol)
        except Exception:
            ctx.book = {"dbi_ring": [], "dev_ring": [],
                        "lambda_ring": [], "latest": {"book": False}}
        try:
            ctx.liq_flow = self.feed.micro_flow(ctx.symbol)
        except Exception:
            ctx.liq_flow = None

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

        # ── §3 emotion feeds ─────────────────────────────────────────
        # E11 herd synchronization: rolling correlation of this asset's
        # returns vs the peer mean (0.5 = uncorrelated baseline)
        herd_corr = None
        try:
            _mine = price_rets[-30:]
            _peer_means = []
            for j in range(len(price_rets)):
                _vals = [peer_series[s][j] for s in SYMBOL_ORDER
                         if s in peer_series and j < len(peer_series[s])]
                if _vals:
                    _peer_means.append(sum(_vals) / len(_vals))
            _pm = _peer_means[-30:]
            _n = min(len(_mine), len(_pm))
            if _n >= 10:
                _a, _b = _mine[-_n:], _pm[-_n:]
                _ma, _mb = sum(_a) / _n, sum(_b) / _n
                _sa = sum((x - _ma) ** 2 for x in _a) ** 0.5
                _sb = sum((x - _mb) ** 2 for x in _b) ** 0.5
                if _sa > 1e-12 and _sb > 1e-12:
                    _cov = sum((_a[i] - _ma) * (_b[i] - _mb)
                               for i in range(_n)) / _n
                    herd_corr = max(-1.0, min(1.0, _cov / (_sa * _sb)))
        except Exception:
            herd_corr = None

        # E10 whale-vs-retail trade sizes from the micro structure ring
        trade_sizes: list[float] = []
        try:
            trade_sizes = list(
                (getattr(self.feed, "micro", None)._qty.get(symbol)  # type: ignore[union-attr]
                 or []))
        except Exception:
            trade_sizes = []

        # E14/E16 headline stamps: newest live item relevant to this asset
        news_last_ts = 0.0
        news_tier = 1.0
        try:
            for it in self.news._live_items():
                if symbol in (it.get("symbols") or ()) or float(
                        it.get("macro_score") or 0) >= 0.6:
                    _ts = float(it.get("ts") or 0.0)
                    if _ts > news_last_ts:
                        news_last_ts = _ts
                        news_tier = float(1.0 / max(1, int(it.get("tier") or 1)))
        except Exception:
            news_last_ts = 0.0

        try:
            spread_bps = float((self.feed.book_status(symbol) or {})
                               .get("spread_bps") or 0.0)
        except Exception:
            spread_bps = 0.0

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
            "herd_corr": herd_corr,
            "trade_sizes": trade_sizes,
            "news_last_ts": news_last_ts,
            "news_tier": news_tier,
            "spread_bps": spread_bps,
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

    def _spillover_for(self, symbol: str, tf: str, ctx) -> dict:
        """Cross-asset spillover + thesis-signed reversal for one lane.

        Inputs are the lane's own context (per-TF closes/highs/lows/volume)
        plus the LEADER's 1-minute tape, so the six-hour cross-asset term
        is measured in wall-clock hours regardless of the lane's bar size.
        Degrades to `available=False` rather than guessing.
        """
        try:
            from .spillover import score as _sp_score
            from .config import TIMEFRAMES
            bar_min = float((TIMEFRAMES.get(tf) or {}).get("minutes") or 22)
            closes = list(getattr(ctx, "P", None) or [])
            highs = list(getattr(ctx, "H", None) or [])
            lows = list(getattr(ctx, "L", None) or [])
            vols = list(getattr(ctx, "V", None) or [])
            taker = list(getattr(ctx, "TB", None) or [])
            market = None
            if symbol != "BTC":
                try:
                    market = list(self.feed.state("BTC").closes or [])
                except Exception:
                    market = None
            return _sp_score(closes, highs, lows, vols, taker,
                             market_closes=market, market="BTC",
                             bar_minutes=bar_min, market_bar_minutes=1.0)
        except Exception as e:  # never let telemetry break a decision
            return {"available": False, "score": 0.0, "direction": 0,
                    "coverage": 0.0, "parts": {}, "error": str(e)[:120]}

    def _apply_brain4(self, symbol: str, tf: str) -> None:
        """B4 Auto-Learning Controller: omega *= (1 + 0.01 * adj[fid]).

        adj is now the per-formula INPUT-DRIVEN delta of DB-4's DAN blocks
        (the population the input actually reaches). It used to be read off
        DN, which the input never touches, so every adjustment was 0.0 —
        a controller with no output. None = not informative this window.
        """
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
            self._last_raws[(symbol, tf)] = raws[tf]
            self.last_traces[(symbol, tf)] = _traces
            if _anoms:
                self.auditor.report_anomalies(symbol, tf, _anoms, raws[tf])
            # §7 genesis: evaluate admitted candidates on this window
            # (features from the same tape — one shared feature build)
            _gfeat = genesis_mod.build_features(ctx.P, ctx.V)
            # Candidate/probation formulas must be frozen at lock time too;
            # otherwise they never receive walk-forward samples and can
            # never reach the active lifecycle state.
            _gids = self.genesis.walk_forward_ids()
            self._genesis_values[(symbol, tf)] = \
                genesis_mod.evaluate_active(_gfeat, _gids)

        # 2) Bgn synchronization of every registered formula across timeframes
        adj: dict[str, dict[int, float]] = {tf: {} for tf in TF_ORDER}
        forward_probs: dict[str, dict[str, dict[str, Any]]] = {
            tf: {} for tf in TF_ORDER}
        for fid in FORMULA_FUNCTIONS:
            raw_by_tf = {tf: raws[tf].get(fid, 0.0) for tf in TF_ORDER}
            synced = BgnSynchronizer.adjust(matrix, raw_by_tf)
            for tf in TF_ORDER:
                adj[tf][fid] = synced[tf]
                # §2.7: once a same-asset / same-horizon bucket has 30
                # settled readings, use its empirical P(up) − 0.5 in the
                # fused signal. The table is keyed by the frozen raw formula
                # reading, so lookup uses the same value recorded at lock.
                # Until then, the original Bgn-synced formula is unchanged.
                if fid not in WEIGHT_IDS:
                    continue
                try:
                    _raw_fv = float(raws[tf].get(fid, 0.0))
                    if not math.isfinite(_raw_fv):
                        raise ValueError("non-finite formula reading")
                    _p_up, _fw_n = self.forward.p_up(
                        fid, symbol, tf, _raw_fv)
                except (TypeError, ValueError, OverflowError):
                    _p_up, _fw_n = None, 0
                forward_probs[tf][str(fid)] = {
                    "p_up": round(float(_p_up), 4)
                    if _p_up is not None else None,
                    "n": int(_fw_n), "used": _p_up is not None,
                }
                if _p_up is not None:
                    adj[tf][fid] = float(_p_up) - 0.5

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
                        if fid in prev_c and _sign(prev_c[fid]) != 0
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
                    from .physics import _NON_DIRECTIONAL as _p_nondirectional
                    for pid in range(1, 22):
                        try:
                            _pv = float(prev_ph.get(pid, 0.0))
                        except (TypeError, ValueError, OverflowError):
                            continue
                        _ps = 2.0 * _pv - 1.0 if pid in _p_nondirectional else _pv
                        if _sign(_ps):
                            self.hist.ring(self.hist.tf_key(
                                symbol, tf, f"pacc{pid}"), 50).append(
                                1.0 if _sign(_ps) == _sign(move) else -1.0)
                    _ufp_s = _sign(float(prev_ph.get(22, 0.0) or 0.0))
                    if _ufp_s:
                        self.hist.ring(self.hist.tf_key(
                            symbol, tf, "pacc22"), 50).append(
                            1.0 if _ufp_s == _sign(move) else -1.0)
                    self.physics.reward_update(symbol, tf, prev_ph, move,
                                               eta=self.learning.rates["physics"],
                                               scale=_scale)
                # CERC emotion-weight rewards (21 channels, eta rule).
                # Score the PREVIOUS pass, not the values just computed below.
                prev_em = self.prev_emotions.get((symbol, tf))
                if prev_em and move != 0:
                    _pev, _pdirs = prev_em[0], prev_em[1]
                    for eid in range(1, 22):
                        _term = float(_pev.get(eid, 0.0) or 0.0) * int(
                            _pdirs.get(eid, 0) or 0)
                        if eid == 21:
                            _term *= 1.0 - 0.5 * float(_pev.get(eid, 0.0) or 0.0)
                        if _sign(_term):
                            self.hist.ring(self.hist.tf_key(
                                symbol, tf, f"eacc{eid}"), 50).append(
                                1.0 if _sign(_term) == _sign(move) else -1.0)
                    _prev_cerc = (float(prev_em[2]) if len(prev_em) > 2 else
                                  sum(float(_pev.get(eid, 0.0) or 0.0)
                                      * int(_pdirs.get(eid, 0) or 0)
                                      * (1.0 - 0.5 * float(_pev.get(eid, 0.0) or 0.0)
                                         if eid == 21 else 1.0)
                                      for eid in range(1, 22)))
                    if _sign(_prev_cerc):
                        self.hist.ring(self.hist.tf_key(
                            symbol, tf, "eacc22"), 50).append(
                            1.0 if _sign(_prev_cerc) == _sign(move) else -1.0)
                    self.emotions.reward_update(
                        symbol, tf, prev_em[0], prev_em[1], _prev_cerc, move,
                        eta=self.learning.rates["emotion"],
                        scale=_scale,
                    )
                # brain.md §6/§7: the connectome vote's own outcome ring
                # (feeds hindsight) and dopamine RPE — boundaries only
                fwv = self._fw_last_vote.get((symbol, tf), 0.0)
                fw_dir = (1 if fwv > 1e-6 else
                          (-1 if fwv < -1e-6 else 0))
                if fw_dir:
                    self.hist.ring(
                        self.hist.tf_key(symbol, tf, "facc_brain"),
                        50).append(1.0 if fw_dir == _sign(move) else -1.0)
                    if self.brains.fw.state == "ready" and move != 0:
                        entry = self.brains.fw.rpe_update(
                            vote=fwv, direction=fw_dir,
                            actual_move=move, horizon=tf)
                        if entry is not None:
                            active = self.brains.fw._last_active_kc
                            if active is not None and len(active):
                                signs = np.where(
                                    self.brains.fw._mbon_approach,
                                    1.0, -1.0).astype(np.float32)
                                self.brains.fw.apply_rpe_weights(
                                    active,
                                    np.arange(len(signs), dtype=np.int64),
                                    signs)
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
                # calibration rings (paired): this window's LOCKED
                # confidence bucket + its realised strict-sign hit — the
                # calibration panel plots hit-rate vs stated confidence
                _cb = self._calib_pending.pop((symbol, tf), None)
                pdir = self.prev_cci_dir.get((symbol, tf), 0)
                if (move != 0 and pdir != 0 and _cb is not None and _cb > 0
                        and calib_should_record(self._calib_done,
                                                (symbol, tf), move)):
                    _bucket = round(min(1.0, max(0.0, _cb)) * 20) / 20.0
                    self.hist.ring(
                        self.hist.tf_key(symbol, tf, "calib_p"), 400
                    ).append(_bucket)
                    self.hist.ring(
                        self.hist.tf_key(symbol, tf, "calib_h"), 400
                    ).append(1.0 if pdir == _sign(move) else 0.0)
                # exchange-physics direction ring: net forced-flow sign
                # graded against the realised move (hindsight 'micro')
                try:
                    _mfl = self.feed.micro_flow(symbol)
                    if move != 0 and _mfl is not None:
                        _mdir = (1 if _mfl > 0.05 else
                                 (-1 if _mfl < -0.05 else 0))
                        if _mdir:
                            self.hist.ring(
                                self.hist.tf_key(symbol, tf,
                                                 "facc_micro"), 50
                            ).append(1.0 if _mdir == _sign(move) else -1.0)
                except Exception:
                    pass
                # consecutive-correct running ratio (0..1, last 10)
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
                advance=record, raws=raws[tf],
            )
            # brain.md: remember the connectome's vote for fusion + RPE
            self._fw_last_vote[(symbol, tf)] = float(
                (brain_res[tf].get("flywire") or {}).get("vote") or 0.0)
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
        tracer_raw: dict[str, float] = {}
        tracer_detail: dict = {}

        def _recompute(g: FormulaContext) -> dict[int, float]:
            # P19 gauge recomputes: full formula pass on transformed prices,
            # record=False so the GRC-cadence rings stay untouched
            return compute_all(g, record=False)

        _mandate_comps: dict = {}
        _mandate_dc: dict = {}
        for tf in TF_ORDER:
            drhe = adj[tf].get(20, 1.0)
            # Hedge-Flow Tracer (id102): real-time cross-asset rotation
            # effect — measured volume-share delta + relative return
            from .tracer import hedge_flow_tracer
            try:
                tracer_raw[tf], tracer_detail = hedge_flow_tracer(
                    self.feed, symbol)
            except Exception:
                tracer_raw[tf], tracer_detail = 0.0, {"reason": "tracer error"}
            self._tracer_last[(symbol, tf)] = tracer_raw[tf]
            comp_core: dict[int, float] = {fid: adj[tf][fid] for fid in FORMULA_FUNCTIONS}
            comp_core[BRAIN_COMPONENT_ID] = brain_sync[tf]
            comp_core[EMOTION_COMPONENT_ID] = cerc_sync[tf]
            comp_core[TRACER_COMPONENT_ID] = tracer_raw[tf]
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
                "tracer": round(tracer_raw.get(tf, 0.0), 6),
                "tracer_detail": tracer_detail,
            }

        # 7b) Bgn synchronization of the part-3 signals across timeframes
        physics_sync = BgnSynchronizer.adjust(
            matrix, {tf: phys_raw[tf]["ufp"]["value"] for tf in TF_ORDER}
        )
        hedge_sync = BgnSynchronizer.adjust(matrix, hedge_raw)
        news_sync = BgnSynchronizer.adjust(
            matrix, {tf: news_agg["value"] for tf in TF_ORDER}
        )

        # 7) Formula 21 per timeframe: 35 weighted components + confidence
        for tf in TF_ORDER:
            atr_tf = self._atr(ctxs[tf])        # per-tf ATR (fixes leak from step 5)
            drhe = adj[tf].get(20, 1.0)
            comp: dict[int, float] = {fid: adj[tf][fid] for fid in FORMULA_FUNCTIONS}
            comp[BRAIN_COMPONENT_ID] = brain_sync[tf]
            comp[EMOTION_COMPONENT_ID] = cerc_sync[tf]
            comp[PHYSICS_COMPONENT_ID] = physics_sync[tf]
            comp[NEWS_COMPONENT_ID] = news_sync[tf]
            comp[HEDGE_COMPONENT_ID] = hedge_sync[tf]
            comp[TRACER_COMPONENT_ID] = tracer_raw.get(tf, 0.0)
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
            # RL DEEPENING: Q-table picks the subsystem mix for this
            # window's measured state; the contextual bandit context is
            # remembered so the issue threshold + settlement scoring
            # use the same context the prediction was made under.
            _rl_state, _rl_ctx = self._rl_window_state(symbol, tf)
            _rl_action = self.rl.choose_mix(_rl_state, f"{symbol}|{tf}")
            self._rl_state[(symbol, tf)] = {
                "state": _rl_state, "action": _rl_action, "ctx": _rl_ctx}
            pack = self.cci.compute(
                symbol, tf, comp, drhe, tme=phys["tme"],
                shares=self.rl.mix_shares(
                    damp_shares(self.learning.subsystem_shares), _rl_action),
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
            # CONFIDENCE CALIBRATION (precision fix): the raw chain
            # |CCI| x DRHE x penalties lands at ~0.001-0.03 in normal
            # markets, so the 0.35 floor demanded near-unanimous
            # extremes and NOTHING could ever issue ("no prediction
            # for a day"). Confidence is now the MEASURED percentile
            # of this score within its own recent history for this
            # asset+TF: 0.35 means "stronger than ~65% of its own
            # recent scores" — reachable when the signal is relatively
            # strong, still gated by every check below. Until 50
            # samples exist the raw value is used (legacy behaviour).
            _hist = self._conf_hist.setdefault((symbol, tf), [])
            _base_conf = float(pack["confidence"])
            fin["confidence_raw"] = round(float(fin["confidence"]), 5)
            # percentile pool: the ladder's own |score| history when it
            # has >= 20 entries (it accumulates from the first GRC),
            # else this run's base-confidence history. Either way the
            # number is a measured rank, not a rescaled guess.
            _pool = [abs(v) for v in
                     self.score_hist.get((symbol, tf), [])
                     if v is not None]
            try:
                _ring_pool = [abs(v) for v in self.hist.ring(
                    self.hist.tf_key(symbol, tf, "score_abs"), 200)]
                if len(_ring_pool) > len(_pool):
                    _pool = _ring_pool
            except Exception:
                pass
            _rank_val = abs(float(pack["cci_adjusted"]))
            if len(_pool) < 8:
                _pool = list(_hist)
                _rank_val = _base_conf
            # Percentile of this score inside its own history — kept as a
            # DIAGNOSTIC only. It used to be the displayed confidence,
            # which meant ANALYSING printed a rank (0.285) while the
            # CONFIDENCE CHAIN printed the raw product (0.0002): two
            # formulas, two numbers, same window. (audit §2.3)
            if len(_pool) >= 8:
                _pct = sum(1 for v in _pool if v <= _rank_val) / len(_pool)
                fin["conf_percentile"] = round(_pct, 4)
            else:
                fin["conf_percentile"] = None
            fin["conf_hist_n"] = len(_pool)
            # ── CALIBRATED CONFIDENCE (audit §2.1/§2.2) ──────────────
            # ONE number, computed once, displayed everywhere. Base is an
            # SNR→tanh mapping of |CCI_adj| over its measured noise floor;
            # every modulator is bounded, so six dampers can no longer
            # multiply a real setup into 0.0002 and jam RECHECK at
            # NO TRADE forever.
            _cal = calibrate_confidence(
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
            fin["confidence"] = _cal["confidence"]
            fin["confidence_base_snr"] = _cal["base_confidence"]
            fin["confidence_modulator"] = _cal["modulator"]
            fin["confidence_chain"] = _cal["steps"]
            fin["snr"] = _cal["snr"]
            fin["noise_floor"] = _cal["noise_floor"]
            fin["issued"] = _cal["issued"]
            # The INPUTS of the chain, published with it. A later stage
            # (the agent/brain fusion at the GRC) rescales `cci_adjusted`
            # and must be able to re-run this exact calibration on the
            # fused number instead of leaving the published confidence
            # describing a stale CCI. Without these, the recompute would
            # silently use defaults and invent a different chain.
            fin["fusion_inputs"] = {
                "experience_quality": br["experience"],
                "p_dominant": br["p_dominant"],
                "state_aligned": bool(aligned),
                "rtp": em["rtp"], "pcv": pcv,
                "wfcp": phys["wfcp"], "drpt": phys["drpt"],
                "gspi": phys["gspi"],
            }
            fin["_pack"] = {"cci_adjusted": float(pack["cci_adjusted"]),
                            "drhe": float(pack.get("drhe", 1.0))}
            fin["strength"] = pack.get("strength", 0.0)
            _wr, _wrn = self._measured_winrate(
                symbol, tf, float(fin["confidence"]))
            fin["winrate"] = _wr
            fin["winrate_n"] = _wrn
            _hist.append(_base_conf)
            if len(_hist) > 400:
                del _hist[:-400]
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
            # exchange-physics micro band (Chrono-Sieve-inspired gauges):
            # net forced-flow share ±10 % on confidence, SAME doctrine as
            # the emotion band — never the direction; silent (0) whenever
            # the tape has no live microstructure (sim)
            try:
                _mflow = self.feed.micro_flow(symbol)
                if _mflow:
                    _mb = 0.10 * float(_mflow) * _dir0
                    if _mb:
                        fin = dict(fin)
                        fin["confidence"] = max(0.0, min(
                            1.0, float(fin["confidence"]) * (1.0 + _mb)))
                        fin["issued"] = fin["confidence"] > CONFIDENCE_FLOOR
                        fin["micro_band"] = round(_mb, 4)
            except Exception:
                pass

            # CALIBRATION shrinkage: once enough settled pairs exist,
            # scale confidence by realised/stated hit-rate (clipped —
            # a learned multiplier, same family as hindsight; silent
            # below n=40 or when stated confidence is noise)
            try:
                cf = self.calibration_factor()
                if cf is not None and abs(cf - 1.0) > 0.01:
                    fin = dict(fin)
                    fin["confidence"] = max(0.0, min(
                        1.0, float(fin["confidence"]) * cf))
                    fin["issued"] = fin["confidence"] > CONFIDENCE_FLOOR
                    fin["calibration_factor"] = round(cf, 3)
            except Exception:
                pass

            # RL ISSUE BANDIT: learned confidence threshold (fresh state
            # =0.35 → identical to the legacy floor until data moves it);
            # a context with its own settled history speaks for itself.
            try:
                fin["issued"] = (float(fin["confidence"])
                                 > self.rl.issue_threshold(
                                     self._rl_state.get((symbol, tf), {})
                                     .get("ctx")))
            except Exception:
                pass

            # PER-ASSET GATE: gold is not BTC — the asset's nature sets
            # its own issue bar (can only tighten the decision here;
            # BNB's steadiness is expressed via the RL bandit learning
            # on top of a slightly lower stated prior). Carried in the
            # payload so every panel can show which bar applied.
            _prof = asset_profile(symbol)
            # The asset may nudge the design floor, not replace it: an
            # uncapped 1.20 multiplier put ETH's bar at 0.42 while the
            # calibrated confidence distribution lives in 0.15–0.45, so
            # nothing could ever issue (the "always NO TRADE" report).
            # §18.2.5: clamp keeps the asset prior from starving a
            # high-bar profile; ISSUE_BAR raises every bar uniformly
            from .config import asset_issue_floor as _issue_floor
            fin["asset_floor"] = _issue_floor(symbol)
            fin["asset_nature"] = _prof["nature"]
            if float(fin["confidence"]) <= fin["asset_floor"]:
                fin["issued"] = False

            # PER-ASSET PREDICTION BEHAVIOUR (asset_behavior.py): the
            # asset's nature tempers the SIGNAL ITSELF — gold amplified
            # in trends / damped in chop, ETH damped in vol blow-offs,
            # BNB pulled toward reversion when stretched. Direction
            # speaks only above the asset's noise gate; tempering can
            # veto an issue, never create one the RL gate refused.
            from .asset_behavior import temper as _temper
            # CONFIDENCE OWNERSHIP (live defect 2026-11-27)
            # ------------------------------------------------
            # The card printed confidence 0.021 while the CONFIDENCE CHAIN
            # — regenerated from the same inputs — printed a total of 0.421.
            # A 20x disagreement is not rounding: `temper()` returns the
            # confidence UNCHANGED for an asset with no tempering rule (BTC
            # has none), and the old line assigned that return value back
            # over the calibrated number. `fin["confidence"]` at this point
            # was whatever the earlier chain left behind, so the calibrated
            # 0.42 was overwritten by a stale 0.02 for exactly the one asset
            # with nothing to say. calibrate_confidence is the single owner
            # of this number; temper may only SCALE it when it states a
            # factor.
            _calibrated_conf = float(fin.get("confidence") or 0.0)
            _beh = _temper(symbol, pack["cci_adjusted"],
                           _calibrated_conf,
                           list(self.feed.state(symbol).closes or []))
            _beh_c = float(_beh.get("confidence") or 0.0)
            if abs(_beh_c - _calibrated_conf) > 1e-9:
                fin["behavior_conf"] = {
                    "before": round(_calibrated_conf, 4),
                    "after": round(_beh_c, 4),
                    "factor": _beh.get("factor"),
                }
            else:
                # nothing was changed: keep the calibrated number and the
                # record of the calibration instead of a verbatim copy
                _beh = dict(_beh)
                _beh["confidence"] = _calibrated_conf
            fin["behavior"] = _beh
            # ── CROSS-ASSET SPILLOVER + THESIS-SIGNED REVERSAL ────────
            # Measured, not assumed: on nine months of real Binance bars
            # the momentum tilt scored IC −0.036 and 48.0% sign accuracy
            # out-of-fold; the same readings taken contrarian — plus the
            # leader's last six hours — scored +0.033 and 51.4%
            # (backtest/results/ALPHA_LAB.md). The block is blended into
            # the fused score BEFORE the side is derived so the card, the
            # chart and the contract cannot disagree about it.
            try:
                _sp = self._spillover_for(symbol, tf, ctxs[tf])
            except Exception:
                _sp = {"available": False, "score": 0.0, "direction": 0,
                       "coverage": 0.0, "parts": {}}
            fin["spillover"] = _sp
            _sp_w = 0.25 * float(_sp.get("coverage") or 0.0)
            if _sp.get("available") and _sp_w > 0:
                _beh = dict(_beh)
                _beh["score"] = ((1.0 - _sp_w) * float(_beh["score"])
                                 + _sp_w * float(_sp["score"]))
                _beh["spillover_weight"] = round(_sp_w, 3)
                fin["behavior"] = _beh
            _gate = 0.02 * float(_prof.get("noise_mult") or 1.0)
            _dir_final = (1 if _beh["score"] > 0 else -1) \
                if abs(_beh["score"]) >= _gate else 0
            # ── NO MORE FLAT CARDS (2026-11-26) ──────────────────────────
            # A fused score inside the noise gate used to yield direction 0,
            # i.e. NO TRADE — an abstention, and the one the user encounters
            # most often because scores are routinely ~0.01-0.03 against a
            # 0.02 x asset-noise gate. "Continuous, without missing any
            # minute" means every window must ship a side; the question is
            # only WHICH side is the better bet when the fused score is not
            # informative. That is answerable from the record instead of
            # guessed:
            #   * the 2-decision trend (app/trend_gate.py) carries a measured
            #     edge of its own — fading it after a chase is worth 56.8 %
            #     while FOLLOWING it as a raw side is only 43.2 %
            #     (`backtest.trend_gate_audit`), so the informative side is
            #     the fade of a chase;
            #   * when even the trend has no reading (fresh boot, no
            #     history), there is nothing measured to lean on and the
            #     engine still must trade, so it takes the tape's own
            #     last-closed-bar direction — the classic, documented
            #     baseline — and SAYS it is uninformative.
            # Both branches are labelled in the payload: a side that came
            # from this fallback must never look like a measured edge.
            if _dir_final == 0:
                _fb = ""
                try:
                    from .trend_gate import trend as _tg_trend
                    _tr = int(_tg_trend(symbol, tf) or 0)
                except Exception:
                    _tr = 0
                if _tr in (-1, 1):
                    # fade the impulse: the measured side of the record
                    _dir_final = -_tr
                    _fb = ("trend-fade (score inside the noise gate; the "
                           "2-decision impulse is the only measured reading "
                           "left, and the record says fade it)")
                else:
                    try:
                        _cl = list(self.feed.state(symbol).closes or [])
                        _last = (1 if _cl[-1] >= _cl[-2] else -1) \
                            if len(_cl) >= 2 else 1
                    except Exception:
                        _last = 1
                    _dir_final = _last
                    _fb = ("last-bar direction (score inside the noise gate "
                           "and no trend reading yet — UNINFORMATIVE: this is "
                           "a coin flip shipped so the window is not skipped)")
                fin["direction_source"] = "noise-gate fallback"
                fin["direction_fallback"] = _fb
            # LOCK HYSTERESIS (the "analysis and chart are not in sync"
            # report): the fused score is routinely 0.03 against a 0.02
            # floor, so its SIGN flips on noise. The window's locked side
            # is drawn on the chart, priced by the contract and graded in
            # the ledger — the analysis must not silently call the opposite
            # of it a minute later. Hold the locked side unless the fresh
            # score is decisively opposite (1.5x the noise gate).
            _cyc_live = self.cycles.get((symbol, tf))
            _locked_dir = int(getattr(_cyc_live, "direction", 0) or 0) \
                if _cyc_live is not None else 0
            fin["locked_direction"] = _locked_dir
            fin["fresh_direction"] = _dir_final
            if _locked_dir and _dir_final and _dir_final != _locked_dir \
                    and abs(float(_beh["score"])) < 1.5 * _gate:
                _dir_final = _locked_dir
                fin["direction_hold"] = (
                    "fresh score opposes the lock but only by "
                    f"{abs(float(_beh['score'])):.4f} < {1.5 * _gate:.4f} "
                    "— locked side held for this window")
            fin["direction_conflict"] = bool(
                _locked_dir and _dir_final and _dir_final != _locked_dir)
            self._last_dirs[(symbol, tf)] = (time.time(),
                                             int(_dir_final or 0))
            fin["confidence"] = _beh["confidence"]
            # ── THE CONFIDENCE TRACE (user report 2026-11-27: "why their is
            # n/a? In explain this why confidence is too low") ────────────
            # Every stage from here to the published number appends
            # (stage, before, after). The card renders it verbatim, so a
            # number that drops by 20x can name the stage that dropped it
            # instead of leaving the reader to guess. The list is built even
            # when nothing moves, so "0.42 in, 0.42 out" is also auditable.
            _ctrace: list[dict] = []

            def _conf_stage(name: str, before: float, after: float,
                            why: str = "") -> None:
                if abs(float(after) - float(before)) > 1e-9:
                    _ctrace.append({"stage": name,
                                    "before": round(float(before), 4),
                                    "after": round(float(after), 4),
                                    "why": why})

            _ctrace.append({"stage": "calibrated chain",
                            "before": None,
                            "after": round(float(fin["confidence"]), 4),
                            "why": "SNR→tanh base × bounded modulators"})
            fin["conf_trace"] = _ctrace
            # docs11-doublecheck §3: EVERY window gets a side. The old
            # chain (RL bandit -> asset floor -> noise gate) fused three
            # different units into one veto and silenced the app for
            # days. It survives only as exec_ok (trader/size gate);
            # `issued` now means "a side was issued", which is always.
            # MOMENTUM-CHASE VETO (2026-11-09, REAL_RESULTS.md §4):
            # on the native 1m tape, windows whose issued side CHASES
            # the window's own momentum by >1 vol-scaled SD are the
            # measured losers (past-W momentum sign accuracy 47.6%
            # pooled, all 5 folds <50% — this market fades the chase
            # at 22–247m horizons). The side is still ISSUED (every
            # window gets a side); only EXECUTION is gated to
            # watch-only, exactly like the <1:1 ratio rule.
            _chase_z = 0.0
            try:
                _P = ctxs[tf].P
                if len(_P) > 8 and _P[0]:
                    _mom = _P[-1] / _P[0] - 1.0
                    _rets = [(_P[i] - _P[i - 1]) / _P[i - 1]
                             for i in range(1, len(_P)) if _P[i - 1]]
                    _sd = _std_pop(_rets) if _rets else 0.0
                    if _sd > 0:
                        _chase_z = (_dir_final * _mom
                                    / (_sd * (len(_P) ** 0.5)))
            except Exception:
                _chase_z = 0.0
            fin["mom_chase_z"] = round(float(_chase_z), 3)
            fin["exec_ok"] = bool(fin.get("issued") and _dir_final != 0
                                  and fin["confidence"] > fin["asset_floor"]
                                  and _chase_z <= 1.0)
            fin["issued"] = True
            # DISPLAY GATE: `issued` stays True by contract (every window
            # gets a side), but no panel may PRINT "ISSUED" for a side that
            # never cleared the floor — that is exactly how the card said
            # ISSUED at 0.176 while ANALYSE simultaneously said NO TRADE.
            try:
                fin["gate_ok"] = bool(_dir_final != 0 and
                                      float(fin["confidence"])
                                      > float(fin["asset_floor"]))
            except (TypeError, ValueError):
                fin["gate_ok"] = False
            fin["issue_note"] = (
                f"cleared the {float(fin['asset_floor']):.3f} floor"
                if fin["gate_ok"] else
                f"below floor ({float(fin.get('confidence') or 0):.3f} "
                f"<= {float(fin['asset_floor']):.3f})")
            # ADVANCED MATHEMATICS LAYER (user-approved items 3-7, 9,
            # 11-16, 20-21): additional voters blended into the fused
            # score (nothing existing is removed or muted), soft
            # gates + RQA scaling confidence, HAR/EVT/RMT sizing.
            _adv = self._advanced_for(symbol, tf, ctxs[tf])
            fin["advanced_vote"] = _adv["vote"]
            fin["adv_soft_gate"] = _adv["soft_gate"]
            # BOUNDED LAYER (fixed 2026-11-26): the advanced block used to
            # multiply confidence by soft_gate × rqa_mult with no floor of its
            # own. On a live sim tape both landed near their lower bounds
            # (gate 0.45, RQA 0.75 ⇒ ×0.34) *after* calibrate_confidence had
            # already produced a floor-clearing 0.27 — so every lane printed
            # confidence 0.009 against a 0.366 floor and no call could ever
            # issue. A calibration must stay inside its documented band; a
            # second unbounded product is exactly the failure
            # calibrate_confidence's docstring says it removed.
            # Both factors are now bounded INSIDE app/advanced.py —
            # soft_gate in [0.85, 1.0] (mean of its three sub-gates, not
            # their triple product) and RQA in [1.0, 1.15] (calibrated
            # against the estimator's own null) — so the product lives in
            # [0.85, 1.15] on its own. The old 0.70 floor chosen here was
            # treating the symptom: it guaranteed a constant −30 % haircut
            # on every window instead of fixing the component that was
            # pinned at its lower bound. A bounding box around a broken
            # measurement is still a broken measurement — the numbers are
            # fixed at their source now, and this clamp only guards against
            # a future regression in either factor.
            _adv_mult = max(0.60,
                            min(1.35, float(_adv["soft_gate"] or 1.0)
                                * float(_adv["rqa_mult"] or 1.0)))
            fin["adv_conf_mult"] = round(_adv_mult, 4)
            fin["adv_conf_mult_rule"] = (
                f"soft_gate {float(_adv['soft_gate'] or 1.0):.3f} × RQA "
                f"{float(_adv['rqa_mult'] or 1.0):.3f} = {_adv_mult:.3f} "
                "(both factors measured in app/advanced.py; this is a "
                "regression guard, not the calibration)")
            _c_before_adv = float(fin["confidence"])
            fin["confidence"] = max(0.0, min(1.0,
                float(fin["confidence"]) * _adv_mult))
            _conf_stage("advanced layer ×", _c_before_adv,
                        float(fin["confidence"]),
                        f"soft_gate × RQA = {_adv_mult:.3f}")
            _beh = dict(_beh)
            _adv_k = self._adv_blend_k(symbol, tf)
            fin["adv_blend_k"] = round(_adv_k, 4)
            _beh["score"] = float(_beh["score"]) + _adv_k * _adv["vote"]
            _tier = self._tier_for(
                symbol, tf, float(_beh["score"]),
                strength=pack.get("strength"),
                direction=int(_dir_final or 0),
                dirs_by_tf=self._sibling_dirs(symbol, tf),
                confidence=float(fin["confidence"]))
            fin.update(_tier)
            fin["raw_score"] = round(float(_beh["score"]), 5)
            # ── volatility-scaled barriers (app/barriers.py) ──────────
            # Geometry, not just direction: a target thinner than 3x the
            # round-trip friction cannot be traded. Published so a
            # watch-only window can say why.
            # (Merge fix: the f07a line read a bare `atr` that was never
            # bound in this scope, so every block below died with a
            # NameError into its except-leg and the telemetry was always
            # None. Bind the lane ATR once, honestly.)
            try:
                _atr_lane = float(self._atr(ctxs[tf]) or 0.0)
            except Exception:
                _atr_lane = 0.0
            try:
                from .barriers import vol_scaled as _vol_barriers
                from .horizon import asset_rt as _asset_rt
                # §18.1: the real Binance round trip (28.9–37.4 bps) —
                # every friction check below charges the SAME number
                _hz_rt = _asset_rt(symbol)
                _px = float(price or 0.0)
                _atr = _atr_lane
                _sig_bps = (_atr / _px * 1e4) if (_px > 0 and _atr > 0) else None
                fin["vol_barriers"] = _vol_barriers(
                    _sig_bps, tf=tf, friction_bps=_hz_rt)
            except Exception:
                fin["vol_barriers"] = None
            # ── MICROSTRUCTURE + HORIZON VIABILITY (deep overhaul) ────
            # OFI / VPIN / Kyle λ / size skew / Hawkes from the lane's own
            # bars, plus the honest question the deep walk-forward raised:
            # does this horizon's σ clear 3x the round trip at all?
            try:
                from .micro_alpha import features as _micro_feats
                from .horizon import classify as _lane_state
                from types import SimpleNamespace as _SNS
                _mb = ctxs[tf]
                # micro_alpha reads lowercase bar fields (c/o/h/l/v, ...);
                # the live context carries P/V/O/H/L. Adapt honestly: cash
                # series pass through, quote volume is close×volume, and
                # taker/trade-count fields stay absent (those readings
                # report None instead of inventing flow).
                _mP = list(getattr(_mb, "P", []) or [])
                _mV = list(getattr(_mb, "V", []) or [])
                _mO = list(getattr(_mb, "O", []) or []) or list(_mP)
                _mH = list(getattr(_mb, "H", []) or []) or list(_mP)
                _mL = list(getattr(_mb, "L", []) or []) or list(_mP)
                _madapt = _SNS(o=_mO, h=_mH, l=_mL, c=_mP, v=_mV,
                               quote_v=[p * v for p, v in zip(_mP, _mV)])
                _mic = _micro_feats(_madapt, len(_mP) - 1)
                fin["micro"] = {k: (None if (v != v) else round(float(v), 4))
                                for k, v in _mic.items()}
                _sig_bps = ((_atr_lane / float(price or 1.0) * 1e4)
                            if price else None)
                _fr = _mic.get("friction_bps")
                # §18.1: the lane gate charges the CERTIFIED round
                # trip (the micro λ stays informational on the card)
                from .horizon import asset_rt as _asset_rt2
                fin["horizon"] = _lane_state(_sig_bps, _asset_rt2(symbol))
                # ── COST GATE: does the *measured* edge clear friction? ──
                # The horizon check asks whether σ is wide enough to be
                # worth trading at all; this asks whether the out-of-fold
                # gross expectancy on record actually pays the round trip.
                # Directional cards may still print, but a blocked lane is
                # never presented as tradable telemetry.
                from .cost_gate import bucket as _edge_bucket, gate as _edge_gate
                # (Merge fix: the f07a map keyed "TF1".. while lanes are
                # "tf1".. so every lane read 60.0. Use the real window.)
                _hz_min = float((TIMEFRAMES.get(tf) or {}).get("minutes")
                                or 22.0)
                # §18.1: the cost gate charges the certified Binance
                # round trip (micro λ is shown, never billed)
                from .horizon import asset_rt as _asset_rt3
                _fric = _asset_rt3(symbol)
                fin["edge"] = _edge_gate(symbol, _edge_bucket(_hz_min), _fric)
                # ── EVENT SLEEVES (multi-year derivatives programme) ──
                # Measured out-of-fold sleeves that trade this asset, with
                # their verdict. `unproven` is a first-class state: positive
                # net expectancy whose bootstrap t is still inside the noise.
                # (Merge: the two fourth-pass positives were retracted per
                # ALPHA_RESEARCH §0 and gate as `blocked`, so the UI — which
                # filters `blocked` — no longer shows them as evidence.
                # Certified sleeves come from app/edge_gate.py instead.)
                from .cost_gate import gates_for as _gates_for
                _sv = _gates_for(symbol)
                fin["sleeves"] = _sv or None
            except Exception as _e:
                fin["micro"] = None
                fin["horizon"] = {"state": "unknown",
                                  "note": f"micro unavailable: {str(_e)[:80]}"}
                fin["edge"] = {"state": "unknown",
                               "note": f"edge gate unavailable: {str(_e)[:80]}"}
                fin["sleeves"] = None
            # RE-JUDGE THE GATE on the FINAL confidence: the advanced layer
            # multiplies confidence by its soft gate AFTER the first check,
            # so the card could print 0.349 "ISSUED" while RECHECK compared
            # 0.349 to a 0.350 floor and failed the same check. One number,
            # one verdict.
            try:
                fin["gate_ok"] = bool(_dir_final != 0 and
                                      float(fin["confidence"])
                                      > float(fin["asset_floor"]))
                # 3-dp, the same rounding the card prints confidence with
                fin["issue_note"] = (
                    f"cleared the {float(fin['asset_floor']):.3f} floor"
                    if fin["gate_ok"] else
                    f"below floor ({float(fin['confidence']):.3f} "
                    f"<= {float(fin['asset_floor']):.3f})")
            except (TypeError, ValueError, KeyError):
                fin["gate_ok"] = False
            # ── HONEST CONFIDENCE (app/calib.py) ─────────────────────
            # The card prints a score magnitude; this is the measured
            # probability that the call wins, from the isotonic map fitted
            # on settled ledger rows. Reported as None until the lane (or
            # the pooled map) has MIN_SAMPLES, never faked with a formula.
            try:
                _cv = self.calib.apply(symbol, tf, fin.get("confidence"))
                fin["p_correct"] = _cv.get("p")
                fin["calib_iso"] = _cv
            except Exception:
                fin["p_correct"] = None
                fin["calib_iso"] = {"source": "error", "p": None, "n": 0}
            pack = dict(pack)
            pack["direction"] = _dir_final
            pack["cci_adjusted_raw"] = pack["cci_adjusted"]
            pack["cci_adjusted"] = _beh["score"]

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
                "backend": br.get("backend", "synthetic-graph"),
                # brain.md §9: every prediction card carries the REAL
                # circuit's vote + its confidence (absent = null/0)
                "flywire": br.get("flywire"),
                "values": {i["id"]: round(i.get("signal", 0.0), 5)
                           for i in br["instances"]},
                "instances": [
                    {k: (round(v, 5) if isinstance(v, float) else
                         ({kk: round(vv, 5) if isinstance(vv, float) else vv
                           for kk, vv in v.items()} if isinstance(v, dict) else v))
                     for k, v in inst.items() if k != "pattern_act"}
                    for inst in br["instances"]
                ],
                "hormones_avg": {k: round(v, 5) for k, v in br["hormones_avg"].items()},
                "dominant": br["dominant"],
                "p_dominant": br["p_dominant"],
                "state_dist": br["state_dist"],
                "state_confident": br.get("state_confident"),
                "experience": round(br["experience"], 5),
                "voters": br.get("voters"),
                "abstained": br.get("abstained") or [],
                "spread": br.get("spread"),
                "agreement_count": br.get("agreement_count"),
                "votes": br.get("votes"),
                # what the controller wanted to change, and why it could not
                "weight_adj": br.get("weight_adj"),
                "weight_adj_reason": br.get("weight_adj_reason"),
            }
            emotion_agg = {
                "aggregate": cerc_sync[tf],
                "polarity": 1 if cerc_sync[tf] > 1e-6 else (
                    -1 if cerc_sync[tf] < -1e-6 else 0
                ),
                "cerc": em["cerc"],
                # E22 two extra, clearly-labelled readings: the signed
                # weighted MEAN of the channels that express a direction,
                # plus how much of the emotional mass has a direction and
                # the bull/bear channel breadth. The spec's raw CERC
                # agreement sum stays in `cerc` untouched.
                "net_emotion": em.get("net_emotion", 0.0),
                "agreement": em.get("agreement", 0.0),
                "breadth": em.get("breadth"),
                "dominant": em.get("dominant"),
                "retail": em.get("retail"),
                "hedge": em.get("hedge"),
                "emotions": em["emotions"],
                "directions": em["directions"],
                "rtp": em["rtp"],
                "tedr": em["tedr"],
                # rebuilt model (see app/emotion.py): every block carries its
                # own citation so the panel can print what it is showing
                "model": em.get("model"),
                "core_affect": em.get("core_affect"),
                "pad": em.get("pad"),
                "panas": em.get("panas"),
                "ekman": em.get("ekman"),
                "primary_emotion": em.get("primary_emotion"),
                "fear_greed": em.get("fear_greed"),
                "action_tendency": em.get("action_tendency"),
                "risk": em.get("risk"),
                # advice §5: measured features + print accounting (D-7)
                "features": em.get("features") or {},
                "zero_reasons": em.get("zero_reasons") or {},
                        "zero_causes": em.get("zero_causes") or {},
                "prints": {"quote_prints": self.feed.quote_prints,
                           "note": "quote prints excluded from volume/side stats"},
            }
            # Section 22: human-readable reasoning, not just numbers
            from .emotion import CHANNELS as _ECH
            _neutral = {i: _ECH[i][3] for i in _ECH}
            _top_e = sorted(em["emotions"].items(),
                            key=lambda kv: -abs(kv[1] - _neutral[int(kv[0])]))[:4]
            _eparts = [
                f"{_ECH[int(eid)][1]} {v:.2f}"
                f"{'▲' if em['directions'].get(int(eid), 0) > 0 else '▼' if em['directions'].get(int(eid), 0) < 0 else '·'}"
                for eid, v in _top_e
                if abs(v - _neutral[int(eid)]) > 0.02
            ]
            _pe = em.get("primary_emotion") or {}
            _fg = em.get("fear_greed") or {}
            _at = em.get("action_tendency") or {}
            _ca = em.get("core_affect") or {}
            emotion_agg["narrative"] = (
                (f"Feels {_pe.get('name', '?')} "
                 f"({100 * float(_pe.get('intensity') or 0):.0f}% prototype "
                 f"match, runner-up "
                 f"{( _pe.get('runner_up') or {}).get('name', '—')}); "
                 f"valence {float(_ca.get('valence') or 0):+.2f}, "
                 f"arousal {float(_ca.get('arousal') or 0):.2f}; "
                 f"fear & greed {_fg.get('index', '—')} "
                 f"({_fg.get('label', '—')}, "
                 f"{round(100 * float(_fg.get('measured_weight') or 0))}% of "
                 f"the published weighting measured); action tendency "
                 f"{_at.get('name', '—')}. ") if _pe else ""
            ) + (
                ("Strongest channels: " + "; ".join(_eparts) + ". ") if _eparts else
                "All 21 channels sit at their neutral reading. "
            ) + (
                f"Net emotion [{em.get('net_emotion', 0.0):+.3f}] "
                f"({em.get('breadth', {}).get('bull', 0)}▲/"
                f"{em.get('breadth', {}).get('bear', 0)}▼, "
                f"{round(100 * em.get('agreement', 0.0))}% directional) · "
                f"CERC agreement sum {em['cerc']:+.4f} — " + (
                    "bulls in control." if em.get("net_emotion", 0.0) > 0.15 else
                    "bears in control." if em.get("net_emotion", 0.0) < -0.15 else
                    "balanced tug-of-war (conflict keeps confidence low)."
                )
            ) + (f" Trauma index RTP {em['rtp']:.2f} > 0.6 — confidence suppressed."
                 if em["rtp"] > 0.6 else "")

            # §18.2.5 CERTAINTY: what the user sees next to the stated
            # confidence is the certified cell's Wilson lower bound on
            # its MEASURED win rate (with n) — a statistic, never a
            # formula score.  Absent until the cell is certified.
            _cell_cert: dict = {}
            try:
                from . import hold_policy as _hp_c
                _hs = _hp_c.summary(symbol, tf)
                if _hs.get("status") == "ok":
                    _wil = _hs.get("win_rate_wilson") or [None, None]
                    _cell_cert = {
                        "cell_certainty": _wil[0],
                        "cell_certainty_n": _hs.get("n_windows"),
                        "cell_win_rate": _hs.get("win_rate"),
                        "cell_win_rate_label": _hs.get("win_rate_label"),
                        "cell_win_rate_net": _hs.get("win_rate_net"),
                        "cell_round_trip_bps": _hs.get("round_trip_bps"),
                        # the target must clear this before a "win" pays
                        "cell_min_tp_bps": _hs.get("min_tp_bps"),
                        "cell_cost_note": _hs.get("cost_note"),
                        "cell_tier": _hs.get("tier"),
                    }
            except Exception:
                _cell_cert = {}
            # ── ONE FINAL CHAIN, ON THE NUMBER THAT SHIPS (2026-11-26) ──
            # Three later stages rescale this window's CCI (the volatility
            # factor, the RL deepening pass, and the agent/brain fusion at
            # the GRC), and each of them used to leave `confidence_chain`
            # describing the ORIGINAL score. Live symptom: a card printing
            # `cci_adjusted −0.1436` beside a chain whose step 1 read
            # `|+0.0000| / floor 0.020 = snr 0.00` — the panel built to let
            # a user audit the confidence was auditing a different number.
            # Rather than replicate the recompute at every mutation site
            # (each one a chance to forget it again), the chain is
            # regenerated ONCE here from the values that are actually about
            # to be published, so agreement is structural.
            try:
                _fk_f = fin.get("fusion_inputs") or {}
                _pack_f = fin.get("_pack") or {}
                if _fk_f and _pack_f:
                    _final_cci = float(fin.get("cci_adjusted")
                                       or pack["cci_adjusted"])
                    _final_drhe = float(fin.get("drhe")
                                        or _pack_f.get("drhe") or 1.0)
                    _fcal_f = calibrate_confidence(
                        {"cci_adjusted": _final_cci, "drhe": _final_drhe},
                        experience_quality=_fk_f.get("experience_quality", 0.5),
                        p_dominant=_fk_f.get("p_dominant", 0.0),
                        state_aligned=_fk_f.get("state_aligned", False),
                        rtp=_fk_f.get("rtp", 0.0),
                        pcv=_fk_f.get("pcv", 0.0),
                        wfcp=_fk_f.get("wfcp", 1.0),
                        drpt=_fk_f.get("drpt", 0.0),
                        gspi=_fk_f.get("gspi", 0.0),
                    )
                    _bfac_f = float((fin.get("behavior") or {})
                                    .get("factor") or 1.0)
                    if _bfac_f != 1.0:
                        _fcal_f = dict(_fcal_f)
                        _fcal_f["confidence"] = (
                            float(_fcal_f["confidence"]) * _bfac_f
                            if _bfac_f < 1.0 else
                            float(_fcal_f["confidence"])
                            * (1.0 + (_bfac_f - 1.0) * 0.5))
                    fin["published_chain"] = _fcal_f["steps"]
                    fin["published_chain_conf"] = round(
                        float(_fcal_f["confidence"]), 4)
                    fin["published_chain_snr"] = _fcal_f["snr"]
                    if abs(float(_fcal_f["confidence"])
                           - float(fin.get("confidence") or 0.0)) > 0.02:
                        # say so rather than pretend the chain equals the
                        # published number
                        fin["chain_gap"] = {
                            "chain": round(float(_fcal_f["confidence"]), 4),
                            "published": round(float(
                                fin.get("confidence") or 0.0), 4),
                            "why": "later stages (fusion/behaviour/advanced "
                                   "layer) moved confidence after the chain",
                        }
            except Exception:
                log.exception("final chain regeneration failed %s|%s",
                              symbol, tf)
            results[tf] = {
                "raw": raws[tf],
                "adj": adj[tf],
                "atr": atr_tf,
                "sigma_cur": sigma_cur,
                "sigma_base": sigma_base,
                "arc": {"value": round(arc, 4), "ratio": ratio_band},
                "cci": pack["cci"],
                "cci_raw": pack["cci"],
                "cci_adjusted_raw": pack.get(
                    "cci_adjusted_raw", pack["cci_adjusted"]),
                "cci_adjusted": pack["cci_adjusted"],
                "forward_probabilities": forward_probs[tf],
                "confidence": fin["confidence"],
                # ── THE CHAIN THAT PRODUCED IT (added 2026-11-26) ──────
                # `finalize_confidence` returns the whole arithmetic
                # (base SNR, modulator, the 8 ordered steps) and the
                # confidence-chain panel renders exactly that spec order.
                # Every one of these fields was previously dropped HERE, so
                # the panel had nothing to draw and reported nulls — the one
                # screen that can answer "why did this window publish 0.009"
                # was blank while the number it audits shipped fine.
                "confidence_base_snr": fin.get("confidence_base_snr"),
                "confidence_modulator": fin.get("confidence_modulator"),
                # the walk-forward edge model's own probability, published
                # with the side it moved (None when no artifact is present)
                "edge_model": self._edge_model,
                # the chain ON THE PUBLISHED NUMBERS (falls back to the
                # original when the final pass could not run)
                "confidence_chain": (fin.get("published_chain")
                                     or fin.get("confidence_chain")),
                "chain_gap": fin.get("chain_gap"),
                "chain_confidence": fin.get("published_chain_conf"),
                "chain_snr": fin.get("published_chain_snr"),
                "fusion_inputs": fin.get("fusion_inputs"),
                "fusion_conf": fin.get("fusion_conf"),
                "_pack": fin.get("_pack"),
                "confidence_raw": fin.get("confidence_raw"),
                "snr": fin.get("snr"),
                "noise_floor": fin.get("noise_floor"),
                "adv_conf_mult": fin.get("adv_conf_mult"),
                "adv_conf_mult_rule": fin.get("adv_conf_mult_rule"),
                "emotion_band": fin.get("emotion_band"),
                "micro_band": fin.get("micro_band"),
                "microflow_band": fin.get("microflow_band"),
                "calibration_factor": fin.get("calibration_factor"),
                "direction_hold": fin.get("direction_hold"),
                "direction_fallback": fin.get("direction_fallback"),
                "strength": fin.get("strength", 0.0),
                "winrate": fin.get("winrate"),
                "winrate_n": fin.get("winrate_n", 0),
                "base_confidence": pack["confidence"],
                "direction": pack["direction"],
                # the side the CCI arithmetic itself produced, preserved
                # before any mandate rule overwrites `direction`
                "direction_cci": pack.get("direction"),
                "direction_source": pack.get("direction_source", "cci"),
                "issued": fin["issued"],
                "gate_ok": fin.get("gate_ok"),
                "p_correct": fin.get("p_correct"),
                "calib_iso": fin.get("calib_iso"),
                "spillover": fin.get("spillover"),
                "vol_barriers": fin.get("vol_barriers"),
                "micro": fin.get("micro"),
                "horizon": fin.get("horizon"),
                "edge": fin.get("edge"),
                "sleeves": fin.get("sleeves"),
                "exec_ok": fin.get("exec_ok"),
                "mom_chase_z": fin.get("mom_chase_z"),
                "tier": fin.get("tier"),
                "p_side": fin.get("p_side"),
                "raw_score": fin.get("raw_score"),
                "noise_floor_s": fin.get("noise_floor_s"),
                "score_floor_ratio": fin.get("score_floor_ratio"),
                "calib_n": fin.get("calib_n"),
                "calib_source": fin.get("calib_source"),
                "tier_rule": fin.get("tier_rule"),
                "calib_provisional": fin.get("calib_provisional"),
                "asset_floor": fin.get("asset_floor"),
                "asset_nature": fin.get("asset_nature"),
                **_cell_cert,
                "behavior": fin.get("behavior"),
                # the confidence audit trail: every stage that moved the
                # published number, so the card can name it (2026-11-27)
                "conf_trace": fin.get("conf_trace"),
                "behavior_conf": fin.get("behavior_conf"),
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
            if record or display:
                # learning state is written ONLY by a recording pass.
                # The display payload below must NOT be gated behind it:
                # ANALYSE computes with record=False, so the dashboard used
                # to show the previous recording pass's systems block —
                # stale lock, null ladder, panels contradicting the chart.
                if record:
                    # reward-rule storage: the 28 weighted components only
                    # (DRHE is a multiplier, not a rewarded signal)
                    comp_w = {fid: comp[fid] for fid in WEIGHT_IDS}
                    self.cci.store_signals(symbol, tf, comp_w)
                    self.prev_comp[(symbol, tf)] = comp_w
                    self.prev_brain_signals[(symbol, tf)] = [
                        inst["h_signal"] for inst in br["instances"]
                    ]
                    self.prev_emotions[(symbol, tf)] = (
                        em["emotions"], em["directions"], float(em.get("cerc", 0.0)))
                    _prev_p = dict(phys["values"])
                    _prev_p[22] = float(phys["ufp"]["value"])
                    self.prev_physics[(symbol, tf)] = _prev_p
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
                    # Per-channel accuracy is graded in the reward phase above,
                    # against the PREVIOUS pass that preceded this realized move.
                    # This pass only updates the sparkline, not its own accuracy.
                    for eid, ev in em["emotions"].items():
                        self.hist.ring(
                            self.hist.tf_key(symbol, tf, f"emospark{eid}"), 100
                        ).append(ev)
                    # E22 (CERC) sparkline — not part of the emotions dict
                    self.hist.ring(
                        self.hist.tf_key(symbol, tf, "emospark22"), 100
                    ).append(em["cerc"])
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
                    str(TRACER_COMPONENT_ID): "Hedge-flow tracer (rotation)",
                    str(EMOTION_COMPONENT_ID): "CERC emotion flow",
                    str(PHYSICS_COMPONENT_ID): "Physics UFP (P22)",
                    str(NEWS_COMPONENT_ID): "News impact aggregate",
                    str(HEDGE_COMPONENT_ID): "HF institutional flow",
                }
                _ew = damp_all(pack.get("weights") or base_w)
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
                _comps.append({
                    "id": "102", "name": "Advanced math layer",
                    "value": round(float(_adv.get("vote") or 0.0), 5),
                    "weight": round(
                        float(fin.get("adv_blend_k", 0.08)), 4),
                    "contribution": round(
                        float(fin.get("adv_blend_k", 0.08))
                        * float(_adv.get("vote") or 0.0), 5)})
                _comps.sort(key=lambda c: -abs(c["contribution"]))
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
                    + (f"Advanced layer {_adv.get('vote', 0):+.3f} "
                       f"(gate ×{_adv.get('soft_gate', 1):.2f}, "
                       f"size ×{_adv.get('size_mult', 1):.2f}). ")
                    + (f"Trend decoherence P6 ×{fin['drpt']:.2f} — choppy market. "
                       if fin.get("drpt", 0) > 0.1 else
                       "Physics multipliers near neutral. ")
                    + (f"PCV conflict penalty {pcv:+.2f}. " if pcv < 0 else "")
                    + ("Confidence cleared the 0.35 issue floor — signal issued."
                       if fin["issued"] else
                       f"Confidence {fin['confidence']:.3f} below the 0.35 floor — "
                       "correct action is watch-only until signals align.")
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
                # per-asset mandate direction rule (measured on 2y):
                # raw ladder side is kept in direction_raw / prev_cci_dir
                # so the ladder's own continuity stays honest; the
                # TRADE side is the rule's output. inv4g needs this
                # pass's TF4 side and is applied after the TF loop.
                _lad_raw = _lad
                from . import mandate_cfg as _mcfg
                if _mcfg.direction_rule(symbol, tf) != "inv4g":
                    _lad = _mcfg.apply_direction(symbol, tf, _lad, 0,
                                                 self.feed)
                pack["direction"] = _lad
                results[tf]["direction"] = _lad
                results[tf]["direction_raw"] = _lad_raw
                results[tf]["ladder"] = {
                    "source": _lad_src, "noise_floor": round(_floor, 5),
                    "score": round(float(pack["cci_adjusted"]), 5),
                    "rung": (_lad_src.split("→")[0][:40] or "fused score"),
                }
                results[tf]["mass"] = self._mass_for_against(_comps, _lad)
                # per-tf inputs the post-loop inv4g pass may need to
                # re-derive display fields with the final side
                _mandate_comps[tf] = _comps
                _mandate_dc[tf] = (float(fin["confidence"]), pcv,
                                   br["polarity"])
                # §8.6 learners: edge guard, temporal twin, BOCPD clock,
                # Hawkes boundary intensity — printed features, never
                # silently absorbed into the score
                results[tf]["learners"] = self._learners(symbol, tf, ctxs[tf])
                results[tf]["advanced"] = _adv
                # adaptive conformal band (ACI state per asset/TF,
                # updated at settlement — see _settle hook)
                _cf = getattr(self, "_conformal", {}).get((symbol, tf))
                if not _cf:
                    _ring = list(self.hist.ring(self.hist.tf_key(
                        symbol, tf, "conf_q"), 400))
                    if _ring:
                        _cf = {"q": float(_ring[-1]),
                               "n": len(_ring), "hits": 0}
                if not (_cf and _cf.get("q")):
                    _cq = list(self.hist.ring(self.hist.tf_key(
                        symbol, tf, "conf_q"), 400))
                    results[tf]["conformal"] = {
                        "q_bps": None, "n": len(_cq),
                        "state": "calibrating — needs settled windows"}
                if _cf and _cf.get("q"):
                    _px = float(ctxs[tf].P[-1]) if ctxs[tf].P else 0.0
                    results[tf]["conformal"] = {
                        "q_bps": round(float(_cf["q"]) * 1e4, 1),
                        "lo": round(_px * (1 - float(_cf["q"])), 6),
                        "hi": round(_px * (1 + float(_cf["q"])), 6),
                        "n": int(_cf.get("n", 0)),
                        "coverage": (round(_cf["hits"] / _cf["n"], 3)
                                     if _cf.get("n") else None),
                        "target_coverage": 0.8}
                try:
                    _gv = self.genesis.live_score(
                        self._genesis_values.get((symbol, tf)) or {})
                    results[tf]["genesis"] = _gv
                except Exception:
                    results[tf]["genesis"] = {"score": 0.0, "weight": 0.0,
                                              "n_active": 0}
                self.prev_cci_dir[(symbol, tf)] = _lad
                # ── WHY THIS DICT IS NO LONGER A HAND-WRITTEN WHITELIST ──
                # A whitelist that is maintained by hand next to a producer
                # that keeps gaining fields can only ever fall behind, and
                # every field it drops fails SILENTLY (it ships as `None`).
                # That is exactly what happened twice: the P(win)/spillover
                # rows read "insufficient" forever even when calibrated, and
                # — found 2026-11-26, still live — `confidence`, `snr`,
                # `confidence_chain`, `confidence_base_snr`,
                # `confidence_modulator` and `adv_conf_mult` were ALL null in
                # /api/systems, so the CONFIDENCE CHAIN panel (the one place
                # a user can audit WHY 0.009 was published) had nothing to
                # draw. The dict now STARTS from the producer's own fields
                # and then overlays the curated/derived ones below, so a new
                # telemetry field is published by default instead of being
                # forgotten. Explicit entries below still win.
                _ds_base = {k: v for k, v in results[tf].items()
                            if not str(k).startswith("_")}
                self.display_systems[(symbol, tf)] = {
                    **_ds_base,
                    # WHEN this block was measured. Everything in
                    # display_systems — formula values, emotion channels,
                    # physics — is computed once per GRC pass and is
                    # therefore BIT-IDENTICAL between passes. The panels
                    # repaint from it every 2 s, so without this stamp the
                    # screen looked like live values that had frozen
                    # ("formula value doesn't change", "sadness for 8
                    # days"). The numbers are correct; they are simply not
                    # new. The UI prints the age so a reading can never be
                    # mistaken for a fresh one.
                    "measured_at": time.time(),
                    # fields the analysis FINAL writes into results[tf]
                    # (p_correct / calib_iso / spillover / winrate …)
                    # used to be dropped by this whitelist, so the hero's
                    # P(win) calibration row and spillover row rendered
                    # "insufficient" forever even when calibrated
                    "p_correct": results[tf].get("p_correct"),
                    "calib_iso": results[tf].get("calib_iso"),
                    "spillover": results[tf].get("spillover"),
                    "winrate": results[tf].get("winrate"),
                    "winrate_n": results[tf].get("winrate_n", 0),
                    "issue_note": results[tf].get("issue_note"),
                    "gate_ok": results[tf].get("gate_ok"),
                    "tps": results[tf].get("tps"),
                    "horizon": results[tf].get("horizon"),
                    "edge": results[tf].get("edge"),
                    "sleeves": results[tf].get("sleeves"),
                    "vol_barriers": results[tf].get("vol_barriers"),
                    "prediction_lock": results[tf].get("prediction_lock"),
                    "tier": results[tf].get("tier"),
                    "p_side": results[tf].get("p_side"),
                    "raw_score": results[tf].get("raw_score"),
                    "noise_floor_s": results[tf].get("noise_floor_s"),
                    "score_floor_ratio": results[tf].get("score_floor_ratio"),
                    "calib_n": results[tf].get("calib_n"),
                    "calib_source": results[tf].get("calib_source"),
                    "tier_rule": results[tf].get("tier_rule"),
                    "calib_provisional": results[tf].get("calib_provisional"),
                    "lock": dict(self.locks.get((symbol, tf)) or {}),
                    "asset_floor": results[tf].get("asset_floor"),
                    "asset_nature": results[tf].get("asset_nature"),
                    "behavior": results[tf].get("behavior"),
                    "ladder": dict(results[tf]["ladder"]),
                    "mass": results[tf]["mass"],
                    "advanced": {
                        "vote": _adv.get("vote"),
                        "soft_gate": _adv.get("soft_gate"),
                        "rqa_mult": _adv.get("rqa_mult"),
                        # the estimator's own null (chance level) and the
                        # excess that actually moved the multiplier — a
                        # multiplier printed alone cannot be audited
                        "rqa_det": _adv.get("rqa_det"),
                        "rqa_null": _adv.get("rqa_null"),
                        "size_mult": _adv.get("size_mult"),
                        "modules": {
                            k: {"vote": (v or {}).get("vote"),
                                "gate": (v or {}).get("gate"),
                                "regime": (v or {}).get("regime"),
                                "hurst": (v or {}).get("hurst"),
                                # a zero vote must say WHY (thin /
                                # non-stationary / no ref) — silent
                                # zeros are how distrust starts
                                "why": ((v or {}).get("why")
                                        if not (v or {}).get("vote")
                                        else None),
                                "z": (v or {}).get("z"),
                                "half_life_bars": (v or {})
                                .get("half_life_bars")}
                            for k, v in (_adv.get("modules")
                                         or {}).items()},
                        "har_rv": _adv.get("har_rv"),
                        "evt": _adv.get("evt"),
                        "rmt_mode_share": (_adv.get("rmt") or {})
                        .get("mode_share"),
                        "conformal": results[tf].get("conformal"),
                    },
                    "genesis": dict(results[tf].get("genesis") or {}),
                    "learners": dict(results[tf].get("learners") or {}),
                    "brain": brain_consensus,
                    "emotion": {
                        "aggregate": round(cerc_sync[tf], 5),
                        "cerc": em["cerc"],
                        "net_emotion": em.get("net_emotion", 0.0),
                        "agreement": em.get("agreement", 0.0),
                        "breadth": em.get("breadth"),
                        "dominant": em.get("dominant"),
                        "model": em.get("model"),
                        "core_affect": em.get("core_affect"),
                        "pad": em.get("pad"),
                        "panas": em.get("panas"),
                        "ekman": em.get("ekman"),
                        "primary_emotion": em.get("primary_emotion"),
                        "fear_greed": em.get("fear_greed"),
                        "action_tendency": em.get("action_tendency"),
                        "risk": em.get("risk"),
                        "retail": em.get("retail"),
                        "hedge": em.get("hedge"),
                        "features": em.get("features") or {},
                        "zero_reasons": em.get("zero_reasons") or {},
                        "zero_causes": em.get("zero_causes") or {},
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
                    # CONTRIBUTION TICKS (user request): which formula /
                    # emotion / physics mechanism / hedge leg / news theme
                    # actually contributed to THIS window, and which not
                    "ticks": {
                        "formulas": {
                            str(c["id"]): {
                                "on": abs(c["contribution"]) >= 1e-5,
                                "c": c["contribution"],
                                "name": c["name"],
                            } for c in _comps
                        },
                        "emotions": {
                            str(i): {"on": abs(float(v)) > 0.01,
                                     "v": round(float(v), 4)}
                            for i, v in em["emotions"].items()
                        },
                        "physics": {
                            str(m["id"]): {
                                "on": bool(m["active"]),
                                "v": m["score"],
                                "name": m["name"],
                                "why": "" if m["active"] else m.get("why", ""),
                            } for m in phys["mechanisms"]
                        },
                        "hedge": {
                            k: {"on": abs(float((phys["hedge"] or {}).get(k, 0.0) or 0.0)) > 1e-6,
                                "v": round(float((phys["hedge"] or {}).get(k, 0.0) or 0.0), 5)}
                            for k in ("ifr", "dpad", "smdi", "combined")
                            if k in (phys["hedge"] or {})
                        },
                        "news": {
                            "aggregate": {"on": abs(news_agg["value"]) > 1e-9,
                                          "v": round(news_agg["value"], 5),
                                          "count": news_agg["count"]},
                            "themes": {
                                str(k): {"on": abs(float(v)) > 1e-9,
                                         "v": round(float(v), 5)}
                                for k, v in (news_agg.get("impacts") or {}).items()
                            },
                        },
                        "brain": {
                            "flywire": {
                                "on": abs(float(
                                    (brain_res[tf].get("flywire") or {}
                                     ).get("vote") or 0.0)) > 1e-6,
                                "v": (brain_res[tf].get("flywire") or {}
                                      ).get("vote"),
                                "ready": bool(
                                    (brain_res[tf].get("flywire") or {}
                                     ).get("ready")),
                            },
                        },
                    },
                    # exchange-server physics gauges (clock-skew z, gap z,
                    # liquidation clusters) — honest absence on the sim tape
                    "micro": self.feed.micro_status(symbol),
                    # live order book: DBI, micro-price, spread, Λ, walls
                    "book": self.feed.book_status(symbol),
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
                        "who": {
                            "formulas": round(sum(
                                abs(c["contribution"]) for c in _comps
                                if c["id"].isdigit()
                                and int(c["id"]) <= 31) / (
                                sum(abs(c["contribution"])
                                    for c in _comps) or 1.0), 4),
                            "brain": round(sum(
                                abs(c["contribution"]) for c in _comps
                                if c["id"] == str(BRAIN_COMPONENT_ID)) / (
                                sum(abs(c["contribution"])
                                    for c in _comps) or 1.0), 4),
                            "emotion": round(sum(
                                abs(c["contribution"]) for c in _comps
                                if c["id"] == str(EMOTION_COMPONENT_ID)) / (
                                sum(abs(c["contribution"])
                                    for c in _comps) or 1.0), 4),
                            "physics": round(sum(
                                abs(c["contribution"]) for c in _comps
                                if c["id"] == str(PHYSICS_COMPONENT_ID)) / (
                                sum(abs(c["contribution"])
                                    for c in _comps) or 1.0), 4),
                            "news": round(sum(
                                abs(c["contribution"]) for c in _comps
                                if c["id"] == str(NEWS_COMPONENT_ID)) / (
                                sum(abs(c["contribution"])
                                    for c in _comps) or 1.0), 4),
                            "hedge": round(sum(
                                abs(c["contribution"]) for c in _comps
                                if c["id"] == str(HEDGE_COMPONENT_ID)) / (
                                sum(abs(c["contribution"])
                                    for c in _comps) or 1.0), 4),
                        "tracer": round(sum(
                                abs(c["contribution"]) for c in _comps
                                if c["id"] == str(TRACER_COMPONENT_ID)) / (
                                sum(abs(c["contribution"])
                                    for c in _comps) or 1.0), 4),
                        },
                        "base_confidence": round(pack["confidence"], 5),
                        "final_confidence": round(fin["confidence"], 5),
                        "penalty_capped": bool(fin.get("penalty_capped")),
                        "penalty": fin.get("penalty", 0.0),
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
                    # Part-4 sync: expose the unmodified CCI and its
                    # double-check-adjusted value separately from the later
                    # behavior/advanced score sent to the direction ladder.
                    "cci": pack["cci"],
                    "cci_raw": pack["cci"],
                    "cci_adjusted_raw": pack.get(
                        "cci_adjusted_raw", pack["cci_adjusted"]),
                    "cci_adjusted": pack["cci_adjusted"],
                    "forward_probabilities": forward_probs[tf],
                    "direction": pack["direction"],
                    "issued": bool(fin["issued"]),
                    "weight_profile": self._weight_profile(symbol, tf, {
                        "raw": raws[tf], "drhe": drhe,
                        "physics": {"ufp": phys["ufp"]["value"]},
                    }),
                    "agent": None,          # patched by _apply_agents
                    # DISPLAY-ONLY verdict: analyse windows never call
                    # _apply_agents, so a None here made the seams panel
                    # show a bare dash forever. The GRC pass still computes
                    # the authoritative (mutating) double-check and
                    # overwrites this entry.
                    "double_check": self._dc_display(
                        pack["direction"], fin["confidence"], pcv,
                        br["polarity"]),
                }
        # ── mandate inv4g (post-loop): the traded side for these cells
        # is the inverse of THIS pass's TF4 raw side, so it can only be
        # applied once every TF has been computed (the search read the
        # same-pass pack["tf4"]). Side-derived display fields (mass,
        # double-check verdict) are re-derived with the final side.
        from . import mandate_cfg as _mcfg_i
        for _tf, _res in list(results.items()):
            if _mcfg_i.direction_rule(symbol, _tf) != "inv4g":
                continue
            _tf4r = results.get("tf4") or {}
            _t4raw = int(_tf4r.get("direction_raw",
                                  _tf4r.get("direction") or 0) or 0)
            _oldd = int(_res.get("direction") or 0)
            _newd = _mcfg_i.apply_direction(symbol, _tf, _oldd, _t4raw,
                                            self.feed)
            if _newd == _oldd:
                continue
            # the mandate rule overwrote the CCI side: say so, and keep the
            # CCI sign visible so the UI can label a fade/inv4g divergence
            # instead of printing BULLISH next to a negative CCI ADJ
            _res["direction_source"] = "mandate:%s" % (
                _mcfg_i.direction_rule(symbol, _tf))
            _res["cci_side"] = _res.get("direction_cci")
            _res["direction"] = _newd
            if _tf in _mandate_comps:
                _res["mass"] = self._mass_for_against(
                    _mandate_comps[_tf], _newd)
            _disp = self.display_systems.get((symbol, _tf))
            if _disp is not None:
                if _tf in _mandate_dc:
                    _c_, _p_, _b_ = _mandate_dc[_tf]
                    _disp["double_check"] = self._dc_display(
                        _newd, _c_, _p_, _b_)
                if "direction" in _disp:
                    _disp["direction"] = _newd
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

    def _sibling_dirs(self, symbol: str, exclude_tf: str) -> dict:
        """Other timeframes' latest directions for the agree gate
        (stale entries >30 min ignored — a silent sibling never vetoes)."""
        now = time.time()
        out = {}
        for (s, t), (ts, d) in list(
                (getattr(self, "_last_dirs", None) or {}).items()):
            if s == symbol and t != exclude_tf and now - ts <= 1800:
                out[t] = int(d)
        return out

    def _tier_for(self, symbol: str, tf: str, score: float,
                  strength: float | None = None, *,
                  direction: int | None = None,
                  dirs_by_tf: dict | None = None,
                  confidence: float | None = None) -> dict:
        """docs11-doublecheck §4: calibrated P(side) + tier. The noise
        floor lives in SCORE units and only sets the tier — it can
        never suppress the side. Provisional Platt: a = 1/median|s|
        (a typical score maps to P≈0.73), labelled provisional until
        40 score samples exist for this asset+TF.

        The scale MUST come from measured score history. It used to fall
        back to `p_side = 0.5` whenever fewer than 10 samples existed — and
        since the ladder only appends one score per *cycle*, a fresh
        process sat at P 0.50 → `watch-only` for hours: the permanent
        "NO TRADE" the user reported. Now the pool falls back to the
        persisted |score| ring (survives restarts), then to the measured
        voter-agreement purity, each labelled with its source.
        """
        import math as _m
        pool = [abs(float(v)) for v in
                (self.score_hist.get((symbol, tf), []) or []) if v is not None]
        source = "in-process score history"
        if len(pool) < 10:
            try:
                ring = self.hist.ring(self.hist.tf_key(symbol, tf, "score_abs"),
                                      200)
                ring_vals = [abs(float(v)) for v in list(ring)
                             if v is not None and abs(float(v)) > 0]
                if len(ring_vals) >= 10:
                    pool, source = ring_vals, "persisted |score| ring"
            except Exception:
                pass
        med = 0.0
        if len(pool) >= 10:
            _o = sorted(pool); med = _o[len(_o) // 2]
        floor_s = max(0.02, 0.5 * med)
        # THE FLOOR AND THE PROBABILITY MUST SHARE ONE SCALE. The score was
        # compared to a floor built from `med`, but P was mapped with `med`
        # itself — when med was tiny the floor defaulted to 0.02 while
        # |score|/med ran to 3+, so the same window read "P 0.97" and
        # "watch-only" at once. P is now the logistic of the score in
        # FLOOR units, and the tier bands are those same units:
        #   <=1.0× floor → watch-only (indistinguishable from noise)
        #   <=1.5×       → quarter
        #   <=2.5×       → half
        #   > 2.5×       → full
        if not pool:
            source = ("absolute floor 0.02 (no score history yet) — provisional"
                      + (f"; voter agreement {float(strength):.2f}"
                         if strength is not None else ""))
        x = abs(float(score)) / floor_s if floor_s > 0 else 0.0
        p_side = 1.0 / (1.0 + _m.exp(-min(20.0, x)))
        if x <= 1.0:
            tier = "watch-only"
        elif x <= 1.5:
            tier = "quarter"
        elif x <= 2.5:
            tier = "half"
        else:
            tier = "full"
        # ── per-asset mandate (2y real measurement, mandate_search):
        # a configured asset only issues on its measured TFs and past
        # its measured filter (§10: all four assets ≥ +0.070R full
        # sample, 6/6 folds positive — PAXG recovered from the engine
        # direction's −0.21R via the 44h-momentum rule on tf3/tf4).
        # Non-measured cells are capped; the side still stands (the
        # ladder never suppresses it), only the tier caps.
        _mandate_note = ""
        # TIER AND CONFIDENCE MUST AGREE (fixed 2026-11-26). The tier band is
        # derived from |score|/floor, so a window can read "full size" on a
        # score that is 4× its noise floor while its CALIBRATED confidence is
        # 0.009 (the advanced multipliers, PCV and DRHE can all cut it) —
        # the live board printed exactly that: `conf 0.01 · tier full ·
        # verdict NO CALL`. A size hint that outranks the published gate is
        # not a hint, it is a contradiction. The tier may now never be more
        # optimistic than the verdict rule allows.
        try:
            from .config import asset_issue_floor as _aif_t
            _band_floor = float(_aif_t(symbol))
        except Exception:
            _band_floor = 0.35
        # NO SIDE, NO SIZE (fixed 2026-11-26, live defect). The board printed
        # `BTC tf4 · direction 0 · tier full · verdict NO CALL` — a card
        # advertising FULL SIZE on a window where the engine had not even
        # picked a side. A tier is a size hint for a trade; with no direction
        # there is no trade to size, so the hint must collapse to watch-only
        # and say why. (The tier never suppresses the side — it never could:
        # this branch only runs when there is nothing to suppress.)
        # ``direction is None`` means the caller did not tell us a side (a
        # pure tier-math probe); only a CALLER-DECLARED flat is a non-trade.
        if direction is not None and int(direction) == 0 \
                and tier != "watch-only":
            tier = "watch-only"
            _mandate_note += (
                " · watch-only: no side was derived this window (the fused "
                "score sat inside the noise gate), so there is nothing to "
                "size — the card must not advertise a tier for a non-trade; "
                "there is nothing to size in a window with no side")
        _conf_for_tier = float(confidence) if confidence is not None else None
        if _conf_for_tier is not None and tier != "watch-only":
            # watch-only at/below the floor; quarter in (floor, floor×1.1]
            if _conf_for_tier <= _band_floor:
                tier = "watch-only"
                _mandate_note += (
                    f" · tier capped at watch-only: confidence "
                    f"{_conf_for_tier:.3f} <= issue floor {_band_floor:.3f} "
                    "(the size hint may never outrank the published gate)")
            elif _conf_for_tier <= _band_floor * 1.1 and tier in (
                    "half", "full"):
                tier = "quarter"
                _mandate_note += (
                    f" · tier cut to quarter: confidence {_conf_for_tier:.3f} "
                    f"is within 10 % of the {_band_floor:.3f} issue floor")
        # ── per-asset mandate (2y real measurement, mandate_search):
        # a configured asset only issues on its measured TFs and past
        # its measured filter (§10: all four assets ≥ +0.070R full
        # sample, 6/6 folds positive — PAXG recovered from the engine
        # direction's −0.21R via the 44h-momentum rule on tf3/tf4).
        # Non-measured cells are capped; the side still stands (the
        # ladder never suppresses it), only the tier caps.
        if tier != "watch-only":
            from . import mandate_cfg as _mcfg
            if not _mcfg.allows(symbol, tf):
                tier = "watch-only"
                _c = _mcfg.MANDATE.get(symbol) or {}
                if _c.get("tfs"):
                    _mandate_note = (
                        f" · mandate: {symbol} issues only on "
                        f"{', '.join(_c['tfs'])} (2y-measured cells)")
                else:
                    _mandate_note = (
                        " · PAXG capped watch-only by the 2y mandate "
                        "(negative EV both modes, both splits)")
            else:
                _mok, _mwhy = _mcfg.filter_ok(
                    symbol, tf, getattr(self, "feed", None))
                if not _mok:
                    tier = "watch-only"
                    _mandate_note = f" · {_mwhy}"
                else:
                    # win-rate search confluence (abstention only):
                    # regime align + cross-TF agree, both causal
                    _cok, _cwhy = _mcfg.confluence_ok(
                        symbol, tf, int(direction or 0),
                        getattr(self, "feed", None),
                        dirs_by_tf or self._sibling_dirs(symbol, tf))
                    if not _cok:
                        tier = "watch-only"
                        _mandate_note = f" · {_cwhy}"
        # HOLD POLICY: every managed cell carries its certified badge
        # (HOLD-OUT VERIFIED / IN-SAMPLE SHADOW / UNGRADEABLE · WARMUP)
        # and measured win rate — the tier may size the position but the
        # hold side itself is never gated (no abstention).
        try:
            from . import hold_policy as _hp_t
            if _hp_t.managed(symbol, tf):
                _hs = _hp_t.summary(symbol, tf)
                _wr = _hs.get("win_rate")
                _lo = _hs.get("confidence")
                _wr_txt = (f"wr {100 * _wr:.1f}%, wilson lo {100 * _lo:.1f}%"
                           if _wr and _lo else
                           "no certified wr — warmup")
                # `_hs['tier']` is the HOLD POLICY'S certification badge
                # ("ROWS VERIFIED · SHIP GATE FAIL"), not this card's size
                # tier — printing it as "tier" put a status sentence in a
                # field every panel reads as full/half/quarter. Name it.
                _mandate_note += (f" · hold-policy certification "
                                  f"[{_hs.get('tier')}] {_wr_txt}")
        except Exception:
            pass
        return {"tier": tier, "p_side": round(p_side, 4),
                "noise_floor_s": round(floor_s, 5),
                "score_floor_ratio": round(x, 3),
                "calib_n": len(pool), "calib_source": source,
                "calib_provisional": len(pool) < 40,
                "tier_rule": ("size hint only — the side is never suppressed "
                              "by the tier; watch-only = below the measured "
                              "noise floor" + _mandate_note)}

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
        try:
            self.hist.ring(self.hist.tf_key(symbol, tf, "score_abs"),
                           200).append(abs(float(score)))
        except Exception:
            pass
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

        # rung 0.5: the WALK-FORWARD-VALIDATED EDGE MODEL.
        # It sits above the fused score because it is the only directional
        # component in this system with a measured out-of-sample edge at the
        # published reward:risk: 39.9 % at 2:1 (CI 38.8-41.1, n=7168) on
        # BTC, 39.5 % on BNB, against a 33.33 % break-even. The ablation in
        # backtest/edge_model.py also showed the 44 formula outputs add
        # nothing (physics 36.2 % vs SHUFFLED physics 37.1 %), so leaning on
        # the fused score first would be leaning on the weaker signal.
        #
        # It is OPTIONAL and additive: no artifact -> this block is skipped
        # and behaviour is byte-identical. It never abstains (side needs only
        # p >= 0.5) and its probability is published, so the reason a side
        # moved is visible rather than inferred.
        try:
            from . import edge_model as _em
            if _em.available():
                _st_e = self.feed.state(symbol)
                _pe = _em.predict(list(_st_e.closes), list(_st_e.highs or
                                                          _st_e.closes),
                                  list(_st_e.lows or _st_e.closes),
                                  list(_st_e.volumes or []),
                                  list(_st_e.times or []))
                if _pe and _pe.get("side") in (-1, 1):
                    self._edge_model = _pe
                    _em_side = int(_pe["side"])
                    return (_em_side,
                            f"edge model P(up) {_pe['prob_up']:.3f} "
                            f"(horizon {_pe['horizon_bars']}h, stop "
                            f"{_pe['stop_atr']}xATR) — measured 39.9% at 2:1 "
                            f"out-of-sample vs 33.3% break-even; side "
                            f"{'UP' if _em_side > 0 else 'DOWN'} "
                            f"(fused score {score:+.4f} would have said "
                            f"{'up' if score > 0 else 'down'})",
                            floor)
        except Exception:
            log.exception("edge model vote failed %s/%s", symbol, tf)

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
                      "ai_review", "recheck", "final")

    def start_analyse(self, symbol: str, tf: str = "tf1",
                      source: str = "manual",
                      request_id: str | None = None) -> dict:
        if symbol not in SYMBOL_ORDER:
            return {"ok": False, "error": "unknown symbol"}
        if tf not in TF_ORDER:
            return {"ok": False, "error": "unknown timeframe"}
        # HOLD POLICY — per-cell ANALYSE timer (user requirement): one
        # click arms a tf-minute (22/44/130/247) background timer on
        # THIS (asset, tf) only.  While it runs the app refuses any
        # other prediction for the same cell; every other cell — and
        # this asset on other timeframes — stays free (4×4 = 16
        # independent predictions).  Only timer expiry re-enables the
        # cell.  The timer starts at the CLICK, before the stages run.
        from . import hold_policy as _hp
        _locks = _hp.cell_locks()
        if _locks.locked(symbol, tf):
            _rem = _locks.remaining(symbol, tf)
            return {"ok": False, "error": "cell_locked", "locked": True,
                    "symbol": symbol, "tf": tf,
                    "locked_until": _locks.until(symbol, tf),
                    "remaining_s": round(_rem, 1),
                    "minutes_left": int(math.ceil(_rem / 60.0)),
                    "why": (f"{symbol} {tf.upper()} is inside its "
                            f"{TIMEFRAMES[tf]['minutes']}-minute analysis "
                            f"window — retry when the countdown ends; other "
                            f"cells are unaffected")}
        _armed = _locks.start(symbol, tf)
        # §1.1 stamping: every analyse_* event carries {asset, timeframe,
        # request_id} so the UI can atomically discard superseded work.
        # Latest-wins: a newer request_id for the same cell replaces the
        # queued one — the client never renders a stale analysis.
        rid = str(request_id) if request_id else None
        if rid:
            self._analyse_reqids[(symbol, tf)] = rid
        if self.analyse_job and self.analyse_job.get("status") == "running":
            same = (self.analyse_job.get("symbol") == symbol
                    and self.analyse_job.get("tf") == tf)
            if same:
                if rid and not self.analyse_job.get("request_id"):
                    self.analyse_job["request_id"] = rid
                return {"ok": True, "symbol": symbol, "tf": tf,
                        "started_at": self.analyse_job["started_at"],
                        "stages": list(self.ANALYSE_STAGES),
                        "already_running": True,
                        "request_id": self.analyse_job.get("request_id")}
            item = (symbol, tf)
            if item not in self._analyse_queue:
                self._analyse_queue.append(item)
            return {"ok": True, "symbol": symbol, "tf": tf,
                    "queued": True, "queue_position": len(self._analyse_queue),
                    "started_at": None,
                    "stages": list(self.ANALYSE_STAGES),
                    "request_id": rid}
        self.analyse_job = {
            "symbol": symbol, "tf": tf, "source": source,
            "started_at": time.time(), "stages": [],
            "status": "running", "result": None,
            "request_id": rid,
        }
        self._analyse_task = asyncio.create_task(
            self._run_analyse(self.analyse_job))
        return {"ok": True, "symbol": symbol, "tf": tf,
                "started_at": self.analyse_job["started_at"],
                "stages": list(self.ANALYSE_STAGES),
                "request_id": rid}

    def _start_queued_analyse(self) -> None:
        """Start the next manual analysis once the running one is done."""
        while self._analyse_queue:
            symbol, tf = self._analyse_queue.pop(0)
            if (self.analyse_job
                    and self.analyse_job.get("symbol") == symbol
                    and self.analyse_job.get("tf") == tf):
                continue
            try:
                self.start_analyse(
                    symbol, tf, source="queued",
                    request_id=self._analyse_reqids.get((symbol, tf)))
                return
            except Exception:
                log.exception("queued analyse failed for %s|%s", symbol, tf)


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
                         "timeframe": job.get("tf"),
                         "request_id": job.get("request_id"),
                         "stage": entry})

    async def _run_analyse(self, job: dict) -> None:
        """User budget: the whole analysis must never exceed 3 minutes.
        The body runs under a hard 175 s watchdog; on timeout the job is
        failed honestly (no silent hang, no partial result shown as done)."""
        try:
            await asyncio.wait_for(self._analyse_body(job), timeout=175.0)
        except asyncio.TimeoutError:
            job["status"] = "error"
            job["error"] = "analysis exceeded the 175 s budget (3 min cap)"
            log.error("analysis for %s exceeded 175 s budget",
                      job.get("symbol"))
            await self.emit({"type": "analyse_done",
                             "symbol": job.get("symbol"),
                             "asset": job.get("symbol"),
                             "tf": job.get("tf"),
                             "timeframe": job.get("tf"),
                             "request_id": job.get("request_id"),
                             "result": None,
                             "error": "exceeded the 175 s budget "
                                      "(3 minute cap)"})
        except Exception as exc:
            # a crash used to vanish as an unhandled task exception —
            # the UI kept "queued" rows forever and the cell stayed
            # burned behind its timer although NO prediction existed
            job["status"] = "error"
            job["error"] = f"analysis failed: {exc!r}"[:300]
            log.exception("analyse body failed for %s|%s",
                          job.get("symbol"), job.get("tf"))
            await self.emit({"type": "analyse_done",
                             "symbol": job.get("symbol"),
                             "asset": job.get("symbol"),
                             "tf": job.get("tf"),
                             "timeframe": job.get("tf"),
                             "request_id": job.get("request_id"),
                             "result": None,
                             "error": job["error"]})
        finally:
            if job.get("status") == "running":      # defensive: never stuck
                job["status"] = "done" if job.get("result") else "error"
            try:
                self._journal_analyse(job)
            except Exception:
                log.exception("analyse journal write failed")
            # a FAILED analysis seals nothing — give the cell back
            # instead of burning its whole window behind a crash
            # ("it locked my eth for timer" with no usable prediction)
            if job.get("status") == "error":
                try:
                    from . import hold_policy as _hp_rel
                    _hp_rel.cell_locks().clear(job["symbol"], job["tf"])
                except Exception:
                    pass
            self._start_queued_analyse()

    def _journal_analyse(self, job: dict) -> None:
        """Issue 3 — durable memory of EVERY analysis: what was decided,
        what it learned (verdict/lock/hindsight) and, when the window
        settles, how it turned out. Append-only JSONL under data/, capped
        so the file cannot grow without bound."""
        path = os.environ.get("ANALYSE_JOURNAL_PATH",
                              os.path.join("data", "analyses.jsonl"))
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        res = job.get("result") or {}
        lock = res.get("prediction_lock") or {}
        rec = {
            "ts": time.time(),
            "started_at": job.get("started_at"),
            "symbol": job.get("symbol"),
            "tf": job.get("tf"),
            "source": job.get("source"),
            "status": job.get("status"),
            "error": job.get("error"),
            "signal": res.get("signal"),
            "confidence": res.get("confidence"),
            "verdict": res.get("verdict"),
            "gate_ok": res.get("gate_ok"),
            "issue_note": res.get("issue_note"),
            "lock_id": lock.get("id"),
            "locked": bool(lock.get("locked")),
            "lock_side": lock.get("side"),
            "recheck": (res.get("recheck") or {}).get("verdict"),
            "recheck_passed": (res.get("recheck") or {}).get("passed"),
            "elapsed_ms": res.get("elapsed_ms"),
        }
        # what it LEARNED to change: the AI correction + hindsight note,
        # when present, are the actionable "needs to change" record
        corr = res.get("ai_correction") or res.get("correction")
        if corr:
            rec["ai_correction"] = {
                "by": corr.get("by"), "direction": corr.get("direction"),
                "reason": (corr.get("reason") or "")[:400]}
        hz = res.get("hindsight") or {}
        if hz:
            rec["hindsight"] = {k: hz.get(k)
                                for k in ("outcome", "note", "rate", "n")
                                if k in hz}
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        # cap: keep the newest 2000 records
        try:
            if os.path.getsize(path) > 4_000_000:
                with open(path, encoding="utf-8") as fh:
                    lines = fh.readlines()
                with open(path, "w", encoding="utf-8") as fh:
                    fh.writelines(lines[-2000:])
        except OSError:
            pass

    async def _analyse_body(self, job: dict) -> None:
        symbol, tf = job["symbol"], job["tf"]
        t_job0 = time.time()
        await self.emit({"type": "analyse_started", "symbol": symbol,
                         "asset": symbol, "tf": tf, "timeframe": tf,
                         "request_id": job.get("request_id"),
                         "started_at": job["started_at"],
                         "stages": list(self.ANALYSE_STAGES)})
        store: dict = {}
        try:
            # 1 ── formulas: every value, every timeframe, timed ----------
            t0 = time.time()
            raws, per_tf_ms = {}, {}
            # EVERY heavy stage of this job runs on a worker thread. The job
            # is auto-triggered every 45 s and is ~10-20 s of pure-CPU
            # analytics; inline in this async def it owned the event loop,
            # so /api/health, the websocket and the snapshot build all
            # stalled behind it. That is the "ENGINE BUSY" banner — measured
            # at clock.handler_ms ~10400 — not a slow network.
            def _stage_formulas():
                out, ms = {}, {}
                for t in TF_ORDER:
                    _gov.cooperate()
                    ctx = self.build_ctx(symbol, t)
                    anoms: list = []
                    f0 = time.perf_counter()
                    out[t] = compute_all(ctx, record=False, anomalies=anoms)
                    ms[t] = round((time.perf_counter() - f0) * 1000, 1)
                return out, ms

            raws, per_tf_ms = await run_pass(_stage_formulas)
            store["raws"] = raws
            # HARD GUARD (user order): a prediction may never seal
            # without formulas actually producing readings — the failed
            # run must fail loudly, not lock a timer on an empty brain
            if not raws or not any(raws.get(t) for t in TF_ORDER):
                raise RuntimeError(
                    "formulas stage produced no readings — refusing to "
                    "seal a prediction that never used the formulas")
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
            def _stage_emotions():
                out = {}
                for t in TF_ORDER:
                    _gov.cooperate()
                    ctx = self.build_ctx(symbol, t)
                    atr = self._atr(ctx)
                    thing2 = self._thing2(symbol, t, ctx, atr, market)
                    affinity = {fid: abs(v) for fid, v in raws[t].items()
                                if abs(v) > 0.7}
                    br = self.brains.evaluate(symbol, t, market, affinity,
                                              thing2, advance=False,
                                              raws=raws[t])
                    _gov.cooperate()   # brain pass is a long quantum
                    em = self.emotions.compute(ctx, market, br, news_agg, None)
                    _gov.cooperate()
                    from .emotion import CHANNELS as _ECH
                    _pe = (em.get("primary_emotion") or {})
                    dom = _pe.get("name") or _ECH[
                        max(em["emotions"], key=em["emotions"].get)][1]
                    # full pass kept for the §5 committee vote (the
                    # contrarian agent reads cerc_vectors + raw channels)
                    out[t] = {"_full": em,
                              "cerc": round(em["cerc"], 4),
                              "top_emotion": str(dom),
                              "fear_greed": (em.get("fear_greed") or {}).get("index"),
                              "valence": (em.get("core_affect") or {}).get("valence"),
                              "brain_state": br["dominant"],
                              "polarity": br["polarity"]}
                return out

            em_out = await run_pass(_stage_emotions)
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
            def _stage_physics():
                out = {}
                for t in TF_ORDER:
                    _gov.cooperate()
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
                    out[t] = {
                        "ufp": round(ph["ufp"]["value"], 4),
                        "wfcp": round(ph["wfcp"], 4),
                        "drpt": round(ph["drpt"], 4),
                        "gspi": round(ph["gspi"], 4),
                        "mech_active": sum(1 for m in mech if m["active"]),
                        "mech_total": len(mech),
                    }
                return out

            phys_out = await run_pass(_stage_physics)
            store["physics"] = phys_out
            pe = phys_out[tf]
            await self._emit_stage(
                job, "physics", "done", t0, time.time(),
                f"UFP {pe['ufp']:+.4f} · {pe['mech_active']}/{pe['mech_total']} "
                f"mechanisms active · WFCP {pe['wfcp']:.3f} DRPT {pe['drpt']:.3f}",
                phys_out)
            # 4 ── news ----------------------------------------------------
            t0 = time.time()
            await run_pass(self.news.refresh)
            agg = await run_pass(self.news.aggregate, symbol)
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
            def _stage_hedge():
                out = {}
                for t in TF_ORDER:
                    ctx = self.build_ctx(symbol, t)
                    ifr = hedgefund.institutional_flow(ctx)
                    dpad = hedgefund.dark_pool_flow(ctx)
                    smdi = hedgefund.smart_money_divergence(
                        ifr["value"], store["emotions"][t]["polarity"])
                    out[t] = {
                        "ifr": round(ifr["value"], 4),
                        "dpad": round(dpad["value"], 4),
                        "smdi": round(smdi, 4),
                        "combined": round(hedgefund.combined_signal(
                            ifr["value"], dpad["value"], smdi), 4),
                    }
                return out

            hedge_out = await run_pass(_stage_hedge)
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
            res_pack = await run_pass(
                self.compute_asset, symbol, record=False, display=True)
            # SAME-NUMBER RULE: the FINAL stage must analyse exactly what the
            # card publishes. Recomputing produced a second, different
            # answer (card 0.176 BULL vs analysis 0.055 FLAT), so the
            # engine shouted NO TRADE at a signal it had just issued.
            # DISPLAY FALLBACK (fixed 2026-11-26): `display_systems` is a
            # whitelist built in compute_asset, and the FINAL analysis box
            # was reading richer fields off it than it carries. When
            # `last_result` was cold (first manual ANALYSE, or a lane whose
            # cycle was just rebuilt) `res` came from the whitelist, so
            # `res["direction"]`/`res["confidence"]` were 0 and `tier` was
            # absent — the result box printed `tp: null · tier: null` next to
            # a perfectly valid locked contract. Fill the gaps from the fresh
            # per-TF pass and, failing that, from the LIVE CYCLE (which always
            # has the sealed side + tier), instead of publishing blanks.
            res = dict(self.last_result.get((symbol, tf)) or {})
            _disp = getattr(self, "display_systems", {}).get((symbol, tf)) or {}
            _fresh = res_pack.get(tf) or {}
            _cyc_pre = self.cycles.get((symbol, tf))
            _fill = {
                "direction": (getattr(_cyc_pre, "direction", None)
                              or _fresh.get("direction")
                              or _disp.get("direction")),
                "confidence": (_fresh.get("confidence")
                               if _fresh.get("confidence") is not None
                               else getattr(_cyc_pre, "confidence", None)),
                "tier": (_fresh.get("tier") or _disp.get("tier")
                         or getattr(_cyc_pre, "tier", None)),
                "issue_note": (_fresh.get("issue_note")
                               or _disp.get("issue_note")
                               or getattr(_cyc_pre, "issue_note", None)),
                "gate_ok": (_fresh.get("gate_ok")
                            if _fresh.get("gate_ok") is not None
                            else getattr(_cyc_pre, "gate_ok", None)),
                "atr": _fresh.get("atr"),
                "ladder": _fresh.get("ladder") or _disp.get("ladder"),
            }
            for _k, _v in _fill.items():
                if res.get(_k) is None and _v is not None:
                    res[_k] = _v
            price = self.feed.prices().get(symbol, 0.0)
            atr = float(res.get("atr") or self._atr(self.build_ctx(symbol, tf)))
            # ── THE SIDE THE APP ACTUALLY PUBLISHES (live defect 2026-11-27)
            # The card said "BUY", the note said "side FLIPPED to SELL", and
            # the sealed TP/SL were the BUY ones. Cause: verdict.resolve() is
            # the one place the trend interaction resolves a side (it can
            # FLIP a chase), but this function took only three of its four
            # return values — `side_out` was discarded — and then published
            # `res["direction"]`, which no longer carried the flip because
            # the double-check pass had overwritten it afterwards. So the
            # note and the side described different trades. Resolve ONCE,
            # here, and publish that side with the levels priced from it.
            direction = int(res.get("direction") or 0)
            conf = float(res.get("confidence") or 0.0)
            _side_resolved = direction
            try:
                from .verdict import resolve as _v_resolve
                _ok_r, _vd_r, _nt_r, _side_r = _v_resolve(
                    symbol, tf, direction, conf, res.get("tier"),
                    res.get("issue_note"))
                if int(_side_r or 0) in (-1, 1) and int(_side_r) != direction:
                    _side_resolved = int(_side_r)
                    res["direction_resolved_from"] = direction
                    res["trend_flip"] = {
                        "applied": True, "from": direction,
                        "to": _side_resolved,
                        "why": str(_nt_r or ""),
                        "source": "verdict.resolve",
                    }
                    log.info("side resolved %s/%s %+d -> %+d by the verdict "
                             "rule", symbol, tf, direction, _side_resolved)
            except Exception:
                log.exception("side resolution failed %s/%s", symbol, tf)
                _side_resolved = direction
            if _side_resolved != direction:
                # keep the pre-flip side visible for the UI's divergence note
                if res.get("direction_cci") is None:
                    res["direction_cci"] = direction
                res["direction"] = _side_resolved
                res["signal"] = ("BUY" if _side_resolved > 0
                                 else "SELL")
                direction = _side_resolved
            minutes = TIMEFRAMES[tf]["targets"]
            # ONE source of truth for levels (user: "which is real?"):
            # the LOCKED cycle levels — the same entry/TP/SL the chart,
            # calls table and (armed) trader use. The old finale mixed
            # confidence-projected TPs (conf~0 -> TP = entry) with a
            # borrowed SL from a possibly opposite cycle: the millions
            # of contradictions in one panel. No lock yet -> build ONE
            # direction-aware 2:1 block anchored at the analysis entry.
            cyc = self.cycles.get((symbol, tf))
            from .levels import from_cycle as _lv_from_cycle, freeze as _lv_freeze
            _lv = _lv_from_cycle(cyc) if cyc is not None else None
            if _lv:
                entry_used = float(_lv["entry"])
                tps = [{"k": t.k, "minutes": t.minutes,
                        "level": round(float(t.level), 6)}
                       for t in (cyc.targets or []) if t.level]
                if not tps:
                    tps = [{"k": 1, "minutes": minutes[-1],
                            "level": round(float(_lv["tp"]), 6)}]
                sl = float(_lv["sl"])
                ratio_used = _lv.get("ratio")
                levels_source = "locked contract — frozen entry/TP/SL"
            else:
                entry_used = price
                tps, sl, ratio_used = [], None, None
                if direction != 0 and price > 0:
                    # EVERY horizon gets its own measured rung (25/50/75/100 %
                    # of the TF window). The old preview built ONE rung at the
                    # horizon, so all four TF cards showed an identical
                    # TP/SL/entry ladder — the user's "should be different"
                    # complaint. Each horizon has its own ATR-scaled distance,
                    # and each TF has its own horizons.
                    from .moves_risk import risk_block as _rb
                    for _m in dict.fromkeys(minutes):
                        try:
                            _blk = _rb(price, direction, int(_m),
                                       min(1.0, conf), [], [], [], atr=atr)
                        except Exception:
                            _blk = None
                        _fin = _lv_freeze(price, direction, _blk) \
                            if _blk else None
                        if not _fin:
                            continue
                        tps.append({"k": len(tps) + 1, "minutes": int(_m),
                                    "level": round(float(_fin["tp"]), 6)})
                        # the contract SL is the widest (horizon) rung's
                        if sl is None or int(_m) >= int(minutes[-1]):
                            sl = float(_fin["sl"])
                            ratio_used = _fin.get("ratio")
                levels_source = "preview only — measured levels, no lock, not executable"
            _t_re = time.time()
            # 6.5 ── RECHECK: every data source, every subsystem, then
            # and only then a prediction (user order). The analysis may
            # NOT contradict the locked call, may NOT issue on stale
            # data, broken levels, a split vote, or sub-floor confidence.
            _pa = {"direction": 0, "score": 0.0, "strong": False,
                   "reason": "price action unavailable"}
            try:
                from .price_action import read as _price_action_read
                _pa_state = self.feed.state(symbol)
                _pa = _price_action_read(
                    list(getattr(_pa_state, "times", []) or []),
                    list(getattr(_pa_state, "opens", []) or []),
                    list(getattr(_pa_state, "closes", []) or []),
                    list(getattr(_pa_state, "highs", []) or []),
                    list(getattr(_pa_state, "lows", []) or []),
                    list(getattr(_pa_state, "volumes", []) or []),
                    price=float(price or 0.0), now=time.time())
            except Exception:
                log.exception("price-action read failed for %s|%s", symbol, tf)
            _voters = {
                "fused": int(res.get("direction") or 0),
                "price_action": int(_pa.get("direction") or 0),
                "emotion": int((store.get("emotions") or {}).get(tf, {})
                               .get("polarity") or 0),
                "physics": (1 if (store.get("physics") or {}).get(tf, {})
                            .get("ufp", 0) > 0 else
                            -1 if (store.get("physics") or {}).get(tf, {})
                            .get("ufp", 0) < 0 else 0),
                "news": (1 if (store.get("news") or {}).get("niv", 0) > 0
                         else -1 if (store.get("news") or {}).get("niv", 0) < 0
                         else 0),
                "hedge": (1 if (store.get("hedge") or {}).get(tf, {})
                          .get("combined", 0) > 0 else
                          -1 if (store.get("hedge") or {}).get(tf, {})
                          .get("combined", 0) < 0 else 0),
                # cross-asset spillover / thesis-signed reversal voter:
                # measured IC +0.033 out-of-fold vs −0.036 for the old
                # momentum tilt (backtest/results/ALPHA_LAB.md)
                "spillover": int(((res.get("spillover") or {})
                                  .get("direction")) or 0),
            }
            _st = self.feed.state(symbol)
            _age = (round(time.time() - float(_st.last_tick), 1)
                    if getattr(_st, "last_tick", 0) else None)
            _dir_lv = int(_lv["direction"]) if _lv else direction
            _tp_lv = (float(_lv["tp"]) if _lv else
                      (tps[-1]["level"] if tps else None))
            from .levels import valid as _lv_valid
            _levels_ok = bool(_lv_valid({
                "entry": entry_used, "tp": _tp_lv, "sl": sl,
                "direction": _dir_lv}) if _tp_lv and sl else False)
            _lstats = {}
            try:
                _lstats = self.ledger.stats(symbol, tf) or {}
            except Exception:
                _lstats = {}
            recheck = _recheck_prediction(
                direction=direction, conf=conf,
                floor=float(res.get("asset_floor") or CONFIDENCE_FLOOR),
                feed_age_s=_age, voters=_voters, levels_ok=_levels_ok,
                ratio=ratio_used,
                lock_dir=int(getattr(cyc, "direction", 0) or 0) if cyc else 0,
                tfs_computed=len(res_pack),
                exp_bps=(_lstats.get("exp_bps_recent")
                         if int(_lstats.get("exp_n_recent") or 0) >= 20
                         else _lstats.get("exp_bps")),
                exp_n=int(_lstats.get("exp_n_recent")
                          if int(_lstats.get("exp_n_recent") or 0) >= 20
                          else (_lstats.get("n") or 0)),
                exp_bps_all=_lstats.get("exp_bps"),
                exp_n_all=int(_lstats.get("exp_n_all")
                              or _lstats.get("n") or 0),
                exp_sd=_lstats.get("exp_sd_recent"),
                pa_conflict=bool(
                    _pa.get("strong") and _pa.get("direction")
                    and direction
                    and int(_pa.get("direction")) != int(direction)))
            await self._emit_stage(
                job, "recheck", "done" if recheck["passed"] else "blocked",
                _t_re, time.time(), recheck["verdict"],
                {"checks": recheck["checks"], "voters": _voters})
            # per-asset weightage — every asset runs its OWN weights
            wts = damp_all(self.cci.get_weights(symbol, tf))
            top_w = sorted(wts.items(), key=lambda kv: -abs(kv[1]))[:8]
            ai = job.get("ai") or {}
            # ── floor the PUBLISHED pair at 1:1 (USER RULE) ─────────────
            # Computed here, once, so the card cannot show a target closer
            # than its stop on either the contract path or the preview
            # fallback. Widens the target; never narrows the measured stop.
            _pub_tp = (float(_tp_lv) if _tp_lv else
                       (float(tps[-1]["level"]) if tps else None))
            _pub_sl = round(sl, 6) if sl else None
            if (_pub_tp and _pub_sl and entry_used
                    and int(direction or 0) in (-1, 1)):
                try:
                    _e = float(entry_used)
                    _d = int(direction)
                    _sd = abs(_e - float(_pub_sl))
                    _td = abs(float(_pub_tp) - _e)
                    if _sd > 0 and (_td < _sd or (_pub_tp - _e) * _d <= 0):
                        _pub_tp = round(_e + _d * _sd, 6)
                        final_floor = {"rr_before_floor": round(
                            _td / _sd if _sd else 0.0, 4)}
                    else:
                        final_floor = {}
                except (TypeError, ValueError):
                    final_floor = {}
            else:
                final_floor = {}
            final = {
                "signal": "BUY" if direction > 0 else ("SELL" if direction < 0 else "FLAT"),
                "direction": direction,
                "confidence": round(conf, 4),
                "issued": bool(res.get("issued")),
                "gate_ok": bool(res.get("gate_ok")),
                "issue_note": res.get("issue_note"),
                # STRENGTH (guide §2/§6): the hero printed 0.00 because the
                # systems payload never carried voter agreement at all. It
                # is the share of directional voters that agree with the
                # fused call, in [0,1]; 0 only when nobody voted.
                "strength": round(
                    (sum(1 for v in _voters.values() if v == direction)
                     / max(1, sum(1 for v in _voters.values() if v != 0)))
                    if direction and any(_voters.values()) else 0.0, 4),
                "voters": _voters,
                "price_action": _pa,
                "provisional": not bool(_lv),
                "price": price,
                "entry": round(entry_used, 6),
                "atr": round(atr, 6),
                "tps": tps,
                # THE HEADLINE TP (fixed 2026-11-26): this box never carried
                # a `tp`/`tier` at all — the UI read `tp: null · tier: null`
                # beside a valid sealed contract (`contract_tp`), which reads
                # as "no target" on the one panel the user opens on purpose.
                # Publish the same number the ladder ends at, and the tier
                # the size hint actually resolved to.
                # THE PUBLISHED PAIR IS FLOORED AT 1:1 (USER RULE).
                # When no contract is locked the card falls back to the
                # horizon target from `tps` and a SEPARATELY measured `sl` —
                # two numbers that were never compared, so the board could
                # publish a target closer than the stop. Live proof
                # 2026-11-27: 7 of 16 cells below 1:1 on this path alone
                # (BNB tf1 0.474, BTC tf1/tf3 0.500, ETH tf1/tf2 0.800,
                # ETH tf3 0.545, PAXG tf1 0.543) while the contract path had
                # already been floored. Flooring at the publish point is
                # what makes the rule true of the CARD, not just of the
                # contract behind it.
                "tp": _pub_tp,
                "sl": _pub_sl,
                "rr_floor": final_floor or None,
                "tp_source": ("locked contract — horizon target"
                              if _lv else
                              "preview ladder — measured, no lock"),
                "tier": (res.get("tier")
                         or getattr(cyc, "tier", None) or "watch-only"),
                "verdict_snapshot": {
                    "verdict": (getattr(cyc, "verdict", None)
                                if cyc is not None else None),
                    "gate_ok": (getattr(cyc, "gate_ok", None)
                                if cyc is not None else None),
                    "issue_note": (getattr(cyc, "issue_note", None)
                                   if cyc is not None else None),
                },
                "ratio": ratio_used,
                "levels_source": levels_source,
                "contract_state": ("locked" if _lv and cyc and cyc.level_contract
                                   else "legacy-locked" if _lv else "preview-only"),
                "contract_id": ((cyc.level_contract or {}).get("id")
                                if _lv and cyc else None),
                "level_contract": (dict(cyc.level_contract)
                                   if _lv and cyc and cyc.level_contract else None),
                "contract_tp": float(_lv["tp"]) if _lv else None,
                "contract_sl": float(_lv["sl"]) if _lv else None,
                "execution_eligible": bool(
                    _lv and cyc and cyc.execution_eligible and cyc.issued
                    and cyc.tier != "watch-only" and not cyc.contract_tp_hit
                    and not cyc.sl_observed and not cyc.pending_reset
                    and getattr(self.feed, "mode", None) == "binance"),
                "raw_signal": ("BUY" if direction > 0 else
                               "SELL" if direction < 0 else "FLAT"),
                "recheck": recheck,
                # the confidence audit trail, on the ANALYSE result too —
                # the CONFIDENCE CHAIN panel lives in this box and had no
                # way to name a stage that moved the number (2026-11-27)
                "conf_trace": res.get("conf_trace"),
                "confidence_chain": (res.get("confidence_chain")
                                     or res.get("published_chain")),
                "snr": res.get("snr"),
                "noise_floor": res.get("noise_floor"),
                "asset_floor": res.get("asset_floor"),
                "prediction": ({"direction": direction,
                                "confidence": round(conf, 4),
                                "entry": round(entry_used, 6),
                                "tp": _tp_lv, "sl": round(sl, 6) if sl else None,
                                "ratio": ratio_used,
                                "kind": "locked-contract" if _lv and cyc and cyc.level_contract else "analysis-preview",
                                "contract_id": ((cyc.level_contract or {}).get("id")
                                                if _lv and cyc else None),
                                "execution_eligible": bool(
                                    _lv and cyc and cyc.execution_eligible and cyc.issued
                                    and cyc.tier != "watch-only" and not cyc.contract_tp_hit
                                    and not cyc.sl_observed and not cyc.pending_reset
                                    and getattr(self.feed, "mode", None) == "binance")}
                               ),   # preview is never executable
                "weightage": [{"id": str(k), "w": round(v, 5)}
                              for k, v in top_w],
                "weightage_note": ("per-asset, per-timeframe weights — "
                                   "every asset analyses with its own mix"),
                "ladder": res.get("ladder"),
                # the four brains and the affect model, verbatim from this
                # analysis' own tf — the panel must show what the decision
                # actually used, not a re-read of the last cycle
                "brains": res.get("brain"),
                "emotions": {k: v for k, v in (res.get("emotion") or {}).items()
                             if k != "emotions"} | {
                    "channels": (res.get("emotion") or {}).get("emotions"),
                    "directions": (res.get("emotion") or {}).get("directions"),
                },
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
            # §5 — the four-agent board votes on EVERY analysis.  The
            # module existed but vote() had never been called (the
            # snapshot's committee block stayed {vote_count: 0, last:
            # null} for every prediction the engine ever made).  Inputs:
            # this run's market bundle + emotion pass, the tape regime
            # and the cell's metacognition status.  The contribution
            # block below reads COMMITTEE.last right after this vote.
            try:
                from .agent_committee import COMMITTEE
                from .metacognition import MC
                from .regime import classify_tape
                _mkt = self._build_market(symbol)
                _emf = ((store.get("emotions") or {}).get(tf) or {}) \
                    .get("_full")
                _cl = list(_mkt.get("closes") or [])
                _regime = classify_tape(_cl[-480:]) if _cl else None
                _st = MC.status(symbol, tf)
                COMMITTEE.vote(_mkt, _emf, _regime, metacog_status=_st,
                               pillar_report=None,
                               allows_capital=MC.allows_capital(symbol, tf),
                               symbol=symbol)
            except Exception:
                pass
            try:
                final["committee"] = self._committee_contribution(
                    symbol, tf, final)
            except Exception:
                pass
            try:
                final["metacog"] = _metacog_block()
            except Exception:
                pass
            if not recheck["passed"]:
                # advisory only: the call stands; the failed checks are
                # printed so the arithmetic can be audited (doc §1)
                final["recheck_advisory"] = recheck["verdict"]
            # The honest verdict: a side can be ISSUED for grading while
            # never clearing the issue floor. The UI and the sealed lock read
            # THIS field, so no panel can advertise "ISSUED" at conf 0.20.
            # REGIME CONFLICT (user order): the TF card's own 5-state
            # chip said IMPULSE_BULL/BEAR opposite to this cell's
            # certified side ("it said impulse bear but gave bullish").
            # The window record stays CONTINUOUS — the certified side is
            # still sealed and graded (no selective skipping) — but the
            # app must not advertise or route the opposite call:
            # gate_ok=False → verdict NO CALL, tier watch-only (below)
            # → no paper/live entry, and the reason names BOTH sides.
            _conflict = self.regime_conflict(symbol, int(direction or 0))
            if _conflict is not None:
                final["regime_conflict"] = _conflict
                # NOT A NO CALL ANY MORE (2026-11-26). This used to set
                # gate_ok=False, which forced verdict NO CALL and tier
                # watch-only: a window with a measured, sealed, graded side
                # was published as nothing at all. The user's standing
                # directive is "raise the win rate WITHOUT abstention", and
                # the disagreement it was reacting to is now handled where it
                # was MEASURED — the certified side that chases the recent
                # impulse is flipped by app/trend_gate.py (56.8 % vs 52.4 %
                # on 861 windows). A regime chip is a STATE, not a measured
                # predictor: flipping on it would invent 2y of evidence we do
                # not have, so this gate keeps its real job — it demotes
                # EXECUTION (no paper/live order) and says so in the note,
                # while the side, the levels, the contract and the grading
                # all stand. Silence was never the honest version of a
                # disagreement; a published, graded, explained side is.
                # ROUTING IS BLOCKED BY `execution_eligible`, WHICH IS THE
                # FIELD THE TRADER ACTUALLY READS — the published tier is a
                # SIZE HINT and stays as measured. Demoting the tier here as
                # well double-counted one decision into two visible fields
                # (the board read "watch-only" and stopped sizing a call it
                # also stopped routing: the same signal, said twice, and the
                # window read like a non-call).
                final["execution_eligible"] = False
                if cyc is not None:
                    cyc.execution_eligible = False
                final["issue_note"] = (
                    f"REGIME DISAGREEMENT — the 5-state chip reads "
                    f"{_conflict['state']} (conf "
                    f"{_conflict.get('regime_conf')}) while the certified "
                    f"side is {'BUY' if direction > 0 else 'SELL'}. The side "
                    f"stands and is graded; only ORDER ROUTING is held back "
                    f"until the tape agrees with itself. A withheld call is "
                    f"not a safer call — it is an ungraded one")
            final["verdict"] = ("ISSUE" if (final.get("gate_ok")
                                            and recheck["passed"])
                                else "WATCH" if final.get("direction")
                                else "NO CALL")
            # SEAL THE PREDICTION. This is the user requirement: pressing
            # ANALYSE locks what the app just said, writes it to history and
            # lets settlement teach the learners. A refused lock is printed
            # with every failing check instead of silently doing nothing.
            # the thought trace is part of the evidence, so it is produced
            # BEFORE the seal and stored in the lock's evidence dict
            _cog = None
            try:
                _cog = self.cognitive(symbol, tf)
            except Exception:
                log.exception("cognition pre-compute failed for %s|%s",
                              symbol, tf)
            try:
                from . import mandate_cfg as _mcfg_t
                _tmode = (str(self.agents.settings.get("system_mode", "old"))
                          if _mcfg_t.mode(symbol) not in ("old", "new")
                          else _mcfg_t.mode(symbol))
                final["prediction_lock"] = self._seal_analysis_lock(
                    symbol, tf, final, store, res, cyc,
                    signature=(_cog or {}).get("signature"),
                    target=("forecast price at horizon end"
                            if _tmode == "new"
                            else "measured TP/SL ladder"))
                # ── STALE-ANCHOR RECOVERY (live defect 2026-11-27) ──────
                # The journal refuses a seal whose entry anchor is not the
                # live market (or whose barriers the market has already
                # passed). Re-anchoring is the repair the user asked for:
                # the SAME call, levels re-measured from the live price,
                # sealed immediately — never an unearned loss and never a
                # silenced window.
                _pl = final.get("prediction_lock") or {}
                if (not _pl.get("locked")
                        and not _pl.get("already")
                        and _pl.get("status") == "refused"
                        and any("anchor" in str(w) or "seal time" in str(w)
                                for w in (_pl.get("why") or []))
                        and cyc is not None):
                    _re = None
                    _live_r = 0.0
                    try:
                        _live_r = float(self.feed.prices().get(symbol, 0.0)
                                        or 0.0)
                        _dir_r = int(final.get("direction") or 0)
                        _sl_bps_r = float((getattr(cyc, "risk", {}) or {})
                                          .get("sl_bps") or 0.0)
                        if _live_r > 0 and _dir_r in (-1, 1) and _sl_bps_r > 0:
                            from .moves_risk import risk_block as _risk_block_r
                            _fresh_r = _risk_block_r(
                                _live_r, _dir_r,
                                int(TIMEFRAMES[tf]["minutes"]),
                                min(1.0, abs(float(final.get("confidence")
                                                   or 0.0)) / 0.35),
                                list(self.feed.state(symbol).closes or []),
                                list(self.feed.state(symbol).highs or []),
                                list(self.feed.state(symbol).lows or []),
                                atr=float(res.get("atr") or 0.0),
                                spread_bps=float(_pl.get("_spread") or 0.0),
                                bar_minutes=1.0)
                            _re = self.predlocks.seal(
                                symbol, tf,
                                direction=_dir_r,
                                entry=float(_fresh_r["entry"]),
                                tp=float(_fresh_r["tp"]),
                                sl=float(_fresh_r["sl"]),
                                confidence=float(final.get("confidence")
                                                 or 0.0),
                                minutes=int(TIMEFRAMES[tf]["minutes"]),
                                gate_ok=bool(final.get("gate_ok")),
                                recheck_ok=(final.get("recheck") or {})
                                .get("passed"),
                                reason=(f"re-anchored on the live price "
                                        f"{_live_r:g} — {final.get('verdict')}"),
                                evidence={"reanchored_from": {
                                    "why": _pl.get("why"),
                                    "entry": float(
                                        (getattr(cyc, "level_contract", {})
                                         or {}).get("entry") or 0.0)}},
                                target="re-anchored on the live price",
                                source=str((self.analyse_job or {})
                                           .get("source") or "analyse"),
                                system_mode=str(self.agents.settings.get(
                                    "system_mode", "old")),
                                now_price=_live_r)
                    except Exception:
                        log.exception("stale-anchor re-seal failed for %s|%s",
                                      symbol, tf)
                        _re = None
                    if _re is not None:
                        _re["reanchored"] = True
                        _re["refused_before"] = _pl.get("why")
                        final["prediction_lock"] = _re
                        log.warning("lock re-anchored %s|%s on the live "
                                    "price %g (was refused: %s)",
                                    symbol, tf, _live_r,
                                    "; ".join(_pl.get("why") or []))
            except Exception:
                log.exception("prediction lock failed for %s|%s", symbol, tf)
                final["prediction_lock"] = {
                    "locked": False, "status": "refused",
                    "why": ["the lock writer raised — see server log"]}
            # publish the same verdict + lock on the CYCLE CARD, so the hero
            # and the timeframe cards read one answer (not "ISSUED" + a lock
            # that says NO CALL)
            if cyc is not None:
                # ONE derivation for the GRC card and the ANALYSE card: the
                # analyse final and the cycle badge come from the same rule,
                # so the hero can never print a verdict its cycle contradicts.
                _ok_a, _verdict_a, _note_a = public_verdict(
                    symbol, tf, int(final.get("direction") or 0),
                    float(final.get("confidence") or 0.0),
                    cyc.tier, final.get("issue_note") or cyc.issue_note)
                final["gate_ok"] = _ok_a
                final["issue_note"] = _note_a
                if _verdict_a != "ISSUE" and str(final.get("verdict")) == "ISSUE":
                    final["verdict"] = _verdict_a
                cyc.verdict = str(final.get("verdict") or _verdict_a)
                cyc.gate_ok = bool(final.get("gate_ok"))
                cyc.issue_note = str(final.get("issue_note") or _note_a)
                cyc.trend_gate = _trend_evidence(
                    symbol, tf, int(final.get("direction") or 0))
                cyc.regime_conflict = _conflict
                cyc.prediction_lock = dict(final.get("prediction_lock") or {})
                if _conflict is not None:
                    # routing held on the next tick; the tier, the side and
                    # the sealed contract are untouched (see the note above)
                    cyc.execution_eligible = False
            # COGNITION: produce and record the thinking trace for the call
            # that was just made (signature + metacognitive snapshot). Failure
            # here must never cost the user their analysis.
            try:
                self._think_and_record(symbol, tf, final, _cog)
            except Exception:
                log.exception("cognition record failed for %s|%s", symbol, tf)
            job["result"] = final
            job["status"] = "done"
            # A FLAT call has no rungs by construction (`tps` stays empty
            # because the preview block only builds levels for a signed
            # direction), so `tps[0]` raised IndexError on every flat tape and
            # threw away an analysis that had already completed — the user saw
            # "analysis is not happening" while the engine had in fact
            # finished and discarded the result. A summary line must never be
            # able to destroy a result, so it degrades instead.
            _tp1 = (f"TP1 {tps[0]['level']}" if tps else
                    "TP1 — (no levels: flat call)")
            _sl = (f"SL {final['sl']}" if final.get("sl") is not None
                   else "SL —")
            await self._emit_stage(
                job, "final", "done", t0, time.time(),
                f"{final['signal']} · conf {final['confidence']:.3f} · "
                f"{_tp1} · {_sl} · "
                f"total {final['elapsed_ms']:.0f} ms", final)
            await self.emit({"type": "analyse_done", "symbol": symbol,
                             "asset": symbol, "tf": tf, "timeframe": tf,
                             "request_id": job.get("request_id"),
                             "result": final})
        except Exception as exc:
            job["status"] = "error"
            job["error"] = repr(exc)
            log.exception("analysis failed for %s", symbol)
            await self.emit({"type": "analyse_done", "symbol": symbol,
                             "asset": symbol, "tf": tf, "timeframe": tf,
                             "request_id": job.get("request_id"),
                             "result": None,
                             "error": repr(exc)})

    async def _analyse_ai_review(self, job: dict, store: dict,
                                 tf: str) -> None:
        """Stage 6: every configured agent reviews the assembled analysis.

        This stage used to consult exactly two things — a Gemini key and an
        installed ``llama_cpp`` — so on a host with neither it printed
        "no AI configured" even when a working Ollama daemon was answering on
        11434, and the whole Agents tab looked dead. It now goes through
        ``AgentManager.review()``, which walks the full ladder (cloud keys →
        local GGUF → Ollama) and returns a payload that always explains
        itself, including which backend answered.

        Review is advisory: it can flag errors and propose a correction, and
        a correction at high confidence is printed in the result, but the
        deterministic ladder underneath is never overwritten.
        """
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
            "You are reviewing an intraday market analysis BEFORE it is "
            "shown to a trader. Find mistakes or contradictions in this "
            "analysis (sign conflicts, stale/absurd numbers, risk that does "
            "not match volatility) and give your own call. "
            'Reply ONLY with JSON: {"errors": ["..."], "fixes": ["..."], '
            '"direction": 1|-1|0, "confidence": 0.0..1.0, '
            '"reason": "one sentence"}. Data:\n'
            + json.dumps(payload, default=str)[:4000]
        )
        # OFFLINE/SLOW-AGENT BUDGET (user order: "agents mostly,
        # always or partially offline"): the advisory review gets a
        # hard 8 s wall — a dead cloud endpoint (12 s transport
        # timeout each) or a CPU-bound llama must never stall the
        # analysis for tens of seconds.  Timeout ⇒ honest skip note.
        try:
            review = await asyncio.wait_for(
                self.agents.review(
                    symbol, tf, store,
                    result={"direction": int(
                        (self.last_result.get((symbol, tf)) or {}).get(
                            "direction") or 0)}),
                timeout=8.0)
        except (asyncio.TimeoutError, TimeoutError):
            review = {"available": False, "backend": "",
                      "note": ("agent review exceeded the 8 s budget — "
                               "slow/offline backend skipped (deterministic "
                               "pipeline only)"),
                      "reviews": {}, "correction": None}
        except Exception as exc:                     # never fail the stage
            log.exception("agent review failed")
            review = {"available": False, "backend": "",
                      "note": f"review stage error: {exc!r}"[:200],
                      "reviews": {}, "correction": None}

        raw_reviews = review.get("reviews") or {}
        reviews: dict[str, dict] = {}
        for name, rv in raw_reviews.items():
            if rv.get("ok"):
                fed = {"direction": rv.get("direction"),
                       "confidence": rv.get("confidence"),
                       "insight": rv.get("reason")}
                parsed = self._parse_ai_review(rv.get("reason") or "", fed)
                parsed["errors"] = list(rv.get("errors") or [])
                parsed["backend"] = rv.get("backend") or name
                reviews[name] = parsed
            else:
                reviews[name] = {"ok": False,
                                 "error": str(rv.get("error") or "failed")}

        t1 = time.time()
        if not review.get("available"):
            note = str(review.get("note") or "no AI available")
            await self._emit_stage(job, "ai_review", "skipped", t0, t1,
                                   note, {"backend": "", "reviews": reviews})
            job["ai"] = {"available": False, "backend": "",
                         "note": note, "reviews": reviews,
                         "correction": None}
            return

        # an AI correction only counts at confidence >= 0.60 (printed, never
        # silent — the deterministic ladder stays visible underneath)
        correction = review.get("correction")
        if not correction:
            for name, rv in reviews.items():
                if (rv.get("ok") and abs(int(rv.get("direction") or 0)) == 1
                        and float(rv.get("confidence") or 0) >= 0.60):
                    correction = {
                        "by": name,
                        "direction": int(rv["direction"]),
                        "confidence": float(rv["confidence"]),
                        "reason": rv.get("reason", "")}
                    break
        job["ai"] = {"available": True,
                     "backend": review.get("backend") or "",
                     "note": review.get("note") or "",
                     "reviews": reviews, "correction": correction}
        bits = []
        for name, rv in reviews.items():
            if rv.get("ok"):
                n_err = len(rv.get("errors") or [])
                verdict = ("CORRECTS to " + ("BUY" if rv["direction"] > 0
                                             else "SELL")
                           if correction and correction.get("by") == name
                           else "concurs")
                bits.append(f"{name}: {n_err} issue(s), {verdict}")
            else:
                bits.append(f"{name}: {rv.get('error', 'failed')}")
        await self._emit_stage(
            job, "ai_review", "done", t0, t1,
            (" · ".join(bits) or "no review") +
            f" [{review.get('backend') or 'agent'}]",
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

    def _sig_weight(self) -> float:
        """Signature voter weight: STRICT floor 0.15 until >=500
        walk-forward graded samples prove accuracy > 52% (user
        directive) — then it may earn 0.6, never more."""
        try:
            n = hits = 0
            for s in SYMBOL_ORDER:
                for t in TF_ORDER:
                    ring = list(self.hist.ring(
                        self.hist.tf_key(s, t, "adv_sig"), 600))
                    n += len(ring)
                    hits += sum(ring)
            if n >= 500 and hits / n > 0.52:
                return 0.6
        except Exception:
            pass
        return 0.15

    def _adv_weight_scales(self, symbol: str, tf: str) -> dict:
        """SIGNED per-module scales from each module's own
        walk-forward accuracy ring for THIS asset+TF — the same
        rule for every asset, timeframe and module (nothing is
        special, nothing is hard-coded):
          acc 65% -> +1.5x, acc 50% -> +0.5x (neutral floor),
          acc 35% -> -1.5x (anti-predictive: votes flipped).
        s = clamp((acc - 0.5) * 6, -1.5, 1.5), magnitude floored
        at 0.5 so every voter keeps voting; arms at n>=40 graded
        samples, before that the module votes raw at scale +1.
        Signature joins the same rule once the engine-wide
        adv_sig count reaches the user-mandated 500 graded
        samples; until then it stays at its floor weight, raw."""
        scales: dict[str, float] = {}
        try:
            sig_ready = False
            try:
                _sn = sum(len(list(self.hist.ring(self.hist.tf_key(
                    s2, t2, "adv_sig"), 600)))
                    for s2 in SYMBOL_ORDER for t2 in TF_ORDER)
                sig_ready = _sn >= 500
            except Exception:
                sig_ready = False
            for mod in ("kalman", "fracdiff", "bocpd", "entropy",
                        "ou_hurst", "jumps", "pairs_ou", "wavelet",
                        "rmt_residual", "signature"):
                if mod == "signature" and not sig_ready:
                    continue
                ring = list(self.hist.ring(
                    self.hist.tf_key(symbol, tf, f"adv_{mod}"), 600))
                if len(ring) >= 40:
                    acc = sum(ring) / len(ring)
                    # deadband 47-53%: inside it the estimate is
                    # within noise of a coin flip — keep the raw
                    # sign at floor weight instead of flipping on
                    # noise. Outside it, scale/flip by evidence.
                    if acc < 0.47:
                        scales[mod] = -max(0.5, min(
                            1.5, abs((acc - 0.5) * 6.0)))
                    elif acc > 0.53:
                        scales[mod] = max(0.5, min(
                            1.5, (acc - 0.5) * 6.0))
                    else:
                        scales[mod] = 0.5
        except Exception:
            pass
        return scales

    def _adv_blend_k(self, symbol: str, tf: str) -> float:
        """Scale-matched blend coefficient for the advanced vote:
        0.6 * median|score| of the last 60 windows, bounded
        [0.04, 0.15] (0.08 before any history) — the vote spans
        +/-1 while CCI scores live in +/-0.1..0.3, so a flat
        coefficient would let one layer outweigh the whole score."""
        hist = [abs(v) for v in
                (self.score_hist.get((symbol, tf), []) or [])
                if v is not None]
        if len(hist) >= 10:
            ordered = sorted(hist)
            med = ordered[len(ordered) // 2]
            return max(0.04, min(0.15, 0.6 * med))
        return 0.08

    def _advanced_for(self, symbol: str, tf: str, ctx) -> dict:
        """Advanced mathematics layer (app/advanced.py) — computed
        once per (asset, TF, bar); failures degrade to a neutral
        block, never to a crashed prediction."""
        try:
            cache = getattr(self, "_adv_cache", None)
            if cache is None:
                cache = self._adv_cache = {}
            st = self.feed.state(symbol)
            key = (symbol, tf, len(st.closes or []),
                   (st.times or [0])[-1] if st.times else 0)
            if key in cache:
                return cache[key]
            from .advanced import advanced_block, rmt_block
            others = {}
            series = {}
            for s2 in SYMBOL_ORDER:
                st2 = self.feed.state(s2)
                closes = list(st2.closes or [])
                series[s2] = closes
                if s2 != symbol:
                    others[s2] = closes
            rmt_cache = getattr(self, "_rmt_cache", None)
            if rmt_cache is None:
                rmt_cache = self._rmt_cache = {}
            rkey = int(time.time() // 60)
            if rkey not in rmt_cache:
                rmt_cache.clear()
                rmt_cache[rkey] = rmt_block(series)
            blk = advanced_block(
                symbol, tf, ctx, others=others,
                full_closes=series.get(symbol) or list(ctx.P or []),
                sig_weight=self._sig_weight(), rmt=rmt_cache[rkey],
                weight_scales=self._adv_weight_scales(symbol, tf),
                minutes=int(TIMEFRAMES[tf]["minutes"]))
            cache.clear()
            cache[key] = blk
            last = getattr(self, "_adv_last", None)
            if last is None:
                last = self._adv_last = {}
            last[(symbol, tf)] = blk
            return blk
        except Exception:
            log.exception("advanced block failed %s/%s", symbol, tf)
            return {"vote": 0.0, "signs": {}, "modules": {},
                    "soft_gate": 1.0, "rqa_mult": 1.0, "har_rv": {},
                    "evt": {}, "rmt": None, "size_mult": 1.0,
                    "weights": {}}

    def advanced_history(self, symbol: str, tf: str) -> dict:
        """Per-module walk-forward accuracy (adv_* rings) — the
        honest scoreboard for the advanced layer: every module
        report its settled sample count and hit rate, and the
        adaptive weight scale the engine currently applies."""
        mods = {}
        try:
            scales = self._adv_weight_scales(symbol, tf)
            for mod in ("kalman", "fracdiff", "bocpd", "entropy",
                        "ou_hurst", "jumps", "pairs_ou", "wavelet",
                        "rmt_residual", "signature"):
                ring = list(self.hist.ring(self.hist.tf_key(
                    symbol, tf, f"adv_{mod}"), 600))
                n = len(ring)
                mods[mod] = {
                    "n": n,
                    "acc": (round(sum(ring) / n, 4) if n else None),
                    "scale": round(scales.get(mod, 1.0), 3)}
            sig_n = len(list(self.hist.ring(self.hist.tf_key(
                symbol, tf, "adv_sig"), 600)))
            cq = list(self.hist.ring(self.hist.tf_key(
                symbol, tf, "conf_q"), 400))
            return {"symbol": symbol, "tf": tf, "modules": mods,
                    "signature_samples": sig_n,
                    "signature_floor": sig_n < 500,
                    "conformal": {"q_bps": (round(cq[-1] * 1e4, 1)
                                            if cq else None),
                                  "n": len(cq)}}
        except Exception:
            log.exception("advanced_history failed")
            return {"symbol": symbol, "tf": tf, "modules": {},
                    "signature_samples": 0, "signature_floor": True,
                    "conformal": {"q_bps": None, "n": 0}}

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

        core_ids = list(range(1, 36))              # F1..F35 panels (35)
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
            "weights": {str(fid): round(damp_all(
                self.cci.get_weights(symbol, tf)).get(fid, 0.0), 5)
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

    def _dc_display(self, direction, confidence, pcv, brain_pol) -> dict | None:
        """Double-check verdict for DISPLAY before the GRC's authoritative
        `_apply_agents` pass (which also applies the mutations). Analyse
        windows never call `_apply_agents` — without this the seams panel
        showed a bare dash forever."""
        try:
            votes = {"pcv": 1 if pcv > 0 else (-1 if pcv < 0 else 0),
                     "agent": int((self.last_agent_signal or {}).get(
                         "direction") or 0),
                     "brain": int(brain_pol or 0)}
            return self.cci.double_check(int(direction or 0),
                                         float(confidence or 0.0), votes)
        except Exception:
            return None

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
            # brain.md §6: × (1 + w_brain × vote × conf) with
            # w_brain = 0.25 × hindsight, 0 when the connectome is absent
            factor = 1.0
            _factors = []
            if gw > 0 and gd > 0:
                factor *= 1.0 + gw * gd * res.get("base_confidence", 0.0)
                _factors.append(f"agent ×{factor:.3f}")
            w_fw = self.brain_fusion_weight(symbol, tf)
            fv = self._fw_last_vote.get((symbol, tf), 0.0)
            if w_fw > 0 and fv:
                _fwf = 1.0 + w_fw * fv * res.get("base_confidence", 0.0)
                factor *= _fwf
                _factors.append(f"brain vote {fv:+.3f} ×{w_fw:.2f} → {_fwf:.3f}")
            # mandate cells trade the side the 2y measurement selected;
            # the GRC fusion re-ladder / double-check override must not
            # silently move it (the walk measured compute_asset output)
            from . import mandate_cfg as _mcfg_g
            _md_lock = (_mcfg_g.cfg(symbol) is not None
                        and _mcfg_g.allows(symbol, tf))
            if factor != 1.0 and not _md_lock:
                res["cci_adjusted"] = res["cci_adjusted"] * factor
                # ── THE PUBLISHED FUSION MUST REACH THE CONFIDENCE
                #    (fixed 2026-11-26) ─────────────────────────────────
                # The confidence chain was computed earlier from the
                # pre-fusion CCI, and this block then scales the CCI by the
                # agent/brain factor WITHOUT recomputing it. Live proof:
                # PAXG tf4 published `cci_adjusted −0.1436` while its own
                # chain read step 1 = `|+0.0000| / floor 0.020 = snr 0.00`
                # and published confidence 0.020 — a card whose confidence
                # describes a number it is not showing, in the one panel
                # built to audit that. (It also meant a window with a real
                # −0.14 net bearish signal could never clear the floor.)
                # The chain is recomputed on the fused CCI so the two agree
                # by construction, and `fusion_conf` records the shift.
                try:
                    _pre_c = float(res.get("confidence") or 0.0)
                    _fk = res.get("fusion_inputs") or {}
                    if not _fk:
                        # no inputs were carried (an older cycle shape) —
                        # skip rather than recalibrate against DEFAULTS,
                        # which would silently invent a different chain.
                        raise KeyError("fusion_inputs not carried")
                    _beh_pre = res.get("behavior") or {}
                    _fcal = calibrate_confidence(
                        {**(res.get("_pack") or {}),
                         "cci_adjusted": float(res["cci_adjusted"]),
                         "drhe": float(res.get("drhe") or 1.0)},
                        experience_quality=_fk.get("experience_quality", 0.5),
                        p_dominant=_fk.get("p_dominant", 0.0),
                        state_aligned=_fk.get("state_aligned", False),
                        rtp=_fk.get("rtp", 0.0),
                        pcv=_fk.get("pcv", 0.0),
                        wfcp=_fk.get("wfcp", 1.0),
                        drpt=_fk.get("drpt", 0.0),
                        gspi=_fk.get("gspi", 0.0),
                    )
                    # the asset-personality temper ran AFTER the original
                    # chain and is a stated multiplication (0.78–1.12) —
                    # apply the SAME factor so the recompute is a re-run of
                    # the same formula, not a second opinion
                    _bfac = float(_beh_pre.get("factor") or 1.0)
                    if _bfac != 1.0:
                        _fcal = dict(_fcal)
                        _fcal["confidence"] = (
                            min(1.0, float(_fcal["confidence"]) * _bfac)
                            if _bfac < 1.0 else
                            min(1.0, float(_fcal["confidence"])
                                * (1.0 + (_bfac - 1.0) * 0.5)))
                    res["confidence"] = _fcal["confidence"]
                    res["confidence_base_snr"] = _fcal["base_confidence"]
                    res["confidence_modulator"] = _fcal["modulator"]
                    res["confidence_chain"] = _fcal["steps"]
                    res["snr"] = _fcal["snr"]
                    res["noise_floor"] = _fcal["noise_floor"]
                    res["fusion_conf"] = {
                        "before": round(_pre_c, 4),
                        "after": round(float(_fcal["confidence"]), 4),
                        "factor": round(float(factor), 4),
                    }
                except Exception:
                    log.exception("fusion confidence recompute failed %s/%s",
                                  symbol, tf)
                _lad2, _src2, _fl2 = self._direction_ladder(
                    symbol, tf, float(res["cci_adjusted"]),
                    (res.get("breakdown") or {}).get("components") or [],
                    res.get("brain") or {},
                    self.prev_cci_dir.get((symbol, tf), 0))
                res["direction"] = _lad2
                res["ladder"] = {"source": _src2, "noise_floor": round(_fl2, 5),
                                 "score": round(float(res["cci_adjusted"]), 5)}
                res["fusion_factors"] = _factors
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
                # the REAL circuit's vote when connected (brain.md §6),
                # legacy consensus only as the pre-connectome fallback
                "brain": (int(np.sign(fv)) if fv
                          else int(res.get("brain", {}).get("polarity", 0))),
            }
            dc = self.cci.double_check(res["direction"], res["confidence"], votes)
            if dc["label"] == "mixed":
                dc["message"] = "MIXED SIGNALS"
            res["double_check"] = dc
            if not dc["issued"]:
                # disagreement is shown, never a veto (docs11-doublecheck
                # §5: pick the stronger side, display both) — cap the
                # tier at quarter as degraded evidence instead
                res["tier"] = "quarter" if res.get("tier") in ("full", "half") else res.get("tier", "watch-only")
                res["degraded_evidence"] = True
            _dc_before = float(res.get("confidence") or 0.0)
            res["confidence"] = dc["confidence"]
            try:
                _t = (res.setdefault("conf_trace", []))
                if abs(float(dc["confidence"]) - _dc_before) > 1e-9:
                    _t.append({
                        "stage": "4-vote double-check",
                        "before": round(_dc_before, 4),
                        "after": round(float(dc["confidence"]), 4),
                        "why": str(dc.get("label") or "")})
            except Exception:
                pass
            if dc["label"] == "overridden" and not _md_lock:
                res["direction"] = dc["direction"]
            # The previous-signal cache must match the final post-agent /
            # double-check direction used by the next learning pass —
            # but a mandate cell's ladder continuity stays on the RAW
            # side compute_asset already stored (the rule is defined
            # against the raw ladder, not against itself).
            if not _md_lock:
                self.prev_cci_dir[(symbol, tf)] = int(res.get("direction") or 0)
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
        # Do not rewrite the settlement lock from a later live recompute.
        # Its side/confidence/reason are the frozen prediction for this W;
        # only a new cycle or an explicit emergency re-lock may replace it.

    def emergency_relock(self, symbol: str, tf: str = "tf1",
                         reason: str = "", *, automatic: bool = False) -> dict:
        """Re-anchor an existing lock after a structural invalidation.

        WINDOW FREEZE (audit §1.3): an active lock is immutable until
        **both** (a) the move from its entry exceeds ``RELOCK_MIN_MOVE_ATR``
        (3.0 ATR) and (b) ``RELOCK_MIN_AGE_S`` (15 min) have elapsed since
        the lock. On top of the freeze, every re-anchor of an active lock
        observes a 15-minute lock-age cooldown. Both manual and automatic
        re-locks require a measurable ATR and a move STRICTLY greater than
        3 ATR. Manual drills with no open entry (audit-row only) keep
        their explicit behaviour. All paths share the six-per-hour N-8 cap
        and the 30 s new-lock halt.
        """
        now = time.time()
        hour = [r for r in self.news.emergency_history
                if r.get("asset") == symbol and now - r.get("ts", 0.0) < 3600.0]
        if len(hour) >= 6:
            self.news.stats_bugs.append({
                "at": now, "code": "N-8",
                "detail": f">6 emergency re-locks in 1 h for {symbol} — BUG"})
            return {"ok": False,
                    "error": "N-8: >6 emergency re-locks in one hour — bug, not a signal"}
        try:
            _last_emergency = float(self.news.last_emergency.get(symbol) or 0.0)
        except (TypeError, ValueError, OverflowError):
            _last_emergency = 0.0
        _cooldown_left = (EMERGENCY_RELOCK_COOLDOWN_S
                          - (now - _last_emergency))
        if _cooldown_left > 0.0:
            return {"ok": False, "cooldown": True,
                    "cooldown_s": round(_cooldown_left, 1),
                    "error": "emergency re-lock cooldown active"}
        price = float(self.feed.prices().get(symbol) or 0.0)
        if price <= 0:
            st0 = self.feed.state(symbol)
            price = float(st0.closes[-1] if st0.closes else 0.0)
        lock = self.locks.get((symbol, tf))
        _old_entry = float((lock or {}).get("price") or 0.0)
        if price <= 0 and _old_entry > 0:
            # never overwrite a REAL entry with0 (it would force a skip)
            return {"ok": False,
                    "error": "no price yet — emergency re-lock refused"}
        # ── WINDOW FREEZE (audit §1.3) ──────────────────────────────────
        # Any re-anchor of an ACTIVE lock must clear both bars: a genuine
        # structural invalidation (>3 ATR away from the locked entry) AND
        # 15 minutes of window elapsed. Without this the news trigger
        # re-locked every 10–30 s, moving the entry while the ladder
        # stayed behind — four entries, four ladders, one window.
        if lock is not None and _old_entry > 0 and price > 0:
            try:
                _atr_freeze = float(self._atr(self.build_ctx(symbol, tf)))
            except Exception:
                _atr_freeze = 0.0
            try:
                _locked_ts = float(lock.get("locked_at") or 0.0)
            except (TypeError, ValueError, OverflowError):
                _locked_ts = 0.0
            _age_s = (now - _locked_ts) if _locked_ts > 0 else 0.0
            _atr_known = math.isfinite(_atr_freeze) and _atr_freeze > 0
            _move_atr = (abs(price - _old_entry) / _atr_freeze
                         if _atr_known else 0.0)
            if _atr_known and _move_atr <= RELOCK_MIN_MOVE_ATR:
                return {"ok": False, "frozen": True, "skipped": True,
                        "move_atr": round(_move_atr, 3),
                        "required_move_atr": RELOCK_MIN_MOVE_ATR,
                        "error": f"window frozen — move {_move_atr:.2f} ATR "
                                 f"must exceed {RELOCK_MIN_MOVE_ATR:.1f} ATR"}
            if _age_s < RELOCK_MIN_AGE_S:
                return {"ok": False, "frozen": True, "skipped": True,
                        # the age bar IS a cooldown: surfaced as one so
                        # callers (and the 68bbb922 pin) can read it as such
                        "cooldown": True,
                        "cooldown_s": round(RELOCK_MIN_AGE_S - _age_s, 1),
                        "move_atr": round(_move_atr, 3),
                        "age_s": round(_age_s, 1),
                        "required_age_s": RELOCK_MIN_AGE_S,
                        "error": f"window frozen — {int(_age_s)}s of "
                                 f"{int(RELOCK_MIN_AGE_S)}s elapsed; a locked "
                                 "contract is immutable for its window"}
        if automatic and lock is None:
            return {"ok": False, "skipped": True,
                    "error": "automatic emergency requires an active lock"}
        if lock is not None:
            try:
                locked_at = float(lock.get("locked_at") or 0.0)
            except (TypeError, ValueError, OverflowError):
                locked_at = 0.0
            if not math.isfinite(locked_at) or locked_at <= 0:
                return {"ok": False, "skipped": True,
                        "error": "emergency gated: lock time unavailable"}
            remaining = EMERGENCY_RELOCK_COOLDOWN_S - (now - locked_at)
            if remaining > 0:
                return {"ok": False, "cooldown": True,
                        "cooldown_s": round(remaining, 1),
                        "error": "emergency gated: lock younger than 15 minutes"}
            if (not math.isfinite(price) or price <= 0 or
                    not math.isfinite(_old_entry) or _old_entry <= 0):
                return {"ok": False, "skipped": True,
                        "error": "emergency needs a valid entry and price"}
            # _atr_known is computed by the freeze guard above. Missing ATR
            # must not let manual re-locks bypass the structural condition.
            if not _atr_known:
                return {"ok": False, "frozen": True, "skipped": True,
                        "error": "emergency gated: ATR unavailable"}
        halt = getattr(self, "_halt_until", None)
        if halt is None:
            halt = self._halt_until = {}
        halt[symbol] = now + 30.0
        if lock is None:
            # drill with no open entry: audit row only, still counts the hour
            lock_id = f"emg-{int(now)}"
            row = self.news.record_emergency(symbol, lock_id, reason,
                                             price, price, ok=True)
            return {"ok": True, "drill": True, "lock_id": lock_id,
                    "original_id": None, "new_entry": price,
                    "halt_s": 30, "row": row}
        old_entry = float(lock.get("price") or price)
        lock["relocks"] = int(lock.get("relocks") or 0) + 1
        lock["relock_at"] = now
        lock["original_price"] = old_entry
        lock["original_id"] = lock.get("lock_id") or (
            f"{symbol}-{tf}-{int(lock.get('locked_at') or 0)}")
        lock["price"] = price
        lock["locked_at"] = now
        lock["lock_id"] = _levels.make_lock_id(lock) or f"emg-{int(now)}"
        lock["reason"] = (f"NEWS_EMERGENCY re-lock: {(reason or 'news event')[:140]}"
                          f" | original entry {old_entry:.6f}")
        lock["emergency"] = True
        # re-anchor the CYCLE's levels at the new entry with the SAME
        # measured bps distances — the old code moved only the lock's
        # entry, leaving TP/SL frozen at the old price (a BUY whose
        # TP sat BELOW its entry, the user's "TP/SL in air")
        _cyc = self.cycles.get((symbol, tf))
        if _cyc is not None and _cyc.risk:
            from .moves_risk import reanchor as _reanchor
            _cyc.risk = _reanchor(_cyc.risk, price)
            _dir = int(_cyc.risk.get("direction") or _cyc.direction or 1)
            for _t in _cyc.targets:
                if _t.status in ("locked", "hit"):
                    _tb = _t.minutes
                    _bps = (float(_cyc.risk["tp_bps"])
                            * math.sqrt(max(1, _tb)
                                        / max(1, int(_cyc.risk["minutes"]))))
                    _t.level = round(price * (1 + _dir * _bps / 1e4), 6)
                    _t.status = "locked"
                    _t.locked_at = now
            _cyc.tp = _cyc.risk["tp"]
            _cyc.sl = _cyc.risk["sl"]
            _cyc.events.append({"ts": now, "type": "relock_reanchor",
                                "new_entry": price})
            self._install_cycle_contract(_cyc, now, reason="emergency_relock")
            self._save_cycle_lock_state(_cyc)
        row = self.news.record_emergency(symbol, lock["lock_id"],
                                         reason or "news emergency",
                                         old_entry, price, ok=True)
        try:
            self._save_locks()
        except Exception:
            pass
        try:
            import asyncio as _a
            if self.emit is not None:
                _a.get_event_loop().create_task(self.emit({
                    "type": "news_emergency", "asset": symbol, "tf": tf,
                    "lock_id": lock["lock_id"],
                    "original_id": lock["original_id"],
                    "reason": reason, "old_entry": round(old_entry, 4),
                    "new_entry": round(price, 4), "at": now, "halt_s": 30}))
        except Exception:
            pass
        return {"ok": True, "drill": False, "lock_id": lock["lock_id"],
                "original_id": lock["original_id"], "old_entry": old_entry,
                "new_entry": price, "halt_s": 30, "row": row}

    def _apply_tape_veto(self, symbol: str, res_pack: dict[str, dict]) -> None:
        """Adverse tape cuts SIZE, graded by how adverse it actually is.

        It used to trip on a flat 4 bps move over three bars whenever
        confidence < 0.62 — on a tape whose per-minute sigma is ~7 bps that
        is ordinary noise, so nearly every window was cut to watch-only and
        the board read as a permanently pessimistic "no trade" system.

        Now the threshold is measured: `need = max(4 bps, 2·sigma_bar·√3)`
        (a two-sigma, three-bar adverse move for THIS asset and timeframe).
        ratio = |move| / need:
          * ratio >= 1.0  → tier cut to watch-only (the tape really is
            contradicting the call), side untouched, reason printed;
          * 0.6 … 1.0     → one notch (full→half→quarter): partial credit,
            size reduced instead of the trade cancelled;
          * below 0.6     → not a contradiction; only the measurement is
            recorded (`tape_z`) so the claim can be audited later.
        """
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
            opposes = ((move > 0 and res["direction"] < 0)
                       or (move < 0 and res["direction"] > 0))
            if not opposes:
                res["tape_z"] = 0.0
                continue
            base_bps = 4e-4
            _ring = self.hist.ring(self.hist.tf_key(symbol, tf, "std_r"),
                                   HIST_STD_R)
            _sig = float(_ring.mean()) if len(_ring) else 0.0
            _need_bps = max(base_bps, 2.0 * _sig * (3.0 ** 0.5)) if _sig else base_bps
            ratio = abs(move) / price / _need_bps
            res["tape_z"] = round(ratio, 3)
            res["tape_need_bps"] = round(_need_bps * 1e4, 2)
            conf = float(res.get("confidence") or 0.0)
            if conf >= 0.62 or ratio < 0.6:
                continue
            # an issued call with no explicit tier is a full-size call
            before = str(res.get("tier") or "full")
            if ratio >= 1.0:
                res["tier"] = "watch-only"     # side still stands (never 0)
                after = "watch-only"
            else:
                after = {"full": "half", "half": "quarter",
                         "quarter": "quarter"}.get(before, before)
            if after == before:
                continue
            res["tier"] = after
            res["tape_veto"] = True
            res["reason"] = (
                f"TAPE VETO {ratio:.2f}× noise ({abs(move) / price * 1e4:+.1f} bps "
                f"over 3 bars vs {_need_bps * 1e4:.1f} bps needed) against the "
                f"call at conf {conf:.2f}: size {before}→{after}, side stands. "
                + str(res.get("reason") or ""))[:420]
            disp = self.display_systems.get((symbol, tf))
            if disp is not None:
                disp["tier"] = after
                disp["reason"] = res["reason"]
                disp["tape_veto"] = True
                disp["tape_z"] = res["tape_z"]
                disp["tape_need_bps"] = res["tape_need_bps"]

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
                for fid in list(WEIGHT_IDS) + ["agent", "brain", "micro"]:
                    if fid == "brain":
                        key_name = "facc_brain"
                    elif fid == "micro":
                        key_name = "facc_micro"
                    else:
                        key_name = (f"facc{fid}" if fid != "agent"
                                    else "facc_agent")
                    key = self.hist.tf_key(symbol, tf, key_name)
                    ring = list(self.hist.ring(key, 50))
                    label = (f"F{fid}"
                             if fid not in ("agent", "brain", "micro")
                             else fid)
                    g = self._grade_ring(ring)
                    decided_n = int(g.get("n") or 0) if g else 0
                    if decided_n < self.HINDSIGHT_MIN_DECIDED:
                        entry = {"label": label, "n": decided_n,
                                 "enough": False,
                                 "note": "not enough windows with decided outcomes to say "
                                         f"(need {self.HINDSIGHT_MIN_DECIDED})"}
                    else:
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
            "last_review": {f"{s}|{t}": v
                            for (s, t), v in self._hindsight_last.items()},
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
        ckey = (symbol, tf)
        last = self._hindsight_last.get(ckey, 0.0)
        snap = self.hindsight()                 # cheap: reads rings only
        enough_n = sum(1 for e in snap["per"].values() if e.get("enough"))
        reviewed_n = self._hindsight_reviewed_n.get(ckey, 0)
        # due by cadence, OR enough newly-decided entries that waiting out
        # the cadence would leave the card stale (cold-start refill)
        if last and now - last < period and enough_n <= reviewed_n + 4:
            return
        self._hindsight_last[ckey] = now
        self._hindsight_reviewed_n[ckey] = enough_n
        for key, entry in snap["per"].items():
            if not entry.get("enough"):
                continue
            sym, t, label = key.rsplit("|", 2)
            if label in ("agent", "micro"):
                continue                     # watch-only: no weight slot
            if label == "brain":
                # brain.md §6: the brain's own hindsight multiplier
                old_b = self._fw_hindsight.get((sym, t), 1.0)
                self._fw_hindsight[(sym, t)] = float(entry["multiplier"])
                # §7 rollback: if the brain got WORSE, revert its last
                # dopamine change (the vote's sign is never hand-edited)
                if (entry.get("z", 0.0) <= -1.0
                        and float(entry["multiplier"]) < old_b
                        and self.brains.fw.state == "ready"):
                    if self.brains.fw.rollback():
                        self.hindsight_changes.append({
                            "at": now, "symbol": sym, "tf": t,
                            "formula": "brain",
                            "from": round(old_b, 3),
                            "to": round(float(entry["multiplier"]), 3),
                            "reason": "RPE rollback (brain worse than "
                                      "before its last change)",
                        })
                continue
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
                         now: float, *, record_id: str | None = None) -> None:
        """Section 10/13 bookkeeping after the lock rules ran."""
        prices = self.feed.prices()
        price = prices.get(symbol, 0.0)
        record = empty_record(now)
        record["symbol"] = symbol      # 4 rows share a ts; without this the
        # graded history cannot attribute a hit or a miss to an asset at all
        record["prices"] = {s: round(p, 6) for s, p in prices.items()}
        # GRADE THE PREVIOUS PASS, not this one. The old code read
        # cci.prev_price AFTER this pass's compute had overwritten
        # it with the current price — the 'move' was always 0, so
        # every formula scored a miss forever (leaderboard 0.0000).
        _grade = getattr(self, "_grade_prev", None)
        if _grade is None:
            _grade = self._grade_prev = {}
        for tf in TF_ORDER:
            res = res_pack.get(tf)
            if not res:
                continue
            cyc = self.cycles.get((symbol, tf))
            evs = [e for e in (cyc.events if cyc else [])
                   if e.get("ts", 0) > now - 260]
            # FIRST-TOUCH grading — see app/grading.py: the old two `any()`
            # scans read an event type that is never emitted, were
            # order-blind, and counted TP1 rungs as targets.
            from .grading import resolve as _resolve_pass
            _verdict = _resolve_pass(evs, tp=(cyc.tp if cyc else None))
            target_hit = bool(_verdict["win"])
            sl_hit = bool(_verdict["loss"])
            sub = {
                "emotion": _sign(res.get("emotion", {}).get("aggregate", 0)),
                "physics": _sign(res.get("physics", {}).get("ufp", 0)),
                "news": _sign(res.get("news_value", 0)),
                # an ABSENT hedge signal is 0 = "no opinion", never a -1
                # "short": scoring the gap as a miss is what drove the
                # leaderboard's hf accuracy to 0.036 while the committee was
                # still counting the abstention as a vote.
                "hf": _sign(res.get("hedge", {}).get("combined", 0)),
                "brain": _sign(res.get("brain", {}).get("polarity", 0)),
                "adv": _sign(float((res.get("advanced") or {})
                                   .get("vote") or 0.0)),
                "core": int(res.get("direction", 0)),
                "final": int(res.get("direction", 0)),
            }
            signals = {int(f): _sign(v)
                       for f, v in (res.get("adj") or {}).items()}
            # SUBSYSTEM COMPONENTS (ids 97..102): they are CCI members just
            # like the formulas, but the old signals dict only carried
            # `adj` (formulas), so brain/emotion/physics/news/hedge/tracer
            # never received a single walk-forward sample — the
            # leaderboard showed "n=0 (needs ≥8)" forever for exactly
            # those rows.
            try:
                from .config import (BRAIN_COMPONENT_ID, EMOTION_COMPONENT_ID,
                                     PHYSICS_COMPONENT_ID, NEWS_COMPONENT_ID,
                                     HEDGE_COMPONENT_ID, TRACER_COMPONENT_ID)
                subs = {
                    BRAIN_COMPONENT_ID:
                        _sign((res.get("brain") or {}).get("polarity") or 0.0),
                    EMOTION_COMPONENT_ID:
                        _sign((res.get("emotion") or {}).get("cerc") or 0.0),
                    PHYSICS_COMPONENT_ID:
                        _sign((res.get("physics") or {}).get("ufp") or 0.0),
                    NEWS_COMPONENT_ID: _sign(res.get("news_value") or 0.0),
                    HEDGE_COMPONENT_ID:
                        _sign((res.get("hedge") or {}).get("combined") or 0.0),
                    TRACER_COMPONENT_ID:
                        _sign(self._tracer_last.get((symbol, tf)) or 0.0),
                }
                signals.update({k: v for k, v in subs.items() if v})
            except Exception:
                pass
            brain_sigs = [float(inst.get("h_signal", 0.0))
                          for inst in res.get("brain", {}).get("instances", [])]
            _prev = _grade.get((symbol, tf))
            if _prev and price > 0 and _prev[1] > 0:
                _ms = _sign(price - _prev[1])
                if _ms:
                    self.learning.record_micro(
                        symbol, tf, _ms, _prev[0], sub, _prev[2],
                        1 if target_hit else None,
                        1 if sl_hit else None,
                    )
                    # grade the ADVANCED modules' previous-pass signs
                    # the same walk-forward way (signature promotion
                    # and module honesty both read these rings)
                    _ag = getattr(self, "_adv_grade", None)
                    if _ag is None:
                        _ag = self._adv_grade = {}
                    _psigns, _pprice = _ag.get((symbol, tf), ({}, 0.0))
                    for _mod, _s in (_psigns or {}).items():
                        if _s:
                            self.hist.ring(self.hist.tf_key(
                                symbol, tf, f"adv_{_mod}"), 600).append(
                                1.0 if _s == _ms else 0.0)
                    if _psigns.get("signature"):
                        self.hist.ring(self.hist.tf_key(
                            symbol, tf, "adv_sig"), 600).append(
                            1.0 if _psigns["signature"] == _ms else 0.0)
            _grade[(symbol, tf)] = (signals, price, brain_sigs)
            _ag2 = getattr(self, "_adv_grade", None)
            if _ag2 is None:
                _ag2 = self._adv_grade = {}
            _ag2[(symbol, tf)] = (
                (res.get("advanced") or {}).get("signs") or {}, price)
            # feed the MSRC return ring from the SAME previous-pass price
            # that graded the pass. (The old name ``prev_price`` did not
            # exist in this scope: every GRC raised NameError after the
            # grade, which silently killed the history row write and the
            # return ring — a root cause of empty module scorecards.)
            _pp = float(_prev[1]) if _prev else 0.0
            if res.get("direction") and _pp > 0 and price > 0:
                self.learning.feed_return(symbol, tf, price / _pp - 1.0)
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
            # realised move from the frozen entry (app/grading.py explains
            # why the old (price/cyc.tp - 1) was the distance REMAINING).
            _entry = 0.0
            try:
                _entry = float((cyc.canonical or {}).get("entry")
                               or (cyc.risk or {}).get("entry") or 0.0)
            except (TypeError, ValueError):
                _entry = 0.0
            _dir = int(cyc.direction or 0) if cyc else 0
            if target_hit or sl_hit:
                if _entry > 0 and price > 0 and _dir:
                    _pnl = round(_dir * (price / _entry - 1.0) * 100, 4)
                elif cyc and (cyc.tp if target_hit else cyc.sl):
                    _hit = float(cyc.tp if target_hit else cyc.sl)
                    _pnl = round(abs(_hit - price) / max(price, 1e-9) * 100, 4)
                else:
                    _pnl = 0.0
            else:
                _pnl = 0.0
            record["outcomes"][tf] = {
                "target_hit": target_hit,
                "sl_hit": sl_hit,
                "pnl_pct": _pnl,
                "resolved": bool(target_hit or sl_hit),
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
            # namespaced by asset: the flat key was overwritten by each
            # asset in the pass, so all four read the LAST one's weights.
            record["weights"][tf] = {str(f): round(w.get(f, 0.0), 5)
                                     for f in WEIGHT_IDS}
            record.setdefault("weights_by_symbol", {}).setdefault(symbol, {})[tf] = {
                str(f): round(w.get(f, 0.0), 5) for f in WEIGHT_IDS}
        try:
            self.historydb.append(record)
        except Exception:
            log.exception("history append failed")

    def _sync_display(self, symbol: str, res_pack: dict[str, dict],
                      now: float | None = None) -> None:
        """Mirror the final per-pass result onto the dashboard snapshot.

        Called after the lock rules so the card and the record cannot
        disagree; the panel is display-only state, never a decision input.
        """
        for tf, res in (res_pack or {}).items():
            disp = self.display_systems.get((symbol, tf))
            if disp is None or not isinstance(res, dict):
                continue
            try:
                if res.get("confidence") is not None:
                    disp["confidence"] = round(float(res["confidence"]), 4)
                if res.get("direction") is not None:
                    disp["direction"] = res["direction"]
                disp["issued"] = bool(res.get("issued"))
                disp["cci_adjusted"] = res.get("cci_adjusted")
                if res.get("double_check"):
                    disp["double_check"] = res["double_check"]
                # the LANE's cycle is the settled value: the SAFE protocol and
                # the last-gate verdict both write there, after _apply_agents
                # already mirrored the pre-lock number onto the card
                cyc = self.cycles.get((symbol, tf))
                if cyc is not None and bool(getattr(cyc, "issued", False)):
                    disp["confidence"] = round(float(cyc.confidence or 0.0), 4)
                    disp["direction"] = int(cyc.direction or 0)
                    disp["issued"] = bool(cyc.issued)
                    _tier_before = getattr(cyc, "tier", disp.get("tier"))
                    disp["tier"] = _tier_before
                    # ── THE TIER AGES WITH THE CONFIDENCE (2026-11-26) ────
                    # This mirror copies the cycle's tier AFTER the SAFE
                    # protocol and the last-gate verdict have moved the
                    # confidence, and it never re-derived it. Live result:
                    # `BNB tf3 · conf 0.019 · tier full · NO CALL` — a card
                    # advertising FULL SIZE on a window 19x below its own
                    # issue floor, while the same rule inside _tier_for had
                    # already decided to cap it. It is the exact drift this
                    # module's own docstrings describe for the BADGE (a
                    # number changed after its label was written), just on
                    # the size hint. Re-derive here, on the same numbers the
                    # card is about to print.
                    try:
                        from .config import asset_issue_floor as _aif_d
                        _dfloor = float(_aif_d(symbol))
                        _dconf = float(cyc.confidence or 0.0)
                        if _dconf <= _dfloor and _tier_before != "watch-only":
                            disp["tier"] = "watch-only"
                            disp["tier_rule"] = (
                                str(disp.get("tier_rule") or "")
                                + f" · tier capped at watch-only at publish: "
                                  f"confidence {_dconf:.3f} <= issue floor "
                                  f"{_dfloor:.3f} (the size hint may never "
                                  f"outrank the published gate)")
                            if cyc is not None:
                                # the cycle is what the next tick reads
                                cyc.tier = "watch-only"
                    except Exception:
                        log.exception("tier re-derivation failed for %s/%s",
                                      symbol, tf)
                    if getattr(cyc, "verdict", ""):
                        disp["verdict"] = cyc.verdict
            except Exception:                     # display must never break a pass
                log.exception("display sync failed for %s/%s", symbol, tf)

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
        # The measured block is frozen at lock time. GRC refreshes may update
        # diagnostics and measured rungs, but never replace the contract pair.
        frozen = cyc.risk or {}
        if _levels.valid({"entry": frozen.get("entry"), "tp": frozen.get("tp"),
                          "sl": frozen.get("sl"), "direction": frozen.get("direction") or cyc.direction}):
            # Keep F28/F29's current-risk read alive as a separate projection;
            # it is diagnostic/sizing input only and cannot mutate the contract.
            try:
                projection = self._risk_pack(symbol, tf, cyc, res)
                if projection and projection[4] is not None:
                    frozen["live_projection"] = {
                        "tp": projection[0], "sl": projection[1],
                        "tp_dist": projection[2], "sl_dist": projection[3],
                        "ratio": projection[4], "as_of": time.time(),
                        "executable": False,
                    }
            except Exception:
                log.exception("live risk projection failed for %s/%s", symbol, tf)
            cyc.tp, cyc.sl = frozen["tp"], frozen["sl"]
            cyc.tp_dist = float(frozen.get("tp_dist") or abs(cyc.tp - frozen["entry"]))
            cyc.sl_dist = float(frozen.get("sl_dist") or abs(frozen["entry"] - cyc.sl))
            cyc.ratio = round(cyc.tp_dist / cyc.sl_dist, 6) if cyc.sl_dist else None
            return
        if cyc.level_contract:
            contract = _levels.validate_cycle_contract(cyc)
            if contract:
                cyc.tp, cyc.sl = contract["tp"], contract["sl"]
                cyc.tp_dist, cyc.sl_dist = contract["tp_dist"], contract["sl_dist"]
                cyc.ratio = contract["ratio"]
            else:
                cyc.execution_eligible = False
            return
        if not cyc.issued:
            return
        tp, sl, tp_dist, sl_dist, ratio = self._risk_pack(symbol, tf, cyc, res)
        if ratio is None and cyc.issued and cyc.active_k is None:
            return
        cyc.tp, cyc.sl = tp, sl
        cyc.tp_dist, cyc.sl_dist = tp_dist, sl_dist
        cyc.ratio = ratio

    def _install_cycle_contract(self, cyc: TFCycle, now: float,
                                reason: str = "") -> dict | None:
        """Freeze, persist and validate this cycle's exact entry/TP/SL pair."""
        lock_row = self.locks.get((cyc.symbol, cyc.tf))
        if not isinstance(lock_row, dict):
            cyc.execution_eligible = False
            return None
        risk = cyc.risk or {}
        entry = _levels._number(risk.get("entry")) or _levels._number(lock_row.get("price"))
        side = int(cyc.direction or 0)
        if entry is None or entry <= 0 or side not in (-1, 1):
            cyc.level_contract = _levels.unavailable_contract(
                symbol=cyc.symbol, timeframe=cyc.tf, cycle=cyc.cycle_id,
                reason="locked entry or direction unavailable")
            cyc.execution_eligible = False
            return None

        # A restart inside an unexpired prediction window must restore the
        # exact serialized contract, not refit it from the newly warmed tape.
        saved = lock_row.get("level_contract")
        saved_lock = lock_row.get("contract_lock")
        saved_cycle = lock_row.get("contract_cycle_id")
        saved_revision = int(lock_row.get("level_revision") or 0)
        saved_end = float(lock_row.get("contract_cycle_end") or 0.0)
        if (reason == "initial_lock" and isinstance(saved_lock, dict)
                and _levels.lock_id_valid(saved_lock)
                and _levels.contract_valid(
                    saved, symbol=cyc.symbol, timeframe=cyc.tf,
                    cycle=saved_cycle, side=side, revision=saved_revision,
                    lock_id=saved_lock.get("lock_id"), entry=entry)
                and saved_end > now):
            cyc.cycle_id = saved_cycle
            cyc.lock_id = saved_lock["lock_id"]
            cyc.level_revision = saved_revision
            cyc.contract_lock = dict(saved_lock)
            cyc.level_contract = dict(saved)
            cyc.cycle_start = float(lock_row.get("contract_cycle_start") or cyc.cycle_start)
            cyc.cycle_end = saved_end
            cyc.contract_tp_hit = bool(lock_row.get("contract_tp_hit"))
            cyc.sl_observed = bool(lock_row.get("sl_observed"))
            cyc.pending_reset = bool(lock_row.get("contract_pending_reset"))
            if "contract_active_k" in lock_row:
                cyc.active_k = lock_row.get("contract_active_k")
            cyc.safe_events = int(lock_row.get("contract_safe_events") or cyc.safe_events)
            cyc.sl_tighten = float(lock_row.get("contract_sl_tighten") or cyc.sl_tighten)
            cyc.issued = bool(lock_row.get("contract_issued", cyc.issued))
            if lock_row.get("contract_tier"):
                # Restore the window's tier, but never let a row written by
                # an older build (different bands / no floor provenance)
                # contradict the freshly computed one: the board then shows
                # "P 0.97 · watch-only" and looks rigged. A restored tier is
                # accepted only when its own band agrees with this window's
                # measured score/floor ratio.
                _rt = str(lock_row["contract_tier"])
                _rx = float(cyc.score_floor_ratio or 0.0)
                _band = ("watch-only" if _rx <= 1.0 else "quarter" if _rx <= 1.5
                         else "half" if _rx <= 2.5 else "full")
                cyc.tier = _rt if (_rx <= 0 or _rt == _band) else _band
            if lock_row.get("contract_confidence") is not None:
                # ── A RESTORED CONFIDENCE MAY NOT INFLATE (2026-11-26) ──
                # The tier three lines up is already protected ("a restored
                # tier is accepted only when its own band agrees with this
                # window's measured score/floor ratio") but the CONFIDENCE
                # was adopted unconditionally. Live: BNB tf1 published
                # 0.853 with its own regenerated chain reading 0.116 —
                # a stored `contract_confidence` from an earlier window
                # overwriting this window's measurement, and every panel
                # reading the inflated one. Same rule as the tier, and the
                # same asymmetry as the card badge: a restore may LOWER a
                # confidence (a saved judgement of a worse setup is
                # information) but may never RAISE it above the measured
                # chain, because that is not a restore, it is a different
                # window's number wearing this one's card.
                _restored = float(lock_row["contract_confidence"])
                _fresh_conf = float(getattr(cyc, "confidence", 0.0) or 0.0)
                if _fresh_conf > 0 and _restored > _fresh_conf + 0.10:
                    # `_install_cycle_contract` does not take a symbol/tf —
                    # naming them here raised NameError on EVERY initial
                    # lock, which the caller caught and logged as "initial
                    # lock failed", so no cycle was ever installed and the
                    # board fell back to preview ladders. Read them off the
                    # cycle instead of inventing a scope.
                    log.info("restored confidence refused %s/%s: stored "
                             "%.3f would inflate this window's measured "
                             "%.3f — keeping the measurement",
                             getattr(cyc, "symbol", "?"),
                             getattr(cyc, "tf", "?"), _restored, _fresh_conf)
                    cyc.restore_note = (
                        f"stored confidence {_restored:.3f} refused; this "
                        f"window measured {_fresh_conf:.3f}")
                else:
                    cyc.confidence = _restored
            saved_targets = lock_row.get("contract_targets")
            if isinstance(saved_targets, list) and len(saved_targets) == len(cyc.targets):
                for target, saved_target in zip(cyc.targets, saved_targets):
                    if not isinstance(saved_target, dict):
                        continue
                    if saved_target.get("status") in {"pending", "locked", "hit", "invalidated"}:
                        target.status = saved_target["status"]
                    level = _levels._number(saved_target.get("level"))
                    target.level = level if level and level > 0 else None
                    target.direction = int(saved_target.get("direction") or 0)
                    target.locked_at = saved_target.get("locked_at")
                    target.hit_ts = saved_target.get("hit_ts")
                    target.locked_cci = saved_target.get("locked_cci")
            if isinstance(lock_row.get("contract_risk"), dict):
                cyc.risk = dict(lock_row["contract_risk"])
            cyc.tp, cyc.sl = saved["tp"], saved["sl"]
            cyc.tp_dist, cyc.sl_dist, cyc.ratio = saved["tp_dist"], saved["sl_dist"], saved["ratio"]
            if cyc.targets:
                cyc.target(len(cyc.targets)).level = cyc.tp
            cyc.execution_eligible = bool(
                cyc.issued and cyc.tier != "watch-only"
                and not saved.get("below_floor") and not cyc.contract_tp_hit
                and not cyc.sl_observed and not cyc.pending_reset
                and getattr(self.feed, "mode", None) == "binance")
            return cyc.level_contract

        # Cap only the TP distance at the user's 4:1 maximum, keeping the
        # measured stop and recording the uncapped evidence in the risk block.
        tp, sl = _levels._number(cyc.tp), _levels._number(cyc.sl)
        lv = {"entry": entry, "tp": tp, "sl": sl, "direction": side}
        if not _levels.valid(lv):
            cyc.level_contract = _levels.unavailable_contract(
                symbol=cyc.symbol, timeframe=cyc.tf, cycle=cyc.cycle_id,
                reason="measured TP/SL failed locked-entry side validation",
                side=side)
            cyc.execution_eligible = False
            return None
        tp_dist, sl_dist = abs(tp - entry), abs(entry - sl)
        rr = tp_dist / sl_dist if sl_dist else 0.0
        if rr > _levels.MAX_RR:
            risk.setdefault("raw_tp_bps", risk.get("tp_bps"))
            risk.setdefault("raw_ratio", rr)
            tp_dist = _levels.MAX_RR * sl_dist
            tp = entry + side * tp_dist
            cyc.tp = round(tp, 8)
            if cyc.targets:
                cyc.target(len(cyc.targets)).level = cyc.tp
            risk["tp"] = cyc.tp
            risk["tp_dist"] = tp_dist
            risk["tp_bps"] = tp_dist / entry * 1e4
            risk["rr"] = risk["ratio"] = _levels.MAX_RR
            rr = _levels.MAX_RR
        # ── THE 1:1 FLOOR (USER RULE), applied to the CONTRACT ──────────
        # This used to demote the cycle to watch-only when the ratio came
        # out below 1:1 — i.e. exactly the abstention the user forbids — and
        # it published the sub-1:1 pair anyway, so the card showed a trade
        # paying less than it risked. Live board 2026-11-27 before this fix:
        # 9 of 16 cells below 1:1 (BNB tf1 0.474, BTC tf1/tf3 0.500, ETH tf1/
        # tf2 0.800, ETH tf3 0.545, PAXG tf2 0.543).
        #
        # The floor WIDENS THE TARGET to meet the measured stop; it never
        # narrows the stop, because the stop is the risk unit and shrinking
        # it to buy a prettier ratio would raise the loss frequency. The
        # window still ships a side at full size — nothing abstains.
        if sl_dist > 0 and rr < 1.0 - 1e-6:
            risk.setdefault("rr_before_floor", round(rr, 6))
            tp_dist = sl_dist
            tp = entry + side * tp_dist
            cyc.tp = round(tp, 8)
            if cyc.targets:
                cyc.target(len(cyc.targets)).level = cyc.tp
            risk["tp"] = cyc.tp
            risk["tp_dist"] = tp_dist
            risk["tp_bps"] = tp_dist / entry * 1e4
            risk["rr"] = risk["ratio"] = 1.0
            risk["rr_floor_applied"] = True
            rr = 1.0
            if not any(e.get("type") == "rr_floor_applied" for e in cyc.events):
                cyc.events.append({"ts": now, "type": "rr_floor_applied",
                                   "from": risk["rr_before_floor"], "to": 1.0})
        cyc.tp_dist, cyc.sl_dist = tp_dist, sl_dist
        cyc.ratio = round(rr, 6) if sl_dist else None

        # Give each issuance/relock a fresh lock fingerprint, distinct from
        # the settlement-window row's mutable metadata.
        contract_lock = {
            "symbol": cyc.symbol, "tf": cyc.tf, "side": side,
            "price": entry, "locked_at": float(now),
            "window_start": float(cyc.cycle_start or now),
            "window_end": float(cyc.cycle_end or now),
            "relocks": int(lock_row.get("relocks") or 0),
        }
        contract_lock["lock_id"] = _levels.make_lock_id(contract_lock)
        revision = max(1, int(lock_row.get("level_revision") or 0) + 1)
        cycle_id = f"{cyc.symbol}:{cyc.tf}:{int(float(cyc.cycle_start or now) * 1000)}"
        frozen = {"entry": entry, "tp": tp, "sl": sl, "direction": side}
        source = "horizon-forecast" if (cyc.forecast or {}).get("mode") == "new" else "measured-moves"
        contract = _levels.make_contract(
            frozen, symbol=cyc.symbol, timeframe=cyc.tf, cycle=cycle_id,
            side=side, revision=revision, lock_id=contract_lock["lock_id"],
            source=source)
        cyc.cycle_id, cyc.lock_id = cycle_id, contract_lock["lock_id"]
        cyc.level_revision, cyc.contract_lock = revision, contract_lock
        cyc.level_contract = contract or _levels.unavailable_contract(
            symbol=cyc.symbol, timeframe=cyc.tf, cycle=cycle_id,
            reason="contract construction failed", lock_id=contract_lock["lock_id"],
            side=side, revision=revision)
        # NEW LOCK = NEW CONTRACT. Hit bookkeeping from an earlier window (or
        # an earlier process, after a restart) used to survive into the new
        # lock: a stale `hit_ts` from 07:53 made `contract_tp_hit` True for
        # the 07:56 window, so a `full` tier reported paper_eligible False
        # and the card read like another "no trade". Only crossings at or
        # after this lock's entry can close THIS contract.
        _prev_lock_id = ((lock_row.get("contract_lock") or {}).get("lock_id"))
        if _prev_lock_id != contract_lock["lock_id"]:
            cyc.contract_tp_hit = False
            cyc.sl_observed = False
            for _t in (cyc.targets or []):
                _hit = getattr(_t, "hit_ts", None)
                if _hit and float(_hit) < float(now):
                    _t.hit_ts = None
                    _t.status = "locked"
                    if getattr(_t, "locked_at", None):
                        pass
        if contract:
            cyc.tp, cyc.sl = contract["tp"], contract["sl"]
            cyc.tp_dist, cyc.sl_dist, cyc.ratio = (
                contract["tp_dist"], contract["sl_dist"], contract["ratio"])
            if cyc.targets:
                cyc.target(len(cyc.targets)).level = cyc.tp
        cyc.execution_eligible = bool(
            contract and cyc.issued and cyc.tier != "watch-only"
            and not contract.get("below_floor") and not cyc.contract_tp_hit
            and not cyc.sl_observed and not cyc.pending_reset
            and getattr(self.feed, "mode", None) == "binance")
        lock_row.update({
            "level_contract": dict(cyc.level_contract),
            "contract_lock": dict(contract_lock),
            "contract_cycle_id": cycle_id,
            "contract_cycle_start": cyc.cycle_start,
            "contract_cycle_end": cyc.cycle_end,
            "contract_risk": dict(cyc.risk or {}),
            "contract_targets": [target.to_dict() for target in cyc.targets],
            "contract_tier": cyc.tier,
            "contract_confidence": cyc.confidence,
            "contract_active_k": cyc.active_k,
            "contract_safe_events": cyc.safe_events,
            "contract_sl_tighten": cyc.sl_tighten,
            "contract_tp_hit": bool(cyc.contract_tp_hit),
            "sl_observed": bool(cyc.sl_observed),
            "contract_pending_reset": bool(cyc.pending_reset),
            "contract_issued": bool(cyc.issued),
            "level_revision": revision,
        })
        self._save_locks()
        return contract

    def _save_cycle_lock_state(self, cyc: TFCycle) -> bool:
        """Persist only a self-consistent locked contract and barrier state."""
        contract = _levels.validate_cycle_contract(cyc)
        if not contract:
            cyc.execution_eligible = False
            return False
        row = self.locks.get((cyc.symbol, cyc.tf))
        if not isinstance(row, dict):
            return False
        row.update({"level_contract": dict(contract),
                    "contract_lock": dict(cyc.contract_lock),
                    "contract_cycle_id": cyc.cycle_id,
                    "contract_cycle_start": cyc.cycle_start,
                    "contract_cycle_end": cyc.cycle_end,
                    "contract_risk": dict(cyc.risk or {}),
                    "contract_targets": [target.to_dict() for target in cyc.targets],
                    "contract_tier": cyc.tier,
                    "contract_confidence": cyc.confidence,
                    "contract_active_k": cyc.active_k,
                    "contract_safe_events": cyc.safe_events,
                    "contract_sl_tighten": cyc.sl_tighten,
                    "level_revision": cyc.level_revision,
                    "sl_observed": bool(cyc.sl_observed),
                    "contract_tp_hit": bool(cyc.contract_tp_hit),
                    "contract_pending_reset": bool(cyc.pending_reset),
                    "contract_issued": bool(cyc.issued)})
        self._save_locks()
        return True

    # ---------------- Signal Lock Protocol + settlement (advice 0.3/0.4)
    def _load_locks(self) -> None:
        try:
            with open(_locks_path(), encoding="utf-8") as fh:
                data = json.load(fh)
            self.locks = {(d["symbol"], d["tf"]): d for d in data.get("locks", [])}
        except (OSError, ValueError, KeyError):
            self.locks = {}

    def _save_locks(self) -> bool:
        path = _locks_path()
        tmp = None
        try:
            directory = os.path.dirname(path) or "."
            os.makedirs(directory, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"locks": list(self.locks.values())}, fh,
                          allow_nan=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
            return True
        except (OSError, TypeError, ValueError):
            if tmp:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            return False

    # -- probability-forecast audit persistence -------------------------
    def _load_prob_audit(self) -> None:
        try:
            with open("data/prob_audit.json", encoding="utf-8") as fh:
                self._prob_audit = {**self._prob_audit, **json.load(fh)}
        except (OSError, ValueError):
            pass

    def _save_prob_audit(self) -> None:
        try:
            import os as _os
            _os.makedirs("data", exist_ok=True)
            with open("data/prob_audit.json", "w", encoding="utf-8") as fh:
                json.dump(self._prob_audit, fh)
        except OSError:
            pass

    def calibration_delta(self) -> float:
        return _calibration_delta(self._prob_audit)

    def prob_audit_summary(self) -> dict:
        a = self._prob_audit
        n = int(a.get("n") or 0)
        if not n:
            return {"n": 0, "note": "no graded probability forecasts yet"}
        mean_pred = a["sum_pred"] / n
        realised = a["tp_first"] / n
        return {"n": n,
                "mean_predicted": round(mean_pred, 4),
                "realised_tp_first": round(realised, 4),
                "gap": round(realised - mean_pred, 4),
                "brier": round(a["brier_sum"] / n, 4),
                "brier_cal": (round(a["brier_cal_sum"] / n, 4) if a.get("brier_cal_sum") is not None else None),
                "calibration_delta": _calibration_delta(a),
                "verdict": ("well calibrated" if abs(realised - mean_pred) < 0.05
                            else "OVERCONFIDENT — predicted more than realised"
                            if mean_pred > realised
                            else "underconfident — realised more than predicted")}

    def _funding_series(self, symbol: str) -> list[float] | None:
        """Recent settled funding rates for ``symbol``, newest last.

        ``app/derivs.py`` has been in the tree (and tested) since the
        derivatives work, but nothing in the runtime ever called it, so
        the formula that consumes carry (F41 FND) always saw ``None`` and
        always voted zero. This is the wiring that makes that block real.

        Returns ``None`` — not an empty list — when the archive is absent,
        which is the honest signal F41 distinguishes from "carry is flat".
        Loaded once per symbol and cached, because the archive is on disk
        and funding only settles three times a day.
        """
        cache = getattr(self, "_funding_cache", None)
        if cache is None:
            cache = self._funding_cache = {}
        if symbol in cache:
            return cache[symbol]
        series: list[float] | None = None
        try:
            from . import derivs as _derivs
            loaded = _derivs.load(symbol)
            fund = (loaded or {}).get("funding") or {}
            vals = fund.get("funding_rate")
            if vals is not None and len(vals):
                arr = [float(v) for v in list(vals)[-8:]
                       if v == v]                      # drop NaN
                series = arr or None
        except Exception:                              # optional module
            series = None
        cache[symbol] = series
        return series

    @staticmethod
    def _grid_window(tf: str, now: float) -> tuple[float, float]:
        """Epoch-aligned window (advice 0.1): start = floor(now/W)*W."""
        ws = TIMEFRAMES[tf]["minutes"] * 60
        start = math.floor(now / ws) * ws
        return float(start), float(start + ws)

    # ══════════════════════════════════════════════════════════════
    # COGNITION — the thinking process, and learning from its own record
    # ══════════════════════════════════════════════════════════════
    THOUGHT_FILE = os.path.join("data", "thoughts.json")
    THOUGHT_KEEP = 400            # per (symbol, tf)

    def _load_thoughts(self) -> None:
        try:
            with open(self.THOUGHT_FILE) as fh:
                d = json.load(fh)
            self._thoughts = {k: list(v)[-self.THOUGHT_KEEP:]
                              for k, v in (d or {}).items()
                              if isinstance(v, list)}
        except (OSError, ValueError):
            self._thoughts = {}

    def _save_thoughts(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.THOUGHT_FILE), exist_ok=True)
            tmp = self.THOUGHT_FILE + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(self._thoughts, fh)
            os.replace(tmp, self.THOUGHT_FILE)
        except OSError:
            pass

    def _graded_rows(self, symbol: str, tf: str) -> list[dict]:
        """Every settled outcome for this window — the evidence for competence.

        The ledger only settles positions it opened; a watch-only call still
        proves (or disproves) the app's confidence, and the sealed prediction
        journal records those. Merging both is what makes metacognition warm
        up at all on a keyless/simulated host.
        """
        rows: list[dict] = []
        try:
            rows += [{"confidence": r.get("confidence"),
                      "result": r.get("result"),
                      "ts": r.get("ts") or 0.0}
                     for r in (self.ledger.rows.get(f"{symbol}|{tf}") or [])
                     if r.get("result") in ("win", "loss")]
        except Exception:
            pass
        try:
            for r in self.predlocks.rows:
                if r.get("symbol") != symbol or r.get("tf") != tf:
                    continue
                if r.get("outcome") not in ("won", "lost"):
                    continue
                rows.append({"confidence": r.get("confidence"),
                             "result": ("win" if r["outcome"] == "won"
                                        else "loss"),
                             "ts": r.get("settled_at") or 0.0})
        except Exception:
            pass
        rows.sort(key=lambda r: float(r.get("ts") or 0.0))
        return rows[-400:]

    def thought_stats(self, ledger_only: bool = True,
                      symbol: str | None = None,
                      tf: str | None = None) -> dict:
        """Metacognition over the app's OWN thought record.

        Only *decided* traces (win/loss) enter the hit-rate. Abstained traces
        are counted separately — an app that refuses to call is not scored as
        if it had called. Traces still waiting for settlement are ``open``.
        With ``symbol``/``tf`` the estimate is scoped to that window; the
        global estimate remains available for context.
        """
        from .cognition import competence
        rows, open_n, abstained = [], 0, 0
        for key, items in (self._thoughts or {}).items():
            sym, _, tfk = key.partition("|")
            if symbol and sym != symbol:
                continue
            if tf and tfk != tf:
                continue
            for rec in items:
                if rec.get("outcome") is None:
                    open_n += 1
                elif rec.get("outcome") == "abstain":
                    abstained += 1
                elif rec.get("outcome") in ("win", "loss"):
                    rows.append({"confidence": rec.get("confidence"),
                                 "result": rec["outcome"],
                                 "ts": rec.get("ts") or 0.0})
        rows.sort(key=lambda r: float(r.get("ts") or 0.0))
        claimed = [float(r["confidence"]) for r in rows
                   if r.get("confidence") is not None]
        mean_claimed = (sum(claimed) / len(claimed)) if claimed else None
        out = competence(mean_claimed, rows)
        out.update({"open": open_n, "abstained": abstained,
                    "decided": len(rows),
                    "scope": f"{symbol or 'all'}|{tf or 'all'}",
                    "rule": ("win/loss only measures the traces that actually "
                             "made a call; abstentions and open traces are "
                             "reported but never folded into the hit-rate")})
        return out

    def cognitive(self, symbol: str, tf: str) -> dict:
        """THE THINKING TRACE for a window (pure read, no state change).

        Every field is derived from the payload the engine already built, so
        this can never disagree with the numbers on the cards.
        """
        from .cognition import think
        res = (self.last_result.get((symbol, tf))
               or self.display_systems.get((symbol, tf)) or {})
        # the verdict lives on the CYCLE card; overlay it on the copy the
        # trace reads so the observe step can narrate the same word the
        # hero shows (never mutate last_result — this is a read path)
        _cyc = self.cycles.get((symbol, tf))
        if _cyc is not None and _cyc.verdict:
            res = {**res, "verdict": _cyc.verdict}
        store = {
            "emotions": {tf: (res.get("emotion") or
                              (self.last_result.get((symbol, tf)) or {})
                              .get("emotion") or {})},
            "physics": {tf: res.get("physics") or {}},
            # a numeric news reading was passed as the dict itself, so a live
            # 0.0 was "withheld" and a live +0.4 became an unreadable float.
            # Wrap it with its own availability flag.
            "news": ({} if res.get("news_value") is None else
                     {"value": res.get("news_value"), "available": True}),
            "hedge": {tf: res.get("hedge") or {}},
            "feed": self.feed.stats(),
        }
        out = think(symbol=symbol, tf=tf, res=res, store=store,
                    graded=self._graded_rows(symbol, tf),
                    floor=float(CONFIDENCE_FLOOR),
                    system_mode=str(self.agents.settings.get("system_mode", "old")))
        cyc = _cyc
        if cyc is not None:
            out["verdict"] = cyc.verdict
            out["prediction_lock"] = dict(cyc.prediction_lock or {})
            out["prediction_open"] = [
                r for r in self.predlocks.open_for(symbol, tf)]
        out["thought_stats"] = self.thought_stats(symbol=symbol, tf=tf)
        out["thought_stats_all"] = self.thought_stats()
        return out

    def _seal_analysis_lock(self, symbol: str, tf: str, final: dict,
                            store: dict, res: dict, cyc,
                            signature: str | None = None,
                            target: str = "measured TP/SL ladder") -> dict:
        """Seal the ANALYSE result as an immutable, graded prediction.

        The evidence saved here is exactly what the decision used: formula
        signs, subsystem votes, brain signals, the AI reply and the weightage.
        Settlement later feeds those numbers back into the same learners that
        grade every other window, so the lock teaches the app rather than
        merely decorating the page.
        """
        raws = (store.get("raws") or {}).get(tf) or {}
        formula_signs = {str(k): _sign(v) for k, v in raws.items()
                         if abs(float(v or 0.0)) > 1e-9}
        brain_sigs = [float(inst.get("h_signal", 0.0))
                      for inst in (res.get("brain") or {}).get("instances", [])]
        tps = final.get("tps") or []
        tp = final.get("contract_tp")
        if tp is None and tps:
            tp = tps[-1].get("level")
        if tp is None:
            tp = final.get("prediction", {}).get("tp")
        sl = final.get("sl") or final.get("contract_sl")
        # NEW system: the prediction being sealed is "price at horizon end",
        # so the lock's target must be that forecast — sealing the measured
        # ladder while labelling the call NEW would grade the wrong number.
        _mode = str(self.agents.settings.get("system_mode", "old"))
        from . import mandate_cfg as _mcfg_s
        _mmode = _mcfg_s.mode(symbol)
        if _mmode in ("old", "new"):
            _mode = _mmode          # per-asset mandate mode wins
        _fc = ((getattr(cyc, "forecast", {}) or {}).get("price")
               if cyc is not None else None)
        if _mode == "new" and _fc:
            tp = float(_fc)
        # CERTIFIED HOLD POLICY — the sealed prediction carries the
        # cell's decoupled ROW geometry (TP = k_row·m_row·q80-scale,
        # SL = m_row·q80-scale) and the hold settlement rules.  The
        # entry stays whatever the lock already froze; the scale is the
        # benchmark's q80 window-move estimator on the live tape, with
        # the engine ATR labelled as the warmup proxy when the tape is
        # too thin to estimate q80 yet.
        hold_block = None
        try:
            from . import hold_policy as _hp_s
            if _hp_s.is_hold_cell(symbol, tf):
                entry_hold = float((getattr(cyc, "level_contract", {})
                                    or {}).get("entry")
                                   or final.get("entry") or 0.0)
                _scale = _hp_s.scale_at_feed(self.feed, symbol, tf)
                _scale_src = "q80 window-move scale (hold.py _scale_at)"
                if not _scale or _scale <= 0:
                    _scale = float(final.get("atr") or 0.0)
                    _scale_src = "engine ATR warmup proxy (q80 not yet estimable)"
                # Every sealed hold lock follows a bounded trade lifecycle.
                # Use guarded-era geometry when certified; the v8 m=999
                # cells use the measured row ladder but still have a
                # maximum hold and TP/SL/side-flip settlement.
                _sum = _hp_s.summary(symbol, tf)
                _era_m = float(_sum.get("era_m") or 0.0)
                _use_era_geometry = bool(_scale and _scale > 0 and _era_m
                                         and 0 < _era_m < 999.0)
                # Every hold-cell lock is bounded, even if warmup data is
                # too thin to derive row geometry (then retain the measured
                # analysis levels but still enforce maximum hold).
                _use_era = True
                _bars_pair = (
                    _hp_s.build_era_bars(
                        symbol, tf, entry_hold,
                        int(final.get("direction") or 0), _scale)
                    if _use_era_geometry else
                    (_hp_s.build_row_bars(
                        symbol, tf, entry_hold,
                        int(final.get("direction") or 0), _scale)
                     if _scale else None))
                # COST FLOOR: a "win" must pay more than the round trip.
                # The certified gross geometry targets k=0.15 x SL, which
                # on the short timeframes lands INSIDE the Binance round
                # trip — an ~85 % hit rate that still loses money per
                # trade. _cost_aware widens such a target to the floor
                # (see app/hold_policy.MIN_TP_OVER_RT); the verdict is
                # recorded on the seal so it is never silent.
                _cost_meta = None
                try:
                    _g = (_hp_s.era_geometry(symbol, tf)
                          if _use_era_geometry else _hp_s.row_geometry(
                              symbol, tf))
                    if _g and _scale and _scale > 0:
                        _cost_meta = _hp_s._cost_aware(
                            symbol, entry_hold,
                            int(final.get("direction") or 0),
                            _scale, float(_g[0]), float(_g[1]))[2]
                except Exception:
                    log.exception("cost-floor check failed for %s|%s",
                                  symbol, tf)
                if _bars_pair:
                    tp, sl = float(_bars_pair[0]), float(_bars_pair[1])
                hold_block = {
                    **_sum,
                    "hold_rules": True,
                    "scale": _scale,
                    "scale_source": _scale_src,
                    "row_tp": tp, "row_sl": sl,
                    # the ratio and the floors this lock was BUILT on, so
                    # no surface has to derive a payoff from the two prices
                    # and guess which geometry it belongs to
                    "row_rr": (_cost_meta or {}).get("rr_live"),
                    "row_rr_certified": (_cost_meta or {}).get("rr_certified"),
                    "rr_floor": (_cost_meta or {}).get("rr_floor"),
                    "cost_floor": _cost_meta,
                    "era_ride": _use_era,
                    "geometry": "guarded_era" if _use_era_geometry else "row",
                    "era_max_hold_s": (_hp_s.ERA_MAX_HOLD_S
                                       if _use_era else None),
                    "settle_rules": _hp_s.row_settle_rules(symbol, tf),
                    "era_geometry": {"k": _sum.get("era_k"),
                                     "m": _sum.get("era_m"),
                                     "ride": _sum.get("ride")},
                    "round_trip_friction_bps": _sum.get("round_trip_bps"),
                    "rule": _mcfg_s.direction_rule(symbol, tf),
                }
        except Exception:
            log.exception("hold-policy seal overlay failed for %s|%s",
                          symbol, tf)
        evidence = {
            "formula_signs": formula_signs,
            "subsystems": {k: _sign(v) if k != "final" else int(v or 0)
                           for k, v in (final.get("voters") or {}).items()},
            "brain_signals": brain_sigs,
            "voters": final.get("voters"),
            "price_action": final.get("price_action"),
            "weightage": final.get("weightage"),
            "ai": {k: v for k, v in (final.get("ai") or {}).items()
                   if k in ("source", "backend", "direction", "confidence",
                            "insight", "available", "note", "model")},
            "levels_source": final.get("levels_source"),
            "recheck": {"passed": (final.get("recheck") or {}).get("passed"),
                        "score": (final.get("recheck") or {}).get("score"),
                        "verdict": (final.get("recheck") or {}).get("verdict")},
            "signature": (signature
                          or (final.get("thinking") or {}).get("signature")),
        }
        if hold_block:
            # Backtest row rules retain a pessimistic SL-first bound;
            # live OHLC ties are unresolvable and are never learned as loss.
            evidence["hold_policy"] = hold_block
        # THE LIVE PRICE AND THE ANCHOR MUST BE THE SAME MARKET. The seal
        # used to accept whatever the level contract remembered, so a
        # restart (or a contract frozen before the last move) stamped a
        # prediction at an anchor tens of bps away from the live quote —
        # with a stop inside that gap, every barrier was already behind
        # the market and the lock graded on the first closed bar. Handing
        # the live quote to the journal lets it refuse that seal by name
        # instead of writing an unearned loss to history.
        try:
            _live_seal = float(self.feed.prices().get(symbol, 0.0) or 0.0)
        except Exception:
            _live_seal = 0.0
        return self.predlocks.seal(
            symbol, tf,
            direction=int(final.get("direction") or 0),
            # the entry is the FROZEN contract anchor when one exists —
            # otherwise the lock would be stamped at a price the level
            # contract never traded at
            entry=float((getattr(cyc, "level_contract", {}) or {}).get("entry")
                        or final.get("entry") or 0.0),
            tp=float(tp or 0.0),
            sl=float(sl or 0.0),
            now_price=(_live_seal or None),
            confidence=float(final.get("confidence") or 0.0),
            minutes=int(TIMEFRAMES[tf]["minutes"]),
            gate_ok=bool(final.get("gate_ok") and
                         (final.get("recheck") or {}).get("passed")),
            recheck_ok=(final.get("recheck") or {}).get("passed"),
            reason=(f"{final.get('verdict')} · conf {final.get('confidence')}"
                    f" · {(final.get('recheck') or {}).get('verdict') or ''}"),
            evidence=evidence,
            target=target,
            source=str((self.analyse_job or {}).get("source") or "analyse"),
            system_mode=str(self.agents.settings.get("system_mode", "old")),
            cycle_id=(getattr(cyc, "cycle_id", None) if cyc else None),
            lock_id=(getattr(cyc, "lock_id", None) if cyc else None),
            # §18 era ride: the lock outlives the window (TP, SL or
            # the certified side flipping settle it first)
            max_hold_s=(float(hold_block.get("era_max_hold_s"))
                        if (hold_block or {}).get("era_ride") else None),
        )

    # ==================================================================
    # prediction SYSTEM selector (old ladder ⇄ new horizon forecast)
    # ==================================================================
    def _retarget_display(self, cyc, mode: str) -> None:
        """Point the DISPLAYED TP at the selected model's target.

        Execution and settlement keep reading the frozen level contract, so
        flipping the selector can never move the barrier a live trade or the
        ledger is graded against. Only what the user sees as "the target"
        changes, immediately — not at the next 22-minute boundary.
        """
        if cyc is None:
            return
        from . import mandate_cfg as _mcfg_r
        _mmode = _mcfg_r.mode(getattr(cyc, "symbol", "") or "")
        if _mmode in ("old", "new"):
            mode = _mmode          # per-asset mandate mode wins
        forecast = dict(cyc.forecast or {})
        risk = dict(cyc.risk or {})
        entry = float((cyc.level_contract or {}).get("entry")
                      or risk.get("entry") or 0.0)
        d = int(cyc.direction or 0)
        if mode == "new":
            fc = forecast.get("price")
            if fc and entry > 0:
                cyc.tp = float(fc)
                cyc.tp_dist = abs(cyc.tp - entry)
                cyc.sl = float(forecast.get("sl") or risk.get("sl") or cyc.sl)
                cyc.sl_dist = (abs(entry - cyc.sl) if cyc.sl else cyc.sl_dist)
                cyc.ratio = (round(cyc.tp_dist / cyc.sl_dist, 6)
                             if cyc.sl_dist else cyc.ratio)
                if cyc.targets:
                    cyc.targets[-1].level = cyc.tp
                forecast["mode"] = "new"
                cyc.forecast = forecast
        else:
            measured_tp = risk.get("tp") or (cyc.level_contract or {}).get("tp")
            if measured_tp:
                cyc.tp = float(measured_tp)
                if entry > 0:
                    cyc.tp_dist = abs(cyc.tp - entry)
                    cyc.ratio = (round(cyc.tp_dist / cyc.sl_dist, 6)
                                 if cyc.sl_dist else cyc.ratio)
                if cyc.targets:
                    cyc.targets[-1].level = cyc.tp
            forecast["mode"] = "old"
            cyc.forecast = forecast
        cyc.events.append({"ts": time.time(), "type": "system_mode",
                           "mode": mode,
                           "note": "display target switched; settlement "
                                   "contract unchanged"})

    def set_system_mode(self, mode: str) -> dict:
        """Switch old/new prediction model everywhere, right now."""
        mode = "new" if str(mode).lower() in ("new", "forecast", "horizon",
                                              "1", "true") else "old"
        self.agents.settings["system_mode"] = mode
        self.agents.save_settings()
        touched = 0
        for cyc in self.cycles.values():
            if cyc is not None:
                self._retarget_display(cyc, mode)
                touched += 1
        self._save_cycles_state()
        return {
            "ok": True, "system_mode": mode, "cycles_retargeted": touched,
            "label": ("NEW — the forecast price at the end of the timeframe "
                      "is the target (stop kept)" if mode == "new" else
                      "OLD — measured TP/SL ladder"),
            "rule": ("the selector changes the model used from this window on; "
                     "a trade/ledger already sealed keeps its contract"),
        }

    def _save_cycles_state(self) -> None:
        """Persist display retargets only if the lock/cycle writer supports it.

        Called from the system switch: a persistence failure must never make
        the switch look broken, so it is deliberately best-effort.
        """
        try:
            self._save_locks()
        except Exception:
            pass

    def _think_and_record(self, symbol: str, tf: str, final: dict,
                          payload: dict | None = None) -> None:
        """Store the trace and its signature against the call it produced."""
        try:
            payload = payload or self.cognitive(symbol, tf)
        except Exception:
            log.exception("cognition failed for %s|%s", symbol, tf)
            return
        key = f"{symbol}|{tf}"
        lock = final.get("prediction_lock") or {}
        # one thought per sealed prediction: re-analysing inside the same
        # locked window must not create an ungradeable duplicate trace
        if lock.get("locked") and lock.get("already"):
            # same full summary the unique path builds (the result box and
            # the tests read every field) — only the LEDGER write is skipped
            final["thinking"] = {
                "signature": payload.get("signature"),
                "n_steps": payload.get("n_steps"),
                "abstain": payload.get("abstain"),
                "trust": (payload.get("competence") or {}).get("trust"),
                "routes_agree": (payload.get("deliberation") or {})
                .get("routes_agree"),
                "consistency": payload.get("consistency"),
                "counterfactual": payload.get("counterfactual"),
                "failing_assumptions": [a["claim"] for a in
                                        payload.get("assumptions") or []
                                        if a.get("holds") is False],
                "duplicate_window": True,
                "note": "a lock already exists for this window; its original "
                        "thought trace is the one that gets graded",
            }
            return
        rec = {"ts": time.time(),
               "symbol": symbol, "tf": tf,
               "lock_id": (lock.get("id") if lock.get("locked") else None),
               "signature": payload.get("signature"),
               "confidence": float(final.get("confidence")
                                   or payload.get("steps", [{}])[0]
                                   .get("data", {}).get("confidence") or 0.0),
               "signal": final.get("signal"),
               "direction": int(final.get("direction") or 0),
               "abstain": bool((payload.get("abstain") or {}).get("abstain")),
               "consistency": bool((payload.get("consistency") or {}).get("ok")),
               "split": bool((payload.get("deliberation") or {}).get("split")),
               "outcome": None}
        self._thoughts.setdefault(key, []).append(rec)
        self._thoughts[key] = self._thoughts[key][-self.THOUGHT_KEEP:]
        self._save_thoughts()
        final["thinking"] = {
            "signature": payload.get("signature"),
            "n_steps": payload.get("n_steps"),
            "abstain": payload.get("abstain"),
            "trust": (payload.get("competence") or {}).get("trust"),
            "routes_agree": (payload.get("deliberation") or {}).get("routes_agree"),
            "consistency": payload.get("consistency"),
            "counterfactual": payload.get("counterfactual"),
            "failing_assumptions": [a["claim"] for a in
                                    payload.get("assumptions") or []
                                    if a.get("holds") is False],
        }

    def _grade_last_thought(self, symbol: str, tf: str, won: bool) -> None:
        """A settled window grades the thought that produced it.

        This is the feedback loop: without it the app could output a
        considered-looking trace for ever and never once learn whether its own
        reasoning was any good.
        """
        key = f"{symbol}|{tf}"
        items = self._thoughts.get(key) or []
        for rec in reversed(items):
            if rec.get("outcome") is None:
                # a trace bound to a sealed lock is graded by THAT lock's
                # settlement (exact id), never by whichever window boundary
                # happens to pass first
                if rec.get("lock_id"):
                    return
                # An ABSTAINED trace has no hit/miss to earn: grading it as a
                # win or a loss taught the competence estimate nothing and
                # inflated the sample with calls the app refused to make.
                rec["outcome"] = ("abstain" if rec.get("abstain")
                                  else ("win" if won else "loss"))
                rec["graded_at"] = time.time()
                self._save_thoughts()
                return

    def _grade_thought_by_lock(self, lock: dict) -> None:
        """Grade the trace that produced this exact lock (by lock id).

        Matching on the lock id is what keeps metacognition honest: the
        thought record can never be graded against a different window's
        outcome just because it happened to be the latest ungraded row.
        """
        lid = lock.get("id")
        if not lid:
            return
        for items in (self._thoughts or {}).values():
            for rec in reversed(items):
                if rec.get("lock_id") == lid and rec.get("outcome") is None:
                    outcome = str(lock.get("outcome") or "")
                    if rec.get("abstain"):
                        rec["outcome"] = "abstain"
                    elif outcome in ("won", "lost"):
                        rec["outcome"] = "win" if outcome == "won" else "loss"
                    else:
                        return          # push/ambiguous: nothing to grade
                    rec["graded_at"] = time.time()
                    rec["settled_bps"] = lock.get("bps")
                    self._save_thoughts()
                    return

    PRED_AUDIT_TTL_S = 5.0

    def _prediction_audit_cached(self) -> dict:
        """The journal's self-audit, rebuilt at most every few seconds.

        Never raises: a failed self-check reports its own failure, because
        a checker that can silently die is the very defect being fixed.
        """
        now = time.time()
        hit = getattr(self, "_pred_audit_cache", None)
        if hit and (now - hit[0]) < self.PRED_AUDIT_TTL_S:
            return hit[1]
        try:
            rep = self.predlocks.audit()
            rep["stamp"] = round(now, 3)
        except Exception as exc:
            rep = {"ok": False, "checked": 0, "issues": [],
                   "error": f"{type(exc).__name__}: {exc}",
                   "rule": ("the self-audit itself failed — that is itself "
                            "an error worth showing"),
                   "stamp": round(now, 3)}
        self._pred_audit_cache = (now, rep)
        return rep

    def learn_from_prediction_lock(self, lock: dict) -> None:
        """Feed a settled sealed lock back into the app's learners.

        The stored evidence is the decision's own reading — not a re-read of
        the latest pass — so the credit (and the blame) lands on what
        actually made the call.
        """
        try:
            symbol, tf = str(lock.get("symbol")), str(lock.get("tf"))
            outcome = str(lock.get("outcome") or "")
            if outcome not in ("won", "lost"):
                self._grade_thought_by_lock(lock)
                return
            move_sign = (1 if outcome == "won" else -1) * int(
                lock.get("direction") or 1)
            ev = lock.get("evidence") or {}
            signals = {}
            for k, v in (ev.get("formula_signs") or {}).items():
                try:
                    fid = int(k)
                except (TypeError, ValueError):
                    continue
                s = _sign(v)
                if s:
                    signals[fid] = s
            subs = {str(k): int(v or 0)
                    for k, v in (ev.get("subsystems") or {}).items()}
            brain = [float(x or 0.0) for x in (ev.get("brain_signals") or [])]
            self.learning.record_micro(
                symbol, tf, move_sign, signals, subs, brain,
                1 if outcome == "won" else None,
                1 if outcome == "lost" else None)
            # calibration ring: stated confidence vs realised sign — the same
            # once-per-window rule as the paired rings above
            conf = float(lock.get("confidence") or 0.0)
            if conf > 0 and calib_should_record(self._calib_done,
                                                (symbol, tf), move_sign):
                self.hist.ring(self.hist.tf_key(symbol, tf, "calib_p"),
                               400).append(round(min(1.0, conf) * 20) / 20.0)
                self.hist.ring(self.hist.tf_key(symbol, tf, "calib_h"),
                               400).append(1.0 if outcome == "won" else 0.0)
            # thought record + agent scorecard
            self._grade_thought_by_lock(lock)
            ai = ev.get("ai") or {}
            if (ai.get("direction") not in (None, 0)
                    and ai.get("source") not in (None, "", "none")):
                self.agents.record_outcome(
                    int(ai.get("direction") or 0) == move_sign)
            self.predlocks.mark_learned(str(lock.get("id")))
        except Exception:
            log.exception("lock learning failed")

    def settle_prediction_locks(self, now: float | None = None) -> list[dict]:
        """Settle due locks and learn from each one that closed."""
        try:
            prices = self.feed.prices()
            extremes = {}
            series_by_sym = {}
            for symbol in SYMBOL_ORDER:
                st = self.feed.state(symbol)
                times = list(getattr(st, "times", []) or [])
                closes = list(getattr(st, "closes", []) or [])
                highs = list(getattr(st, "highs", []) or [])
                lows = list(getattr(st, "lows", []) or [])
                minute_now = (int(now if now is not None else time.time()) // 60) * 60
                # A bar with timestamp == current minute is still forming;
                # never let its partial high/low decide a sealed outcome.
                closed_n = sum(1 for ts in times if float(ts) < minute_now)
                # only the bars printed since the newest open lock matter;
                # passing the session extreme would settle a lock on a touch
                # that happened before it existed.
                opens = [r for r in (self.predlocks.rows or [])
                         if r.get("symbol") == symbol
                         and r.get("status") == "open"]
                if opens:
                    # Explicit empty series means "no closed bars yet";
                    # don't fall back to the forming live quote for TP/SL.
                    series_by_sym[symbol] = []
                if opens and closes:
                    since = min(float(r.get("created_at") or 0.0)
                                for r in opens)
                    n = min(closed_n, len(closes), len(times),
                            len(highs), len(lows))
                    lo_i = max(0, n - int(max(1, ((now or time.time()) - since) / 60)))
                    # The lock can only learn from completed OHLC bars.
                    # No aggregate extremes: they erase sequence and used
                    # to turn two touches inside the same candle into a loss.
                    if n > lo_i:
                        series_by_sym[symbol] = [
                            (float(times[j]), float(highs[j]), float(lows[j]))
                            for j in range(lo_i, n)
                        ]
            # §18: the CURRENT certified side per (symbol, tf) — an
            # open era ride exits at market when it no longer matches
            # (the flip IS the era exit)
            sides_now = {}
            for (sym, tfn), cyc in (self.cycles or {}).items():
                try:
                    sides_now[(sym, tfn)] = int(
                        getattr(cyc, "direction", 0) or 0)
                except Exception:
                    continue
            # bar_s: the tape is 1-minute, so a bar stamped at ts covers
            # [ts, ts+60). Without the width the journal cannot tell that
            # a bar which ENDS after the seal still opened before it, and
            # graded the lock on pre-seal tape (an immediate, unearned
            # loss). Passing it lets a straddling bar be skipped.
            rows = self.predlocks.settle_all(prices, extremes={},
                                              now=now,
                                              series=series_by_sym,
                                              sides_now=sides_now,
                                              bar_s=60.0)
            for row in rows:
                self.learn_from_prediction_lock(row)
            return rows
        except Exception:
            log.exception("prediction-lock settlement failed")
            return []

    # ==================================================================
    # HOLD POLICY — live era accounting + the 16-cell status block
    # ==================================================================
    def _committee_contribution(self, symbol: str, tf: str,
                                final: dict) -> dict:
        """§5 FOUR-AGENT COMMITTEE — a REAL, bounded contribution to the
        prediction: the board's vote share moves confidence ±5 % (never
        the side — the certified hold side is parity-locked) and the
        result carries what the brain did so the panel can show it."""
        try:
            from .agent_committee import COMMITTEE as _CMT
            _last = _CMT.last
            if _last is None:
                final["committee"] = {"effect": "committee has not voted yet",
                                      "direction_held": True}
                return final["committee"]
            if not final.get("direction"):
                final["committee"] = {"effect": "no side — board silent",
                                      "direction_held": True}
                return final["committee"]
            # agent directions are float conviction in [-1, +1] — the
            # old int() truncation turned ±0.6 into 0 and made every
            # board look permanently split
            _dirs = [(1 if float(getattr(v, "direction", 0) or 0) > 0.05
                      else -1 if float(getattr(v, "direction", 0) or 0)
                      < -0.05 else 0)
                     for v in (_last.votes or [])]
            _nz = [d for d in _dirs if d]
            _agree = (sum(1 for d in _nz if d == int(final["direction"]))
                      / len(_nz)) if _nz else 0.5
            # An uncalibrated board vote is not accuracy evidence. Keep it
            # visible as telemetry, but never let it inflate/deflate the
            # displayed probability; only realised outcomes may calibrate it.
            _c0 = float(final.get("confidence") or 0.0)
            _eff = ("committee confirms (telemetry only; confidence unchanged)"
                    if _agree >= 0.75 else
                    "committee challenges (telemetry only; confidence unchanged)"
                    if _agree <= 0.25 else
                    "committee split (telemetry only; confidence unchanged)")
            final["committee"] = {
                **_last.as_dict(),
                "agree_share": round(_agree, 3),
                "effect": _eff,
                "confidence_before": _c0,
                "confidence_after": final["confidence"],
                "direction_held": True,
                "rule": ("telemetry only; confidence changes only from "
                         "realised calibration, and the certified side "
                         "is never flipped by the board"),
            }
        except Exception:
            log.exception("committee contribution failed for %s|%s",
                          symbol, tf)
            final["committee"] = {"effect": "committee unavailable",
                                  "direction_held": True}
        return final["committee"]

    def _hold_tick(self, now: float) -> None:
        """Advance every (asset, tf) cell's era position.

        * window boundary → certified side + entry + q80 scale feed the
          accountant (opens / flips / extends the era run),
        * while inside a position, every CLOSED native bar runs the
          SL-first / TP-second first-touch check (SL wins ties),
        * side runs consume exactly one era record each; each close
          pays the real round-trip friction (24 bps, PAXG 36).
        """
        from bisect import bisect_left
        from . import hold_policy as _hp
        from . import hold_era as _he
        minute_now = (int(now) // 60) * 60
        for symbol in SYMBOL_ORDER:
            try:
                st = self.feed.state(symbol)
                times = list(st.times)
                closes = list(st.closes)
                highs = list(getattr(st, "highs", []) or [])
                lows = list(getattr(st, "lows", []) or [])
            except Exception:
                continue
            if not times or len(closes) != len(times):
                continue
            for tf in TF_ORDER:
                if not _hp.managed(symbol, tf):
                    continue
                key = (symbol, tf)
                cur = self._hold_cursor.get(key)
                if cur is None:
                    cur = {"win": 0.0, "bar_i": -1}
                    self._hold_cursor[key] = cur
                W = TIMEFRAMES[tf]["minutes"] * 60
                # window boundary (epoch grid, hold.py's start anchor)
                s = int(times[-1]) // W * W
                if s > float(cur["win"]):
                    cur["win"] = float(s)
                    side, _meta = _hp.side_at_feed(
                        self.feed, symbol, tf, now=s + 1)
                    i_e = bisect_left(times, s) - 1
                    entry_px = float(closes[i_e]) if i_e >= 0 else 0.0
                    scale = _hp.scale_at_feed(self.feed, symbol, tf, now=s)
                    if side and entry_px > 0:
                        # §20 TSMOM gate — slow-trend sign vs side at
                        # the run start (batch parity: window-entry
                        # returns, warmup → take).  Consulted only by
                        # the accountant when a NEW run would start.
                        gate_ok = True
                        gl = _hp.era_gate(symbol, tf)
                        if gl > 0:
                            j = bisect_left(times, float(s) - gl * 60.0)
                            # last bar at/before s−L (batch uses the
                            # window-entry bar — ± one window, same
                            # formula, causal)
                            if j >= 1:
                                r = (float(closes[i_e])
                                     / float(closes[j - 1]) - 1.0)
                                gate_ok = not (int(side) * r < 0.0)
                        _he.accountant(symbol, tf).on_window(
                            side, entry_px, float(scale or 0.0), float(s),
                            gate_ok=gate_ok)
                # closed-bar first-touch scan while inside the position
                acct = _he.accountant(symbol, tf)
                i0 = int(cur.get("bar_i", -1))
                closed_i = max((j for j, ts in enumerate(times)
                                if float(ts) < minute_now), default=-1)
                new_bars = [(float(times[j]), float(highs[j] or closes[j]),
                             float(lows[j] or closes[j]))
                            for j in range(i0 + 1, closed_i + 1)]
                if new_bars and acct.in_position:
                    acct.on_bars(new_bars)
                # Do not advance the cursor over a forming candle. The feed
                # updates that same timestamp when it finally closes; advancing
                # too early permanently skipped its OHLC range.
                cur["bar_i"] = max(i0, closed_i)

    def hold_cells_block(self, now: float | None = None) -> dict:
        """The live 16-cell picture: certified tier + win rate +
        Wilson confidence, ANALYSE countdowns, era position, last lock."""
        from . import hold_policy as _hp
        from . import hold_era as _he
        now = float(now if now is not None else time.time())
        try:
            prices = self.feed.prices()
        except Exception:
            prices = {}
        locks = _hp.cell_locks()
        out: dict[str, dict] = {}
        for symbol, tf in _hp.known_cells():
            row = _hp.summary(symbol, tf)
            rem = locks.remaining(symbol, tf, now)
            row["analyse"] = {
                "locked": rem > 0,
                "remaining_s": round(rem, 1),
                "locked_until": locks.until(symbol, tf),
                "can_analyse": rem <= 0,
                "tf_minutes": TIMEFRAMES[tf]["minutes"],
            }
            try:
                row["era"] = _he.accountant(symbol, tf).snapshot(
                    prices.get(symbol))
            except Exception:
                row["era"] = {"error": "era accountant unavailable"}
            latest = self.predlocks.latest(symbol, tf, 1)
            row["last_lock"] = latest[0] if latest else None
            res = self.last_result.get((symbol, tf)) or {}
            row["live_direction"] = int(res.get("direction") or 0)
            row["rule"] = None
            try:
                from . import mandate_cfg as _mcfg_c
                row["rule"] = _mcfg_c.direction_rule(symbol, tf)
            except Exception:
                pass
            out[f"{symbol}|{tf}"] = row
        return out

    def explain_ctx(self, view: str, symbol: str, tf: str) -> dict:
        """Everything the tab explainer needs, built from live state.
        No computation of its own — it only READS the engine."""
        res = self.last_result.get((symbol, tf)) or {}
        disp = self.display_systems.get((symbol, tf)) or {}
        cyc = self.cycles.get((symbol, tf))
        price = self.feed.prices().get(symbol, 0.0)
        atr = float(res.get("atr") or 0.0)
        bd = res.get("breakdown") or {}
        ctx: dict = {
            "phase": "live" if self.phase == "live" else self.phase,
            "price": price, "atr": atr,
            "direction": int(res.get("direction") or 0),
            "direction_label": ("BUY" if (res.get("direction") or 0) > 0
                                else "SELL" if (res.get("direction") or 0) < 0
                                else "FLAT"),
            "signal": ("BUY" if (res.get("direction") or 0) > 0
                       else "SELL" if (res.get("direction") or 0) < 0
                       else "FLAT"),
            "confidence": res.get("confidence"),
            "issued": bool(res.get("issued")),
            "reason": res.get("reason") or "",
            "cci_raw": res.get("cci_raw"),
            "cci_adjusted": res.get("cci_adjusted"),
            "votes": bd.get("votes") or {},
            "penalty_capped": bd.get("penalty_capped"),
            "penalty": bd.get("penalty"),
            "top": (bd.get("components") or [])[:6],
            "ratio": getattr(cyc, "ratio", None),
            "sl": getattr(cyc, "sl", None),
        }
        # provisional TP/SL the same way the lock builds them
        tps = []
        direction = ctx["direction"] or (1 if cyc is None else
                                         int(getattr(cyc, "direction", 0) or 0))
        conf = abs(float(ctx["confidence"] or 0.0))
        minutes = TIMEFRAMES[tf]["targets"]
        if price and atr:
            for k, mins in enumerate(minutes, start=1):
                tps.append({
                    "k": k, "minutes": mins,
                    "level": round(self._project_level(
                        price, direction or 1, conf, atr, mins), 6)})
        ctx["tps"] = tps
        if not ctx.get("sl") and price and direction:
            ctx["sl"] = round(price - direction * atr, 6)
        view = (view or "dashboard").lower()
        if view == "formulas":
            ctx["zero_reasons"] = (res.get("formula_zero_reasons")
                                   or disp.get("formula_zero_reasons") or {})
        elif view == "emotion":
            em = disp.get("emotion") or {}
            ctx.update(cerc=em.get("cerc"), aggregate=em.get("aggregate"),
                       zero_reasons=em.get("zero_reasons") or {},
                       flow=(em.get("narrative") or {}).get("headline")
                       if isinstance(em.get("narrative"), dict)
                       else (em.get("narrative") or ""))
        elif view == "physics":
            ph = disp.get("physics") or {}
            ctx.update(ufp=ph.get("ufp"), wfcp=ph.get("wfcp"),
                       drpt=ph.get("drpt"), gspi=ph.get("gspi"),
                       mechanisms=ph.get("mechanisms") or [])
        elif view == "news":
            nw = disp.get("news") or {}
            ctx.update(value=nw.get("value"), polarity=nw.get("polarity"),
                       count=nw.get("count"), coverage=nw.get("coverage"),
                       impacts=nw.get("impacts") or {})
        elif view in ("hedge", "hedgefund"):
            hf = disp.get("hedge") or {}
            ctx.update(hedge=hf)
        return ctx

    def calibration_factor(self) -> float | None:
        """Realised/stated hit-rate ratio from the calibration rings.

        Guards: needs n≥40 settled pairs (same floor as the panel) and a
        mean stated confidence ≥0.15 (below that the ratio is noise);
        clipped to [0.6, 1.4] so a bad sample can never wreck confidence.
        """
        try:
            ps: list[float] = []
            hs: list[float] = []
            for symbol in SYMBOL_ORDER:
                for tf in TF_ORDER:
                    pp = list(self.hist.ring(
                        self.hist.tf_key(symbol, tf, "calib_p"), 400))
                    hh = list(self.hist.ring(
                        self.hist.tf_key(symbol, tf, "calib_h"), 400))
                    n = min(len(pp), len(hh))
                    ps.extend(float(x) for x in pp[:n])
                    hs.extend(float(x) for x in hh[:n])
            if len(ps) < 40:
                return None
            stated = sum(ps) / len(ps)
            if stated < 0.15:
                return None
            realised = sum(hs) / len(hs)
            return round(max(0.6, min(1.4, realised / max(stated, 1e-9))), 3)
        except Exception:
            return None

    BRANCH_TTL_S = 8.0

    def branch_stats_cached(self, symbol: str, tf: str, d: dict,
                            price: float, now: float) -> dict | None:
        """branch_stats() memoised for a few seconds.

        The bar-by-bar first-touch replay is exact but expensive (tens of
        millions of comparisons per snapshot across 16 windows). Its inputs
        only change when a cycle rolls, a barrier moves or a new 1-minute
        bar closes, so cache on that signature with a short TTL and a price
        drift guard; the numbers themselves are untouched.
        """
        try:
            st = self.feed.state(symbol)
            sig = (d.get("cycle_start"), d.get("cycle_id"),
                   int(d.get("direction") or 0), d.get("tp"), d.get("sl"),
                   len(st.times))
        except Exception:
            return self.branch_stats(d, price, now, tf)
        key = (symbol, tf)
        # per-window jitter: without it all 16 windows recompute in the same
        # snapshot (a 1.9 s spike); spread over the TTL the board stays < 0.4 s
        ttl = self.BRANCH_TTL_S + (abs(hash(key)) % 7) * 0.9
        hit = self._branch_cache.get(key)
        if hit is not None:
            at, sig0, price0, out = hit
            if (sig0 == sig and now - at < ttl
                    and price0 > 0 and abs(price / price0 - 1.0) < 0.0025):
                return out
        out = self.branch_stats(d, price, now, tf)
        self._branch_cache[key] = (now, sig, price, out)
        if len(self._branch_cache) > 64:
            self._branch_cache.clear()
        if out is not None:
            self._branch_cache[key] = (now, sig, price, out)
        return out

    def branch_stats(self, d: dict, price: float, now: float,
                     tf: str) -> dict | None:
        """PROBABILITY BRANCHES (old-app inspiration): where price can go
        inside this window — quantile bands, P(target before stop),
        P(close down). Driftless-Brownian approximations on the remaining
        window (labelled `approx`); never presented as measurements."""
        try:
            dirn = int(d.get("direction") or 0)
            sl = float(d.get("sl") or 0.0)
            # the barrier pair is the cycle's gated TP/SL (locked-entry
            # anchored) — targets[0] is the 5-minute sub-frame, whose
            # tiny distance made the panel read 0%/0%/100%
            targets = d.get("targets") or []
            tp1 = float(d.get("tp") or 0.0) or (
                float((targets[0] or {}).get("level") or 0.0)
                if targets else 0.0)
            # A barrier that has ALREADY been touched is a resolved window,
            # not a reason to blank the panel. The old early-return threw
            # away the measured distribution too, so the whole
            # Probability-Branches chart (bands, median, NOW marker,
            # target / stop zones) rendered empty for the rest of the
            # window. Keep the empirical/fan computation, then state the
            # outcome plainly: a first touch of the target IS P(target
            # before stop) = 100% for this window.
            resolved: dict | None = None
            if dirn in (-1, 1) and price > 0 and tp1 > 0 and sl > 0:
                tp_touched = (price >= tp1 if dirn > 0 else price <= tp1)
                sl_touched = (price <= sl if dirn > 0 else price >= sl)
                hit_name = ("TP" if tp_touched else "SL" if sl_touched else None)
                if (d.get("contract_tp_hit") or d.get("sl_observed")
                        or tp_touched or sl_touched):
                    tpf = bool(d.get("contract_tp_hit") or tp_touched)
                    slf = bool(d.get("sl_observed") or sl_touched) and not tpf
                    resolved = {
                        "tp_first": tpf,
                        "sl_first": slf,
                        "barrier": ("target already touched" if tpf else
                                    "stop already touched" if slf else
                                    f"price at/past {hit_name}"),
                    }
            if not dirn or price <= 0 or sl <= 0 or tp1 <= 0:
                return None
            rem_s = float(d.get("countdown_s") or 0)
            if rem_s <= 0:
                rem_s = max(1.0, float(d.get("cycle_end") or 0) - now)
            candle_min = float(
                (TIMEFRAMES.get(tf) or {}).get("minutes", 1) or 1)
            # ── EMPIRICAL probabilities (the real fix): replay the
            # actual 1-minute bar path over every historical start on
            # this asset's tape, translate today's TP/SL distances (in
            # bps) onto each start, and COUNT first touches. No
            # driftless formula, no invented distribution — measured
            # frequencies with the sample size printed next to them.
            # A bar that hits both barriers counts as STOP first
            # (conservative; also counted in `ambig`).
            _sym = d.get("symbol") or "BTC"
            # self may be None in the pure-function contract tests; the
            # empirical replay then simply has no tape and the driftless
            # fallback below answers instead of raising.
            _feed = getattr(self, "feed", None)
            _st = _feed.state(_sym) if _feed is not None else None
            _cl = list(_st.closes or []) if _st is not None else []
            _hi = list(_st.highs or _cl) if _st is not None else []
            _lo = list(_st.lows or _cl) if _st is not None else []
            _rem_min = max(1, int(round(rem_s / 60.0)))
            _nb = min(len(_cl), len(_hi), len(_lo))
            if _nb >= _rem_min + 20 and price > 0:
                _tpb = abs(tp1 - price) / price
                _slb = abs(price - sl) / price
                _n = _tpn = _sln = _amb = _up = 0
                _rets = {0.25: [], 0.5: [], 0.75: [], 1.0: []}
                # PRECISION PASS: unconditional frequencies mix every
                # regime together. Also count the subset of historical
                # starts whose SETUP matches now — (a) 5-bar momentum
                # agreeing with the direction, (b) realised-vol regime
                # (10-bar vol above/below the tape median). Conditional
                # frequencies on matched setups are sharper estimates;
                # they are reported alongside, never instead of, the
                # unconditional ones, each with its own n.
                _nc = _tpc = _slc = 0                    # momentum-matched
                _rets_c = {0.25: [], 0.5: [], 0.75: [], 1.0: []}
                # ONE pass: 10-bar rolling volatility per index (the old
                # `_vol_at` rebuilt the same window for every start —
                # O(n^2) and a third of the snapshot build)
                _vol_idx: list = [None] * _nb
                # direction-matched returns: windows whose own 5-bar
                # momentum agreed with TODAY'S side. The fan is drawn from
                # this subset when it is thick enough, so a bullish window
                # carries the drift of bullish windows instead of the
                # unconditional median (which sloped into the SL).
                _rets_d = {0.25: [], 0.5: [], 0.75: [], 1.0: []}
                for _vi in range(10, _nb):
                    _vr = [(_cl[_vj] - _cl[_vj - 1]) / _cl[_vj - 1]
                           for _vj in range(_vi - 9, _vi + 1)
                           if _cl[_vj - 1] > 0]
                    if len(_vr) >= 8:
                        _vm = sum(_vr) / len(_vr)
                        _vol_idx[_vi] = (sum((x - _vm) ** 2 for x in _vr)
                                         / len(_vr)) ** 0.5
                _vols = [v for v in _vol_idx if v is not None]
                _vol_med = sorted(_vols)[len(_vols) // 2] if _vols else 0.0

                def _vol_at(i0):
                    return _vol_idx[i0] if 0 <= i0 < _nb else None
                _cur_vol = _vol_at(_nb - 1)
                _cur_mom = (1 if _nb >= 6 and _cl[-1] > _cl[-6]
                            else -1 if _nb >= 6 else 0)
                _maxi = _nb - _rem_min
                # sample the historical starts: 1-in-1 for short windows,
                # 1-in-4 for the 4-hour one (still thousands of starts, and
                # `n` is printed next to every probability, so the reader
                # always knows the sample size)
                _step = (1 if _maxi <= 400 else
                         2 if _maxi <= 2000 else 4)
                for _i0 in range(0, _maxi, _step):
                    _p0 = _cl[_i0]
                    if _p0 <= 0:
                        continue
                    _tpl = _p0 * (1 + dirn * _tpb)
                    _sll = _p0 * (1 - dirn * _slb)
                    _oc = 0                              # 1 tp / -1 sl first
                    for _j in range(_i0 + 1, _i0 + _rem_min + 1):
                        _ht = (_hi[_j] >= _tpl) if dirn > 0 else (_lo[_j] <= _tpl)
                        _hs = (_lo[_j] <= _sll) if dirn > 0 else (_hi[_j] >= _sll)
                        if _ht and _hs:
                            _amb += 1
                            _oc = -1
                            break
                        if _ht:
                            _oc = 1
                            break
                        if _hs:
                            _oc = -1
                            break
                    if _oc > 0:
                        _tpn += 1
                    elif _oc < 0:
                        _sln += 1
                    _n += 1
                    # matched-setup subset: momentum agrees AND vol regime
                    # matches the current one (both measured, both cheap)
                    _mom_i = (1 if _i0 >= 5 and _cl[_i0] > _cl[_i0 - 5]
                              else -1 if _i0 >= 5 else 0)
                    _v_i = _vol_at(_i0)
                    _match = (_mom_i == dirn and _cur_mom == dirn
                              and (_v_i is None or _cur_vol is None
                                   or (_v_i > _vol_med) == (_cur_vol > _vol_med)))
                    if _match:
                        _nc += 1
                        if _oc > 0:
                            _tpc += 1
                        elif _oc < 0:
                            _slc += 1
                        for _f, _lst in _rets_c.items():
                            _k = _i0 + max(1, int(round(_rem_min * _f)))
                            _lst.append(_cl[_k] / _p0 - 1.0)
                    if _mom_i == dirn:
                        for _f, _lst in _rets_d.items():
                            _k = _i0 + max(1, int(round(_rem_min * _f)))
                            _lst.append(_cl[_k] / _p0 - 1.0)
                    if _cl[_i0 + _rem_min] > _p0:
                        _up += 1
                    for _f, _lst in _rets.items():
                        _k = _i0 + max(1, int(round(_rem_min * _f)))
                        _lst.append(_cl[_k] / _p0 - 1.0)
                if _n >= 20:
                    def _q(vals, qv):
                        s = sorted(vals)
                        return s[min(len(s) - 1, max(0, int(qv * len(s))))]
                    _bands = []
                    for _qv in (0.05, 0.25, 0.5, 0.75, 0.95):
                        _bands.append({
                            "f": _qv,
                            "level": round(price * (1 + _q(_rets[1.0], _qv)), 6)})
                    _end = _rets[1.0]
                    _mu = sum(_end) / len(_end)
                    _sig = (sum((x - _mu) ** 2 for x in _end)
                            / len(_end)) ** 0.5
                    # ── DRIFT-ALIGNED FAN (audit §4.2) ────────────────
                    # Draw the cone from the most specific honest subset:
                    # setup-matched (momentum+vol) > direction-matched
                    # (momentum) > all windows. The old code always drew
                    # the unconditional quantiles, so a BULLISH window's
                    # q50 could slope straight into the SL. The subset and
                    # its n are printed, and if the median still points
                    # against the call the payload says so ("against")
                    # instead of implying support that is not in the tape.
                    if _nc >= 50 and len(_rets_c[1.0]) >= 50:
                        _fan_src = _rets_c
                        _fan_basis = f"setup-matched (momentum+vol) n={_nc}"
                    elif len(_rets_d[1.0]) >= 30:
                        _fan_src = _rets_d
                        _fan_basis = (f"direction-matched (momentum) "
                                      f"n={len(_rets_d[1.0])}")
                    else:
                        _fan_src = _rets
                        _fan_basis = f"all windows n={_n}"
                    _fan = [{"f": 0.0, "q05": round(price, 6),
                             "q25": round(price, 6), "q50": round(price, 6),
                             "q75": round(price, 6), "q95": round(price, 6)}]
                    for _f in (0.25, 0.5, 0.75, 1.0):
                        _vals = _fan_src.get(_f) or _rets[_f]
                        _fan.append({
                            "f": _f,
                            "q05": round(price * (1 + _q(_vals, 0.05)), 6),
                            "q25": round(price * (1 + _q(_vals, 0.25)), 6),
                            "q50": round(price * (1 + _q(_vals, 0.50)), 6),
                            "q75": round(price * (1 + _q(_vals, 0.75)), 6),
                            "q95": round(price * (1 + _q(_vals, 0.95)), 6)})
                    _drift_bps = (round((_fan[-1]["q50"] / price - 1.0) * 1e4, 1)
                                  if price else 0.0)
                    _fan_drift = ("flat" if abs(_drift_bps) < 1.0
                                  else "with" if (_drift_bps > 0) == (dirn > 0)
                                  else "against")

                    def _wilson(k, n):
                        if not n:
                            return [0.0, 0.0]
                        z, ph = 1.96, k / n
                        den = 1 + z * z / n
                        cen = (ph + z * z / (2 * n)) / den
                        half = z * ((ph * (1 - ph) / n
                                     + z * z / (4 * n * n)) ** 0.5) / den
                        return [round(max(0.0, cen - half), 4),
                                round(min(1.0, cen + half), 4)]
                    # (the fan itself is built once, above, from the
                    # most specific honest subset — no second rebuild)
                    out = {
                        "approx": True,
                        "method": "empirical-first-touch",
                        "fan_basis": _fan_basis,
                        # does the median path point the way the call
                        # does? "with" / "against" / "flat" — measured,
                        # never assumed (audit §4.2)
                        "fan_drift": _fan_drift,
                        "drift_bps": _drift_bps,
                        "n": _n, "ambig": _amb,
                        # Wilson 95% intervals: a probability without
                        # its uncertainty is a claim; with n it is a
                        # measurement. EV = counted expectancy of the
                        # barrier pair in bps from the current price.
                        "ci_target": _wilson(_tpn, _n),
                        "ci_stop": _wilson(_sln, _n),
                        "ev_bps": round((_tpn / _n) * _tpb * 1e4
                                        - (_sln / _n) * _slb * 1e4, 1),
                        "p_target_cond": (round(_tpc / _nc, 4)
                                          if _nc >= 30 else None),
                        "n_cond": _nc,
                        "ev_bps_cond": (round((_tpc / _nc) * _tpb * 1e4
                                              - (_slc / _nc) * _slb * 1e4, 1)
                                        if _nc >= 30 else None),
                        "p_target": round(_tpn / _n, 4),
                        "p_stop": round(_sln / _n, 4),
                        "p_between": round(1 - _tpn / _n - _sln / _n, 4),
                        "p_close_down": round(1 - _up / _n, 4),
                        "sigma": round(price * _sig, 4),
                        "rem_s": int(rem_s),
                        "now": round(now, 3),
                        "entry": round(price, 6),
                        "tp": round(tp1, 6),
                        "sl": round(sl, 6),
                        "direction": dirn,
                        "bands": _bands,
                        "fan": _fan,
                    }
                    return PredictionEngine._apply_resolved(out, resolved)
            _lr_map = getattr(self, "last_result", None) or {}
            lr = _lr_map.get((d.get("symbol") or "BTC", tf)) or {}
            atr = float(lr.get("atr") or 0.0)
            if atr <= 0:
                atr = abs(price) * 1e-3
            # sigma over the remaining time (per-candle ATR → sqrt time)
            t_min = max(rem_s / 60.0, 1e-3)
            sigma = atr * math.sqrt(max(t_min, 1e-3) / max(candle_min, 1.0))
            # the fan must be VISIBLE: ATR-of-1m-candles made σ so small
            # the cone glued itself to the price line (user: "only white
            # green and red line nothing else"). Floor σ by the measured
            # block σ (per-horizon, sqrt-scaled to the remaining time).
            _sig_bps = float((d.get("risk") or {}).get("sigma_bps") or 0.0)
            _hz_min = float((d.get("risk") or {}).get("minutes")
                            or candle_min or 1)
            if _sig_bps > 0 and price > 0:
                sigma = max(sigma, price * _sig_bps / 1e4
                            * math.sqrt(t_min / max(_hz_min, 1.0)))
            up = (tp1 - price) * dirn          # room to target (dir side)
            down = (price - sl) * dirn          # room to stop (against)
            if up <= 0 or down <= 0:
                up, down = max(up, 1e-9), max(down, 1e-9)
            # driftless: per-barrier reach-by-T (erfc marginal) decides
            # how much probability mass can hit ANYTHING; the distance
            # share decides WHICH barrier comes first. This keeps the
            # monotonic intuition: nearer target → higher P(target).
            sq2 = math.sqrt(2.0)
            hit_up = math.erfc(up / (max(sigma, 1e-12) * sq2))
            hit_dn = math.erfc(down / (max(sigma, 1e-12) * sq2))
            hit_mass = min(1.0, hit_up + hit_dn - hit_up * hit_dn)
            p_first = down / (up + down)        # P(target first), no drift
            p_target = p_first * hit_mass
            p_stop = (1.0 - p_first) * hit_mass
            p_between = 1.0 - hit_mass
            # quantile bands over the remaining window (approx)
            z = {0.05: -1.6449, 0.25: -0.6745, 0.5: 0.0,
                 0.75: 0.6745, 0.95: 1.6449}
            bands = []
            for f, qv in z.items():
                sig_f = atr * math.sqrt(max(t_min * f, 1e-4)
                                        / max(candle_min, 1.0))
                if _sig_bps > 0 and price > 0:
                    sig_f = max(sig_f, price * _sig_bps / 1e4
                                * math.sqrt(max(t_min * f, 1e-4)
                                            / max(_hz_min, 1.0)))
                bands.append({"f": f, "level": round(price + qv * sig_f, 6)})
            out = {
                "approx": True,
                "method": "driftless-fallback (thin tape)",
                "n": 0, "ambig": 0,
                "ci_target": None, "ci_stop": None,
                "ev_bps": round(p_target * abs(tp1 - price) / price * 1e4
                                - p_stop * abs(price - sl) / price * 1e4, 1)
                if price > 0 else None,
                "p_target": round(p_target, 4),
                "p_stop": round(p_stop, 4),
                "p_between": round(p_between, 4),
                "p_close_down": 0.5,           # driftless: honest coin flip
                "sigma": round(sigma, 4),
                "rem_s": int(rem_s),
                "now": round(now, 3),
                "entry": round(price, 6),
                "tp": round(tp1, 6),
                "sl": round(sl, 6),
                "direction": dirn,
                "bands": bands,
                "fan": [{"f": f,
                         "q05": round(price - 1.6449 * sigma * (f ** 0.5), 6),
                         "q25": round(price - 0.6745 * sigma * (f ** 0.5), 6),
                         "q50": round(price, 6),
                         "q75": round(price + 0.6745 * sigma * (f ** 0.5), 6),
                         "q95": round(price + 1.6449 * sigma * (f ** 0.5), 6)}
                        for f in (0.0, 0.25, 0.5, 0.75, 1.0)],
            }
            return PredictionEngine._apply_resolved(out, resolved)
        except Exception:
            return None

    @staticmethod
    def _apply_resolved(out: dict, resolved: dict | None) -> dict:
        """Fold an already-observed barrier into the branch payload.

        The probabilities become the realised first-touch (1/0), the
        distribution stays visible for the remaining time, and the
        payload says in words that this window is decided.
        """
        if not resolved:
            return out
        out = dict(out)
        out["resolved"] = resolved
        out["method"] = "measured first-touch — barrier already observed"
        if resolved.get("tp_first"):
            out["p_target"] = 1.0
            out["p_stop"] = 0.0
            out["p_between"] = 0.0
        elif resolved.get("sl_first"):
            out["p_target"] = 0.0
            out["p_stop"] = 1.0
            out["p_between"] = 0.0
        out["note"] = (
            f"{resolved.get('barrier')} — this window is decided; the bands "
            "show where price can still travel before it closes")
        return out

    def brain_fusion_weight(self, symbol: str, tf: str) -> float:
        """brain.md §6: 0.25 start, hindsight moves it, 0 when the
        connectome is absent (absence contributes exactly nothing)."""
        if self.brains.fw.state != "ready":
            return 0.0
        mult = self._fw_hindsight.get((symbol, tf), 1.0)
        mult = max(0.25, min(1.5, float(mult)))
        return round(0.25 * mult, 6)

    def _rl_window_state(self, symbol: str, tf: str) -> tuple[str, str]:
        """Measured (Q-table state, bandit context) for this window from
        the tape itself: recent-vs-baseline vol and drift in sigma
        units. Thin tape -> the neutral state/context, never a guess."""
        from .rlsys import window_state, context_key
        try:
            closes = list(self.feed.state(symbol).closes or [])
        except Exception:
            closes = []
        if len(closes) < 30:
            state = "midvol-chop"
            return state, context_key(symbol, tf, state.split("-")[0])
        rets = [(closes[i] - closes[i - 1]) / closes[i - 1]
                for i in range(1, len(closes)) if closes[i - 1]]
        recent = rets[-20:]
        sig_r = _std_pop(recent) if recent else 0.0
        sig_b = _std_pop(rets) if rets else 0.0
        ratio = (sig_r / sig_b) if sig_b > 0 else 1.0
        drift = abs(sum(recent)) / (sig_r * (len(recent) ** 0.5)) \
            if sig_r > 0 else 0.0
        state = window_state(ratio, drift)
        return state, context_key(symbol, tf, state.split("-")[0])

    def _measured_winrate(self, symbol: str, tf: str,
                          conf: float) -> tuple[float | None, int]:
        """Measured strict-sign hit-rate of settled windows whose
        locked confidence sat in this call's bucket (±0.1). Returns
        (rate, n); (None, n) until n >= 10 — never invented."""
        try:
            ps = list(self.hist.ring(
                self.hist.tf_key(symbol, tf, "calib_p"), 400))
            hs = list(self.hist.ring(
                self.hist.tf_key(symbol, tf, "calib_h"), 400))
            pairs = [(p, h) for p, h in zip(ps, hs)
                     if abs(p - conf) <= 0.1]
            if len(pairs) < 10:
                return None, len(pairs)
            return round(sum(h for _, h in pairs) / len(pairs), 4), len(pairs)
        except Exception:
            return None, 0

    def _settle_and_lock(self, symbol: str, tf: str, now: float,
                         res: dict) -> None:
        """Settle the previous window in the ledger, then freeze the new
        signal. Mid-window refashes (same grid window) keep the lock.

        docs11 §4.9 halt = 'no NEW locks for30 s after an emergency
        re-lock' — it must NEVER block settling: a boundary landing
        inside the halt used to return before the ledger was written and
        that window was lost forever (the all-zero settlement history)."""
        prev = self.locks.get((symbol, tf))
        # TREND GATE bookkeeping: one sampled decision close per settled
        # window, recorded BEFORE the next window is priced, so the gate's
        # "trend" is the engine's own decision series (what the interaction
        # was measured on) and never a reading it has not yet seen.
        try:
            from . import trend_gate as _tg
            if prev is not None and float(prev.get("price") or 0.0) > 0:
                _tg.observe(symbol, tf, float(prev["price"]))
        except Exception:
            log.exception("trend-gate observe failed for %s|%s", symbol, tf)
        # Prediction, contract, probability audit and settlement all share
        # the same lock-anchored [start, end) interval (W from issuance).
        start = float(now)
        end = start + TIMEFRAMES[tf]["minutes"] * 60.0
        if prev is not None:
            _pending = prev.get("pending_settlement")
            if not isinstance(_pending, dict):
                try:
                    _prev_end = float(prev.get("window_end") or 0.0)
                except (TypeError, ValueError, OverflowError):
                    _prev_end = 0.0
                if _prev_end <= 0.0:
                    try:
                        _prev_start = float(prev.get("window_start") or 0.0)
                    except (TypeError, ValueError, OverflowError):
                        _prev_start = 0.0
                    if _prev_start > 0.0:
                        _prev_end = _prev_start + TIMEFRAMES[tf]["minutes"] * 60.0
                    else:
                        _locked = float(prev.get("locked_at") or now)
                        _prev_end = self._grid_window(tf, _locked)[1]
                if now < _prev_end:
                    return                  # lock holds for its full W horizon
            if isinstance(_pending, dict):
                # Freeze the first boundary's result across disk errors/restarts;
                # retries must not settle the same window at a later quote.
                try:
                    _side = int(_pending.get("side") or 0)
                except (TypeError, ValueError, OverflowError):
                    _side = 0
                _entry = float(_pending.get("entry") or 0.0)
                price = float(_pending.get("settle_price") or 0.0)
                _window_end = _pending.get("window_end")
                _touch = _pending.get("first_touch", "unavailable")
                _barriers = bool(_pending.get("barriers_available"))
                _ambiguous = bool(_pending.get("ambiguous"))
                _record_id = str(_pending.get("record_id") or "")
            else:
                price = self.feed.prices().get(symbol, 0.0)
                # First-touch grading is distinct from directional window
                # grading. Only a validated frozen level contract may establish
                # barriers; absent/invalid barriers are never called "neither".
                _touch, _barriers, _ambiguous = "unavailable", False, False
                _pc = self.cycles.get((symbol, tf))
                try:
                    _side = int(prev.get("side") or 0)
                except (TypeError, ValueError, OverflowError):
                    _side = 0
                if _pc is not None and _side in (-1, 1):
                    _contract = _pc.level_contract or {}
                    _barriers = _levels.contract_valid(
                        _contract, symbol=symbol, timeframe=tf,
                        cycle=_pc.cycle_id, side=_side,
                        revision=_pc.level_revision, lock_id=_pc.lock_id)
                    if _barriers:
                        _locked_at = float(prev.get("locked_at") or
                                           prev.get("computed_at") or 0.0)
                        _events = sorted(
                            (event for event in _pc.events
                             if event.get("type") in
                             ("target_hit", "tp_hit", "sl_crossed")
                             and float(event.get("ts") or 0.0) >= _locked_at),
                            key=lambda event: float(event.get("ts") or 0.0))
                        _first = _events[0] if _events else None
                        _touch = ("target" if _first and _first.get("type") in
                                  ("target_hit", "tp_hit") else
                                  "stop" if _first else "neither")
                        if _first:
                            _first_ts = float(_first.get("ts") or 0.0)
                            _same_time_types = {
                                event.get("type") for event in _events
                                if float(event.get("ts") or 0.0) == _first_ts
                            }
                            if ("sl_crossed" in _same_time_types and
                                    _same_time_types & {"target_hit", "tp_hit"}):
                                _touch, _ambiguous = "ambiguous", True
                _entry = float(prev.get("price") or 0.0)
                _window_end = prev.get("window_end")
                try:
                    _window_start = float(prev.get("window_start") or 0.0)
                    if not math.isfinite(_window_start) or _window_start <= 0:
                        _window_start = float(prev.get("locked_at") or now)
                except (TypeError, ValueError, OverflowError):
                    _window_start = float(prev.get("locked_at") or now)
                _record_id = str(prev.get("settlement_id") or
                                 f"live-v1:{symbol}:{tf}:"
                                 f"{int(_window_start * 1000)}")
                _pending = {
                    "record_id": _record_id, "entry": _entry,
                    "settle_price": price, "side": _side,
                    "window_end": _window_end, "first_touch": _touch,
                    "barriers_available": _barriers,
                    "ambiguous": _ambiguous,
                }
            _row = self.ledger.settle(
                symbol, tf, _entry, price, _side, window_end=_window_end,
                record_id=_record_id, first_touch=_touch,
                barriers_available=_barriers, ambiguous=_ambiguous)
            if not _row.get("_persisted", True):
                prev["pending_settlement"] = _pending
                self.locks[(symbol, tf)] = prev
                self._save_locks()
                return
            if _row.get("_duplicate"):
                # A crash may happen after the ledger fsync but before the
                # lock file is cleared. Continue from the committed row.
                _entry = float(_row.get("entry") or _entry)
                price = float(_row.get("settle") or price)
                _side = int(_row.get("side") or _side)
                _touch = _row.get("first_touch") or _touch
                _barriers = bool(_row.get("barriers_available", _barriers))
                _ambiguous = bool(_row.get("ambiguous", _ambiguous))
            prev.pop("pending_settlement", None)
            # feed the isotonic calibrator a (confidence, outcome) sample —
            # only settled, non-seeded rows, never the live window
            try:
                if (not _row.get("seeded")) and _row.get("result") in ("win", "loss"):
                    self.calib.observe(symbol, tf, _row.get("conf"),
                                       _row.get("result") == "win")
                    _clk = getattr(self, "_calib_saved_at", 0.0)
                    if time.time() - _clk > 120:
                        self._calib_saved_at = time.time()
                        self.calib.save()
            except Exception:
                pass
            # adaptive conformal state: learn this window's realised
            # |return| so the next band is honest (Gibbs-Candes ACI)
            try:
                _entry = float(prev.get("price") or 0.0)
                if _entry > 0 and price > 0:
                    from .advanced import conformal_update
                    _cf_all = getattr(self, "_conformal", None)
                    if _cf_all is None:
                        _cf_all = self._conformal = {}
                    _cf_all[(symbol, tf)] = conformal_update(
                        _cf_all.get((symbol, tf), {}),
                        abs(price / _entry - 1.0))
                    self.hist.ring(self.hist.tf_key(
                        symbol, tf, "conf_q"), 400).append(
                        float(_cf_all[(symbol, tf)]["q"]))
            except Exception:
                pass
            # close the trading book for this asset if a position was
            # opened under the optional execution module.
            #
            # If the account is armed and has keys, the position is
            # flattened AT THE EXCHANGE first (reduce-only market + cancel
            # of the TP/SL brackets). Before this, settlement popped the
            # local book while the position stayed open on Binance with
            # orphaned brackets — the app said "closed", the exchange said
            # "open", and the one-position-per-symbol gate meant the app
            # could never manage it again. When the account is not live
            # there is nothing at the exchange to flatten, so the local
            # book is booked exactly as before.
            try:
                _cfg = getattr(self.trader, "cfg", {}) or {}
                _live = bool(_cfg.get("enabled") and _cfg.get("api_key")
                             and _cfg.get("api_secret"))
                _recon = (self.trader.reconcile(symbol, price)
                          if _live else None)
                _row_close = self.trader.close_position(
                    symbol, price, _row.get("result"))
                if _recon and _row_close is not None:
                    _row_close["exchange_closed"] = bool(_recon.get("closed"))
                    _row_close["exchange_why"] = _recon.get("why")
            except Exception:
                log.exception("settlement close failed for %s", symbol)
            # grade the probability forecast made for THAT window:
            # only windows where a barrier was actually decided count
            try:
                _pred = self._prob_pred.pop((symbol, tf), None)
                if (_pred and _pred.get("p_target") is not None
                        and abs(float(_pred.get("cycle_start") or 0)
                                - float(prev.get("window_start") or 0)) < 1.0
                        and _touch in ("target", "stop")):
                    _y = 1.0 if _touch == "target" else 0.0
                    _pa = self._prob_audit
                    _pa["n"] += 1
                    _pa["sum_pred"] += float(_pred["p_target"])
                    _pa["tp_first"] += int(_y)
                    _pa["brier_sum"] += (float(_pred["p_target"]) - _y) ** 2
                    if _pred.get("p_target_cal") is not None:
                        _pa["brier_cal_sum"] = (_pa.get("brier_cal_sum") or 0.0) + (float(_pred["p_target_cal"]) - _y) ** 2
                    self._save_prob_audit()
            except Exception:
                pass
            # REINFORCEMENT loop on a settled window: score the issue
            # bandit, then run the error-finder over every reward ring
            try:
                _ctx = prev.get("rl_ctx")
                was_issued = (float(prev.get("confidence") or 0.0)
                              > self.rl.issue_threshold(_ctx))
                _won = (_row.get("result") == "win"
                        if _row.get("result") in ("win", "loss")
                        else None)
                self.rl.on_settle(issued=was_issued, won=_won, context=_ctx)
                # METACOGNITION: grade the thought that produced this call, so
                # the app learns whether its own reasoning was good
                if _won is not None:
                    self._grade_last_thought(symbol, tf, _won)
                # Q-table: grade the mix this window actually traded
                # under (decided windows only)
                self.rl.q_on_settle(
                    prev.get("rl_state"), prev.get("rl_action"),
                    (1.0 if _won else -1.0) if _won is not None else None)
                self.rl.audit(self)
            except Exception:
                log.exception("rl settle loop failed")
            # Hedge-Flow Tracer direction ring (grades component102 like
            # every other input: sign vs realised move)
            try:
                _tv = self._tracer_last.get((symbol, tf))
                if move_ok := (float(prev.get("price") or 0.0) > 0):
                    _move = price - float(prev.get("price") or 0.0)
                    if _tv is not None and abs(_tv) > 0.02 and _move != 0:
                        self.hist.ring(
                            self.hist.tf_key(symbol, tf, "facc_tracer"), 50
                        ).append(1.0 if ((_tv > 0) == (_move > 0)) else -1.0)
            except Exception:
                pass
            # forward table: settle the PREVIOUS lock's frozen readings
            if prev.get("fv"):
                try:
                    with guard("settle.forward_record"):
                        self.forward.record(symbol, tf,
                                            {int(k): float(v)
                                             for k, v in prev["fv"].items()},
                                            float(prev.get("price") or 0.0),
                                            price, record_id=_record_id)
                except Exception:
                    pass
            # §7 genesis: score the admitted set walk-forward on the
            # settled move (never in-sample)
            if prev.get("genesis_values"):
                try:
                    with guard("settle.genesis_score"):
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
                    with guard("settle.meta_learn"):
                        self.meta.note_pick(prev["inputs"], up)
                        self.meta.record(symbol, tf, prev["inputs"], up)
                except Exception:
                    pass
            # graded exactly once — drop it so a halted pass below (or a
            # later GRC) can never double-grade the same window
            self.locks.pop((symbol, tf), None)
            try:
                self._save_locks()
            except Exception:
                pass
        # docs11 §4.9: halt gates ONLY the new lock — history above is
        # always written first (settlement must never be skipped)
        if time.time() < float(getattr(self, "_halt_until", {})
                               .get(symbol, 0.0)):
            return
        _dom = ((res.get("emotion") or {}).get("dominant")
                or (res.get("brain") or {}).get("dominant"))
        dom = _lock_emotion_text(_dom)
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
        # docs11 §5 path 4 — LEARNED THEME INPUTS: the per-theme net
        # contribution from the evaluation layer folds into the news source
        # column, so meta fitting, the evidence ledger and hindsight all
        # learn from theme priors (same inputs dict, frozen at lock time).
        ti = (self.news.last_eval.get(symbol) or {}).get("theme_inputs") or {}
        if ti:
            net = sum(float(v) for v in ti.values())
            inputs["news"] = round(
                max(-1.0, min(1.0, float(inputs.get("news") or 0.0) + net)), 5)
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
            # entry price: live price, else the LAST CLOSE on the tape —
            # a lock created before the first tick had price0 and every
            # settlement of that window graded as 'skip' (entry missing)
            "price": round(
                self.feed.prices().get(symbol, 0.0)
                or (self.feed.state(symbol).closes[-1]
                    if self.feed.state(symbol).closes else 0.0), 6),
            "reason": str(res.get("reason") or "")[:500],
            "emotion": str(dom or ""),
            "locked_at": now, "computed_at": now,
            "window_start": start, "window_end": end,
            "relocks": 0, "relock_at": None,
            "fv": fv,
            "inputs": inputs,
            "genesis_values": dict(self._genesis_values.get((symbol, tf)) or {}),
        }
        # calibration: remember this window's confidence for the settle
        # pass at the NEXT boundary (paired with the realised move)
        self._calib_pending[(symbol, tf)] = round(
            float(res.get("confidence") or 0.0), 4)
        self._maybe_review_hindsight(symbol, tf, now)   # cadence-gated §8.3
        self._save_locks()

    # ── guide §1: per-timeframe mini-chart series ─────────────────────
    def _provisional_cycle_dict(self, symbol: str, tf: str, now: float,
                                price: float) -> dict | None:
        """Read-only preview card for a TF that has no locked cycle yet.

        Everything here is labelled `provisional` — the UI prints PREVIEW
        and never treats these levels as a contract."""
        from .verdict import describe as vd_describe
        res = (self.last_result.get((symbol, tf))
               or self.display_systems.get((symbol, tf)) or {})
        if not res:
            return None
        direction = int(res.get("direction") or 0)
        conf = float(res.get("confidence") or 0.0)
        mins = list(TIMEFRAMES[tf]["targets"])
        tps = res.get("tps") or []
        lv_by_k = {int(t.get("k") or i + 1): t.get("level")
                   for i, t in enumerate(tps) if isinstance(t, dict)}
        targets = [{"k": i + 1, "minutes": m, "level": lv_by_k.get(i + 1),
                    "status": "preview" if lv_by_k.get(i + 1) else "open",
                    "direction": direction, "locked_at": None,
                    "hit_ts": None, "locked_cci": None}
                   for i, m in enumerate(mins)]
        end = next_grc_ts(now)
        # ONE call to the shared rule, so the four published fields describe
        # the same decision (they were three separate calls before).
        _pv_pre = vd_describe(symbol, tf, direction, conf, res.get("tier"))
        return {
            "provisional": True,
            "symbol": symbol, "tf": tf, "W": int(TIMEFRAMES[tf]["minutes"]),
            "direction": direction, "confidence": round(conf, 4),
            "strength": float(res.get("strength") or 0.0),
            "issued": bool(res.get("issued")),
            # same rule as a locked card (app/verdict.py) — a PREVIEW that
            # reads WATCH while its own confidence is below the floor was the
            # other half of the same reported contradiction
            "verdict": _pv_pre["verdict"],
            "gate_ok": _pv_pre["gate_ok"],
            "issue_note": _pv_pre["issue_note"],
            "trend_gate": _pv_pre["trend_gate"],
            "prediction_lock": {},
            "tier": res.get("tier") or "watch-only",
            "p_side": res.get("p_side"),
            "entry": (res.get("entry")
                      or (res.get("lock") or {}).get("price")
                      or (price or None)),
            "tp": (targets[-1]["level"] if targets else None),
            "sl": res.get("sl"),
            "sl_provisional": True,
            "ratio": res.get("ratio"),
            "targets": targets,
            "countdown_s": (max(0, int(end - now)) if end else None),
            "cycle_start": None, "cycle_end": end,
            "risk": {"note": (res.get("levels_source")
                              or "preview only — no locked contract yet"),
                     "provisional": True},
            "canonical": None, "level_contract": None, "contract_id": None,
            "contract_state": "preview", "levels_ok": True,
            "winrate": None, "winrate_n": 0,
            "execution_eligible": False, "issued_at": None,
            "levels_source": res.get("levels_source"),
        }

    def _bars_1m(self, symbol: str) -> list[dict]:
        st = self.feed.state(symbol)
        n = min(len(st.times), 1200)
        start = len(st.times) - n
        return [{"t": st.times[i], "o": st.opens[i], "h": st.highs[i],
                 "l": st.lows[i], "c": st.closes[i], "v": st.volumes[i]}
                for i in range(start, len(st.times))]

    def tf_charts(self, symbol: str, bars_1m: list[dict] | None = None) -> dict:
        """tf_charts[asset][tf] = build_tf_candles(tape, tf).

        One resampling rule server-side (step + lookback per TF), so every
        card that draws a TF draws the SAME series, and each TF keeps its
        own epoch (epoch_start_tf) instead of inheriting tf1's clock."""
        from .chartlink import build_tf_candles, TF_CHART_SPEC
        if bars_1m is None:
            bars_1m = self._bars_1m(symbol)
        cache = getattr(self, "_tf_chart_cache", None)
        if cache is None:
            cache = self._tf_chart_cache = {}
        sig = (len(bars_1m),
               int(bars_1m[-1]["t"]) if bars_1m else 0,
               round(float(bars_1m[-1].get("c") or 0), 8) if bars_1m else 0.0)
        prev = cache.get(symbol)
        if prev and prev[0] == sig:
            return prev[1]
        epochs = getattr(self, "epoch_start_tf", None)
        if epochs is None:
            epochs = self.epoch_start_tf = {}
        out: dict = {}
        for tf in TF_CHART_SPEC:
            spec = TF_CHART_SPEC[tf]
            step = max(1, int(spec["step"]))
            W = int(TIMEFRAMES.get(tf, {}).get("minutes", 22) or 22)
            # four completed windows + the current one; the guide's
            # 60/50/48/48 lookbacks are the MINI-chart minimum, and using
            # them for the main chart truncated it to a sliver (the chart
            # looked like it "vanished" after switching TF/asset).
            need = 4 * int(math.ceil(W / step)) + 8
            series = build_tf_candles(bars_1m, tf,
                                     lookback=max(int(spec["lookback"]), need))
            out[tf] = series
            if series["candles"]:
                epochs.setdefault(symbol, {})[tf] = int(series["candles"][0]["t"])
        cache[symbol] = (sig, out)
        return out

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
        _mins = TIMEFRAMES[tf]["targets"]
        # LOCK FIRST, then anchor EVERYTHING to the lock's entry price:
        # the settlement lock, the targets, the TP/SL block and the
        # displayed "entry (locked)" are one number. (Before this, the
        # block anchored to the compute price while the lock froze the
        # live price seconds later — two different "entries" on screen,
        # 75,672 in the trace vs 76,182 in the hero.)
        self._settle_and_lock(symbol, tf, now, res)
        _lock_now = self.locks.get((symbol, tf)) or {}
        _price0 = (float(_lock_now.get("price") or 0.0)
                   or res.get("_price", 0.0)
                   or self.feed.prices().get(symbol, 0.0))
        # anchor sanity: at boot the lock can carry the backfill
        # tape's end price while the live feed already trades far
        # away (a 170k anchor under a 79k market froze a nonsense
        # forecast/levels). If the anchor disagrees with the live
        # price by >20%, the live price wins.
        #
        # 20 % IS NOT A SANITY BOUND (live defect 2026-11-27). A lock
        # stamped at 4,923.24 while ETH traded 4,952.50 is only 59 bps out
        # — it sailed under the old threshold — yet its 5.5 bp stop landed
        # 32 bps BEHIND the market, so the first completed bar "touched"
        # it and the history recorded a loss seconds after ANALYSE, before
        # price moved at all. The anchor must BE the market it is graded
        # against: anything beyond the seal tolerance is re-anchored here,
        # at the source, and the re-anchor is recorded on the cycle.
        _live = float(self.feed.prices().get(symbol, 0.0) or 0.0)
        _anchor_skew_bps = (abs(_price0 / _live - 1.0) * 1e4
                            if _live > 0 and _price0 > 0 else 0.0)
        if _live > 0 and _price0 > 0 and (
                abs(_price0 / _live - 1.0) > 0.20
                or _anchor_skew_bps > 2.0):
            if abs(_price0 / _live - 1.0) <= 0.20:
                # small skew: record it so the page can say the anchor was
                # refreshed rather than pretending it was always the tape
                self._anchor_reanchored = getattr(
                    self, "_anchor_reanchored", {})
                self._anchor_reanchored[(symbol, tf)] = {
                    "ts": now, "from": _price0, "to": _live,
                    "skew_bps": round(_anchor_skew_bps, 2)}
                log.warning("anchor re-anchored %s|%s: %g -> live %g "
                            "(%.1f bps skew)", symbol, tf, _price0, _live,
                            _anchor_skew_bps)
            _price0 = _live
        # MEASURED-MOVES TP/SL (docs11-charts B-1): levels come from
        # this asset's measured excursions at each horizon — q80 MAE
        # stop, clip(1.5·SL·f(edge), q50, q95) target — frozen here.
        # The old conf·ATR·√min projection made ~2 bp targets that
        # barely changed window to window. USER RULE kept: the TF's
        # own horizon TP (LAST target) is calculated FIRST.
        from .moves_risk import risk_block as _risk_block
        # ── TREND INTERACTION: FLIP, NEVER WITHHOLD (2026-11-26) ────────
        # The user directive is "raise the win rate WITHOUT abstention". The
        # measured interaction in app/trend_gate.py says a call that chases
        # the recent 2-decision trend wins 46.0 % while its fade wins 56.2 %
        # on 861 real windows. This is the ONE place the side of a new lock
        # is written, so it is the one place the flip is applied: the side
        # changes, the entry does not, and every level below is re-measured
        # for the new side from the same locked anchor. Grading stays
        # continuous (every window still gets exactly one sealed side), and
        # the deciding evidence travels on the cycle.
        _dir_raw = res["direction"] or 1
        _dir = int(_dir_raw)
        _trend_flip: dict | None = None
        try:
            from .trend_gate import check as _tg_check
            _tdec = _tg_check(symbol, tf, int(_dir))
            if _tdec.get("flipped") and int(_tdec.get("side_out") or 0) in (-1, 1):
                _dir = int(_tdec["side_out"])
                _trend_flip = {
                    "applied": True, "from": int(_dir_raw), "to": _dir,
                    "why": str(_tdec.get("why") or ""),
                    "trend": _tdec.get("trend"),
                    "trend_bps": _tdec.get("trend_bps"),
                    "min_trend_bps": _tdec.get("min_trend_bps"),
                    "measured": _tdec.get("measured"),
                }
                res = dict(res)
                res["direction_before_flip"] = int(_dir_raw)
                res["direction"] = _dir
                res["trend_flip"] = _trend_flip
                log.info("trend flip %s/%s %+d -> %+d (%.1f bps vs %.1f bps "
                         "min)", symbol, tf, int(_dir_raw), _dir,
                         float(_tdec.get("trend_bps") or 0.0),
                         float(_tdec.get("min_trend_bps") or 0.0))
            elif _tdec.get("blocked"):
                # legacy TREND_GATE=abstain mode: the withhold semantic is
                # kept for A/B, and it is still visible on the cycle.
                _trend_flip = {"applied": False, "withheld": True,
                               "why": str(_tdec.get("why") or "")}
                res = dict(res)
                res["trend_withheld"] = _trend_flip
        except Exception:
            log.exception("trend rule failed for %s/%s", symbol, tf)
            _dir = int(_dir_raw)
            _trend_flip = None
        res["_trend_flip"] = _trend_flip
        _edge = min(1.0, abs(float(res.get("confidence") or 0.0)) / 0.35)
        _st = self.feed.state(symbol)
        _closes = list(_st.closes or [])
        _highs = list(_st.highs or _closes)
        _lows = list(_st.lows or _closes)
        try:
            _spread = float((self.feed.book_status(symbol) or {})
                            .get("spread_bps") or 0.0)
        except Exception:
            _spread = 0.0
        # 1-MINUTE spacing declared explicitly: `minutes` is a HOLDING
        # PERIOD, never a bar count. The backtest harness used to feed a 1h
        # tape into this same function, which measured 22 HOURS of excursion
        # for a 22-minute horizon (see app/moves_risk.moves_stats).
        _BAR_MIN = 1.0
        _blocks = {m: _risk_block(_price0, _dir, m, _edge, _closes,
                                  _highs, _lows, atr=res.get("atr", 0.0),
                                  spread_bps=_spread,
                                  bar_minutes=_BAR_MIN)
                   for m in dict.fromkeys(_mins)}
        # The trend rule's "is this move a trend or tick noise?" filter needs
        # THIS asset's measured σ for THIS window — the same measurement that
        # just priced the TP/SL, never a second invented volatility scale.
        try:
            from . import trend_gate as _tg_sig
            _hz_key = _mins[-1]
            _hz_block = _blocks.get(_hz_key) or {}
            _tg_sig.set_window_sigma(
                symbol, tf,
                float(_hz_block.get("sigma_bps")
                      or _hz_block.get("measured_sigma_bps") or 0.0) or None)
        except Exception:
            log.exception("trend sigma publish failed for %s/%s", symbol, tf)
        # per-asset MANDATE geometry (2y measured, mandate_search):
        # sl ×= k, tp = sl × R on the raw measured block — exactly the
        # geometry the EV numbers were computed on, so the generic
        # asset profile and TF envelope are skipped for this cell.
        from . import mandate_cfg as _mcfg_c
        _mscale = _mcfg_c.level_scale(symbol, tf)
        _mandate_cell = _mscale is not None
        if _mscale:
            _k, _Rm = _mscale
            # NOTE: sl_bps is scaled FIRST and tp_bps is then k x R times the
            # ORIGINAL stop, because reading the already-scaled stop made
            # the published ratio R and no longer k x R — a mandate cell
            # asking for 2:1 shipped 1.667:1 on eight of sixteen cells
            # (BTC tf2/tf4, ETH tf1/tf2, BNB tf2/tf3, PAXG tf3/tf4) with
            # nothing on the card to say the number had moved.
            for _b in _blocks.values():
                _sl0 = float(_b["sl_bps"])
                _b["sl_bps"] = round(_sl0 * _k, 3)
                _b["tp_bps"] = round(_sl0 * _k * _Rm, 3)
                _b["tp"] = round(_price0 * (1 + _dir * _b["tp_bps"] / 1e4), 6)
                _b["sl"] = round(_price0 * (1 - _dir * _b["sl_bps"] / 1e4), 6)
                _b["rr"] = (round(_b["tp_bps"] / _b["sl_bps"], 3)
                            if _b["sl_bps"] else None)
                _b["below_floor"] = bool(
                    _b["sl_bps"] > 0 and _b["tp_bps"] / _b["sl_bps"] < 1.0)
                _b["note"] = ((_b.get("note") or "")
                              + f" · mandate k={_k} × SL, R={_Rm} "
                                "(2y-measured)")
        # per-asset nature scales the measured distances (gold wider,
        # BNB tighter, ETH wider) — bps scaled, levels recomputed from
        # the same locked anchor so the invariant cannot break
        _prof = asset_profile(symbol)
        if (not _mandate_cell
                and (_prof["sl_mult"] != 1.0 or _prof["tp_mult"] != 1.0)):
            for _b in _blocks.values():
                # asset nature scales the 1 unit (SL); TP scales by the
                # SAME factor so the block's measured ratio survives
                _b["sl_bps"] = round(_b["sl_bps"] * _prof["sl_mult"], 3)
                _b["tp_bps"] = round(_b["tp_bps"] * _prof["sl_mult"]
                                     * (_prof["tp_mult"] / _prof["sl_mult"]), 3)
                _b["below_floor"] = bool(
                    _b["sl_bps"] > 0 and _b["tp_bps"] / _b["sl_bps"] < 1.0)
                _b["tp"] = round(_price0 * (1 + _dir * _b["tp_bps"] / 1e4), 6)
                _b["sl"] = round(_price0 * (1 - _dir * _b["sl_bps"] / 1e4), 6)
                _b["rr"] = round(_b["tp_bps"] / _b["sl_bps"], 3) \
                    if _b["sl_bps"] else None
                _b["note"] = (_b.get("note", "")
                              + f" · asset profile ×{_prof['sl_mult']} SL"
                                f" TP = 2x unit ({symbol} nature)")
        # TF ENVELOPE (user rule): a longer timeframe can never carry a
        # smaller TP/SL range than a shorter one. The measured q80 MAE is
        # noisy at small samples and can invert (tf2 < tf1 on a quiet tape),
        # which reads as "the range does not scale with the timeframe".
        # Enforce a monotone floor of +3% per step on the bps distances and
        # re-anchor the levels at the SAME locked entry; the measured value
        # is kept in `measured_*_bps` so nothing is hidden.
        _prev_sl = _prev_tp = 0.0
        for _m in sorted(_blocks):
            _b = _blocks[_m]
            _b["measured_sl_bps"] = _b.get("sl_bps")
            _b["measured_tp_bps"] = _b.get("tp_bps")
            if (not _mandate_cell and _prev_sl > 0
                    and _b["sl_bps"] < _prev_sl * 1.03):
                _b["sl_bps"] = round(_prev_sl * 1.03, 3)
                _b["envelope"] = "raised to keep SL increasing with the timeframe"
            if (not _mandate_cell and _prev_tp > 0
                    and _b["tp_bps"] < _prev_tp * 1.03):
                # ── THE ENVELOPE RAN ON TP ALONE AND BROKE THE RATIO ────
                # This raised TP to keep targets increasing with the
                # timeframe but left SL where it was, so the PUBLISHED
                # reward:risk collapsed. Live: BTC tf2 shipped TP 7 bps
                # against SL 7 bps (1.00:1) and BNB tf4 1.28:1, while tf1
                # showed 2.00:1 — the same badge quoting three strategies.
                # Break-even is 62.5 % at 0.6:1 and 33.3 % at 2:1, so this
                # is not cosmetic.
                #
                # The monotone-timeframe rule is kept, but it is now applied
                # to the RANGE and the target is then re-derived from it:
                # SL is widened with the timeframe, and TP follows at
                # exactly RR_TARGET x the resulting stop. The measured q80
                # MAE stays in `measured_*_bps`, so the envelope is still
                # visible and nothing is hidden.
                from .moves_risk import RR_TARGET as _RRT_ENV
                _b["sl_bps"] = round(max(_b["sl_bps"], _prev_sl * 1.03), 3)
                _b["tp_bps"] = round(_RRT_ENV * _b["sl_bps"], 3)
                _b["envelope"] = ("range raised to keep SL increasing with "
                                  f"the timeframe; TP re-fixed at {_RRT_ENV:g}x "
                                  "the stop")
            _b["tp"] = round(_price0 * (1 + _dir * _b["tp_bps"] / 1e4), 6)
            _b["sl"] = round(_price0 * (1 - _dir * _b["sl_bps"] / 1e4), 6)
            _b["rr"] = (round(_b["tp_bps"] / _b["sl_bps"], 3)
                        if _b["sl_bps"] else None)
            _b["below_floor"] = bool(
                _b["sl_bps"] > 0 and _b["tp_bps"] / _b["sl_bps"] < 1.0)
            if _b.get("envelope"):
                _b["note"] = (_b.get("note", "")
                              + f" · TF envelope: {_b['envelope']}")
            _prev_sl, _prev_tp = float(_b["sl_bps"]), float(_b["tp_bps"])
        _horizon_block = _blocks[_mins[-1]]     # horizon TP ALWAYS calculated FIRST
        # CANONICAL LADDER (audit §1.2): TP1 is a monotone fraction of the
        # entry → TF-TP span, built in ONE place (app/contracts.py). It used
        # to be the separate 5-minute measured block, which printed $4
        # above the entry while the ANALYSING panel's "Rung 1" printed
        # +86 bps — two formulas, two ladders, one window.
        # TP2/TP3 keep the sequential-lock status (a target is only
        # *tracked* once the one before it is hit) but their frozen price
        # is published on the canonical contract, so no panel ever prints
        # "— OPEN" for a level the lock already knows.
        from .contracts import build_ladder as _build_ladder
        _rungs = _build_ladder(_price0, _dir, _horizon_block["tp"],
                               list(_mins))
        _rung1 = (_rungs[0]["level"] if _rungs
                  else _blocks[_mins[0]]["tp"])
        targets: list[TargetState] = []
        for _k, _minutes in enumerate(_mins, start=1):
            if _k == len(_mins):
                _level, _status = _horizon_block["tp"], "locked"
            elif _k == 1:
                _level, _status = _rung1, "locked"
            else:
                # TP2/TP3 stay level-less on the cycle BY DESIGN (a rung is
                # only tracked once the one before it is hit) — their frozen
                # price is published on the canonical contract instead, so no
                # panel prints "OPEN" for a level the lock already knows.
                _level, _status = None, "pending"
            targets.append(TargetState(
                k=_k, minutes=_minutes, level=_level, status=_status,
                direction=_dir, locked_at=now,
                locked_cci=res.get("cci_adjusted")))
        # window anchored to the LOCK (user: a timeframe is the time
        # FROM the prediction — 'after 22 minutes, what price?').
        # The epoch grid previously made the real duration anything
        # from seconds to W (countdown drift). Settlement keying in
        # _settle_and_lock keeps its own grid, unaffected.
        _g_start, _g_end = now, now + W * 60
        # public verdict for the card: the contract keeps `issued` true so a
        # call can be graded, but the badge must say what actually cleared.
        # THE FLIP IS ALREADY IN res["direction"] (applied above, from the
        # ONE decision this window's side came from), so the rule is asked
        # about the side that will actually be published — never the raw one,
        # which would flip a second time and produce the original side back.
        _gate_ok, _verdict, _issue_note, _side_rule = resolve_verdict(
            symbol, tf, int(res.get("direction") or 0),
            float(res.get("confidence") or 0.0), res.get("tier"),
            res.get("issue_note"))
        cyc = TFCycle(
            symbol=symbol,
            tf=tf,
            W=W,
            direction=(_side_rule or res["direction"]),
            confidence=res["confidence"],
            cci=res["cci_adjusted"],
            drhe=res["drhe"],
            issued=res["issued"],
            verdict=_verdict,
            gate_ok=_gate_ok,
            issue_note=_issue_note,
            trend_gate=_trend_evidence(symbol, tf,
                                       int(res.get("direction") or 0)),
            trend_flip=(_trend_flip or {}),
            tier=str(res.get("tier") or "watch-only"),
            p_side=float(res.get("p_side") or 0.5),
            calib_n=int(res.get("calib_n") or 0),
            calib_source=str(res.get("calib_source") or ""),
            noise_floor_s=float(res.get("noise_floor_s") or 0.0),
            score_floor_ratio=float(res.get("score_floor_ratio") or 0.0),
            tier_rule=str(res.get("tier_rule") or ""),
            cycle_start=_g_start,
            cycle_end=_g_end,
            strength=float(res.get("strength") or 0.0),
            winrate=res.get("winrate"),
            winrate_n=int(res.get("winrate_n") or 0),
            targets=targets,
            active_k=1,   # every issued side is tracked + graded
            warning=warning,
            refreshed_at=now,
            qdpc_tau=res.get("qdpc_tau"),
        )
        # ── NEW system (horizon forecast): the forecast price at the
        # end of this timeframe IS the target; SL stays from the
        # measured block. Magnitude = σ-scaled expected move, tilted
        # by signal strength — labelled a model forecast, and graded
        # against the realised price at the horizon like any call.
        _mode = str(self.agents.settings.get("system_mode", "old"))
        _mmode = _mcfg_c.mode(symbol)
        if _mmode in ("old", "new"):
            _mode = _mmode          # per-asset mandate mode wins
        _sigma = max(float(_horizon_block.get("sigma_bps") or 0.0),
                     float(_horizon_block.get("q50") or 0.0))
        # TAIL-UNIT SCALING (2y real-data mandate): the old formula
        # (factor × σ) is a MEAN-scaled target measured against a
        # q80-MAE stop — on real bars that ratio is structurally
        # 0.65–0.80, so EVERY new-mode window fell below the 1:1
        # floor and the model could never issue. The stop is the
        # tape's own tail unit; the target is now that unit tilted
        # by signal strength. σ stays in the payload for display.
        #
        # THE TILT MAY NEVER PUSH A SIGNAL UNDER 1:1 (user rule, absolute).
        # The old band started at 0.7 because 0.35 × 2.0 = 0.7 — any signal
        # that was not almost saturated landed below the floor and the cell
        # went "watch-only" permanently. That is what the board showed for
        # six days: 12 of 16 cells NOT TRADEABLE, 4 ungradeable. The tilt
        # now modulates how FAR ABOVE the floor a target sits (1:1 .. 2:1)
        # instead of deciding whether the trade is legal at all. A weak
        # signal still gets the smallest legal target and still carries the
        # weak score in its tier/issue_note — it is no longer silenced.
        _factor = (0.5 + 0.5 * min(1.0, abs(float(
            res.get("cci_adjusted") or res.get("cci") or 0.0)) / 0.08))
        _sl_bps = float(_horizon_block.get("sl_bps") or 0.0)
        # WHICH target ratio to aim at, most specific first:
        #   1. the ratio this exact (asset, tf) cell measured out of sample
        #      (app/mandate_cfg.py CELL_RR — only 3 cells clear both the
        #      1:1 floor and positive holdout EV);
        #   2. the per-asset mandate override;
        #   3. the global default.
        # The scan in docs/WINRATE_AT_1_TO_1.md found the optimal ratio is
        # NOT constant across cells — tf1/tf2 have zero positive-EV points
        # at >= 1:1 while tf4 has 52 of 84 — so a single global knob was
        # the wrong shape. A cell with no measurement keeps the default
        # rather than being silenced.
        _rr_cell = _mcfg_c.new_rr_cell(symbol, tf)
        _rr_n = _rr_cell or _mcfg_c.new_rr(symbol) or NEW_TARGET_RR
        _rr_src = ("measured cell ratio" if _rr_cell
                   else "mandate override" if _mcfg_c.new_rr(symbol)
                   else "global default")
        _mag = _factor * _rr_n * _sl_bps if _sl_bps > 0 \
            else _factor * _sigma
        _fc_price = round(_price0 * (1 + _dir * _mag / 1e4), 6) \
            if _price0 > 0 else None
        cyc.forecast = {
            "mode": _mode, "price": _fc_price,
            "move_bps": round(_mag, 2), "minutes": W,
            "end_ts": _g_end,
            "direction": (1 if (_fc_price or 0) >= _price0 else -1)
            if _fc_price else int(_dir),
            # where this window's target ratio came from, and — when it was
            # measured — the out-of-sample evidence behind it. A ratio
            # presented without its provenance is a number nobody can audit.
            "rr_target": round(float(_rr_n), 4),
            "rr_source": _rr_src,
            "rr_evidence": (dict(_mcfg_c.CELL_RR.get((symbol, tf)) or {})
                            if _rr_cell else None),
        }
        if _mode == "new" and _fc_price and _price0 > 0:
            cyc.tp = _fc_price
            cyc.sl = _horizon_block["sl"]
            cyc.tp_dist = abs(_fc_price - _price0)
            cyc.sl_dist = abs(_price0 - cyc.sl)
            cyc.ratio = (round(cyc.tp_dist / cyc.sl_dist, 6)
                         if cyc.sl_dist else None)
            cyc.direction = cyc.forecast["direction"]
        # user rule: below 1:1 is not worth trading -> watch-only
        if (cyc.ratio is not None and cyc.ratio < 1.0):
            cyc.tier = "watch-only"
            cyc.events.append({"ts": now, "type": "below_ratio_floor",
                               "ratio": cyc.ratio})
        cyc.events.append({"ts": now, "type": "cycle_fresh", "reason": reason})
        cyc.risk = dict(_horizon_block)
        if _mode == "new" and _fc_price and _price0 > 0:
            # NEW system: end-price forecast is TP; the measured catastrophic
            # stop remains unchanged. Preserve both raw and effective evidence.
            cyc.risk["raw_tp_bps"] = cyc.risk.get("tp_bps")
            cyc.risk["tp"] = cyc.tp
            cyc.risk["tp_dist"] = cyc.tp_dist
            cyc.risk["tp_bps"] = cyc.tp_dist / _price0 * 1e4
            cyc.risk["rr"] = cyc.risk["ratio"] = cyc.ratio

        # ── ONE CANONICAL CONTRACT PER (ASSET, TF, CYCLE) ───────────────
        # Frozen here, read by every panel: entry, TF-horizon TP, TP1..TP3
        # rungs, SL, ATR, both confidences, digest. The ladder invariant
        # (ENTRY < TP1 < TP2 < TP3 < TF-TP for a BUY, mirrored for a SELL)
        # is enforced by app/contracts.py and re-verified on read.
        from .contracts import build_contract as _build_contract
        _eff_dir = int(cyc.direction or _dir or 1)
        _eff_dir = 1 if _eff_dir >= 0 else -1
        _eff_tp = float(cyc.tp or _horizon_block.get("tp") or 0.0)
        _eff_sl = float(cyc.sl or _horizon_block.get("sl") or 0.0)
        # ── SIDE/LEVEL CONSISTENCY (live defect 2026-11-26) ─────────────
        # 9 of 16 cells shipped a TP/SL on the WRONG side of their own
        # entry: e.g. BTC tf1 published direction −1 with tp 67521 > entry
        # 66467 > sl 65836, i.e. the levels of a BUY under a SELL badge. A
        # user reading that sees a target ABOVE the market on a short — the
        # exact "TP/SL/entry don't match" complaint, and it is unfixable by
        # any amount of chart explanation.
        #
        # Cause: the paper trail showed risk["direction"] = +1 while the
        # cycle published −1, so the pair was frozen for one side and
        # stamped with the other. Rather than chase every writer, the
        # single site that builds the contract now VERIFIES the pair against
        # the side it is about to publish and re-prices it for that side at
        # the SAME entry when they disagree. The measured distances are
        # preserved (app/levels.mirror reflects the pair about its entry);
        # re-pricing is the fallback, and `levels_repaired` records which
        # happened so a silent correction can never look like a measurement.
        _pair_ok = (_eff_tp > 0 and _eff_sl > 0 and _price0 > 0 and
                    ((_eff_tp > _price0 > _eff_sl) if _eff_dir > 0
                     else (_eff_tp < _price0 < _eff_sl)))
        if _price0 > 0 and not _pair_ok and _eff_tp > 0 and _eff_sl > 0:
            _mir = None
            try:
                from .levels import mirror as _mirror
                _mir = _mirror({"entry": _price0, "tp": _eff_tp,
                                "sl": _eff_sl,
                                "direction": -_eff_dir}, _eff_dir)
            except Exception:
                _mir = None
            if _mir:
                _eff_tp, _eff_sl = float(_mir["tp"]), float(_mir["sl"])
                _repair = "mirrored about the entry (distances preserved)"
            else:
                _fresh = _risk_block(_price0, _eff_dir, _mins[-1], _edge,
                                     _closes, _highs, _lows,
                                     atr=res.get("atr", 0.0),
                                     spread_bps=_spread, bar_minutes=1.0)
                _eff_tp, _eff_sl = float(_fresh["tp"]), float(_fresh["sl"])
                _repair = ("re-measured for the published side (the inherited "
                           "pair could not be mirrored)")
            cyc.tp, cyc.sl = _eff_tp, _eff_sl
            cyc.tp_dist = abs(_eff_tp - _price0)
            cyc.sl_dist = abs(_price0 - _eff_sl)
            cyc.ratio = (round(cyc.tp_dist / cyc.sl_dist, 6)
                         if cyc.sl_dist else None)
            cyc.risk = {**(cyc.risk or {}), "tp": _eff_tp, "sl": _eff_sl,
                        "tp_dist": cyc.tp_dist, "sl_dist": cyc.sl_dist,
                        "ratio": cyc.ratio, "direction": _eff_dir,
                        "tp_bps": round(cyc.tp_dist / _price0 * 1e4, 2),
                        "sl_bps": round(cyc.sl_dist / _price0 * 1e4, 2)}
            cyc.levels_repaired = _repair
            cyc.events.append({"ts": now, "type": "levels_repaired",
                               "direction": _eff_dir, "how": _repair})
            log.warning("levels repaired %s/%s: side %+d, pair was on the "
                        "wrong side of %.6f — %s", symbol, tf, _eff_dir,
                        _price0, _repair)
        cyc.canonical = _build_contract(
            symbol=symbol, timeframe=tf, direction=_eff_dir,
            entry=_price0, horizon_tp=_eff_tp, sl=_eff_sl,
            minutes=list(_mins), atr=float(res.get("atr") or 0.0),
            base_confidence=float(res.get("confidence_base_snr")
                                  or res.get("confidence") or 0.0),
            final_confidence=float(res.get("confidence") or 0.0),
            cci_adjusted=float(res.get("cci_adjusted") or 0.0),
            cycle_id=cyc.cycle_id, locked_at=now,
            source="measured-moves+canonical-ladder",
        )
        # NOTE: cyc.targets keeps its sequential hit-tracking order (a
        # pending target carries no level so the touch detector ignores
        # it). The FROZEN price of every rung lives on cyc.canonical, and
        # the UI falls back to it — that is what removes the "— OPEN"
        # cells without letting an untracked rung fire a hit.
        self.cycles[(symbol, tf)] = cyc
        self._apply_risk(symbol, tf, cyc, res)
        if cyc.sl is None and _price0 > 0:
            # unissued (the common case): the measured-moves block IS
            # the displayed TP/SL — frozen, on the correct side of the
            # entry, instead of a 1×ATR seed that changed every snapshot
            cyc.tp = _horizon_block["tp"]
            cyc.sl = _horizon_block["sl"]
            cyc.tp_dist = abs(_horizon_block["tp"] - _price0)
            cyc.sl_dist = abs(_price0 - _horizon_block["sl"])
            cyc.ratio = _horizon_block["rr"]
        if cyc.ratio is None:
            cyc.ratio = _horizon_block.get("rr")
        self._install_cycle_contract(cyc, now, reason=reason)
        # OPTIONAL execution: the trader re-checks every safety gate
        # itself (disabled / unarmed / sub-floor / wrong-side levels
        # are refused + logged) — this hook can never fire a random
        # prediction by accident.
        try:
            self.trader.consider({
                "symbol": symbol, "tf": tf,
                "issued": bool(cyc.execution_eligible),
                "execution_eligible": bool(cyc.execution_eligible),
                "feed_mode": getattr(self.feed, "mode", None),
                "feed_simulated": getattr(self.feed, "mode", None) != "binance",
                "feed_age_s": round(now - float(
                    self.feed.state(symbol).last_tick or 0), 1)
                if self.feed.state(symbol).last_tick else 999.0,
                "direction": int(cyc.direction or 0),
                "entry": (cyc.level_contract or {}).get("entry"),
                "tp": (cyc.level_contract or {}).get("tp"),
                "sl": (cyc.level_contract or {}).get("sl"),
                "contract": dict(cyc.level_contract or {}),
                "contract_lock": dict(cyc.contract_lock or {}),
                "contract_id": (cyc.level_contract or {}).get("id"),
                "cycle_id": cyc.cycle_id, "lock_id": cyc.lock_id,
                "level_revision": cyc.level_revision,
                "tp_bps": _horizon_block.get("tp_bps"),
                "sl_bps": _horizon_block.get("sl_bps"),
                # advanced sizing layers: HAR-RV + EVT + RMT mode
                # dampen size (never lever up); EVT also bounds the
                # catastrophic stop distance actually placed
                "size_mult": float(((self.last_result.get(
                    (symbol, tf)) or {}).get("advanced") or {})
                    .get("size_mult", 1.0)),
                "sl_max_bps": float((((self.last_result.get(
                    (symbol, tf)) or {}).get("advanced") or {})
                    .get("evt") or {}).get("cat_stop_bps") or 0.0),
                "window_end": _g_end})
        except Exception:
            log.exception("trading consider failed")
        return cyc

    # ==================================================================
    # Initial Analysis Phase
    # ==================================================================
    async def _analysis_loop(self) -> None:
        """Readiness evaluation + prominent countdown broadcast (1 Hz)."""
        last_beat = 0.0
        while not self._stop.is_set() and self.phase == "analysis":
            now = time.time()
            # this loop ticks at 4 Hz; the dry-run inside readiness is a full
            # pipeline pass whenever it is due, so it must not run on the loop
            await run_pass(self._update_readiness)
            ready = all(self.readiness.values())
            # computable = everything except full-history bars, with
            # at least MIN_BARS_COMPUTE bars present. Locking then
            # (with warning if bars are still short) beats holding
            # the user behind a 4-minute overlay for one slow item.
            core = all(v for k, v in self.readiness.items()
                       if k != "bars")
            computable = core and getattr(self, "_enough_any", False)
            expired = now >= self.phase_deadline

            if ready or computable or expired:
                warn = not ready
                self.phase_warning = warn
                await self._lock_initial(warning=warn)
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
        self._enough_any = enough_any
        self.readiness["brains"] = self.brains.ready
        self.readiness["emotions"] = self.emotions.ready
        self.readiness["news"] = self.news.ready
        self.readiness["physics"] = self.physics.ready
        if enough_any and not self.readiness["formulas"]:
            # Dry-run the whole pipeline ONCE to let every subsystem report
            # readiness. It must not repeat every GRC: this dry-run costs the
            # same as a full pass (seconds of CPU × 6 assets), and the old
            # code re-ran it on EVERY GRC until physics warmed up — ~10 s of
            # blocking work every 4 minutes, i.e. the "ENGINE BUSY" banner
            # coming back over and over. Attempts now back off exponentially
            # (1st GRC, then ×2 up to 10 min) and the number of attempts plus
            # the last error are reported, so a stuck dry-run is visible in
            # /api/health rather than silently burning the box.
            try:
                attempt = int(self._readiness_tries) + 1
            except AttributeError:
                attempt = 1
            now = time.time()
            due = getattr(self, "_readiness_next_at", 0.0) or 0.0
            if now >= due:
                self._readiness_tries = attempt
                try:
                    for symbol in SYMBOL_ORDER:
                        self.compute_asset(symbol, record=False, display=False)
                    self.readiness["formulas"] = True
                    self.readiness["weights"] = True
                    self._readiness_error = ""
                except Exception as exc:
                    self._readiness_error = repr(exc)[:200]
                    log.exception("analysis dry-run failed")
                backoff = min(600.0, 30.0 * (2 ** max(0, attempt - 1)))
                self._readiness_next_at = time.time() + backoff

    async def _lock_initial(self, warning: bool) -> None:
        if self.locked:
            return                      # claim immediately (analysis loop and
                                        # a GRC during phase may race here)
        self.locked = True
        now = time.time()
        for symbol in SYMBOL_ORDER:
            try:
                # the initial lock computes all six assets; inline it stalled
                # the loop right at the moment the UI expects its first frame
                res_pack = await run_pass(
                    self.compute_asset, symbol, record=True, display=True)
            except Exception:
                log.exception("initial compute failed for %s", symbol)
                res_pack = self.last_result
            for tf in TF_ORDER:
                res = res_pack.get(tf)
                if not res:
                    continue
                self.last_result[(symbol, tf)] = res
                try:
                    self.fresh_cycle(symbol, tf, now, res,
                                     reason="initial_lock", warning=warning)
                except Exception as exc:
                    # One failing lane used to abort the WHOLE initial lock:
                    # phase never reached "live", _tick_loop never ran and no
                    # TF ever got a contract (the "waiting for the first
                    # lock…" dead end). Log, record and keep going.
                    log.exception("initial lock failed for %s/%s", symbol, tf)
                    errs = getattr(self, "cycle_errors", None)
                    if errs is None:
                        errs = self.cycle_errors = {}
                    errs[f"{symbol}:{tf}"] = repr(exc)[:200]
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
            self.grc_started_at = time.time()
            _prof: dict[str, float] = {}
            self.grc_profile = _prof

            def _t(label, since):
                _prof[label] = round((time.time() - since) * 1000, 1)
                return time.time()
            now = time.time()
            self.last_grc = ts
            self.grc_count += 1

            if self.phase == "analysis":
                # non-recording pass: the ONE official recording pass for the
                # initial lock happens in _lock_initial (never double-append
                # GRC baselines within a single cycle)
                for symbol in SYMBOL_ORDER:
                    try:
                        # OFF THE EVENT LOOP. compute_asset is seconds of pure
                        # CPU per asset; run inline in this async def it froze
                        # /api/health and the websocket for the whole pass,
                        # which the client saw as "ENGINE BUSY" and which made
                        # the brain chip fall back to its offline text.
                        res_pack = await run_pass(
                            self.compute_asset, symbol,
                            record=False, display=False)
                        for tf in TF_ORDER:
                            self.last_result[(symbol, tf)] = res_pack[tf]
                    except Exception:
                        log.exception("analysis GRC compute failed")
                _t("analysis_compute", now)
                await run_pass(self._update_readiness)
                _t("analysis_readiness", now)
                if all(self.readiness.values()):
                    await self._lock_initial(warning=False)
                return

            if not self.locked:
                return

            _done = 0
            for symbol in SYMBOL_ORDER:
                try:
                    # yield to the loop between assets when it is starved
                    # (the pass is a burst; the board must stay responsive)
                    await asyncio.sleep(0)
                    _gov.cooperate()
                    # same reason as above: keep the loop answering while the
                    # heavy per-asset pipeline runs on a worker thread
                    res_pack = await run_pass(
                        self.compute_asset, symbol, record=True, display=True)
                except Exception:
                    log.exception("GRC compute failed for %s", symbol)
                    continue
                # one progress frame per finished asset: the client's silence
                # detector stays fed even though the work is off-loop now
                _done += 1
                await self.emit({"type": "grc_progress",
                                 "done": _done, "of": len(SYMBOL_ORDER),
                                 "symbol": symbol, "ts": time.time()})
                # Section 11 agent signal + multiplier, Section 16 double-check
                try:
                    await self._apply_agents(symbol, res_pack)
                except Exception:
                    log.exception("agent pass failed for %s", symbol)
                for tf in TF_ORDER:
                    res = res_pack[tf]
                    self.last_result[(symbol, tf)] = res
                    # lock rules are per-lane bookkeeping (measured 154 ms);
                    # they mutate this lane's cycle only, so they stay in
                    # order but off the loop
                    await run_pass(self._apply_grc_lock_rules,
                                   symbol, tf, now, res)
                # Sections 10/13 learning + history bookkeeping (263 ms of
                # rings/scorecards per asset — the second biggest loop-bound
                # cost in the pass after compute_asset)
                try:
                    await run_pass(self._record_learning, symbol, res_pack, now)
                except Exception:
                    log.exception("learning bookkeeping failed for %s", symbol)
                # The display snapshot is synced inside _apply_agents, which
                # runs BEFORE the lock rules — so any lock event that moves
                # the number (an SL SAFE penalty, a confidence refresh, a
                # direction override) left the card showing the pre-lock
                # value while the graded record carried the new one. Re-sync
                # once the per-asset pass is complete.
                self._sync_display(symbol, res_pack)
            _t("compute+agents+locks+learning", now)
            _mark = time.time()
            # Section 10 macro (every 15 GRCs) + deep (every 247 min)
            try:
                await run_pass(self.learning.maybe_macro, self.grc_count, self)
                await run_pass(self.learning.maybe_deep, self)
                await run_pass(self.learning.save, self)
            except Exception:
                log.exception("macro/deep learning cycle failed")
            await run_pass(self.cci.save)
            await run_pass(self.physics.save)
            _t("learning_saves", _mark)
            _mark = time.time()
            await self._broadcast_full("grc")
            _t("broadcast", _mark)
            # what the pass ACTUALLY cost, once it is fully done — the clock's
            # handler_ms only covers the coroutine, which now hands its CPU to
            # worker threads; without this the health panel under-reported.
            self.grc_ms = round((time.time() - self.grc_started_at) * 1000, 1)

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
                    price, int(cyc.direction or t4.direction or 0),
                    res["confidence"], res["atr"], t4.minutes
                )
                t4.locked_at = now
                t4.locked_cci = res["cci_adjusted"]
                cyc.events.append({"ts": now, "type": "extreme_override",
                                   "contract_changed": False,
                                   "candidate_direction": t4.direction})

        if not res["issued"]:
            # NO TRADE — insufficient confidence. Levels are NEVER nulled
            # or re-derived from spot here (that nulling + snapshot
            # re-seeding is what put a BUY stop above its entry).
            # A flipped/flat call re-predicts FROM NOW with a fresh
            # frozen block; same side keeps its frozen lock levels.
            new_dir = int(res.get("direction") or 0)
            if new_dir == 0 or new_dir != int(cyc.direction or 0):
                self.fresh_cycle(symbol, tf, now, res,
                                 reason="retake_direction")
                return
            cyc.issued = False
            cyc.confidence = res["confidence"]
            cyc.cci = res["cci_adjusted"]
            cyc.drhe = res["drhe"]
            cyc.active_k = None
            cyc.refreshed_at = now
            # BADGE + CONFIDENCE ARE ONE NUMBER. This branch refreshed the
            # confidence every GRC but never re-derived the badge, so a card
            # could read "ISSUED / cleared the floor" while its own number was
            # 0.009 against a 0.385 floor (the reported mismatch error).
            cyc.gate_ok, cyc.verdict, cyc.issue_note = public_verdict(
                symbol, tf, int(cyc.direction or 0), cyc.confidence, cyc.tier,
                None)
            cyc.trend_gate = _trend_evidence(symbol, tf,
                                             int(cyc.direction or 0))
            cyc.regime_conflict = self.regime_conflict(
                symbol, int(cyc.direction or 0))
            self._save_cycle_lock_state(cyc)
            return

        if not cyc.issued or cyc.direction == 0:
            # confidence recovered (or first live GRC) -> fresh lock
            self.fresh_cycle(symbol, tf, now, res, reason="confidence_recovered")
            return

        prev_cci = cyc.cci
        new_cci = res["cci_adjusted"]
        shift = abs(new_cci - prev_cci)

        # --- Tp1: lowest lock strength — adjusts at every GRC before being hit
        #
        # SIDE IS FROZEN. This used to project rung 1 on
        # `res["direction"] or t1.direction` — the LIVE lean — and then write
        # that lean back onto the rung. On a live SHORT the tape leaned long,
        # so rung 1 was rebuilt ABOVE the entry (BNB|tf1: short at 439.47 with
        # TP1 at 439.64), i.e. a target on the losing side of its own trade,
        # and every later rung inherited the flipped side. A cycle's side is
        # the locked prediction (§3): rungs may move in PRICE, never in SIDE.
        t1 = cyc.target(1)
        if t1.status == "locked":
            _side1 = int(cyc.direction or t1.direction or 0)
            # Rung 1 stays free to adjust (lowest lock strength, by design)
            # but is CLAMPED into the trade's own region. The old write was
            # unconstrained and produced targets on the LOSING side of their
            # own trade — 11 of 16 lanes in data/locks.json (BNB tf4: short at
            # 443.06, rung 1 at 493.50, stop at 466.46).
            _rc = cyc.level_contract or {}
            _e1 = float(_rc.get("entry") or (cyc.risk or {}).get("entry")
                        or price)
            _tp1 = float(_rc.get("tp") or cyc.tp or 0.0)
            _sl1 = float(_rc.get("sl") or cyc.sl or 0.0) or None
            _m1 = [t.minutes for t in cyc.targets]
            _fall1 = rung_level(_e1, _side1, _tp1, _m1,
                                (_m1[-1] if _m1 else None), 1, _sl1)
            t1.level = clamp_rung(
                self._project_level(
                    price, _side1, res["confidence"], res["atr"], t1.minutes),
                _e1, _side1, _sl1, fallback=_fall1, fallback_sl=_sl1,
            ) or t1.level
            t1.direction = _side1
            t1.locked_at = now
            t1.locked_cci = new_cci
            # The measured rung may be refreshed; the cycle side and its
            # frozen contract remain the original locked prediction.

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
        # The badge is re-derived from the number it sits next to — it used to
        # be skipped whenever a cycle stayed open, so a card kept the badge
        # from its FIRST analysis forever ("ISSUED" at conf 0.009).
        #
        # NOTE: the trend gate is deliberately NOT fed here. A GRC repeats the
        # SAME price for an open window, and its ring only holds three
        # samples, so observing per GRC filled it with [p, p, p] — the trend
        # became 0 on every cell within minutes and the gate silently stopped
        # withholding anything (caught on the live board: 16/16 cells
        # "trend 0"). The gate's sampled series is one entry per SETTLED
        # window (see _settle_and_lock) plus the warm replay, and nothing else.
        cyc.gate_ok, cyc.verdict, cyc.issue_note = public_verdict(
            symbol, tf, int(cyc.direction or 0), cyc.confidence, cyc.tier,
            res.get("issue_note"))
        cyc.trend_gate = _trend_evidence(symbol, tf, int(cyc.direction or 0))
        # The chip and the badge are compared on EVERY card write, not only
        # inside ANALYSE. This is the path a GRC takes, and it is the one
        # that produced the screenshot: a ▲ BULLISH* headline beside a
        # 📉 impulse bear chip, with no disagreement banner in sight.
        cyc.regime_conflict = self.regime_conflict(
            symbol, int(cyc.direction or 0))
        if cyc.regime_conflict:
            cyc.execution_eligible = False
        self._apply_risk(symbol, tf, cyc, res)
        self._save_cycle_lock_state(cyc)

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
        t.hit_ts = now
        cyc.events.append({"ts": now, "type": "target_hit", "k": k, "level": t.level})
        if (cyc.tp is not None and t.level is not None
                and abs(float(t.level) - float(cyc.tp))
                <= max(1e-7, abs(float(cyc.tp)) * 1e-10)):
            cyc.contract_tp_hit = True
            cyc.execution_eligible = False

        if k in (1, 2):
            # "The moment Tp1 is reached, immediately calculate and lock Tp2
            #  using all current data" (same for Tp2 -> Tp3)
            res = self._event_result(symbol, tf)
            self.last_result[(symbol, tf)] = res
            # Sub-target recalculation must not mutate the cycle side or its
            # entry-bound TP/SL contract after the barrier has been locked.
            nxt = cyc.target(k + 1)
            from .moves_risk import risk_block as _rb2
            _st2 = self.feed.state(symbol)
            nxt.level = _rb2(
                self.feed.prices().get(symbol, 0.0),
                cyc.direction, nxt.minutes,
                min(1.0, abs(float(res.get("confidence") or 0.0)) / 0.35),
                list(_st2.closes or []), list(_st2.highs or []),
                list(_st2.lows or []), atr=res.get("atr", 0.0))["tp"]
            nxt.direction = cyc.direction
            nxt.status = "locked"
            nxt.locked_at = now
            nxt.locked_cci = res["cci_adjusted"]
            cyc.active_k = k + 1
            cyc.confidence = res["confidence"]
            cyc.cci = res["cci_adjusted"]
            # the badge describes the number beside it — re-derive it here too
            cyc.gate_ok, cyc.verdict, cyc.issue_note = public_verdict(
                symbol, tf, int(cyc.direction or 0), cyc.confidence, cyc.tier,
                None)
            cyc.trend_gate = _trend_evidence(symbol, tf,
                                             int(cyc.direction or 0))
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
        self._save_cycle_lock_state(cyc)

    # -- SL Hit Protocol ----------------------------------------------------
    async def _on_sl_hit(self, symbol: str, tf: str, now: float) -> None:
        cyc = self.cycles.get((symbol, tf))
        if cyc is None or not cyc.issued:
            return
        original_direction = cyc.direction
        cyc.sl_observed = True
        cyc.execution_eligible = False
        if not any(e.get("type") == "sl_crossed" for e in cyc.events):
            cyc.events.append({"ts": now, "type": "sl_crossed",
                               "sl": cyc.sl, "observed": True})
        # E2/E9 history: this cycle experienced an SL hit (both branches)
        self.hist.record_sl_hit(symbol)
        # FULL RECALCULATION EVENT: every formula across all subsystems,
        # from scratch, using the latest available data
        res = self._event_result(symbol, tf)
        self.last_result[(symbol, tf)] = res
        new_direction = res["direction"]
        # a recalc whose fused score sits INSIDE the noise floor has not
        # actually reversed — its side is a weak lean (the ladder says
        # so). Closing a position on a coin-flip recalc is wrong: treat
        # a sub-floor recalc as "no reversal" (SAFE branch).
        _floor = float((res.get("ladder") or {}).get("noise_floor") or 0.02)
        if new_direction != original_direction and \
                abs(float(res.get("cci_adjusted") or 0.0)) <= _floor:
            new_direction = original_direction

        if new_direction == original_direction:
            # SAFE — BE CAUTIOUS (amber): prediction stays active,
            # SL tightened by 15%, next sub-target recalculated, confidence -0.15
            cyc.safe_events += 1
            cyc.sl_tighten *= (1.0 - SL_TIGHTEN_FACTOR)
            cyc.confidence = max(0.0, cyc.confidence - SL_CONFIDENCE_PENALTY)
            # docs11-doublecheck: the side stays issued + graded; the
            # penalty is expressed as a one-step tier cut
            _steps = ["watch-only", "quarter", "half", "full"]
            if cyc.tier in _steps and _steps.index(cyc.tier) > 0:
                cyc.tier = _steps[_steps.index(cyc.tier) - 1]
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
            # the penalty above changed the number; the badge follows it
            cyc.gate_ok, cyc.verdict, cyc.issue_note = public_verdict(
                symbol, tf, int(cyc.direction or 0), cyc.confidence, cyc.tier,
                None)
            cyc.trend_gate = _trend_evidence(symbol, tf,
                                             int(cyc.direction or 0))
            self._apply_risk(symbol, tf, cyc, res)
            cyc.refreshed_at = now
            cyc.safe_alert = True
            await self.emit({
                "type": "sl_event",
                "kind": "safe",
                "symbol": symbol,
                "tf": tf,
                # the toast used to read only the message — "SAFE" with
                # no clue WHICH trade.  The text now names the cell,
                # the side and the lock id; the raw fields are here too
                "side": "BUY" if original_direction > 0 else "SELL",
                "lock_id": str(getattr(cyc, "lock_id", "") or ""),
                "message": (
                    f"{symbol} {tf.upper()} — SAFE, BE CAUTIOUS "
                    f"(SL hit vs {'BUY' if original_direction > 0 else 'SELL'}"
                    f" · lock {str(getattr(cyc, 'lock_id', '') or '')[:10]})"),
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
            cyc.tier = "watch-only"   # side stays issued + graded (doc §3)
            cyc.pending_reset = True
            cyc.close_alert = True
            # Keep the frozen entry/TP/SL visible for audit; only execution
            # eligibility and the next-cycle state change after a barrier.
            cyc.execution_eligible = False
            cyc.refreshed_at = now
            await self.emit({
                "type": "sl_event",
                "kind": "close",
                "symbol": symbol,
                "tf": tf,
                # user complaint: "it told me to cancel trade but not
                # which one" — the toast must identify the position
                "side": "BUY" if original_direction > 0 else "SELL",
                "lock_id": str(getattr(cyc, "lock_id", "") or ""),
                "message": (
                    f"{symbol} {tf.upper()} — CLOSE POSITION IMMEDIATELY "
                    f"(SL hit vs {'BUY' if original_direction > 0 else 'SELL'}"
                    f" · lock {str(getattr(cyc, 'lock_id', '') or '')[:10]})"),
                "ts": now,
            })
        self._save_cycle_lock_state(cyc)

    # ==================================================================
    # 1 Hz tick loop: hit detection, countdown rollovers, phase beat
    # ==================================================================
    async def _tick_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(1.0)
            if self.phase != "live" or not self.locked:
                continue
            now = time.time()
            # AUTO-PREDICT (user toggle): keep the full staged analysis
            # fresh without a click — one run at most every 45 s, never
            # stacked on a running job (manual ▶ ANALYSE behaves as before)
            # DEFAULT = MANUAL (user directive): the loop only runs when
            # the user explicitly turned auto-predict ON; only the user
            # can change it.  The auto scope also honours the per-cell
            # ANALYSE timers (a locked cell is never auto-run).
            if (self.agents.settings.get("auto_predict", False)
                    and not os.environ.get("PYTEST_CURRENT_TEST")
                    and (self.analyse_job is None
                         or self.analyse_job.get("status") != "running")
                    and now - self._last_auto_analyse >= 45.0):
                self._last_auto_analyse = now
                try:
                    from . import hold_policy as _hp_auto
                    _sym_a = SYMBOL_ORDER[0]
                    if not _hp_auto.cell_locks().locked(_sym_a, "tf1"):
                        self.start_analyse(_sym_a, "tf1")
                except Exception:
                    log.exception("auto-predict analyse failed")
            # BRAIN SELF-HEAL: whatever flipped the connectome to
            # absent (failed build, dead worker, restart), retry the
            # load every 60 s without the user hunting for RECONNECT.
            try:
                if now - getattr(self, "_last_brain_watch", 0) > 60:
                    self._last_brain_watch = now
                    # heal() starts ensure() on a WORKER thread: the load is
                    # 8.5 s of CPU and running it here froze the event loop
                    # (health/WS dead -> the UI banner said "engine offline")
                    self.heal()
            except Exception:
                pass
            prices = self.feed.prices()
            # SEALED PREDICTIONS settle on the live tape (TP touch, SL touch,
            # or strict sign at their own horizon), then teach the learners.
            try:
                settled = self.settle_prediction_locks(now)
                if settled:
                    await self.emit({"type": "prediction_settled",
                                     "locks": settled[-6:],
                                     "stats": self.predlocks.stats()})
            except Exception:
                log.exception("prediction-lock tick failed")
            # HOLD ERA ACCOUNTING: every cell advances its own
            # one-position-per-side-era state (window boundaries +
            # closed-bar first-touch checks, real round-trip friction)
            try:
                self._hold_tick(now)
            except Exception:
                log.exception("hold era tick failed")
            for symbol in SYMBOL_ORDER:
                price = prices.get(symbol, 0.0)
                if price <= 0:
                    continue
                for tf in TF_ORDER:
                    cyc = self.cycles.get((symbol, tf))
                    if cyc is None:
                        # SELF-HEAL: if the initial lock (or a rollover)
                        # failed, this lane would never show a contract.
                        # Retry at most every 30 s, off the lock path.
                        retry = getattr(self, "_cycle_retry", None)
                        if retry is None:
                            retry = self._cycle_retry = {}
                        _k = (symbol, tf)
                        if now - float(retry.get(_k, 0.0)) > 30.0:
                            retry[_k] = now
                            try:
                                _res = self._event_result(symbol, tf)
                                self.last_result[_k] = _res
                                self.fresh_cycle(symbol, tf, now, _res,
                                                 reason="self_heal")
                            except Exception:
                                log.exception("self-heal lock failed %s/%s",
                                              symbol, tf)
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
                        # THE STOP CANNOT BE THE ONLY BARRIER MEASURED. This
                        # block tested SL first (and `continue`d), then TP —
                        # so a tick that touched BOTH always booked the loss,
                        # and a contract TP could only be booked inside this
                        # branch. Resolve the pair, then dispatch; a genuine
                        # same-tick tie still goes to the stop (app/grading.py).
                        _sl_touched = cyc.sl is not None and not cyc.sl_observed and (
                            (price <= cyc.sl and cyc.direction > 0)
                            or (price >= cyc.sl and cyc.direction < 0))
                        _tp_touched = cyc.tp is not None and not cyc.contract_tp_hit and (
                            (price >= cyc.tp and cyc.direction > 0)
                            or (price <= cyc.tp and cyc.direction < 0))
                        if _sl_touched:
                            await self._on_sl_hit(symbol, tf, now)
                            # a resolved loss ends this lane's window: the TP
                            # of a dead contract must not be booked afterwards
                            continue
                        if _tp_touched:
                            cyc.contract_tp_hit = True
                            cyc.execution_eligible = False
                            cyc.events.append(
                                {"ts": now, "type": "tp_hit", "tp": cyc.tp}
                            )
                            self._save_cycle_lock_state(cyc)
                    elif not cyc.issued:
                        # NOT issued (below the confidence floor): the
                        # locked TP/SL levels are still displayed, so
                        # track them observationally — price crossing
                        # TP1 while the card still said LOCK/open is
                        # exactly the "TP/SL in air" the user reported.
                        for t in cyc.targets:
                            if t.status == "locked" and t.level is not None:
                                d = t.direction or cyc.direction
                                if (d > 0 and price >= t.level) or (
                                        d < 0 and price <= t.level):
                                    t.status = "hit"
                                    t.hit_ts = now
                                    cyc.events.append(
                                        {"ts": now, "type": "target_hit",
                                         "k": t.k, "level": t.level,
                                         "observed": True})
                                    if cyc.tp is not None and abs(float(t.level) - float(cyc.tp)) <= max(1e-7, abs(float(cyc.tp)) * 1e-10):
                                        cyc.contract_tp_hit = True
                                        cyc.execution_eligible = False
                        if cyc.sl is not None and cyc.direction:
                            against = (price <= cyc.sl
                                       and cyc.direction > 0) or (
                                           price >= cyc.sl
                                           and cyc.direction < 0)
                            if against and not cyc.sl_observed:
                                cyc.sl_observed = True
                                cyc.events.append(
                                    {"ts": now, "type": "sl_crossed",
                                     "sl": cyc.sl, "observed": True})
                        if cyc.contract_tp_hit or cyc.sl_observed:
                            cyc.execution_eligible = False
                            self._save_cycle_lock_state(cyc)
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
    def _brain_live(self) -> dict:
        """One source of truth for the brain chip, read live every snapshot.

        `state` is the component's own state machine (absent | building |
        ready); the per-(asset,tf) display copy can lag by a GRC, so a chip
        driven from it reports "offline" over a loaded connectome. The chip
        text always comes from the component (`fw.chip()`), and the grace
        fields let the UI hold a green chip through a rebuild rather than
        flashing red.
        """
        fw = getattr(getattr(self, "brains", None), "fw", None)
        if fw is None:
            return {"state": "absent", "ready": False,
                    "chip": {"kind": "absent", "text": "brain — not wired"},
                    "pass_age_s": None, "load_ms": None}
        age = (round(time.time() - fw.last_pass_at, 1)
               if getattr(fw, "last_pass_at", 0) else None)
        return {
            "state": fw.state,
            "ready": fw.state == "ready",
            "chip": fw.chip(),
            "pass_age_s": age,
            "pass_ms": getattr(fw, "last_pass_ms", None),
            "load_ms": getattr(fw, "load_ms", None),
            "neurons": (fw.meta or {}).get("neurons"),
            "last_error": (fw.last_error or "")[:200] if fw.last_error else "",
        }

    def heal(self) -> dict:
        """Repair the parts that used to need a manual restart.

        * flywire state == absent  -> re-run ensure() on a worker thread
          (the load itself is 8.5 s of CPU: doing it on the event loop used
          to freeze the whole app, which is what the UI reported as
          "engine offline").
        * brain pass stalled       -> tell the pass thread to re-arm.

        Returns a small report so /api/health can show what was done.
        """
        out: dict[str, Any] = {"at": time.time()}
        fw = getattr(getattr(self, "brains", None), "fw", None)
        if fw is not None:
            out["brain_state"] = fw.state
            out["pass_age_s"] = (round(time.time() - fw.last_pass_at, 1)
                                 if getattr(fw, "last_pass_at", 0) else None)
            th = getattr(self, "_fw_heal_thread", None)
            # "building" is not a healthy state — it is an UNPROVEN one. Ask
            # the component whether the attempt is still alive; this is what
            # turns a silent, killed or wedged build into an automatic retry
            # instead of an indefinite "brain offline — vote withheld".
            if fw.state == "building":
                try:
                    sup = fw.supervise()
                    out["supervise"] = sup
                    out["action"] = ("flywire-build-" + str(sup.get("action")))
                except Exception as exc:      # pragma: no cover - defensive
                    out["supervise"] = {"error": repr(exc)[:200]}
                    out["action"] = "supervise-failed"
            elif fw.state == "absent" and (th is None or not th.is_alive()):
                def _load() -> None:
                    try:
                        fw.ensure()
                    except Exception:
                        log.exception("flywire self-heal failed")
                self._fw_heal_thread = threading.Thread(
                    target=_load, name="flywire-heal", daemon=True)
                self._fw_heal_thread.start()
                out["action"] = "flywire-ensure-thread-started"
            elif (fw.state == "ready" and out["pass_age_s"] is not None
                  and out["pass_age_s"] > 120):
                try:
                    fw._start_pass_thread()
                    out["action"] = "brain-pass-rearmed"
                except Exception as exc:      # pragma: no cover - defensive
                    out["action"] = f"pass-rearm failed: {exc!r}"
            else:
                out.setdefault("action", "none")
        return out

    def _refresh_flywire_display(self) -> None:
        """Display must never LIE about the brain: the connectome can
        hot-swap in BETWEEN GRC passes, and the user saw 'brain absent'
        on the dashboard while the brain tab showed ready. On every
        snapshot read, any display entry whose flywire state disagrees
        with the live component is re-evaluated (a vote costs ~5 ms)."""
        fw = self.brains.fw
        ready_now = fw.state == "ready"
        for (sym, tf), disp in self.display_systems.items():
            brain = disp.get("brain")
            if brain is None:
                continue
            f = brain.get("flywire")
            cur = bool(f and f.get("ready"))
            if cur == ready_now:
                continue
            fresh = self.brains._flywire_eval(
                sym, tf, self._last_raws.get((sym, tf)))
            brain["flywire"] = fresh
            self._fw_last_vote[(sym, tf)] = float(fresh.get("vote") or 0.0)
            tk = (disp.get("ticks") or {}).get("brain")
            if tk is not None:
                tk["flywire"] = {
                    "on": abs(float(fresh.get("vote") or 0.0)) > 1e-6,
                    "v": fresh.get("vote"),
                    "ready": bool(fresh.get("ready")),
                }

    # ------------------------------------------------------------------
    # CERTIFIED SLEEVES (app/edge_gate.py + app/sleeves.py)
    # ------------------------------------------------------------------
    def sleeve_block(self, now: float | None = None) -> dict[str, Any]:
        """Live status of every certified sleeve, plus honest coverage.

        The registry is produced by ``backtest/edge_scan.py`` and is the
        single source of truth: a sleeve that the gate did not promote to
        ``open`` cannot emit a state here. Results are cached for a few
        seconds because the snapshot is built far more often than the
        15m bar grid can change.
        """
        ttl = 5.0
        stamped = self._sleeve_cache
        if (
            stamped is not None
            and now is not None
            and stamped[0] + ttl > now
        ):
            return stamped[1]
        try:
            reg = load_sleeve_registry()
            block = evaluate_sleeves(self.feed, registry=reg, now=now)
        except Exception:
            log.exception("sleeve evaluation failed")
            block = {"signals": [], "assets": {}, "certified": [],
                     "summary": {}, "note": "sleeve evaluation failed"}

        # every UI asset gets a row, even with nothing certified — silence
        # is reported as silence, never as a signal
        # THE EDGE SCAN NEVER TESTED SOME ASSETS, and "none" used to read as
        # "we tested it and it failed" — the two are different facts and the
        # user asked which one they were looking at. The registry records
        # which assets it actually covered; anything outside that set is
        # reported as UNTESTED, with the scan's own candidate count.
        _covered = set((block.get("assets") or {}).keys())
        _tested = int((block.get("summary") or {}).get("total") or 0)
        for symbol in SYMBOL_ORDER:
            if symbol in _covered:
                continue
            block["assets"][symbol] = {
                "sleeve_id": "",
                "asset": symbol,
                "status": "none",
                "state": "none",
                "untested": True,
                "note": (f"the edge scan ({_tested} candidates across the "
                         f"covered assets) never produced a candidate for "
                         f"{symbol}, so there is nothing to certify and "
                         f"nothing to reject"),
                "gate_note": ("not a failure and not a live signal — this "
                              "asset was outside the scan's search space"),
            }
        block["coverage"] = sorted(block["assets"].keys())

        # drive the paper portfolio from the same certified signals
        try:
            block["portfolio"] = self._step_portfolio(block)
        except Exception:
            log.exception("portfolio step failed")
            block["portfolio"] = {"error": "portfolio step failed"}

        self._sleeve_cache = (now if now is not None else time.time(), block)
        return block

    def regime_conflict(self, symbol: str, direction: int) -> dict | None:
        """The 5-state regime read on THIS asset's tape vs a certified side.

        One definition, used by BOTH writers of a card (ANALYSE and the GRC
        refresh). It used to exist only inside the analyse path, so every
        card produced by a GRC refresh carried the chip and no conflict
        record — which is exactly the screenshot the user sent: a headline
        **▲ BULLISH\*** beside a chip reading **📉 impulse bear · conf 0.20**,
        with no disagreement banner anywhere on the card. The chip and the
        badge disagreed because only one of the two paths compared them.

        Returns ``None`` when the tape is not in an impulse state, when the
        side is flat, or when they AGREE. Never raises: a failed
        classification is the absence of a conflict claim, not a silent
        "disagrees".
        """
        try:
            if int(direction) not in (-1, 1):
                return None
            from .regime import classify_tape as _ctape
            st = self.feed.state(symbol)
            rq = _ctape(list(st.closes or []),
                        highs=list(st.highs or []) or None,
                        lows=list(st.lows or []) or None,
                        volumes=list(getattr(st, "volumes", None)
                                     or []) or None)
            want = {"IMPULSE_BULL": 1, "IMPULSE_BEAR": -1}.get(
                str(rq.get("state") or ""), 0)
            if not want or want == int(direction):
                return None
            return {"state": str(rq.get("state")),
                    "label": str(rq.get("label") or ""),
                    "regime_conf": rq.get("confidence"),
                    "regime_wants": want,
                    "certified_side": int(direction)}
        except Exception:
            log.exception("regime classification failed for %s", symbol)
            return None

    def _step_portfolio(self, block: dict[str, Any]) -> dict[str, Any]:
        """Expose the existing paper book without fabricating live fills.

        The old integration filled at the latest one-minute *close* while
        claiming a next-15-minute-bar open, and also consumed simulated tape.
        Freeze entries, exits and marks until the execution adapter can prove
        source, timing and OHLC parity with the research fill contract. The
        existing ledger is preserved; it is not evidence of verified fills.
        """
        # VERIFIED EXECUTION (2026-11-27). The book was frozen because the
        # two things a paper fill must prove — a NEXT-BAR OPEN and REAL-TAPE
        # provenance — were not implemented. They are now (app/exec_live.py),
        # so the freeze is lifted exactly as far as the tape supports it:
        # real Binance tape fills; simulated or gapped tape reports the
        # precise rule that blocks it, per asset. The book is never paused
        # without naming the fact that paused it.
        try:
            from . import exec_live as _xl
            return _xl.step_live_book(self, block)
        except Exception:
            log.exception("verified paper execution failed")
            snap = self.portfolio.snapshot()
            snap["execution_status"] = "paused"
            snap["execution_reason"] = (
                "Paper execution paused: the verified-fill driver raised — "
                "see the server log for the traceback.")
            snap["entered_this_bar"] = []
            return snap

    def _current_meta_inputs(self) -> dict:
        """Latest frozen subsystem readings, for the meta gauge."""
        for (sym, tf) in (("BTC", "tf1"),) + tuple(self.locks.keys()):
            lock = self.locks.get((sym, tf))
            if lock and lock.get("inputs"):
                return dict(lock["inputs"])
        sysd = self.display_systems.get(("BTC", "tf1")) or {}
        ph = sysd.get("physics") or {}
        ufp = ph.get("ufp") if isinstance(ph, dict) else None
        return {
            "score": float(sysd.get("cci_adjusted") or 0.0),
            "brain": int((((sysd.get("brain") or {}).get("polarity")) or 0)),
            "physics": float((ufp or {}).get("value") if isinstance(ufp, dict)
                             else (ufp or 0.0)),
            "emotion": float(((sysd.get("emotion") or {}).get("aggregate"))
                             or 0.0),
            "news": float((sysd.get("news_value")) or 0.0),
            "agent": int((((sysd.get("agent") or {}).get("direction")) or 0)),
        }


    def build_snapshot(self) -> dict[str, Any]:
        now = time.time()
        # §1.1 every snapshot is versioned at birth — the broadcast path
        # and the REST-primed cache serve the SAME monotonic counter
        self._state_version += 1
        try:
            self._refresh_flywire_display()
        except Exception:
            log.exception("flywire display refresh failed")
        assets: dict[str, Any] = {}
        prices = self.feed.prices()
        for symbol in SYMBOL_ORDER:
            st = self.feed.state(symbol)
            base = st.day_open or (st.closes[0] if st.closes else 0.0)
            price = prices.get(symbol, 0.0)
            tfs: dict[str, Any] = {}
            for tf in TF_ORDER:
                cyc = self.cycles.get((symbol, tf))
                d = cyc.to_dict(now) if cyc else None
                if d is None:
                    # PROVISIONAL CARD: a TF with a computed signal but no
                    # locked cycle yet used to render an empty hero
                    # ("waiting for the first lock…") even though the
                    # analysis had a call, a confidence and a preview
                    # ladder. Publish those, explicitly flagged preview.
                    d = self._provisional_cycle_dict(symbol, tf, now, price)
                if d:
                    # the walk-forward edge model's vote, on the CARD.
                    # It is published from the analysis result, not from the
                    # cycle, because the rung re-runs every pass while the
                    # cycle may still be the previous lock — a card must show
                    # the probability that went into ITS side, not a stale one.
                    try:
                        _emv = (self.last_result.get((symbol, tf)) or {}).get(
                            "edge_model")
                        if _emv:
                            d["edge_model"] = _emv
                    except Exception:
                        log.exception("edge model publish failed %s/%s",
                                      symbol, tf)
                    # BADGE INVARIANT at the one place every card is printed.
                    # Confidence can change between analyses (SL penalties,
                    # tier cuts, ageing), so the published badge is checked
                    # against the published number here: it may be
                    # DOWNGRADED (ISSUE -> WATCH -> NO CALL) and never
                    # upgraded, and gate_ok/verdict/issue_note are always
                    # shipped together with the confidence they describe.
                    try:
                        from .verdict import reconcile_card as _vcard
                        _vcard(symbol, tf, d)
                    except Exception:
                        log.exception("verdict reconcile failed for %s|%s",
                                      symbol, tf)
                # LEVELS GATE (rewrite): the snapshot NEVER invents a
                # level from the current price. Levels come only from
                # the cycle's frozen block anchored at the LOCKED entry
                # (app/levels.py). Valid levels pass untouched; a
                # wrong-side/missing pair is repaired from the frozen
                # block at the locked entry or withheld (UI shows
                # "levels pending") — a BUY stop above its entry can
                # never reach the screen again.
                if d and cyc is not None:
                    lv = _levels.from_cycle(cyc)
                    d["level_contract"] = (dict(cyc.level_contract)
                                            if cyc.level_contract else None)
                    d["contract_id"] = ((cyc.level_contract or {}).get("id"))
                    d["contract_state"] = ("locked" if lv and cyc.level_contract
                                            else "legacy" if lv else "unavailable")
                    if lv:
                        # ── THE SCREEN'S OWN GUARANTEE (2026-11-26) ──────
                        # The contract layer now repairs a pair that arrives
                        # on the wrong side of the entry, but a display must
                        # not depend on an upstream fix having run: if this
                        # pair is still inverted against the side the card is
                        # about to print, mirror it (distances preserved) and
                        # SAY SO. Nine of sixteen live cells were shipping
                        # `SELL` above an entry with its targets on the buy
                        # side — a screen must never be the place where a
                        # trader discovers that.
                        _lv_side = int(d.get("direction") or 0) or \
                            int(cyc.direction or 0)
                        _lv_ok = (_lv_side == 0
                                  or (float(lv["tp"]) > float(lv["entry"])
                                      > float(lv["sl"]) if _lv_side > 0
                                      else float(lv["tp"]) < float(lv["entry"])
                                      < float(lv["sl"])))
                        _lv_note = ""
                        if not _lv_ok:
                            try:
                                from .levels import mirror as _mir2
                                _fix = _mir2(
                                    {"entry": lv["entry"], "tp": lv["tp"],
                                     "sl": lv["sl"], "direction": -_lv_side},
                                    _lv_side)
                            except Exception:
                                _fix = None
                            if _fix:
                                lv = {**lv, "tp": _fix["tp"], "sl": _fix["sl"],
                                      "tp_dist": _fix["tp_dist"],
                                      "sl_dist": _fix["sl_dist"],
                                      "ratio": _fix["ratio"],
                                      "mirrored": True}
                                _lv_note = (" · levels mirrored onto the "
                                            "published side (the stored pair "
                                            "was inverted)")
                        # ── THE 1:1 FLOOR ON THE SCREEN (USER RULE) ─────
                        # `mirror` preserves distances but nothing compared
                        # them, and the legacy frozen block was never floored
                        # at all. Live board 2026-11-27: 7 of 16 cells
                        # published a target closer than their stop (BNB tf1
                        # 0.474, BTC tf1/tf3 0.500, ETH tf1/tf2 0.800, ETH
                        # tf3 0.545, PAXG tf2 0.544). This is the one place
                        # the card's levels are written, so the invariant is
                        # enforced here: the target is widened to meet the
                        # measured stop, the stop never moves, and the card
                        # says when it happened.
                        try:
                            from .levels import floor_rr as _rrfloor
                            _lv_floored = _rrfloor(lv, _lv_side) or lv
                        except Exception:
                            _lv_floored = lv
                        if (float(_lv_floored["tp"]) != float(lv["tp"])):
                            _lv_note = ((_lv_note or "") +
                                        " · target raised to 1:1 (it sat "
                                        f"closer than the stop: "
                                        f"{_lv_floored.get('rr_floor_applied')}:1)")
                            _lv_floored["mirrored"] = lv.get("mirrored", False)
                        lv = _lv_floored
                        d["tp"], d["sl"] = lv["tp"], lv["sl"]
                        d["tp_dist"], d["sl_dist"] = lv["tp_dist"], lv["sl_dist"]
                        d["ratio"] = lv["ratio"]
                        d["entry"] = lv["entry"]
                        d["rr_floor"] = ({"before": lv.get("rr_floor_applied")}
                                         if lv.get("rr_floor_applied") else None)
                        d["levels_ok"] = bool(_lv_ok or _lv_note)
                        d["levels_mirrored"] = lv.get("mirrored") is True
                        if _lv_note:
                            d["levels_note"] = _lv_note.strip(" ·")
                            d["issue_note"] = ((d.get("issue_note") or "")
                                               + _lv_note)
                        _live_tape = getattr(self.feed, "mode", None) == "binance"
                        _open = bool(cyc.issued
                                     and cyc.tier != "watch-only"
                                     and not cyc.contract_tp_hit
                                     and not cyc.sl_observed
                                     and not cyc.pending_reset)
                        d["execution_eligible"] = bool(
                            cyc.execution_eligible and _open and _live_tape)
                        # PAPER eligibility is separate from order routing: a
                        # simulator tape must never send an order, but it CAN
                        # carry a live signal — calling that "no trade" made
                        # the whole board look broken on sim data.
                        d["paper_eligible"] = bool(_open)
                        # window/lock provenance: how old the anchor is and
                        # whether anything re-anchored it (must stay 0 — a
                        # silent mid-window re-lock is the churn report)
                        _row = self.locks.get((symbol, tf)) or {}
                        d["relocks"] = int(_row.get("relocks") or 0)
                        d["relocked_at"] = _row.get("relock_at")
                        d["lock_age_s"] = (
                            round(now - float(_row.get("locked_at") or now), 1)
                            if _row else None)
                        d["entry_age_s"] = (
                            round(now - float(cyc.cycle_start or now), 1))
                        # ── ANCHOR DRIFT (user report 2026-11-26) ────────
                        # "I got tp of 551 on BNB when market was on 556."
                        # The levels are anchored at the LOCKED entry, which
                        # is CORRECT (a target must be measured from where
                        # the position was opened, not from wherever price
                        # has since wandered). What was missing is the ONE
                        # number that makes that honest on screen: how far
                        # the live tape has moved away from the anchor since
                        # the lock. Without it the card prints an entry the
                        # market is no longer at, next to a target that looks
                        # wrong, and the user has to do the bps arithmetic
                        # themselves. Published here so every panel can say
                        # "anchor 439.43 · now 440.15 (+16 bps) · 4 m ago".
                        try:
                            _live_p = float(self.feed.prices().get(symbol)
                                            or 0.0)
                            _ent_p = float(lv.get("entry") or 0.0)
                            if _live_p > 0 and _ent_p > 0:
                                d["live_price"] = round(_live_p, 6)
                                d["anchor_drift_bps"] = round(
                                    (_live_p / _ent_p - 1.0) * 1e4, 2)
                                # has price already gone THROUGH the target
                                # we are still printing as "ahead"? That is
                                # stale-by-definition and must be labelled.
                                _tpv = float(lv.get("tp") or 0.0)
                                if _tpv > 0 and int(cyc.direction or 0):
                                    _past = ((_live_p - _tpv) * int(cyc.direction) >= 0)
                                    d["tp_already_reached"] = bool(
                                        _past and not cyc.contract_tp_hit)
                        except Exception:
                            pass
                        # why a high-P window is still watch-only: name the
                        # rule and the measured number, never a bare label
                        _ds = self.display_systems.get((symbol, tf)) or {}
                        if _ds.get("tape_veto"):
                            d["tape_veto"] = True
                            d["tape_z"] = _ds.get("tape_z")
                            d["tape_need_bps"] = _ds.get("tape_need_bps")
                        _rank = {"watch-only": 0, "quarter": 1,
                                 "half": 2, "full": 3}
                        _xr = float(d.get("score_floor_ratio") or 0.0)
                        _band = ("full" if _xr > 2.5 else "half" if _xr > 1.5
                                 else "quarter" if _xr > 1.0 else "watch-only")
                        # explain ONLY a real cut below the measured band —
                        # a window that is simply "half" needs no excuse
                        if str(d.get("tier")) == "watch-only" or \
                                _rank.get(str(d.get("tier")), 0) < _rank[_band]:
                            _why = []
                            if getattr(cyc, "pending_reset", False):
                                _why.append("window closed by a barrier — "
                                            "side invalidated, no new entry")
                            if getattr(cyc, "contract_tp_hit", False):
                                _why.append("target already touched in this window")
                            if d.get("ratio") is not None and \
                                    float(d["ratio"]) < 1.0:
                                _why.append(f"ratio {float(d['ratio']):.2f} < 1:1")
                            _nf = d.get("noise_floor_s") or 0
                            if _nf and abs(float(getattr(cyc, "cci", 0.0) or 0.0)) <= _nf:
                                _why.append(
                                    f"|score| {abs(float(cyc.cci or 0.0)):.4f} ≤ "
                                    f"noise floor {_nf:.4f} "
                                    f"({float(d.get('score_floor_ratio') or 0):.2f}× floor)")
                            _safe = int(getattr(cyc, "safe_events", 0) or 0)
                            if _safe:
                                _why.append(
                                    f"{_safe} SAFE event(s) cut size one step each "
                                    f"(SL tightened, confidence −0.15 per event)")
                            if _ds.get("tape_veto"):
                                _why.append(
                                    f"tape veto {float(_ds.get('tape_z') or 0):.2f}× "
                                    f"noise against the call")
                            _ev = [str(e.get("type")) for e in
                                   (getattr(cyc, "events", None) or [])]
                            if "below_ratio_floor" in _ev and \
                                    not any("ratio" in x for x in _why):
                                _why.append("ratio below 1:1 at lock "
                                            "(below_ratio_floor event)")
                            if "sl_crossed" in _ev:
                                _why.append("stop was crossed in this window")
                            if not _why:
                                # never a bare label: print the window's own
                                # event trace so the cut is auditable
                                _why.append("tier cut at lock by "
                                            + (", ".join(_ev[-3:]) or
                                               "an unrecorded rule"))
                            d["tier_why"] = "; ".join(_why)
                        d["execution_blocked_by"] = (
                            [] if d["execution_eligible"] else
                            (["simulated tape — orders are not routed"]
                             if _open and not _live_tape else
                             ["no open call: " + ", ".join(
                                 k for k, v in (
                                     ("side not issued", not cyc.issued),
                                     ("size tier watch-only",
                                      cyc.tier == "watch-only"),
                                     ("target already touched",
                                      cyc.contract_tp_hit),
                                     ("stop already observed", cyc.sl_observed),
                                     ("cycle resetting", cyc.pending_reset),
                                 ) if v)]))
                        d["sl_provisional"] = bool(
                            (cyc.risk or {}).get("provisional"))
                    else:
                        d["tp"] = d["sl"] = d["entry"] = None
                        d["levels_ok"] = False
                        d["execution_eligible"] = False
                        if cyc.level_contract:
                            d["level_contract"] = _levels.unavailable_contract(
                                symbol=symbol, timeframe=tf, cycle=cyc.cycle_id,
                                reason="cycle lock/level contract failed validation",
                                lock_id=cyc.lock_id, side=cyc.direction,
                                revision=cyc.level_revision)
                # SYSTEM SELECTOR (display): in NEW mode the target shown is
                # the horizon forecast. The level CONTRACT keeps the measured
                # ladder (settlement/execution read that), so this is the one
                # place where the displayed TP may legitimately differ.
                if (d and cyc is not None
                        and str(self.agents.settings.get("system_mode", "old"))
                        == "new"):
                    _fc = (cyc.forecast or {}).get("price")
                    if _fc:
                        _entry_d = (float((cyc.level_contract or {})
                                          .get("entry") or 0.0) or price)
                        d["tp"] = float(_fc)
                        d["tp_dist"] = (round(abs(float(_fc) - _entry_d), 6)
                                        if _entry_d else None)
                        if d.get("sl") is not None and _entry_d:
                            _sl_d = abs(_entry_d - float(d["sl"]))
                            d["ratio"] = (round(abs(float(_fc) - _entry_d)
                                                / _sl_d, 6) if _sl_d else None)
                        if d.get("targets"):
                            d["targets"][-1]["level"] = float(_fc)
                        d["system_mode"] = "new"
                        d["tp_source"] = "forecast price at horizon end"
                if d and price > 0:
                    bs = self.branch_stats_cached(symbol, tf, d, price, now)
                    if bs:
                        d["branches"] = bs
                        # remember this window's forecast ONCE (first
                        # snapshot of the cycle) for the settle audit
                        _pk = (symbol, tf)
                        _cs = d.get("cycle_start")
                        _praw = bs.get("p_target")
                        _d = self.calibration_delta()
                        _pcal = (max(0.01, min(0.99, _praw + _d))
                                 if _praw is not None else None)
                        bs["p_target_raw"] = _praw
                        bs["p_target_cal"] = (round(_pcal, 4)
                                              if _pcal is not None else None)
                        bs["cal_delta"] = _d
                        if _cs and self._prob_pred.get(_pk, {}) \
                                .get("cycle_start") != _cs:
                            self._prob_pred[_pk] = {
                                "cycle_start": _cs,
                                "p_target": _praw,
                                "p_target_cal": _pcal,
                                "method": bs.get("method")}
                if d:
                    _lr = self.last_result.get((symbol, tf)) or {}
                    _adv2 = _lr.get("advanced") or \
                        getattr(self, "_adv_last", {}).get((symbol, tf))
                    if _adv2:
                        _conf = _lr.get("conformal")
                        if not _conf:
                            _cqn = len(list(self.hist.ring(
                                self.hist.tf_key(symbol, tf, "conf_q"),
                                400)))
                            _conf = {"q_bps": None, "n": _cqn,
                                     "state": ("calibrating — needs "
                                               "settled windows")}
                        d["advanced"] = {
                            "vote": _adv2.get("vote"),
                            "soft_gate": _adv2.get("soft_gate"),
                            "rqa_mult": _adv2.get("rqa_mult"),
                            "size_mult": _adv2.get("size_mult"),
                            "modules": {
                                k: {"vote": (v or {}).get("vote"),
                                    "gate": (v or {}).get("gate"),
                                    "why": ((v or {}).get("why")
                                            if not (v or {}).get("vote")
                                            else None),
                                    "z": (v or {}).get("z"),
                                    "half_life_bars": (v or {})
                                    .get("half_life_bars")}
                                for k, v in (_adv2.get("modules")
                                             or {}).items()},
                            "har_rv": _adv2.get("har_rv"),
                            "evt": _adv2.get("evt"),
                            "rmt_mode_share": (_adv2.get("rmt") or {})
                            .get("mode_share"),
                            "conformal": _conf,
                        }
                # §1.3/§2/§4 — the decoupled context every TF card carries:
                # 5-state regime (classify_tape on THIS asset's tape —
                # never BTC's), the isolated Θ(a,τ) cell parameters, and
                # the metacognition CONFIRM/SHADOW status for this cell
                if d is not None:
                    try:
                        from .regime import classify_tape
                        _rtf = classify_tape(
                            list(st.closes or []),
                            highs=list(st.highs or []) or None,
                            lows=list(st.lows or []) or None,
                            volumes=list(st.volumes or []) or None)
                        d["regime"] = _rtf
                    except Exception:
                        log.exception("regime classify failed for %s|%s",
                                      symbol, tf)
                        d["regime"] = None
                    try:
                        from .cell_registry import cell_params
                        _cp = cell_params(symbol, tf)
                        d["cell"] = _cp.to_dict()
                    except Exception:
                        d["cell"] = None
                    try:
                        from .metacognition import MC as _MC
                        d["metacog_status"] = _MC.status(symbol, tf)
                    except Exception:
                        d["metacog_status"] = "SHADOW"
                tfs[tf] = d
            # PRICE PROVENANCE — the chart, the rail and the analysis all
            # quote a different moment in time; saying which one is which is
            # the difference between "chart and analysis don't match" and a
            # readable board. `price_asof` is when the feed produced this
            # price; `chart` is the last closed/live candle the chart draws.
            _last_t = (float(st.times[-1]) if st.times else None)
            _last_c = (float(st.closes[-1]) if st.closes else None)
            assets[symbol] = {
                "price": price,
                "price_asof": (float(getattr(st, "last_tick", 0) or 0) or None),
                "feed_age_s": (round(now - float(st.last_tick), 1)
                               if getattr(st, "last_tick", 0) else None),
                "chart": {
                    "last_bar_t": _last_t,
                    "last_bar_close": _last_c,
                    "age_s": (round(now - _last_t, 1)
                              if _last_t else None),
                    "delta_bps": (round((price / _last_c - 1.0) * 1e4, 2)
                                  if _last_c else None),
                    "note": ("last candle the chart draws vs the live price "
                             "printed by the rail — same tape, different "
                             "instant"),
                },
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
            "state_version": self._state_version,
            "snapshot_ts": now,
            "phase": self.phase,
            "system_mode": str(
                self.agents.settings.get("system_mode", "old")),
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
            "sleeves": self.sleeve_block(now=now),
            "prob_audit": self.prob_audit_summary(),
            # §1.4 / §4 / §5 — the decoupled subsystems' telemetry:
            # one shared snapshot clock with the 5 named countdowns, the
            # metacognition book (weights + CONFIRM/SHADOW board), and the
            # four-agent committee's last vote
            "sync": _sync_block(),
            "metacog": _metacog_block(),
            "committee": _committee_block(),
            # the certified hold policy's 16 cells: tier + win rate +
            # Wilson confidence + ANALYSE countdowns + era position
            "cells": self.hold_cells_block(now),
            "bgn": {
                sym: {
                    tf: {t2: round(v, 4) for t2, v in row.items()}
                    for tf, row in (mat or {}).items()}
                for sym, mat in (self.bgn.last_matrix or {}).items()
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
            # LIVE brain state, straight off the component every snapshot.
            # The chip used to read only the per-(asset,tf) display copy,
            # which is rebuilt at GRC — so a window that had not been
            # re-analysed yet painted "brain offline" over a healthy
            # connectome (the "brain disconnects at random" report).
            # (This dict used to declare "brain" twice — the AUTHORITATIVE
            # block above was silently overwritten by this one, and the
            # connectome's own vote was dropped. One entry now carries
            # both; the duplicate key is gone.)
            "brain": {**self._brain_live(),
                      "vote": getattr(self.brains, "last_vote", None)},
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
            # pass the CURRENT frozen subsystem readings so the panel can
            # show the predictability gauge (1 − H₂) and this window's
            # P(up); the payload was called with no inputs, so the gauge
            # never appeared even after the 40-window record was complete.
            "meta": self.meta.payload(self._current_meta_inputs()),
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
                    sym: {
                        tf: [round(x, 4) for x in
                             self.learning.brain_influence(sym, tf)]
                        for tf in TF_ORDER}
                    for sym in SYMBOL_ORDER
                },
                "formula_accuracy": {
                    str(f): (round(a, 4) if (a := self.learning.formula_accuracy(f)) is not None else None)
                    for f in list(WEIGHT_IDS)[:12]
                },
                "probation_count": len(self.learning.probation),
            },
            "agents": self.agents.status(),
            # cosmetic settings the dashboard needs live (haptics off must
            # actually stop the vibration). NEVER the secret key fields.
            "settings": {
                "haptics": self.agents.settings.get("haptics", True),
                "sound": self.agents.settings.get("sound", True),
                "theme": self.agents.settings.get("theme", "dark"),
                # "" when the last write reached disk. A non-empty string means
                # a setting is live in memory only and WILL be lost on restart,
                # so the dashboard must say so instead of showing it as saved.
                "save_error": getattr(self.agents, "last_save_error", ""),
            },
            # exceptions swallowed by best-effort paths (learners, ring
            # buffers). Zero is the healthy answer; a rising count means a
            # panel is showing stale numbers.
            "silence": _silence_report(),
            "feed": self.feed.stats(),
            "ledger": self.ledger.api_payload(SYMBOL_ORDER, TF_ORDER),
            "locks": [dict(v) for v in self.locks.values()],
            "hindsight": self.hindsight(),
            "watchdog": {
                "seconds_since_window": round(time.time() - self.last_window_ts, 1),
                "stalled": (time.time() - self.last_window_ts) >
                            (TIMEFRAMES["tf4"]["minutes"] * 60 + 45),
            },
            # SELF-EXAMINATION OF THE SEALED RECORD. The app checks its own
            # journal on every snapshot and publishes what it finds, so an
            # incoherent row becomes a visible in-app error instead of a
            # number the user has to notice for us. Cost is a linear scan of
            # at most KEEP (400) rows — sub-millisecond.
            "prediction_audit": self._prediction_audit_cached(),
            "agent_signal": self.last_agent_signal or {},
        }

    async def _broadcast_full(self, reason: str) -> None:
        # assembling the snapshot is hundreds of ms of CPU; awaiting an
        # asyncio.Event around a synchronous build (the old shape) still
        # blocked every other coroutine, including /api/health
        snap = await run_pass(self.build_snapshot)
        snap["reason"] = reason
        # state_version/snapshot_ts are stamped inside build_snapshot —
        # the UI drops any out-of-order snapshot (§1.1 atomic discard)
        self.last_snapshot = snap
        # HAND THE SAME PAYLOAD TO THE REST CACHE. Without this, every GRC
        # built the snapshot twice: once here, then again on the next
        # /api/snapshot poll ~1 s later (the broadcast never populated the
        # cache). Under the old 0.75 s TTL those two builds plus a poll could
        # chain together and hold the CPU. The snapshot is the authoritative
        # representation of this moment, so the poll can serve it verbatim.
        try:
            from . import main as _m
            _m.prime_snapshot_cache(snap)
        except Exception:
            pass
        await self.emit(snap)
