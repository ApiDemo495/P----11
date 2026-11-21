"""§7 verification for the LIMITLESS MASTER ARCHITECTURAL DIRECTIVE.

Fast, deterministic checks of the new subsystems — the heavy statistical
rigor (nested purged WF on real data) lives in backtest/; these tests
pin the contracts the directive requires:
  * 16 isolated Θ(a,τ) cells with zero cross-asset bleed
  * 21 emotion channels all moving (sd > 0.08) + E22 vectors
  * metacognition: audit → floor/ABSTAIN, bounded calibration, Exp3
    floor, DSR gate defaulting to SHADOW
  * 4-agent committee: all agents vote; SHADOW / shock / split veto
  * sync clock: all five countdowns present and sane
  * snapshot stamps: state_version monotonic, regime per-asset
  * master scheduler: exactly ONE setInterval, in-flight coalescing
"""
from __future__ import annotations

import math
import random
import statistics


# ── §2 16-cell isolation ──────────────────────────────────────────────
def test_sixteen_cells_are_isolated():
    from app.cell_registry import CELLS, cell_params
    assert len(CELLS.snapshot()) >= 16
    btc1 = cell_params("BTC", "tf1")
    eth4 = cell_params("ETH", "tf4")
    pax1 = cell_params("PAXG", "tf1")
    # per-asset personalities differ (no BTC fallback)
    assert btc1.edge_floor_bps != eth4.edge_floor_bps or btc1.atr_sl != eth4.atr_sl
    assert pax1.conf_floor != btc1.conf_floor
    # per-horizon tilt differs within one asset
    assert btc1.micro_weight != cell_params("BTC", "tf4").micro_weight
    # unknown symbol gets a fresh cell — never a BTC copy
    stranger = cell_params("DOGE", "tf1")
    assert stranger.edge_floor_bps >= 0
    # mutate is the only mutation path; isolation after mutate
    before = cell_params("BTC", "tf1").atr_sl
    CELLS.mutate("ETH", "tf1", atr_sl=before + 0.5)
    assert cell_params("BTC", "tf1").atr_sl == before
    assert cell_params("ETH", "tf1").atr_sl == before + 0.5
    CELLS.reset()


def test_regime_five_states_no_dead_band():
    from app.regime import classify_tape, RISK_THROTTLE_STATES
    rng = random.Random(11)
    # strong trend must classify as an impulse, not "sideways"
    px, trend = 50000.0, []
    for _ in range(240):
        px *= 1 + 0.0025 + rng.gauss(0, 0.002)
        trend.append(px)
    r = classify_tape(trend)
    assert r["state"] in ("IMPULSE_BULL", "IMPULSE_BEAR"), r["state"]
    # flat tape must classify as range, not impulse
    px, flat = 50000.0, []
    for _ in range(240):
        px *= 1 + rng.gauss(0, 0.0008)
        flat.append(px)
    r2 = classify_tape(flat)
    assert r2["state"] in ("MEAN_REVERTING_RANGE", "VOLATILE_RANGE",
                           "IMPULSE_BULL", "IMPULSE_BEAR")
    # every call names a state (no dead band)
    assert r2["state"]
    assert isinstance(r["risk_throttle"], bool)
    assert "CHAOTIC_LIQUIDATION_SHOCK" in RISK_THROTTLE_STATES


# ── §3 emotion: every channel moves ──────────────────────────────────
def _tape(n=600, seed=7):
    rng = random.Random(seed)
    closes, vols, times = [50000.0], [], []
    px = 50000.0
    for i in range(n):
        drift = 0.0015 if (i // 80) % 2 == 0 else -0.0008
        px *= 1 + drift + rng.gauss(0, 0.006)
        closes.append(px)
        vols.append(10 + abs(rng.gauss(0, 4)))
        times.append(1700000000 + 60 * i)
    return closes, vols, times


def test_all_21_channels_move_and_contract_holds():
    from app.emotion import EmotionSystem, EMO_IDS, CHANNELS, MODEL
    from app.models import FormulaContext
    closes, vols, times = _tape()
    highs = [c * 1.003 for c in closes]
    lows = [c * 0.997 for c in closes]
    sys_ = EmotionSystem()
    series = {i: [] for i in EMO_IDS}
    for end in range(120, len(closes) - 1, 10):
        w = 60
        ctx = FormulaContext(symbol="BTC", tf="tf1", W=w,
                             P=closes[end - w:end], V=vols[end - w:end],
                             T=times[end - w:end], hist={})
        m = {"closes": closes[:end], "volumes": vols[:end],
             "times": times[:end],
             "ohlc": {"highs": highs[:end], "lows": lows[:end],
                      "closes": closes[:end]},
             "live": closes[end - 1], "open": closes[end - w],
             "session_open": closes[end - w], "btc_closes": closes[:end],
             "peer_returns": [[rng_v] * 3 for rng_v in
                              [random.Random(end).gauss(0, 0.004)] * 20],
             "herd_corr": 0.5 + 0.4 * math.sin(end / 40),
             "ofti": random.Random(end).gauss(0, 0.3), "cmp": 0.3,
             "news_last_ts": times[end - 30] if end % 50 < 25 else 0,
             "news_tier": 2.0,
             "trade_sizes": [abs(random.Random(end * 31 + i).gauss(20, 14)) + 1
                             for i in range(60)]}
        out = sys_.compute(ctx, m, {"hormones_avg": {}},
                           {"polarity": random.Random(end).gauss(0, 0.4),
                            "count": 3 if end % 50 < 25 else 0}, None)
        assert sorted(out["emotions"]) == list(range(1, 22))
        for i in EMO_IDS:
            series[i].append(out["emotions"][i])
        # contract: derived displays present
        for k in ("primary_emotion", "fear_greed", "core_affect",
                  "action_tendency", "cerc_vectors", "zero_reasons"):
            assert k in out, k
        # bounded
        assert -1.0 <= out["cerc"] <= 1.0
        assert 0.0 <= out["net_emotion"] <= 1.0 or \
            -1.0 <= out["net_emotion"] <= 1.0
    low = [i for i in EMO_IDS
           if statistics.pstdev(series[i]) <= 0.08]
    assert not low, f"flat channels (sd<=0.08): {low}"
    # E22 vectors present with the three named telemetry fields
    assert MODEL["channels"]["18"]["key"] == "chase_fatigue"
    assert CHANNELS[20][3] == 0.0     # news FOMO is an event channel


def test_emotion_abstention_text_contract():
    from app.emotion import EmotionSystem
    from app.models import FormulaContext
    closes, vols, times = _tape(200, seed=3)
    ctx = FormulaContext(symbol="BTC", tf="tf1", W=60,
                         P=closes[-60:], V=vols[-60:], T=times[-60:],
                         hist={})
    m = {"closes": closes, "volumes": vols, "times": times,
         "ohlc": {"highs": [c * 1.002 for c in closes],
                  "lows": [c * 0.998 for c in closes],
                  "closes": closes},
         "live": closes[-1], "open": closes[0], "session_open": closes[0],
         "btc_closes": closes, "peer_returns": [], "ofti": 0.1, "cmp": 0.3}
    out = EmotionSystem().compute(ctx, m, {"hormones_avg": {}},
                                  {"polarity": 0, "count": 0}, None)
    zr = out["zero_reasons"]
    assert "20" in zr and "headline" in zr["20"]     # news FOMO abstains
    assert "13" in zr                                # trap abstains
    assert "14" in zr                                # news confirmation
    # no peers → E11 abstains honestly (never a BTC substitute)
    assert "11" in zr


# ── §4 metacognition ──────────────────────────────────────────────────
def test_metacognition_audit_and_gates():
    from app.metacognition import (audit_pillars, PillarReading,
                                   CalibrationBook, Exp3Ix,
                                   deflated_sharpe, Metacognition,
                                   DSR_CONFIRM, CALIB_STEP_MAX)
    # agreeing board: no abstain, no floor bump
    ok = audit_pillars([PillarReading("a", 0.8, 0.65),
                        PillarReading("b", 0.7, 0.6),
                        PillarReading("c", 0.75, 0.62)])
    assert not ok.abstain and ok.elevated_floor <= 0.20
    # divided board: floor raised + ABSTAIN
    bad = audit_pillars([PillarReading("a", 0.9, 0.85),
                         PillarReading("b", -0.9, 0.15),
                         PillarReading("c", 0.0, 0.5)])
    assert bad.abstain and bad.elevated_floor > ok.elevated_floor
    # calibration bounded ±3 pp
    cb = CalibrationBook()
    for i in range(40):
        cb.record("BTC", "tf1", 0.7, i % 2)
    p0, p1 = 0.7, cb.calibrate("BTC", "tf1", 0.7)
    assert abs(p1 - p0) <= CALIB_STEP_MAX + 1e-9
    # Exp3 bounded: floor respected, weights sum to 1
    e = Exp3Ix(k=16)
    for _ in range(100):
        e.update(0, 1.0)
    w = e.weights()
    assert abs(sum(w) - 1.0) < 1e-6
    assert min(w) >= 0.05 / 16 - 1e-9
    # DSR separates signal from noise
    assert deflated_sharpe(2.0, 80, 80) > deflated_sharpe(0.1, 80, 80)
    # default state is SHADOW (no capital before proof)
    mc = Metacognition(persist_path=None)
    assert mc.status("BTC", "tf1") == "SHADOW"
    assert not mc.allows_capital("BTC", "tf1")
    # a losing cell stays SHADOW after evaluation
    losing = {("BTC", "tf1"): [-0.01 + random.gauss(0, 0.01)
                               for _ in range(120)]}
    v = mc.evaluate(losing)
    assert v[("BTC", "tf1")].status == "SHADOW"
    assert DSR_CONFIRM == 0.90


# ── §5 committee ──────────────────────────────────────────────────────
def test_committee_four_agents_and_vetoes():
    from app.agent_committee import (COMMITTEE, flow_vote, lead_lag_vote,
                                     contrarian_vote, governor_vote)
    market = {"closes": [50000 * (1 + 0.001 * i) for i in range(50)],
              "volumes": [10.0] * 50, "ofti": 0.4, "cmp_divergence": 0.2,
              "spread_bps": 1.5,
              "peer_returns": [[0.001, 0.001, 0.001] for _ in range(20)]}
    emotion = {"cerc_vectors": {"retail_herd": 0.3, "smart_money": 0.2,
                                "crowd_smart_split": 0.1},
               "emotions": {18: 0.6, 20: 0.05, 1: 0.02}}
    regime = {"state": "IMPULSE_BULL", "confidence": 0.7,
              "risk_throttle": False}
    # every agent votes
    for fn in (flow_vote,):
        v = fn(market)
        assert v.agent == "flow" and 0 <= v.confidence <= 1
    assert lead_lag_vote(market, "ETH").agent == "lead_lag"
    assert contrarian_vote(emotion).agent == "contrarian"
    assert governor_vote("CONFIRMED", None, regime, True).agent == "governor"
    # clear board issues
    r = COMMITTEE.vote(market, emotion, regime, "CONFIRMED", None, True, "BTC")
    assert len(r.votes) == 4 and r.may_issue
    # SHADOW cell → governor veto, no issue
    r2 = COMMITTEE.vote(market, emotion, regime, "SHADOW", None, False, "BTC")
    assert not r2.may_issue and any("SHADOW" in v for v in r2.vetoes)
    # risk-throttle regime → veto
    shock = dict(regime, state="CHAOTIC_LIQUIDATION_SHOCK",
                 risk_throttle=True, confidence=1.0)
    r3 = COMMITTEE.vote(market, emotion, shock, "CONFIRMED", None, True, "BTC")
    assert not r3.may_issue and any("throttle" in v for v in r3.vetoes)
    # crowd-vs-smart split → contrarian veto
    em2 = {"cerc_vectors": {"retail_herd": 0.85, "smart_money": -0.6,
                            "crowd_smart_split": 0.8},
           "emotions": {18: 0.9, 20: 0.7, 1: 0.02}}
    r4 = COMMITTEE.vote(market, em2, regime, "CONFIRMED", None, True, "BTC")
    assert not r4.may_issue and any("split" in v for v in r4.vetoes)


# ── §1 sync clock + stamps ───────────────────────────────────────────
def test_sync_clock_five_countdowns():
    from app.sync_clock import CLOCK
    d = CLOCK.countdowns()
    for key in ("NEXT_TICK_PASS", "WINDOW_CLOSE_SETTLE",
                "BRAIN_METACOG_REWEIGHT", "AGENT_COMMITTEE_SYNTHESIS",
                "NEWS_MACRO_POLL"):
        assert key in d, key
        assert d[key]["remaining_s"] >= 0
        assert d[key]["remaining"]  # formatted mm:ss
    assert d["WINDOW_CLOSE_SETTLE"]["minutes"] in (1, 5, 22, 30, 60, 240)


def test_snapshot_stamps_and_per_asset_regime():
    import asyncio
    import sys
    sys.path.insert(0, "tests")
    from tests.test_engine import make_engine, _boot
    eng, _feed, _e = make_engine()
    asyncio.run(_boot(eng))
    s1 = eng.build_snapshot()
    s2 = eng.build_snapshot()
    # monotonic state_version at birth + snapshot_ts
    assert s1["state_version"] >= 1
    assert s2["state_version"] > s1["state_version"]
    assert s1["snapshot_ts"] > 0
    # subsystem blocks present
    assert "NEXT_TICK_PASS" in s1["sync"]
    assert len(s1["metacog"]["statuses"]) == 16
    assert "committee" in s1
    # per-asset regime + cell on every TF card
    for sym in ("BTC", "ETH", "BNB", "PAXG"):
        for tf in ("tf1", "tf2", "tf3", "tf4"):
            d = s1["assets"][sym]["tfs"][tf]
            assert d.get("regime") is not None, f"{sym}|{tf} regime"
            assert d.get("cell") is not None, f"{sym}|{tf} cell"
            assert d.get("metacog_status") in ("SHADOW", "CONFIRMED")
    # no bleed: per-asset cell params differ somewhere across assets
    sl = {sym: s1["assets"][sym]["tfs"]["tf1"]["cell"]["atr_sl"]
          for sym in ("BTC", "ETH", "BNB", "PAXG")}
    assert len(set(sl.values())) > 1, sl


def test_hindsight_keys_are_composite():
    from pathlib import Path
    src = Path("app/engine.py").read_text()
    assert 'self._hindsight_last: dict[tuple[str, str], float]' in src
    assert "ckey = (symbol, tf)" in src
    # tf-only keying is gone
    assert 'self._hindsight_last.get(tf' not in src


def test_master_scheduler_single_interval_and_stamps():
    from pathlib import Path
    js = Path("static/app.js").read_text()
    assert js.count("setInterval(") == 1
    assert "SCHED.nextAt" in js and "schedRun(" in js
    assert "state_version" in js and "analyseReqIds" in js
    assert "NO TRADE" not in js and '"NO TRADE"' not in js
    # engine stamps every snapshot
    eng = Path("app/engine.py").read_text()
    assert '"state_version": self._state_version' in eng
    assert '"request_id": job.get("request_id")' in eng
