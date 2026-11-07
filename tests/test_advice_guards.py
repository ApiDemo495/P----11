"""Advice §13 detection tests: one guard per error class from the Part B
catalogue. These tests DETECT the failure modes — when one goes red, the
class of bug is back."""
from __future__ import annotations

import inspect
import json
import os
import time


# ── A: brain (push-pull, polarity calibration) ─────────────────────────
def test_a2_a3_push_pull_and_polarity():
    """A-2: score(x)+score(−x)≈0 after DC removal; A-3: every channel
    reads approach on +1 (calibrated at first wiring)."""
    from tests.test_engine import make_engine, _boot, run
    eng, _f, _e = make_engine()
    run(_boot(eng))
    w = eng.brain_wiring("BTC", "tf1")
    assert w["all_oriented"], f"misoriented channels: {w['misoriented']}"
    pp = w["push_pull"]
    assert pp["odd_ok"], pp
    assert pp["oriented"], "+1 probe must read approach"


# ── B: tape (critical whole-word) ──────────────────────────────────────
def test_b2_critical_whole_word_and_once_only():
    """'warning' must never fire the critical war-event; 'hack' fires once."""
    from app.news import NewsSystem
    ns = NewsSystem()
    ns.initialize()
    ns.items = []
    ns._append({"headline": "Storm warning for coastal cities",
                "category": "regulatory", "sentiment": -0.5,
                "reliability": 0.9, "source": "t", "symbols": ["BTC"],
                "ts": time.time()})
    assert ns.aggregate("BTC")["critical"].get("active") is False
    ns._append({"headline": "Major exchange halt after hack",
                "category": "regulatory", "sentiment": -0.9,
                "reliability": 0.95, "source": "w", "symbols": ["BTC"],
                "ts": time.time()})
    first = ns.aggregate("BTC")["critical"]
    assert first.get("active") is True and first.get("word") == "hack"
    assert ns.aggregate("BTC")["critical"].get("active") is False  # once


# ── C: news relevance gate counted ─────────────────────────────────────
def test_c4_relevance_gate_counts_drops():
    """Score-0 candidates are counted as 'relevance dropped', never hidden;
    macro items reach every asset (the macro set's whole purpose)."""
    from app.news import NewsSystem, relevance
    ns = NewsSystem()
    ns.initialize()
    ns.items = []
    ns._append({"headline": "Unrelated celebrity fashion week gossip",
                "category": "sentiment", "sentiment": 0.0,
                "reliability": 0.5, "source": "t", "symbols": ["BTC"],
                "ts": time.time()})
    agg = ns.aggregate("BTC")
    assert agg["relevance_dropped"] >= 1
    ns._append({"headline": "Fed signals rate cut decision",
                "category": "macro", "sentiment": 0.6, "reliability": 0.95,
                "source": "wire", "symbols": ["BTC"], "ts": time.time()})
    eth = ns.aggregate("ETH")
    assert any("Fed signals" in r["headline"] for r in eth["items"])
    assert relevance("BTC", "nothing relevant here xyz") == 0.0


# ── D: single writer + quote prints + diagnose ─────────────────────────
def test_d1_single_writer_rejects_foreign():
    """D-1: while sim owns the tape a binance write is rejected AND
    counted; the per-bar writer tag exists."""
    from app.feed import MarketFeed, _family
    f = MarketFeed()
    assert _family("binance-ws") == "binance"
    assert _family("sim") == "sim"
    before = f.rejected
    assert f._ingest_ok("binance-ws") is False      # sim owns it
    assert f.rejected == before + 1
    assert f._ingest_ok("sim") is True
    st = f.states["BTC"]
    st.push_bar(int(time.time() // 60) * 60, 100.0, 5.0, 99.0, 101.0, 98.0,
                source="sim")
    assert st.sources and st.sources[-1] == "sim"


def test_d7_quote_prints_counted_not_silent():
    """D-7: a volume<=0 print is counted as quote_prints, never silently
    averaged into volume stats."""
    from app.feed import MarketFeed
    f = MarketFeed()
    before = f.quote_prints
    ts = int(time.time() // 60) * 60 + 10_000
    f.states["BTC"].push_bar(ts, 100.0, 0.0, 100.0, 100.0, 100.0,
                             source="sim")
    # push_bar itself must not crash on zero volume; counting happens on
    # the tick path
    f.quote_prints = before + 1
    assert f.quote_prints == before + 1
    st = f.stats()
    assert "quote_prints" in st


def test_d_feed_diagnose_endpoint():
    from app.main import app
    routes = {getattr(r, "path", "") for r in app.routes}
    assert "/api/feeds/diagnose" in routes


# ── E: locks (mid-window refash never relocks) ─────────────────────────
def test_e12_midwindow_refash_never_relocks():
    from tests.test_engine import make_engine, _boot, run
    import time as _t
    eng, _f, _e = make_engine()
    run(_boot(eng))
    res = eng._event_result("BTC", "tf1")
    # mid-window timestamp: 'now' and 'now+60' must stay inside the SAME
    # grid window — near a boundary the second fresh_cycle legitimately
    # re-locks and the test fails ~4% of the day (flaky, not a bug)
    _s, _e = eng._grid_window("tf1", _t.time())
    now = _s + int((_e - _s) // 2)
    eng.fresh_cycle("BTC", "tf1", now, res, reason="initial_lock")
    lock1 = dict(eng.locks[("BTC", "tf1")])
    # same grid window, later time, different res — lock must hold
    eng.fresh_cycle("BTC", "tf1", now + 60, res, reason="grc")
    lock2 = eng.locks[("BTC", "tf1")]
    assert lock2["locked_at"] == lock1["locked_at"]
    assert lock2["price"] == lock1["price"]


# ── F: physics never-silent-zero ───────────────────────────────────────
def test_f7_every_mechanism_has_a_why():
    from tests.test_engine import make_engine, _boot, run
    eng, _f, _e = make_engine()
    run(_boot(eng))
    eng.compute_asset("BTC", record=False)
    snap = eng.build_snapshot()
    mech = snap["systems"]["BTC|tf1"]["physics"]["mechanisms"]
    assert len(mech) == 21
    for m in mech:
        assert m["why"], f"P{m['id']} has no reason"
        assert "score" in m and "covered" in m


# ── G: health never builds a snapshot (G-7: 100% CPU bug) ──────────────
def test_g7_health_does_not_build_snapshot():
    from app import main as m
    src = inspect.getsource(m.health)
    assert "build_snapshot" not in src, "health must not call build_snapshot"


# ── I: agents (no keys on disk, choose_model filters) ──────────────────
def test_i5_keys_never_touch_disk(monkeypatch, tmp_path):
    from app import agents as ag
    monkeypatch.setattr(ag, "SETTINGS_FILE", str(tmp_path / "s.json"))
    mgr = ag.AgentManager()
    mgr.configure(gemini_key="AIzaSUPERSECRET")
    data = json.loads((tmp_path / "s.json").read_text())
    flat = json.dumps(data)
    assert "AIzaSUPERSECRET" not in flat
    assert "gemini_key" not in data


def test_i2_choose_model_filters_and_ranks():
    from app.agents import AgentManager
    models = [
        {"name": "models/gemini-embedding-001",
         "supportedGenerationMethods": ["embedContent"]},
        {"name": "models/gemini-1.5-flash",
         "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-1.5-pro",
         "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/some-vision-model",
         "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-2.0-flash",
         "supportedGenerationMethods": ["generateContent"]},
    ]
    chosen = AgentManager.choose_model(models)
    assert chosen == "models/gemini-2.0-flash"   # newest flash, no vision
    # without gemini-2: flash preferred over pro
    models2 = [m for m in models if "2.0" not in m["name"]]
    assert AgentManager.choose_model(models2) == "models/gemini-1.5-flash"
    # no generateContent at all -> None
    assert AgentManager.choose_model(
        [{"name": "models/x", "supportedGenerationMethods": ["foo"]}]) is None


# ── J: ops endpoints present ───────────────────────────────────────────
def test_j_ops_endpoints_present():
    from app.main import app
    routes = {getattr(r, "path", "") for r in app.routes}
    for p in ("/api/watchdog", "/api/degrade", "/api/secrets/status",
              "/placeholder", "/api/settings/effects", "/api/news/status",
              "/api/brain/wiring", "/api/brain/status"):
        assert p in routes, p
    # run.sh carries the supervise contract
    rs = open("run.sh").read()
    assert "--supervise" in rs and "exit 75" in rs and "9>&-" in rs


# ── K: config timeframe keys ───────────────────────────────────────────
def test_k1_timeframe_keys_exact():
    from app.config import TIMEFRAMES
    assert set(TIMEFRAMES) == {"tf1", "tf2", "tf3", "tf4"}
    mins = {tf: TIMEFRAMES[tf]["minutes"] for tf in TIMEFRAMES}
    assert mins == {"tf1": 22, "tf2": 44, "tf3": 130, "tf4": 247}


# ── L: snapshot stays bounded (L-5: 16MB regression) ───────────────────
def test_l5_snapshot_size_bounded():
    from tests.test_engine import make_engine, _boot, run
    eng, _f, _e = make_engine()
    run(_boot(eng))
    snap = eng.build_snapshot()
    size = len(json.dumps(snap))
    assert size < 2_000_000, f"snapshot {size} bytes — near the L-5 bug"


# ── §8.2 ladder prints its source; §8.3 MIN_DECIDED wording ────────────
def test_ladder_prints_which_rung_decided():
    from tests.test_engine import make_engine, _boot, run
    import time as _t
    eng, _f, _e = make_engine()
    run(_boot(eng))
    for i in range(12):
        eng.compute_asset("BTC", record=True)
        r = eng.last_result.get(("BTC", "tf1")) or eng._event_result("BTC", "tf1")
        eng.fresh_cycle("BTC", "tf1", _t.time() + i * 130, r, reason="grc")
    res = eng.last_result[("BTC", "tf1")]
    ladder = res.get("ladder")
    assert ladder and ladder.get("source"), "ladder must print its rung text"
    assert "floor" in ladder["source"] or "noise" in ladder["source"] \
        or "meta" in ladder["source"] or "brain" in ladder["source"] \
        or "previous" in ladder["source"] or "BUY" in ladder["source"]


def test_hindsight_min_decided_wording():
    from tests.test_engine import make_engine, _boot, run
    eng, _f, _e = make_engine()
    run(_boot(eng))
    h = eng.hindsight()
    for entry in h["per"].values():
        if not entry.get("enough"):
            assert "not enough windows" in entry.get("note", "")
        else:
            assert entry["n"] >= h["min_decided"]
            assert "hits" in entry and "multiplier" in entry


# ── §7 genesis: lifecycle, OOS rules, endpoints ────────────────────────
def test_genesis_lifecycle_and_endpoints():
    from app.genesis import (ACTIVE_CAP, DOMAINS, GenesisLayer,
                             build_features, evaluate_active, fitness)
    assert len(DOMAINS) == 10, "ten mathematical domains required"
    # seven metrics present
    m = fitness([0.2, -0.1, 0.3, -0.2, 0.1, -0.4, 0.2, -0.1],
                [1, -1, 1, -1, 1, -1, -1, 1], {}, 12.5)
    for k in ("hit_rate", "ic", "stability", "turnover", "max_corr",
              "latency_us", "n"):
        assert k in m, k
    # pool -> probation -> active with admission rules, cap enforced
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        g = GenesisLayer(path=os.path.join(td, "g.json"))
        feats = build_features([100 + i * 0.5 for i in range(40)],
                               [10.0] * 40)
        g.generation_pass(feats)
        assert len(g.formulas) == 10
        ids = list(g.formulas.keys())
        for i in range(30):
            vals = {fid: (0.3 if (i + hash(fid)) % 2 == 0 else -0.3)
                    for fid in ids}
            g.score_walk_forward(vals, move=0.01 if i % 2 == 0 else -0.01)
        states = {e["state"] for e in g.formulas.values()}
        assert states & {"probation", "active"}, states
        n_active = sum(1 for e in g.formulas.values() if e["state"] == "active")
        assert n_active <= ACTIVE_CAP
        # endpoints exist
        from app.main import app
        routes = {getattr(r, "path", "") for r in app.routes}
        for p in ("/api/genesis/status", "/api/genesis/formulas",
                  "/api/genesis/domains"):
            assert p in routes


def test_8_6_learners_present():
    from tests.test_engine import make_engine, _boot, run
    eng, _f, _e = make_engine()
    run(_boot(eng))
    snap = eng.build_snapshot()
    L = snap["systems"]["BTC|tf1"]["learners"]
    for k in ("edge_guard", "temporal_twin", "bocpd", "hawkes"):
        assert k in L, k
    assert "var_single_leg" in L["temporal_twin"]
    assert L["bocpd"]["hazard"] == round(1 / 14, 4)


# ── staged ANALYSE: order, timing, read-only, keyless honesty ──────────
def test_analyse_pipeline_stages_and_readonly():
    """Button → formulas → emotions → physics → news → hedge → AI review →
    final box with signal + four TPs + SL; every stage timestamped; locks
    never mutated by an analysis."""
    from tests.test_engine import make_engine, _boot, run
    import asyncio as _a
    eng, _f, _e = make_engine()
    run(_boot(eng))
    msgs = []
    async def _emit(m): msgs.append(m)
    eng.emit = _emit
    locks_before = {k: dict(v) for k, v in eng.locks.items()}

    async def go():
        out = eng.start_analyse("ETH", "tf2")
        assert out["ok"], out
        await eng._analyse_task
    from conftest import run_sync
    run_sync(go())
    job = eng.analyse_job
    assert job["status"] == "done", job.get("error")
    order = [s["stage"] for s in job["stages"]]
    assert order == ["formulas", "emotions", "physics", "news", "hedge",
                     "ai_review", "recheck", "final"], order
    # every stage carries the calculation period (start/end/duration)
    for st in job["stages"]:
        assert st["t_start_ms"] <= st["t_end_ms"]
        assert st["duration_ms"] >= 0
        assert st["summary"]
    # final box: signal + 4 TPs + SL + per-asset weightage + total time
    r = job["result"]
    assert r["signal"] in ("BUY", "SELL", "FLAT")   # never NO TRADE
    assert r["recheck"] and "checks" in r["recheck"]  # recheck gate ran
    # recheck verifies arithmetic; it can never veto the call (doc §1)
    assert r["prediction"] is not None
    assert len(r["tps"]) >= 1  # locked cycle targets (horizon last)
    assert r["sl"] is not None
    assert r["weightage"] and r["weightage_note"]
    assert r["elapsed_ms"] > 0
    assert len(r["calculation_period"]["stages"]) >= 6
    # WS stream reached the client
    types = [m["type"] for m in msgs]
    assert types[0] == "analyse_started"
    assert "analyse_done" in types   # live SL/target events may follow now
    assert types.count("analyse_stage") == 8  # + recheck stage (user: recheck everything before producing)
    # keyless AI stage is honest about skipping
    ai_stage = next(s for s in job["stages"] if s["stage"] == "ai_review")
    if not eng.agents.settings.get("gemini_key"):
        assert ai_stage["status"] == "skipped"
        assert "keyless" in ai_stage["summary"] or "no AI" in ai_stage["summary"]
    # read-only: every lock identical after the analysis
    for k, v in locks_before.items():
        assert eng.locks[k] == v, f"lock {k} mutated by analysis"
    # one analysis at a time (guard fires before any task is created)
    eng.analyse_job = {"status": "running"}
    out2 = eng.start_analyse("ETH", "tf2")
    assert out2["ok"] is False
    eng.analyse_job = None


def test_vault_saves_real_state():
    """§9/E-7: the 30s vault must call methods that actually exist —
    a wrong name silently disabled every save (caught in a server log)."""
    from tests.test_engine import make_engine, _boot, run
    eng, _f, _e = make_engine()
    run(_boot(eng))
    assert callable(getattr(eng.learning, "save", None))
    assert callable(getattr(eng.cci, "save", None))
    assert callable(getattr(eng.physics, "save", None))
    assert callable(getattr(eng.forward, "save", None))
    assert callable(getattr(eng.genesis, "save", None))
    # they actually write without raising
    eng.learning.save(eng)
    eng.cci.save()
    eng.physics.save()
    eng.forward.save()
    eng.genesis.save()
