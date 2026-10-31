"""Part 4 tests — auto-learning, agents, history DB, two-tier weights,
double-check, and the UI contract."""
from __future__ import annotations

import json
import os
import time

import pytest

from app import config
from app.agents import AgentManager, build_prompt, parse_agent_reply, _mask
from app.cci import CCIEngine, uniform_prior
from app.historydb import HistoryDB, empty_record, _day_stats
from app.learning import HMM2, LearningSystem, _pearson, _clamp


# ══════════════════════ Section 10 — learning core ══════════════════════

def test_config_part4_constants():
    assert config.ETA_MICRO == 0.02
    assert config.ETA_EMOTION == 0.015
    assert config.ETA_PHYSICS == 0.015
    assert config.BRAIN_W_MIN == 0.1 and config.BRAIN_W_MAX == 0.4
    assert config.EMOTION_W_MIN == 0.01 and config.EMOTION_W_MAX == 0.12
    assert config.PHYSICS_W_MIN == 0.01 and config.PHYSICS_W_MAX == 0.12
    assert config.MACRO_PERIOD_GRC == 15
    assert config.DEEP_PERIOD_MINUTES == 247
    assert config.ACCURACY_TARGET == 0.90
    assert config.HIGH_ALERT_LOW == 0.85 and config.HIGH_ALERT_HIGH == 0.92


def test_subsystem_shares_sum_one():
    assert abs(sum(config.SUBSYSTEM_SHARES_INIT.values()) - 1.0) < 1e-12
    assert config.SUBSYSTEM_SHARES_INIT["core"] == 0.45
    assert config.SUBSYSTEM_SHARES_INIT["emotion"] == 0.15
    assert config.SUBSYSTEM_SHARES_INIT["physics"] == 0.15
    assert config.SUBSYSTEM_SHARES_INIT["brain"] == 0.10
    assert config.SUBSYSTEM_SHARES_INIT["news"] == 0.07
    assert config.SUBSYSTEM_SHARES_INIT["hf"] == 0.08


def test_learning_rates_adaptive_high_alert():
    ls = LearningSystem()
    ls.update_adaptive_rates()
    assert ls.high_alert is False or ls.rates["micro"] == 0.04
    # simulate worst tf accuracy below 85% -> high alert doubles rates
    for tf in config.TF_ORDER:
        ls.tf_hits[tf].extend([0] * 60)
    ls._refresh_tf_hr()
    ls.update_adaptive_rates()
    assert ls.high_alert is True
    assert ls.rates["micro"] == 0.04
    assert ls.rates["emotion"] == 0.03
    assert ls.rates["physics"] == 0.03
    assert ls.probation_threshold == config.HIGH_ALERT_PROBATION
    # all timeframes above 92% -> back to normal
    for tf in config.TF_ORDER:
        ls.tf_hits[tf].clear()
        ls.tf_hits[tf].extend([1] * 60)
    ls._refresh_tf_hr()
    ls.update_adaptive_rates()
    assert ls.high_alert is False
    assert ls.rates["micro"] == config.ETA_MICRO


def test_record_micro_accuracy_and_brain_influence():
    ls = LearningSystem()
    signals = {1: 1, 2: -1, 3: 1, 21: 1}
    sub = {"emotion": 1, "physics": -1, "news": 1, "hf": 1, "brain": 1, "core": 1, "final": 1}
    ls.record_micro("BTC", "tf1", 1, signals, sub, [0.2, -0.1, 0.3, 0.1], 1, None)
    assert ls.formula_accuracy(1) == 1.0          # agreed with move
    assert ls.formula_accuracy(2) == 0.0          # disagreed
    w = ls.brain_influence("BTC", "tf1")
    assert abs(sum(w) - 1.0) < 1e-9               # renormalized
    assert w[0] > 0.25                            # correct brain gained
    assert w[1] < 0.25                            # wrong brain lost
    assert min(w) >= 0.09


def test_brain_influence_clamped_and_renormalized():
    ls = LearningSystem()
    for _ in range(200):
        ls.record_micro("BTC", "tf1", 1, {1: 1}, {"core": 1, "final": 1}, [1, -1, 0, 0], None, None)
    w = ls.brain_influence("BTC", "tf1")
    assert abs(sum(w) - 1.0) < 1e-9
    assert w[0] <= 0.4 + 1e-9


def test_performance_multiplier_axis():
    ls = LearningSystem()
    r = ls._acc_ring(ls.acc_tf, (21, "tf1"), 100)
    r.extend([1] * 50)
    for tf in ("tf2", "tf3", "tf4"):
        rr = ls._acc_ring(ls.acc_tf, (21, tf), 100)
        rr.extend([0] * 50)
    m = ls.performance_multiplier(21, "tf", "tf1")
    assert m > 1.0                                # outperforming its peers
    assert 0.5 <= m <= 2.0
    assert ls.performance_multiplier(99, "tf", "tf1") == 1.0   # unknown -> neutral


def test_macro_cycle_probation_and_bonus(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LEARN_STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr("app.learning.LEARN_STATE_FILE", str(tmp_path / "state.json"))
    ls = LearningSystem()
    engine = type("E", (), {})()
    cci = CCIEngine()
    for s in config.SYMBOL_ORDER:
        for tf in config.TF_ORDER:
            cci.weights[(s, tf)] = uniform_prior()
    engine.cci = cci

    class _Hist:
        @staticmethod
        def tf_key(symbol, tf, name):
            return f"{symbol}|{tf}|{name}"

        @staticmethod
        def ring(key, n):
            from collections import deque
            return deque([0.1, 0.2, 0.15], maxlen=n)

    engine.hist = _Hist()

    class _Brains:
        @staticmethod
        def macro_feedback(metrics):
            ls.macro_tone = 0.0
            return 0.0

    engine.brains = _Brains()
    # formula 1 -> accurate (bonus path), formula 2 -> terrible (probation)
    for _ in range(20):
        ls.record_micro("BTC", "tf1", 1, {1: 1, 2: -1, 3: 1},
                        {"core": 1, "final": 1}, [1, 0, 0, 0], None, None)
    w_before = dict(cci.get_weights("BTC", "tf1"))
    ls.macro_cycle(engine)
    w_after = cci.get_weights("BTC", "tf1")
    assert w_after[2] < w_before[2]               # probation halved the loser
    assert (config.SYMBOL_ORDER[0], "tf1", 2) in ls.probation
    assert abs(sum(w_after.values()) - 1.0) < 1e-9


def test_deep_cycle_subsystem_share_update(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LEARN_STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr("app.learning.LEARN_STATE_FILE", str(tmp_path / "state.json"))
    ls = LearningSystem()
    engine = type("E", (), {})()
    cci = CCIEngine()
    for s in config.SYMBOL_ORDER:
        for tf in config.TF_ORDER:
            cci.weights[(s, tf)] = uniform_prior()
    engine.cci = cci

    class _Hist:
        @staticmethod
        def tf_key(symbol, tf, name):
            return f"{symbol}|{tf}|{name}"

        @staticmethod
        def ring(key, n):
            from collections import deque
            return deque([0.1] * 8, maxlen=n)

    engine.hist = _Hist()

    class _Brains:
        @staticmethod
        def diagnostic_inhibit(tf, fids, until):
            _Brains.last = (tf, fids, until)

        @staticmethod
        def force_anchor(a):
            pass

    engine.brains = _Brains()
    engine.bgn = type("B", (), {"force_anchor": staticmethod(lambda a: None)})()

    # emotion subsystem always agreed + prediction correct -> contribution +
    for _ in range(12):
        ls.record_micro("BTC", "tf1", 1,
                        {1: 1, 2: -1, 3: 1, 4: 1, 5: 1},
                        {"emotion": 1, "physics": -1, "news": 1, "hf": 1,
                         "brain": 1, "core": 1, "final": 1},
                        [1, 0, 0, 0], None, None)
    # low hit rate -> diagnostic mode
    for tf in config.TF_ORDER:
        ls.tf_hits[tf].extend([0] * 30)
    ls._refresh_tf_hr()
    before = dict(ls.subsystem_shares)
    ls.deep_cycle(engine)
    shares = ls.subsystem_shares
    assert abs(sum(shares.values()) - 1.0) < 1e-9     # renormalized (L1 rule)
    assert shares["emotion"] >= before["emotion"] - 1e-9   # agreed -> gained
    assert shares["physics"] <= before["physics"] + 1e-9   # disagreed -> lost
    lo, hi = config.SUBSYSTEM_BEARINGS["emotion"]
    assert lo <= shares["emotion"] <= hi


def test_hmm_baum_welch_and_posterior():
    h = HMM2()
    import math, random
    random.seed(495)
    xs = [0.01 * math.sin(i / 3) + random.gauss(0, 0.002) for i in range(120)]
    ll = h.fit(xs, iters=4)
    assert math.isfinite(ll)
    p = h.state_probability(xs)
    assert 0.0 <= p <= 1.0
    d = h.to_dict()
    h2 = HMM2()
    h2.load_dict(d)
    assert h2.mu == pytest.approx(h.mu)


def test_learning_persistence_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LEARN_STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr("app.learning.LEARN_STATE_FILE", str(tmp_path / "state.json"))
    ls = LearningSystem()
    ls.subsystem_shares["emotion"] = 0.2
    ls.brain_w[("BTC", "tf1")] = [0.4, 0.2, 0.2, 0.2]
    ls.macro_count = 7
    ls.save()
    assert os.path.exists(config.LEARN_STATE_FILE)
    ls2 = LearningSystem()
    ls2.load_state()
    assert ls2.macro_count == 7
    assert ls2.subsystem_shares["emotion"] == pytest.approx(0.2)
    assert ls2.brain_influence("BTC", "tf1")[0] == pytest.approx(0.4)


def test_pearson_and_clamp():
    assert _pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert _pearson([1, 1, 1], [2, 3, 4]) == 0.0
    assert _clamp(5, 0, 1) == 1.0
    assert _clamp(-1, 0, 1) == 0.0


# ══════════════════════ Section 11 — agents ══════════════════════

def test_agent_prompt_contains_all_fields():
    p = build_prompt({
        "prices": {"BTC": 100.0}, "returns": {"BTC": 0.001},
        "cci_dir": 1, "cci_conf": 0.62, "top5": [{"formula": 11, "weight": 0.05, "signal": 0.4}],
        "brain_consensus": 1, "headline": "ETF inflows", "news_sentiment": 0.3, "cerc": 0.12,
    })
    assert "Prices" in p and "10-minute returns" in p
    assert "CCI prediction" in p and "Top-5 formula signals" in p
    assert "brain consensus" in p and "Latest news" in p
    assert "Emotion composite" in p and "JSON" in p


def test_parse_agent_reply_json_and_heuristic():
    r = parse_agent_reply('{"direction": 1, "confidence": 0.8, "insight": "strong flow"}')
    assert r["direction"] == 1 and r["confidence"] == 0.8
    r2 = parse_agent_reply('{"direction": −1, "confidence": 0.4}')     # unicode minus
    assert r2["direction"] == -1 and r2["confidence"] == 0.4
    r3 = parse_agent_reply("The data confirm bullish momentum with confidence 72")
    assert r3["direction"] == 1 and r3["confidence"] == pytest.approx(0.72)


def test_mask_key():
    assert _mask("") == ""
    assert _mask("AIzaShort") == "AIza•hort"
    m = _mask("AIzaSyEXAMPLEKEY1234567")
    assert m.startswith("AIza") and m.endswith("567") and "•" in m


def test_agent_failover_reuses_signal():
    am = AgentManager()
    am.settings["gemini_key"] = "test-key-123"
    am.last_good["gemini"] = {"direction": 1, "confidence": 0.7, "insight": "kept"}
    am.fail_streak["gemini"] = 1
    merged = am._merge(None, None)          # cloud down, local down
    # _merge itself returns none-source; fallback happens in query(). Check merge paths:
    assert am._merge({"direction": 1, "confidence": 0.7, "insight": "k"}, None)["weight"] \
        == config.AGENT_GEMINI_WEIGHT
    assert am._merge(None, {"direction": -1, "confidence": 0.6})["weight"] \
        == config.AGENT_LOCAL_WEIGHT
    both = am._merge({"direction": 1, "confidence": 0.8},
                     {"direction": -1, "confidence": 0.8})
    assert both["weight"] == config.AGENT_COMBINED_WEIGHT
    assert both["direction"] == 0            # opposite agents cancel


def test_agent_no_keys_status():
    am = AgentManager()
    st = am.status()
    assert st["slots"][0]["status"] in ("not configured", "disabled")
    assert st["local"]["available"] in (True, False)


# ══════════════════════ Section 13 — history DB ══════════════════════

def test_historydb_append_and_days(tmp_path):
    db = HistoryDB(str(tmp_path))
    rec = empty_record()
    rec["outcomes"] = {"tf1": {"target_hit": True, "sl_hit": False}}
    db.append(rec)
    rec2 = empty_record()
    rec2["outcomes"] = {"tf1": {"target_hit": False, "sl_hit": True}}
    db.append(rec2)
    days = db.days()
    assert days and days[0]["grcs"] == 2
    assert days[0]["targets_hit"] == 1 and days[0]["sls_hit"] == 1
    rows = db.day(days[0]["date"])
    assert len(rows) == 2
    assert "predictions" in rows[0] and "weights" in rows[0]


def test_history_retention_compression(tmp_path):
    db = HistoryDB(str(tmp_path))
    old_day = "2020-01-01"
    rec = empty_record()
    rec["outcomes"] = {"tf1": {"target_hit": True, "sl_hit": False}}
    # write an old file directly
    with open(db._path(old_day), "w") as fh:
        json.dump([rec, rec], fh)
    n = db.compress_old()
    assert n == 1
    assert not os.path.exists(db._path(old_day))
    summaries = json.load(open(os.path.join(str(tmp_path), "weekly_summaries.json")))
    assert summaries and summaries[0]["date"] == old_day
    assert summaries[0]["targets_hit"] == 2


def test_day_stats_accuracy():
    rows = [
        {"outcomes": {"tf1": {"target_hit": True}, "tf2": {"target_hit": False, "sl_hit": True}}},
    ]
    s = _day_stats("2026-01-01", rows)
    assert s["predictions"] == 2
    assert s["accuracy"] == pytest.approx(0.5)


# ══════════════════════ Section 14/16 — two-tier + double-check ══════════════════════

def test_effective_weights_two_tier():
    w = uniform_prior()
    shares = dict(config.SUBSYSTEM_SHARES_INIT)
    eff = CCIEngine.effective_weights(w, shares)
    assert eff[config.BRAIN_COMPONENT_ID] == shares["brain"]
    assert eff[config.EMOTION_COMPONENT_ID] == shares["emotion"]
    assert eff[config.PHYSICS_COMPONENT_ID] == shares["physics"]
    assert eff[config.NEWS_COMPONENT_ID] == shares["news"]
    assert eff[config.HEDGE_COMPONENT_ID] == shares["hf"]
    core_sum = sum(v for k, v in eff.items() if k in config.CORE_SIGNAL_IDS)
    assert core_sum == pytest.approx(shares["core"], abs=1e-9)
    assert abs(sum(eff.values()) - 1.0) < 1e-9


def test_double_check_confirmed_mixed_overridden():
    dc = CCIEngine.double_check(1, 0.5, {"pcv": 1, "agent": 1, "brain": -1})
    assert dc["label"] == "confirmed" and dc["issued"] is True
    assert dc["confidence"] == pytest.approx(0.525)          # ×1.05
    dc2 = CCIEngine.double_check(1, 0.5, {"pcv": -1, "agent": 0, "brain": 1})
    assert dc2["label"] == "mixed" and dc2["confidence"] == pytest.approx(0.425)
    dc3 = CCIEngine.double_check(1, 0.5, {"pcv": -1, "agent": -1, "brain": -1})
    assert dc3["label"] == "overridden" and dc3["direction"] == -1
    assert dc3["issued"] is False
    dc4 = CCIEngine.double_check(1, 0.5, {"pcv": 0, "agent": 0, "brain": 0})
    assert dc4["label"] == "standalone"


def test_cci_compute_respects_shares():
    eng = CCIEngine()
    shares = dict(config.SUBSYSTEM_SHARES_INIT)
    shares["emotion"] = 0.25                    # up-shared emotion
    shares["core"] = 0.35
    adjusted = {f: 0.0 for f in range(1, 32)}
    for f in range(1, 20):
        adjusted[f] = 0.5 if f % 2 else -0.5
    adjusted[config.EMOTION_COMPONENT_ID] = 1.0     # strong emotion signal
    adjusted[config.PHYSICS_COMPONENT_ID] = 0.0
    adjusted[config.NEWS_COMPONENT_ID] = 0.0
    adjusted[config.BRAIN_COMPONENT_ID] = 0.0
    adjusted[config.HEDGE_COMPONENT_ID] = 0.0
    p1 = eng.compute("BTC", "tf1", dict(adjusted), 1.0, shares=shares)
    flat = eng.compute("BTC", "tf1", dict(adjusted), 1.0,
                       shares=dict(config.SUBSYSTEM_SHARES_INIT))
    # emotion weight difference must move the CCI
    assert p1["cci"] != flat["cci"]
    assert "weights_used" in p1


def test_engine_learning_wiring():
    from app.engine import PredictionEngine
    from app.feed import MarketFeed
    eng = PredictionEngine(MarketFeed(), emit=lambda m: None)
    assert eng.learning is not None
    assert eng.agents is not None
    assert eng.historydb is not None
    assert eng.learning.subsystem_shares["core"] == pytest.approx(0.45)
    # double-check + agents hooks exist
    assert hasattr(eng, "_apply_agents")
    assert hasattr(eng, "_record_learning")


def test_brain_influence_and_inhibition_api():
    from app.brain import BrainSystem
    b = BrainSystem()
    b.set_influence("BTC", "tf1", [0.4, 0.2, 0.2, 0.2])
    assert b.influence_for("BTC", "tf1") == [0.4, 0.2, 0.2, 0.2]
    tone = b.macro_feedback({1: 0.9, 2: 0.1})
    assert -0.5 <= tone <= 0.5
    b.diagnostic_inhibit("tf1", [3, 7], time.time() + 100)
    assert b._inhibited[("tf1", 3)] > time.time()
    st = b.state_dict()
    assert "pattern_valence" in st and "influence" in st
    b2 = BrainSystem()
    b2.load_state(st)
    assert b2.influence_for("BTC", "tf1") == [0.4, 0.2, 0.2, 0.2]


def test_bgn_deep_anchor():
    from app.history import HistoryStore
    from app.bgn import BgnSynchronizer
    hist = HistoryStore()
    bgn = BgnSynchronizer(hist)
    anchor = {"tf1": {"tf2": 0.55}, "tf2": {"tf1": 0.55}}
    bgn.force_anchor(anchor)
    coh = bgn._coherence("BTC", "tf1", "tf2")     # rolling empty -> anchor alone
    assert coh == pytest.approx(0.55)


def test_emotion_and_physics_eta_clamps():
    from app.emotion import ETA as E_ETA, W_CLAMP_LO, W_CLAMP_HI
    from app.physics import ETA_PHYSICS, PHYSICS_W_MIN, PHYSICS_W_MAX
    assert E_ETA == 0.015
    assert W_CLAMP_LO == 0.01 and W_CLAMP_HI == 0.12
    assert ETA_PHYSICS == 0.015
    assert PHYSICS_W_MIN == 0.01 and PHYSICS_W_MAX == 0.12


def test_main_part4_routes_present():
    import importlib
    main = importlib.import_module("app.main")
    routes = {getattr(r, "path", None) for r in main.app.routes}
    for p in ("/api/learning", "/api/learning/macro", "/api/learning/deep",
              "/api/learning/reset", "/api/history/days", "/api/history/day",
              "/api/history/replay", "/api/history/leaderboard",
              "/api/settings", "/api/agents/status", "/api/agents/test/{slot}",
              "/api/state/export", "/api/state/import", "/api/systems"):
        assert p in routes, f"missing route {p}"


def test_ui_contract_static_assets():
    assert os.path.exists("static/index.html")
    html = open("static/index.html").read()
    for tab in ("dashboard", "brains", "formulas", "emotion", "physics",
                "news", "agents", "history", "settings"):
        assert f'tab-{tab}' in html
    assert "#0a0a0f" in open("static/style.css").read()
    assert "#00e5ff" in open("static/style.css").read()
    assert "#ff1744" in open("static/style.css").read()
    assert "#ffd600" in open("static/style.css").read()
    assert os.path.exists("static/vendor/lightweight-charts.standalone.production.js")
    assert os.path.exists("static/vendor/katex.min.js")
    js = open("static/app.js").read()
    assert "TAB_LOADERS" in js and "MIXED SIGNALS" not in js  # label comes from backend


# ── Part 4: agents are OPTIONAL (Drosophila-mode guarantee) ────────────
def test_agent_mode_keyless_runs_drosophila(monkeypatch, tmp_path):
    """Without any API key the app reports 'drosophila' mode and never
    blocks; adding a key flips it to cloud; local model counts too."""
    from app import agents as ag
    monkeypatch.setattr(ag, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    m = ag.AgentManager()
    m.local.available = False
    st = m.status()
    assert st["mode"] == "drosophila"
    assert st["has_cloud_keys"] is False
    # every cloud slot is merely "not configured" — nothing throws
    assert all(s["status"] in ("not configured", "disabled") for s in st["slots"])
    # paste a key → cloud mode, persisted, reloaded
    m.configure(gemini_key="AIzaTESTKEY")
    assert m.status()["mode"] == "cloud"
    assert ag.AgentManager().status()["has_cloud_keys"] is True
    # keyless + local model → local mode
    m.configure(gemini_key="")
    m.local.available = True
    assert m.status()["mode"] == "local"
