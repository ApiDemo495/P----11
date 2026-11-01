"""Local Ollama formula auditor + Groq committee synthesis (user prompts)."""
import asyncio
import json


# ───────────────────────── auditor ─────────────────────────
def test_auditor_trigger_accuracy_below_45():
    from app.auditor import FormulaAuditor, ACCURACY_WINDOW
    a = FormulaAuditor()
    # full window of wrong predictions: reward -1 → accuracy 0%
    ring = [-1.0] * ACCURACY_WINDOW
    a.check_accuracy("BTC", "tf1", 5, ring)
    assert a.triggers["accuracy"] == 1
    # cooldown: same formula+kind again within 10 min does not re-trigger
    a.check_accuracy("BTC", "tf1", 5, ring)
    assert a.triggers["accuracy"] == 1
    # healthy ring does NOT trigger
    a.check_accuracy("BTC", "tf1", 6, [1.0] * ACCURACY_WINDOW)
    assert a.triggers["accuracy"] == 1
    # short window ignored
    a.check_accuracy("BTC", "tf1", 7, [-1.0] * 10)
    assert a.triggers["accuracy"] == 1


def test_auditor_anomaly_collector_and_offline_graceful():
    from app.auditor import FormulaAuditor
    a = FormulaAuditor()
    a._available = False; a._ping_at = 1e18      # force "offline"
    a.report_anomalies("BTC", "tf1", [(3, "nan", None)], {3: 0.0})
    assert a.triggers["exceptions"] >= 1
    assert a.triggers["skipped_offline"] >= 1    # no daemon → recorded, no crash
    st = a.status()
    assert st["model"] == "qwen2.5-coder:3b"
    assert "setup_local_llm" in st["hint"]


def test_compute_all_collects_anomalies():
    from app.formulas import FORMULA_FUNCTIONS
    from app.formulas import compute_all
    # monkeypatch one formula to raise, another to return NaN
    orig = dict(FORMULA_FUNCTIONS)
    try:
        def boom(ctx): raise ZeroDivisionError("div by zero")
        def nan(ctx): return float("nan")
        ids = list(FORMULA_FUNCTIONS)[:2]
        FORMULA_FUNCTIONS[ids[0]] = boom
        FORMULA_FUNCTIONS[ids[1]] = nan
        anoms = []
        from tests.test_engine import make_engine, _boot, run
        eng, _f, _e = make_engine(); run(_boot(eng))
        ctx = eng.build_ctx("BTC", "tf1")
        compute_all(ctx, record=False, anomalies=anoms)
        kinds = {k for _, k, _ in anoms}
        assert "exception" in kinds and "nan" in kinds
        # collector optional → old behavior preserved
        out = compute_all(ctx, record=False)
        assert all(v == v for v in out.values())
    finally:
        FORMULA_FUNCTIONS.clear(); FORMULA_FUNCTIONS.update(orig)


def test_setup_local_llm_script_exists_and_is_safe():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    sh = (root / "scripts" / "setup_local_llm.sh").read_text()
    assert "qwen2.5-coder:3b" in sh
    assert "ollama.com/install.sh" in sh
    assert "ollama pull" in sh


# ───────────────────────── groq ─────────────────────────
def test_groq_catalog_packs_and_key_handling():
    import tempfile, os
    from app import agents as ag
    old = ag.SETTINGS_FILE
    with tempfile.TemporaryDirectory() as td:
        ag.SETTINGS_FILE = os.path.join(td, "s.json")
        try:
            am = ag.AgentManager()
            ids = [p["id"] for p in am.catalog_status()["packs"]]
            assert "groq-70b" in ids and "groq-8b" in ids
            out = am.import_pack("groq-70b", "gsk_demo")
            assert out["ok"] and out["slot"] == "groq"
            assert am.groq_key == "gsk_demo"
            st = am.status()
            assert st["groq"]["configured"] is True
            assert st["groq"]["primary"] == "llama-3.3-70b-versatile"
            assert st["groq"]["triage"] == "llama-3.1-8b-instant"
        finally:
            ag.SETTINGS_FILE = old


def test_groq_synthesize_parses_and_bounds():
    import tempfile, os
    from app import agents as ag
    old = ag.SETTINGS_FILE
    with tempfile.TemporaryDirectory() as td:
        ag.SETTINGS_FILE = os.path.join(td, "s.json")
        try:
            am = ag.AgentManager()
            am.settings["groq_key"] = "gsk_demo"
            fake = {"ok": True, "text": json.dumps({
                "regime": "Trending",
                "weight_adjustments": {"5": 2.5, "9": 0.1, "bad": "x"},
                "confidence_score": 0.8,
                "rationale": "momentum aligned across timeframes"})}
            am._groq_call = lambda *a, **k: asyncio.sleep(0, result=fake)
            out = asyncio.get_event_loop().run_until_complete(
                am.synthesize({"symbol": "BTC", "direction": 1}))
            assert out["ok"] and out["regime"] == "Trending"
            # adjustments bounded to [0.5, 1.5], bad entries dropped
            assert out["weight_adjustments"]["5"] == 1.5
            assert out["weight_adjustments"]["9"] == 0.5
            assert "bad" not in out["weight_adjustments"]
            assert out["confidence_score"] == 0.8
            # invalid regime rejected
            fake["text"] = json.dumps({"regime": "sideways",
                                       "confidence_score": 0.5})
            out2 = asyncio.get_event_loop().run_until_complete(
                am.synthesize({}))
            assert not out2["ok"]
        finally:
            ag.SETTINGS_FILE = old


def test_groq_backoff_returns_failure_without_key():
    from app.agents import AgentManager
    am = AgentManager()
    am.settings.pop("groq_key", None)
    out = asyncio.get_event_loop().run_until_complete(am.synthesize({}))
    assert out["ok"] is False and out["reason"] == "no GROQ_API_KEY"


def test_engine_synthesis_endpoint_and_status_payload():
    from tests.test_engine import make_engine, _boot, run
    eng, _f, _e = make_engine(); run(_boot(eng))
    # no key → synthesis silently skipped
    run(eng._apply_groq_synthesis("BTC", {}))
    # status payload includes auditor + groq
    st = eng.agents.status()
    assert "groq" in st
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    r = c.get("/api/auditor/status")
    assert r.status_code == 200 and r.json()["model"] == "qwen2.5-coder:3b"
    r2 = c.get("/api/agents/status")
    body = r2.json()
    assert "auditor" in body["agents"] and "groq" in body["agents"]
