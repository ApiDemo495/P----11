"""UX/sync contracts — the features added in the 0.00/UX pass must stay
wired: explainer (qwen + fallback), contribution ticks, AUTO-PREDICT
setting, dashboard-only ANALYSE button, and the doc/index surface.

These tests exist because the failure mode they guard is *drift*: one
layer gains a feature and the others silently lag behind.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- explainer
def test_explainer_fallback_is_deterministic_and_grounded():
    from app.explainer import fallback, _tp_sl_block

    ctx = {
        "phase": "live", "price": 75000.0, "atr": 100.0,
        "confidence": 0.42, "direction": 1, "signal": "BUY",
        "reason": "score above noise floor",
        "top": [{"name": "PEM", "contribution": 0.03,
                 "weight": 0.05, "value": 0.6}],
        "votes": {"pcv": 1, "brain": -1},
        "tps": [{"k": 1, "minutes": 5, "level": 75030.0},
                {"k": 2, "minutes": 10, "level": 75045.0}],
        "sl": 74900.0, "ratio": 1.4,
    }
    text = fallback("dashboard", "BTC", "tf1", ctx)
    assert "TP/SL MATH" in text
    assert "75030" in text and "74900" in text
    assert "TARGET_SCALE=0.5" in text
    # second call identical (deterministic — no LLM involved)
    assert fallback("dashboard", "BTC", "tf1", ctx) == text
    # TP formula is the engine's own: entry + dir*|conf|*atr*sqrt(min)*0.5
    block = _tp_sl_block(ctx)
    assert "√minutes_k" in block


def test_explainer_status_shape_and_offline_safety():
    from app import explainer

    st = explainer.status()
    for key in ("available", "model", "url", "source", "fallback_only"):
        assert key in st, key
    assert st["model"] == "qwen2.5-coder:3b"
    # never raises, never blocks on a missing daemon
    assert isinstance(st["source"], str) and st["source"]


@pytest.mark.asyncio
async def test_explain_returns_text_even_without_ollama():
    from app import explainer

    out = await explainer.explain("physics", "BTC", "tf1",
                                  {"ufp": 0.1, "wfcp": 0.9, "drpt": 0.2,
                                   "gspi": 0.01, "mechanisms": []})
    assert out["ok"] is True
    assert out["text"]
    assert out["source"] in ("qwen2.5-coder", "local-fallback")


# ------------------------------------------------------------------- ticks
def test_ticks_groups_and_engine_ctx():
    from tests.test_engine import make_engine

    engine, feed, _ = make_engine()
    ctx = engine.explain_ctx("dashboard", "BTC", "tf1")
    # TP/SL projection uses the engine's own formula
    assert "tps" in ctx and "sl" in ctx
    assert ctx["signal"] in ("BUY", "SELL", "FLAT")
    # the six tick groups are declared where the display systems build
    src = (ROOT / "app" / "engine.py").read_text()
    for group in ('"formulas":', '"emotions":', '"physics":',
                  '"hedge":', '"news":', '"brain":', '"ticks":'):
        assert group in src, group


# ----------------------------------------------------------- auto-predict
def test_auto_predict_setting_default_and_toggle():
    from tests.test_engine import make_engine

    engine, _, _ = make_engine()
    assert engine.agents.settings.get("auto_predict") is True
    engine.agents.configure(auto_predict=False)
    assert engine.agents.settings["auto_predict"] is False
    engine.agents.configure(auto_predict=True)
    assert engine.agents.settings["auto_predict"] is True
    # cadence guard exists and is pytest-safe
    src = (ROOT / "app" / "engine.py").read_text()
    assert "_last_auto_analyse" in src
    assert 'os.environ.get("PYTEST_CURRENT_TEST")' in src


def test_settings_endpoint_roundtrip():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        before = c.get("/api/settings").json()["settings"]
        assert "auto_predict" in before
        try:
            r = c.put("/api/settings", json={"auto_predict": False})
            assert r.json()["settings"]["auto_predict"] is False
        finally:
            c.put("/api/settings", json={"auto_predict": True})


# ------------------------------------------------------------- UI contracts
def test_analyse_button_dashboard_only_contract():
    html = (ROOT / "static" / "index.html").read_text()
    js = (ROOT / "static" / "app.js").read_text()
    assert html.count('id="btn-analyse"') == 1
    assert 'id="auto-chip"' in html
    assert "function syncAnalyseButton" in js
    assert 'S.view === "dashboard"' in js.split("function syncAnalyseButton")[1][:400]
    assert "bindAutoChip" in js


def test_fx_formatter_and_explain_panels_present():
    js = (ROOT / "static" / "app.js").read_text()
    assert "function fx(" in js
    assert "function ensureExplainPanel" in js
    assert 'window.toggleWhyExplain' in js
    assert "CONTRIBUTION TICKS" in js
    # leaderboard/day field fixes (the UNDEFINED / empty-day bugs)
    assert "r.name || r.symbol || r.tf" in js
    assert "d.records || d.rows || d.predictions" in js


def test_docs_and_requirements_in_sync():
    req = (ROOT / "requirements.txt").read_text()
    assert "pyarrow" in req and "scipy" in req      # flywire build deps
    readme = (ROOT / "README.md").read_text()
    # README must describe the current surface, not an older era
    for token in ("FlyWire", "/api/explain", "AUTO-PREDICT",
                  "221 tests", "CONTRIBUTION TICKS", "CONNECTOME_CACHE"):
        assert token in readme, token
    api = (ROOT / "API_SETUP.md").read_text()
    for token in ("flywire_783.npz", "qwen2.5-coder:3b",
                  "setup_local_llm.sh", "no fallback matrix"):
        assert token in api, token
    docs = (ROOT / "docs.html").read_text()
    assert "FlyWire" in docs and "/api/explain" in docs


def test_reports_exist_and_are_linked():
    for name in ("NEWS-REPORT.md", "BRAIN-REPORT.md", "UX-ZERO-REPORT.md"):
        assert (ROOT / "docs" / name).exists(), name
    readme = (ROOT / "README.md").read_text()
    for name in ("NEWS-REPORT", "BRAIN-REPORT", "UX-ZERO-REPORT"):
        assert name in readme, name
