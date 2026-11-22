"""Repairs for app code earlier sessions left broken (diagnosed live).

1. ``GET /api/history`` 500'd for every caller: the handler stored the
   bound method ``engine.hindsight`` (truthy callable) instead of
   calling it, and ``JSONResponse`` sits outside the per-section
   try/except, so one bad leaf killed the whole console.
2. ``display_systems`` (the snapshot's ``systems`` block) dropped the
   fin fields the hero reads — ``p_correct``/``calib_iso``/``spillover``
   (P(win) calibration + spillover rows) rendered "insufficient"
   forever even when calibrated.
3. The engine emitted ``prediction_settled`` on every lock settlement
   and the UI never handled it — books/WR tables stayed stale until a
   REST poll.
4. ``haptic()`` read ``S.snap.settings.haptics`` that no snapshot ever
   carried, so "haptics off" could never reach the client.
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _one_client():
    """ONE lifespan per file — a second `with TestClient(app)` runs the
    lifespan twice and the app's own tests document the breakage."""
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_history_console_serves_json():
    with _one_client() as c:
        r = c.get("/api/history")
        snap = c.get("/api/snapshot").json()
    _check_history(r)
    _check_snapshot(snap)


def _check_snapshot(snap):
    st = snap.get("settings")
    assert isinstance(st, dict), "snapshot.settings missing"
    assert isinstance(st.get("haptics"), bool)
    assert not any(k.endswith("_key") or k == "gemini_key" for k in st)
    src = open(os.path.join(ROOT, "app", "engine.py")).read()
    wl = src.split("self.display_systems[(symbol, tf)] = {", 1)[1][:6000]
    for k in ("p_correct", "calib_iso", "spillover", "winrate",
              "winrate_n", "issue_note", "gate_ok", "tps"):
        assert f'"{k}": results[tf].get(' in wl, \
            f"display_systems whitelist dropped {k}"


def _check_history(r):
    assert r.status_code == 200, r.text[:300]
    data = r.json()
    assert data.get("ok") is True
    hz = data.get("hindsight")
    assert isinstance(hz, dict), type(hz)
    # the §8.3 re-grade builder (a METHOD on the engine) must have been
    # CALLED — never a bound-method object, never an error section
    assert "error" not in (hz or {}), (hz or {}).get("error")
    assert hz
    # the defensive serializer must not have needed to stringify leaves
    assert not any("not serializable" in json.dumps(v, default=str)
                   for v in data.values())


def test_snapshot_settings_and_whitelist_pins():
    # pure source + artefact pins — no second lifespan
    _check_snapshot({"settings": {"haptics": True}})


def test_ui_handles_prediction_settled():
    js = open(os.path.join(ROOT, "static", "app.js")).read()
    # the WS dispatcher must route it — engine emits it every settle
    assert 'msg.type === "prediction_settled"' in js
    assert "function onPredictionSettled" in js
    # handler must actually refresh the sealed-predictions table
    m = re.search(r"function onPredictionSettled.*?(?=\nfunction )",
                  js, re.S)
    assert m, "handler not found"
    assert "loadSealedPredictions" in m.group(0)
    assert "S.predStats" in m.group(0)
