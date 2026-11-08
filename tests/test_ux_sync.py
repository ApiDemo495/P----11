"""UX/sync contracts — the features added in the 0.00/UX pass must stay
wired: explainer (qwen + fallback), contribution ticks, AUTO-PREDICT
setting, dashboard-only ANALYSE button, and the doc/index surface.

These tests exist because the failure mode they guard is *drift*: one
layer gains a feature and the others silently lag behind.
"""
from __future__ import annotations

import json
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

    # NOTE: the file's ONLY `with TestClient(app)` — a second lifespan in
    # the same pytest process re-binds engine asyncio primitives to a new
    # loop and crashes shutdown (RuntimeError: Event bound to a different
    # event loop). Every app-level HTTP contract in this file lives here.
    with TestClient(app) as c:
        before = c.get("/api/settings").json()["settings"]
        assert "auto_predict" in before
        try:
            r = c.put("/api/settings", json={"auto_predict": False})
            assert r.json()["settings"]["auto_predict"] is False
        finally:
            c.put("/api/settings", json={"auto_predict": True})
        # confidence calibration (stated vs realised) — honest cold start
        cal = c.get("/api/history/calibration").json()
        for k in ("n", "enough", "min_n", "bins", "brier", "note"):
            assert k in cal, k
        assert cal["enough"] is False and cal["n"] == 0
        # settlement liveness — the zero-wall must explain itself
        led = c.get("/api/ledger").json()
        for k in ("settle_calls", "last_settle_ago_s", "totals",
                  "rows_total"):
            assert k in led, k
        # system status: history + code-freshness rows exist
        st = c.get("/api/system/status").json()
        labels = [r["label"] for r in st.get("rows") or []]
        assert "history" in labels and "code" in labels, labels
        # leaderboard carries n so the UI can refuse to speak below 8
        lb = c.get("/api/history/leaderboard").json()
        assert all("n" in r for r in (lb.get("leaderboard") or []))
        # ids97-101 are subsystem components — never mislabeled F97…
        subs = [r for r in lb.get("leaderboard") or [] if r["formula"] >= 97]
        assert subs and all(r.get("subsystem") and "subsystem" in r["name"]
                            for r in subs), subs


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
    assert "d.records || d.rows" in js            # flattened per-GRC rows


def test_docs_and_requirements_in_sync():
    req = (ROOT / "requirements.txt").read_text()
    assert "pyarrow" in req and "scipy" in req      # flywire build deps
    readme = (ROOT / "README.md").read_text()
    # README must describe the current surface, not an older era
    for token in ("FlyWire", "/api/explain", "AUTO-PREDICT",
                  "CONTRIBUTION TICKS", "CONNECTOME_CACHE"):
        assert token in readme, token
    # test count must track the real collection (no stale "54 tests")
    m = re.search(r"#\s*(\d+) tests", readme)
    assert m, "README must state the test count"
    assert int(m.group(1)) >= 230, f"stale test count in README: {m.group(1)}"
    ini = (ROOT / "pytest.ini").read_text()
    assert "asyncio_mode = strict" in ini
    assert "no:cacheprovider" in ini
    pins = (ROOT / "requirements.txt").read_text()
    assert re.search(r"pytest==9\.", pins), pins
    assert "pytest-asyncio==1." in pins
    api = (ROOT / "API_SETUP.md").read_text()
    for token in ("flywire_783.npz", "qwen2.5-coder:3b",
                  "setup_local_llm.sh", "no fallback matrix"):
        assert token in api, token
    docs = (ROOT / "docs.html").read_text()
    assert "FlyWire" in docs and "/api/explain" in docs


def test_news_aggregate_rows_carry_ts():
    """The 'news stuck at 05:30' bug: aggregate() repacks items and used
    to drop `ts` → the table rendered Date(0) for every row."""
    from app.news import NewsSystem

    ns = NewsSystem()
    ns.refresh()
    agg = ns.aggregate("BTC")
    rows = agg.get("items") or []
    assert rows, "aggregate must expose scored items"
    import time as _t
    for r in rows[:5]:
        assert isinstance(r.get("ts"), (int, float)), r
        assert r["ts"] > _t.time() - 86400 * 2, r.get("ts")  # not epoch


def test_dashboard_only_chrome_and_zigzag_chart():
    js = (ROOT / "static" / "app.js").read_text()
    # asset/timeframe pickers live on the dashboard only
    fn = js.split("function syncAnalyseButton")[1].split("\n}")[0]
    assert "seg-assets" in fn and "seg-tf" in fn
    # charts draw per-bar high→low wicks (real-market zigzag)
    assert "class PriceChart" in (ROOT / "static" / "LineChart.js").read_text()  # own PriceChart engine replaces candle ChartX
    # grid rows must not stretch (the blank-space bug)
    css = (ROOT / "static" / "style.css").read_text()
    assert "align-content: start" in css
    assert "align-self: start" in css
    # ticker reloads on newest-ts change too
    assert "NEWS_LAST_TS" in js



def test_markdown_preview_remap_and_fast_autostart():
    """Preview fix must be backgrounded/only-if-missing, inside the
    one-file boot (start.py editor_extras daemon thread)."""
    start = (ROOT / "start.py").read_text()
    assert "def editor_extras" in start
    assert "--install-extension" in start
    assert "threading.Thread(target=work, daemon=True).start()" in start



def test_autostart_docs_off_critical_path():
    start = (ROOT / "start.py").read_text()
    assert "make_docs_html.py" in start
    assert "fetch_real_brain" not in start


def test_brain_display_refresh_and_provisional_sl():
    """Dashboard must not LIE about the brain between GRC passes (user
    saw absent+weight0 while the brain tab was ready), and below the
    confidence floor the calls table must still show a SEEDED SL/ratio."""
    import asyncio
    from tests.test_engine import make_engine, run, _boot
    eng, _f, _e = make_engine()
    run(_boot(eng))
    # force the component ready with raws available, stale display says absent
    saved = eng.brains.fw.state
    try:
        eng.brains.fw.state = "ready"
        eng._last_raws[("BTC", "tf1")] = {i: 0.4 for i in range(1, 32)}
        eng.display_systems[("BTC", "tf1")] = {
            "brain": {"flywire": {"ready": False, "vote": None}},
            "ticks": {"brain": {"flywire": {"ready": False, "on": False}}},
        }
        eng._refresh_flywire_display()
        f = eng.display_systems[("BTC", "tf1")]["brain"]["flywire"]
        assert f["ready"] is True, f
        tk = eng.display_systems[("BTC", "tf1")]["ticks"]["brain"]["flywire"]
        assert tk["ready"] is True
        # snapshot path invokes the refresh itself
        eng.display_systems[("BTC", "tf1")]["brain"]["flywire"]["ready"] = False
        eng.build_snapshot()
        f2 = eng.display_systems[("BTC", "tf1")]["brain"]["flywire"]
        assert f2["ready"] is True
        # provisional SL/ratio code present + calibration stash wired
        src = (ROOT / "app" / "engine.py").read_text()
        assert "sl_provisional" in src and "_calib_pending" in src
    finally:
        eng.brains.fw.state = saved
        run(eng.stop())


def test_calibration_and_day_flatten_ui_contract():
    js = (ROOT / "static" / "app.js").read_text()
    assert "needs ≥8" in js
    assert "rc.predictions" in js        # day detail flattens per-GRC records
    assert "hs-calib" in (ROOT / "static" / "index.html").read_text()
    assert "sl_provisional" in js
    # chart: denser window + strong wicks
    lc = (ROOT / "static" / "LineChart.js").read_text()
    assert "length > 1200" in lc and "createLinearGradient" in lc  # PriceChart caps pts, gradient area


def test_old_app_inspiration_bundle():
    """The old-agent-app inspiration pass: SYSTEM panel, probability
    branches, who-decided share, conviction note, brain read-out/pathways
    /channel matrix, dominant emotion, live-value ticker — all wired."""
    # server surface
    main = (ROOT / "app" / "main.py").read_text()
    assert "/api/system/status" in main
    assert "channel_map" in main
    eng = (ROOT / "app" / "engine.py").read_text()
    assert "def branch_stats" in eng and '"who": {' in eng
    fw = (ROOT / "app" / "flywire.py").read_text()
    for token in ("_last_whole", "_last_mb", "channel_matrix",
                  "pathways", "hop_counts", "top_mbons"):
        assert token in fw, token
    # branch math is sane: closer target → higher P(target), probs sum to1
    from tests.test_engine import make_engine, run, _boot
    import asyncio
    e2, _ff, _ee = make_engine()
    run(_boot(e2))
    try:
        d = {"symbol": "BTC", "direction": 1, "sl": 90.0,
             "countdown_s": 600, "targets": [{"level": 110.0}]}
        bs = e2.branch_stats(d, 100.0, __import__("time").time(), "tf1")
        # parts are4-decimal display values — sum within rounding
        assert bs and abs(bs["p_target"] + bs["p_stop"] +
                          bs["p_between"] - 1.0) < 1e-3
        assert 0.0 <= bs["p_target"] <= 1.0 and len(bs["bands"]) == 5
        d2 = dict(d, targets=[{"level": 101.0}])
        bs2 = e2.branch_stats(d2, 100.0, __import__("time").time(), "tf1")
        assert bs2["p_target"] > bs["p_target"]   # nearer target wins
    finally:
        run(e2.stop())
    # UI surface
    js = (ROOT / "static" / "app.js").read_text()
    html = (ROOT / "static" / "index.html").read_text()
    for token in ("loadSystemPanel", "renderBranches", "tickFormulaValues",
                  "WHO DECIDED", "CONVICTION", "em-dominant",
                  "br-readout", "br-pathways", "br-chan-matrix"):
        assert token in js or token in html, token
    for anchor in ("sys-panel", "branch-canvas", "fl-ticker",
                   "br-readout", "em-dominant"):
        assert anchor in html, anchor


def test_visibility_sync_and_flat_fixes():
    """User round: 'brain not connected to dashboard / charts flat /
    zeros without reason / app looks flatty'."""
    html = (ROOT / "static" / "index.html").read_text()
    js = (ROOT / "static" / "app.js").read_text()
    css = (ROOT / "static" / "style.css").read_text()
    # brain connection visible on EVERY tab (rail LED + toolbar chip)
    assert 'id="led-brain"' in html and 'id="brain-chip"' in html
    assert 'led("#led-brain"' in js and 'brain-chip' in js
    # robust chart range (percentile) + clamped labels
    lc = (ROOT / "static" / "LineChart.js").read_text()
    assert "q95" in js and "padR = spark ? 2 : 62" in lc  # empirical fan quantiles + right price axis
    # emotion cells print small-but-real values (0.004 ≠ 0.00)
    assert "fx(v, 2)" in js
    # physics strip: cause tag + why in title/hint
    assert "m.cause" in js and "[${esc(m.cause)}]" in js
    # depth pass exists (flatty fix)
    assert "DEPTH PASS" in css and "box-shadow" in css.split("DEPTH PASS")[0][-4000:]
    # formulas charts: never overflow the screen + never flat-by-scaling
    assert ".spark { display: block; width: 100%" in css          # was: default300px canvas
    assert "#fl-forward, #fl-genesis" in css and "overflow-x: auto" in css
    assert ".card { background: var(--panel); border: 1px solid var(--line); border-radius: 9px; padding: 12px 14px; min-width: 0;" in css
    assert "<th>tick</th>" not in js                 # tick merged into the name cell
    assert "S.fvRings" in js and "q95" in js   # live rings + empirical fan range
    assert "devicePixelRatio" in (ROOT / "static" / "LineChart.js").read_text()  # HiDPI in PriceChart


def test_history_not_updating_guards():
    """User: 'app not updating + history all zero'. Guards: locks never
    store entry0 (boot race → skip), emergency refuses price0, and the
    UI header must surface settlement liveness."""
    eng_src = (ROOT / "app" / "engine.py").read_text()
    assert "LAST CLOSE on the tape" in eng_src          # lock price fallback
    assert "emergency re-lock refused" in eng_src       # no price0 clobber
    led_src = (ROOT / "app" / "ledger.py").read_text()
    assert "settle_calls" in led_src and "last_settle_ts" in led_src
    html = (ROOT / "static" / "index.html").read_text()
    assert 'id="led-live"' in html
    js = (ROOT / "static" / "app.js").read_text()
    assert "no window settled yet" in js                # honest empty state
    assert "last_settle_ago_s" in js


def test_settle_halt_and_curl_and_gap_fixes():
    fw = (ROOT / "app" / "flywire.py").read_text()
    # curl-fallback states count as success (a full download was being
    # marked failed → brain stayed absent)
    assert 'str(v).startswith("downloaded")' in fw
    eng = (ROOT / "app" / "engine.py").read_text()
    # halt gates only the new lock — settlement writes first
    assert "halt gates ONLY the new lock" in eng
    css = (ROOT / "static" / "style.css").read_text()
    assert "minmax(112px, 1fr)" in css       # fills wide stat rows (physics gap)
    js = (ROOT / "static" / "app.js").read_text()
    assert "weak lean inside noise floor" in js   # tf3-style lean honesty


def test_dashboard_rewrite_tf_focus_candles_and_tp_first():
    """User: rewrite the dashboard — focus mode (chosen TF + its
    sub-timeframe TPs only), candle charts instead of flat lines, and the
    TF-horizon TP always calculated & shown first."""
    html = (ROOT / "static" / "index.html").read_text()
    js = (ROOT / "static" / "app.js").read_text()
    eng = (ROOT / "app" / "engine.py").read_text()
    # focus-mode hero + revert button + sub-timeframe strip
    for token in ("tf-hero", "chart-binance", "chart-tv", "chart-own", "subtf-strip", "btn-all-tf"):
        assert token in html, token
    assert "tfFocus" in js and 'localStorage.setItem("tfFocus"' in js
    assert "function renderHero" in js and "renderDashboardMode" in js
    # calls table follows focus + TF-horizon TP first
    assert 'TF.filter(tf => !S.tfFocus || tf === S.focusTf)' in js
    assert "TF-TP (horizon)" in js
    assert "_ord.unshift(_ord.pop())" in js        # [3,0,1,2] order
    # engine precomputes the horizon target first
    assert "ALWAYS calculated FIRST" in eng
    # charts: CANDLES on the quad cards (grid + scale) — the HERO is
    # now the user's requested line-chart-from-now (candles crushed the
    # live price into the corner under a multi-hour trend)
    assert "createLinearGradient" in (ROOT / "static" / "LineChart.js").read_text()  # PriceChart area gradient
    assert "grid + y axis (nice-number ticks)" in (ROOT / "static" / "LineChart.js").read_text()  # v2: price axis always drawn
    assert "function drawMainChart()" in js
    # the stray `s` reference in renderCalls is gone (would throw)
    assert "(s?.ladder" not in js and " s?.ladder" not in js


def test_reports_exist_and_are_linked():
    for name in ("NEWS-REPORT.md", "BRAIN-REPORT.md", "UX-ZERO-REPORT.md"):
        assert (ROOT / "docs" / name).exists(), name
    readme = (ROOT / "README.md").read_text()
    for name in ("NEWS-REPORT", "BRAIN-REPORT", "UX-ZERO-REPORT"):
        assert name in readme, name


