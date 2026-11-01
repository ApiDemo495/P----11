"""UI ↔ HTML contract (r2 rewrite): guard the new seams permanently."""
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "static"
APP_JS = (STATIC / "app.js").read_text()
INDEX_HTML = (STATIC / "index.html").read_text()


def test_no_tradingview_dependency():
    """Own canvas chart only — the TradingView lib was removed for speed."""
    assert "LightweightCharts" not in APP_JS
    assert "lightweight-charts" not in INDEX_HTML
    assert "class ChartX" in APP_JS


def test_bottom_tab_bar_and_nine_views():
    assert 'id="tabs"' in INDEX_HTML
    for v in ("dashboard", "formulas", "emotion", "physics", "news",
              "brains", "agents", "history", "settings"):
        assert f'id="view-{v}"' in INDEX_HTML, v
        assert f'data-view="{v}"' in INDEX_HTML, v


def test_boot_overlay_updated_before_content_render():
    """applySnapshot must drive the boot overlay first; a render error can
    never freeze the initial-analysis page again."""
    i = APP_JS.index("function applySnapshot")
    block = APP_JS[i:i + 400]
    assert "renderBoot" in block
    assert block.index("renderBoot") < block.index("renderQuads")
    assert "try { renderQuads" in block


def test_toolbar_toggles_assets_and_main_timeframes():
    """Global toolbar: asset segment × 4 main timeframes. Sub-windows are
    TP ladders of those4 — no separate sub-timeframe toggles anywhere."""
    assert 'id="toolbar"' in INDEX_HTML
    assert 'id="seg-assets"' in INDEX_HTML and 'id="seg-tf"' in INDEX_HTML
    assert "buildToolbar" in APP_JS and "setFocusTf" in APP_JS
    assert "tf-bar" not in INDEX_HTML and "fl-tf" not in INDEX_HTML
    # sub-timeframe toggles must not exist in any view
    assert "tf-mini" not in INDEX_HTML


def test_calls_and_targets_table():
    """TP/SL ladder for every timeframe, shown below the dashboard."""
    assert 'id="calls-panel"' in INDEX_HTML
    assert "renderCalls" in APP_JS and "TP1" in APP_JS
    assert "SL" in APP_JS


def test_why_panel_reads_breakdown():
    assert 'id="why-panel"' in INDEX_HTML
    assert "breakdown" in APP_JS and "multipliers" in APP_JS
    assert "SUBSYSTEM VOTES" in APP_JS


def test_app_js_cache_busted():
    assert 'src="/app.js?v=' in INDEX_HTML
    assert 'href="/style.css?v=' in INDEX_HTML


def test_per_tf_candle_endpoints():
    """Charts load per-timeframe aggregated candles, not one shared series."""
    assert "/api/candles?symbol=${S.symbol}&tf=${tf}" in APP_JS


def test_emotion_reasoning_and_formula_logic_present():
    assert "narrative" in APP_JS            # emotion reasoning panel
    assert "FORMULA_LOGIC" in APP_JS        # per-formula logic one-liners
    assert "P_LOGIC" in APP_JS              # per-physics-formula logic


def test_docs_route_renders_markdown_without_editor_extensions():
    """markdown.showPreview missing in the editor must never block docs:
    the app serves README + API_SETUP itself at /docs."""
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    r = c.get("/docs")
    assert r.status_code == 200
    body = r.text
    assert "README.md" in body and "API_SETUP.md" in body
    assert "<h3" in body or "<h4" in body   # headings actually rendered
    assert "back to app" in body


def test_offline_banner_and_error_trap_in_js():
    js = open("static/app.js").read()
    assert "_netBanner" in js                 # engine-offline detection
    assert 'window.addEventListener("error"' in js
    assert "unhandledrejection" in js


def test_real_drosophila_brain_dataset():
    """The brain system must load the real Janelia hemibrain v1.2 data."""
    from pathlib import Path as _P
    root = _P(__file__).resolve().parent.parent
    import json as _json
    br = _json.load(open(root / "data" / "brain_real.json"))
    assert br["stats"]["neurons"] == 21739
    assert br["stats"]["synapses"] == 14329229
    hubs = [h["type"] for h in br["hubs"]]
    assert "APL" in hubs and "DPM" in hubs
    assert (root / "scripts" / "fetch_real_brain.py").exists()
    assert "hemibrain" in APP_JS or "hemibrain" in (root / "app" / "brain.py").read_text()
