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


def test_focus_timeframe_selector_exists():
    assert 'id="tf-bar"' in INDEX_HTML
    assert "S.focusTf" in APP_JS
    assert "TF_META" in APP_JS


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
