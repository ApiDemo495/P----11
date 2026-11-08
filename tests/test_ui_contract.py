"""UI ↔ HTML contract (r2 rewrite): guard the new seams permanently."""
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "static"
APP_JS = (STATIC / "app.js").read_text()
INDEX_HTML = (STATIC / "index.html").read_text()


def test_no_tradingview_dependency():
    """Three sources, one toggle: Binance candles (vendored lib, Binance
    klines), TradingView iframe (TV mode), own PriceChart. No hero."""
    assert "LightweightCharts" in APP_JS and "bnMake" in APP_JS
    assert "lightweight-charts" in INDEX_HTML
    assert "class PriceChart" in (STATIC / "LineChart.js").read_text() and "hero-chart" not in APP_JS


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
    block = APP_JS[i:i + 700]
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
    """Charts load ONE dense minute series for every TF card.

    (Superseded contract: they used to fetch per-TF aggregated candles —
    tf2/3/4 then had only 4-5 points and rendered as smooth fake curves.
    The user asked for real-market zigzag precision, so the raw 1-minute
    tape with high/low wicks drives all four cards; TP/lock overlays
    remain per-TF. The endpoint itself still accepts tf= aggregation.)"""
    assert "/api/candles?symbol=${S.symbol}&tf=tf1&n=260" in APP_JS
    assert "main.pts[main.pts.length - 1]" in (STATIC / "LineChart.js").read_text()  # PriceChart engine
    # every TF chart still receives data in the loop
    assert "S.charts[tf].setData" in APP_JS or "c.setData(" in APP_JS


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
    # dataset ships in assets/ (tracked); data/ copy is a runtime fallback
    _ds = next((root / d / "brain_real.json" for d in ("assets", "data")
                if (root / d / "brain_real.json").exists()), None)
    assert _ds is not None, "brain_real.json missing from assets/ and data/"
    br = _json.load(open(_ds))
    assert br["stats"]["neurons"] == 21739
    assert br["stats"]["synapses"] == 14329229
    hubs = [h["type"] for h in br["hubs"]]
    assert "APL" in hubs and "DPM" in hubs
    assert (root / "scripts" / "fetch_real_brain.py").exists()
    assert "hemibrain" in APP_JS or "hemibrain" in (root / "app" / "brain.py").read_text()


def test_markdown_preview_fixed_all_paths():
    """Editor shows 'markdown.showPreview not found' (no built-in markdown).
    Fix: recommend + auto-install Markdown Preview Enhanced, which bundles
    its OWN preview engine, and keep the zero-install /docs viewer."""
    import json as _json
    from pathlib import Path as _P
    root = _P(__file__).resolve().parent.parent
    recs = _json.load(open(root / ".vscode" / "extensions.json"))["recommendations"]
    assert "shd101wyy.markdown-preview-enhanced" in recs
    dc = _json.load(open(root / ".devcontainer" / "devcontainer.json"))
    dc_exts = dc["customizations"]["vscode"]["extensions"]
    assert "shd101wyy.markdown-preview-enhanced" in dc_exts
    start = (root / "start.py").read_text()
    assert "markdown-preview-enhanced" in start and "--install-extension" in start
    # in-app viewer still the zero-dependency path
    from fastapi.testclient import TestClient
    from app.main import app
    assert TestClient(app).get("/docs").status_code == 200


def test_docs_html_zero_extension_viewer():
    """docs.html must exist, be self-contained, and render the docs —
    the fix for editors with NO built-in markdown."""
    from pathlib import Path as _P
    root = _P(__file__).resolve().parent.parent
    h = (root / "docs.html").read_text(encoding="utf-8")
    assert "PREDICT · docs" in h
    assert "README.md" in h and "API_SETUP.md" in h
    assert "<h3" in h or "<h2" in h          # headings really rendered
    assert "markdown.showPreview" in h       # banner explains the fix
    assert (root / "scripts" / "make_docs_html.py").exists()



def test_setup_is_fast_by_default_ai_opt_in():
    """The 10-20 min llama-cpp source build + model downloads must be
    OFF by default so Codespaces boots in ~1 min; provisioning lives
    in start.py and installs core requirements only."""
    from pathlib import Path as _P
    root = _P(__file__).resolve().parent.parent
    start = (root / "start.py").read_text()
    assert "requirements.txt" in start and "requirements-agents" not in start
    assert "--retries" in start and "make_docs_html" in start

