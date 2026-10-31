"""UI ↔ HTML contract: selector mismatches froze the 4:00 overlay once —
guard the seam permanently (see commit history: q-agent-tf- vs q-agent-)."""
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "static"
APP_JS = (STATIC / "app.js").read_text()
INDEX_HTML = (STATIC / "static" if False else STATIC / "index.html").read_text()


def test_quadrant_agent_selector_matches_html_id():
    # html: <span id="q-agent-tf1"> … js must query #q-agent- + tf
    assert 'id="q-agent-tf1"' in INDEX_HTML
    assert '$("#q-agent-" + tf)' in APP_JS
    assert '$("#q-agent-tf-" + tf)' not in APP_JS


def test_overlay_updates_before_dashboard_render():
    # applySnapshot + WS snapshot path must update the overlay FIRST and
    # swallow renderDashboard errors — a quadrant bug can never freeze 4:00
    i_apply = APP_JS.index("function applySnapshot")
    block = APP_JS[i_apply:i_apply + 400]
    assert block.index("updateOverlay") < block.index("renderDashboard")
    assert "try { renderDashboard" in block
    i_snap = APP_JS.index('case "snapshot":')
    ws_block = APP_JS[i_snap:i_snap + 500]
    assert ws_block.index("updateOverlay") < ws_block.index("renderDashboard")


def test_app_js_cache_busted():
    assert 'src="/app.js?v=' in INDEX_HTML
    assert 'href="/style.css?v=' in INDEX_HTML


def test_every_dynamic_quadrant_id_exists_in_html_or_is_self_created():
    # ids the JS builds dynamically must exist in index.html
    for tf in ("tf1", "tf2", "tf3", "tf4"):
        for stem in ("q-", "dir-", "ratio-", "tplist-", "gauge-", "q-cci-", "q-agent-"):
            assert f'id="{stem}{tf}"' in INDEX_HTML, f"missing {stem}{tf}"
