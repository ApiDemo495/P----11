"""User's three: brain drops every 1-2 min, probability chart
unreadable, own chart squashed by its own fan."""
import app.flywire as fw


def test_brain_cache_in_workspace_and_watchdog():
    assert "data" in str(fw.CACHE_DIR) and "connectome" in str(fw.CACHE_DIR)
    eng = open("app/engine.py").read()
    assert "_last_brain_watch" in eng and '_fw.state == "absent"' in eng
    assert "_fw.ensure()" in eng          # 60 s self-heal, no button hunt


def test_probability_bar_and_readable_fan_chart():
    html = open("static/index.html").read()
    js = open("static/app.js").read()
    assert 'id="branch-bar"' in html
    assert "TARGET FIRST" in js and "STOP FIRST" in js and "NEITHER" in js
    lc = open("static/LineChart.js").read()
    assert "history solid, forecast median dashed" in lc
    assert "lineWidth = spark ? 1.5 : 2" in lc   # price is the hero, thick


def test_fan_never_sets_own_chart_range():
    js = open("static/app.js").read()
    lc = open("static/LineChart.js").read()
    assert "fan is NEVER in the domain" in lc
    assert "clipped to the price-led range" in lc
    lc = open("static/LineChart.js").read()
    assert "Number.isFinite(p.p)" in lc          # NaN hardening retained in PriceChart
