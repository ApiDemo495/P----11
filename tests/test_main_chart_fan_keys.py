"""Regression (user screenshot): the OWN main chart rendered blank with
a 'NaN' crosshair. Cause: _drawOwn read fan keys q90/q10 while the
engine payload carries q05/q95 — undefined fanned into the price range
and NaN-poisoned every coordinate. Quads (no fan) were unaffected."""
JS = open("static/app.js").read()
LC = open("static/LineChart.js").read()


def test_main_chart_uses_real_fan_keys():
    assert "series(\"q50\")" in JS and "lo: p2.q05" in JS   # fan keys q05..q95 in both charts
    assert "p.q90" not in JS and "p.q10" not in JS


def test_linechart_drops_non_finite_values():
    assert "Number.isFinite(p.p)" in LC          # points filtered
    assert "Number.isFinite(L.p)" in LC          # levels guarded
    assert "main.pts.length < 2" in LC           # never draws on NaN range
