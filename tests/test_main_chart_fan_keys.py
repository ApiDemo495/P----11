"""Regression (user screenshot): the OWN main chart rendered blank with
a 'NaN' crosshair. Cause: _drawOwn read fan keys q90/q10 while the
engine payload carries q05/q95 — undefined fanned into the price range
and NaN-poisoned every coordinate. Quads (no fan) were unaffected."""
JS = open("static/app.js").read()


def test_main_chart_uses_real_fan_keys():
    assert "p.q95, p.q05" in JS
    assert "p.q90" not in JS and "p.q10" not in JS


def test_linechart_drops_non_finite_values():
    assert "Number.isFinite(p.p)" in JS          # points filtered
    assert "Number.isFinite(L.p)" in JS          # levels guarded
    assert "d.pts.length < 2) return" in JS      # never draws on NaN range
