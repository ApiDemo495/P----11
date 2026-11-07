"""Hero line-chart-from-now + visible fan (user: candles 72-75k while
price >76k; branches chart showed only white/green/red lines).

The hero is a line chart: realised path up to a NOW divider, forward
fan for the remaining window, TP/SL as future barriers, last point ==
live price. The fan sigma is floored by the measured block sigma so
the cone is actually visible instead of gluing to the price line.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_fan_sigma_floored_by_measured_block():
    src = (ROOT / "app" / "engine.py").read_text()
    i = src.index("def branch_stats")
    seg = src[i:i + 14000]
    assert "sigma_bps" in seg and "price * _sig_bps / 1e4" in seg


def test_own_main_chart_is_line_from_now():
    """docs11-charts Part C supersedes the line-from-now hero: the OWN
    main chart is server-tape candlesticks (Lightweight Charts) with
    entry/TP/SL price lines and lock markers."""
    js = (ROOT / "static" / "app.js").read_text()
    assert "function drawMainChart()" in js and "function drawOwnChart()" in js
    assert "addCandlestickSeries" in js and "createPriceLine" in js
    assert "setMarkers" in js and "LOCK" in js
    assert "hero-chart" not in js and "drawHeroChart" not in js  # hero removed (user order)


def test_pricehist_seeded_from_candles():
    js = (ROOT / "static" / "app.js").read_text()
    assert "seed the realised-path history" in js
    assert "S.priceHist = [...seeded, ...tail].slice(-900)" in js
