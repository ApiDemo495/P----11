"""Chart toggle (OWN ⇄ TradingView) applies to the main chart AND every
sub-timeframe card, each with its own Binance interval."""
JS = open("static/app.js").read()
HTML = open("static/index.html").read()


def test_toggle_covers_all_timeframes():
    assert 'id="seg-chart"' in HTML and 'data-chart="tv"' in HTML
    assert 'id="qtv-${tf}"' in JS and 'id="qbn-${tf}"' in JS  # per-card TV + Binance slots
    assert 'id="qc-${tf}"' in JS                     # own chart slot kept
    assert 'fill(slot, tf)' in JS and 'bnEnsure' in JS  # TV per-TF; Binance lazy-built
    assert 'BINANCE:BTCUSDT' in JS and 'TV_INT' in JS
    assert '"tf1": "5"' not in JS or 'tf1: "5"' in JS  # interval map exists
    assert 'grid.style.display = tv' not in JS       # grid no longer hidden in TV mode


def test_toggle_persisted_and_reapplied():
    assert 'localStorage.setItem("chartSrc"' in JS
    assert JS.count("applyChartSrc()") >= 3          # bind + symbol + focus-TF


def test_three_modes_and_no_hero():
    assert 'data-chart="binance"' in HTML and 'data-chart="own"' in HTML
    assert 'id="chart-binance"' in HTML and 'id="hero-chart"' not in HTML
    assert "drawHeroChart" not in JS and "bnFeed" in JS and "addCandlestickSeries" in JS
    assert "/api/candles?symbol=${S.symbol}&tf=${tf}&n=180" in JS
