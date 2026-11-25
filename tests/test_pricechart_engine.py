"""PriceChart (static/LineChart.js) replaced the deleted LineChart v2
and the hand-rolled probability canvas. Rules come from
docs/line-chart-research.md."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LC = (ROOT / "static" / "LineChart.js").read_text()
JS = (ROOT / "static" / "app.js").read_text()
API = (ROOT / "app" / "main.py").read_text()


def test_old_charts_deleted_new_engine_everywhere():
    assert "class LineChart" not in JS and "class PriceChart" in LC
    assert "new PriceChart" in JS                      # main + quads + spark
    assert 'src="/LineChart.js' in (ROOT / "static" / "index.html").read_text()
    assert "XF = f =>" not in JS                       # old fan canvas gone


def test_research_rules_implemented():
    assert "pcTickStep" in LC and "1/2/5" in LC        # Heckbert nice ticks
    assert "pcDecimals(yStep)" in LC                   # precision from step
    assert "pcLTTB" in LC                              # shape-preserving downsample
    assert "3 * medianDt" in LC                        # gaps break the path
    assert "fan is NEVER in the domain" in LC          # fan clipped, price leads
    assert "magnet" in LC.lower()                      # crosshair snaps to real points
    assert 'setAttribute("role", "img")' in LC and "aria-label" in LC
    assert "devicePixelRatio" in LC                    # HiDPI
    assert "VOL " in LC                                # volume sub-pane labelled


def test_contract_levels_scale_chart_but_rungs_stay_annotations():
    assert "if (L.contract)" in LC
    assert "non-contract rungs and LIQ marks remain" in LC
    assert "offScale" in LC and "edge tag" in LC       # non-contract far levels stay pinned
    assert "function quadChart" in JS and "handleScroll: false" in JS  # doc C-5 mini candle cards
    assert "q-spark" not in JS                         # red noise line deleted (doc C-5)
    assert "Math.min(12" in LC                         # sparse volume bars capped


def test_own_chart_live_candle_never_freezes():
    # regression (user screenshot): candles fell behind the live price
    # because setData keyed on the cycle only and bucket rolls skipped
    assert "dataKey" in JS and "bars[bars.length - 1].time" in JS
    assert "inst.last = { time: bt, open: inst.last.close" in JS
    assert "Own.rangeKey" in JS          # reframe only on window change


def test_candle_fetches_reject_stale_symbol_and_key_history_revision():
    # A slower request for the previously selected asset must never populate
    # the current asset; same-time/same-length feed replacements still redraw.
    assert "const symbol = S.symbol;" in JS
    assert "S._candleLoadSeq = seq;" in JS
    assert "seq !== S._candleLoadSeq || symbol !== S.symbol" in JS
    # per-TF server charts: a card bound to /api/tf-charts keys on the
    # server's revision, otherwise on the local history revision
    assert "S.tfBarsRevision[tf] = useSrv ? String(srv.revision" in JS
    assert ": historyRevision;" in JS
    assert "const sourceRevision =" in JS and "S.tfBarsRevision?.[S.focusTf]" in JS
    assert "String(md.revision ||" in JS and "candleRevisionFallback(md.bars)" in JS


def test_probability_chart_uses_shared_engine():
    assert "cvHost._pc.setModel" in JS
    assert "lo: p2.q05, lo2: p2.q25, mid: p2.q50, hi2: p2.q75, hi: p2.q95" in JS  # nested fan bands
    assert "TARGET FIRST" in JS                        # one-glance bar kept


def test_one_price_truth_and_current_bucket():
    # snapshots sync S.prices (REST lulls must not leave the chart on
    # a stale tick price); the chart forces the current bucket to the
    # snapshot price and drops future candles
    assert "the 1 Hz tick stream is the live price truth" in JS
    assert "const liveP = (S.prices?.[S.symbol] > 0) ? S.prices[S.symbol] : a.price;" in JS
    assert "b2.time <= bt0" in JS and "lb.close = liveP" in JS
    assert "S.priceHist.push({ t: serverNow(), p: _pp })" in JS
    assert "if (bt > inst.last.time)" in JS        # quad bars advance too


def test_own_chart_keeps_at_least_30_real_candles():
    # §6 warmup: boot asks for the whole 7-day backfill window and the
    # server serves up to BACKFILL_BARS — the old 1000-bar cap starved
    # the chart to 16 h (WARMING_UP / n=0 dead states)
    assert "&tf=tf1&n=10080" in JS
    assert "n = max(10, min(BACKFILL_BARS, n))" in API
    assert "Math.max(4 * windowMin, 30 * intervalMin" in JS
    assert 'tf === "tf1" ? 90 * intervalMin : 0' in JS
    # the view window is a LOGICAL bar range now: setVisibleRange(frome/to)
    # compressed the candles into the left of the canvas whenever the tape
    # or the cycle end did not sit where the time window expected
    assert "setVisibleLogicalRange" in JS
    assert "const futureBars = Math.max(2," in JS
    # 2026-11-26: the banner must distinguish MISSING data from SLOW load -
    # "history warming" was printed for a genuinely empty tape too, so users
    # could not tell that no bars were coming.
    assert "no source bars for" in JS
    assert "history partial: ${tapeArr.length}/${minVisibleBars}" in JS
    assert "notes.push(`history warming" not in JS


def test_own_chart_shows_existing_median_forecast_by_default():
    assert "The model's q50 median estimate is always visible" in JS
    assert "inst.fan50.setData(median.length > 1 ? median : [])" in JS
    assert 'inst.fan25.setData(fanOn ? series("q25") : [])' in JS
    assert "MODEL ESTIMATE · q50" in JS
    assert 'const source = [method, basis].filter(Boolean).join(" · ");' in JS
    assert "not a promise" in JS
    assert "forecast unavailable · insufficient data" in JS
