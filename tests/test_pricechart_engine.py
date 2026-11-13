"""PriceChart (static/LineChart.js) replaced the deleted LineChart v2
and the hand-rolled probability canvas. Rules come from
docs/line-chart-research.md."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LC = (ROOT / "static" / "LineChart.js").read_text()
JS = (ROOT / "static" / "app.js").read_text()


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
    assert "S.tfBarsRevision[tf] = historyRevision;" in JS
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
