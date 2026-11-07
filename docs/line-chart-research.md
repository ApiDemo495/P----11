# How to build a proper line chart — research notes

Written 2026-11-07 before deleting the old `LineChart` (app.js v2) and the
hand-rolled probability/fan canvas, and rebuilding both as one new
component: `static/LineChart.js` (`PriceChart`).

Method: broad survey of the data-visualisation and financial-charting
literature — style guides (FT, Datawrapper, UK Civil Service, Statistics
Canada), Tufte scholarship, charting-library documentation (TradingView
Lightweight Charts, D3, Chart.js, uPlot ecosystem writing), the Bank of
England fan-chart tradition, accessibility/palette research (Okabe–Ito /
Color Universal Design, WCAG), and rendering/performance writing
(LTTB downsampling, HiDPI canvas). ~25 sources were read in depth; the
rules below are the points those sources agree on, with the disagreements
called out.

## The rules the sources agree on

### 1. The line is the data; everything else is scaffolding
- Maximise the **data-ink ratio** (Tufte): remove any ink that does not
  carry data. Gridlines thin and faint, behind the data; no boxes,
  shadows, 3D, or heavy axis frames. (Performance Magazine's summary is
  a useful corrective: tick *labels* and a light axis measurably help,
  so "erase" applies to decoration, not to reading aids.)
- One series = one strong stroke. The price line is drawn **last and
  thickest** (~2 px), everything auxiliary (moving averages, fans,
  levels) is thinner/lighter. Lines are ≤ ~4 before the chart becomes
  spaghetti (Useful Data Tips; numberanalytics).
- End the line with a **dot on the last point** — Tufte's sparkline
  convention; it anchors "where we are now" (Tufte sparkline literature).
- Show **gaps as gaps**: when data is missing, break the path instead of
  interpolating a lie (Statistics Canada; Useful Data Tips).

### 2. Axes: nice numbers, honest ranges, adaptive precision
- Y tick values come from a **"nice number" algorithm** (Heckbert 1990,
  as used by D3's `ticks()`/`nice()`): steps of 1/2/5 × 10^k, domain
  snapped outward to tick boundaries. Arbitrary 4-equal-slices axes
  (what the old chart did) produce unreadable labels like 76,013.37.
- Decimal precision of labels is **derived from the tick step**, not from
  the asset: a 25-unit step gets 0 decimals, a 0.25 step gets 2. Same
  rule D3's `tickFormat` applies.
- **Zero baseline: not required for line/price charts.** Bar charts must
  start at zero; for line charts the consensus (Quartz, Chad Skelton's
  survey, Useful Data Tips, every trading terminal) is a truncated axis
  is the norm for prices — position and slope stay honest — *provided*
  the axis labels are clear. Extending BTC to zero would erase all
  signal. Our range therefore fits series + labelled levels, with
  ~6 % headroom, and never lets an auxiliary element (the forecast fan)
  set the range.
- X (time) ticks come from a **step ladder** (1s, 5s, 15s, 1m, 5m, 15m,
  1h, 6h, 1d …) chosen so ~4–6 labels fit; format adapts to span
  (HH:MM:SS under a few minutes, HH:MM intraday, "7 Nov 12:00" beyond a
  day). Never label every point (Infogram; Datawrapper Academy).
- Labels ≥ 10 px, muted colour, right-aligned price scale on the right
  (financial convention: Binance, TradingView).

### 3. Label directly, don't make the reader commute
- **Direct labelling beats legends** (FT's Burn-Murdoch; UK Civil
  Service analysis function; Useful Data Tips rule 4; graffiy's review):
  TP/SL/ENTRY tags sit *on* their lines, the last price sits *on* the
  axis, overlay lines get end labels. WCAG adds a hard reason: colour
  alone must not carry meaning — a text tag is also the accessible path.
- Titles/labels carry only what the reader needs; selective annotation
  of extremes (H/L) is worth its ink.

### 4. Colour: conventional, restrained, colour-blind safe
- Financial convention: green up `#26a69a`, red down `#ef5350`
  (TradingView defaults, Binance). One accent series; context in grey.
- Where categorical colour is needed, **Okabe–Ito** (Color Universal
  Design; Wong, *Nature Methods*) stays separable under deuteranopia,
  protanopia and tritanopia; never rely on hue alone (add dash/label).
- Backgrounds and fills stay subtle: area gradient under the line at
  low alpha (Binance area-chart look), gridlines ≈ 8–14 % alpha.

### 5. Interaction: crosshair with magnet snap + readout
- TradingView Lightweight Charts' documented model: crosshair lines on
  both axes, **Magnet mode** snapping to the nearest data point, and
  axis labels that follow the crosshair. A free-floating readout of
  interpolated fantasy prices is worse than snapping to a real point.
- Tooltip/readout shows time, price, change vs. the visible start, and
  volume when present — next to the point, not in a distant legend.
- Touch: pointer events (mouse+touch unified), generous hit behaviour,
  redraw on resize; no hover-only information that mobile can't reach
  (finance-UX literature: 48 px touch targets for controls; charts are
  read-first on small screens, axes get *fewer* ticks, not smaller).

### 6. Financial-chart anatomy (Binance / TradingView)
- Price pane on top, **volume pane below at ~15–20 % height**, volume
  bars coloured by bar direction at ~50 % alpha (Binance app; Sencha
  and Medium finance-UX write-ups: volume belongs in its own pane
  because it has a different scale — it must never compress price).
- **Last-price line + axis chip**: a dashed horizontal at the last
  price with a filled tag on the price scale (Lightweight Charts'
  `priceLineVisible` + `lastValueVisible`, default-on for a reason).
- **Autoscale to the visible range** with margins (Lightweight Charts
  `autoscaleInfoProvider` margins), and price-format precision derived
  from the instrument/step.
- Moving-average overlay in a muted contrasting stroke (white/grey),
  thinner than price.

### 7. Forecast / probability (fan) charts: the Bank of England rules
From the fan-chart literature (Bank of England 1996 → IMF/ECB standard;
metricgate, visualizing.org, Highcharts, usefuldatatips):
- History is a **solid line**; the forecast region is visually distinct
  (divider at "now", median path **dashed**), because solid = fact.
- **Nested bands** (dark inner → light outer) encode where probability
  mass concentrates; ≤ 3 bands or it turns to mush. Bands must be
  labelled in plain language ("range of likely outcomes"), never left
  as unexplained shading.
- The fan **widens with horizon**; a constant-width band is a lie.
- Always show the historical line behind the fan so plausibility can be
  judged; state n and basis in text near the chart.
- Known failure mode relevant to us: if the fan's extremes set the
  y-range, the realised line is crushed flat — the fan is **clipped to
  the price-led range**, never the other way round.

### 8. Rendering & performance engineering
- **HiDPI**: canvas backing store = CSS size × devicePixelRatio (cap 2),
  context scaled — the standard fix for blurry canvas lines (MDN-derived
  Stack Overflow consensus; cmdcolin's analysis).
- **Downsampling**: more points than pixels is waste; the standard is
  **LTTB** (Largest-Triangle-Three-Buckets) which preserves peaks and
  shape (Steinarsson; used by Grafana/Plotly-resampler) — target ≈ the
  pixel width of the plot.
- Redraw via `requestAnimationFrame` coalescing; keep per-frame work
  O(points on screen).
- Round line joins/caps so dense 1-bar-per-pixel segments don't spike.

### 9. Accessibility
- Canvas gets `role="img"` and an `aria-label` summarising the series
  (first/last/min/max/last time) — the chart must say something to a
  screen reader; fans specifically need a text summary of the range
  (visualizing.org's fan-chart guidance).
- Text contrast: axis labels muted but ≥ ~4.5:1 on the background;
  meaning never by colour alone (tags + dashes on levels).

## What was wrong with the old charts (audit before deletion)
1. Y axis: 4 arbitrary slices of a padded range → non-round labels.
2. Fixed 9 px type, labels colliding with level tags.
3. Fan/levels/volume behaviours accreted as patches across three
   rewrites; two separate hand-rolled canvases (main + branches)
   implemented the same ideas inconsistently.
4. Crosshair reported an interpolated price at the cursor rather than
   snapping to a real point; no change/volume in the readout.
5. No LTTB/downsampling, no gap-breaking, no ARIA summary.
6. Probability chart duplicated ~130 lines of canvas code with its own
   scales, its own axis math, its own bugs.

## The new component (static/LineChart.js — `PriceChart`)
One engine implementing the rules above, used by: the OWN main chart,
the four sub-timeframe cards, the sparklines, and the probability
(history + measured fan) chart. API:

```js
const ch = new PriceChart(el, { spark: false });
ch.setData(bars)            // OHLC-ish bars → close line + volume
ch.tick(price)              // live tick append/update
ch.setModel({               // full control (main + probability charts)
  series: [{ pts: [{t, p, v}], color?, width? }],
  overlays: [{ pts, color, label }],       // e.g. moving average
  levels: [{ p, label, color, dash }],
  fan: { originT, pts: [{t, lo, mid, hi}] },
  nowT, t0, t1, emptyText
});
ch.setLines(levels)         // levels-only update (quad cards)
```

Key internals: Heckbert nice ticks + step-derived precision; time-step
ladder; fan excluded from the domain and clipped; magnet crosshair with
tooltip; LTTB downsample to plot width; gap-breaking; volume sub-pane;
last-price dashed line + axis tag; direct level tags; `role="img"`
ARIA summary; sparkline mode keeps only line + end dot (Tufte).

## Sources (read in depth)
1. Statistics Canada — *Data Visualization: Best Practices* (line charts,
   ticks, gridlines, shading uncertainty)
   https://www150.statcan.gc.ca/n1/pub/89-26-0005/892600052022001-eng.htm
2. Infogram — *Do This, Not That: Line Charts* (even axes, selective text)
   https://infogram.com/blog/do-this-not-that-line-charts/
3. Useful Data Tips — *Line Chart Best Practices: 7 Rules* and
   *Showing Uncertainty: Error Bars, Bands, Fan Charts*
   https://usefuldatatips.com/tips/visualization/line-chart-best-practices
4. UK Civil Service Analysis Function — *Line charts* (direct labelling,
   WCAG colour-alone rule)
   https://analysisfunction.civilservice.gov.uk/support/communicating-analysis/introduction-to-data-visualisation-e-learning/module-8-line-charts/
5. Financial Times — *Visual Vocabulary* (chart choice; FT chart style)
   https://github.com/Financial-Times/chart-doctor/blob/main/visual-vocabulary/README.md
6. iMeDD Lab — Burn-Murdoch (FT): direct labels, strategic colour,
   annotation https://lab.imedd.org/en/from-data-to-storytelling-concept-and-design-tips-from-the-financial-times-john-burn-murdoch
7. Datawrapper Academy — *How to create a line chart* / *Customizing
   your multiple lines chart* (custom ticks, range rounding, grid choice)
   https://academy.datawrapper.de/article/23-how-to-create-a-line-chart
8. Quartz — *It's OK not to start your y-axis at zero* (stock-chart
   convention) https://qz.com/418083/its-ok-not-to-start-your-y-axis-at-zero
9. Chad Skelton — *Bar charts should always start at zero. But what
   about line charts?* http://www.chadskelton.com/2018/06/bar-charts-should-always-start-at-zero.html
10. Heckbert — *Nice Numbers for Graph Labels* (Graphics Gems) via
    PharmaSUG-2019-DV-005 and the Stack Overflow implementation thread
    https://www.lexjansen.com/pharmasug/2019/DV/PharmaSUG-2019-DV-005.pdf
11. D3 axis/tick formatting tutorials (scale.ticks, tickFormat precision)
    https://library.fridoverweij.com/docs/d3tutorial/axes.html
12. TradingView Lightweight Charts — *Crosshair* tutorial (magnet mode)
    https://tradingview.github.io/lightweight-charts/tutorials/customization/crosshair
13. Lightweight Charts API — SeriesOptionsCommon (lastValueVisible,
    priceLineVisible, autoscaleInfoProvider margins, priceFormat)
    https://tradingview.github.io/lightweight-charts/docs/api/interfaces/SeriesOptionsCommon
14. Metricgate — *Fan Chart (Forecast Uncertainty)* (50/80/95 nested
    bands, Bank of England) https://metricgate.com/docs/fan-chart-uncertainty/
15. visualizing.org — *Fan Chart: definition, best practices* (≤3–4
    bands, keep history visible, ARIA fallbacks)
    https://www.visualizing.org/fan-chart
16. Highcharts — *Predictive analytics visualization* (solid history vs
    dashed forecast, temporal markers)
    https://www.highcharts.com/blog/use-cases/predictive-analytics-visualization/
17. Okabe-Ito / Color Universal Design palette references (Wong 2011)
    https://figcanvas.com/blog/okabe-ito-palette-hex-rgb-codes and
    https://thenode.biologists.com/data-visualization-with-flying-colors/research/
18. Tufte sparkline literature (end-point dot, word-sized graphics,
    final-value labels) https://www.line-graph-maker.com/resources/line-charts/sparkline-matrix
19. Performance Magazine — *How data-ink ratio imposed minimalism*
    (evidence-based caveats on erasing ticks/axes)
    https://www.performancemagazine.org/data-ink-ratio-minimalism-data-visualization/
20. LTTB downsampling (Steinarsson) explainers — shape-preserving,
    target ≈ pixel width https://www.pythoncharts.com/python/downsample-time-series-python/
21. HiDPI canvas rendering (devicePixelRatio scaling)
    https://stackoverflow.com/questions/8696631/canvas-drawings-like-lines-are-blurry
22. UX Stack Exchange — *How to optimize candlestick charts* (subtle
    gridlines, whitespace, data-ink in finance)
    https://ux.stackexchange.com/questions/66968/how-to-optimize-candlestick-charts-from-finance
23. Finance-app chart UX (volume in own pane, hover values, simplicity)
    https://medium.com/@extej/extej/ux-tips-for-enhancing-the-usability-of-finance-app-charts-and-graphs-0843d723b57f
24. Binance chart reading guides (price pane + volume below, last-price
    tag on the axis) https://wickread.com/en/chart/read-on-binance/
25. Storytelling with Charts / Number Analytics — line-chart craft
    (2–3 series max, markers/grid in moderation, typography 8–12 pt)
    https://www.storytellingwithcharts.com/blog/best-practices-for-line-charts-in-financial-presentations/
