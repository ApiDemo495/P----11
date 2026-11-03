# NEWS REBUILD — report (docs11-news.md §10 format)

Scope: `docs11-news.md` (Drosophila Trader v2.0 news component), adapted to
this app's four assets (BTC / PAXG / ETH / BNB) and 22/44/130/247-minute
windows as the source document itself instructs. Acceptance tests:
`tests/test_news_docs11.py` (12/12), full suite **211 passed**.

## What changed vs the advice

- **Poll → evaluate split.** The poller no longer "decides anything".
  Ingest is rate-limited fetching; evaluation is pure arithmetic
  (`NewsSystem.evaluate`) and runs every 4 s from `_news_ticker` plus on
  demand. The ticker publishes the `NEWS` WS frame (coverage, four
  horizons' impact/NIV/SMD, top theme, critical count) with REST mirrors
  at `/api/news/impact` and `/api/news/status`.
- **Per-horizon arithmetic added beside the existing Section-7 aggregate.**
  Half-lives 11/22/65/123 min (W/2), decay 0.5^(age/half-life), impact =
  tanh(Σ credibility·decay·novelty·magnitude·impact_prior·relevance),
  NIV with τ_h = W/4 over the five newest relevant items, SMD =
  tanh(NIV − tape acceleration). The original `tf_impact` + Section-7
  `aggregate` are untouched (both are user-spec'd) — the new indices live
  in `evaluate()`/`impact()` and the four horizon chips.
- **Relevance gate is now enforced at ingest** (docs §4.1): an item that is
  0-relevance for all four assets is dropped **and counted**
  (`filtered`, three examples, `relevance_dropped` in the aggregate) —
  never shown as a zero-impact row. Tiers: direct 1.0, macro 0.7, geo 0.6
  (geo keywords added: war/strike/missile/sanction/tariff/blockade/…).
- **Novelty/habituation added**: Jaccard ≥ 0.6 near-duplicates weigh 0,
  the k-th distinct headline on a theme weighs 1/k, a continuously-present
  theme halves every 2 h and resets after a 90-minute gap. Chronological
  (only earlier arrivals habituate later ones).
- **Emergency re-lock implemented** (`engine.emergency_relock`): the only
  path that mutates an already-locked entry — re-opens at the current
  price, keeps the original entry in the reason (`original_price`,
  `relocks`, `relock_at`), then halts all new locks for 30 s. Manual/drill
  via `POST /api/news/emergency`; auto via critical-headline + NIV-swing
  trigger in the ticker. Rate guard: the 7th re-lock inside one hour is
  rejected with `N-8: … bug, not a signal` and logged to `stats_bugs`.
- **Ingest hardening (docs §3)**: key ring of up to 3 slots
  (`NEWSAPI_KEYS` env / `data/keyring.json` / legacy key), free=900 s vs
  paid=60 s cadence with auto plan detection from `X-RateLimit-Limit`
  (≤100 → free), 429 → exponential backoff 60→3000 s honouring
  `X-RateLimit-Reset` + key rotation, 426 → plan-limited cooldown,
  401/403 → key rotation, named browser-style `User-Agent`, conditional
  GET (`If-None-Match` / `If-Modified-Since`, 304 counted as a poll).
- **Coverage header** (`coverage_string`) and per-source lines
  (`source_lines`: tier, state, backoff level, 429 drops) feed the news
  card; SEED/SYNTHETIC items are labelled and never counted as live.

## What is new

- `app/themes.json` — 14 themes × per-asset impact priors (+ magnitude,
  severity, hot flags). Equity/USD columns from the doc's table were
  folded into explicit values for this app's four assets (`_note` field).
- Item record fields per docs §4.1: tier, credibility,
  relevance_by_asset, theme, theme_k, magnitude, impacts, synthetic.
- `evaluate()` / `impact()` / `set_tape_accel()` / `critical()` /
  `coverage_string()` / `source_lines()` / `record_emergency()`.
- Endpoints: `GET /api/news/feed`, `GET /api/news/impact`,
  enriched `GET /api/news/status`, upgraded `POST /api/news/poll`,
  `POST /api/news/emergency`.
- News card UI: coverage line + plan/key/poll header, source lines with
  backoff, four horizon chips (impact/NIV/SMD), NIV 90-s trace canvas,
  dropped-topic examples, emergency history, ⚡ drill button; WS `NEWS`
  ticker keeps chips live; `news_emergency` events toast.
- `tests/test_news_docs11.py` — the 12 acceptance tests from §8, written
  against this implementation.

## What is skipped and why

- **Nothing from §1–§10 is dropped.** Two adaptations, both with reasons:
  1. *Equity / USD columns* in the theme table: this app trades
     BTC/PAXG/ETH/BNB only — values were transcribed explicitly per asset
     instead of an unusable column.
  2. *NewsAPI pricing* ("read at build time"): the sandbox cannot reach
     newsapi.org, so the price note is carried from the spec text
     (free 100 req/day, paid ≈ $449/mo) and flagged as spec-derived in
     `/api/news/status.price_note`.
- Wire tests run against the local fixture/sim path; a real NewsAPI key
  is optional (keyless = core never blocks, per standing rule).

## Limitations

- Habituation halving (2 h) and the 90-min theme gap are wall-clock
  mechanisms: in a quiet sim stream a theme rarely reaches k>2, so the
  1/k curve is proven by tests rather than seen live.
- `SMD` uses tape acceleration from the last refresh; during boot grace
  (sim-only tape) it is near 0 by construction.
- NIV history for the swing trigger is per-process (90-s window); it does
  not survive restarts.
- 429 quota-window detection depends on NewsAPI exposing
  `X-RateLimit-*`; without headers the backoff ladder (60→3000 s) is the
  fallback.

## Numbers that moved

- Full suite: **199 → 211 passed** (12 new acceptance tests).
- Relevance gate: unrelated headlines now counted (e.g. fixture wire of 6
  → 4 kept, 2 counted with examples) instead of silently entering the
  feed.
- Emergency path: original entry preserved verbatim in the lock reason;
  30-s halt verified by test; 7th-in-hour rejected (N-8).

## Three things still weak

1. **Single-wire ingest**: only NewsAPI is wired as a real HTTP source;
   the docs' multi-wire (Reuters/CoinDesk/AP…) arrives through tier
   classification of whatever the API returns, not separate per-desk
   pollers with their own cadences.
2. **Emergency auto-trigger threshold** (NIV swing > 0.02 over 90 s) is a
   chosen constant, not a calibrated one — it fires rarely by design, but
   it has not been tuned against real volatility.
3. **Theme prior table is hand-authored**: `app/themes.json` encodes the
   doc's economics (war → gold up, hack → majors down, ETF → majors up)
   with stated magnitudes; there is no feedback loop yet that measures
   realized impact vs prior and corrects it.
