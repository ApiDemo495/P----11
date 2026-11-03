# API Setup Guide

## 1. Gemini API (Primary Agent)

1. Go to https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Select your Google Cloud project (or create a new one)
4. Copy the generated API key
5. Paste it into the Settings tab → Gemini API Key field
6. Free tier: 15 requests per minute, 1 million tokens per day
7. The app uses approximately 15 requests per hour (one per GRC), well within free tier limits

## 2. Brain data — FlyWire v783 (automatic, no key) + optional neuPrint

**No key, no action needed.** The brain is the real FlyWire v783
connectome: `Connectivity_783.parquet` (~96 MB) + neuron annotations
(~30 MB) are downloaded once from GitHub (with resume, size checks and a
curl fallback), built into `flywire_783.npz` in a worker process, and
cached outside the repo:

```
~/.cache/drosophila-trader/connectome/     (override: $CONNECTOME_CACHE)
  Connectivity_783.parquet
  Supplemental_file1_neuron_annotations.tsv
  flywire_783.npz          ~192 MB, BUILD_VERSION inside — never committed
```

- First boot starts with the brain **absent** (fusion weight 0, both
  brain formulas say `brain absent`) and hot-swaps in when the build
  finishes — first prediction is never blocked.
- `POST /api/brain/reconnect` re-runs the five-step verification and
  restarts a failed build; `GET /api/brain/status` shows each step,
  durations, counts and the chip.
- If a cached npz is from an older build (`BUILD_VERSION` mismatch) it
  is rebuilt automatically (error B-8).

**Optional neuPrint** (`NEUPRINT_URL` + `NEUPRINT_TOKEN`, Settings tab):
extra queries only — reach/auth/query steps appear in the verification
table; a 401 reads `invalid key`, a timeout reads `unreachable`, and the
key is held **in process memory only** (never `.env`, never a file —
a lint test greps the repo for token patterns). FlyWire works fully
without it.

There is **no fallback matrix of any size**: absent means absent.

## 3. NewsAPI (Live News Feed)

1. Go to https://newsapi.org
2. Click "Get API Key" and register for a free account
3. Copy the API key from your dashboard
4. Paste it into the Settings tab → NewsAPI Key field
5. Free tier: 100 requests per day, 1000 articles per request
6. The app makes one request every 4 minutes (360 per day), which exceeds the free tier. To stay within limits, the app caches news for 20 minutes and only fetches new articles every 20 minutes (3 requests per hour, 72 per day).
7. For full real-time news (every 4 minutes), upgrade to the Developer plan ($449/month) which allows unlimited requests.

Backup options for news:

- Backup 1: CryptoPanic API (https://cryptopanic.com/developers/api/) — free tier allows 50 requests per hour, crypto-specific news
- Backup 2: CoinGecko News API (https://www.coingecko.com/en/api) — free, no key required, limited to crypto news
- Backup 3: RSS feed aggregation (the app parses RSS feeds from CoinDesk, CoinTelegraph, and The Block directly, no API key needed, but less structured data)

## 4. Price Data API (Live Market Data)

The app uses the Binance public API for real-time price and volume data. No API key is required for public market data endpoints.

Endpoints used:

- WebSocket: wss://stream.binance.com:9443/ws/btcusdt@kline_1m (1-minute candlestick stream)
- REST: https://api.binance.com/api/v3/ticker/24hr (24-hour statistics)

For PAXG: use PAXGUSDT pair
For ETH: use ETHUSDT pair
For BNB: use BNBUSDT pair

Backup price data sources:

- Backup 1: CoinGecko API (https://api.coingecko.com/api/v3/simple/price)
- Backup 2: Kraken WebSocket (wss://ws.kraken.com)
- Backup 3: Coinbase WebSocket (wss://ws-feed.exchange.coinbase.com)

The app automatically detects which price source has the lowest latency and uses it as primary, with the others as failover.

## 5. Local Model Dependencies (two independent local models)

No API key required for either.

### 5a. Response agent — llama-cpp (auto-picked by RAM)

Installed by `scripts/setup.sh` via `requirements-agents.txt`
(`llama-cpp-python`, `huggingface-hub`); the model downloads on first
launch through `scripts/download_models.py`:

- ≥14 GB RAM → Phi-3-mini-4k-instruct Q4_K_M
- ≥7 GB RAM → TinyLlama-1.1B-Chat Q4_K_M
- else → Qwen2-0.5B-Instruct Q4_K_M (~2.5 GB disk, 5–10 min)

### 5b. Quant debugger + tab explainer — Ollama `qwen2.5-coder:3b`

Used by `app/auditor.py` (formula exceptions / accuracy audits) and
`app/explainer.py` (the "WHY — plain-language readout" on every tab,
including the TP/SL math). One command installs everything:

```bash
bash scripts/setup_local_llm.sh          # install Ollama + start + pull
bash scripts/setup_local_llm.sh --status # check daemon + model
```

If the daemon is up but the model is missing, the app starts
`ollama pull qwen2.5-coder:3b` **by itself** (background, non-blocking).
While it downloads — or if Ollama is not installed — the explainer falls
back to a deterministic readout built from the engine's own numbers, so
a tab is never blank. Model/env overrides: `OLLAMA_URL` (default
`http://127.0.0.1:11434`), `OLLAMA_MODEL` (default `qwen2.5-coder:3b`).
~2 GB RAM — fits an 8 GB / 4-core Codespace with headroom.

## 6. Environment quick reference

| Variable | Purpose |
|---|---|
| `FEED_MODE` | `auto` / `binance` / `sim` — tape source |
| `PORT` | HTTP port (default 8000) |
| `SKIP_SETUP=1` | skip bootstrap in `run.sh` |
| `OLLAMA_URL` / `OLLAMA_MODEL` | local qwen endpoint / tag |
| `CONNECTOME_CACHE` | FlyWire download + npz cache dir |
| `NEUPRINT_URL` / `NEUPRINT_TOKEN` | optional extra brain queries (memory only) |
| `GEMINI_API_KEY` | primary agent |
| `NEWSAPI_KEY` | live news (falls back to seeded stream) |
