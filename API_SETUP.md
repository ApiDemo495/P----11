# API Setup Guide

## 1. Gemini API (Primary Agent)

1. Go to https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Select your Google Cloud project (or create a new one)
4. Copy the generated API key
5. Paste it into the Settings tab → Gemini API Key field
6. Free tier: 15 requests per minute, 1 million tokens per day
7. The app uses approximately 15 requests per hour (one per GRC), well within free tier limits

## 2. neuPrint API (Drosophila Brain Data)

1. Go to https://neuprint.janelia.org
2. Click "Sign In" and create an account (free)
3. After signing in, go to your profile and copy your API token
4. Paste it into the Settings tab → neuPrint API Key field
5. The app uses neuPrint to download the Drosophila hemibrain connectome data on first launch
6. The connectome data is approximately 500MB and is cached locally after the first download
7. Subsequent launches use the cached data and do not make API calls to neuPrint
8. Rate limit: 100 requests per hour (only relevant during initial download)

Backup options for brain data if neuPrint is unavailable:

- Backup 1: FlyWire connectome API (https://flywire.ai) — requires free account
- Backup 2: Pre-downloaded connectome file from GitHub release (the app provides a direct download link)
- Backup 3: Minimal synthetic brain (the app generates a random graph with 10,000 nodes and scale-free topology that approximates Drosophila connectivity patterns — less accurate but functional)

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

## 5. Local Model Dependencies

No API key required. The app automatically downloads the model on first launch. Required packages are installed via pip:

- llama-cpp-python
- huggingface-hub (for model download)

The model download requires approximately 2.5GB of disk space and 5-10 minutes on a typical Codespaces connection.
