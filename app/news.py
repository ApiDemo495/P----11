"""Section 7 — News system with categories and weightages.

Every news item is classified into one of the seven spec categories, scored

    Impact_Score(n) = Category_Weight(n) * Sentiment(n)
                      * Freshness(n) * Source_Reliability(n)

with Freshness = exp(-0.05 * age_minutes) and Sentiment in [-1, +1].
The aggregate

    News_Impact_Aggregate(t) = sum(impact of items younger than 247 min)

is clamped to [-1, +1] and feeds the CCI (component id 100), the Extreme
Override check, the Drosophila brain inputs and the risk display.

Per-timeframe impacts follow the spec table: a timeframe of duration D_min
receives Impact_Score * min(1, D_min / duration) when duration >= D_min.

Live ingestion: NewsAPI (NEWS_API_KEY) when configured, with the seeded /
simulated store as the always-available backup source (sandbox egress and
Codespaces firewalls otherwise block the API).
"""
from __future__ import annotations

import math
import random
import time
from typing import Any

from .config import (
    NEWS_AGGREGATE_WINDOW,
    NEWS_API_KEY,
    NEWS_API_URL,
    NEWS_CATEGORIES,
    NEWS_FRESHNESS_RATE,
    NEWS_MID_SOURCES,
    NEWS_POLL_SECONDS,
    NEWS_QUERY,
    NEWS_RELIABILITY_LOW,
    NEWS_RELIABILITY_MID,
    NEWS_RELIABILITY_TOP,
    NEWS_TOP_SOURCES,
    SYMBOL_ORDER,
    TIMEFRAMES,
    TF_ORDER,
)

# --------------------------------------------------------------------------
# Category classification: first matching keyword wins (most specific first).
# Order follows spec example sets; unmatched headlines fall back to the
# sentiment category (lowest weight, shortest duration — documented default).
# --------------------------------------------------------------------------
_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("regulatory", (
        "sec ", "sec-", "securities", "ruling", "regulat", "ban ", "banned",
        "central bank", "cbdc", "tax ", "taxation", "government", "law ",
        "lawsuit", "legal", "court", "compliance", "clampdown", "restrict",
    )),
    ("institutional", (
        "etf", "hedge fund", "bank ", "banking", "custody", "institutional",
        "treasury", "acquisition", "buys ", "buying ", "adoption", "adopt",
        "spot ", "fund manager", "asset manager", "allocation",
    )),
    ("protocol", (
        "upgrade", "fork", "exploit", "vulnerability", "smart contract",
        "mainnet", "testnet", "release", "validator", "congestion",
        "protocol", "burn ", "staking", "airdrop", "hard fork",
    )),
    ("macro", (
        "cpi", "fed ", "fomc", "rate decision", "interest rate", "inflation",
        "employment", "nonfarm", "jobs ", "gdp", "recession", "dollar",
        "treasury yield", "macro",
    )),
    ("structure", (
        "hack", "hacked", "depeg", "liquidation", "delist", "listing",
        "exchange outage", "outage", "halted", "insolvent", "bankrupt",
        "market structure", "order book",
    )),
    ("geopolitical", (
        "war ", "sanction", "trade agreement", "election", "geopolit",
        "military", "conflict", "tariff", "embargo", "unrest",
    )),
    ("sentiment", (
        "tweet", "reddit", "viral", "trending", "influencer", "google trends",
        "survey", "poll ", "sentiment", "social", "meme", "hype",
    )),
]

# Deterministic local sentiment lexicon -> [-1, +1] (Part 4's agent layer
# replaces this scorer when an LLM is configured).
_BULL = (
    "approv", "adoption", "inflow", "upgrade", "partnership", "record",
    "surge", "rally", "bullish", "breakout", "growth", "accumulat",
    "launch", "boost", "soar", "gain", "support", "approval", "buy",
)
_BEAR = (
    "ban ", "hack", "exploit", "lawsuit", "outflow", "crash", "bearish",
    "downgrade", "depeg", "liquidat", "fraud", "scam", " plunge", "drop",
    "selloff", "fear", "reject", "suspend", "probe", "risk",
)
# Macro risk-on / risk-off phrasings wires ACTUALLY use (error C-2) — the
# crypto-desk word list alone scored a real world wire at exactly 0.0.
_MACRO_BULL = (
    "ceasefire", "de-escalat", "rate cut", "stimulus", "etf inflow",
    "inflows", "dovish", "cooling inflation", "jobs beat", "rally",
    "easing", "bailout", "rescue", "agreement", "truce",
)
_MACRO_BEAR = (
    "tariff", "tariffs", "strike", "strikes", "drone attack", "missile",
    "invasion", "war ", "sanction", "sanctions", "yields climb",
    "rate hike", "hawkish", "hot inflation", "cpi", "recession",
    "default", "halt", "halted", "outage", "bankruptcy", "collapse",
    "escalat",
)

# Per-asset relevance keywords + the macro set that hits everything
# (advice §4.1 relevance.py; error C-4: "News irrelevant" -> count drops).
_ASSET_KEYS: dict[str, tuple[str, ...]] = {
    "BTC": ("bitcoin", "btc", "crypto", "etf", "halving"),
    "PAXG": ("gold", "paxg", "bullion", "xau"),
    "ETH": ("ethereum", "eth", "ether", "layer 2", "staking"),
    "BNB": ("bnb", "binance", "chain", "token"),
}
_MACRO_KEYS = (
    "fed", "rate", "inflation", "cpi", "recession", "etf", "exchange",
    "regulat", "sec", "central bank", "enforcement", "monetary",
    "yield", "dollar", "recession",
)
# docs11 §4.1 geo tier: wires phrase risk in these, not in crypto words
_GEO_KEYS = (
    "war", "strike", "strikes", "missile", "troops", "drone", "attack",
    "sanction", "sanctions", "tariff", "tariffs", "blockade", "embargo",
    "invasion", "escalat", "airstrike", "bombing", "explosion",
)


def relevance(symbol: str, headline: str) -> float:
    """0..1 per-asset relevance (docs11 §4.1): direct asset terms 1.0,
    macro 0.7, geo 0.6, else 0.0. Zero-for-every-asset items are dropped
    at ingest AND counted with examples (N-5), never hidden."""
    h = (headline or "").lower()
    if any(k in h for k in _ASSET_KEYS.get(symbol, ())):
        return 1.0
    if any(k in h for k in _MACRO_KEYS):
        return 0.7
    words = set(h.split())
    if any((k in words) if len(k) <= 6 else (k in h) for k in _GEO_KEYS):
        return 0.6
    return 0.0

# ── theme -> signed per-asset impact priors (docs11 §4.4) ──────────────
# Data, not code: app/themes.json. Whole-word keywords, most specific first.
import json as _json
import os as _os
from pathlib import Path

_THEMES_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                             "themes.json")
try:
    with open(_THEMES_PATH, encoding="utf-8") as _fh:
        THEMES: list[dict] = (_json.load(_fh).get("themes") or [])
except (OSError, ValueError):            # pragma: no cover
    THEMES = []


def classify_theme(headline: str) -> dict | None:
    """First matching theme (keywords most-specific-first in file order)."""
    h = (headline or "").lower()
    words = set(h.replace("-", " ").split())
    for th in THEMES:
        for kw in th["keywords"]:
            if " " in kw:
                if kw in h:
                    return th
            elif kw in words:
                return th
    return None


def source_tier(source: str) -> int:
    """1 = wire/tier-1 desks, 2 = quality press, 3 = everything else."""
    s = (source or "").lower()
    if any(t in s for t in NEWS_TOP_SOURCES):
        return 1
    if any(m in s for m in NEWS_MID_SOURCES):
        return 2
    return 3


TIER_CREDIBILITY = {1: 1.0, 2: 0.7, 3: 0.45}


def enrich_item(item: dict) -> dict:
    """docs11 §4: the full computed record at INGEST — tier, credibility,
    per-asset relevance, theme, magnitude, signed impacts, scope."""
    headline = str(item.get("headline") or "")
    tier = source_tier(str(item.get("source") or ""))
    item["tier"] = tier
    item.setdefault("credibility", TIER_CREDIBILITY[tier])
    item.setdefault("relevance_by_asset",
                    {sym: relevance(sym, headline) for sym in SYMBOL_ORDER})
    th = classify_theme(headline)
    item["theme"] = th["name"] if th else None
    item["magnitude"] = th["magnitude"] if th else 0.3
    item["impacts"] = dict(th["impacts"]) if th else {}
    item["scope"] = th["scope"] if th else "asset"
    item.setdefault("synthetic", str(item.get("source")) in ("seed", "sim"))
    return item


# Seed headlines (pre-launch content so the pipeline is never empty at boot).
_SEED: list[tuple[str, str, float, int, tuple[str, ...]]] = [
    # (headline, category, sentiment, age_min, symbols)
    ("Spot ETF net inflows accelerate for a fourth session",
     "institutional", 0.8, 12, ("BTC",)),
    ("Miners rotate treasury reserves to exchanges",
     "structure", -0.3, 40, ("BTC",)),
    ("SEC signals review of staking services",
     "regulatory", -0.5, 25, ("ETH", "BTC")),
    ("Real yields slip after soft labor print",
     "macro", 0.45, 28, ("PAXG",)),
    ("Central bank gold purchases slow QoQ",
     "institutional", -0.4, 70, ("PAXG",)),
    ("L2 settlement volume hits monthly high",
     "protocol", 0.42, 18, ("ETH",)),
    ("Large validator queue exits observed",
     "protocol", -0.38, 55, ("ETH",)),
    ("Launchpool listing sparks platform demand",
     "institutional", 0.44, 30, ("BNB",)),
    ("Regulatory hearing scheduled next week",
     "regulatory", -0.5, 90, ("BNB",)),
    ("Community sentiment turns euphoric on social channels",
     "sentiment", 0.5, 6, ("BTC", "ETH", "BNB")),
]


def classify_category(headline: str) -> str:
    """Keyword classifier for the seven spec categories (documented default:
    'sentiment' when nothing matches)."""
    low = headline.lower()
    for cat, keys in _CATEGORY_KEYWORDS:
        for k in keys:
            if k in low:
                return cat
    return "sentiment"


def sentiment_score(headline: str, body: str = "") -> float:
    """Local lexicon sentiment in [-1, +1]: (bull_hits - bear_hits) / total,
    0.0 when the lexicon sees nothing. Macro wire phrasings count too (C-2)."""
    low = (headline + " " + body).lower()
    bull = sum(1 for k in _BULL if k in low) + sum(1 for k in _MACRO_BULL if k in low)
    bear = sum(1 for k in _BEAR if k in low) + sum(1 for k in _MACRO_BEAR if k in low)
    total = bull + bear
    if total == 0:
        return 0.0
    return max(-1.0, min(1.0, (bull - bear) / total))


def source_reliability(source: str) -> float:
    s = (source or "").lower()
    if any(t in s for t in NEWS_TOP_SOURCES):
        return NEWS_RELIABILITY_TOP
    if any(m in s for m in NEWS_MID_SOURCES):
        return NEWS_RELIABILITY_MID
    return NEWS_RELIABILITY_LOW


def impact_score(category: str, sentiment: float, age_minutes: float,
                 reliability: float) -> float:
    """Impact_Score(n) = Category_Weight * Sentiment * Freshness * Reliability."""
    cat = NEWS_CATEGORIES.get(category) or NEWS_CATEGORIES["sentiment"]
    freshness = math.exp(-NEWS_FRESHNESS_RATE * max(0.0, age_minutes))
    return cat["weight"] * sentiment * freshness * reliability


def tf_impact(score: float, duration: int, tf: str) -> float:
    """Section 7 timeframe distribution: TF receives the score scaled by
    min(1, tf_minutes / duration), only when duration covers that TF."""
    tf_min = TIMEFRAMES[tf]["minutes"]
    if duration < tf_min:
        return 0.0
    return score * min(1.0, tf_min / duration)


class NewsSystem:
    def __init__(self, now: float | None = None) -> None:
        self.ready = False
        self._t0 = now if now is not None else time.time()
        self.items: list[dict[str, Any]] = []
        self._rng = random.Random(49507)
        self._last_poll = 0.0
        self.source = "seed"
        self._fingerprints: dict[str, float] = {}   # fp -> first ts
        self.source_stats: dict[str, dict] = {}     # heartbeat per source
        # docs11 §4 evaluation state (pure arithmetic, no network)
        self.filtered = 0                           # all-asset relevance 0
        self.filtered_examples: list[str] = []
        self._theme_arrival: dict[str, float] = {}  # theme -> last arrival
        self._theme_streak: dict[str, float] = {}   # theme -> streak start
        self._theme_count: dict[str, int] = {}      # theme -> k since reset
        self._counted_heads: list[set] = []         # token sets (novelty)
        self.tape_accel: dict[str, float] = {}
        self.last_emergency: dict[str, float] = {}
        # docs11 §3: NewsAPI plan + key ring (3 slots) + conditional GET
        self.api_keys: list[str] = []
        self.key_slot: int = 0
        self.plan: str = _os.environ.get("NEWSAPI_PLAN", "auto")  # auto/free/paid
        self._etag: dict[str, tuple[str, str]] = {}   # url -> (etag, last-mod)
        self._load_keyring()
        self.emergency_history: list[dict] = []
        self.stats_bugs: list[dict] = []      # symbol -> accel (SMD)
        self._niv_hist: dict[str, list] = {}        # symbol -> [(ts, niv)]
        self.last_eval: dict[str, dict] = {}        # symbol -> evaluate()

    # -- lifecycle ----------------------------------------------------------
    def initialize(self) -> None:
        self.ready = True
        self._seed()
        self.refresh()

    def _seed(self) -> None:
        now = time.time()
        self.items = []
        for headline, cat, sent, age, syms in _SEED:
            category = NEWS_CATEGORIES.get(cat)
            if category is None:
                cat_key = classify_category(headline)
            else:
                cat_key = cat
            self._append({
                "headline": headline,
                "category": cat_key,
                "sentiment": sent,
                "reliability": NEWS_RELIABILITY_MID,
                "source": "seed",
                "symbols": list(syms),
                "ts": now - age * 60.0,
            })

    @staticmethod
    def _fingerprint(headline: str) -> str:
        """Normalised title (advice C-5): repeats keep first timestamp."""
        return " ".join(headline.lower().split())[:120]

    def _append(self, item: dict[str, Any]) -> bool:
        """False when the fingerprint was already seen (dedupe)."""
        now = time.time()
        enrich_item(item)
        # docs11 §4.1 / error N-5: an item irrelevant to EVERY asset is
        # dropped at ingest and COUNTED with examples — never hidden.
        # Synthetic/seed packs are the labelled offline layer (excepted).
        rb = item.get("relevance_by_asset") or {}
        if not item.get("synthetic") and all(v == 0.0 for v in rb.values()):
            self.filtered += 1
            head = str(item.get("headline") or "")[:90]
            if head not in self.filtered_examples:
                self.filtered_examples = (self.filtered_examples + [head])[-3:]
            st = self.source_stats.setdefault(item.get("source", "?"),
                                              {"ok": 0, "dupes": 0,
                                               "errors": 0, "last_ok": 0.0})
            st["filtered"] = st.get("filtered", 0) + 1
            return False
        fp = self._fingerprint(str(item.get("headline") or ""))
        first = self._fingerprints.get(fp)
        if first is not None and now - first < 600.0:
            # heartbeat: counted as a duplicate drop, not hidden
            st = self.source_stats.setdefault(item.get("source", "?"),
                                              {"ok": 0, "dupes": 0, "errors": 0,
                                               "last_ok": 0.0})
            st["dupes"] += 1
            return False
        self._fingerprints[fp] = now
        dur = NEWS_CATEGORIES[item["category"]]["duration"]
        item["duration"] = dur
        item["arrival_impact"] = impact_score(
            item["category"], item["sentiment"], 0.0, item["reliability"]
        )
        st = self.source_stats.setdefault(item.get("source", "?"),
                                          {"ok": 0, "dupes": 0, "errors": 0,
                                           "last_ok": 0.0})
        st["ok"] += 1
        st["last_ok"] = now
        # novelty clock (docs11 §4.5): k-th distinct headline on a theme
        # since a 90-minute gap; a new gap resets the theme to full weight
        th = item.get("theme")
        if th:
            last = self._theme_arrival.get(th, 0.0)
            if now - last > 5400.0:                 # 90-min gap -> reset
                self._theme_count[th] = 0
                self._theme_streak[th] = now
            self._theme_arrival[th] = now
            self._theme_count[th] = self._theme_count.get(th, 0) + 1
            item["theme_k"] = self._theme_count[th]
        else:
            item["theme_k"] = 1
        self._counted_heads.append(
            (float(item.get("ts") or now),
             set(str(item.get("headline") or "").lower().split()),
             id(item), len(self._counted_heads)))
        if len(self._counted_heads) > 400:
            self._counted_heads = self._counted_heads[-400:]
        self.items.append(item)
        return True

    def refresh(self, ) -> None:
        """Drop expired items, poll the live source when configured, and top
        up the simulated stream (backup source) so live demos keep moving."""
        now = time.time()
        self.items = [
            it for it in self.items
            if (now - it["ts"]) / 60.0 < NEWS_AGGREGATE_WINDOW
        ]
        if (self.api_keys or NEWS_API_KEY) and \
                now - self._last_poll >= self._poll_interval():
            self._last_poll = now
            self._poll_newsapi(now)
        # keep the feed stocked: the UI must never show an empty table
        if self.source != "newsapi":
            by_sym: dict[str, int] = {}
            for it in self.items:
                for sym in it.get("symbols", []):
                    by_sym[sym] = by_sym.get(sym, 0) + 1
            for _ in range(24):
                if all(by_sym.get(s, 0) >= 3 for s in SYMBOL_ORDER):
                    break
                self._sim_tick(now, force=True)
                for sym in (self.items[-1].get("symbols", []) if self.items else []):
                    by_sym[sym] = by_sym.get(sym, 0) + 1
            # CONTINUOUS STREAM (user request): drip one fresh headline
            # roughly every 8 s while the ticker calls us every 4 s — the
            # News tab keeps receiving items without anyone pressing
            # refresh. Rate-limited so the 600 s dedupe and the
            # evaluation layer stay meaningful.
            youngest = max((float(it.get("ts") or 0) for it in self.items),
                           default=0.0)
            recent_n = sum(1 for it in self.items
                           if now - float(it.get("ts") or 0) < 300.0)
            if now - youngest > 8.0 and recent_n < 40:
                self._sim_tick(now, force=True)

    # -- live source (docs11 §3 ingest layer) -------------------------------
    def _load_keyring(self) -> None:
        """Up to 3 slots: NEWSAPI_KEYS env (comma), then data/keyring.json,
        then the legacy single NEWS_API_KEY. Slots rotate on 429/401."""
        keys: list[str] = []
        env = _os.environ.get("NEWSAPI_KEYS", "")
        keys += [k.strip() for k in env.split(",") if k.strip()]
        try:
            kr = Path(__file__).resolve().parent.parent / "data" / "keyring.json"
            if kr.exists():
                data = _json.loads(kr.read_text())
                if isinstance(data, list):
                    keys += [str(k) for k in data if k][: 3 - len(keys)]
        except Exception:
            pass
        if not keys and NEWS_API_KEY:
            keys = [NEWS_API_KEY]
        self.api_keys = keys[:3]

    def _key(self) -> str:
        if not self.api_keys:
            return NEWS_API_KEY or ""
        return self.api_keys[self.key_slot % len(self.api_keys)]

    def _rotate_key(self) -> None:
        if len(self.api_keys) > 1:
            self.key_slot = (self.key_slot + 1) % len(self.api_keys)

    def _backoff(self, name: str, reset_s: float | None = None) -> float:
        """docs11 §3.3 — exponential 60 s → 3000 s, unless the response
        carries a rate-limit reset (then honour it exactly)."""
        st = self.source_stats.setdefault(
            name, {"ok": 0, "dupes": 0, "errors": 0, "last_ok": 0.0})
        level = int(st.get("backoff_level") or 0) + 1
        st["backoff_level"] = level
        wait = reset_s if reset_s else min(3000.0, 60.0 * (2 ** level))
        st["cooldown_until"] = time.time() + wait
        return wait

    def _poll_interval(self) -> float:
        """Paid = 60 s, free = 900 s (docs11 §3.2); unknown starts at the
        free cadence and adapts the moment a limit header proves the plan."""
        if self.plan == "paid":
            return 60.0
        if self.plan == "free":
            return 900.0
        return float(_os.environ.get("NEWS_POLL_SECONDS", "900"))

    def _http_get(self, url: str, params: dict, name: str):
        """Named UA + conditional GET + 429/426/401 handling. Returns the
        httpx response, or None when the source is cooling down."""
        import httpx
        headers = {
            # docs11 §3.4: an honest, identifiable browser-style UA
            "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
            "Accept": "application/json",
        }
        cached = self._etag.get(url)
        if cached:
            if cached[0]:
                headers["If-None-Match"] = cached[0]
            if cached[1]:
                headers["If-Modified-Since"] = cached[1]
        params = dict(params or {})
        params["apiKey"] = self._key()
        try:
            resp = httpx.get(url, params=params, headers=headers, timeout=8.0)
        except Exception:
            st = self.source_stats.setdefault(
                name, {"ok": 0, "dupes": 0, "errors": 0, "last_ok": 0.0})
            st["errors"] = int(st.get("errors") or 0) + 1
            return None
        st = self.source_stats.setdefault(
            name, {"ok": 0, "dupes": 0, "errors": 0, "last_ok": 0.0})
        if resp.status_code == 304:
            st["not_modified"] = int(st.get("not_modified") or 0) + 1
            return resp
        if resp.status_code == 429:
            st["dropped_429"] = int(st.get("dropped_429") or 0) + 1
            st["last_429"] = time.time()
            reset_s = None
            try:
                remaining = resp.headers.get("X-RateLimit-Remaining")
                if remaining is not None and int(remaining) == 0:
                    if self.plan == "auto":
                        limit = int(resp.headers.get("X-RateLimit-Limit") or 0)
                        self.plan = "free" if limit and limit <= 100 else "paid"
                    reset = resp.headers.get("X-RateLimit-Reset")
                    if reset and int(reset) > 0:
                        reset_s = max(0.0, int(reset) - time.time())
            except Exception:
                pass
            self._backoff(name, reset_s)
            self._rotate_key()
            return resp
        if resp.status_code == 426:
            # plan-limited: the source itself says upgrade required
            st["plan_limited"] = True
            st["cooldown_until"] = time.time() + 3600.0
            return resp
        if resp.status_code in (401, 403):
            st["bad_key"] = True
            self._rotate_key()
            return resp
        # remember validators for the next conditional GET
        self._etag[url] = (resp.headers.get("ETag") or "",
                           resp.headers.get("Last-Modified") or "")
        if resp.status_code == 200 and self.plan == "auto":
            try:
                limit = int(resp.headers.get("X-RateLimit-Limit") or 0)
                if limit:
                    self.plan = "free" if limit <= 100 else "paid"
            except Exception:
                pass
        return resp

    def _poll_newsapi(self, now: float) -> None:
        try:                                   # pragma: no cover - external
            name = "newsapi"
            st = self.source_stats.get(name) or {}
            if float(st.get("cooldown_until") or 0.0) > now:
                return                          # backing off — no request
            resp = self._http_get(
                NEWS_API_URL,
                {"q": NEWS_QUERY, "pageSize": 30, "sortBy": "publishedAt",
                 "language": "en"},
                name,
            )
            if resp is None or resp.status_code == 304:
                return
            if resp.status_code != 200:
                return
            data = resp.json()
            if data.get("status") != "ok":
                return
            for art in data.get("articles", []):
                title = art.get("title") or ""
                if not title:
                    continue
                source = ((art.get("source") or {}).get("name") or "")
                ts = _parse_ts(art.get("publishedAt")) or now
                self._append({
                    "headline": title,
                    "category": classify_category(title),
                    "sentiment": sentiment_score(
                        title, (art.get("description") or "")
                    ),
                    "reliability": source_reliability(source),
                    "source": source,
                    "symbols": _symbol_tags(title),
                    "ts": ts,
                })
            self.source = "newsapi"
        except Exception:
            self.source = self.source or "sim"

    # -- backup source: deterministic simulator -----------------------------
    def _sim_tick(self, now: float, force: bool = False) -> None:
        # roughly one new headline every 1.5 minutes on average (forced when
        # the feed runs thin so the News tab is never empty)
        if not force and self._rng.random() > 0.12:
            return
        cat = self._rng.choice(list(NEWS_CATEGORIES))
        sent = round(self._rng.uniform(-1.0, 1.0), 2)
        tier = self._rng.random()
        rel = (NEWS_RELIABILITY_TOP if tier > 0.75 else
               NEWS_RELIABILITY_MID if tier > 0.35 else
               NEWS_RELIABILITY_LOW)
        src = self._rng.choice(
            ["Reuters", "CoinDesk", "CryptoBlog", "forum-post", "X feed",
             "Bloomberg", "The Block"]
        )
        sym = self._rng.choice(SYMBOL_ORDER)
        headline = _SIM_TEMPLATES[cat][self._rng.randrange(
            len(_SIM_TEMPLATES[cat])
        )].format(s=sym)
        self.source = "sim"
        self._append({
            "headline": headline,
            "category": cat,
            "sentiment": sent,
            "reliability": rel,
            "source": src,
            "symbols": [sym],
            "ts": now,
        })

    # -- reads --------------------------------------------------------------
    def _live_items(self) -> list[dict[str, Any]]:
        now = time.time()
        return [
            it for it in self.items
            if 0.0 <= (now - it["ts"]) / 60.0 < NEWS_AGGREGATE_WINDOW
        ]

    def stats(self) -> Dict:
        """Honesty stats for the news pipe (advice item 6): per-source
        heartbeats + dupe counter. Used by NIV + the UI source table."""
        now = time.time()
        per_hour: dict[str, float] = {}
        for it in self.items:
            src = str(it.get("source") or "?")
            if now - float(it.get("ts") or 0) <= 3600:
                per_hour[src] = per_hour.get(src, 0.0) + 1.0
        return {
            "sources": {k: dict(v) for k, v in self.source_stats.items()},
            "items_per_hour": {k: round(v, 1) for k, v in per_hour.items()},
            "dupes": sum(v.get("dupes", 0) for v in self.source_stats.values()),
            "items": len(self.items),
            "critical_fired": len(getattr(self, "_critical_fired", {}) or {}),
        }

    # critical-event vocabulary: WHOLE-WORD match + the headline must also
    # name a market term + ≤10 min old + fire once per fingerprint (B-2:
    # substring "war" fired on "warning"/"forward" and broke locks hourly)
    _CRITICAL_WORDS = ("war", "hack", "ban", "halt", "attack", "sanctions",
                       "collapse", "depeg", "exploit", "invasion", "outage",
                       "default")
    _MARKET_TERMS = ("bitcoin", "btc", "eth", "bnb", "gold", "paxg",
                     "exchange", "etf", "fed", "rate", "sec", "binance",
                     "coinbase", "market", "price")

    def critical(self, symbol: str, within_min: float = 30.0) -> list[dict]:
        """docs11 §4.4 — recent tier-1/2 headlines carrying this asset
        (recency window in minutes, chosen by the caller)."""
        now = time.time()
        return [it for it in self._live_items()
                if (it.get("reliability") or 0.0) >= 0.8
                and float((it.get("relevance_by_asset") or {}).get(symbol, 0.0)) > 0
                and now - float(it.get("ts") or 0) <= within_min * 60.0]

    def _critical_check(self, items: list, now: float) -> dict:
        import re
        fired = getattr(self, "_critical_fired", None)
        if fired is None:
            fired = self._critical_fired = {}
        for it in items:
            head = str(it.get("headline") or "")
            low = head.lower()
            age_min = (now - float(it.get("ts") or 0)) / 60.0
            if age_min > 10.0:
                continue
            words = set(re.findall(r"[a-z]+", low))
            hit = [w for w in self._CRITICAL_WORDS
                   if w in words or any(m.startswith(w) for m in words)]
            has_market = any(t in low for t in self._MARKET_TERMS)
            if not hit or not has_market:
                continue
            fp = self._fingerprint(head)
            if fp in fired:
                continue                    # fires once per fingerprint
            fired[fp] = now
            self._critical_fired = {k: v for k, v in fired.items()
                                    if now - v < 3600}
            return {"active": True, "headline": head,
                    "word": hit[0], "age_min": round(age_min, 1),
                    "fired_at": now, "source": it.get("source")}
        # NIV-swing trigger (separate from headline trigger, advice §4.1)
        return {"active": False}

    def aggregate(self, symbol: str) -> dict:
        """Signed News Impact Aggregate + per-TF impacts for one asset."""
        now = time.time()
        live = self._live_items()
        # relevance gate (C-4): candidates carry this symbol; score == 0
        # candidates are DROPPED AND COUNTED. Macro-scored items (≥0.6)
        # reach EVERY asset even when the tag list omits it — that is what
        # the macro set is for; a macro item never silently vanishes.
        candidates = [it for it in live if symbol in (it.get("symbols") or ())]
        dropped = sum(1 for it in candidates
                      if relevance(symbol, str(it.get("headline") or "")) == 0.0)
        mine = [it for it in candidates
                if relevance(symbol, str(it.get("headline") or "")) > 0.0]
        seen = {id(it) for it in mine}
        for it in live:
            if id(it) in seen:
                continue
            if relevance(symbol, str(it.get("headline") or "")) >= 0.6:
                mine.append(it)
        total = 0.0
        bull = 0.0
        bear = 0.0
        rows: list[dict] = []
        impacts = {tf: 0.0 for tf in TF_ORDER}
        for it in mine:
            age = (now - it["ts"]) / 60.0
            score = impact_score(
                it["category"], it["sentiment"], age, it["reliability"]
            )
            total += score
            if score >= 0:
                bull += score
            else:
                bear += score
            dur = it["duration"]
            for tf in TF_ORDER:
                impacts[tf] += tf_impact(score, dur, tf)
            rows.append({
                # ts MUST survive the repack — without it the table
                # renders Date(0) = 05:30 IST for every row and the feed
                # looks frozen at app-open time (user report)
                "ts": it["ts"],
                "headline": it["headline"],
                "category": it["category"],
                "sentiment": round(it["sentiment"], 3),
                "impact": round(score, 4),
                "duration": dur,
                "age_min": round(age, 1),
                "source": it["source"],
                "reliability": it["reliability"],
                "synthetic": bool(it.get("synthetic")),
            })
        # NIV with resting evidence (advice C-3): divide by (sum|w| + W0)
        # so a single stale headline can never pin the aggregate.
        wsum = sum(abs(impact_score(it["category"], it["sentiment"],
                                    (now - it["ts"]) / 60.0,
                                    it["reliability"])) for it in mine) or 0.0
        niv = max(-1.0, min(1.0, total / (wsum + 0.5))) if mine else 0.0
        total = max(-1.0, min(1.0, total))
        agg = {
            # signed Section-7 aggregate (component 100 / override / brains)
            "value": round(total, 4),
            # legacy split view (UI + E17/E20 consumers)
            "bull": round(max(0.0, min(1.0, bull)), 4),
            "bear": round(max(0.0, min(1.0, -bear)), 4),
            "polarity": round(total, 4),
            "count": len(mine),
            "impacts": {tf: round(max(-1.0, min(1.0, impacts[tf])), 4)
                        for tf in TF_ORDER},
            "source": self.source,
            "niv": round(niv, 4),
            "resting_evidence": 0.5,
            # in-window zeros + ingest-gate drops (N-5): never hidden
            "relevance_dropped": dropped + self.filtered,
            "critical": self._critical_check(live, now),
            "heartbeats": {k: {kk: (round(vv, 1) if isinstance(vv, float) else vv)
                               for kk, vv in v.items()}
                           for k, v in self.source_stats.items()},
            "items": rows,
            "_opposing": {"bull": max(0.0, min(1.0, bull)),
                          "bear": max(0.0, min(1.0, -bear))},
        }
        return agg

    # ==================================================================
    # docs11 §4 EVALUATION layer — pure arithmetic over the cache, no
    # network. Runs every 4 s (the ticker) and on demand; this is the
    # layer that is actually "alive", not the poller.
    # ==================================================================
    def _novelty(self, item: dict, now: float) -> tuple[float, dict]:
        """(novelty weight, trace) — Jaccard near-dups weigh 0; the k-th
        distinct headline on a theme weighs 1/k; a theme continuously on
        the wire halves every 2 h and resets after a 90-minute gap."""
        head = set(str(item.get("headline") or "").lower().split())
        item_ts = float(item.get("ts") or now)
        dup = False
        # locate my ingestion order by token set (id() can be reused by
        # GC'd seed items, so identity matching is not safe here)
        my_seq = max((seq for _t, hset, _pid, seq in self._counted_heads
                      if hset == head), default=10 ** 9)
        for prev_ts, prev, prev_id, prev_seq in self._counted_heads:
            # chronological: only headlines ingested EARLIER can habituate
            # this one (equal timestamps fall back to ingestion order)
            if (prev_seq >= my_seq or prev_ts > item_ts
                    or not head or not prev or prev == head):
                continue
            j = len(head & prev) / max(1, len(head | prev))
            if j >= 0.6:
                dup = True
                break
        if dup:
            return 0.0, {"reason": "near-duplicate (Jaccard ≥ 0.6)"}
        k = int(item.get("theme_k") or 1)
        base = 1.0 / max(1, k)
        th = item.get("theme")
        habit = 1.0
        if th:
            last = self._theme_arrival.get(th, 0.0)
            streak = self._theme_streak.get(th, now)
            if now - last > 5400.0:
                habit = 1.0                          # 90-min gap -> reset
            else:
                hours = max(0.0, (now - streak) / 3600.0)
                habit = 0.5 ** (hours / 2.0)         # halves every 2 h
        return base * habit, {"k": k, "habituation": round(habit, 4)}

    def evaluate(self, symbol: str, now: float | None = None) -> dict:
        """Per asset × per horizon: impact index (docs11 §4.7), NIV with
        τ_h = W/4 and W₀ = 0.5 (§4.8), SMD = tanh(NIV − tape accel), and
        the top-4 drivers with their full weight traces."""
        now = now if now is not None else time.time()
        accel = float(self.tape_accel.get(symbol, 0.0))
        out: dict[str, Any] = {"symbol": symbol, "evaluated_at": now,
                               "tape_accel": accel, "horizons": {}}
        # scored layer excludes labelled synthetic/seed items (docs11 §4.7:
        # synthetic coverage is visible in the feed, never in the drivers)
        relevant = [
            it for it in self._live_items()
            if not it.get("synthetic")
            and float((it.get("relevance_by_asset") or {}).get(symbol, 0.0)) > 0.0
        ]
        for tf in TF_ORDER:
            W = TIMEFRAMES[tf]["minutes"]
            halflife = W / 2.0
            tau = W / 4.0
            total = 0.0
            drivers: list[dict] = []
            niv_num = niv_den = 0.0
            newest: list[dict] = []
            for it in relevant:
                age_min = (now - float(it.get("ts") or now)) / 60.0
                cred = float(it.get("credibility") or 0.45)
                rel = float((it.get("relevance_by_asset") or {}).get(symbol, 0.0))
                # skip beyond 12 half-lives of this horizon (docs11 §4.6)
                if age_min > 12.0 * halflife:
                    continue
                novelty, ntrace = self._novelty(it, now)
                decay = 0.5 ** (age_min / halflife)
                mag = float(it.get("magnitude") or 0.3)
                signed = float((it.get("impacts") or {}).get(symbol, 0.0))
                contrib = (cred * decay * novelty * mag * signed * rel)
                total += contrib
                drivers.append({
                    "headline": str(it.get("headline"))[:110],
                    "theme": it.get("theme"), "age_min": round(age_min, 1),
                    "novelty": round(novelty, 4), "k": ntrace.get("k"),
                    "habituation": ntrace.get("habituation"),
                    "contribution": round(contrib, 5),
                    "synthetic": bool(it.get("synthetic")),
                })
                newest.append(it)
            # NIV: five newest relevant items, τ_h = W/4, W₀ = 0.5
            newest.sort(key=lambda x: -float(x.get("ts") or 0))
            for it in newest[:5]:
                age_min = (now - float(it.get("ts") or now)) / 60.0
                w = (float(it.get("credibility") or 0.45)
                     * rel if (rel := float((it.get("relevance_by_asset") or {}).get(symbol, 0.0))) > 0 else 0.0)
                w *= math.exp(-age_min / max(tau, 1e-9))
                niv_num += float(it.get("sentiment") or 0.0) * w
                niv_den += w
            niv = niv_num / (niv_den + 0.5) if (niv_den + 0.5) > 0 else 0.0
            drivers.sort(key=lambda d: -abs(d["contribution"]))
            out["horizons"][tf] = {
                "impact": round(math.tanh(total), 4),
                "niv": round(niv, 4),
                "smd": round(math.tanh(niv - accel), 4),
                "half_life_min": halflife,
                "tau_min": tau,
                "drivers": drivers[:4],
                "n_items": len(drivers),
            }
        # NIV history for the sentiment-swing trigger (|ΔNIV| over 90 s)
        hist = self._niv_hist.setdefault(symbol, [])
        hist.append((now, out["horizons"]["tf1"]["niv"]))
        self._niv_hist[symbol] = [h for h in hist if now - h[0] <= 90.0]
        swing = 0.0
        if len(self._niv_hist.get(symbol, [])) >= 2:
            vals = [v for _t, v in self._niv_hist[symbol]]
            swing = abs(vals[-1] - vals[0])
        out["niv_swing_90s"] = round(swing, 4)
        # theme inputs for the learned layer (docs11 §5 path 4)
        theme_inputs: dict[str, float] = {}
        for d in drivers:
            if d["theme"] and not d["synthetic"]:
                theme_inputs[f"news:theme:{d['theme']}"] = round(
                    theme_inputs.get(f"news:theme:{d['theme']}", 0.0)
                    + d["contribution"], 5)
        out["theme_inputs"] = theme_inputs
        # docs11 §4.9 / error N-8: CRITICAL event + sentiment swing
        # triggers an emergency re-lock; >1/hour is a BUG and is rejected.
        crit = self.critical(symbol, max(10.0, TIMEFRAMES["tf1"]["minutes"] / 10.0))
        out["critical_count"] = len(crit)
        trig = None
        if crit and swing > 0.02 and (now - self.last_emergency.get(symbol, 0.0)) > 60.0:
            hour = [t for t in self.emergency_history
                    if t.get("asset") == symbol and now - t.get("ts", 0.0) < 3600.0]
            if len(hour) >= 6:
                self.stats_bugs.append({
                    "at": now, "code": "N-8",
                    "detail": f">6 emergency re-locks in 1 h for {symbol} — BUG"})
            else:
                trig = {"reason": "NEWS_EMERGENCY: " + "; ".join(
                            str(c.get("headline"))[:70] for c in crit[:3]),
                        "niv_swing_90s": swing}
        out["emergency_trigger"] = trig
        self.last_eval[symbol] = out
        return out

    def impact(self, symbol: str, tf: str) -> dict:
        """GET /api/news/impact?asset=&horizon= — index + drivers."""
        ev = self.last_eval.get(symbol) or self.evaluate(symbol)
        h = (ev.get("horizons") or {}).get(tf)
        if h is None:
            return {"ok": False, "error": "unknown horizon"}
        return {"ok": True, "asset": symbol, "horizon": tf, **h,
                "theme_inputs": ev.get("theme_inputs", {}),
                "niv_swing_90s": ev.get("niv_swing_90s", 0.0)}

    def set_tape_accel(self, symbol: str, accel: float) -> None:
        """Engine feeds tape acceleration for SMD (news-vs-price)."""
        self.tape_accel[symbol] = float(accel)

    def impact_against(self, symbol: str, direction: int) -> float:
        """Opposing news impact magnitude in [0,1] (Extreme Override > 0.8)."""
        value = self.aggregate(symbol)["value"]
        if direction >= 0:
            return max(0.0, -value)
        return max(0.0, value)

    def agrees_with(self, symbol: str, direction: int) -> bool:
        """Neutral aggregate counts as agreement (risk-layer convention)."""
        value = self.aggregate(symbol)["value"]
        if abs(value) < 0.05:
            return True
        return (1 if value > 0 else -1) == direction or abs(value) < 0.15

    def arrival_scores_last(self, minutes: int) -> list[float]:
        """Arrival impact scores within `minutes` (QDPC event counting)."""
        now = time.time()
        return [
            it["arrival_impact"] for it in self.items
            if 0.0 <= (now - it["ts"]) / 60.0 <= minutes
        ]

    def arrival_minutes_last(self, minutes: int) -> dict[int, float]:
        """Minute epoch -> summed arrival impact score for the last
        `minutes` (P9 hot flags, P21 significant-event counting)."""
        now = time.time()
        cutoff = now - minutes * 60.0
        buckets: dict[int, float] = {}
        for it in self.items:
            if it["ts"] >= cutoff:
                minute = int(it["ts"] // 60) * 60
                buckets[minute] = buckets.get(minute, 0.0) + it["arrival_impact"]
        return buckets

    def status(self) -> dict:
        live = self._live_items()
        return {
            "ready": self.ready,
            "items": len(live),
            "total": len(self.items),
            "source": self.source,
            "categories": len(NEWS_CATEGORIES),
            "filtered": self.filtered,
            "filtered_examples": self.filtered_examples,
            "coverage": self.coverage_string(),
            "sources": self.source_lines(),
            "horizons": {sym: ev.get("horizons")
                         for sym, ev in self.last_eval.items() if ev},
            "emergency_history": self.emergency_history[-10:],
            "plan": self.plan,
            "keys": len(self.api_keys),
            "poll_interval_s": self._poll_interval(),
            # pricing reference (docs11 §3.5): free = 100 req/day,
            # paid ≈ $449/mo — read from the spec at build time, not live
            "price_note": "free 100 req/day (429 cools the quota window); "
                          "paid plan unlimited-ish 60s cadence",
        }

    # ------------------------------------------------------------- docs11 §3.6
    def coverage_string(self) -> str:
        """Honest header, e.g. 'LIVE Reuters CoinDesk · last 15 min 7
        items · 2 topics skipped for no crypto relevance · rate limited
        1 sources'. SEED/SYNTHETIC items never claim to be live (N-2)."""
        now = time.time()
        live, down = [], []
        for name, st in self.source_stats.items():
            cool = float(st.get("cooldown_until") or 0.0)
            if cool > now:
                down.append(name)
            elif now - float(st.get("last_ok") or 0.0) < 300.0:
                live.append(name)
        recent = len([i for i in self.items
                      if now - float(i.get("ts") or 0) < 900.0
                      and not i.get("synthetic")])
        seeds = len(self.items) - recent
        parts = []
        if live:
            parts.append("LIVE " + " ".join(sorted(live)))
        if seeds:
            parts.append(f"SEED x{seeds}")
        parts.append(f"last 15 min {recent} items")
        if self.filtered:
            parts.append(
                f"{self.filtered} topics skipped for no crypto relevance")
        if down:
            parts.append(f"rate limited {len(down)} sources")
        if not any(not it.get("synthetic") for it in self.items):
            parts.append("coverage: synthetic")
        return " · ".join(parts)

    def source_lines(self) -> list[dict]:
        """Per-source lines for the news card (state + backoff level)."""
        now = time.time()
        out = []
        for name, st in sorted(self.source_stats.items()):
            cool = float(st.get("cooldown_until") or 0.0)
            age = now - float(st.get("last_ok") or 0.0)
            state = ("backing off" if cool > now else
                     "live" if age < 300.0 else "idle")
            out.append({
                "source": name, "tier": source_tier(name),
                "state": state,
                "cooldown_s": round(max(0.0, cool - now)),
                "backoff_level": int(st.get("backoff_level") or 0),
                "dropped_429": int(st.get("dropped_429") or 0),
                "ok": int(st.get("ok") or 0),
                "dupes": int(st.get("dupes") or 0),
                "filtered": int(st.get("filtered") or 0),
                "last_ok_age_s": round(age, 1),
            })
        return out

    def record_emergency(self, asset: str, lock_id: str, reason: str,
                         old_entry: float, new_entry: float,
                         ok: bool = True) -> dict:
        """Audit row for every emergency re-lock (acceptance 6 / N-8)."""
        row = {"ts": time.time(), "asset": asset, "lock_id": lock_id,
               "reason": reason, "old_entry": old_entry,
               "new_entry": new_entry, "ok": ok}
        self.emergency_history.append(row)
        self.emergency_history = self.emergency_history[-50:]
        if ok:
            self.last_emergency[asset] = row["ts"]
        return row


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _symbol_tags(headline: str) -> list[str]:
    low = headline.lower()
    tags: list[str] = []
    if any(k in low for k in ("bitcoin", "btc", "spot etf")):
        tags.append("BTC")
    if any(k in low for k in ("ethereum", "eth ", "ether")):
        tags.append("ETH")
    if any(k in low for k in ("bnb", "binance")):
        tags.append("BNB")
    if any(k in low for k in ("gold", "paxg")):
        tags.append("PAXG")
    return tags or ["BTC", "ETH", "BNB", "PAXG"]


def _parse_ts(value: Any) -> float | None:      # pragma: no cover - external
    if not value or not isinstance(value, str):
        return None
    try:
        from datetime import datetime, timezone
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        ).astimezone(timezone.utc).timestamp()
    except ValueError:
        return None


_SIM_TEMPLATES: dict[str, list[str]] = {
    "regulatory": [
        "SEC issues guidance on {s} custody rulemaking",
        "Lawmakers propose new tax treatment for {s} staking",
        "Central bank reiterates caution on {s} adoption",
    ],
    "institutional": [
        "Fund manager discloses larger {s} spot position",
        "Bank announces {s} custody service rollout",
        "ETF provider files amended {s} registration",
    ],
    "protocol": [
        "{s} mainnet upgrade activates performance patch",
        "Developers patch vulnerability in {s} contract",
        "{s} network congestion eases after fee spike",
    ],
    "macro": [
        "CPI print lands hot; {s} reacts to rate path shift",
        "Fed minutes point to extended hold; {s} consolidates",
        "GDP beat lifts risk appetite across {s} pairs",
    ],
    "structure": [
        "Exchange reports partial outage in {s} order book",
        "Liquidation cluster cleared in {s} perpetuals",
        "Stablecoin depeg fears pressure {s} liquidity",
    ],
    "sentiment": [
        "Social chatter on {s} spikes to weekly high",
        "Influencer thread on {s} goes viral overnight",
        "Search interest in {s} trends upward",
    ],
    "geopolitical": [
        "New sanctions package complicates {s} flows",
        "Election uncertainty lifts safe-haven {s} demand",
        "Trade tensions resurface; {s} volatility bid",
    ],
}
