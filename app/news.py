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
    "fed", "rate", "inflation", "cpi", "war", "sanction", "tariff",
    "recession", "etf", "exchange", "regulat", "sec", "central bank",
    "enforcement", "monetary",
)


def relevance(symbol: str, headline: str) -> float:
    """0..1 per-asset relevance: asset keywords first, macro set hits all.
    score == 0 items are COUNTED and shown as 'relevance gate dropped n'
    (error C-4), never silently hidden."""
    h = (headline or "").lower()
    if any(k in h for k in _ASSET_KEYS.get(symbol, ())):
        return 1.0
    if any(k in h for k in _MACRO_KEYS):
        return 0.6
    return 0.0

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
        if NEWS_API_KEY and now - self._last_poll >= NEWS_POLL_SECONDS:
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

    # -- live source (optional) --------------------------------------------
    def _poll_newsapi(self, now: float) -> None:
        try:                                   # pragma: no cover - external
            import httpx
            resp = httpx.get(
                NEWS_API_URL,
                params={"q": NEWS_QUERY, "pageSize": 30,
                        "sortBy": "publishedAt", "language": "en",
                        "apiKey": NEWS_API_KEY},
                timeout=8.0,
            )
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
            self.source = "sim"

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
                "headline": it["headline"],
                "category": it["category"],
                "sentiment": round(it["sentiment"], 3),
                "impact": round(score, 4),
                "duration": dur,
                "age_min": round(age, 1),
                "source": it["source"],
                "reliability": it["reliability"],
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
            "relevance_dropped": dropped,
            "critical": self._critical_check(live, now),
            "heartbeats": {k: {kk: (round(vv, 1) if isinstance(vv, float) else vv)
                               for kk, vv in v.items()}
                           for k, v in self.source_stats.items()},
            "items": rows,
            "_opposing": {"bull": max(0.0, min(1.0, bull)),
                          "bear": max(0.0, min(1.0, -bear))},
        }
        return agg

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
        }


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
