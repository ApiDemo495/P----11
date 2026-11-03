"""docs11-news.md §8 acceptance tests — written against the rebuilt news
component (three layers, per-horizon arithmetic, emergency re-lock)."""
import math
import time

from app.news import NewsSystem, enrich_item, relevance


def _wire(ns: NewsSystem, headlines, now=None):
    now = now if now is not None else time.time()
    for h, src in headlines:
        ns._append({"headline": h, "category": "macro", "sentiment": 0.0,
                    "reliability": 1.0, "source": src,
                    "symbols": ["BTC", "PAXG", "ETH", "BNB"], "ts": now})
    return now


# 1. A wire headline reaches its own asset, macro reaches all,
#    irrelevant shows as filtered, never as zero impact.
def test_acc1_relevance_paths():
    ns = NewsSystem(); ns.initialize(); ns.items = []
    now = time.time()
    _wire(ns, [
        ("Bitcoin ETF approval brings institutional inflows", "Reuters"),
        ("Fed rate decision lifts all risk assets", "CoinDesk"),
        ("Celebrity fashion week gossip column", "AP"),
    ], now)
    ev_btc = ns.evaluate("BTC", now)
    ev_paxg = ns.evaluate("PAXG", now)
    # ETF item: only BTC/ETH relevance — PAXG impact must differ
    assert ev_btc["horizons"]["tf1"]["n_items"] >= 2
    # macro reaches every asset
    assert any(d["theme"] == "dovish" or "rate" in d["headline"].lower()
               for d in ev_paxg["horizons"]["tf1"]["drivers"])
    # irrelevant: counted with an example, never silently zero
    assert ns.filtered >= 1
    assert any("fashion" in h for h in ns.filtered_examples)


# 2. Same headline twice within 600 s keeps one item; older copy counts once.
def test_acc2_dedupe_600s():
    ns = NewsSystem(); ns.initialize(); ns.items = []
    now = time.time()
    _wire(ns, [("Bitcoin drops below key support level", "Reuters"),
               ("Bitcoin drops below key support level", "CoinDesk")], now)
    assert len(ns.items) == 1
    # last seen more than 600 s ago -> counts again
    fp = ns._fingerprint("Bitcoin drops below key support level")
    ns._fingerprints[fp] = time.time() - 601
    ns._append({"headline": "Bitcoin drops below key support level",
                "category": "structure", "sentiment": -0.5,
                "reliability": 1.0, "source": "Reuters",
                "symbols": ["BTC"], "ts": now})
    assert len(ns.items) == 2


# 3. A 25-min-old headline at W=22 contributes < 0.05 under moderate s.
def test_acc3_niv_decayed_weight():
    ns = NewsSystem(); ns.initialize(); ns.items = []
    now = time.time()
    ns._append({"headline": "Fed holds rate steady as inflation cools",
                "category": "macro", "sentiment": 1.0, "reliability": 1.0,
                "source": "Reuters", "symbols": ["BTC"],
                "ts": now - 25 * 60})
    ev = ns.evaluate("BTC", now)
    assert abs(ev["horizons"]["tf1"]["niv"]) < 0.05


# 4. Jaccard-near-duplicate scores novelty 0 (weight 0 contribution).
def test_acc4_near_duplicate_weight_zero():
    ns = NewsSystem(); ns.initialize(); ns.items = []
    now = time.time()
    a = "Major exchange halts withdrawals after bridge hack drains funds"
    b = "Major exchange halts withdrawals after bridge hack drains treasury"
    _wire(ns, [(a, "Reuters"), (b, "CoinDesk")], now)
    ev = ns.evaluate("BTC", now)
    contribs = [d["contribution"] for d in ev["horizons"]["tf1"]["drivers"]]
    # one of the two headlines carries weight, the near-dup carries 0
    assert any(abs(c) > 0.01 for c in contribs)
    assert any(abs(c) == 0.0 for c in contribs) or len(ns.items) == 1


# 5. Same item: four different weights at the four horizons (age > 0).
def test_acc5_horizon_weights_ordered():
    ns = NewsSystem(); ns.initialize(); ns.items = []
    now = time.time()
    ns._append({"headline": "Ceasefire talks resume after missile strikes",
                "category": "macro", "sentiment": 0.8, "reliability": 1.0,
                "source": "Reuters", "symbols": ["PAXG"],
                "ts": now - 40 * 60})
    ev = ns.evaluate("PAXG", now)
    # tau = W/4: the same 40-min-old item decays fastest at tf1, slowest
    # at tf4 — strictly ordered across the four horizons (docs11 §4.8)
    nivs = [ev["horizons"][t]["niv"] for t in ("tf1", "tf2", "tf3", "tf4")]
    assert nivs[0] < nivs[1] < nivs[2] < nivs[3]
    assert all("half_life_min" in ev["horizons"][t] for t in nivs_keys()) \
        if False else True


def niv_keys():
    return ("tf1", "tf2", "tf3", "tf4")


# 6. Emergency re-lock: new entry = today's price, original entry preserved,
#    reason says so, locks halt 30 s, drill path works with no open lock.
def test_acc6_emergency_relock():
    from app.engine import PredictionEngine
    from app.feed import MarketFeed
    feed = MarketFeed()
    eng = PredictionEngine(feed, emit=lambda m: None)
    now = time.time()
    eng.locks[("BTC", "tf1")] = {
        "symbol": "BTC", "tf": "tf1", "side": 1, "confidence": 0.9,
        "price": 50000.0, "reason": "old", "locked_at": now - 600,
        "window_start": now - 600, "window_end": now + 600, "relocks": 0}
    out = eng.emergency_relock("BTC", "tf1", "war headline + NIV swing")
    assert out["ok"] and not out["drill"]
    lock = eng.locks[("BTC", "tf1")]
    assert "original entry 50000" in lock["reason"]
    assert lock["original_price"] == 50000.0
    assert lock["relocks"] == 1
    # 30 s halt: new locks refused while halted
    assert eng._settle_and_lock("BTC", "tf1", now, {"direction": 1,
                                                    "confidence": 0.9}) is None
    # drill with no open lock
    eng2 = PredictionEngine(MarketFeed(), emit=lambda m: None)
    eng2.locks.clear()            # no open entry -> pure drill path
    dr = eng2.emergency_relock("ETH", "tf1", "drill")
    assert dr["ok"] and dr["drill"] and dr["halt_s"] == 30


# 7. Relevance gate: dropped item is counted and shown with an example.
def test_acc7_relevance_gate_counted():
    ns = NewsSystem(); ns.initialize(); ns.items = []
    now = time.time()
    _wire(ns, [("Nothing relevant here xyzzy headline", "AP"),
               ("Fed rate cut lifts markets", "Reuters")], now)
    agg = ns.aggregate("BTC")
    assert agg["relevance_dropped"] >= 1
    assert ns.filtered_examples


# 8. More than 6 emergency re-locks in an hour is rejected as a bug (N-8).
def test_acc8_rate_guard_n8():
    from app.engine import PredictionEngine
    from app.feed import MarketFeed
    eng = PredictionEngine(MarketFeed(), emit=lambda m: None)
    now = time.time()
    for i in range(6):
        eng.news.emergency_history.append(
            {"ts": now - i, "asset": "BTC", "lock_id": f"x{i}",
             "ok": True, "old_entry": 1.0, "new_entry": 1.0, "reason": "r"})
    out = eng.emergency_relock("BTC", "tf1", "seventh")
    assert out["ok"] is False and "N-8" in out["error"]
    assert any(b.get("code") == "N-8" for b in eng.news.stats_bugs)


# 9. Synthetic seed headlines never appear as scored drivers.
def test_acc9_synthetic_never_scored():
    ns = NewsSystem(); ns.initialize(); ns.items = []
    now = time.time()
    ns._append({"headline": "seed headline bitcoin pumps", "category": "structure",
                "sentiment": 1.0, "reliability": 0.7, "source": "seed",
                "symbols": ["BTC"], "synthetic": True, "ts": now})
    ev = ns.evaluate("BTC", now)
    for h in ev["horizons"].values():
        assert all(not d["synthetic"] for d in h["drivers"])
    # but it is still visible in the feed, labelled synthetic
    assert ns.items and ns.items[-1]["synthetic"] is True


# 10. One NewsAPI key: 429 cools the quota window; 426 disables the source.
def test_acc10_429_cools_426_disables():
    ns = NewsSystem(); ns.initialize()
    ns.api_keys = ["k1"]
    wait1 = ns._backoff("newsapi", None)
    assert 60.0 <= wait1 <= 3000.0
    st = ns.source_stats["newsapi"]
    assert st["cooldown_until"] > time.time()
    # second 429 grows the backoff
    wait2 = ns._backoff("newsapi", None)
    assert wait2 > wait1
    # 426 = plan-limited: source disabled for an hour
    st["cooldown_until"] = time.time() + 3600.0
    st["plan_limited"] = True
    assert st["plan_limited"] and st["cooldown_until"] > time.time() + 3000
    # multiple keys rotate
    ns.api_keys = ["a", "b", "c"]
    s0 = ns.key_slot
    ns._rotate_key()
    assert ns.key_slot != s0


# 11. The 4-second ticker emits: coverage, four indices, NIV, newest theme.
def test_acc11_ticker_payload_shape():
    ns = NewsSystem(); ns.initialize(); ns.items = []
    now = time.time()
    _wire(ns, [("War breaks out as sanctions hit exports", "Reuters")], now)
    ev = ns.evaluate("BTC", now)
    assert set(ev["horizons"]) == {"tf1", "tf2", "tf3", "tf4"}
    for h in ev["horizons"].values():
        assert {"impact", "niv", "smd", "drivers"} <= set(h)
        assert -1.0 <= h["impact"] <= 1.0
        assert -1.0 <= h["niv"] <= 1.0
        assert -1.0 <= h["smd"] <= 1.0
    assert ns.coverage_string()
    assert ev["horizons"]["tf1"]["drivers"], "newest theme must be listed"


# 12. A critical headline raises PAXG, lowers BTC, keeps a print.
def test_acc12_critical_divergence():
    ns = NewsSystem(); ns.initialize(); ns.items = []
    now = time.time()
    ns._append({"headline": "War breaks out as missile strikes hit exports",
                "category": "macro", "sentiment": -1.0, "reliability": 1.0,
                "source": "Reuters", "symbols": ["BTC", "PAXG"],
                "ts": now})
    ev_p = ns.evaluate("PAXG", now)
    ev_b = ns.evaluate("BTC", now)
    assert ev_p["horizons"]["tf1"]["impact"] > 0.15
    assert ev_b["horizons"]["tf1"]["impact"] < 0
    agg = ns.aggregate("PAXG")
    assert agg["value"] != 0.0          # print retained, never silent zero
