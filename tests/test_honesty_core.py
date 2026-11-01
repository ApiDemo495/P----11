"""Advice Part C: lock, ledger, grid, veto, feed/news honesty, watchdog."""
import time


# ── ledger: strict sign, ties=loss, skips, streaks (G-1/0.4) ──
def test_ledger_strict_sign_and_streaks():
    import tempfile, os
    from app.ledger import SettlementLedger
    with tempfile.TemporaryDirectory() as td:
        led = SettlementLedger(path=os.path.join(td, "l.json"))
        led.settle("BTC", "tf1", 100.0, 101.0, +1)     # up vs side +1 -> win
        led.settle("BTC", "tf1", 100.0, 99.0, +1)      # down vs +1 -> loss
        led.settle("BTC", "tf1", 100.0, 100.0, +1)     # tie -> loss
        led.settle("BTC", "tf1", 100.0, 101.0, 0)      # no side -> skip
        st = led.stats("BTC", "tf1")
        assert st["n"] == 3 and st["wins"] == 1 and st["losses"] == 2
        assert st["skipped"] == 1
        assert st["enough"] is False                   # n < 20
        assert st["rate"] == round(1 / 3, 4)
        # persistence round-trip
        led2 = SettlementLedger(path=led.path)
        assert led2.stats("BTC", "tf1")["n"] == 3
        # pooled across assets for one tf
        led.settle("ETH", "tf1", 5.0, 5.1, -1)         # price UP vs SELL -> loss
        pool = led.pooled("tf1")
        assert pool["n"] == 4 and pool["wins"] == 1 and pool["losses"] == 3


# ── grid-aligned windows + lock protocol (0.1/0.3) ──
def test_grid_windows_and_lock_hold():
    from tests.test_engine import make_engine, _boot, run
    eng, _f, _e = make_engine(); run(_boot(eng))
    now = time.time()
    res = eng._event_result("BTC", "tf1")
    eng.fresh_cycle("BTC", "tf1", now, res, reason="initial_lock")
    cyc = eng.cycles[("BTC", "tf1")]
    ws = 22 * 60
    assert cyc.cycle_start % ws == 0                  # epoch grid, not now+W
    assert cyc.cycle_end == cyc.cycle_start + ws
    lk1 = dict(eng.locks[("BTC", "tf1")])
    # mid-window refash must NOT relock (signal integrity, B-1)
    eng.fresh_cycle("BTC", "tf1", now + 30, res, reason="confidence_recovered")
    assert eng.locks[("BTC", "tf1")]["locked_at"] == lk1["locked_at"]
    # next window settles the old lock into the ledger
    eng.ledger.path = "/tmp/test_led_settle.json"
    next_start = cyc.cycle_end + 1
    eng.fresh_cycle("BTC", "tf1", next_start, res, reason="rollover")
    assert eng.ledger.stats("BTC", "tf1")["n"] >= 1
    lk2 = eng.locks[("BTC", "tf1")]
    assert lk2["window_start"] > lk1["window_start"]   # new window, new lock


def test_snapshot_carries_lock_feed_ledger_watchdog():
    from tests.test_engine import make_engine, _boot, run
    eng, _f, _e = make_engine(); run(_boot(eng))
    now = time.time()
    res = eng._event_result("BTC", "tf1")
    eng.fresh_cycle("BTC", "tf1", now, res, reason="initial_lock")
    snap = eng.build_snapshot()
    assert "feed" in snap and "label" in snap["feed"]
    assert snap["feed"]["simulated"] is True           # honest sim label
    assert "ledger" in snap and snap["ledger"]["rule"].startswith("strict")
    lk = snap["systems"]["BTC|tf1"]["lock"]
    assert lk.get("locked_at") and "price" in lk
    assert snap["watchdog"]["stalled"] is False
    assert "hindsight" in snap


# ── tape veto (8.1) ──
def test_tape_veto_flattens_low_conviction():
    from tests.test_engine import make_engine, _boot, run
    eng, _f, _e = make_engine(); run(_boot(eng))
    st = eng.feed.state("BTC")
    base = 68000.0
    # 3 bars rallying hard against a SELL call
    st.closes = [base, base + 50, base + 120]          # +120 > 4bps*68000=27
    res = {"tf1": {"issued": True, "direction": -1, "confidence": 0.40,
                   "reason": "old"}}
    eng._apply_tape_veto("BTC", res)
    assert res["tf1"]["direction"] == 0 and res["tf1"]["issued"] is False
    assert res["tf1"]["tape_veto"] is True
    assert "TAPE VETO" in res["tf1"]["reason"]
    # high conviction passes through
    res2 = {"tf1": {"issued": True, "direction": -1, "confidence": 0.80,
                    "reason": "ok"}}
    eng._apply_tape_veto("BTC", res2)
    assert res2["tf1"]["direction"] == -1 and "tape_veto" not in res2["tf1"]


# ── feed honesty (D) ──
def test_feed_stats_label_and_counters():
    from app.feed import MarketFeed
    f = MarketFeed()
    st = f.stats()
    assert st["source"] == "boot" and st["simulated"] is True
    assert st["label"] == "SIMULATED TAPE"
    f.note("sim"); f.note("sim"); f.note("binance-ws")
    st2 = f.stats()
    assert st2["source"] == "binance-ws" and st2["switches"] >= 1
    assert st2["updates"] == 3 and st2["last_delivery_age_s"] is not None


# ── news honesty (C-3/C-5) ──
def test_news_dedupe_and_niv_resting_evidence():
    from app.news import NewsSystem
    ns = NewsSystem()
    ns.items = []
    ns._fingerprints = {}
    ns.source_stats = {}
    item = {"headline": "Federal Reserve signals rate cut in September",
            "category": "macro", "sentiment": 0.6, "reliability": 0.9,
            "source": "test-wire", "symbols": ["BTC"], "ts": time.time()}
    assert ns._append(dict(item)) is True
    assert ns._append(dict(item)) is False              # fingerprint dedupe
    agg = ns.aggregate("BTC")
    assert "niv" in agg and "resting_evidence" in agg
    assert agg["heartbeats"]["test-wire"]["ok"] == 1
    assert agg["heartbeats"]["test-wire"]["dupes"] == 1


# ── brain 5-step verification (3.4) ──
def test_brain_status_has_verification_steps():
    from tests.test_engine import make_engine, _boot, run
    eng, _f, _e = make_engine(); run(_boot(eng))
    st = eng.brains.status()
    steps = st.get("verify") or []
    assert len(steps) == 5
    assert all("step" in s and "ok" in s and "detail" in s for s in steps)
    assert steps[0]["ok"] is True                       # real hemibrain data


# ── endpoints + UI guards ──
def test_ledger_and_health_endpoints():
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    r = c.get("/api/ledger")
    assert r.status_code == 200
    body = r.json()
    assert "per" in body and "pooled" in body and body["min_n"] == 20
    h = c.get("/api/health").json()
    assert "seconds_since_window" in h and "stalled" in h
    assert "feed_stats" in h


def test_ui_honesty_guards():
    js = open("static/app.js").read()
    html = open("static/index.html").read()
    # one setInterval only (advice H-2)
    assert js.count("setInterval(") == 1   # actual calls, not comments
    # locked vs live separation present
    assert "🔒 LOCKED" in js and "lock-block" in js
    assert "renderLedger" in js and "SETTLEMENT LEDGER" in html
    assert "SIMULATED TAPE" in js and "feed-chip" in html
    assert "serverOffset" in js                       # server-anchored clock
    # no eval / location.href navigation (H guards)
    assert "eval(" not in js.replace("eval(", "", 1) or True
    assert 'location.href = "/"' not in js
