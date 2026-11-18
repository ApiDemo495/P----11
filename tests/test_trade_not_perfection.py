"""Not-perfection: the gate must trade on measured edge, not unanimity.

Reports that drove this file:

* "recheck system not working it always say no trade ... like pessimistic
  model, it always seems negativity" — the analysis gate was a flat AND of
  seven checks, so one soft disagreement blocked every call;
* "trades are changing in mid time" — the simulator's live 1 Hz step used
  PER-MINUTE drift/vol per second (60x too fast), which hit TP/SL within
  seconds and let synthetic news drive an emergency re-lock every minute;
* "chart price and analyse price aren't matching" — the board never said
  which instant each price belonged to.
"""
import math
import random
import time

from app.engine import _recheck_prediction

GOOD = dict(direction=1, conf=0.42, floor=0.35, feed_age_s=2.0,
            voters={"fused": 1, "emotion": 1, "physics": 1, "news": -1,
                    "hedge": 1},
            levels_ok=True, ratio=2.0, lock_dir=1, tfs_computed=4)


def test_a_split_vote_near_the_floor_still_trades():
    """The market is never unanimous: 2/5 agreeing at 0.30/0.35 must be
    callable (smaller size is the tier's job, not a veto)."""
    r = _recheck_prediction(**{**GOOD, "conf": 0.30, "lock_dir": 0,
                               "voters": {"fused": 1, "emotion": -1,
                                          "physics": -1, "news": 0,
                                          "hedge": 1}})
    assert r["passed"], r["verdict"]
    assert r["score"] >= r["threshold"]
    # …and it is honest about what was not perfect
    assert "subsystem consensus" in r["failed"]
    assert "confidence above floor" in r["failed"]


def test_no_conviction_is_still_refused():
    """Half the floor is not an opinion, it is noise."""
    r = _recheck_prediction(**{**GOOD, "conf": 0.10})
    assert not r["passed"]
    assert "conviction within reach of the floor" in r["hard_failed"]
    assert "NOT" not in r["verdict"] or "NO TRADE" in r["verdict"]


def test_verdict_names_what_would_have_to_change():
    r = _recheck_prediction(**{**GOOD, "conf": 0.30, "ratio": 0.7})
    assert not r["passed"]
    assert "levels valid + ratio ≥ 1:1" in r["hard_failed"]
    assert r["threshold"] and r["score"] is not None


def test_synthetic_news_can_never_relock_a_trade():
    from app.news import NewsSystem

    ns = NewsSystem.__new__(NewsSystem)          # no network, no threads
    now = time.time()
    ns.items = [
        {"ts": now, "source": "sim", "synthetic": True, "reliability": 0.95,
         "relevance_by_asset": {"BTC": 0.9}, "headline": "MARKET CRASH sim"},
        {"ts": now, "source": "seed", "synthetic": True, "reliability": 0.92,
         "relevance_by_asset": {"BTC": 0.8}, "headline": "HALT trading seed"},
    ]
    assert ns.critical("BTC") == []                 # cannot trigger
    assert len(ns.critical("BTC", include_synthetic=True)) == 2  # display only


def test_simulator_live_step_is_calibrated_per_minute():
    """60 one-second steps must produce the documented per-minute sigma."""
    from app.feed import SIM_BASE_VOL_MIN

    rng = random.Random(11)
    for sym in ("BTC", "ETH"):
        sigma_1s = SIM_BASE_VOL_MIN[sym] / math.sqrt(60.0)
        p = 100.0
        marks = []
        for _ in range(2000):
            for _s in range(60):
                p *= math.exp(rng.gauss(0.0002 / 60.0, sigma_1s))
            marks.append(p)
        rets = [(b - a) / a for a, b in zip(marks, marks[1:])]
        mu = sum(rets) / len(rets)
        sd = (sum((r - mu) ** 2 for r in rets) / len(rets)) ** 0.5
        assert abs(sd - SIM_BASE_VOL_MIN[sym]) < 0.25 * SIM_BASE_VOL_MIN[sym], (
            f"{sym}: 1-min sigma {sd:.6f} vs target {SIM_BASE_VOL_MIN[sym]}")


def test_simulator_regimes_are_stored_in_minutes():
    src = open("app/feed.py").read()
    assert "reg[\"left\"] -= 1.0 / 60.0" in src        # 1 Hz -> minutes
    assert "reg[\"drift\"] / 60.0" in src              # drift per minute
    assert "SIM_BASE_VOL_MIN[sym] / math.sqrt(60.0)" in src
    assert "SIM_REGIME_MIN" in src


def test_tier_never_silently_defaults_to_a_coin_flip():
    src = open("app/engine.py").read()
    seg = src[src.index("def _tier_for"):src.index("def _direction_ladder")]
    assert "persisted |score| ring" in seg          # survives restarts
    assert "voter-agreement purity" in seg          # measured fallback
    assert "calib_source" in seg
    assert "tier_rule" in seg


def test_snapshot_reports_price_provenance():
    src = open("app/engine.py").read()
    assert '"price_asof"' in src and '"delta_bps"' in src
    js = open("static/app.js").read()
    assert "PRICE PROVENANCE" in js
    assert "same tape, one instant behind" in js


def test_insignificant_negative_expectancy_does_not_veto():
    """-21 bp/window with t ≈ -1.3 is noise, not evidence: it must not shut
    the gate (the "always NO TRADE" report), but it must be printed."""
    r = _recheck_prediction(**{**GOOD, "exp_bps": -21.0, "exp_n": 60,
                               "exp_sd": 124.0, "exp_bps_all": -7.5,
                               "exp_n_all": 437})
    assert "measured expectancy (settled ledger)" not in r["hard_failed"]
    assert "measured expectancy (settled ledger)" in r["failed"]
    _exp = [c for c in r["checks"]
            if c["name"] == "measured expectancy (settled ledger)"][0]
    assert "not significant" in _exp["detail"]
    assert "t=-1.31" in _exp["detail"]
    assert r["passed"], r["verdict"]

    # a significant negative edge still blocks
    r2 = _recheck_prediction(**{**GOOD, "exp_bps": -21.0, "exp_n": 400,
                                "exp_sd": 100.0})
    assert "measured expectancy (settled ledger)" in r2["hard_failed"]
    assert not r2["passed"]
    assert "measured expectancy" in r2["verdict"]
    assert "significant" in r2["verdict"]      # the reason, not a bare label


def test_tier_why_is_never_a_bare_label():
    """A watch-only window must carry the measured reason it was cut."""
    src = open("app/engine.py").read()
    assert '"tier_why"' in src
    assert "noise floor" in src and "tape veto" in src
