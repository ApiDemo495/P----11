"""Analysis recheck gate: a prediction is produced ONLY after every
data source and subsystem is rechecked. Anything less is NO TRADE
with the failed checks named — the old finale produced a SELL from
conf 0.001 with TP = entry."""
from app.engine import _recheck_prediction

GOOD = dict(direction=1, conf=0.42, floor=0.35, feed_age_s=2.0,
            voters={"fused": 1, "emotion": 1, "physics": 1, "news": -1,
                    "hedge": 1},
            levels_ok=True, ratio=2.0, lock_dir=1, tfs_computed=4)


def test_all_checks_pass_produces_prediction():
    r = _recheck_prediction(**GOOD)
    assert r["passed"] and r["failed"] == []


def test_subfloor_confidence_is_no_trade():
    r = _recheck_prediction(**{**GOOD, "conf": 0.001})
    assert not r["passed"] and "confidence above floor" in r["failed"]


def test_contradicting_the_lock_is_blocked():
    r = _recheck_prediction(**{**GOOD, "direction": -1,
                               "voters": {"fused": -1, "emotion": -1,
                                          "physics": -1, "news": -1,
                                          "hedge": -1}})
    assert not r["passed"] and "agrees with locked call" in r["failed"]


def test_split_vote_and_stale_feed_blocked():
    r = _recheck_prediction(**{**GOOD, "feed_age_s": 99,
                               "voters": {"fused": 1, "emotion": -1,
                                          "physics": -1, "news": 0,
                                          "hedge": 1}})
    assert not r["passed"]
    assert "live data" in r["failed"] and "subsystem consensus" in r["failed"]


def test_broken_levels_blocked():
    r = _recheck_prediction(**{**GOOD, "levels_ok": False, "ratio": 1.2})
    assert not r["passed"] and "levels valid + 2:1" in r["failed"]
