"""Sealed prediction locks: the contract behind ▶ ANALYSE.

Pins the behaviour the user asked for: a call becomes an immutable record,
one per (asset, timeframe, window), saved to history, settled by a measured
touch or the strict sign at expiry, and fed back into the learners.
"""
from __future__ import annotations

import json

import pytest

from app.predlock import PredictionJournal


def _j(tmp_path) -> PredictionJournal:
    return PredictionJournal(str(tmp_path / "locks.json"))


def test_seal_requires_a_signed_finite_ladder(tmp_path):
    j = _j(tmp_path)
    flat = j.seal("BTC", "tf1", direction=0, entry=100, tp=101, sl=99,
           confidence=0.4)
    assert flat["locked"] is False and flat["status"] == "refused"
    assert any("direction" in w.lower() or "flat" in w.lower()
               for w in flat["why"])
    wrong = j.seal("BTC", "tf1", direction=1, entry=100, tp=99, sl=101,
           confidence=0.4)
    assert wrong["locked"] is False
    assert wrong["why"] and "wrong side" in wrong["why"][0].lower()
    bad = j.seal("BTC", "tf1", direction=1, entry=100, tp=float("nan"), sl=99,
           confidence=0.4)
    assert bad["locked"] is False
    assert j.rows == []                       # a refusal writes nothing


def test_one_immutable_lock_per_window(tmp_path):
    j = _j(tmp_path)
    a = j.seal("BTC", "tf2", direction=1, entry=100, tp=104, sl=99, confidence=0.4,
               minutes=44, created_at=10_000)
    assert a["locked"] and not a["already"]
    b = j.seal("BTC", "tf2", direction=-1, entry=300, tp=200, sl=400, confidence=0.4,
               minutes=44, created_at=10_100)
    assert b["locked"] and b["already"]
    assert b["id"] == a["id"] and len(j.rows) == 1
    # a different tf is a different window
    c = j.seal("BTC", "tf3", direction=1, entry=100, tp=110, sl=95, confidence=0.4,
               minutes=130, created_at=10_100)
    assert c["locked"] and not c["already"]


def test_settle_uses_tp_before_sl_and_strict_sign_at_expiry(tmp_path):
    j = _j(tmp_path)
    win = j.seal("ETH", "tf1", direction=1, entry=100, tp=102, sl=99, confidence=0.4,
                 minutes=22, created_at=0)
    j.settle("ETH", "tf1", 102.5, now=60)
    row = next(r for r in j.rows if r["id"] == win["id"])
    assert row["outcome"] == "won" and row["settle_reason"].startswith("TP")

    loss = j.seal("ETH", "tf1", direction=-1, entry=100, tp=98, sl=101, confidence=0.4,
                  minutes=22, created_at=10_000)
    j.settle("ETH", "tf1", 101.2, now=10_060)
    row = next(r for r in j.rows if r["id"] == loss["id"])
    assert row["outcome"] == "lost"

    # expiry with no touch: sign decides
    push = j.seal("BNB", "tf1", direction=1, entry=100, tp=110, sl=90, confidence=0.4,
                  minutes=22, created_at=20_000)
    assert j.settle("BNB", "tf1", 100.0, now=21_320)
    row = next(r for r in j.rows if r["id"] == push["id"])
    assert row["outcome"] == "push"
    assert j.stats()["hit_rate"] == 0.5       # 1 won / 1 lost; the push is excluded


def test_extremes_decide_a_touch_inside_one_sample(tmp_path):
    j = _j(tmp_path)
    r = j.seal("BTC", "tf1", direction=1, entry=100, tp=102, sl=99, confidence=0.4,
               minutes=22, created_at=0)
    j.settle("BTC", "tf1", 100.4, high=102.5, low=98.5, now=60)
    row = next(x for x in j.rows if x["id"] == r["id"])
    assert row["outcome"] == "ambiguous"
    assert "both" in row["settle_reason"].lower()


def test_settled_lock_is_appended_to_history(tmp_path, monkeypatch):
    seen = []

    class DB:
        def append(self, record):
            seen.append(record)
            return record

    import app.historydb as hdb
    monkeypatch.setattr(hdb, "HistoryDB", lambda: DB())
    j = _j(tmp_path)
    r = j.seal("BTC", "tf1", direction=1, entry=100, tp=101, sl=99,
               confidence=0.4, minutes=22, created_at=0)
    j.settle("BTC", "tf1", 101.5, now=60)
    assert seen, "a settled lock must reach HistoryDB"
    assert seen[-1]["type"] == "prediction_lock"
    assert seen[-1]["id"] == r["id"]


def test_stats_are_scoped_and_persist_across_reload(tmp_path):
    path = tmp_path / "locks.json"
    j = PredictionJournal(str(path))
    j.seal("BTC", "tf1", direction=1, entry=100, tp=101, sl=99,
           confidence=0.4, minutes=22, created_at=0)
    j.settle("BTC", "tf1", 100.5, now=1_400)
    j.seal("ETH", "tf3", direction=-1, entry=100, tp=95, sl=105,
           confidence=0.4, minutes=130, created_at=10_000)
    j2 = PredictionJournal(str(path))
    assert len(j2.rows) == 2
    assert j2.stats("BTC", "tf1")["graded"] == 1
    assert j2.stats("ETH", "tf3")["open"] == 1
    assert json.loads(path.read_text())["version"] == 1


def test_refusal_names_every_failure(tmp_path):
    j = _j(tmp_path)
    out = j.seal("BTC", "tf1", direction=0, entry=0, tp=0, sl=0, confidence=0.0)
    assert out["locked"] is False
    assert len(out["why"]) >= 1 and "direction" in out["why"][0].lower()
