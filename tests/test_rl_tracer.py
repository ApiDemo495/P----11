"""RL controllers + Hedge-Flow Tracer acceptance.

User: 'use RL to find errors and fix them, boost accuracy (target 90%),
fix every formula/emotion/physics, add a hedge tracer with weightage'.

Honesty spine: the target tracker reports the REAL number + gap; RL
climbs toward 90% — it never rewrites history to fake reaching it.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from app.rlsys import RLSystem, damp_shares

ROOT = Path(__file__).resolve().parent.parent


def _seed_rings(engine, name: str, vals: list[float],
                symbol: str = "BTC", tf: str = "tf1") -> None:
    for v in vals:
        engine.hist.ring(engine.hist.tf_key(symbol, tf, name), 50).append(v)


def test_error_finder_damps_and_recovers():
    from tests.test_engine import make_engine
    rl = RLSystem(path=os.path.join(tempfile.mkdtemp(), "rl.json"))
    eng, _f, _e = make_engine()
    # a broken formula: pooled reward below 0.40 over >=8 windows
    _seed_rings(eng, "facc7", [-1.0] * 6 + [1.0] * 2 + [-1.0] * 2)
    taken = rl.audit(eng)
    assert rl.damp.get(7) == 0.75, rl.damp
    assert taken and taken[0]["kind"] == "formula"
    # still bad after the cooldown -> deepens (floor 0.4)
    _seed_rings(eng, "facc7", [-1.0] * 10, symbol="PAXG")
    rl.audit(eng)
    assert rl.damp[7] < 0.75 and rl.damp[7] >= 0.4
    # heals -> weight restored (isolated engine: pooled mean must be
    # able to clear REWARD_OK — the broken engine's old negatives would
    # keep dragging the pool down, which is correct behaviour)
    eng2, _f2, _e2 = make_engine()
    rl.active[7] = {"at": 0, "kind": "formula", "id": 7,
                    "reward": 0.2, "n": 8, "action": "damp",
                    "state": "active"}
    rl.damp[7] = 0.75
    _seed_rings(eng2, "facc7", [1.0] * 12)
    rl.audit(eng2)
    assert 7 not in rl.damp
    assert any(a.get("state") == "recovered" for a in rl.actions)
    # damp_shares keeps the pile normalised at 1.0 — it reads the GLOBAL
    # singleton (the same one cci.compute uses), so damp there and clean up
    from app.rlsys import get_rl
    g = get_rl()
    had = g.damp.get(97)
    g.damp[97] = 0.5
    try:
        sh = damp_shares({"core": 0.42, "brain": 0.09, "emotion": 0.14,
                          "physics": 0.14, "news": 0.07, "hf": 0.08,
                          "tracer": 0.06})
        assert sum(sh.values()) == pytest.approx(1.0, abs=1e-9)
        assert sh["brain"] < 0.09        # damped below its raw share
    finally:
        if had is None:
            g.damp.pop(97, None)
        else:
            g.damp[97] = had


def test_error_finder_persistence_and_version_guard():
    d = tempfile.mkdtemp()
    p1 = os.path.join(d, "rl.json")
    rl = RLSystem(path=p1)
    rl.damp[5] = 0.75
    rl.bandit["bucket"] = 3
    rl.save()
    rl2 = RLSystem(path=p1)
    assert rl2.damp.get(5) == 0.75
    assert rl2.bandit["bucket"] == 3
    # stale build -> drop (B-8)
    with open(p1) as fh:
        data = json.load(fh)
    data["build_version"] = -1
    with open(p1, "w") as fh:
        json.dump(data, fh)
    rl3 = RLSystem(path=p1)
    assert not rl3.damp and rl3.bandit["bucket"] == 2


def test_issue_bandit_scores_and_repicks():
    rl = RLSystem(path=os.path.join(tempfile.mkdtemp(), "rl.json"))
    assert rl.issue_threshold() == pytest.approx(0.35)   # legacy default
    # rival buckets with measured reward — the re-pick must be able to
    # leave the default bucket when a rival scores better
    rl.bandit["stats"]["0"] = {"n": 5, "reward": 4.0}    # mean +0.8
    rl.bandit["stats"]["2"] = {"n": 2, "reward": -2.0}   # default looks bad
    for _ in range(8):
        rl.on_settle(issued=True, won=True)
    assert rl.bandit["committed"] == 0          # re-picked after 8
    assert any(a.get("kind") == "bandit" for a in rl.actions)
    assert rl.bandit["bucket"] != 2             # left the bad default
    assert 0.25 <= rl.issue_threshold() <= 0.45
    # no-trade windows are opportunity-neutral (reward 0)
    n_before = rl.bandit["stats"][str(rl.bandit["bucket"])]["n"]
    rl.on_settle(issued=False, won=None)
    st = rl.bandit["stats"][str(rl.bandit["bucket"])]
    assert st["n"] == n_before + 1


def test_target_tracker_never_fakes_90():
    from tests.test_engine import make_engine
    rl = RLSystem(path=os.path.join(tempfile.mkdtemp(), "rl.json"))
    eng, _f, _e = make_engine()
    eng.ledger.path = os.path.join(tempfile.mkdtemp(), "led.json")
    eng.ledger.rows = {}
    st = rl.target_status(eng.ledger)
    assert st["target"] == 0.90 and st["n"] == 0
    assert "no settled windows" in st["status"]
    for _ in range(20):
        eng.ledger.settle("BTC", "tf1", 100.0, 101.0, 1, window_end=time.time())
        eng.ledger.settle("ETH", "tf1", 100.0, 99.0, 1, window_end=time.time())
    st2 = rl.target_status(eng.ledger)
    assert st2["n"] == 40
    assert st2["current"] == pytest.approx(0.5)
    assert st2["gap_pp"] == pytest.approx(40.0, abs=0.1)
    assert "TARGET NOT MET" in st2["status"]


def test_hedge_flow_tracer_sign_and_thin_tape():
    from app.tracer import hedge_flow_tracer, MIN_BARS
    from tests.test_engine import make_engine
    eng, feed, _e = make_engine()
    # thin tape -> honest zero with a reason
    v, d = hedge_flow_tracer(feed, "BTC")
    assert v == 0.0 and "reason" in d
    # seed bars: BTC suddenly absorbs MOST of the volume -> rotation >0
    for sym in ("BTC", "PAXG", "ETH", "BNB"):
        x = feed.states[sym]
        base = {"BTC": 80000.0, "PAXG": 4000.0,
                "ETH": 3000.0, "BNB": 600.0}[sym]
        for i in range(MIN_BARS):
            x.closes.append(base * (1 + 0.00001 * i))
            x.volumes.append(1.0 + (10.0 if (sym == "BTC"
                              and i >= MIN_BARS - 30) else 0.0))
            x.times.append(1700000000 + i * 60)
            x.opens.append(base)
            x.highs.append(base)
            x.lows.append(base)
    v2, d2 = hedge_flow_tracer(feed, "BTC")
    assert -1.0 <= v2 <= 1.0
    assert v2 > 0.0, (v2, d2)          # BTC volume share ROTATED up
    assert d2["share_now"] > d2["share_base"]
    assert "rel_ret" in d2


def test_tracer_enters_the_weighted_36():
    from app.config import (WEIGHT_IDS, TRACER_COMPONENT_ID,
                            N_WEIGHT_COMPONENTS, SUBSYSTEM_SHARES_INIT)
    from app.formulas import FORMULA_FUNCTIONS
    assert TRACER_COMPONENT_ID in WEIGHT_IDS
    assert N_WEIGHT_COMPONENTS == len(WEIGHT_IDS) == 36
    assert SUBSYSTEM_SHARES_INIT["tracer"] == pytest.approx(0.06)
    assert sum(SUBSYSTEM_SHARES_INIT.values()) == pytest.approx(1.0)
    # tracer is a SUBSYSTEM component — not a formula (registry stays 31)
    assert TRACER_COMPONENT_ID not in FORMULA_FUNCTIONS


def test_engine_rl_wiring_and_contracts():
    from tests.test_engine import make_engine, run, _boot
    eng, _f, _e = make_engine()
    # the global RL file is shared across tests in this process — pin the
    # bucket so the contract is deterministic (fresh default =0.35)
    eng.rl.bandit["bucket"] = 2
    eng.rl.bandit["committed"] = 0
    assert eng.rl.issue_threshold() == pytest.approx(0.35)  # legacy default
    run(_boot(eng))
    try:
        eng.compute_asset("BTC", record=False, display=True)
        snap = eng.build_snapshot()
        comps = ((snap["systems"]["BTC|tf1"].get("breakdown") or {})
                 .get("components") or [])
        assert any(str(c["id"]) == "102" for c in comps), \
            "hedge-flow tracer must enter the CCI component list"
    finally:
        run(eng.stop())
    # surface contracts
    main = (ROOT / "app" / "main.py").read_text()
    assert "/api/rl/status" in main
    html = (ROOT / "static" / "index.html").read_text()
    assert "rl-panel" in html
    js = (ROOT / "static" / "app.js").read_text()
    assert "loadRL" in js and "36 weighted components" in js
    cci = (ROOT / "app" / "cci.py").read_text()
    assert "damp_all" in cci
