"""RL deepening acceptance: winner boost, contextual bandit, Q-table.

All three learn ONLY from the same strict-sign settlement rewards as
the original error-finder/bandit. Fresh state must be exactly legacy:
factor 1.0, threshold 0.35, balanced mix.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from app.rlsys import (BOOST_CAP, MIX_ACTIONS, RLSystem, context_key,
                       window_state)


def _rl() -> RLSystem:
    return RLSystem(path=os.path.join(tempfile.mkdtemp(), "rl.json"))


def _seed(engine, name, vals, symbol="BTC", tf="tf1"):
    for v in vals:
        engine.hist.ring(engine.hist.tf_key(symbol, tf, name), 50).append(v)


# ---------------------------------------------------------- winner boost
def test_boost_earns_deepens_caps_and_withdraws():
    from tests.test_engine import make_engine
    rl = _rl()
    eng, _f, _e = make_engine()
    _seed(eng, "facc7", [1.0] * 9 + [-1.0])          # reward 0.8, n=10
    rl.audit(eng)
    assert rl.boost.get(7) == pytest.approx(1.10)
    assert rl.factor(7) == pytest.approx(1.10) and 7 not in rl.damp
    # sustained winning after the cooldown deepens, never past the cap
    for _ in range(12):
        _seed(eng, "facc7", [1.0] * 10, symbol="PAXG")
        rl.audit(eng)
        assert rl.boost[7] <= BOOST_CAP
    assert rl.boost[7] == pytest.approx(BOOST_CAP)
    # reward fades -> boost withdrawn BEFORE any damp is applied
    eng2, _f2, _e2 = make_engine()
    _seed(eng2, "facc7", [1.0] * 4 + [-1.0] * 6)     # reward -0.2
    rl.audit(eng2)
    assert 7 not in rl.boost and 7 not in rl.damp    # withdrawn, not damped
    assert any(a.get("state") == "boost-withdrawn" for a in rl.actions)
    # and only NOW, still bad on a fresh engine, does the damp engage
    rl.audit(eng2)
    assert rl.damp.get(7) == pytest.approx(0.75)


def test_boost_never_stacks_on_a_damped_component():
    from tests.test_engine import make_engine
    rl = _rl()
    eng, _f, _e = make_engine()
    _seed(eng, "facc9", [-1.0] * 9 + [1.0])
    rl.audit(eng)
    assert rl.damp.get(9) == pytest.approx(0.75) and 9 not in rl.boost


# ------------------------------------------------------ contextual bandit
def test_contextual_bandit_learns_per_context(monkeypatch):
    import random as _random
    monkeypatch.setattr(_random, "random", lambda: 0.99)  # never explore
    rl = _rl()
    ctx_a = context_key("BTC", "tf1", "highvol")
    ctx_b = context_key("BNB", "tf4", "lowvol")
    # unseen contexts fall back to the global (legacy) threshold
    assert rl.issue_threshold(ctx_a) == pytest.approx(0.35)
    # context A: committed bucket keeps winning -> after COMMIT_N the
    # context re-picks on ITS OWN stats; global stats learned too
    rl.contexts[ctx_a] = RLSystem._fresh_bandit()
    rl.contexts[ctx_a]["stats"]["4"] = {"n": 5, "reward": 5.0}  # mean +1.0
    # the committed bucket scores worse (+0.5) over the commitment, so
    # the context's own re-pick must move it to the better bucket 0.45
    for won in (True, True, True, True, True, True, False, False):
        rl.on_settle(issued=True, won=won, context=ctx_a)
    assert ctx_a in rl.contexts
    assert rl.contexts[ctx_a]["committed"] == 0       # re-picked
    assert rl.issue_threshold(ctx_a) == pytest.approx(0.45)
    # context B never settled a window: still the global bucket
    assert ctx_b not in rl.contexts
    assert rl.issue_threshold(ctx_b) == rl.issue_threshold()
    # global bandit scored every one of those windows as well
    assert rl.bandit["stats"]["2"]["n"] >= 8


# --------------------------------------------------------------- Q-table
def test_window_state_buckets():
    assert window_state(1.6, 2.0) == "highvol-trend"
    assert window_state(1.6, 0.2) == "highvol-chop"
    assert window_state(0.5, 0.0) == "lowvol-chop"
    assert window_state(1.0, 1.5) == "midvol-trend"


def test_qtable_cold_state_plays_balanced_then_learns():
    rl = _rl()
    # cold state: legacy mix, no guessing
    assert rl.choose_mix("highvol-trend", "BTC|tf1") == "balanced"
    # feed decided rewards: physics-heavy wins, balanced loses
    for _ in range(4):
        rl.q_on_settle("highvol-trend", "physics-heavy", 1.0)
        rl.q_on_settle("highvol-trend", "balanced", -1.0)
    cell = rl.q["highvol-trend"]
    assert cell["physics-heavy"]["q"] > 0 and cell["balanced"]["q"] < 0
    # greedy pick (exploration aside) must now favour physics-heavy:
    # sample many choices, the learned action dominates
    picks = [rl.choose_mix("highvol-trend", "BTC|tf1") for _ in range(60)]
    assert picks.count("physics-heavy") > picks.count("balanced")
    # undecided windows never grade the mix
    rl.q_on_settle("highvol-trend", "physics-heavy", None)
    assert rl.q["highvol-trend"]["physics-heavy"]["n"] == 4
    # mix application renormalises to 1.0 and actually tilts physics up
    base = {"core": 0.42, "brain": 0.09, "emotion": 0.14, "physics": 0.14,
            "news": 0.07, "hf": 0.08, "tracer": 0.06}
    mixed = RLSystem.mix_shares(base, "physics-heavy")
    assert sum(mixed.values()) == pytest.approx(1.0)
    assert mixed["physics"] > base["physics"]
    assert RLSystem.mix_shares(base, "balanced") == base
    assert set(MIX_ACTIONS) >= {"balanced", "core-heavy", "physics-heavy",
                                "sentiment-heavy", "brain-heavy"}


def test_deep_state_persists_and_version_guard_still_drops_stale():
    import json
    d = tempfile.mkdtemp()
    p = os.path.join(d, "rl.json")
    rl = RLSystem(path=p)
    rl.boost[3] = 1.21
    rl.q_on_settle("lowvol-chop", "core-heavy", 1.0)
    rl.on_settle(issued=True, won=True, context="ETH|tf2|midvol")
    rl.save()
    rl2 = RLSystem(path=p)
    assert rl2.boost.get(3) == pytest.approx(1.21)
    assert rl2.q["lowvol-chop"]["core-heavy"]["n"] == 1
    assert "ETH|tf2|midvol" in rl2.contexts
    st = rl2.status()
    assert st["boost"] and st["qtable"] and st["contexts"]
    assert st["factors"]["3"] == pytest.approx(1.21)
    with open(p) as fh:
        data = json.load(fh)
    data["build_version"] = -1
    with open(p, "w") as fh:
        json.dump(data, fh)
    rl3 = RLSystem(path=p)
    assert not rl3.boost and not rl3.q and not rl3.contexts


def test_engine_wiring_deep_rl():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eng = open(os.path.join(root, "app", "engine.py")).read()
    assert "choose_mix" in eng and "q_on_settle" in eng
    assert '"rl_state"' in eng and '"rl_ctx"' in eng
    js = open(os.path.join(root, "static", "app.js")).read()
    assert "WINNER BOOSTS" in js and "CONTEXTUAL BANDITS" in js
    assert "Q-TABLE" in js
