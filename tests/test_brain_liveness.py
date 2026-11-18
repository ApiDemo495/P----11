"""The four real circuits must actually answer their inputs.

Root cause this file locks down: every readout was taken from raw
activations of a recurrent attractor, so the numbers described the wiring,
not the window — DB-3 returned the identical +0.3059 for opposite inputs,
DB-4 sat at ±0.04 because its readout population was never driven, and the
DB-2 readout was pinned. The fix reads the INPUT-DRIVEN response
(driven − undriven) and adds an abstain path, a real softmax posterior for
DB-3 and per-formula deltas for DB-4.
"""
import numpy as np
import pytest


def _real_brains():
    from app.flywire import get_flywire
    fw = get_flywire()
    try:
        fw.ensure()
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"connectome unavailable: {exc!r}")
    if fw.state != "ready":
        pytest.skip(f"connectome not ready: {fw.state}")
    from app.real_brains import RealBrains
    return RealBrains(fw)


@pytest.fixture(scope="module")
def rb():
    return _real_brains()


def _probe(sign: float) -> dict:
    from app.real_brains import RealBrains
    return RealBrains._probe_inputs(sign)


def test_every_brain_separates_bull_from_bear(rb):
    """A brain that cannot tell a bullish window from a bearish one is not
    working, whatever its activation field looks like."""
    st = rb.self_test()
    assert set(st) == {"DB-1", "DB-2", "DB-3", "DB-4"}
    for bid, v in st.items():
        assert v["responds"], f"{bid} did not respond: {v}"
        assert v["flips"], f"{bid} did not flip sign: bull {v['bull']} bear {v['bear']}"
        assert v["separation"] > 0.05, f"{bid} barely separates: {v['separation']}"
        assert v["bull"] > 0 and v["bear"] < 0


def test_db3_posterior_is_a_real_distribution(rb):
    a = rb.evaluate_instance("DB-3", "__t__", "p", _probe(1.0), advance=False)
    b = rb.evaluate_instance("DB-3", "__t__", "p", _probe(-1.0), advance=False)
    for r in (a, b):
        dist = r["state_dist"]
        assert len(dist) == 5
        assert abs(sum(dist.values()) - 1.0) < 1e-6
        assert max(dist.values()) >= 0.2          # never below uniform
    assert a["dominant"] != b["dominant"] or abs(a["signal"] - b["signal"]) > 0.2
    assert a["signal"] * b["signal"] < 0          # opposite windows, opposite sides


def test_db4_readout_follows_its_own_input_and_splits_formulas(rb):
    """DB-4's input lands in DAN; it used to be read off DN (never driven),
    so every per-formula adjustment was 0.0."""
    from app.config import WEIGHT_IDS
    up = rb.evaluate_instance("DB-4", "__t__", "p", _probe(1.0), advance=False)
    dn = rb.evaluate_instance("DB-4", "__t__", "p", _probe(-1.0), advance=False)
    assert up["signal"] > 0 > dn["signal"]
    assert set(up["weight_adj"]) == set(WEIGHT_IDS)
    assert up["informative"] is True
    spread = max(abs(v) for v in up["weight_adj"].values())
    assert spread > 1e-4, "all per-formula deltas identical — controller blind"


def test_db1_reports_its_selectivity_honestly(rb):
    r = rb.evaluate_instance("DB-1", "__t__", "p", _probe(1.0), advance=False)
    assert r["signal"] > 0
    assert "selectivity" in r
    # measured: the approach/avoid split is ~1% of their common motion, so a
    # large selectivity would mean the wiring changed — say so loudly
    assert abs(r["selectivity"]) < 0.5


def test_zero_input_abstains_instead_of_inventing_a_side(rb):
    inert = {k: (0.0 if isinstance(v, float) else
                 ([0.0] * len(v) if isinstance(v, list) and v and
                  isinstance(v[0], float) else v))
             for k, v in _probe(1.0).items()}
    # a genuinely empty window: all inputs zero
    inert["formula_perf"] = {i: 0.0 for i in range(1, 36)}
    inert["cc_inputs"] = [0.0] * 5
    inert["norm_returns20"] = [0.0] * 20
    inert["tp_dist"] = inert["sl_dist"] = 0.0
    inert["news_value"] = 0.0
    for bid in ("DB-1", "DB-2", "DB-3", "DB-4"):
        r = rb.evaluate_instance(bid, "__t__", "p", inert, advance=False)
        assert r["input_energy"] == 0.0
        assert r["abstain"] is True, f"{bid} claimed a side with no input"
        assert r["signal"] == 0.0


def test_consensus_ignores_abstaining_brains():
    """`_evaluate_real` must not average in a brain that did not vote."""
    import types
    from app.brain import BrainSystem
    bs = BrainSystem.__new__(BrainSystem)          # no connectome build

    class FakeReal:
        def evaluate_instance(self, bid, symbol, tf, thing2, advance=True):
            sig = {"DB-1": 0.0, "DB-2": 0.6, "DB-3": 0.4, "DB-4": -0.9}[bid]
            return {"id": bid, "role": bid, "signal": sig, "h_signal": sig,
                    "hormones": {k: 0.5 for k in
                                 ("dopamine", "serotonin", "octopamine",
                                  "tyramine")},
                    "neurons": 1, "synapses": 1, "backend": "fake",
                    "ready": True, "input_energy": 0.0 if bid == "DB-1" else 1.0,
                    "response": 0.0 if bid == "DB-1" else 1.0,
                    "abstain": bid == "DB-1", "experience": 0.5,
                    "state_dist": None, "weight_adj": None,
                    "informative": True}

    bs.instances = [types.SimpleNamespace(id=f"DB-{i}", role="r", ready=True)
                    for i in range(1, 5)]
    bs._influence = {}
    bs._flywire_eval = lambda *a, **k: {"ready": False}
    out = bs._evaluate_real(FakeReal(), "BTC", "tf1", {}, False, None)
    assert out["voters"] == 3
    assert out["abstained"] == ["DB-1"]
    # (0.6 + 0.4 − 0.9) / 3, the abstainer is not in the denominator
    assert out["consensus"] == pytest.approx((0.6 + 0.4 - 0.9) / 3.0, abs=1e-4)
    assert out["polarity"] == 1
    assert out["agreement_count"] == "2/3"
    assert out["spread"] == pytest.approx(1.5, abs=1e-4)
