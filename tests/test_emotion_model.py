"""Contract for the literature-grounded affect model (app/emotion.py).

Replaces the old bespoke channel formulas with published constructs:
Russell (1980) core affect, Mehrabian & Russell (1974) PAD, PANAS,
Ekman (1992) basic emotions, prospect theory (λ = 2.25), Damasio's somatic
marker, CSAD herding, BGS extrapolation, Frijda action tendencies and the
alternative.me Fear & Greed composition — measured over the components this
app can actually measure, with the rest declared missing.
"""
import math

import pytest


def _ctx(closes, market=None):
    from tests.test_part2 import make_ctx, _market_for
    return (make_ctx(closes, W=22), market if market is not None else
            _market_for(closes))


def _em():
    from app.emotion import EmotionSystem
    return EmotionSystem()


def _noisy(seed, drift, vol, n=200, start=100.0, end=None):
    import random
    rng = random.Random(seed)
    p, out = start, []
    for _ in range(n):
        p *= math.exp(rng.gauss(drift, vol))
        out.append(p)
    return out + ([end] if end is not None else [])


def _run(closes, news=("polarity", 0.0, 0)):
    ctx, market = _ctx(closes)
    return _em().compute(ctx, market,
                         {"hormones_avg": {"serotonin": 0.6}},
                         {"polarity": news[1], "count": news[2]}, None)


def test_every_channel_carries_a_published_citation():
    from app.emotion import CHANNELS, MODEL
    assert len(CHANNELS) == 21            # E22 is the CERC composite
    for cid, meta in CHANNELS.items():
        key, name, cite, neutral = meta
        assert key and name and cite, f"channel {cid} is incomplete"
        assert 0.0 <= neutral <= 1.0
        # the UI measures against the channel's own neutral, so it ships
        shipped = MODEL["channels"][str(cid)]
        assert shipped["neutral"] == neutral and shipped["name"] == name
    assert "Russell" in " ".join(MODEL["citations"])
    assert "Ekman" in " ".join(MODEL["citations"])


def test_net_emotion_is_bounded_and_signed():
    for tape in (_noisy(1, 0.0015, 0.004), _noisy(2, -0.0015, 0.006),
                 _noisy(3, 0.0, 0.006), _noisy(4, 0.0, 0.02)):
        out = _run(tape)
        assert -1.0 <= out["net_emotion"] <= 1.0, out["net_emotion"]
        assert -1.0 <= out["cerc"] <= 1.0


def test_momentum_channel_tracks_the_actual_gap():
    """The old `(live − MA30)/(σ√n · live)` divided by the price twice and
    read ~0.35 on a roaring uptrend."""
    up = _run(_noisy(12, 0.0022, 0.0025))["emotions"]
    dn = _run(_noisy(13, -0.0022, 0.0025))["emotions"]
    assert up[18] > 0.80, up[18]
    assert dn[18] < 0.20, dn[18]


def test_primary_emotion_discriminates_bull_from_bear():
    bull = _run(_noisy(7, 0.0018, 0.0035))
    bear = _run(_noisy(8, -0.0018, 0.0045))
    assert bull["core_affect"]["valence"] > 0.2
    assert bear["core_affect"]["valence"] < -0.2
    assert bull["primary_emotion"]["name"] == "enjoyment"
    assert bear["primary_emotion"]["name"] in ("anger", "fear", "sadness",
                                               "disgust")
    assert bull["action_tendency"]["direction"] == 1
    assert bull["action_tendency"]["name"] == "approach"


def test_dominant_is_named_and_graded_not_pinned():
    from app.emotion import CHANNELS
    out = _run(_noisy(10, 0.0, 0.01))
    dom = out["dominant"]
    names = {m[1] for m in CHANNELS.values()} | {"Net emotion"}
    assert dom["name"] in names, dom["name"]
    assert dom["cite"]
    # graded inside [0,1]: a pinned 1.000 crown means the channel saturated
    assert 0.0 <= dom["intensity"] < 1.0, dom["intensity"]
    assert dom["runner_up"]["name"] != dom["name"]
    assert dom["intensity"] >= dom["runner_up"]["intensity"]


def test_dominant_ranks_on_deviation_from_its_own_neutral():
    from app.emotion import CHANNELS
    out = _run(_noisy(10, 0.0, 0.01))
    devs = {i: abs(float(out["emotions"][i]) - CHANNELS[i][3])
            for i in range(1, 22)}
    win = max(devs, key=lambda i: devs[i])
    assert out["dominant"]["id"] == win
    assert out["dominant"]["intensity"] == pytest.approx(devs[win], abs=2e-3)


def test_fear_greed_only_claims_what_it_measures():
    fg = _run(_noisy(11, 0.001, 0.005))["fear_greed"]
    assert set(fg["components"]) == {"volatility", "momentum_volume"}
    assert fg["measured_weight"] == pytest.approx(0.5)
    # alternative.me's dominance slot isBTC market-cap share — NOT our breadth
    assert "dominance" in fg["unmeasured"]
    assert "breadth_component" in fg
    assert 0.0 <= fg["index"] <= 100.0
    assert fg["label"] in ("Extreme fear", "Fear", "Neutral", "Greed",
                           "Extreme greed")


def test_fear_greed_moves_with_the_tape():
    greed = _run(_noisy(12, 0.0022, 0.0025))["fear_greed"]["index"]
    fear = _run(_noisy(13, -0.0022, 0.0025))["fear_greed"]["index"]
    assert greed > fear + 15, (greed, fear)
    assert greed > 50 > fear


def test_ekman_prototypes_are_separated_states():
    from app.emotion import ekman_scores
    # channel space: 0…1 with 0.5 neutral (v, a, d)
    calm_good = ekman_scores(0.925, 0.65, 0.75)
    panic = ekman_scores(0.10, 0.95, 0.15)
    assert calm_good["enjoyment"]["intensity"] > 0.5
    assert panic["fear"]["intensity"] > 0.5
    assert calm_good["fear"]["intensity"] < 0.1
    assert panic["enjoyment"]["intensity"] < 0.1


def test_missing_inputs_are_declared_not_faked():
    out = _run(_noisy(14, 0.0, 0.005), news=("polarity", 0.0, 0))
    zr = out["zero_reasons"]
    assert "20" in zr and "headlines" in zr["20"].lower()
    # somatic marker needs settled outcomes: says so instead of a number
    assert "13" in zr
    # herding needs a peer cross-section
    assert "14" in zr
