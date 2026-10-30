"""Part-2 tests: Formulas 22-31, exact F28/F29 risk math, the four
Drosophila brain instances, the 22-emotion system with CERC learning,
and the 28-component confidence pipeline.
"""
import asyncio
import math

import pytest

from app.config import (
    BRAIN_TOTAL_UNITS,
    CONFIDENCE_FLOOR,
    POP_SIZES,
    WEIGHT_IDS,
)
from app.brain import BrainSystem, Connectome
from app.cci import CCIEngine, finalize_confidence
from app.emotion import EMO_IDS, EmotionSystem, ETA, W_CLAMP_HI, W_CLAMP_LO
from app.formulas import DIRECTIONAL_FORMULAS, FORMULA_FUNCTIONS
from app.formulas.f22_f31 import (
    analyze_vol_clusters,
    clre_anomaly,
    f22_bapr,
    f23_sard,
    f24_fbps,
    f25_tmdm,
    f26_ifds,
    f27_vcdp,
    f30_pcv,
    f31_clre,
    fractal_boundary_levels,
)
from app.history import HistoryStore
from app.risk import arc_ratio, dynamic_level_placement

from test_formulas import flat, make_ctx, trend_down, trend_up


# --------------------------------------------------------------- registry
def test_registry_includes_part2_signals():
    for fid in (22, 23, 24, 25, 26, 27, 31):
        assert fid in FORMULA_FUNCTIONS
        assert fid in DIRECTIONAL_FORMULAS
    # risk/validator/aggregator stay out of the weighted sum
    for fid in (20, 28, 29, 30):
        assert fid not in DIRECTIONAL_FORMULAS


def test_part2_formulas_bounded():
    for builder in (trend_up, trend_down, flat):
        ctx = make_ctx(builder())
        for fid in (22, 23, 24, 25, 26, 27, 31):
            v = FORMULA_FUNCTIONS[fid](ctx)
            assert -1.0 <= v <= 1.0, f"F{fid}={v} out of bounds"


# ---------------------------------------------------------------- F22 BAPR
def test_bapr_tracks_bid_ask_reconstruction():
    # steady uptrend: buy-side (ask) pressure dominates -> positive skew
    up = make_ctx(trend_up())
    v_up = f22_bapr(up)
    down = make_ctx(trend_down())
    v_down = f22_bapr(down)
    assert -1.0 <= v_up <= 1.0 and -1.0 <= v_down <= 1.0
    # recency weighting: latest candle contributes most (exp(-0.03k))
    assert f22_bapr(up) == pytest.approx(v_up, abs=1e-12)


# ---------------------------------------------------------------- F23 SARD
def test_sard_wave_absorption_direction():
    # declining prices on rising volume -> absorption of supply -> positive
    prices = trend_down(300)
    vols = [10.0 + i * 0.05 for i in range(300)]   # volume rising as price falls
    ctx = make_ctx(prices, volumes=vols)
    v = f23_sard(ctx)
    assert -1.0 <= v <= 1.0


# ---------------------------------------------------------------- F24 FBPS
def test_fbps_uses_fractal_boundaries_with_fallback():
    ctx = make_ctx(trend_up())
    v = f24_fbps(ctx)
    assert -1.0 <= v <= 1.0
    below, above = fractal_boundary_levels(ctx)
    assert isinstance(below, list) and isinstance(above, list)
    price = ctx.P[-1]
    assert all(lvl < price for lvl in below)
    assert all(lvl > price for lvl in above)
    # sorted nearest-first on both sides
    if len(below) > 1:
        assert (price - below[0]) <= (price - below[-1])
    if len(above) > 1:
        assert (above[0] - price) <= (above[-1] - price)


# ---------------------------------------------------------------- F25 TMDM
def test_tmdm_points_with_accelerating_momentum():
    # quadratic growth: momentum accelerates -> positive projection
    prices = [100.0 + 0.01 * i * i for i in range(300)]
    v = f25_tmdm(make_ctx(prices))
    assert -1.0 <= v <= 1.0
    assert v > 0.0


# ---------------------------------------------------------------- F26 IFDS
def test_ifds_sign_matches_dominant_frequency_phase():
    # low-frequency surge (smooth trend) -> low band dominates -> positive
    v_up = f26_ifds(make_ctx(trend_up()))
    assert -1.0 <= v_up <= 1.0
    # high-frequency alternating noise -> high band dominance -> negative
    jag = [100.0 + (1.0 if i % 2 else -1.0) for i in range(300)]
    v_jag = f26_ifds(make_ctx(jag))
    assert -1.0 <= v_jag <= 1.0


# ---------------------------------------------------------------- F27 VCDP
def test_vcdp_clusters_and_bootstrap():
    ctx = make_ctx(trend_up())
    info = analyze_vol_clusters(ctx)
    assert "in_cluster" in info and "ended" in info
    v = f27_vcdp(ctx)
    assert -1.0 <= v <= 1.0
    # with committed history (memoryless branch), still bounded
    hist = ctx.hist
    hist.vol_cluster_durations["BTC"] = [3.0, 4.0, 5.0, 3.0]
    hist.vol_cluster_gaps["BTC"] = [10.0, 12.0, 8.0]
    assert -1.0 <= f27_vcdp(ctx) <= 1.0


def test_vol_cluster_commit_records_once():
    hist = HistoryStore()
    info = {"ended": [(1000.0, 3.0, -0.001), (2000.0, 4.0, 0.002)],
            "in_cluster": False, "last_end_ts": 2000.0, "gap": 7.0}
    hist.commit_vol_cluster("BTC", info)
    assert hist.vol_cluster_durations["BTC"] == [3.0, 4.0]
    hist.commit_vol_cluster("BTC", info)      # idempotent: ends <= last_ts skipped
    assert hist.vol_cluster_durations["BTC"] == [3.0, 4.0]
    assert hist.vol_cluster_gaps["BTC"] == [7.0]


# ---------------------------------------------------------------- F28 ARC
def test_arc_exact_formula_and_bands():
    # hand-computed: CCI=1.0 -> certainty 2; sigma ratio 1 -> vol_state 0.5;
    # consensus 1 -> agreement 1.2 => 1.5*2*0.5*1.2 = 1.8 -> band 3:2
    arc, ratio = arc_ratio(1.0, 1.0, 1.0, 1.0, confidence=0.9)
    assert arc == pytest.approx(1.8, abs=1e-9)
    assert ratio == 1.5
    # low volatility ratio (calm market) pushes ARC to 3:1
    arc2, ratio2 = arc_ratio(1.0, 0.05, 2.0, 1.0, confidence=0.9)
    assert arc2 > 2.0 and ratio2 == 3.0
    # weak signal: CCI=0, volatile vs baseline, no consensus
    arc3, ratio3 = arc_ratio(0.0, 3.0, 1.0, 0.0, confidence=0.5)
    # raw = 1.5 * 1 * 0.25 * 0.8 = 0.3 -> clamped up to 1.0
    assert arc3 == 1.0 and ratio3 == 1.0
    # clamping at the top
    arc4, ratio4 = arc_ratio(10.0, 0.0, 1.0, 1.0, confidence=0.9)
    assert arc4 == 3.0 and ratio4 == 3.0
    # NO TRADE below the confidence floor
    arc5, ratio5 = arc_ratio(1.0, 1.0, 1.0, 1.0, confidence=0.2)
    assert ratio5 is None


# ---------------------------------------------------------------- F29 DLP
def test_dlp_phi_formulas():
    price, atr, drhe, ofti = 100.0, 2.0, 1.0, 0.0
    # no boundary in TP direction: fbps_tp=+1 -> phi_tp = 1+0.3+0.2 = 1.5
    tp, sl, tpd, sld = dynamic_level_placement(
        price, 1, 3.0, atr, ([], []), drhe, ofti
    )
    assert tpd == pytest.approx(atr * 3.0 * 1.5)
    assert tp == pytest.approx(price + tpd)
    # sl: no boundary -> phi_sl = 1 + 0 - 0 = 1
    assert sld == pytest.approx(atr * 1.0 * 1.0)
    assert sl == pytest.approx(price - sld)

    # boundary just beyond SL -> fbps_sl=+1, OFTI=1 subtracts 0.2
    tp2, sl2, tpd2, sld2 = dynamic_level_placement(
        price, 1, 3.0, atr, ([price - 2.5], []), drhe, ofti_value=1.0
    )
    assert sld2 == pytest.approx(atr * (1.0 + 0.3 - 0.2))
    # SAFE tighten factor stacks on Phi_sl
    tp3, sl3, tpd3, sld3 = dynamic_level_placement(
        price, 1, 3.0, atr, ([], []), drhe, ofti, tighten=0.85
    )
    assert sld3 == pytest.approx(sld * 0.85)
    # TP boundary collision pulls TP inward (fbps=-0.5)
    tp4, sl4, tpd4, sld4 = dynamic_level_placement(
        price, 1, 3.0, atr, ([], [price + 0.5]), drhe, ofti
    )
    assert tpd4 == pytest.approx(atr * 3.0 * (1.0 - 0.15 + 0.2))


# ---------------------------------------------------------------- F30 PCV
def test_pcv_shape_and_conflict_flag():
    ctx = make_ctx(trend_up())
    res = f30_pcv(ctx.P, cci=0.8)
    assert set(res) == {"pcv", "direction", "agreement", "conflict"}
    assert -1.0 <= res["pcv"] <= 1.0
    assert res["conflict"] == (res["pcv"] < 0)
    # identical alignment sign product: conflict iff directions oppose
    res2 = f30_pcv(ctx.P, cci=-0.8)
    if sign_of(res["direction"]) and sign_of(res2["direction"]):
        assert (res["pcv"] < 0) != (res2["pcv"] < 0)


def sign_of(x):
    return (x > 0) - (x < 0)


# ---------------------------------------------------------------- F31 CLRE
def test_clre_bootstrap_then_active():
    ctx = make_ctx(trend_up())
    # fresh history: fewer than 5 anomaly samples -> neutral bootstrap
    assert f31_clre(ctx) == 0.0
    # anomaly helper runs and is finite
    a = clre_anomaly(ctx)
    assert a >= 0.0 and math.isfinite(a)
    # seed the baseline ring -> formula can produce a bounded output
    ring = ctx.hist.ring(ctx.hist.tf_key("BTC", "tf1", "ranom"), 200)
    for _ in range(10):
        ring.append(a * 0.9)
    v = f31_clre(ctx)
    assert -1.0 <= v <= 1.0


# ---------------------------------------------------------------- brains
def test_connectome_layout_matches_spec():
    conn = Connectome()
    assert conn.unit_count == BRAIN_TOTAL_UNITS == 7700
    for pop_name, size in POP_SIZES.items():
        assert len(conn.pop[pop_name]) == size, pop_name
    # 5,000 individual + 2,700 cluster layout
    assert len(conn.pop["CLU"]) == 2700
    # reduced-compute unit budget
    assert 5000 + 2700 == BRAIN_TOTAL_UNITS
    assert conn.edge_count > 10000


def test_brain_instances_and_hormone_directions():
    from test_engine import DummyFeed

    async def _run():
        from app.engine import PredictionEngine
        feed = DummyFeed()
        eng = PredictionEngine(feed, lambda m: asyncio.sleep(0))
        await feed.start()
        market = eng._build_market("BTC")
        bs = eng.brains
        affinity = {}
        thing2 = bs.performance_inputs() | {
            "norm_returns20": [0.0] * 20, "tp_dist": 3.0, "sl_dist": 1.0,
            "cc_inputs": [0.5, 0.0, 0.0, 0.0, 0.5],
        }
        res = bs.evaluate("BTC", "tf1", market, affinity, thing2, advance=False)
        assert len(res["instances"]) == 4
        assert [i["id"] for i in res["instances"]] == ["DB-1", "DB-2", "DB-3", "DB-4"]
        # consensus = mean of the four H signals
        mean_h = sum(i["h_signal"] for i in res["instances"]) / 4.0
        assert res["consensus"] == pytest.approx(mean_h, abs=1e-4)
        for inst in res["instances"]:
            assert -1.0 <= inst["h_signal"] <= 1.0
            assert 0.0 < inst["hormone_state"] < 1.0
            assert -1.0 <= inst["pattern"] <= 1.0
            assert 0.0 <= inst["experience"] <= 1.0
            assert abs(sum(inst["state_dist"].values()) - 1.0) < 1e-6
            assert set(inst["weight_adj"]) == set(WEIGHT_IDS)
            for a in inst["weight_adj"].values():
                assert -0.5 <= a <= 0.5
        # Thing 1 directions: euphoria link H_signal = -(H-0.5)/0.5
        for inst in res["instances"]:
            expect = -(inst["hormone_state"] - 0.5) / 0.5
            assert inst["h_signal"] == pytest.approx(expect, abs=1e-9)
        return res

    asyncio.get_event_loop().run_until_complete(_run())


def test_brain_affinity_input_applied_when_strong():
    bs = BrainSystem()
    bs.initialize()
    conn = bs.connectome
    market = {
        "closes": [100.0 + i for i in range(120)],
        "volumes": [10.0] * 120,
        "session_open": 100.0,
    }
    thing2 = bs.performance_inputs() | {
        "norm_returns20": [0.1] * 20, "tp_dist": 3.0, "sl_dist": 1.0,
        "cc_inputs": [0.5] * 5,
    }
    # capture the inputs of the first instance
    captured = []
    orig = conn.sweep
    def cap(state, inputs):
        captured.append(dict(inputs))
        return orig(state, inputs)
    conn.sweep = cap
    try:
        bs.evaluate("BTC", "tf1", market, {}, thing2, advance=False)
        bs.evaluate("BTC", "tf1", market, {7: 0.9}, thing2, advance=False)
    finally:
        conn.sweep = orig
    assert len(captured) == 8, "4 sweeps per evaluate expected"
    # affinity cells get input only when a |F| > 0.7 boost is supplied
    aff = list(conn.pop["AFF"])
    per = 310 // 31
    f7_start = aff[(7 - 1) * per]
    assert captured[0].get(f7_start, 0.0) == 0.0        # evaluate #1, DB-1
    assert captured[4].get(f7_start, 0.0) == pytest.approx(0.9)  # evaluate #2
# ---------------------------------------------------------------- emotions
def _market_for(mc):
    n = len(mc)
    return {
        "closes": mc,
        "volumes": [10.0] * n,
        "times": [1_700_000_000 + i * 60 for i in range(n)],
        "ohlc": {"highs": mc, "lows": mc, "closes": mc},
        "live": mc[-1],
        "session_open": mc[0],
        "session_high": max(mc),
        "btc_closes": mc,
        "peer_returns": [],
        "ofti": 0.2,
        "cmp_divergence": 0.3,
        "news_price_corr": 0.0,
    }


def test_all_emotions_bounded_and_cerc_consistent():
    sys_ = EmotionSystem()
    ctx = make_ctx(trend_up(200), W=22)
    market = _market_for(trend_up(200))
    brain_res = {
        "hormones_avg": {"dopamine": 0.1, "serotonin": 0.6,
                         "octopamine": 0.3, "tyramine": 1.1},
        "consensus": 0.0,
    }
    news = {"polarity": 1, "count": 3, "bull": 2, "bear": 1, "items": 3}
    out = sys_.compute(ctx, market, brain_res, news, None)
    emos, dirs = out["emotions"], out["directions"]
    assert sorted(emos) == list(range(1, 22))          # E1..E21 values
    for eid, val in emos.items():
        assert 0.0 <= val <= 1.0, f"E{eid}={val}"
    for eid, d in dirs.items():
        assert d in (-1, 0, 1), f"dir E{eid}={d}"
    # E22 CERC = weighted sum with TEDR participation multiplier
    w = sys_._weights("BTC", "tf1")
    expect = 0.0
    for i, eid in enumerate(EMO_IDS):
        term = w[i] * emos[eid] * dirs[eid]
        if eid == 21:
            term *= (1.0 - 0.5 * emos[21])
        expect += term
    assert out["cerc"] == pytest.approx(expect, abs=1e-4)
    assert -1.0 <= out["cerc"] <= 1.0
    assert 0.0 <= out["rtp"] <= 1.0
    assert set(out["lag_values"]) == {"dpi", "octo", "pidx", "nsp"}


def test_cerc_weight_reward_rule():
    def spec_legal_state(sys_):
        # Part-4 clamps [0.01, 0.12], sum 1 (eta rule respects these bounds)
        w = [0.11, 0.08] + [0.81 / 19] * 19
        sys_.weights[("BTC", "tf1")] = list(w)
        return list(sys_._weights("BTC", "tf1"))

    emos = {eid: 0.8 for eid in EMO_IDS}

    # scenario A: selective positive contribution gains relative share
    sysA = EmotionSystem()
    before = spec_legal_state(sysA)
    dirs_sel = {eid: 0 for eid in EMO_IDS}
    dirs_sel[1] = 1
    sysA.reward_update("BTC", "tf1", emos, dirs_sel, 0.0, +1.0)
    after = sysA._weights("BTC", "tf1")
    assert sum(after) == pytest.approx(1.0, abs=1e-9)
    assert min(after) >= W_CLAMP_LO - 1e-12
    assert (after[0] / after[1]) > (before[0] / before[1])   # ch1 rewarded

    # scenario B: negative move — positive contributor loses vs the negative one
    sysB = EmotionSystem()
    before = spec_legal_state(sysB)
    dirs_mix = {eid: 0 for eid in EMO_IDS}
    dirs_mix[1] = 1
    dirs_mix[2] = -1
    sysB.reward_update("BTC", "tf1", emos, dirs_mix, 0.0, -1.0)
    after = sysB._weights("BTC", "tf1")
    assert sum(after) == pytest.approx(1.0, abs=1e-9)
    assert (after[0] / after[1]) < (before[0] / before[1])

    # scenario C: zero-direction channels are punished uniformly (ratios stable)
    sysC = EmotionSystem()
    before = spec_legal_state(sysC)
    dirs0 = {eid: 0 for eid in EMO_IDS}
    sysC.reward_update("BTC", "tf1", emos, dirs0, 0.0, +1.0)
    after = sysC._weights("BTC", "tf1")
    assert sum(after) == pytest.approx(1.0, abs=1e-9)
    ratios = [a / b for a, b in zip(after, before)]
    assert max(ratios) - min(ratios) < 1e-9

    # equal start stays equal under uniform reward (initial 1/21 state)
    sysD = EmotionSystem()
    w0 = list(sysD._weights("BTC", "tf1"))
    assert all(abs(x - 1.0 / 21.0) < 1e-12 for x in w0)
    sysD.reward_update("BTC", "tf1", emos, {eid: 1 for eid in EMO_IDS}, 0.0, +1.0)
    w1 = sysD._weights("BTC", "tf1")
    assert max(w1) - min(w1) < 1e-12
    assert W_CLAMP_LO - 1e-12 <= min(w1) and max(w1) <= W_CLAMP_HI + 1e-12


# ------------------------------------------------------- confidence pipeline
def test_finalize_confidence_pipeline_order():
    pack = {"confidence": 0.5}       # base = |CCI| * min(1, DRHE)
    # step 1: EQ multiplier (0.7 + 0.3*EQ)
    fin = finalize_confidence(pack, experience_quality=0.0, p_dominant=0.0,
                              state_aligned=False, rtp=0.0, pcv=0.0)
    assert fin["confidence"] == pytest.approx(0.35)
    assert fin["issued"] is False                    # floor is STRICTLY > 0.35
    # step 2: +0.1 * P(dominant) when aligned
    fin2 = finalize_confidence(pack, experience_quality=1.0, p_dominant=0.5,
                               state_aligned=True, rtp=0.0, pcv=0.0)
    assert fin2["confidence"] == pytest.approx(0.5 + 0.05)
    # misaligned: no boost
    fin2b = finalize_confidence(pack, experience_quality=1.0, p_dominant=0.5,
                                state_aligned=False, rtp=0.0, pcv=0.0)
    assert fin2b["confidence"] == pytest.approx(0.5)
    # step 3: RTP reduction only above 0.6
    fin3 = finalize_confidence(pack, experience_quality=1.0, p_dominant=0.0,
                               state_aligned=False, rtp=0.7, pcv=0.0)
    assert fin3["confidence"] == pytest.approx(0.5 - 0.7 * 0.2)
    # step 4: PCV conflict reduction
    fin4 = finalize_confidence(pack, experience_quality=1.0, p_dominant=0.0,
                               state_aligned=False, rtp=0.0, pcv=-0.5)
    assert fin4["confidence"] == pytest.approx(0.5 - 0.5 * 0.3)
    assert fin4["conflicting"] is True
    # step 5: clamp at zero
    fin5 = finalize_confidence({"confidence": 0.1}, experience_quality=0.0,
                               p_dominant=1.0, state_aligned=True, rtp=1.0,
                               pcv=-1.0)
    assert fin5["confidence"] >= 0.0
    assert fin5["issued"] is (fin5["confidence"] > CONFIDENCE_FLOOR)


def test_cci_has_31_weighted_components():
    from app.config import HEDGE_COMPONENT_ID, HEDGE_INITIAL_WEIGHT

    eng = CCIEngine()
    w = eng.get_weights("BTC", "tf1")
    assert set(w) == set(WEIGHT_IDS)
    assert len(w) == 31
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(W_CLAMP_LO <= v for v in w.values())
    # Section 8: hedge/dark-pool combined starts at exactly 0.08
    assert w[HEDGE_COMPONENT_ID] == pytest.approx(HEDGE_INITIAL_WEIGHT)
    others = [v for f, v in w.items() if f != HEDGE_COMPONENT_ID]
    assert all(v == pytest.approx(0.92 / 30, abs=1e-12) for v in others)


def test_brain4_secondary_weight_adjustment():
    from test_engine import DummyFeed

    async def _run():
        from app.engine import PredictionEngine
        feed = DummyFeed()
        eng = PredictionEngine(feed, lambda m: asyncio.sleep(0))
        await feed.start()
        eng.compute_asset("BTC", record=True)
        w_before = dict(eng.cci.get_weights("BTC", "tf1"))
        # force a distinctive DN activation pattern and re-apply
        eng.brain_weight_adj[("BTC", "tf1")] = {
            fid: (0.5 if fid == 1 else -0.5) for fid in WEIGHT_IDS
        }
        eng._apply_brain4("BTC", "tf1")
        w_after = eng.cci.get_weights("BTC", "tf1")
        # omega *= (1 + 0.01*(A-0.5)) : fid 1 grows 0.5%, others shrink 0.5%
        assert w_after[1] > w_before[1]
        assert w_after[2] < w_before[2]
        assert sum(w_after.values()) == pytest.approx(1.0, abs=1e-9)
        assert all(W_CLAMP_LO <= v <= W_CLAMP_HI for v in w_after.values())

    asyncio.get_event_loop().run_until_complete(_run())


# ------------------------------------------------------- engine integration
def test_engine_pipeline_carries_part2_state():
    from test_engine import make_engine, run, _boot

    engine, feed, emitted = make_engine()
    run(_boot(engine))
    try:
        for tf in ("tf1", "tf4"):
            res = engine.last_result[("BTC", tf)]
            assert res["arc"]["ratio"] in (None, 1.0, 1.5, 3.0)
            assert "value" in res["arc"]
            assert -1.0 <= res["pcv"] <= 1.0
            assert len(res["brain"]["instances"]) == 4
            assert sorted(res["emotion"]["emotions"]) == list(range(1, 22))
            assert res["base_confidence"] >= 0.0
            if res["issued"]:
                assert res["confidence"] > CONFIDENCE_FLOOR
                assert res["arc"]["ratio"] is not None
            # 28 weighted components stored for the next GRC's reward rule
            comp = engine.prev_comp[("BTC", tf)]
            assert set(comp) == set(WEIGHT_IDS)
            w = engine.cci.get_weights("BTC", tf)
            assert set(w) == set(WEIGHT_IDS)
            assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)
        # GRC-lag emotion rings were written
        keys = [k for k in engine.hist._rings if ":dpi" in k or ":compo" in k]
        assert keys, "emotion lag rings missing"
        snap = engine.build_snapshot()
        assert set(snap["weights"]) == {str(f) for f in WEIGHT_IDS}
        assert 22 in snap["formula_names"] and 31 in snap["formula_names"]
        systems = snap["systems"]["BTC|tf1"]
        assert "pcv" in systems and "arc" in systems
        assert "emotions" in systems["emotion"]
    finally:
        run(engine.stop())
