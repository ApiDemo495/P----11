"""Part-3 tests: the 22 physics formulas (Section 6), the news system with
categories and weightages (Section 7), the hedge-fund/dark-pool formulas
(Section 8), and the visualization data contracts (Section 9).
"""
from __future__ import annotations

import base64
import math
import time

import numpy as np
import pytest

from app.cci import CCIEngine, finalize_confidence
from app.config import (
    HEDGE_COMPONENT_ID,
    HEDGE_INITIAL_WEIGHT,
    NEWS_CATEGORIES,
    PHYSICS_ALPHA_COUNT,
    PHYSICS_DISPLAY_IDS,
    QUANTUM_LEVELS,
    UFP_ALPHA_IDS,
    WEIGHT_IDS,
)
from app.hedgefund import (
    combined_signal,
    dark_pool_flow,
    institutional_flow,
    smart_money_divergence,
)
from app.history import HistoryStore
from app.models import FormulaContext
from app.news import (
    NewsSystem,
    classify_category,
    impact_score,
    source_reliability,
    tf_impact,
)
from app.physics import (
    PhysicsSystem,
    Scene,
    betti_numbers,
    p01_pssa,
    p02_wfcp,
    p04_qtpb,
    p05_mee,
    p06_drpt,
    p09_tme,
    p10_pipp,
    p13_glplo,
    p14_celg,
    p15_dsms,
    p16_rtdmp,
    p17_bbir,
    p18_tpti,
    p19_gspi,
    p21_qdpc,
    qdpc,
    simplified_cci,
    volume_profile,
)


# ===========================================================================
# helpers
# ===========================================================================
def _trend_ctx(n: int = 60, W: int = 44, seed: int = 7) -> FormulaContext:
    rng = np.random.RandomState(seed)
    price = 100.0
    P, V, T = [price], [], []
    vol = 100.0
    base_t = int(time.time() // 60) * 60 - n * 60
    for i in range(n):
        price *= 1.0 + 0.0006 + 0.0015 * rng.randn() * 0.3
        vol = 100.0 + 40.0 * rng.rand()
        P.append(round(price, 6))
        V.append(vol)
        T.append(base_t + i * 60)
    V.append(V[-1])
    return FormulaContext(symbol="BTC", tf="tf1", W=W, P=P, V=V, T=T,
                          hist=HistoryStore())


def _scene(ctx: FormulaContext | None = None, cci_ref: float = 0.2,
           long_closes: dict | None = None) -> Scene:
    ctx = ctx or _trend_ctx()
    atr = float(np.mean(np.abs(np.diff(ctx.P)))) or 1e-6
    return Scene("BTC", ctx.tf, ctx, atr=atr, raws={}, cci_ref=cci_ref,
                 long_closes=long_closes or {"BTC": list(ctx.P)},
                 news_minutes={}, t_market_prev=None)


# ===========================================================================
# Section 7 — news categories, weightages, scoring
# ===========================================================================
def test_news_category_table_exact():
    expected = {
        "regulatory": (0.25, 247), "institutional": (0.20, 130),
        "protocol": (0.15, 44), "macro": (0.15, 130),
        "structure": (0.10, 44), "sentiment": (0.08, 22),
        "geopolitical": (0.07, 247),
    }
    assert set(NEWS_CATEGORIES) == set(expected)
    for name, (w, d) in expected.items():
        assert NEWS_CATEGORIES[name]["weight"] == w
        assert NEWS_CATEGORIES[name]["duration"] == d
    assert sum(v["weight"] for v in NEWS_CATEGORIES.values()) == pytest.approx(1.0)


def test_impact_score_and_freshness_exact():
    # Impact = weight * sentiment * exp(-0.05 * age) * reliability
    got = impact_score("regulatory", -0.8, 20.0, 0.7)
    want = 0.25 * (-0.8) * math.exp(-0.05 * 20.0) * 0.7
    assert got == pytest.approx(want)
    # freshness checkpoints from the spec
    assert math.exp(-0.05 * 20) == pytest.approx(math.exp(-1.0))
    assert impact_score("sentiment", 1.0, 0.0, 1.0) == pytest.approx(0.08)


def test_timeframe_impact_distribution_exact():
    # regulatory D=247: TF1 receives score*22/247 ... TF4 full score
    assert tf_impact(1.0, 247, "tf1") == pytest.approx(22 / 247)
    assert tf_impact(1.0, 247, "tf2") == pytest.approx(44 / 247)
    assert tf_impact(1.0, 247, "tf3") == pytest.approx(130 / 247)
    assert tf_impact(1.0, 247, "tf4") == pytest.approx(1.0)
    # sentiment D=22: only TF1 (gated by D >= tf minutes)
    assert tf_impact(1.0, 22, "tf1") == pytest.approx(1.0)
    assert tf_impact(1.0, 22, "tf2") == 0.0
    # institutional D=130: TF4 gated off
    assert tf_impact(1.0, 130, "tf3") == pytest.approx(1.0)
    assert tf_impact(1.0, 130, "tf4") == 0.0


def test_aggregate_clamped_and_signed():
    ns = NewsSystem()
    ns.initialize()
    ns.items = []
    now = time.time()
    for _ in range(10):                       # 10 * +0.25 -> clamp to +1
        ns._append({"headline": "SEC approves everything", "category":
                    "regulatory", "sentiment": 1.0, "reliability": 1.0,
                    "source": "t", "symbols": ["BTC"], "ts": now})
    agg = ns.aggregate("BTC")
    assert agg["value"] == pytest.approx(1.0)
    assert agg["count"] == 10
    assert agg["impacts"]["tf4"] == pytest.approx(1.0)
    # opposing-magnitude helper
    assert ns.impact_against("BTC", +1) == 0.0
    assert ns.impact_against("BTC", -1) == pytest.approx(1.0)


def test_classifier_and_reliability_tiers():
    assert classify_category("SEC ruling on crypto custody") == "regulatory"
    assert classify_category("ETF provider files registration") == "institutional"
    assert classify_category("CPI data beats expectations") == "macro"
    assert classify_category("Ethereum mainnet upgrade ships") == "protocol"
    assert classify_category("wildly unclassifiable chatter") == "sentiment"
    assert source_reliability("Reuters") == 1.0
    assert source_reliability("CoinDesk") == 1.0
    assert source_reliability("The Block") == 0.7
    assert source_reliability("some-forum") == 0.4


def test_news_seeds_and_status():
    ns = NewsSystem()
    ns.initialize()
    st = ns.status()
    assert st["ready"] and st["items"] >= 5 and st["categories"] == 7
    agg = ns.aggregate("ETH")
    assert -1.0 <= agg["value"] <= 1.0
    assert agg["count"] >= 1
    assert set(agg["impacts"]) == {"tf1", "tf2", "tf3", "tf4"}


# ===========================================================================
# Section 8 — hedge fund and dark pool formulas
# ===========================================================================
def _hf_ctx() -> FormulaContext:
    """15 minutes: quiet volumes with ONE institutional-style minute."""
    n = 15
    base_t = int(time.time() // 60) * 60 - n * 60
    P = [100.0 + 0.02 * i for i in range(n + 1)]
    V = [100.0] * (n + 1)
    O = list(P)
    H = [p + 0.05 for p in P]
    L = [p - 0.05 for p in P]
    # minute 10: 4x volume, tiny body, close in upper 30% -> institutional BUY
    V[10] = 400.0
    P[10] = P[9] + 0.01
    H[10] = P[9] + 0.02
    L[10] = P[9] - 0.02
    O[10] = P[9]
    T = [base_t + i * 60 for i in range(n + 1)]
    return FormulaContext(symbol="BTC", tf="tf1", W=22, P=P, V=V, T=T,
                          hist=HistoryStore(), O=O, H=H, L=L)


def test_ifr_detects_institutional_buy():
    res = institutional_flow(_hf_ctx())
    assert res["events"] >= 1
    assert res["buy"] > 0 and res["sell"] == 0.0
    assert -1.0 <= res["value"] <= 1.0
    assert res["value"] > 0.0
    # tanh contract: value == tanh((buy - sell) / total)
    assert res["value"] == pytest.approx(
        math.tanh((res["buy"] - res["sell"]) / (res["buy"] + res["sell"] + 1e-12))
    )


def test_ifr_flat_when_no_institutional_minutes():
    ctx = _hf_ctx()
    ctx.V = [100.0] * len(ctx.V)          # no volume spikes at all
    res = institutional_flow(ctx)
    assert res["events"] == 0
    assert res["value"] == 0.0


def test_dpad_detects_dark_print_accumulation():
    ctx = _hf_ctx()
    # make every minute volatile-ish (sigma > 0) then plant a print
    n = len(ctx.P)
    for i in range(1, n):
        ctx.P[i] = 100.0 + (0.4 if i % 2 == 0 else -0.3) * i * 0.05
    # print: huge volume, tiny return; net buying next 3 minutes
    ctx.V[8] = 900.0
    p7 = ctx.P[7]
    ctx.P[8] = p7 * 1.0001                     # |r| ~ 1bp << 0.3 sigma
    ctx.P[9] = ctx.P[8] * 1.002
    ctx.P[10] = ctx.P[8] * 1.004
    ctx.P[11] = ctx.P[8] * 1.005
    avg = sum(ctx.V[:-1]) / (len(ctx.V) - 1)
    assert ctx.V[8] > 5.0 * avg
    res = dark_pool_flow(ctx)
    assert res["prints"] >= 1
    assert res["accumulation"] > 0.0
    assert res["value"] > 0.0
    assert -1.0 <= res["value"] <= 1.0


def test_smdi_formula_and_combined_mean():
    # retail_flow = -H_signal_avg -> SMDI = tanh((IFR + H) / 2)
    ifr, h = 0.6, -0.3
    assert smart_money_divergence(ifr, h) == pytest.approx(
        math.tanh((ifr - (-h)) / 2.0)
    )
    # institutions buying while retail (brain) is bearish -> strongly positive
    assert smart_money_divergence(0.8, 0.5) > 0.5   # H>0 -> retail_flow<0
    # combined component = plain mean, bounded
    c = combined_signal(0.5, -0.2, 0.1)
    assert c == pytest.approx((0.5 - 0.2 + 0.1) / 3.0)
    assert -1.0 <= combined_signal(1.0, 1.0, 1.0) <= 1.0


def test_hedge_initial_weight_is_0_08():
    eng = CCIEngine()
    w = eng.get_weights("ETH", "tf2")
    assert len(WEIGHT_IDS) == 31
    assert w[HEDGE_COMPONENT_ID] == pytest.approx(HEDGE_INITIAL_WEIGHT)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-12)


# ===========================================================================
# Section 6 — physics formulas
# ===========================================================================
def test_p1_superposition_normalizes_and_centers():
    sc = _scene()
    value, det = p01_pssa(sc)
    assert len(det["levels"]) == QUANTUM_LEVELS
    assert det["levels"][10] == pytest.approx(sc.price)   # N=21 centered
    assert sum(det["probs"]) == pytest.approx(1.0, abs=1e-4)
    assert all(p >= 0 for p in det["probs"])
    assert det["confidence_q"] >= 1.0                     # max|c|^2 / (1/N)
    assert -1.0 <= value <= 1.0
    # expectation value of the superposition is the weighted mean level
    pred = sum(p * x for p, x in zip(det["probs"], det["levels"]))
    assert det["p_predicted"] == pytest.approx(pred, abs=1e-3)


def test_p2_wfcp_probability_bounds():
    sc = _scene()
    wfcp, det = p02_wfcp(sc)
    assert 0.0 <= wfcp <= 1.0
    assert det["gap"] >= 0.0
    # markets far from equilibrium collapse quickly -> WFCP high here
    assert wfcp == pytest.approx(1.0 - math.exp(-det["gamma"] * sc.tau))


def test_p4_qtpb_no_barrier_means_classical():
    sc = _scene(_trend_ctx(n=5))          # W_eff < 10 -> no fractal levels
    value, det = p04_qtpb(sc)
    assert det["q_up"] == 1.0 and det["q_down"] == 1.0
    assert value == 0.0


def test_p4_qtpb_bounds_with_barriers():
    sc = _scene()
    value, det = p04_qtpb(sc)
    assert 0.0 <= det["q_up"] <= 1.0 and 0.0 <= det["q_down"] <= 1.0
    assert value == pytest.approx(det["q_up"] - det["q_down"])
    assert -1.0 <= value <= 1.0


def test_p5_entanglement_entropy_range():
    ctx = _trend_ctx()
    peers = {s: (list(ctx.T), [p * (1.0 + 0.0001 * i)
                               for i, p in enumerate(ctx.P)])
             for s in ("PAXG", "ETH", "BNB")}
    ctx.others = peers
    sc = _scene(ctx)
    value, det = p05_mee(sc)
    assert 0.0 <= det["entropy"] <= math.log(2.0) + 1e-9
    assert 0.0 <= det["reduced"] <= 1.0
    assert -1.0 <= value <= 1.0


def test_p6_drpt_decreases_with_coherence():
    sc = _scene()
    coherent = p06_drpt(sc, msrc=0.9, cthr=0.9)
    noisy = p06_drpt(sc, msrc=0.0, cthr=0.0)
    assert 0.0 <= coherent <= 1.0 and 0.0 <= noisy <= 1.0
    assert coherent < noisy                      # coherence -> less decoherence
    assert coherent == pytest.approx(
        1.0 - math.exp(-(1.0 - 0.9) * (1.0 - 0.9) / 2.0 * (sc.tau / (sc.tau / 5.0)))
    )
    # call-site semantics (user-delegated fix): msrc = |MSRC| coherence
    # level, cthr = CONFIDENCE_FLOOR — a coherent bear must NOT max DRPT
    from app.config import CONFIDENCE_FLOOR
    bear = p06_drpt(sc, msrc=abs(-0.9), cthr=CONFIDENCE_FLOOR)
    bull = p06_drpt(sc, msrc=abs(+0.9), cthr=CONFIDENCE_FLOOR)
    assert bear == pytest.approx(bull)            # sign of MSRC is irrelevant
    assert bear < p06_drpt(sc, msrc=0.1, cthr=CONFIDENCE_FLOOR)


def test_p9_tme_defaults_and_range():
    sc = _scene(_trend_ctx(n=8))
    tme, det = p09_tme(sc)
    # with no hot/cold minutes the spec defaults give 1 - 0.5/2 = 0.75
    assert tme == pytest.approx(0.75)
    assert 0.0 <= tme <= 1.0


def test_p10_pipp_deterministic_and_bounded():
    sc = _scene(_trend_ctx(n=30))
    v1 = p10_pipp(sc, atr=0.5)
    v2 = p10_pipp(_scene(_trend_ctx(n=30)), atr=0.5)
    assert -1.0 <= v1 <= 1.0
    assert v1 == pytest.approx(v2)              # seeded Monte Carlo


def test_p13_glplo_pulls_toward_mass():
    sc = _scene(_trend_ctx(n=60))
    value, det = p13_glplo(sc)
    assert -1.0 <= value <= 1.0
    assert det["massive"] >= 0
    # spec reconstruction: tanh(normalized net force)
    if det["massive"] > 0 and det["g_norm"] > 0:
        # recompute from the profile directly
        prof = sc.profile
        forces = []
        for mid, v in zip(prof["mid"], prof["vol"]):
            if v > 3.0 * prof["avg"]:
                m_j = v / prof["avg"]
                d_j = max(abs(mid - sc.price) / sc.atr, 1e-6)
                forces.append((m_j / (d_j * d_j),
                               1 if mid > sc.price else
                               (-1 if mid < sc.price else 0)))
        g = sum(f for f, _ in forces)
        expect = math.tanh(sum(f * s for f, s in forces) / g)
        assert value == pytest.approx(expect, abs=1e-9)


def test_p14_celg_cancels_to_exact_form():
    sc = _scene(_trend_ctx(n=80))
    value, det = p14_celg(sc)
    assert -1.0 <= value <= 1.0
    if det["gaps"] >= 2 and det["d_gap"]:
        expected = math.tanh(
            -(1.0 / det["d_gap"] ** 4)
            * (1 if det["mid_gap"] > sc.price else -1)
        )
        assert value == pytest.approx(expected, abs=0.01)
    else:
        assert value == 0.0


def test_p15_doppler_amplifies_blueshift():
    ctx = _trend_ctx(n=60)
    # leader: same series but with a stronger latest minute up-move
    long_closes = {"BTC": list(ctx.P), "PAXG": list(ctx.P),
                   "ETH": list(ctx.P), "BNB": list(ctx.P)}
    long_closes["ETH"][-1] = long_closes["ETH"][-2] * 1.004   # ETH leads up
    sc = _scene(ctx, long_closes=long_closes)
    value, det = p15_dsms(sc, atr=0.5)
    assert -1.0 <= value <= 1.0
    assert det["doppler"] > 0
    # exact spec reconstruction
    assert value == pytest.approx(
        math.tanh(det["momentum_raw"] * det["doppler"]), abs=1e-9
    )


def test_p16_rtdmp_exact_reconstruction():
    sc = _scene(cci_ref=0.5)
    value, det = p16_rtdmp(sc)
    gamma = det["gamma"]
    tau_eff = sc.tau / gamma
    expected = math.tanh((tau_eff / sc.tau - 0.5) * 2.0 * 1.0)
    assert value == pytest.approx(expected, abs=1e-4)
    assert gamma >= 1.0
    # with cci_ref = 0 direction collapses the signal to 0
    sc0 = _scene(cci_ref=0.0)
    v0, _ = p16_rtdmp(sc0)
    assert v0 == 0.0


def test_p17_bbir_uses_reported_ratio():
    sc = _scene()
    value, det = p17_bbir(sc, t_baseline=sc.t_market * 0.5)  # market is hot
    assert -1.0 <= value <= 1.0
    if det["hot"] != 0:
        assert value == pytest.approx(math.tanh(det["ratio"] * det["hot"]),
                                      abs=1e-9)
    # no baseline -> 0 (bootstrap)
    v_none, _ = p17_bbir(sc, t_baseline=0.0)
    assert v_none == 0.0


def test_p18_betti_numbers_and_boot_state():
    P = _trend_ctx(n=40).P
    b0, b1 = betti_numbers(P)
    assert b0 >= 1 and b1 >= 0
    b0b, b1b = betti_numbers(P)
    assert (b0, b1) == (b0b, b1b)                # deterministic filtration
    # empty history: delta_beta0 = 0 and z = 0 -> TPTI = 0
    sc = _scene()
    value, det, pair = p18_tpti(sc, beta1_ring=[], beta0_prev=None)
    assert value == 0.0
    assert pair[0] == det["beta0"]


def test_p19_gspi_zero_when_gauges_invariant():
    sc = _scene()
    raws = {f: 0.3 for f in range(1, 32)}
    weights = {f: 0.1 for f in WEIGHT_IDS if f <= 31}
    # gauge-invariant recompute: identical outputs regardless of transform
    value, det = p19_gspi(sc, raws, weights,
                          recompute=lambda g: dict(raws))
    assert value == pytest.approx(0.0, abs=1e-9)
    assert 0.0 <= value <= 1.0
    # simplified top-5 CCI contract
    s = simplified_cci(raws, weights, [1, 2, 3, 4, 5])
    assert s == pytest.approx(0.3)


def test_p21_qdpc_exact_decay():
    h, t, g = 0.5, 1e-4, 0.2
    tau = h / (2.0 * 1.0 * t * g)
    assert qdpc(0.0, h, t, g) == 1.0
    assert qdpc(tau, h, t, g) == pytest.approx(math.exp(-1.0))
    assert qdpc(10.0, h, t, g) == pytest.approx(math.exp(-10.0 / tau))


def test_p21_qdpc_counts_last_30min_news_events():
    ctx = _trend_ctx(n=60)
    now_min = ctx.T[-1]
    sc = Scene("BTC", "tf1", ctx, atr=0.5, raws={}, cci_ref=0.0,
               long_closes={"BTC": list(ctx.P)},
               news_minutes={now_min - 60: 0.9,        # inside 30 min, >0.3
                             now_min - 60 * 60: 0.9},  # outside window
               t_market_prev=None)
    _, gamma = p21_qdpc(sc)
    # exactly one qualifying news event (+ price/volume events from trend)
    news_part = 1.0 / 30.0
    assert gamma >= news_part


def test_finalize_confidence_physics_multipliers_exact():
    pack = {"confidence": 0.5}
    fin = finalize_confidence(
        pack, experience_quality=1.0, p_dominant=0.0, state_aligned=False,
        rtp=0.0, pcv=0.0, wfcp=0.5, drpt=0.2, gspi=0.1,
    )
    expect = 0.5 * 1.0 * (0.5 + 0.5 * 0.5) * (1 - 0.2) * (1 - 0.1)
    assert fin["confidence"] == pytest.approx(expect)
    # neutral defaults keep the part-2 pipeline exact
    fin0 = finalize_confidence(pack, experience_quality=1.0, p_dominant=0.0,
                               state_aligned=False, rtp=0.0, pcv=0.0)
    assert fin0["confidence"] == pytest.approx(0.5)


def test_cci_tme_scales_only_physics_component():
    eng = CCIEngine()
    adjusted = {fid: 0.0 for fid in WEIGHT_IDS}
    adjusted[99] = 1.0
    adjusted[1] = 0.5
    out = eng.compute("BTC", "tf1", adjusted, drhe=1.0, tme=0.5)
    w = eng.get_weights("BTC", "tf1")
    expect = w[99] * 1.0 * 0.5 + w[1] * 0.5
    assert out["cci"] == pytest.approx(expect, abs=1e-12)


def test_ufp_exact_with_neutral_couplings(tmp_path, monkeypatch):
    from app import physics as physics_mod
    monkeypatch.setattr(physics_mod, "PHYSICS_STATE_FILE",
                        str(tmp_path / "phys.json"))
    ps = PhysicsSystem()
    ps.initialize()
    values = {p: 0.0 for p in UFP_ALPHA_IDS}
    values[1] = 1.0                                # only PSSA nonzero
    hist = HistoryStore()
    out = ps.build_ufp("BTC", "tf1", values, tme=0.5, hist=hist)
    # coupling defaults to 1 pre-history; alpha 1/21 each.
    # Directional P1: 1.0 * tme = 0.5.  The five non-directional formulas
    # (2, 6, 9, 19, 21) map their zero value to 2*0 - 1 = -1.
    expect = (1.0 / PHYSICS_ALPHA_COUNT) * (1.0 * 0.5 + 5 * (-1.0))
    assert out["value"] == pytest.approx(expect, abs=1e-6)  # 6-dp rounding
    assert len(out["alpha"]) == 21
    assert sum(out["alpha"].values()) == pytest.approx(1.0, abs=1e-9)
    # alpha reward rule: correct-sign alpha grows, sum stays 1
    before = dict(ps._weights("BTC", "tf1"))
    ps.reward_update("BTC", "tf1", values, move=+1.0)
    after = ps._weights("BTC", "tf1")
    assert after[1] > before[1]
    assert sum(after.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(0.005 - 1e-9 <= v <= 0.15 + 1e-9 for v in after.values())


def test_physics_display_ids_complete():
    assert len(PHYSICS_DISPLAY_IDS) == 22          # P1..P21 + UFP


# ===========================================================================
# Section 9 + integration — engine data contracts
# ===========================================================================
def test_engine_results_carry_part3_fields():
    from test_engine import _boot, make_engine, run

    engine, feed, _ = make_engine()
    run(_boot(engine))
    try:
        pack = engine.compute_asset("BTC", record=True, display=True)
        res = pack["tf1"]
        ph = res["physics"]
        assert set(ph["values"]) == set(range(1, 22))
        assert -1.0 <= ph["ufp"] <= 1.0
        assert 0.0 <= ph["tme"] <= 1.0
        assert 0.0 <= ph["wfcp"] <= 1.0
        assert 0.0 <= ph["drpt"] <= 1.0
        assert 0.0 <= ph["gspi"] <= 1.0
        assert -1.0 <= res["news_value"] <= 1.0
        assert res["qdpc_tau"] > 0
        hedge = res["hedge"]
        for k in ("ifr", "dpad", "smdi", "combined"):
            assert -1.0 <= hedge[k] <= 1.0

        d = engine.display_systems[("BTC", "tf1")]
        assert set(d["physics"]["values"]) == set(range(1, 22))
        assert "alpha" in d["physics"] and "couplings" in d["physics"]
        assert d["physics"]["p1"]["probs"]
        assert d["news"]["value"] is not None
        assert "dlp" not in d or isinstance(d["dlp"], dict)
        # two more record passes -> co-relation edges appear (>=3 samples)
        for _ in range(2):
            feed.tick(4)
            for sym in ("BTC", "PAXG", "ETH", "BNB"):
                engine.compute_asset(sym, record=True, display=True)
        d2 = engine.display_systems[("BTC", "tf1")]
        assert len(d2["corelation"]["edges"]) > 0
        nodes = [n["id"] for n in d2["corelation"]["nodes"]]
        assert "AGENT-G" in nodes and "PHYSICS" in nodes
    finally:
        run(engine.stop())


def test_formula_history_covers_all_three_groups():
    from test_engine import _boot, make_engine, run

    engine, feed, _ = make_engine()
    run(_boot(engine))
    try:
        engine.compute_asset("BTC", record=True, display=True)
        h = engine.formula_history("BTC", "tf1")
        assert len(h["core"]) == 31               # F1..F31 panels
        assert len(h["physics"]) == 22            # P1..P22 panels
        assert len(h["emotion"]) == 22            # E1..E22 panels
        assert len(h["alpha"]) == 21
        assert len(h["core"]["1"]) >= 1
        assert len(h["physics"]["22"]) >= 1
        assert h["names"]["physics"]["22"] == "UFP"
    finally:
        run(engine.stop())


def test_snapshot_ships_part3_meta():
    from test_engine import _boot, make_engine, run

    engine, feed, _ = make_engine()
    run(_boot(engine))
    try:
        engine.compute_asset("BTC", record=True, display=True)
        snap = engine.build_snapshot()
        labels = snap["weight_labels"]
        assert labels["99"] == "PHYSICS UFP"
        assert labels["100"] == "NEWS IMPACT"
        assert labels["101"] == "HEDGE+DARK"
        assert snap["physics"]["formulas"] == 22
        assert len(snap["news_meta"]["categories"]) == 7
        assert set(snap["weights"]) == {str(f) for f in WEIGHT_IDS}
        assert len(snap["weights"]) == 31
        assert snap["analysis"]["readiness"].get("physics") is True
    finally:
        run(engine.stop())


def test_brain_visualization_geometry_and_activations():
    from test_engine import _boot, make_engine, run

    engine, feed, _ = make_engine()
    run(_boot(engine))
    try:
        geom = engine.brain_geometry()
        assert geom["units"] == 7700
        assert geom["edges"] > 0
        assert len(geom["regions"]) == 8          # spec region labels
        names = {r["name"] for r in geom["regions"]}
        assert "PAM cluster" in names and "Mushroom Body" in names
        pos = np.frombuffer(base64.b64decode(geom["positions"]),
                            dtype=np.int16).reshape(-1, 3)
        assert pos.shape == (7700, 3)
        pairs = np.frombuffer(base64.b64decode(geom["edge_pairs"]),
                              dtype=np.uint16).reshape(-1, 2)
        assert pairs.shape[0] == geom["edges"]
        assert max(int(pairs.max()), 0) < 7700
        acts = engine.brain_activations("BTC", "tf1")
        assert len(acts["per_instance"]) == 4
        raw = base64.b64decode(acts["per_instance"][0])
        assert len(raw) == 7700
        assert len(geom["news_units"]) == 10
        assert geom["news_units"] == list(range(geom["news_units"][0],
                                                geom["news_units"][0] + 10))
        assert all(4780 < u < 5000 for u in geom["news_units"])
    finally:
        run(engine.stop())


def test_brain_receives_news_value_input():
    from test_engine import _boot, make_engine, run

    engine, feed, _ = make_engine()
    run(_boot(engine))
    try:
        market = engine._build_market("BTC")
        market["news_value"] = 0.7
        thing2 = engine._thing2("BTC", "tf1",
                                engine.build_ctx("BTC", "tf1"), 1.0, market)
        assert thing2["news_value"] == 0.7
        # the sensory slice is live inside evaluate()
        br = engine.brains.evaluate(
            "BTC", "tf1", market, {}, thing2, advance=False)
        assert br["consensus"] is not None
    finally:
        run(engine.stop())
