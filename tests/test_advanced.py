"""Advanced mathematics layer (app/advanced.py) — each approved
module pinned on synthetic tape where the right answer is known."""
import math
import random
from types import SimpleNamespace

from app.advanced import (
    advanced_block,
    bocpd_vote, conformal_update, entropy_block, evt_tail,
    fracdiff_vote, har_rv, jumps_vote, kalman_vote, ou_hurst_block,
    pairs_vote, rqa_multiplier, signature_vote, wavelet_vote,
)


def _trend(n=160, drift=0.0015, noise=0.0004, seed=7):
    import random
    rnd = random.Random(seed)
    p, out = 100.0, []
    for _ in range(n):
        p *= 1 + drift + rnd.gauss(0, noise)
        out.append(p)
    return out


def _chop(n=200, seed=3):
    import random
    rnd = random.Random(seed)
    return [100 + 0.6 * math.sin(i / 9.0) + rnd.gauss(0, 0.05)
            for i in range(n)]


def test_kalman_follows_trend_not_chop():
    assert kalman_vote(_trend())["vote"] > 0.25
    assert abs(kalman_vote(_chop())["vote"]) < 0.85


def test_fracdiff_keeps_memory_of_trend():
    assert fracdiff_vote(_trend())["vote"] > 0.2


def test_bocpd_detects_break_and_tightens_gate():
    calm = _trend(120, drift=0.0004, noise=0.0003, seed=5)
    broken = [x * (1 - 0.004 * (i + 1)) for i, x in
              enumerate(calm[-30:])]
    out = bocpd_vote(calm + broken)
    assert out["vote"] < 0
    assert out["gate"] <= 1.0 and out["run_len"] >= 1


def test_entropy_numbers_in_range():
    out = entropy_block(_chop())
    assert 0.0 <= out["perm_entropy"] <= 1.0
    assert 0.45 <= out["gate"] <= 1.0


def test_ou_hurst_reverting_tape_fades_stretch():
    out = ou_hurst_block(_chop(), W=44)
    assert out["regime"] in ("reverting", "dead-zone", "trending")
    assert -1.0 <= out["vote"] <= 1.0
    assert out["half_life_bars"] > 0


def test_jumps_post_jump_retrace_vote():
    p = _trend(120, drift=0.0, noise=0.0003, seed=11)
    p[-3] = p[-4] * 1.03            # +3% jump three bars ago
    p[-2] = p[-3] * 1.0002
    p[-1] = p[-2] * 0.9998
    out = jumps_vote(p)
    assert out["jump_share"] >= 0.0
    # a detected +jump must produce a non-positive (retrace) vote
    assert out["vote"] <= 0.0


def test_pairs_vote_needs_reference():
    assert pairs_vote("BTC", _trend(), {"BTC": _trend()})["vote"] == 0.0
    out = pairs_vote("ETH", _trend(200, drift=0.002),
                     {"BTC": _trend(200, drift=0.001)})
    assert -1.0 <= out["vote"] <= 1.0


def test_wavelet_trend_positive():
    assert wavelet_vote(_trend())["vote"] > 0


def test_har_and_evt_sizing_layers_bounded():
    p = _trend(1500, drift=0.0002, noise=0.001, seed=13)
    har = har_rv(p)
    assert har["sigma_bar"] and 0.45 <= har["size_mult"] <= 1.0
    evt = evt_tail(p)
    assert evt["q99_bps"] and evt["q99_bps"] > evt["thr_bps"]
    assert 0.4 <= evt["size_mult"] <= 1.0
    assert evt["cat_stop_bps"] > evt["q99_bps"]


def test_rqa_is_only_a_multiplier():
    out = rqa_multiplier(_chop())
    assert 0.85 <= out["mult"] <= 1.15
    assert "vote" not in out


def test_rqa_does_not_punish_ordinary_noise():
    """The live defect: `0.75 + 0.4·DET` returned 0.75 (a silent −25 %
    confidence cut) on a plain random walk, on every cell, forever. An
    unremarkable tape must be NEUTRAL (mult 1.0), and only MEASURED
    determinism may lift it."""
    import random as _rnd
    rng = _rnd.Random(9)
    px = [100.0]
    for _ in range(600):
        px.append(px[-1] * (1 + rng.gauss(0, 0.0012)))
    out = rqa_multiplier(px)
    assert out["determinism"] < 0.5
    assert out["mult"] >= 1.0, out          # never a penalty for being noise
    assert out["mult"] <= 1.15


def test_signature_floor_cap_declared():
    out = signature_vote(_trend(), [10.0] * 160)
    assert out["weight_cap"] == 0.15
    assert -1.0 <= out["vote"] <= 1.0


def test_conformal_band_adapts_to_misses():
    st = {"q": 0.001, "n": 0, "hits": 0}
    for _ in range(60):
        st = conformal_update(st, 0.004)     # always exceeding
    assert st["q"] > 0.001, "band must widen when it keeps missing"
    assert st["n"] == 60


def test_weight_scales_clamped_and_applied():
    P = _trend(300, 0.0008, 0.0006, seed=23)
    ctx = SimpleNamespace(x=1, p=P[-1], P=P, Pfull=P, O=P, H=P, L=P, V=P)
    base = advanced_block("BTCUSDT", "tf3", ctx)
    scaled = advanced_block("BTCUSDT", "tf3", ctx,
                            weight_scales={"kalman": 99.0,
                                           "wavelet": 0.01})
    # scales clamp to [0.5, 1.5]: kalman share grows but stays finite
    wk = scaled["weights"]["kalman"] / max(1e-9, base["weights"]["kalman"])
    assert 1.0 < wk <= 1.5001
    ww = scaled["weights"]["wavelet"] / max(1e-9, base["weights"]["wavelet"])
    assert 0.4999 <= ww < 1.0
    # signature is exempt from adaptive scaling (floor-governed)
    assert scaled["weights"]["signature"] == base["weights"]["signature"]


def test_bocpd_conjugate_detects_regime_break():
    rnd = random.Random(31)
    calm = [100.0 * (1 + rnd.gauss(0, 0.0005)) ** i for i in range(120)]
    last = calm[-1]
    jump = [last * (1.03 ** (i + 1)) * (1 + rnd.gauss(0, 0.0005))
            for i in range(30)]
    blk = bocpd_vote(calm + jump)
    assert 0.0 <= blk["p_break"] <= 1.0
    # after a persistent 3%/bar break the run-length posterior must
    # notice (most likely run length resets well below calm length)
    assert blk["run_len"] < 120


def test_evt_pwm_fat_tail_positive_shape():
    rnd = random.Random(41)
    P, price = [100.0], 100.0
    for i in range(1600):
        step = rnd.gauss(0, 0.001)
        if i % 97 == 0:
            step += rnd.choice([-1, 1]) * 0.012   # fat-tail shocks
        price *= 1 + step
        P.append(price)
    out = evt_tail(P)
    assert out["q99_bps"] is not None and out["q99_bps"] > 0
    assert 0.25 <= out["size_mult"] <= 1.0
    assert out["cat_stop_bps"] >= out["q99_bps"]


def test_horizon_changes_the_answer():
    # same tape, different forecast horizons must NOT collapse to
    # the same vote (tf-invariance was the "un-synced" smell)
    P = _trend(300, 0.0009, 0.0005, seed=5)
    ctx = SimpleNamespace(P=P, V=P, W=22, p=P[-1])
    b22 = advanced_block("BTCUSDT", "tf1", ctx, full_closes=P,
                         minutes=22)
    b247 = advanced_block("BTCUSDT", "tf4", ctx, full_closes=P,
                          minutes=247)
    assert b22["modules"]["kalman"]["horizon"] == 22
    assert b247["modules"]["kalman"]["horizon"] == 247
    assert b22["vote"] != b247["vote"]
    assert (b22["modules"]["entropy"].get("vote")
            != b247["modules"]["entropy"].get("vote")
            or b22["modules"]["wavelet"]["vote"]
            != b247["modules"]["wavelet"]["vote"])


def test_rings_persistence_roundtrip():
    from app.history import HistoryStore
    from app.learning import _rings_restore, _rings_snapshot

    class _E:
        pass
    e1, e2 = _E(), _E()
    e1.hist, e2.hist = HistoryStore(), HistoryStore()
    e1.hist.ring("BTCUSDT:tf3:adv_kalman", 600).append(1.0)
    e1.hist.ring("BTCUSDT:tf3:conf_q", 400).append(0.0042)
    e1.hist.ring("BTCUSDT:tf3:unrelated", 50).append(9.0)
    snap = _rings_snapshot(e1)
    assert "BTCUSDT:tf3:adv_kalman" in snap
    assert "BTCUSDT:tf3:unrelated" not in snap
    _rings_restore(e2, snap)
    assert list(e2.hist.ring("BTCUSDT:tf3:adv_kalman", 600)) == [1.0]
    assert list(e2.hist.ring("BTCUSDT:tf3:conf_q", 400)) == [0.0042]


def test_pairs_intercept_recovers_spread():
    # cointegrated pair with a LEVEL intercept: y = 1.5 + 0.8 x +
    # OU spread. Without the intercept the spread z is garbage and
    # the fade vote points the wrong way.
    rnd = random.Random(3)
    x, y = [11.0], [1.5 + 0.8 * 11.0]
    spread = 0.0
    for _ in range(239):
        x.append(x[-1] + rnd.gauss(0, 0.002))
        spread = 0.92 * spread + rnd.gauss(0, 0.004)
        y.append(1.5 + 0.8 * x[-1] + spread)
    # push the spread far up at the end: vote must fade it (short y)
    y[-1] += 0.05
    out = pairs_vote("ETH", [math.exp(v) for v in y],
                     {"BTC": [math.exp(v) for v in x]})
    assert out["vote"] < 0 and out["z"] > 1.0


def test_signature_translation_invariant():
    rnd = random.Random(9)
    P = [100.0]
    for _ in range(80):
        P.append(P[-1] * (1 + rnd.gauss(0.0004, 0.001)))
    V = [1000 + 300 * math.sin(i / 5.0) for i in range(len(P))]
    a = signature_vote(P, V)["vote"]
    b = signature_vote([v * math.exp(5.0) for v in P], V)["vote"]
    assert abs(a - b) < 0.05          # level shift must not move it


def test_rqa_periodic_beats_noise():
    per = [100.0]
    pat = [0.002, -0.001, 0.0015, -0.002] * 40
    for r in pat:
        per.append(per[-1] * (1 + r))
    rnd = random.Random(4)
    noi = [100.0]
    for _ in range(160):
        noi.append(noi[-1] * (1 + rnd.gauss(0, 0.0016)))
    assert (rqa_multiplier(per)["determinism"]
            > rqa_multiplier(noi)["determinism"])


def test_evt_q99_tracks_empirical():
    rnd = random.Random(17)
    P, px = [100.0], 100.0
    rets = []
    for i in range(3000):
        step = rnd.gauss(0, 0.0008)
        if i % 89 == 0:
            step += rnd.choice([-1, 1]) * 0.01
        rets.append(abs(step))
        px *= 1 + step
        P.append(px)
    out = evt_tail(P)
    emp = sorted(rets)[int(0.99 * len(rets))] * 1e4
    assert out["q99_bps"] is not None
    assert 0.5 * emp < out["q99_bps"] < 2.2 * emp


def test_bocpd_quiet_on_pure_noise():
    rnd = random.Random(21)
    P = [100.0]
    for _ in range(170):
        P.append(P[-1] * (1 + rnd.gauss(0, 0.001)))
    out = bocpd_vote(P)
    # with the prior-predictive factor in the CP mass, featureless
    # noise must NOT read as a permanent regime break
    assert out["gate"] >= 0.9 and out["p_break"] < 0.1


def test_ou_deviation_half_life_finite():
    rnd = random.Random(27)
    lp = [11.0]
    dev = 0.0
    for _ in range(259):
        dev = 0.90 * dev + rnd.gauss(0, 0.002)
        lp.append(11.0 + dev + 0.00002 * len(lp))
    out = ou_hurst_block([math.exp(v) for v in lp], 22)
    assert out["half_life_bars"] < 60      # ~ln2/ln(1/0.9) = 6.6
    assert abs(out["z"]) > 0.1             # z actually moves now


def test_pairs_basket_works_for_btc_too():
    # no asset is the reference: BTC fades vs the peer basket.
    # Common factor f; every asset = level + beta*f + OU spread.
    rnd = random.Random(11)
    f = [0.0]
    for _ in range(239):
        f.append(f[-1] + rnd.gauss(0, 0.002))
    series = {}
    for name, (lvl, beta) in {"ETH": (8.0, 0.9), "BNB": (6.5, 1.1),
                              "PAXG": (7.8, 0.7)}.items():
        sp, s = 0.0, []
        for i in range(240):
            sp = 0.92 * sp + rnd.gauss(0, 0.002)
            s.append(math.exp(lvl + beta * f[i] + sp))
        series[name] = s
    sp, btc = 0.0, []
    for i in range(240):
        sp = 0.92 * sp + rnd.gauss(0, 0.002)
        btc.append(math.exp(11.0 + f[i] + sp))
    btc[-1] *= math.exp(0.05)        # BTC stretched above basket
    out = pairs_vote("BTC", btc, series)
    assert out.get("ref", "").startswith("basket")
    assert out["vote"] < 0           # fades its own stretch


def test_signed_scale_flips_module_contribution():
    P = _trend(300, 0.0009, 0.0005, seed=5)
    ctx = SimpleNamespace(P=P, V=P, W=22, p=P[-1])
    raw = advanced_block("BTCUSDT", "tf1", ctx, full_closes=P,
                         minutes=22)
    flipped = advanced_block(
        "BTCUSDT", "tf1", ctx, full_closes=P, minutes=22,
        weight_scales={k: -1.5 for k in raw["modules"]})
    assert raw["vote"] != 0
    assert flipped["vote"] == -raw["vote"] or \
        (flipped["vote"] * raw["vote"] < 0
         and abs(abs(flipped["vote"]) - abs(raw["vote"])) < 0.05)
    # magnitude floor: a tiny supplied scale still votes at 0.5x
    tiny = advanced_block("BTCUSDT", "tf1", ctx, full_closes=P,
                          minutes=22,
                          weight_scales={"kalman": -0.05})
    assert tiny["weights"]["kalman"] < 0
    from app.advanced import MODULE_WEIGHTS
    assert abs(tiny["weights"]["kalman"]) \
        >= 0.5 * MODULE_WEIGHTS["kalman"] - 1e-9


def test_rqa_multiplier_is_calibrated_against_its_own_null():
    """Second half of the same fix (2026-11-26). A raw `det` on an ORDINARY
    tape is ~0.44, not 0 — that is the estimator's small-sample diagonal
    bias. The first fix (1.0 + 0.4*det) therefore landed on the 1.15 CAP on
    all 16 live cells, every window: a constant multiplier carries no
    information and silently inflates every published confidence by 15 %,
    which is the SAME defect as the old -25 % with the sign flipped.
    The gate now measures its own null and only counts determinism above
    it."""
    import random as _rnd
    from app.advanced import _DET_NULL, _det_on
    # the published null must be a plausible chance level for this estimator
    assert 0.30 < _DET_NULL < 0.60
    # a fixed-seed random walk reports determinism at/near that null
    rng = _rnd.Random(11)
    px = [100.0]
    for _ in range(700):
        px.append(px[-1] * (1 + rng.gauss(0, 0.0013)))
    out = rqa_multiplier(px)
    assert abs(out["determinism"] - _DET_NULL) < 0.10
    # and it is NEUTRAL, not capped high
    assert out["mult"] < 1.02, out
    assert out["det_excess"] < 0.1
    # the null is published so the panel can show both numbers
    assert out["determinism_null"] == round(_DET_NULL, 3)


def test_a_truly_recurrent_tape_earns_more_than_an_ordinary_one():
    """The multiplier must DISCRIMINATE — otherwise it is a constant and
    should not exist. A periodic tape has to beat an ordinary walk on the
    published mult, not just on raw det."""
    import random as _rnd
    per = [100.0]
    for r in [0.002, -0.001, 0.0015, -0.002] * 60:
        per.append(per[-1] * (1 + r))
    rng = _rnd.Random(4)
    noi = [100.0]
    for _ in range(240):
        noi.append(noi[-1] * (1 + rng.gauss(0, 0.0016)))
    a, b = rqa_multiplier(per), rqa_multiplier(noi)
    assert a["det_excess"] > b["det_excess"]
    assert a["mult"] > b["mult"]
    assert a["mult"] <= 1.15


def test_soft_gate_is_a_bounded_scale_not_a_triple_product_veto():
    """The advanced block multiplied three module gates and clamped at 0.45.
    A product of three numbers in [0,1] is almost always below 0.45 on real
    tape: measured live, 4 of 16 cells sat exactly on the floor and the rest
    at 0.57-0.62 — a −38 % to −55 % confidence haircut on EVERY window,
    justified by no measurement, applied after the published confidence had
    already cleared its floor. (Same defect class as the RQA 0.75 fixed the
    same day.) The gate is now one mean-based uncertainty penalty in
    [0.85, 1.0], and the raw product is still published for audit."""
    from app.advanced import advanced_block
    import random as _rnd
    out = None
    for seed in (3, 17, 91):
        rng = _rnd.Random(seed)
        px, vol = [], []
        p = 100.0
        for _ in range(400):
            p *= 1 + rng.gauss(0, 0.0015)
            px.append(p)
            vol.append(1000 + rng.random() * 200)
        ctx = SimpleNamespace(P=px, V=vol, W=22, p=px[-1])
        blk = advanced_block("BTCUSDT", "tf1", ctx, full_closes=px,
                             minutes=22)
        sg = blk["soft_gate"]
        assert 0.85 <= sg <= 1.0, sg          # a scale, never a veto
        assert blk["soft_gate_raw"] is not None  # the product is published
        # with sub-gates near 1 the gate must NOT be pinned at a bound
        mods = blk["modules"]
        gs = [mods[n].get("gate", 1.0) for n in ("entropy", "ou_hurst", "bocpd")]
        if all(g > 0.5 for g in gs):
            assert sg > 0.85 + 1e-9, (sg, gs)
        out = blk
    assert out is not None
    assert "one uncertainty penalty" in out["soft_gate_rule"]


def test_advanced_confidence_multiplier_varies_across_tape():
    """A multiplier that is the same constant on every cell carries no
    information and is applied to every confidence in the app. Both factors
    must be able to move."""
    from app.advanced import advanced_block
    import random as _rnd
    vals = []
    for seed in (1, 2, 3, 4, 5):
        rng = _rnd.Random(seed)
        px, vol = [], []
        p = 100.0
        drift = 0.0005 * (seed % 3)
        for _ in range(400):
            p *= 1 + rng.gauss(drift, 0.001 * (1 + seed % 4))
            px.append(p)
            vol.append(800 + rng.random() * 400)
        ctx = SimpleNamespace(P=px, V=vol, W=22, p=px[-1])
        blk = advanced_block("BTCUSDT", "tf1", ctx, full_closes=px,
                             minutes=22)
        vals.append((round(blk["soft_gate"], 4), round(blk["rqa_mult"], 4)))
    assert len({v[0] for v in vals}) > 1, vals   # soft gate is not a constant


def test_confidence_chain_telemetry_reaches_the_api_payload():
    """The CONFIDENCE CHAIN panel is the only screen that can answer "why did
    this window publish 0.009". Every field it renders was dropped inside
    compute_asset when `results[tf]` was built, so the panel drew nulls while
    the number it audits shipped fine. The producer's chain must reach
    /api/systems."""
    import inspect
    from app import engine as _eng
    src = inspect.getsource(_eng)
    for field in ("confidence_base_snr", "confidence_modulator",
                  "confidence_chain", "snr", "adv_conf_mult"):
        assert f'"{field}": fin.get("{field}")' in src, field
    # and the display_systems dict must start from the producer's own fields
    # instead of a hand-maintained whitelist that silently drops new ones
    assert "_ds_base = {k: v for k, v in results[tf].items()" in src
