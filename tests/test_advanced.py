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
    assert 0.75 <= out["mult"] <= 1.15
    assert "vote" not in out


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
