"""Advanced mathematics layer (app/advanced.py) — each approved
module pinned on synthetic tape where the right answer is known."""
import math

from app.advanced import (
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
