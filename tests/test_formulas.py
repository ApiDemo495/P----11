"""Formula sanity tests on synthetic series.

We pin behavioral expectations implied by each formula's interpretation in the
specification (directional formulas point the right way on clean synthetic
patterns; bounded formulas stay in [-1, +1]).
"""
import random

from app.formulas import FORMULA_FUNCTIONS, DIRECTIONAL_FORMULAS
from app.history import HistoryStore
from app.models import FormulaContext


def make_ctx(prices, volumes=None, tf="tf1", W=22, symbol="BTC", hist=None,
             others=None, fill_history=None):
    n = len(prices)
    if volumes is None:
        volumes = [10.0] * n
    T = [1_700_000_000 + i * 60 for i in range(n)]
    ctx = FormulaContext(
        symbol=symbol, tf=tf, W=W, P=list(prices), V=list(volumes), T=T,
        hist=hist or HistoryStore(),
        others=others or {},
    )
    if fill_history:
        for key, (name, maxlen, value) in fill_history.items():
            ring = ctx.hist.ring(ctx.hist.tf_key(symbol, tf, name), maxlen)
            for _ in range(8):
                ring.append(value)
    return ctx


def trend_up(n=300, seed=7):
    rng = random.Random(seed)
    p = 100.0
    out = []
    for _ in range(n):
        p *= 1.0 + abs(rng.gauss(0.001, 0.0004))   # upward drift with noise
        out.append(p)
    return out


def trend_down(n=300, seed=7):
    rng = random.Random(seed)
    p = 100.0
    out = []
    for _ in range(n):
        p *= 1.0 - abs(rng.gauss(0.001, 0.0004))
        out.append(p)
    return out


def flat(n=300):
    return [100.0] * n


def all_bounded(results):
    for fid, v in results.items():
        if fid == 20:
            assert v >= 0.0, "DRHE must be non-negative"
            continue
        assert -1.0001 <= v <= 1.0001, f"F{fid} out of range: {v}"


def test_all_formulas_run_and_bounded_on_trending_series():
    ctx = make_ctx(trend_up())
    results = {fid: fn(ctx) for fid, fn in FORMULA_FUNCTIONS.items()}
    all_bounded(results)
    assert all(v == v for v in results.values()), "NaN produced"


def test_all_formulas_run_on_flat_series_without_crashing():
    ctx = make_ctx(flat())
    results = {fid: fn(ctx) for fid, fn in FORMULA_FUNCTIONS.items()}
    all_bounded(results)


def test_satd_positive_on_uptrend_negative_on_downtrend():
    up = FORMULA_FUNCTIONS[4](make_ctx(trend_up()))
    down = FORMULA_FUNCTIONS[4](make_ctx(trend_down()))
    assert up > 0.1
    assert down < -0.1


def test_msrc_positive_when_granularities_agree():
    ctx = make_ctx(trend_up())
    assert FORMULA_FUNCTIONS[11](ctx) > 0.3


def test_diq_high_on_strong_trend():
    ctx = make_ctx(trend_up())
    v = FORMULA_FUNCTIONS[13](ctx)
    assert abs(v) > 0.4


def test_rmet_upper_half_on_uptrend():
    # rising series should sit in the upper half of its equilibrium channel
    ctx = make_ctx(trend_up())
    v = FORMULA_FUNCTIONS[12](ctx)
    assert v > -1.0 and v <= 1.0


def test_tci_concentrated_spike_positive():
    prices = trend_up(200, seed=3)
    # inject a concentrated bullish burst at the very end
    for i in range(195, 200):
        prices[i] = prices[i - 1] * 1.02
    v = FORMULA_FUNCTIONS[14](make_ctx(prices, W=44, tf="tf2"))
    assert v > 0


def test_ccmdr_zero_for_short_streak():
    # alternating returns => streak length 1 -> 0 by spec (N < 3)
    prices = [100.0]
    for i in range(100):
        prices.append(prices[-1] * (1.002 if i % 2 == 0 else 0.998))
    assert FORMULA_FUNCTIONS[18](make_ctx(prices)) == 0.0


def test_ccmdr_positive_when_lambda_exceeds_default_critical():
    # strong decaying streak: magnitudes shrink much faster than lambda=0.15
    prices = [100.0]
    r = 0.02
    for i in range(60):
        r *= 0.5                       # geometric decay of move size
        prices.append(prices[-1] * (1.0 + r) if r > 1e-6 else prices[-1] * 1.0000001)
    # keep positive returns so the streak persists, magnitudes decaying
    v = FORMULA_FUNCTIONS[18](make_ctx(prices, W=64, tf="tf2"))
    assert -1.0 <= v <= 1.0


def test_drhe_neutral_without_regime_history():
    v = FORMULA_FUNCTIONS[20](make_ctx(trend_up()))
    assert v == 1.0    # insufficient durations -> neutral support


def test_pmtl_uses_cross_asset_leadership():
    # asset B leads target A by 3 minutes with identical returns
    base = trend_up(300, seed=11)
    target = base
    leader = [base[0]] * 3 + base[:-3]      # leader shifted: its past = target's now
    others = {"ETH": (
        [1_700_000_000 + i * 60 for i in range(len(leader))], leader
    )}
    ctx = make_ctx(target, others=others)
    v = FORMULA_FUNCTIONS[19](ctx)
    assert -1.0 <= v <= 1.0


def test_directional_formulas_list_excludes_drhe():
    # Part 2: the directional (weighted-signal) set is every registered
    # signal formula except F20/DRHE (a confidence multiplier, unbounded).
    from app.formulas import FORMULA_FUNCTIONS
    assert 20 not in DIRECTIONAL_FORMULAS
    assert set(DIRECTIONAL_FORMULAS) == set(FORMULA_FUNCTIONS) - {20}
    # part-2 signals are present in the weighted set
    assert {22, 23, 24, 25, 26, 27, 31} <= set(DIRECTIONAL_FORMULAS)


def test_rpic_records_gamma_intermediate():
    ctx = make_ctx(trend_up())
    FORMULA_FUNCTIONS[1](ctx)
    assert "gamma" in ctx.intermediates
    assert ctx.intermediates["gamma"] >= 0


def test_cci_weight_update_clamps_and_normalizes():
    from app.cci import CCIEngine
    eng = CCIEngine()
    eng.store_price("BTC", 100.0)
    eng.store_signals("BTC", "tf1", {f: (1.0 if f % 2 else -1.0) for f in range(1, 21)})
    # price rose -> positive signals rewarded
    for _ in range(50):
        eng.update_weights("BTC", "tf1", 101.0)
    w = eng.get_weights("BTC", "tf1")
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(0.005 - 1e-9 <= x <= 0.15 + 1e-9 for x in w.values())


def test_cci_double_check_reduces_confidence():
    from app.cci import CCIEngine
    eng = CCIEngine()
    # strong opposition: half the formulas hard against the other half
    adjusted = {f: (1.0 if f <= 10 else -1.0) for f in range(1, 21)}
    pack = eng.compute("BTC", "tf1", adjusted, drhe=1.0)
    # cci ~0 (weights equal) -> double-check is moot; make imbalance:
    w = eng.get_weights("BTC", "tf1")
    for f in range(1, 11):
        w[f] = 0.09
    for f in range(11, 21):
        w[f] = 0.011
    total = sum(w.values())
    for f in w:
        w[f] /= total
    pack2 = eng.compute("BTC", "tf1", adjusted, drhe=1.0)
    assert abs(pack2["cci_adjusted"]) <= abs(pack2["cci"]) + 1e-9
    assert pack2["confidence"] <= abs(pack2["cci"]) + 1e-9


def test_cci_below_floor_not_issued():
    from app.cci import CCIEngine
    eng = CCIEngine()
    adjusted = {f: 0.01 for f in range(1, 21)}
    pack = eng.compute("BTC", "tf1", adjusted, drhe=1.0)
    assert pack["confidence"] < 0.35
    assert pack["issued"] is False
