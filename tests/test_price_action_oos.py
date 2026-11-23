"""OOS runner's streaming price-action predicate must match app runtime."""
import numpy as np

from app.price_action import read
from backtest.price_action_oos import PAStream, _metrics


def test_streaming_oos_pa_matches_app_read_on_causal_prefixes():
    rng = np.random.default_rng(71)
    n = 360
    t = 1_700_000_000 + np.arange(n) * 60
    o = np.empty(n); h = np.empty(n); l = np.empty(n); c = np.empty(n)
    c[0] = 100.0
    for i in range(n):
        if i:
            o[i] = c[i-1]
            c[i] = o[i] * np.exp(rng.normal(0, 0.002))
        else:
            o[i] = c[i]
        h[i] = max(o[i], c[i]) * (1 + rng.uniform(0, 0.006))
        l[i] = min(o[i], c[i]) * (1 - rng.uniform(0, 0.006))
    v = rng.uniform(10, 100, n)
    stream = PAStream(t, o, h, l, c, v)
    for cut in (60, 75, 120, 181, 240, 359, 360):
        stream.advance(cut)
        expected = read(t[:cut], o[:cut], c[:cut], h[:cut], l[:cut], v[:cut],
                        price=float(c[cut-1]), now=float(t[cut-1] + 60))
        actual = stream.read()
        assert actual["direction"] == expected["direction"]
        assert actual["score"] == expected["score"]
        assert actual["strong"] == expected["strong"]


def test_net_trade_metrics_exclude_ambiguous_and_charge_24bps():
    allowed = np.array([True, True, True, False])
    result = np.array([1, 2, -1, 3], dtype=np.int8)
    gross = np.array([50.0, np.nan, -10.0, np.nan])
    stats = _metrics(allowed, result, gross)
    assert stats["issued_n"] == 3
    assert stats["graded_n"] == 2 and stats["ambiguous_n"] == 1
    # 50bp gross TP -> 26bp net; -10bp stop -> -34bp net.
    assert stats["net_ev_bps"] == -4.0
    assert stats["strict_wr_net"] == 0.5
    assert stats["first_touch_wr"] == 0.5
    assert stats["pf_net"] == round(26 / 34, 6)
