"""Precision pass: probabilities conditioned on matched setups
(momentum agreement + vol regime), each with its own n; the drawn fan
comes from the matched subset only when it has >= 50 windows."""
from app.engine import PredictionEngine


class _St:
    closes = []; highs = []; lows = []


def _feed_with(closes):
    class _F:
        def state(self, sym):
            st = _St()
            st.closes = closes
            st.highs = [c * 1.001 for c in closes]
            st.lows = [c * 0.999 for c in closes]
            return st
    return _F()


def test_branch_stats_reports_conditioned_probabilities():
    eng = PredictionEngine.__new__(PredictionEngine)
    import math
    closes = [100.0 * (1 + 0.002 * math.sin(i / 7.0) + 0.0004 * i / 50)
              for i in range(600)]
    eng.feed = _feed_with(closes)
    eng.last_result = {}
    d = {"symbol": "BTC", "direction": 1, "sl": 99.0, "tp": 102.0,
         "countdown_s": 600, "cycle_end": 0, "targets": [], "risk": {}}
    out = eng.branch_stats(d, 100.0, 0.0, "tf1")
    assert out and out["method"] == "empirical-first-touch"
    assert out["n"] >= 100 and "n_cond" in out
    if out["n_cond"] >= 30:
        assert 0.0 <= out["p_target_cond"] <= 1.0
        assert out["ev_bps_cond"] is not None
    assert out["fan_basis"] and ("n=" in out["fan_basis"])
    fan = out["fan"]
    assert fan[0]["q50"] == 100.0 or abs(fan[0]["q50"] - 100.0) < 1e-6
