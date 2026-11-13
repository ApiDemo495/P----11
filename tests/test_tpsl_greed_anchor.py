"""TP/SL 'in air' + Greed round 2 + probability-fan anchor.

From the user's screenshots on a live up-day:
* TP1 sat above price already crossed, still labelled LOCK — hit
  detection only ran for issued cycles, and levels are locked at the
  cycle start while the display never showed the anchor entry.
* The probability fan was spread across the whole cycle although the
  bands are quantiles of the REMAINING window from now.
* E2 (Greed) still pegged at exactly 1.00: pen mixed the day-scale
  session open with the last-60m low, so any up day saturated it.
* The LOCKED banner printed a raw Python dict for the emotion.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _rising_market(n=220):
    closes = [100.0 * (1.001 ** i) for i in range(n)]   # steady up-tape
    return {
        "closes": closes, "volumes": [10.0] * n,
        "times": [1700000000 + i * 60 for i in range(n)],
        "ohlc": {"highs": [c * 1.001 for c in closes],
                 "lows": [c * 0.999 for c in closes], "closes": closes},
        "live": closes[-1], "session_open": closes[0],
        "session_high": closes[-1] * 1.001,
        "btc_closes": closes, "peer_returns": [],
        "cmp_divergence": 0.0, "news_price_corr": 0.0,
        "octo_move": None,
    }


def test_e2_not_pegged_on_a_rising_tape():
    from app.emotion import EmotionSystem
    from app.models import FormulaContext
    from tests.test_engine import make_engine
    eng, _f, _e = make_engine()
    ctx = FormulaContext(symbol="BTC", tf="tf1", W=22,
                         P=_rising_market()["closes"][-60:],
                         V=[10.0] * 60, T=[1700000000] * 60,
                         hist=eng.hist)
    out = EmotionSystem().compute(ctx, _rising_market(),
                                  {"hormones_avg": {}}, {"polarity": 0,
                                                         "count": 0}, None)
    e2 = out["emotions"][2]
    assert e2 < 0.999, f"E2 still pegged at {e2} on a plain rising tape"
    assert out["dominant"]["id"] >= 1


def test_lock_emotion_is_human_text_not_a_dict():
    eng = (ROOT / "app" / "engine.py").read_text()
    assert 'isinstance(_dom, dict)' in eng
    assert 'f"E{_dom[' in eng and "intensity" in eng


def test_unissued_cycles_track_target_hits():
    eng = (ROOT / "app" / "engine.py").read_text()
    assert '"observed": True' in eng and 't.status = "hit"' in eng
    from app.models import TFCycle
    assert "sl_observed" in TFCycle.__dataclass_fields__


def test_branch_fan_anchored_at_now():
    eng = (ROOT / "app" / "engine.py").read_text()
    assert '"now": round(now, 3)' in eng
    js = (ROOT / "static" / "app.js").read_text()
    assert "originT: tN" in js                  # fan rooted at now (PriceChart model)
    lc = (ROOT / "static" / "LineChart.js").read_text()
    assert "levels: in-range = line + direct tag; off-scale = edge tag" in lc
    assert '{ p: b.tp, label: "TP", color: "#26a69a", dash: true }' in js


def test_chart_labels_and_liq_filter():
    js = (ROOT / "static" / "app.js").read_text()
    lc = (ROOT / "static" / "LineChart.js").read_text()
    assert "padL + plotW + 5" in lc                          # price labels on the right axis (PriceChart)
    assert "Math.abs(p - _ref) / _ref <= 0.02" in js      # far LIQ dropped
    assert "entry (locked contract)" in js and "from entry" in js  # frozen contract anchor shown
