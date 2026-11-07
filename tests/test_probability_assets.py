"""Empirical probabilities + per-asset prediction behaviour.

Probability: first-touch outcomes REPLAYED over the asset's own bar
path (counted, with n), not a driftless erfc formula. Behaviour: gold
amplifies in trends / damps in chop, ETH damps in vol blow-offs, BNB
pulls toward reversion when stretched, BTC is the reference.
"""
from pathlib import Path

from app.asset_behavior import regime, temper

ROOT = Path(__file__).resolve().parent.parent


def _trend(n=80, step=0.001):
    out, p = [], 100.0
    for _ in range(n):
        p *= (1 + step)
        out.append(p)
    return out


def _chop(n=80):
    out, p = [], 100.0
    for i in range(n):
        p *= (1 + (0.001 if i % 2 else -0.001))
        out.append(p)
    return out


def test_gold_amplifies_trend_damps_chop():
    t = temper("PAXG", 0.20, 0.30, _trend())
    assert t["factor"] == 1.12 and t["score"] > 0.20
    c = temper("PAXG", 0.20, 0.30, _chop())
    assert c["factor"] == 0.82 and c["score"] < 0.20


def test_eth_damps_vol_blowoff_btc_reference():
    closes = _chop(70) + [100.0 * (1 + 0.02 * ((i % 3) - 1)) for i in range(15)]
    e = temper("ETH", 0.30, 0.40, closes)
    assert e["factor"] == 0.78 and e["confidence"] < 0.40
    b = temper("BTC", 0.30, 0.40, closes)
    assert b["score"] == 0.30 and b["factor"] == 1.0


def test_bnb_reversion_pull_against_stretch():
    closes = _trend(80, 0.004)          # stretched far above the mean
    r = temper("BNB", 0.05, 0.30, closes)
    assert r["score"] < 0.05            # pulled back toward reversion
    assert "reversion pull" in r["action"]


def test_regime_measures_trend_efficiency():
    assert regime(_trend())["trend_eff"] > 0.9
    assert regime(_chop())["trend_eff"] < 0.1


def test_branch_stats_is_empirical_with_n():
    src = (ROOT / "app" / "engine.py").read_text()
    i = src.index("def branch_stats")
    seg = src[i:i + 19000]
    assert "empirical-first-touch" in seg and '"n": _n' in seg
    assert "STOP first" in seg                      # conservative both-hit
    assert "driftless-fallback (thin tape)" in seg  # honest fallback only


def test_engine_wires_temper_before_issue():
    src = (ROOT / "app" / "engine.py").read_text()
    assert "from .asset_behavior import temper as _temper" in src
    assert 'fin["behavior"] = _beh' in src
    assert "noise_mult" in src


def test_charts_true_window_axis_and_series():
    js = (ROOT / "static" / "app.js").read_text()
    assert "function realisedSeries(" in js
    assert "d.cycle_start" in js and "nowT: now" in js  # PriceChart: true window axis + NOW divider
    assert "S.candleSeries = { symbol: S.symbol, bars: seeded }" in js
    assert "loadCandles().catch(() => {})" in js     # 30 s refresh wired
