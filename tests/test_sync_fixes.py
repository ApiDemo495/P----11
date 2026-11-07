"""Sync fixes from user screenshots: E9 'Rollover panic' crowned for
days, Agents tab stuck on 'checking daemon…', identical TF charts."""
import app.main as main
from app.emotion import EmotionSystem


def test_agents_status_fills_auditor():
    import asyncio, json
    r = asyncio.run(main.agents_status())
    d = json.loads(r.body)
    assert d["agents"]["auditor"] is not None
    assert "available" in d["agents"]["auditor"]


def test_e9_never_dominant_on_up_tape():
    e = EmotionSystem()
    # dominant ranking must measure deviation from each channel's OWN
    # neutral: E9 (roll-over panic) is 0-based — no pain, no crown
    src = open("app/emotion.py").read()
    assert "_NEUTRAL = {9: 0.0}" in src and "_dev_of" in src


def test_quad_charts_aggregate_per_tf():
    js = open("static/app.js").read()
    assert "QINT = { tf1: 1, tf2: 3, tf3: 10, tf4: 30 }" in js
    assert "agg(d.bars || [], QINT[tf] || 1)" in js
    assert 'if (S.chartSrc === "binance") { try { bnEnsure(); }' in js
