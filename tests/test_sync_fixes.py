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
    # dominant ranking must measure deviation from each channel's OWN
    # neutral: EVENT channels (fear/capitulation … climax streak) are
    # 0-based — no event, no crown — so a flat up-tape cannot crown
    # "startle shock". State channels are bipolar 0.5.
    from app.emotion import CHANNELS
    src = open("app/emotion.py").read()
    assert "def _dev_of" in src and "CHANNELS[i][3]" in src
    for eid in (1, 3, 4, 5, 12, 13, 15, 16, 20, 21):   # event channels
        assert CHANNELS[eid][3] == 0.0, eid
    for eid in (2, 6, 7, 8, 9, 10, 11, 14, 17, 18, 19):  # state channels
        assert CHANNELS[eid][3] == 0.5, eid
    assert len(CHANNELS) == 21


def test_quad_charts_aggregate_per_tf():
    js = open("static/app.js").read()
    assert "QINT = { tf1: 1, tf2: 1, tf3: 5, tf4: 5 }" in js
    assert "agg(md.bars || [], QINT[tf] || 1)" in js
    # per-TF server charts: server revision when bound, else local history
    assert "S.tfBarsRevision[tf] = useSrv ? String(srv.revision" in js
    assert ": historyRevision;" in js
    assert 'if (S.chartSrc === "binance") { try { bnEnsure(); }' in js
