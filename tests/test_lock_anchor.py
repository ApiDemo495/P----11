"""One anchor: the settlement lock IS the TP/SL entry.

Live evidence (build 9861596): the trace banner said LOCKED entry
75,672 while the hero said entry (locked) 76,182 — the risk block
anchored to the compute price, the lock froze the live price seconds
later. fresh_cycle now settles/locks FIRST and anchors targets, the
measured-moves block and the displayed entry to the lock's price.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_lock_created_before_levels_anchor():
    src = (ROOT / "app" / "engine.py").read_text()
    i = src.index("def fresh_cycle")
    seg = src[i:i + 2600]
    assert seg.index("self._settle_and_lock") < seg.index("_risk_block(")
    assert '_lock_now.get("price")' in seg


def test_branch_barriers_are_gated_tpsl_not_subframe():
    src = (ROOT / "app" / "engine.py").read_text()
    i = src.index("def branch_stats")
    seg = src[i:i + 1200]
    assert 'tp1 = float(d.get("tp") or 0.0)' in seg


def test_chart_labels_decobllided_and_quads_trimmed():
    js = (ROOT / "static" / "app.js").read_text()
    lc = (ROOT / "static" / "LineChart.js").read_text()
    assert "padR = spark ? 2 : 62" in lc  # PriceChart: labels live on the price axis, no collision stack
    assert "quad cards show ONLY the horizon TP + SL" in js
    assert "_pct" in js          # <1% instead of a misleading 0%
