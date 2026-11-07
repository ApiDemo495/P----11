"""User rule: TP:SL = 2:1 everywhere. SL is the 1 unit (measured q80
adverse excursion), TP exactly 2 units — in the risk block, through
per-asset profile scaling, and in the analysis finale (which must use
the locked cycle levels, never confidence-projected TP = entry)."""
from app.moves_risk import risk_block
from app.levels import freeze, valid


def _tape(n=400, vol=0.001):
    import math
    c = [100.0 * (1 + vol * math.sin(i / 9.0)) for i in range(n)]
    h = [x * 1.001 for x in c]; l = [x * 0.999 for x in c]
    return c, h, l


def test_block_is_exactly_two_to_one():
    c, h, l = _tape()
    for d in (1, -1):
        b = risk_block(100.0, d, 22, 0.5, c, h, l, atr=0.1)
        assert abs(b["tp_bps"] - 2.0 * b["sl_bps"]) < 0.05
        assert abs(b["rr"] - 2.0) < 0.01
        f = freeze(100.0, d, b)
        assert valid(f) and abs(f["ratio"] - 2.0) < 0.01


def test_spread_does_not_break_ratio():
    c, h, l = _tape()
    b = risk_block(100.0, 1, 22, 0.5, c, h, l, atr=0.1, spread_bps=4.0)
    assert abs(b["tp_bps"] / b["sl_bps"] - 2.0) < 0.01


def test_analysis_uses_locked_levels_not_projection():
    js = open("app/engine.py").read()
    seg = js[js.index("# 7 ── final"):js.index("job[\"result\"] = final")]
    assert "_lv_from_cycle(cyc)" in seg
    assert "_project_level" not in seg          # TP = entry bug factory gone
    assert "locked cycle" in seg
    assert "2:1 block" in seg                   # no-lock fallback is 2:1 too
