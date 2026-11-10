"""User rule (current): TP:SL 2:1 preferred; measured ratios from
1:1 to 4:1 are shown, above 4:1 is capped, below 1:1 is flagged
below_floor (watch-only, never executes). The provisional ATR path
uses exactly 2:1. The analysis finale must use the locked cycle
levels, never a confidence-projected TP = entry."""
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
        # measured path: ratio inside the sanctioned band, cap 4:1,
        # sub-1:1 honestly flagged (watch-only downstream)
        assert 0.0 < b["rr"] <= 4.0 + 1e-9
        assert b["below_floor"] == (b["rr"] < 1.0)
        f = freeze(100.0, d, b)
        assert valid(f) and abs(f["ratio"] - b["rr"]) < 0.01
    # provisional path (too few windows): exactly the 2:1 preference
    b = risk_block(100.0, 1, 22, 0.5, c[:30], h[:30], l[:30], atr=0.1)
    assert b["provisional"] and abs(b["rr"] - 2.0) < 0.01


def test_spread_does_not_break_ratio():
    c, h, l = _tape()
    b0 = risk_block(100.0, 1, 22, 0.5, c, h, l, atr=0.1)
    b = risk_block(100.0, 1, 22, 0.5, c, h, l, atr=0.1, spread_bps=4.0)
    # spread widens both legs; ratio stays in-band and near the
    # no-spread ratio (it is NOT forced back to exactly 2:1)
    assert 0.0 < b["rr"] <= 4.0 + 1e-9
    assert abs(b["rr"] - b0["rr"]) < 0.35


def test_analysis_uses_locked_levels_not_projection():
    js = open("app/engine.py").read()
    seg = js[js.index("# 7 ── final"):js.index("job[\"result\"] = final")]
    assert "_lv_from_cycle(cyc)" in seg
    assert "_project_level" not in seg          # TP = entry bug factory gone
    assert "locked cycle" in seg
    assert "2:1 block" in seg                   # no-lock fallback is 2:1 too
