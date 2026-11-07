"""docs11-doublecheck: every window gets a side; gates become tiers."""
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
JS = (ROOT / "static" / "app.js").read_text()
ENG = (ROOT / "app" / "engine.py").read_text()


def test_no_no_trade_ui():
    assert "no trade" not in JS and "NO TRADE" not in JS


def test_issued_means_side_issued():
    assert 'fin["issued"] = True' in ENG
    assert 'fin["exec_ok"]' in ENG               # execution keeps its gate
    assert '"issued": bool(cyc.issued and cyc.tier != "watch-only")' in ENG


def test_recheck_and_vetoes_are_advisory():
    assert 'final["signal"] = "NO TRADE"' not in ENG
    assert "recheck_advisory" in ENG
    assert 'res["direction"] = 0' not in ENG     # tape veto can't flatten
    assert 'res["tier"] = "watch-only"' in ENG


def test_tier_math():
    assert "def _tier_for" in ENG
    assert 'floor_s = max(0.02, 0.5 * med)' in ENG   # floor in score units


def test_tier_thresholds_pure():
    # replicate the tier rule: P from |s|/median via logistic
    import math
    def p_side(s, med): return 1 / (1 + math.exp(-abs(s) / med)) if med > 0 else 0.5
    assert p_side(0.0, 0.1) == 0.5
    assert abs(p_side(0.1, 0.1) - 0.731) < 0.01   # typical score ≈ 0.73 (doc §4)
