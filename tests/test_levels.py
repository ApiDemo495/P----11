"""TP/SL levels rewrite — the locked-entry invariant.

The live bug: BUY locked at 75,460 displayed SL 75,606 (ABOVE entry)
because the NO-TRADE GRC branch nulled the levels and the snapshot
re-seeded a stop from the CURRENT price. These tests pin the rewrite:
levels come only from the frozen block at the locked entry, are
guarded on every snapshot, and can never cross to the wrong side.
"""
from pathlib import Path

import pytest

from app import levels

ROOT = Path(__file__).resolve().parent.parent

BLOCK = {"tp_bps": 80.0, "sl_bps": 66.0, "provisional": False}


def test_buy_levels_anchor_to_locked_entry_not_spot():
    lv = levels.freeze(75460.0, 1, BLOCK)
    assert lv["tp"] > 75460.0 > lv["sl"]
    # spot later at 75,719 changes nothing — levels are frozen numbers
    assert lv["sl"] == pytest.approx(75460.0 * (1 - 66 / 1e4))


def test_sell_levels_mirror():
    lv = levels.freeze(80000.0, -1, BLOCK)
    assert lv["tp"] < 80000.0 < lv["sl"]


def test_guard_repairs_wrong_side_from_block_never_spot():
    bad = {"tp": 76072.0, "sl": 75606.0, "entry": 75460.0, "direction": 1}
    assert not levels.valid(bad)              # SL above BUY entry
    fixed = levels.guard(bad, BLOCK)
    assert fixed["tp"] > 75460.0 > fixed["sl"]


def test_guard_withholds_what_it_cannot_repair():
    assert levels.guard({"tp": 1, "sl": 2, "entry": 0, "direction": 1},
                        BLOCK) is None
    assert levels.freeze(0.0, 1, BLOCK) is None
    assert levels.freeze(100.0, 1, {"tp_bps": 0, "sl_bps": 0}) is None


def test_snapshot_never_seeds_from_current_price():
    src = (ROOT / "app" / "engine.py").read_text()
    assert "price - dirn * atr" not in src
    assert "LEVELS GATE" in src and "_levels.from_cycle" in src


def test_no_trade_grc_keeps_frozen_levels():
    src = (ROOT / "app" / "engine.py").read_text()
    i = src.index("# NO TRADE")
    seg = src[i:i + 900]
    assert "cyc.tp = cyc.sl = None" not in seg        # never nulled
    assert 'reason="retake_direction"' in src        # flip -> re-predict


def test_apply_risk_never_overwrites_unissued():
    src = (ROOT / "app" / "engine.py").read_text()
    i = src.index("def _apply_risk")
    assert "if not cyc.issued:" in src[i:i + 400]
