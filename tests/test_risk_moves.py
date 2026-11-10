"""docs11-charts Part B acceptance (adapted to 22/44/130/247m windows).

1. Levels from MEASURED moves: BUY tp>entry>sl, SELL mirrored, rr>=1.
2. σ cap (C-7): one 300bp print cannot push tp_bps past q95-ish.
3. Moves table updates (C-6): a calmer tape gives smaller bps.
4. n<5 windows -> provisional ATR fallback, labelled.
5. Emergency re-lock re-anchors the same bps at the new entry.
6. Ledger records level outcome as its own column (B-3).
"""
from __future__ import annotations

import math
import os
import tempfile

from app.moves_risk import moves_stats, reanchor, risk_block


def _tape(n=600, drift=0.0002, vol=0.001, jump_at=None):
    closes, highs, lows = [], [], []
    p = 100.0
    for i in range(n):
        p *= (1 + drift + vol * math.sin(i * 1.7))
        if jump_at is not None and i == jump_at:
            p *= 1.03                      # one 300bp print
        closes.append(p)
        highs.append(p * 1.001)
        lows.append(p * 0.999)
    return closes, highs, lows


def test_levels_from_measured_moves_both_sides():
    c, h, l = _tape()
    for direction in (1, -1):
        b = risk_block(100.0, direction, 22, 0.5, c, h, l, atr=0.1)
        assert b["n"] >= 5 and not b["provisional"]
        if direction > 0:
            assert b["tp"] > 100.0 > b["sl"]
        else:
            assert b["tp"] < 100.0 < b["sl"]
        # band rule: (0, 4:1]; sub-1:1 is allowed but must be flagged
        assert 0.0 < b["rr"] <= 4.0 and b["below_floor"] == (b["rr"] < 1.0)
        assert b["tp_bps"] > 5   # not the old ~2bp toy


def test_sigma_cap_against_one_bad_print():
    c, h, l = _tape()
    clean = risk_block(100.0, 1, 22, 0.5, c, h, l, atr=0.1)
    cj, hj, lj = _tape(jump_at=300)
    dirty = risk_block(100.0, 1, 22, 0.5, cj, hj, lj, atr=0.1)
    st = moves_stats(cj, hj, lj, 22)
    assert dirty["tp_bps"] <= max(st["q95"], clean["tp_bps"]) * 1.05 + 1
    assert dirty["sigma_bps"] <= st["q80"] / 1.2816 + 0.01


def test_moves_table_updates_between_tapes():
    c1, h1, l1 = _tape(vol=0.002)
    c2, h2, l2 = _tape(vol=0.0004)
    wild = risk_block(100.0, 1, 22, 0.5, c1, h1, l1, atr=0.1)
    calm = risk_block(100.0, 1, 22, 0.5, c2, h2, l2, atr=0.1)
    assert wild["tp_bps"] != calm["tp_bps"]     # C-6: never frozen numbers
    assert wild["tp_bps"] > calm["tp_bps"]


def test_provisional_below_five_windows():
    c, h, l = _tape(n=30)
    b = risk_block(100.0, 1, 22, 0.5, c, h, l, atr=0.5)
    assert b["provisional"] and "provisional" in b["note"]
    assert b["tp"] > 100.0 > b["sl"]


def test_reanchor_keeps_bps_at_new_entry():
    c, h, l = _tape()
    b = risk_block(100.0, 1, 22, 0.5, c, h, l, atr=0.1)
    r = reanchor(b, 110.0)
    assert r["tp"] > 110.0 > r["sl"]              # BUY sides still correct
    assert abs((r["tp"] / 110.0 - 1) * 1e4 - b["tp_bps"]) < 0.2
    r2 = reanchor(risk_block(100.0, -1, 22, 0.5, c, h, l, atr=0.1), 90.0)
    assert r2["tp"] < 90.0 < r2["sl"]             # SELL mirrored


def test_engine_relock_reanchors_and_ledger_level_outcome():
    from app.ledger import SettlementLedger
    led = SettlementLedger(path=os.path.join(tempfile.mkdtemp(), "l.json"))
    row = led.settle("BTC", "tf1", 100.0, 101.0, 1, level_outcome="TP first")
    assert row["level_outcome"] == "TP first"     # B-3: separate column
    st = led.stats("BTC", "tf1")
    assert st["tp_first_n"] == 1 and st["tp_first_pct"] == 1.0
    # engine wiring contracts
    eng = open(os.path.join(os.path.dirname(__file__), "..",
                            "app", "engine.py")).read()
    assert "_reanchor(_cyc.risk, price)" in eng
    assert "level_outcome=_lvl" in eng
    assert "risk_block as _risk_block" in eng
