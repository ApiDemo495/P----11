"""The per-TF backtest must grade each call over its OWN window.

Found by re-deriving from the committed CSV: the tape is hourly, TF_W is in
MINUTES, and the settle index was ``bars[t + W]``. So

    tf1 "22 minutes"  was graded 22 HOURS later
    tf4 "247 minutes" was graded 247 HOURS (10.3 days) later

Every cell in backtest/results/per_tf_full_engine_real.csv is therefore a
60x-670x-stretched horizon, which is also why its time grid steps 22 h.
"""
from __future__ import annotations

from backtest.per_tf_real import TF_W, bars_for, STEP


def test_minutes_are_converted_to_bars():
    # hourly tape
    assert bars_for(22, 3600) == 1
    assert bars_for(44, 3600) == 1
    assert bars_for(130, 3600) == 2
    assert bars_for(247, 3600) == 4
    # on a 5-minute tape the same windows are real
    assert bars_for(22, 300) == 4
    assert bars_for(247, 300) == 49


def test_horizon_is_not_stretched_by_a_factor_of_sixty():
    for tf, minutes in TF_W.items():
        n = bars_for(minutes)
        # the graded horizon in hours must equal the window in minutes
        assert n * 1 == max(1, round(minutes / 60)), tf
        # and must never equal the raw minute count (the old bug)
        if minutes > 60:
            assert n != minutes, tf


def test_settle_and_barrier_scan_use_the_same_converted_index():
    src = open("backtest/per_tf_real.py", encoding="utf-8").read()
    assert "_nb = bars_for(W)" in src
    assert "settle = bars[t + _nb].c" in src
    assert "bars[t + 1:t + _nb + 1]" in src
    # the buggy index must not survive anywhere
    assert "bars[t + W].c" not in src
    assert "bars[t + 1:t + W + 1]" not in src
