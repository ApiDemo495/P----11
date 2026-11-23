"""Price-action input is closed-bar, strict, and context telemetry only."""
from app.price_action import _trigger, read


def test_strict_pin_requires_rejection_wick():
    direction, note = _trigger(100.0, 101.0, 95.0, 100.5,
                               100.0, 101.0, 99.0, 100.2)
    assert direction == 1
    assert "bullish pin" in note
    # A small wick cannot qualify as a pin.
    direction, _ = _trigger(100.0, 100.4, 99.8, 100.2,
                            100.0, 100.3, 99.8, 100.1)
    assert direction == 0


def test_strict_engulfing_requires_full_body_and_close_beyond():
    direction, note = _trigger(99.0, 102.0, 98.0, 101.0,
                               101.0, 102.0, 99.0, 100.0)
    assert direction == 1 and "engulfing" in note
    direction, _ = _trigger(100.0, 100.8, 99.8, 100.5,
                            101.0, 102.0, 99.0, 100.0)
    assert direction == 0


def test_price_action_never_reads_forming_bar():
    # Exactly 60 minutes of bars, with the final timestamp in the current
    # minute: only 59 are closed, so the result must be neutral.
    minute_now = 20_000 * 60
    times = [minute_now - (59 - i) * 60 for i in range(60)]
    values = [100.0 + i * 0.01 for i in range(60)]
    out = read(times, values, values, [x + 0.1 for x in values],
               [x - 0.1 for x in values], [1.0] * 60,
               price=values[-1], now=minute_now + 10)
    assert out["direction"] == 0 and out["score"] == 0
    assert out["strong"] is False


def test_price_action_is_neutral_on_insufficient_closed_history():
    out = read(None, [100.0] * 59, [100.0] * 59, [101.0] * 59,
               [99.0] * 59, [1.0] * 59)
    assert out["direction"] == 0 and out["score"] == 0
