"""docs11-charts (7) Part D — desync defence + honest references."""
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
JS = (ROOT / "static" / "app.js").read_text()
FEED = (ROOT / "app" / "feed.py").read_text()
ENG = (ROOT / "app" / "engine.py").read_text()


def test_desync_watchdog_on_chart():
    assert "CHART DESYNC" in JS and "live bar stalled" in JS
    assert "desync > 0.002" in JS                    # spec D-4 threshold
    assert "S._desyncAt" in JS                       # rebuild throttled


def test_sim_tape_labelled_on_chart():
    assert "SIM TAPE — simulated feed" in JS         # doc D-3 honesty


def test_ist_render_adapter_only_at_library_boundary():
    assert "const lt = t => t + TZ_OFF" in JS and "19800" in JS
    assert JS.count("lt(") >= 10                     # every LWC crossing


def test_ticker_reference_is_exchange_24h_open():
    assert "ticker/24hr" in FEED and "openPrice" in FEED   # doc D-6


def test_hit_ts_recorded_and_serialised():
    from app.models import TargetState
    t = TargetState(k=1, minutes=5, level=1.0, status="hit")
    t.hit_ts = 42.0
    assert t.to_dict()["hit_ts"] == 42.0
    assert ENG.count("t.hit_ts = now") == 2          # both first-touch sites
