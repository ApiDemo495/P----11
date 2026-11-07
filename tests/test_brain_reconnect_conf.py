"""Brain resilience + confidence calibration.

1) A corrupt cached parquet/npz is PURGED so the next build downloads
   clean artefacts (was: same poisoned file retried for a day).
2) Failed builds auto-retry (bounded), manual RECONNECT resets budget.
3) Confidence is calibrated to the score's own history percentile —
   the raw chain (~0.001) could never cross the 0.35 floor, so no
   prediction could ever issue."""
from app.flywire import FlyWireBrain
import app.engine as eng


def test_purge_only_on_corrupt_errors(tmp_path, monkeypatch):
    import app.flywire as fw
    monkeypatch.setattr(fw, "CACHE_DIR", tmp_path)
    (tmp_path / fw.PARQUET).write_bytes(b"x" * 10)
    b = FlyWireBrain.__new__(FlyWireBrain)
    assert b._purge_corrupt(RuntimeError("could not deserialize thrift page header")) is True
    assert not (tmp_path / fw.PARQUET).exists()
    (tmp_path / fw.PARQUET).write_bytes(b"x" * 10)
    assert b._purge_corrupt(RuntimeError("network timeout")) is False
    assert (tmp_path / fw.PARQUET).exists()   # network errors keep cache


def test_header_reconnect_button():
    html = open("static/index.html").read()
    js = open("static/app.js").read()
    assert 'id="btn-brain"' in html and "⟳ BRAIN" in html
    assert 'api("/api/brain/reconnect", { method: "POST" })' in js


def test_confidence_calibration_wired():
    src = open("app/engine.py").read()
    assert "_conf_hist" in src and "conf_percentile" in src
    assert "len(_pool) >= 20" in src          # measured rank once 20 scores exist
    assert 'fin["confidence_raw"]' in src     # raw never hidden
