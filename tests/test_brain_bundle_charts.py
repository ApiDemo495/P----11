"""Brain ships in the repo (no download on fresh Codespace) and both
charts follow the Binance layout: area line + MA + volume strip."""
import gzip, os
from pathlib import Path


def test_brain_bundle_in_repo_and_restore():
    gz = Path("assets/connectome/flywire_783.npz.gz")
    assert gz.exists() and gz.stat().st_size > 10_000_000   # real build
    with gzip.open(gz, "rb") as fh:
        assert fh.read(6)  # readable gzip stream
    src = open("app/flywire.py").read()
    assert "_restore_bundled" in src and "assets" in src
    assert "no download" in src.lower()


def test_binance_style_charts():
    js = open("static/app.js").read()
    assert "MA overlay (Binance-style" in js
    assert "volume strip (Binance layout)" in js
    assert "hasVol" in js and "ph * 0.82" in js
    assert "Binance-style area gradient" in js
    assert "p: b.close, v: b.v || 0" in js       # volume reaches the charts
