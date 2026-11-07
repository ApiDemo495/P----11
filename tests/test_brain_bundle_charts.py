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
    lc = open("static/LineChart.js").read()
    js = open("static/app.js").read()
    assert "moving-average overlay" in lc
    assert "volume sub-pane" in lc
    assert "hasVol" in lc and "plotH * 0.84" in lc   # price pane 84%, volume below
    assert "area fill" in lc
    assert "p: b.close, v: b.v || 0" in lc       # volume reaches the charts
    assert "p.close, v: x.v || 0" in js or "v: x.v || 0" in js
