"""NEW app/chartlink.py — measured chart<->price link + watchdog."""
import time

from app.chartlink import LinkWatchdog, all_reports, link_report


class _St:
    def __init__(self, live, closes, times, sources=None):
        self.live_price, self.opens = live, list(closes)
        self.closes, self.times = list(closes), list(times)
        self.highs, self.lows = list(closes), list(closes)
        self.sources = list(sources or ["sim"] * len(times))

    def indices_for_family(self, family):
        from app.feed import _family
        return [i for i, source in enumerate(self.sources[:len(self.times)])
                if _family(source) == family]


class _Feed:
    mode, source = "sim", "sim"
    def __init__(self): self._s = {}
    def state(self, sym): return self._s[sym]


def test_link_report_measures_delta_and_age():
    f = _Feed(); now = int(time.time())
    f._s["BTC"] = _St(100.0, [99.0, 100.0], [now - 120, now - 60])
    r = link_report(f, "BTC", now=float(now))
    assert r["delta_bps"] == 0.0 and r["bar_age_s"] == 60 and r["ok"]


def test_link_report_flags_desync():
    f = _Feed(); now = int(time.time())
    f._s["BTC"] = _St(103.0, [100.0], [now - 60])     # 3 % below live
    r = link_report(f, "BTC", now=float(now))
    assert r["delta_bps"] > 200 and r["ok"] is False


def test_link_report_flags_foreign_history_even_when_last_bar_matches():
    f = _Feed(); f.mode, f.source = "binance", "binance-ws"
    now = int(time.time())
    f._s["BTC"] = _St(68_769.0, [75_920.0, 68_769.0],
                      [now - 120, now - 60], ["sim", "binance-ws"])
    r = link_report(f, "BTC", now=float(now))
    assert r["delta_bps"] == 0.0
    assert r["last_bar_close"] == 68_769.0
    assert r["foreign_bars"] == 1
    assert r["source_mix_ok"] is False and r["ok"] is False


def test_watchdog_logs_persistent_desync_only():
    import asyncio
    f = _Feed(); now = int(time.time())
    f._s["BTC"] = _St(103.0, [100.0], [now - 60])
    wd = LinkWatchdog(f, ["BTC"])

    async def drive():
        wd.last = all_reports(f, ["BTC"])
        for _ in range(3):
            wd.last = all_reports(f, ["BTC"])
            wd.checks += 1
            for sym, r in wd.last["assets"].items():
                if not r["ok"]:
                    wd._streak[sym] = wd._streak.get(sym, 0) + 1
                    if wd._streak[sym] == 3:
                        wd.incidents.append({"ts": time.time(), "symbol": sym,
                                             "delta_bps": r["delta_bps"],
                                             "bar_age_s": r["bar_age_s"],
                                             "live_price": r["live_price"],
                                             "last_bar_close": r["last_bar_close"],
                                             "mode": r["mode"], "source": r["source"]})
    asyncio.run(drive())
    assert len(wd.incidents) == 1 and wd.incidents[0]["symbol"] == "BTC"


def test_endpoint_and_panel_exist():
    import pathlib
    root = pathlib.Path(__file__).parent.parent
    assert "/api/chart-link" in (root / "app" / "main.py").read_text()
    js = (root / "static" / "app.js").read_text()
    assert "pollChartLink" in js and "repairChartLink" in js
    assert 'id="link-panel"' in (root / "static" / "index.html").read_text()
    assert 'id="btn-repair-link"' in (root / "static" / "index.html").read_text()
