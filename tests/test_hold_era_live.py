"""Live era accounting must not guess the order of a both-touch OHLC bar."""


def test_era_accountant_records_ambiguous_without_pnl_or_cost_guess(tmp_path):
    from app.hold_era import EraAccountant

    acct = EraAccountant("BTC", "tf3", str(tmp_path / "era.json"))
    acct.k, acct.m = 1.0, 1.0
    acct.fr_bps = 12.0
    acct.on_window(1, 100.0, 10.0, 1_000.0)
    assert acct.in_position and acct.tp_px == 110.0 and acct.sl_px == 90.0

    closed = acct.on_bars([(1_060.0, 111.0, 89.0)])
    assert len(closed) == 1
    assert closed[0]["outcome"] == "ambiguous"
    assert closed[0]["exit"] is None
    assert closed[0]["pnl_gross"] is None
    assert closed[0]["pnl_net"] is None
    assert closed[0]["friction_bps"] is None
    assert not acct.in_position
    assert acct.totals["eras"] == 0
    assert acct.totals["ambiguous"] == 1
    assert acct.totals["friction_unresolved"] == 1
    assert acct.totals["net"] == 0.0
    assert acct.snapshot()["era_win_rate"] is None


def test_era_accountant_still_grades_single_barrier_touch(tmp_path):
    from app.hold_era import EraAccountant

    acct = EraAccountant("BTC", "tf3", str(tmp_path / "era.json"))
    acct.k, acct.m = 1.0, 1.0
    acct.fr_bps = 12.0
    acct.on_window(1, 100.0, 10.0, 1_000.0)
    closed = acct.on_bars([(1_060.0, 111.0, 99.0)])
    assert len(closed) == 1 and closed[0]["outcome"] == 1
    assert closed[0]["pnl_net"] < closed[0]["pnl_gross"]
    assert acct.totals["touches"] == 1
    assert acct.totals["ambiguous"] == 0


def test_hold_tick_does_not_skip_bar_that_was_forming_last_tick(monkeypatch,
                                                               tmp_path):
    import time
    from tests.test_engine import make_engine
    from app import hold_era, hold_policy

    eng, feed, _ = make_engine()
    acct = hold_era.EraAccountant("BTC", "tf1", str(tmp_path / "era.json"))
    acct.k, acct.m = 1.0, 1.0
    acct.on_window(1, 100.0, 10.0, 0.0)
    monkeypatch.setattr(hold_policy, "managed",
                        lambda symbol, tf: symbol == "BTC" and tf == "tf1")
    monkeypatch.setattr(hold_era, "accountant", lambda symbol, tf: acct)

    base = int(time.time() // (22 * 60)) * (22 * 60) + 2 * 60
    st = feed.state("BTC")
    st.push_bar(base - 60, 100.0, 1.0, 100.0, 100.0, 100.0)
    st.push_bar(base, 100.0, 1.0, 100.0, 100.0, 100.0)
    s = base // (22 * 60) * (22 * 60)
    eng._hold_cursor[("BTC", "tf1")] = {"win": float(s), "bar_i": 0}

    # At first call, the bar timestamp equal to this minute is still forming.
    eng._hold_tick(base + 10.0)
    assert eng._hold_cursor[("BTC", "tf1")]["bar_i"] == 0
    assert acct.in_position

    # Feed closes that same timestamp with a both-touch range, then starts
    # a new forming minute. The next tick must process index 1 exactly once.
    st.push_bar(base, 100.0, 1.0, 100.0, 111.0, 89.0)
    st.push_bar(base + 60, 100.0, 1.0, 100.0, 100.0, 100.0)
    eng._hold_tick(base + 70.0)
    assert not acct.in_position
    assert acct.totals["ambiguous"] == 1
    assert eng._hold_cursor[("BTC", "tf1")]["bar_i"] == 1
