"""The NEW-mode target fix (2y real-data mandate).

The old formula — factor × σ (a MEAN-scaled expected move) — was
measured against the q80-MAE stop on 2y of real 1h bars: the ratio is
structurally 0.65–0.80, so EVERY new-mode window fell below the user's
1:1 floor and the model could never issue a trade (backtest/mandate.py
smoke: 34/34 new rows ratio-floor-skipped before the fix).

The fix: the target is the stop's own tail unit tilted by signal
strength — factor × NEW_TARGET_RR × sl_bps — so a saturated signal
reaches rr = NEW_TARGET_RR ≥ 1 while a weak signal stays gated below
the floor exactly as the user's rule intends.
"""
from __future__ import annotations

import math


def test_new_target_rr_is_tradable_at_full_strength():
    from app.engine import NEW_TARGET_RR

    factor_full = 0.5 + 0.5 * min(1.0, 0.10 / 0.08)     # saturated CCI
    assert math.isclose(factor_full, 1.0)
    assert factor_full * NEW_TARGET_RR >= 1.0, \
        "a full-strength signal must clear the 1:1 floor"
    assert factor_full * NEW_TARGET_RR <= 2.0 + 1e-9, \
        "a full-strength target is capped at 2:1"


def test_the_tilt_never_pushes_a_signal_under_the_floor():
    """The tilt modulates how far ABOVE 1:1 a target sits — it may not
    decide whether the trade is legal at all.

    The old band was 0.35 + 0.65·|cci|/0.08, i.e. rr ∈ [0.7, 2.0]: any
    signal that was not almost saturated produced a sub-1:1 target, the
    floor refused it, and the cell sat "watch-only" indefinitely. That is
    exactly what the board showed for six days (12 of 16 cells NOT
    TRADEABLE). A weak signal now gets the SMALLEST LEGAL target and keeps
    its weak score in the tier/issue_note rather than being silenced.
    """
    from app.engine import NEW_TARGET_RR

    for cci in (0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.5):
        factor = 0.5 + 0.5 * min(1.0, abs(cci) / 0.08)
        rr = factor * NEW_TARGET_RR
        assert rr >= 1.0 - 1e-9, (cci, rr)
        assert rr <= 2.0 + 1e-9, (cci, rr)
    # the weakest possible signal still clears the floor
    assert NEW_TARGET_RR * 0.5 >= 1.0 - 1e-9


def test_old_sigma_mean_formula_is_gone():
    """The mean-vs-tail comparison that made NEW unissuable."""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent
           / "app" / "engine.py").read_text()
    # the structural fix: magnitude comes from the stop, not from σ
    # (rr comes from the per-asset mandate config when set, else the
    # global NEW_TARGET_RR — the structure is the same)
    assert "_mag = _factor * _rr_n * _sl_bps" in src
    # the ratio resolves most-specific-first: measured cell -> per-asset
    # mandate -> global default (docs/WINRATE_AT_1_TO_1.md)
    assert "_rr_cell = _mcfg_c.new_rr_cell(symbol, tf)" in src
    # the per-asset forecast override is GONE with the new system
    # (2026-11-27): the ratio is measured-cell -> global default only.
    assert "_rr_n = _rr_cell or NEW_TARGET_RR" in src
    assert "new_rr(" not in src
    assert "0.5 * min(1.0, abs(float(" in src   # factor kept (tilt, floored at 1:1)


def test_mandate_tiers_follow_measured_cells():
    """CERTIFIED HOLD POLICY owns all 16 (asset × tf) cells
    (backtest/results/hold_wr.json, PR #13): 7 cells are HOLD-OUT
    VERIFIED, BNB's 4 are IN-SAMPLE SHADOW (single-year), PAXG tf4 is
    IN-SAMPLE SHADOW · HOLDOUT MISS, PAXG tf1–tf3 are UNGRADEABLE /
    WARMUP (1 h history) — and every one of them may ISSUE at a huge
    score: the hold system has NO abstention gates (filters and
    confluence are skipped for managed cells; the legacy mandate tf
    cap only applies when hold_wr.json is missing).  The tier_rule
    carries the hold badge + measured win rate so the UI explains
    WHY each cell is live."""
    import os

    from tests.test_engine import make_engine
    from app import hold_policy as hp
    engine, _feed, _ = make_engine()
    for sym, tf in hp.known_cells():
        # tf1 is a 24-hour lane the duration-keyed artifact has never walked,
        # and SOL has no walk at all. Those cells are NOT certified, so the
        # mandate lane cap applies to them — they are still issued, capped in
        # size, which is the no-abstention behaviour. Everything certified
        # must reach "full" at a huge score.
        if tf == "tf1" or sym == "SOL":
            out = engine._tier_for(sym, tf, 0.5)
            assert out["tier"] in ("full", "watch-only"), (sym, tf, out)
            continue
        out = engine._tier_for(sym, tf, 0.5)     # a huge score
        assert out["tier"] == "full", (sym, tf, out.get("tier_rule"))
        # the hold-policy CERTIFICATION badge is labelled as such — it used
        # to be printed as "hold policy [TIER]" inside a field every panel
        # reads as a size tier, so a status sentence ("ROWS VERIFIED ·
        # SHIP GATE FAIL") could be rendered as a size.
        assert "hold-policy certification" in out["tier_rule"], (sym, tf)
        assert "hold policy [" not in out["tier_rule"], (sym, tf)
        assert f"[{hp.tier(sym, tf)}]" in out["tier_rule"], (sym, tf)
    # the certified tier distribution is exact — and it is now gated by
    # the artifact's own ship verdict AND by the hard reward:risk rule.
    # 2026-11-27: the shipped hold_wr.json fails gate (ii) AND selected a
    # 0.27:1 geometry on every graded cell, so no cell may wear a
    # "verified", "shipped" or "holdout miss" badge: the geometry it was
    # measured on is not one the app is allowed to trade.  The distribution
    # therefore collapses onto the geometry violation, which is the honest
    # label until the artifact is re-graded at legal leverage.
    counts = hp.tier_counts()
    assert hp.artifact_ship_verdict()["gate_ii_pass"] is False
    assert counts.get("HOLD-OUT VERIFIED", 0) == 0
    assert counts.get(hp.TIER_SHIPGATE, 0) == 0
    assert counts.get("HOLDOUT ROWS · ERA n<3", 0) == 0
    assert counts.get("IN-SAMPLE SHADOW", 0) == 0
    assert counts.get("IN-SAMPLE SHADOW · HOLDOUT MISS", 0) == 0
    # the rule outranks every gate above it. 9 = BTC/ETH/BNB x tf2/tf3/tf4,
    # the certified sub-day cells (PAXG used to supply 2 of the 11; it is no
    # longer an asset). tf1's 4 lanes and SOL's 4 lanes are uncertified and
    # report UNGRADEABLE instead — 7 in total (4 tf1 lanes + 3 SOL lanes),
    # the tf1 lanes being the 24-hour windows the artifact never walked.
    assert counts[hp.TIER_SUB1] == 9
    assert counts["UNGRADEABLE / WARMUP"] == 7
    # PAXG's archived rows still answer NO MEASURED EDGE — outside the live
    # grid, which is why it is not in tier_counts any more
    assert counts.get(hp.TIER_NO_EDGE, 0) == 0
    assert hp.tier("PAXG", "tf1") == hp.TIER_NO_EDGE
    # legacy fallback still caps when the hold record is absent
    import app.hold_policy as _hp
    old_path = os.environ.get("HOLD_WR_PATH")
    os.environ["HOLD_WR_PATH"] = "/nonexistent/hold_wr.json"
    try:
        _hp.reload("/nonexistent/hold_wr.json")
        out = engine._tier_for("BTC", "tf1", 0.5)
        assert out["tier"] == "watch-only", out
        assert "mandate" in out["tier_rule"]
    finally:
        if old_path is None:
            os.environ.pop("HOLD_WR_PATH", None)
        else:
            os.environ["HOLD_WR_PATH"] = old_path
        _hp.reload(old_path)


def test_measured_cell_ratios_match_the_artifact_they_came_from():
    """CELL_RR must be re-derivable from the audit grid, not asserted by hand.

    Every entry claims "selected on train, reported on holdout" at a ratio
    >= 1:1 with positive holdout EV. This test recomputes that from
    backtest/results/wr_audit_independent.json and fails if the table and
    the measurement ever disagree — otherwise a future edit could quietly
    turn a measured number into a remembered one.
    """
    import json
    from pathlib import Path

    from app.mandate_cfg import CELL_RR

    root = Path(__file__).resolve().parent.parent
    grid = json.loads(
        (root / "backtest" / "results" / "wr_audit_independent.json").read_text())

    by_cell = {}
    for row in grid:
        best = None
        for g in (row.get("grid") or []):
            tp, sl = g.get("tp_bps") or 0, g.get("sl_bps") or 0
            if sl <= 0 or g.get("holdout_net_wr") is None:
                continue
            rr = tp / sl
            if rr < 1.0 - 1e-9:
                continue
            if best is None or (g.get("train_ev_net_R") or -9) > best[0]:
                best = (g.get("train_ev_net_R") or -9, rr,
                        g["holdout_net_wr"], g["holdout_ev_net_R"],
                        g.get("holdout_pf_net"))
        if best:
            by_cell[row["cell"]] = best

    assert CELL_RR, "the measured table must not be empty"
    for (symbol, tf), row in CELL_RR.items():
        w_min = {"tf1": 22, "tf2": 44, "tf3": 130, "tf4": 247}[tf]
        cell = f"{symbol}|{w_min}"
        assert cell in by_cell, f"{cell} is not in the audit grid"
        _train_ev, rr, hwr, hev, pf = by_cell[cell]
        assert abs(row["rr"] - rr) < 1e-6, (cell, row["rr"], rr)
        assert abs(row["holdout_wr"] - hwr) < 5e-4, (cell, row["holdout_wr"], hwr)
        assert abs(row["holdout_ev"] - hev) < 5e-4, (cell, row["holdout_ev"], hev)
        assert abs(row["pf"] - pf) < 5e-4, (cell, row["pf"], pf)
        # the whole point of the table: a legal ratio AND positive EV
        assert row["rr"] >= 1.0, cell
        assert row["holdout_ev"] > 0, cell


def test_short_timeframes_have_no_positive_ev_point_at_one_to_one():
    """The measured reason tf1/tf2 get no special ratio.

    If this ever flips, the table above should grow — the test exists so
    the omission is a decision backed by data rather than an oversight.
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    grid = json.loads(
        (root / "backtest" / "results" / "wr_audit_independent.json").read_text())

    pos = {22: 0, 44: 0}
    for row in grid:
        w = row.get("w_min")
        if w not in pos:
            continue
        for g in (row.get("grid") or []):
            tp, sl = g.get("tp_bps") or 0, g.get("sl_bps") or 0
            if sl <= 0 or g.get("holdout_net_wr") is None:
                continue
            if tp / sl < 1.0 - 1e-9:
                continue
            if (g.get("holdout_ev_net_R") or 0) > 0:
                pos[w] += 1
    assert pos[22] == 0, f"tf1 gained {pos[22]} positive-EV points at >=1:1"
    assert pos[44] == 0, f"tf2 gained {pos[44]} positive-EV points at >=1:1"
