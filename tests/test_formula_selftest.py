"""Advice §2.5/§2.6/§2.2: self-test tapes, zero reasons, double checks."""
from app.formulas.selftest import TAPES, run_selftest
from app.formulas.trace import (
    DOUBLE_CHECK, PRIMARY_GUARD, run_double_check, zero_reason,
)


def test_selftest_tapes_all_green():
    r = run_selftest()
    assert r["ok"], r["failures"][:8]
    assert set(r["tapes"]) == set(TAPES)


def test_bull_bear_opposite_sign():
    r = run_selftest()
    assert r["claims"]["bull_bear_opposite"] is True


def test_flat_median_quiet():
    r = run_selftest()
    assert abs(r["tapes"]["FLAT"]["median"]) <= 0.35


def test_every_formula_has_identity_and_guard():
    from app.formulas import FORMULA_FUNCTIONS
    for fid in FORMULA_FUNCTIONS:
        assert fid in DOUBLE_CHECK, f"F{fid} missing declared identity"
        assert DOUBLE_CHECK[fid], f"F{fid} empty identity"
        assert fid in PRIMARY_GUARD, f"F{fid} missing data guard"


def test_zero_reason_contract():
    assert zero_reason(1, 0.0, 8) == "needs 15 bars, had 8"
    assert "genuinely flat" in zero_reason(1, 0.0, 40)
    assert "inner gate" in zero_reason(6, 0.0, 40)
    assert zero_reason(1, 0.42, 40) is None        # non-zero -> no reason needed


def test_double_check_labels_and_failure_mode():
    ok = run_double_check(1, 0.25, 0.25)
    assert ok["ok"] and ok["label"] == "✓✓ verified" and ok["identity"]
    bad = run_double_check(1, 0.25, -0.25)        # replay mismatch
    assert not bad["ok"] and bad["label"].startswith("✗")
    oor = run_double_check(1, 3.7, 3.7)           # out of range
    assert not oor["ok"]
