"""Forecast calibration: predictions correct themselves from graded
outcomes — and switch off if they stop helping. NOT 'perfect':
measured, bounded, self-disabling."""
from app.engine import _calibration_delta


def _audit(n, pred, realised, brier=0.20, brier_cal=None):
    a = {"n": n, "sum_pred": pred * n, "tp_first": int(realised * n),
         "brier_sum": brier * n}
    if brier_cal is not None:
        a["brier_cal_sum"] = brier_cal * n
    return a


def test_no_calibration_before_30_graded():
    assert _calibration_delta(_audit(29, 0.62, 0.40)) == 0.0


def test_overconfidence_corrected_down():
    assert _calibration_delta(_audit(100, 0.62, 0.54)) == -0.08


def test_correction_clamped():
    assert _calibration_delta(_audit(100, 0.90, 0.10)) == -0.15


def test_self_disables_when_calibration_hurts():
    a = _audit(100, 0.62, 0.54, brier=0.18, brier_cal=0.22)
    assert _calibration_delta(a) == 0.0
    a2 = _audit(100, 0.62, 0.54, brier=0.22, brier_cal=0.18)
    assert _calibration_delta(a2) == -0.08
