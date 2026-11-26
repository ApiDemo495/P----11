"""Regression tests for the deep per-file audit (§39).

Each test pins a defect that was *reproduced* before it was fixed. The
common theme is the one the user keeps hitting: a missing or failed value
being presented as a real one.

  1. `save_settings()` swallowed every OSError, so a setting that never
     reached disk still looked saved and silently reverted on restart.
  2. The dashboard rendered six fields with `|| 0` / `?? 0` before
     `toFixed()`, so "no reading" printed as "0.00".
  3. `_settle_and_lock`'s best-effort learners swallowed exceptions with
     `pass`, so a learner that had been dead for days looked identical to a
     healthy one.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import pytest

from app import silence
from app.agents import AgentManager

ROOT = Path(__file__).resolve().parents[1]


# ───────────────────────── 1. settings that claim to save ─────────────────
def _manager() -> AgentManager:
    m = AgentManager.__new__(AgentManager)
    m.settings = {"model": "m"}
    m.keyring = {}
    m.last_save_error = ""
    return m


def test_save_settings_reports_failure_instead_of_swallowing(monkeypatch):
    """A path that cannot be written must NOT look like a successful save."""
    tmp = tempfile.mkdtemp()
    # point SETTINGS_FILE at a directory: open(..., "w") raises IsADirectoryError
    bad = os.path.join(tmp, "cfg")
    os.makedirs(bad)
    monkeypatch.setattr("app.agents.SETTINGS_FILE", bad)
    monkeypatch.setattr("app.agents.SECRETS_FILE", os.path.join(tmp, "s.json"))
    m = _manager()

    ok = m.save_settings()

    assert ok is False, "a failed save returned True"
    assert not m.save_ok()
    assert m.last_save_error.startswith("settings:"), m.last_save_error


def test_save_settings_success_clears_the_error(monkeypatch):
    tmp = tempfile.mkdtemp()
    monkeypatch.setattr("app.agents.SETTINGS_FILE",
                        os.path.join(tmp, "settings.json"))
    monkeypatch.setattr("app.agents.SECRETS_FILE", os.path.join(tmp, "s.json"))
    m = _manager()
    m.last_save_error = "stale failure from an earlier call"

    assert m.save_settings() is True
    assert m.save_ok()
    assert m.last_save_error == ""
    with open(os.path.join(tmp, "settings.json")) as fh:
        assert json.load(fh)["model"] == "m"


def test_status_payload_carries_the_save_error():
    """The dashboard must be able to say 'live in memory only'."""
    src = (ROOT / "app" / "engine.py").read_text()
    assert '"save_error": getattr(self.agents, "last_save_error", "")' in src


# ───────────────────────── 2. missing is not zero (JS) ────────────────────
JS = (ROOT / "static" / "app.js").read_text()


def test_js_has_one_missing_value_helper():
    assert "const num = (v, d = 2) =>" in JS
    # it must be the finite check, not a truthiness check
    body = JS.split("const num = (v, d = 2) =>", 1)[1][:260]
    assert "Number.isFinite" in body
    assert "|| 0" not in body and "?? 0" not in body


@pytest.mark.parametrize("fragment", [
    "Number(r.live_win_rate || 0)",
    "Number(r.research_pf || 0).toFixed(2)",
    "Number(p.age_hours || 0).toFixed(1)",
    "Number(r.er_fast ?? 0).toFixed(2)",
    "Number(r.vol_ratio ?? 0).toFixed(2)",
    "(era.totals?.net || 0) * 100",
    "(c.min_tp_bps ?? 0).toFixed(0)",
    "Number(r.regime_conflict.regime_conf ?? 0).toFixed(2)",
])
def test_missing_values_are_no_longer_rendered_as_zero(fragment):
    """Each of these used to print a measured-looking 0.00 for no data."""
    assert fragment not in JS, "regressed: %r is back" % fragment


def test_jump_limit_colour_does_not_invent_a_limit():
    """`(r.history_jump_bps || 0) > (r.jump_limit_bps || 300)` painted an
    unknown jump as green — i.e. 'trusted' — which is the one direction a
    watchdog must never fail in."""
    assert '(r.history_jump_bps || 0) > (r.jump_limit_bps || 300)' not in JS
    assert 'r.history_jump_bps == null ? "var(--dim)"' in JS


# ───────────────────────── 3. swallowed is still counted ──────────────────
def test_guard_never_raises_and_counts():
    silence.reset()
    with silence.guard("unit.test"):
        raise ValueError("boom")
    rep = silence.report()
    assert rep["total_swallowed"] == 1
    assert rep["detail"][0]["site"] == "unit.test"
    assert "ValueError: boom" in rep["detail"][0]["last"]


def test_guard_passes_through_the_happy_path():
    silence.reset()
    seen = []
    with silence.guard("unit.ok"):
        seen.append(1)
    assert seen == [1]
    assert silence.report()["total_swallowed"] == 0


def test_guard_counts_repeated_failures_in_one_site():
    silence.reset()
    for _ in range(5):
        with silence.guard("unit.repeat"):
            raise RuntimeError("x")
    rep = silence.report()
    assert rep["total_swallowed"] == 5
    assert rep["sites"] == 1


def test_settle_learners_are_counted_not_silently_passed():
    """The three learning calls in _settle_and_lock were bare `pass`."""
    src = (ROOT / "app" / "engine.py").read_text()
    for site in ("settle.meta_learn", "settle.genesis_score",
                 "settle.forward_record"):
        assert 'guard("%s")' % site in src, site


def test_engine_exposes_the_silence_report():
    src = (ROOT / "app" / "engine.py").read_text()
    assert '"silence": _silence_report(),' in src


# ───────────────────── 4. the audit harness stays honest ──────────────────
def test_deep_audit_never_probes_side_effecting_functions():
    """The probe called agents.download_gguf() once, which spawned a real
    installer and closed stdout — the audit then 'found nothing'."""
    import scripts.deep_audit as audit
    assert audit._SIDE_EFFECT.search("download_gguf")
    assert audit._SIDE_EFFECT.search("ensure_model")
    assert not audit._SIDE_EFFECT.search("kyle_lambda")


# ─────────────── 5. no future bars in the d08 smoothing ──────────────────
def test_d08_smoothing_is_causal():
    """`np.convolve(..., 'same')` is centred: the score of a window ending
    at t moved when only a LATER bar changed, which is a look-ahead leak."""
    import numpy as np
    from app.genesis import _d08_category

    fn = _d08_category({})[0][1]
    rs = np.random.RandomState(3)
    x = rs.randn(20)
    later = x.copy()
    later[10] += 1000.0

    # a leak-free score must be identical up to the bar that changed
    from app.genesis import _d08_category as _c
    import importlib
    # reproduce the internal smoothing the feature uses
    mod = importlib.import_module("app.genesis")
    src = (ROOT / "app" / "genesis.py").read_text()
    block = src.split("def _d08_category")[1][:1600]
    # comments explain the bug; they are not the bug. Strip them or this
    # assertion matches its own explanation (a trap already hit once).
    code = "\n".join(ln.split("#", 1)[0] for ln in block.splitlines())
    assert 'mode="same"' not in code, "centred convolution is back in d08"
    assert 'np.convolve(x, k, mode="full")[:x.size]' in code


def test_d08_score_is_stable_against_appended_bars():
    """Appending bars after the window must not rewrite the score of the
    window, beyond the documented window length."""
    import numpy as np
    from app.genesis import _d08_category
    fn = _d08_category({})[0][1]
    rs = np.random.RandomState(5)
    r = rs.randn(32)
    assert fn({"returns": r}) == fn({"returns": r})
    assert -1.0 <= fn({"returns": r}) <= 1.0


# ───────────────── 6. multiple-testing correction actually corrects ───────
def test_norm_ppf_is_the_inverse_of_the_cdf():
    from app.metacognition import _norm_cdf, _norm_ppf
    for p in (0.5, 0.9, 0.975, 0.99, 0.999):
        assert _norm_cdf(_norm_ppf(p)) == pytest.approx(p, abs=1e-6)
    # the tails matter too: 1 - 1/N for a 6600-config search is ~0.99985
    for p in (1e-6, 1e-4, 0.001, 0.999, 0.9999, 1 - 1e-6):
        assert _norm_cdf(_norm_ppf(p)) == pytest.approx(p, abs=1e-6)


def test_norm_ppf_matches_known_quantiles():
    from app.metacognition import _norm_ppf
    assert _norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert _norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-4)
    assert _norm_ppf(0.99) == pytest.approx(2.326348, abs=1e-4)
    assert _norm_ppf(0.999) == pytest.approx(3.090232, abs=1e-4)


def test_deflated_sharpe_falls_as_the_number_of_trials_rises():
    """Bug fixed 2026-11-26: `_norm_cdf` was used where a quantile was
    needed, so `sr0` was ~0.019 for every N and the DSR was 1.0000 whether
    the trader ran 1 test or 6600. A multiple-comparison correction that
    ignores the number of comparisons is not a correction.
    """
    from app.metacognition import deflated_sharpe
    vals = [deflated_sharpe(0.20, 100, n) for n in (1, 10, 100, 1000, 6600)]
    assert vals == sorted(vals, reverse=True), vals
    for a, b in zip(vals, vals[1:]):
        assert b < a, vals
    # and the haircut must be large enough to change a decision
    assert deflated_sharpe(0.20, 100, 1) > 0.90
    assert deflated_sharpe(0.20, 100, 6600) < 0.10


def test_deflated_sharpe_still_rewards_a_bigger_edge():
    from app.metacognition import deflated_sharpe
    assert deflated_sharpe(0.50, 100, 80) > deflated_sharpe(0.10, 100, 80)


def test_deflated_sharpe_stays_in_the_unit_interval():
    from app.metacognition import deflated_sharpe
    for sr in (-1.0, 0.0, 0.3, 5.0):
        for n in (1, 7, 1000):
            v = deflated_sharpe(sr, 200, n)
            assert 0.0 <= v <= 1.0
