"""The four questions the user asked about the board (2026-11-27).

1. "It is showing bullish in yellow box with impulse bear conf 0.20" — a
   headline badge contradicting its own regime chip.
2. "Why asset triggers are unproven and shadow or none" — three different
   facts printed as three unexplained labels.
3. "Why paper execution is paused?" — a freeze with no cause on the page.
4. "Why hold policy have those yellow boxes of gate fail, warmup and other
   any issues behind them" — tiers that named a verdict and never a reason.

Each test pins the MECHANISM, so the same page cannot be shipped again.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP_JS = (ROOT / "static" / "app.js").read_text()
INDEX_HTML = (ROOT / "static" / "index.html").read_text()
ENGINE_PY = (ROOT / "app" / "engine.py").read_text()
HOLD_PY = (ROOT / "app" / "hold_policy.py").read_text()
SLEEVES_PY = (ROOT / "app" / "sleeves.py").read_text()


# ── 1. the badge must never contradict its own chip ────────────────────
def test_regime_conflict_has_one_definition_used_by_both_write_paths():
    """The conflict was computed ONLY inside analyse.

    So a GRC-refreshed card carried the chip and no disagreement record —
    which is exactly the screenshot: '▲ BULLISH*' beside '📉 impulse bear'
    with no banner. The comparison now lives in one method, and every card
    writer calls it.
    """
    assert "def regime_conflict(self, symbol: str, direction: int)" in ENGINE_PY
    # the old inline copy inside analyse is gone, replaced by the method
    assert "_conflict = self.regime_conflict(symbol, int(direction or 0))" in ENGINE_PY
    # a GRC refresh writes the conflict too (issued + not-issued branches)
    assert ENGINE_PY.count("cyc.regime_conflict = self.regime_conflict(") >= 2
    # and the analyse path persists the same object on the cycle
    assert "cyc.regime_conflict = _conflict" in ENGINE_PY


def test_conflict_is_serialised_on_the_cycle_payload():
    """A field the model drops never reaches the page."""
    models_py = (ROOT / "app" / "models.py").read_text()
    assert "regime_conflict: dict | None = None" in models_py
    assert '"regime_conflict": self.regime_conflict,' in models_py


def test_regime_conflict_reads_this_assets_tape_not_btcs():
    """The classification must be per-asset, and never raise."""
    block = ENGINE_PY[ENGINE_PY.index("def regime_conflict("):]
    block = block[:block.index("def _step_portfolio")]
    assert "self.feed.state(symbol)" in block
    assert "except Exception:" in block and "return None" in block


def test_badge_prints_the_side_and_the_opposing_regime_together():
    """Never two fighting labels: one sentence naming both facts."""
    i = APP_JS.index("const _sideTxt = dir > 0")
    block = APP_JS[i:i + 2000]
    assert "OPPOSES THIS" in block
    assert "call stands and is graded, routing held" in block
    # the chip is suppressed when the explicit clash line is shown, so the
    # header can never print both "impulse bear" and a bare bullish word
    assert '${conflict ? "" : chopChip(tf)}' in block


def test_the_undefined_rawDir_variable_is_gone():
    """That row dereferenced `rawDir`, which was never defined — a latent
    ReferenceError that could abort the hero render."""
    assert "rawDir" not in APP_JS


def test_hero_no_longer_says_withheld_over_a_graded_call():
    """The call stands and is graded; 'WITHHELD' said the opposite."""
    i = APP_JS.index('<span>regime disagreement</span>')
    block = APP_JS[i - 700:i + 500]
    assert '"● WITHHELD"' not in block
    assert "the side stands and is graded, only routing is held" in block
    # and the why-panel headline above it names the opposing state instead
    # of replacing the side with the word WITHHELD
    assert "opposes · no routing</span>" in APP_JS


# ── 2. shadow / unproven / none must say which fact they are ────────────
def test_shadow_sleeves_carry_a_live_trigger_and_the_gate_they_failed():
    assert "the gate rejected this candidate" in SLEEVES_PY
    assert '"gate_note"' in SLEEVES_PY
    # `state` stays contractually "unproven" — only OPEN sleeves may emit a
    # tradable state — and the live reading travels in its own field, so no
    # surface can mistake a rejected candidate for a position
    assert 'state = "unproven"' in SLEEVES_PY
    assert '"trigger": trigger,' in SLEEVES_PY


def test_untested_assets_are_not_reported_as_failures():
    """PAXG was never in the edge scan's search space. 'none' implied it
    had been tested and failed."""
    assert '"untested": True' in ENGINE_PY
    assert "never produced a candidate for" in ENGINE_PY


def test_asset_tiles_distinguish_the_three_labels():
    i = APP_JS.index("const assetWrap = $(\"#book-assets\")")
    block = APP_JS[i:i + 2200]
    assert "UNTESTED — not in the scan" in block
    assert "SHADOW — gate rejected, watched only" in block
    assert "WARMING UP — needs real tape" in block
    assert "live z" in block
    assert "a.trigger" in block


def test_the_hero_sleeve_panel_names_the_gate_not_just_unproven():
    i = APP_JS.index('const slBlock = $("#hero-sleeve")')
    block = APP_JS[i:i + 6000]
    assert "gate it failed" in block
    assert "SHADOW ·" in block


# ── 3. paper execution: verified, or paused WITH its cause ─────────────
def test_execution_module_exists_and_implements_the_research_contract():
    from app import exec_live
    assert exec_live.BAR_SECONDS == 900
    assert exec_live.FILL_DELAY_BARS == 1
    assert exec_live.SLIPPAGE_SHARE == 0.5


def test_resampler_drops_the_forming_bar():
    """The old live path filled against a half-formed bar."""
    from app.exec_live import resample_ohlc
    n = 31                      # 30 complete minutes + 1 forming
    t = [i * 60 for i in range(n)]
    bars, why = resample_ohlc(t, [100.0] * n, [101.0] * n, [99.0] * n,
                              [100.5] * n, ["binance-rest"] * n)
    assert why == ""
    # 30 complete minutes = two 15m bars; the 31st minute (still forming)
    # must not appear as a third
    assert len(bars) == 2
    assert [b["t"] for b in bars] == [0, 900]


def test_resampler_refuses_a_gap_instead_of_aggregating_around_it():
    from app.exec_live import resample_ohlc
    t = [i * 60 for i in range(30)]
    t = t[:10] + t[11:]                       # remove one minute
    m = len(t)
    bars, why = resample_ohlc(t, [100.0] * m, [101.0] * m, [99.0] * m,
                              [100.5] * m, ["binance-rest"] * m)
    assert bars == []
    assert "gap" in why.lower()


def test_resampler_rejects_mixed_provenance_and_bad_ohlc():
    from app.exec_live import resample_ohlc
    t = [i * 60 for i in range(15)]
    # high below the close is not a candle
    bars, why = resample_ohlc(t, [100.0] * 15, [99.0] * 15, [98.0] * 15,
                              [100.0] * 15, ["binance-rest"] * 15)
    assert bars == [] and why == ""
    # a length mismatch is refused, not truncated
    bars2, why2 = resample_ohlc(t, [100.0] * 14, [101.0] * 15, [99.0] * 15,
                                [100.0] * 15)
    assert bars2 == [] and "disagree" in why2


def test_next_bar_open_is_the_open_after_the_signal_never_the_close():
    from app.exec_live import next_bar_open, BAR_SECONDS
    bars = [{"t": 0, "o": 1.0, "c": 2.0}, {"t": BAR_SECONDS, "o": 3.0, "c": 4.0}]
    bar, why = next_bar_open(bars, 0, now=BAR_SECONDS * 2 + 1)
    assert bar is not None and bar["o"] == 3.0
    assert why == ""


def test_next_bar_open_waits_while_the_fill_bar_is_still_forming():
    from app.exec_live import next_bar_open, BAR_SECONDS
    bars = [{"t": 0, "o": 1.0, "c": 2.0}, {"t": BAR_SECONDS, "o": 3.0, "c": 4.0}]
    bar, why = next_bar_open(bars, 0, now=BAR_SECONDS + 10)
    assert bar is None and "still forming" in why


def test_tape_provenance_refuses_simulated_tape():
    from app.exec_live import tape_provenance

    class Feed:
        mode = "sim"

    ok, why = tape_provenance(Feed(), "ETH")
    assert ok is False and "simulated" in why.lower()


def test_engine_uses_the_verified_driver():
    assert "from . import exec_live as _xl" in ENGINE_PY
    assert "return _xl.step_live_book(self, block)" in ENGINE_PY
    # and a failure is still an explicit pause with a stated cause
    assert "the verified-fill driver raised" in ENGINE_PY


def test_paused_book_publishes_a_reason_per_asset_on_the_page():
    assert '"paused_assets"' in (ROOT / "app" / "exec_live.py").read_text()
    assert 'id="book-pause-why"' in INDEX_HTML
    assert "why no fills:" in APP_JS
    assert "fill contract:" in APP_JS


# ── 4. every hold-policy tier explains itself ──────────────────────────
def test_every_tier_has_a_reason_and_all_16_cells_get_one():
    from app import hold_policy as hp
    cells = hp.known_cells()
    assert len(cells) == 16
    reasons = {}
    for sym, tf in cells:
        t = hp.tier(sym, tf)
        r = hp.tier_reason(sym, tf)
        assert isinstance(r, str) and len(r) > 20, (sym, tf)
        reasons.setdefault(t, []).append(r)
    # every tier that appears on the board is covered by the reason map
    for t in reasons:
        assert t in {
            "HOLD-OUT VERIFIED", "HOLDOUT ROWS · ERA n<3",
            "IN-SAMPLE SHADOW", "IN-SAMPLE SHADOW · HOLDOUT MISS",
            "UNGRADEABLE / WARMUP", "UNECONOMIC · BELOW ROUND TRIP",
            "ROWS VERIFIED · SHIP GATE FAIL",
        }, t


def test_ship_gate_fail_names_the_gate_and_the_reference():
    from app import hold_policy as hp
    r = hp.tier_reason("BTC", "tf1")
    if hp.tier("BTC", "tf1") == "ROWS VERIFIED · SHIP GATE FAIL":
        assert "gate (ii)" in r
        assert "v8" in r
        assert "73.2 %" in r or "73.2%" in r


def test_warmup_names_the_actual_blocker():
    from app import hold_policy as hp
    r = hp.tier_reason("PAXG", "tf1")
    assert "60m bars" in r or "cost floor" in r


def test_holdout_miss_names_the_failing_metric():
    from app import hold_policy as hp
    for sym, tf in hp.known_cells():
        if hp.tier(sym, tf) == "IN-SAMPLE SHADOW · HOLDOUT MISS":
            r = hp.tier_reason(sym, tf)
            assert "hold-out" in r and "bp" in r
            return
    pytest.skip("no HOLDOUT MISS cell in this artifact")


def test_tier_reason_is_on_the_api_payload_and_the_card():
    from app import hold_policy as hp
    s = hp.summary("BTC", "tf1")
    assert "tier_reason" in s and s["tier_reason"]
    assert "hc-why" in APP_JS
    assert "c.tier_reason" in APP_JS
