"""Centralized synchronized clock + per-component countdown timers (§1.4).

Every subsystem pass inside one cycle is stamped with the SAME
``snapshot_ts`` so Formulas, Brain, Emotion, Physics, News, Agents and
Charts all describe the identical market instant. The UI header reads
``SyncClock.countdowns()`` for the live schedule:

  * ``NEXT_TICK_PASS``           1.0 s cadence
  * ``WINDOW_CLOSE_SETTLE``      exact MM:SS to the focused TF boundary
  * ``BRAIN_METACOG_REWEIGHT``   online weight/calibration update (GRC)
  * ``AGENT_COMMITTEE_SYNTHESIS`` multi-agent consensus audit
  * ``NEWS_MACRO_POLL``          headline ingestion cycle (60 s)

Anchoring: exchange UTC when the feed reports delivery timestamps,
local UTC otherwise — never the browser clock. Countdowns are computed
from the anchor, so a delayed pass shows a negative remainder instead
of silently drifting the schedule.
"""
from __future__ import annotations

import time
from typing import Any

from .config import GRC_INTERVAL_SECONDS, NEWS_POLL_SECONDS, TIMEFRAMES

TICK_PERIOD_S = 1.0
METACOG_PERIOD_S = GRC_INTERVAL_SECONDS          # reweight at every GRC
COMMITTEE_PERIOD_S = 3 * GRC_INTERVAL_SECONDS    # synthesis every 3rd GRC


class SyncClock:
    """One clock, one snapshot instant, five visible countdowns."""

    def __init__(self) -> None:
        self._snapshot_ts: float = 0.0
        self._anchor_src: str = "local-utc"
        self._last_pass: dict[str, float] = {}   # component -> last fire ts
        self._passes: int = 0

    # ── anchoring ────────────────────────────────────────────────────
    def server_now(self, feed=None) -> float:
        """Exchange-UTC now when the feed has a recent delivery, else local UTC."""
        if feed is not None:
            try:
                age = time.time() - float(getattr(feed, "last_delivery", 0.0) or 0.0)
                if 0.0 <= age < 5.0:
                    self._anchor_src = "exchange-utc"
                    return time.time()
            except Exception:
                pass
        self._anchor_src = "local-utc"
        return time.time()

    def begin_pass(self, feed=None) -> float:
        """Stamp the single snapshot_ts every subsystem of this pass shares."""
        self._snapshot_ts = self.server_now(feed)
        self._passes += 1
        return self._snapshot_ts

    @property
    def snapshot_ts(self) -> float:
        return self._snapshot_ts or time.time()

    @property
    def anchor_source(self) -> str:
        return self._anchor_src

    # ── component scheduling ─────────────────────────────────────────
    def note_pass(self, component: str, now: float | None = None) -> None:
        self._last_pass[component] = now if now is not None else time.time()

    def _remaining(self, component: str, period: float,
                   now: float) -> float:
        last = self._last_pass.get(component, 0.0)
        if not last:
            return period                     # cold: full period ahead
        elapsed = now - last
        return period - elapsed

    @staticmethod
    def _mmss(seconds: float) -> str:
        s = max(0, int(round(seconds)))
        return f"{s // 60:02d}:{s % 60:02d}"

    def window_close(self, tf: str, now: float | None = None) -> dict[str, Any]:
        """Exact remaining MM:SS until the next close of this timeframe.

        Windows are aligned to the exchange minute grid: a 22-minute TF
        closes at epoch minutes where ``minute % 22 == 0`` (the tape's
        own bar boundaries), so all four cards countdown to their real
        settlement instants off one clock.
        """
        now = self.snapshot_ts if now is None else now
        minutes = int(TIMEFRAMES.get(tf, {}).get("minutes", 22) or 22)
        epoch_min = now / 60.0
        into = epoch_min % minutes
        remain_min = (minutes - into) if into > 0 else float(minutes)
        remain_s = remain_min * 60.0 - (now % 60.0 if into > 0 else 0.0)
        # recompute cleanly: seconds to next multiple of `minutes`
        period_s = minutes * 60.0
        remain_s = period_s - (now % period_s)
        return {
            "tf": tf,
            "minutes": minutes,
            "remaining_s": round(remain_s, 3),
            "remaining": self._mmss(remain_s),
            "ends_at": now + remain_s,
        }

    def countdowns(self, focus_tf: str = "tf1",
                   now: float | None = None) -> dict[str, Any]:
        now = self.snapshot_ts if now is None else now
        win = self.window_close(focus_tf, now)
        return {
            "snapshot_ts": round(now, 3),
            "anchor": self._anchor_src,
            "passes": self._passes,
            "NEXT_TICK_PASS": {
                "period_s": TICK_PERIOD_S,
                "remaining_s": round(self._remaining(
                    "tick", TICK_PERIOD_S, now), 3),
                "remaining": self._mmss(self._remaining(
                    "tick", TICK_PERIOD_S, now)),
            },
            "WINDOW_CLOSE_SETTLE": {
                **win,
                "label": f"{focus_tf.upper()} {win['minutes']}m",
            },
            "BRAIN_METACOG_REWEIGHT": {
                "period_s": METACOG_PERIOD_S,
                "remaining_s": round(self._remaining(
                    "metacog", METACOG_PERIOD_S, now), 3),
                "remaining": self._mmss(self._remaining(
                    "metacog", METACOG_PERIOD_S, now)),
            },
            "AGENT_COMMITTEE_SYNTHESIS": {
                "period_s": COMMITTEE_PERIOD_S,
                "remaining_s": round(self._remaining(
                    "committee", COMMITTEE_PERIOD_S, now), 3),
                "remaining": self._mmss(self._remaining(
                    "committee", COMMITTEE_PERIOD_S, now)),
            },
            "NEWS_MACRO_POLL": {
                "period_s": NEWS_POLL_SECONDS,
                "remaining_s": round(self._remaining(
                    "news", NEWS_POLL_SECONDS, now), 3),
                "remaining": self._mmss(self._remaining(
                    "news", NEWS_POLL_SECONDS, now)),
            },
        }


# one process-wide clock: the whole engine shares one snapshot instant
CLOCK = SyncClock()
