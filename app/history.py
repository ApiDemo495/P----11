"""GRC-cycle history ring buffers.

Every baseline / rolling statistic referenced by the formulas ("last 500 GRC
cycles", "last 100 GRC cycles", ...) lives here.  Entries are appended exactly
once per GRC — never between GRCs — so baselines keep their GRC cadence even
when the engine recomputes predictions on events (Tp2 lock, SL protocol,
rollover, manual override).
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from .config import HIST_REGIME, HIST_REGIME_DURATIONS


class Ring:
    __slots__ = ("maxlen", "_q")

    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self._q: deque[float] = deque(maxlen=maxlen)

    def append(self, v: float) -> None:
        self._q.append(float(v))

    def values(self) -> list[float]:
        return list(self._q)

    def last(self) -> float | None:
        return self._q[-1] if self._q else None

    def __len__(self) -> int:
        return len(self._q)

    def vmax(self) -> float:
        return max((abs(v) for v in self._q), default=0.0)

    def mean(self) -> float:
        if not self._q:
            return 0.0
        return sum(self._q) / len(self._q)

    def std(self) -> float:
        n = len(self._q)
        if n < 2:
            return 0.0
        m = self.mean()
        return math.sqrt(sum((v - m) ** 2 for v in self._q) / n)

    def median(self) -> float:
        if not self._q:
            return 0.0
        s = sorted(self._q)
        n = len(s)
        mid = n // 2
        return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


class HistoryStore:
    """All GRC-cadence histories, keyed by 'symbol:tf:name' or 'symbol:name'."""

    def __init__(self) -> None:
        self._rings: dict[str, Ring] = {}
        # completed durations per (symbol, tf, regime-type) — Formula 20
        self.regime_durations: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        self._run: dict[tuple[str, str], tuple[str, int]] = {}
        # Formula 18 auto-learning: completed directional streaks per symbol
        self.streak_lams: dict[str, deque[tuple[int, float]]] = defaultdict(
            lambda: deque(maxlen=200)
        )
        self.prev_streak: dict[str, tuple[int, int, float]] = {}  # sym -> (sign, N, lambda)
        # Formula 27 auto-learning: volatility clusters + inter-cluster gaps
        self.vol_cluster_durations: dict[str, list[float]] = defaultdict(list)
        self.vol_cluster_gaps: dict[str, list[float]] = defaultdict(list)
        self._vol_last_end_ts: dict[str, float] = {}   # last committed cluster end
        # Emotion E2/E9: SL-hit cycle history per symbol (last 10 events)
        self.sl_hit_events: dict[str, deque[int]] = defaultdict(
            lambda: deque(maxlen=10)
        )
        # Engine-maintained consecutive-correct / 10 ratio per symbol (UI + E-ctx)
        self.win_ratio: dict[str, float] = defaultdict(float)
        self._correct_run: dict[str, int] = defaultdict(int)

    # -- generic ring access ------------------------------------------------
    def ring(self, key: str, maxlen: int) -> Ring:
        r = self._rings.get(key)
        if r is None:
            r = Ring(maxlen)
            self._rings[key] = r
        return r

    # -- convenience key builders ------------------------------------------
    @staticmethod
    def tf_key(symbol: str, tf: str, name: str) -> str:
        return f"{symbol}:{tf}:{name}"

    @staticmethod
    def sym_key(symbol: str, name: str) -> str:
        return f"{symbol}:{name}"

    # -- Formula 20 regime bookkeeping -------------------------------------
    def append_regime(self, symbol: str, tf: str, regime: str) -> None:
        """Append the GRC's regime label and close out finished durations."""
        key = (symbol, tf)
        cur = self._run.get(key)
        if cur is None:
            self._run[key] = (regime, 1)
        elif cur[0] == regime:
            self._run[key] = (regime, cur[1] + 1)
        else:
            dkey = (symbol, tf, cur[0])
            durations = self.regime_durations[dkey]
            durations.append(cur[1])
            if len(durations) > HIST_REGIME_DURATIONS:
                del durations[: len(durations) - HIST_REGIME_DURATIONS]
            self._run[key] = (regime, 1)
        self.ring(self.tf_key(symbol, tf, "regime"), HIST_REGIME).append(
            float(_regime_to_float(regime))
        )

    def regimes(self, symbol: str, tf: str) -> list[str]:
        vals = self.ring(self.tf_key(symbol, tf, "regime"), HIST_REGIME).values()
        return [_float_to_regime(v) for v in vals]

    def regime_ages(self, symbol: str, tf: str) -> tuple[int, int]:
        """Return (age_in_GRC_steps_of_current_run, total_history_len)."""
        labels = self.regimes(symbol, tf)
        if not labels:
            return 0, 0
        cur = labels[-1]
        age = 1
        for lab in reversed(labels[:-1]):
            if lab != cur:
                break
            age += 1
        return age, len(labels)

    # -- Formula 18 streak learning ----------------------------------------
    def commit_streak(self, symbol: str, streak_sign: int, n: int, lam: float) -> None:
        """Called by the engine once per GRC with the live streak observation.

        When the streak direction changes, the *previous* (ended) streak's
        lambda is committed to history — those completed streaks are what
        lambda_critical is averaged from.
        """
        prev = self.prev_streak.get(symbol)
        if streak_sign == 0 or n < 3:
            if prev is not None:
                self.streak_lams[symbol].append((prev[1], prev[2]))
                self.prev_streak.pop(symbol, None)
            return
        if prev is not None and prev[0] != streak_sign:
            self.streak_lams[symbol].append((prev[1], prev[2]))
        self.prev_streak[symbol] = (streak_sign, n, lam)

    def lambda_critical(self, symbol: str, n: int) -> float | None:
        """Mean lambda of the last HIST_CCMDR_WINDOW completed streaks with
        length within +/-2 of n.  None when no historical streak is available
        (formula defaults to 0.15)."""
        entries = self.streak_lams.get(symbol)
        if not entries:
            return None
        match = [lam for (len_, lam) in entries if abs(len_ - n) <= 2]
        if not match:
            return None
        window = match[-50:]
        return sum(window) / len(window)

    # -- Formula 27 volatility-cluster learning -----------------------------
    def commit_vol_cluster(self, symbol: str, info: dict) -> None:
        """Commit clusters that ENDED since the previous GRC (once each)."""
        last_ts = self._vol_last_end_ts.get(symbol, 0.0)
        durations = self.vol_cluster_durations[symbol]
        gaps = self.vol_cluster_gaps[symbol]
        ended = info.get("ended") or []
        # ended entries arrive oldest-first; commit strictly after last_ts
        for (end_ts, duration, _mean_r) in ended:
            if end_ts > last_ts:
                durations.append(float(duration))
                if len(durations) > HIST_REGIME_DURATIONS:
                    del durations[: len(durations) - HIST_REGIME_DURATIONS]
                self._vol_last_end_ts[symbol] = end_ts
                last_ts = end_ts
        # gap between the last committed cluster and now
        if last_ts and info.get("last_end_ts") == last_ts and not info.get("in_cluster"):
            gap_min = float(info.get("gap") or 0)
            if gap_min > 0:
                if not gaps or gaps[-1] != gap_min:
                    gaps.append(gap_min)
                    if len(gaps) > HIST_REGIME_DURATIONS:
                        del gaps[: len(gaps) - HIST_REGIME_DURATIONS]

    # -- Emotion E2 / E9 SL-hit cycles + win-streak ratio --------------------
    def record_sl_hit(self, symbol: str) -> None:
        self.sl_hit_events[symbol].append(1)

    def record_no_sl_hit(self, symbol: str) -> None:
        """Called when a cycle closes without an SL hit (keeps cycles aligned)."""
        self.sl_hit_events[symbol].append(0)

    def p_hit(self, symbol: str) -> float:
        """E2: fraction of SL hits over the last 10 completed cycles."""
        ev = self.sl_hit_events.get(symbol)
        if not ev:
            return 0.0
        return sum(ev) / len(ev)

    def sl_hit_ratio_recent(self, symbol: str) -> float:
        """E9: hit ratio over the last 5 completed cycles (hit = trauma)."""
        ev = self.sl_hit_events.get(symbol)
        if not ev:
            return 0.0
        last5 = list(ev)[-5:]
        return sum(last5) / len(last5)

    def commit_win_streak(self, symbol: str, correct: bool) -> None:
        """Consecutive-correct running ratio: run length capped and /10."""
        if correct:
            self._correct_run[symbol] = min(10, self._correct_run[symbol] + 1)
        else:
            self._correct_run[symbol] = 0
        self.win_ratio[symbol] = self._correct_run[symbol] / 10.0


_REGIME_MAP = {
    "volatile_trending": 1.0,
    "trending": 2.0,
    "transitional": 3.0,
    "volatile_ranging": 4.0,
    "calm_ranging": 5.0,
}
_FLOAT_REGIME = {v: k for k, v in _REGIME_MAP.items()}


def _regime_to_float(r: str) -> float:
    return _REGIME_MAP.get(r, 3.0)


def _float_to_regime(v: float) -> str:
    return _FLOAT_REGIME.get(round(v), "transitional")
