"""Anti-look-ahead guard.

A predictor never receives the bar list. It receives a SliceView
built by DataSlice.at(t), which STRUCTURALLY contains only
bars[:t+1] (a fresh list — there is no reference to the future to
peek at, and no index arithmetic that could off-by-one into it).
"""
from __future__ import annotations

from .data import Bar


class SliceView:
    __slots__ = ("bars", "t")

    def __init__(self, bars: list[Bar], t: int):
        self.bars = bars          # exactly bars[:t+1] of the full series
        self.t = t

    def __len__(self) -> int:
        return len(self.bars)

    @property
    def closes(self) -> list[float]:
        return [b.c for b in self.bars]

    @property
    def highs(self) -> list[float]:
        return [b.h for b in self.bars]

    @property
    def lows(self) -> list[float]:
        return [b.l for b in self.bars]

    @property
    def last(self) -> Bar:
        return self.bars[-1]


class DataSlice:
    def __init__(self, bars: list[Bar]):
        if not bars:
            raise ValueError("empty series")
        self._bars = list(bars)

    def __len__(self) -> int:
        return len(self._bars)

    def at(self, t: int) -> SliceView:
        if not 0 <= t < len(self._bars):
            raise IndexError(t)
        return SliceView(self._bars[:t + 1], t)   # the ONLY way in

    def future(self, t: int, end: int) -> list[Bar]:
        """Settlement-side access (runner/replay only — predictors
        never see this object)."""
        return self._bars[t + 1:end]
