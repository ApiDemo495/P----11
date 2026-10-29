"""Small pearson helper kept separate to avoid a circular import."""
from __future__ import annotations

from .formulas._helpers import pearson


def pearson_or_default(x: list[float], y: list[float]) -> float:
    return abs(pearson(x, y))
