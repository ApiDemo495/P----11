"""Pure-python statistical helpers shared by all 31 formulas."""
from __future__ import annotations

import math


def sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def std_pop(xs: list[float]) -> float:
    """Population standard deviation (divides by N, matching the spec)."""
    if not xs:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def skewness(xs: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    m = mean(xs)
    m2 = sum((x - m) ** 2 for x in xs) / n
    if m2 <= 1e-30:
        return 0.0
    m3 = sum((x - m) ** 3 for x in xs) / n
    return m3 / (m2 ** 1.5)


def excess_kurtosis(xs: list[float]) -> float:
    n = len(xs)
    if n < 4:
        return 0.0
    m = mean(xs)
    m2 = sum((x - m) ** 2 for x in xs) / n
    if m2 <= 1e-30:
        return 0.0
    m4 = sum((x - m) ** 4 for x in xs) / n
    return m4 / (m2 * m2) - 3.0


def pearson(x: list[float], y: list[float]) -> float:
    n = min(len(x), len(y))
    if n < 3:
        return 0.0
    x, y = x[-n:], y[-n:]
    mx, my = mean(x), mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx <= 1e-30 or sy <= 1e-30:
        return 0.0
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    return max(-1.0, min(1.0, cov / (sx * sy)))


def linear_slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of ys against xs."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    den = sum((a - mx) ** 2 for a in xs)
    if den <= 1e-30:
        return 0.0
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / den


def ema(values: list[float], period: int) -> list[float]:
    """EMA chain: alpha = 2/(period+1), seeded with the first value."""
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1.0 - alpha) * out[-1])
    return out


def pct_returns(prices: list[float]) -> list[float]:
    """r(i) = (P(i) - P(i-1)) / P(i-1) for consecutive samples."""
    out = []
    for i in range(1, len(prices)):
        prev = prices[i - 1]
        out.append((prices[i] - prev) / prev if abs(prev) > 1e-30 else 0.0)
    return out


def abs_changes(prices: list[float]) -> list[float]:
    return [prices[i] - prices[i - 1] for i in range(1, len(prices))]


def safe_tanh(x: float) -> float:
    if x != x or x in (float("inf"), float("-inf")):
        return 1.0 if (x == x and x > 0) else (-1.0 if x == x else 0.0)
    return math.tanh(x)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
