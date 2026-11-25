"""Exact product bound from a box and rational fleet support inequalities."""

from collections.abc import Iterable
from fractions import Fraction
from math import lcm


def maximize(
    q_cap: int | Fraction,
    p_total: int | Fraction,
    supports: Iterable[tuple[int | Fraction, int | Fraction, int | Fraction]],
    *,
    p_step: int | Fraction | None = None,
    q_step: int | Fraction | None = None,
) -> Fraction:
    """Maximize q*p/p_total over the nonnegative box and every a*q+b*p <= S.

    Each supplied step restricts that coordinate to nonnegative multiples of
    the step. Steps must come from the original model, and an omitted step
    leaves its coordinate continuous. An empty population gives score zero.
    """
    q_cap, p_total = Fraction(q_cap), Fraction(p_total)
    p_step = None if p_step is None else Fraction(p_step)
    q_step = None if q_step is None else Fraction(q_step)
    if q_cap < 0 or p_total < 0:
        raise ValueError("box limits must be nonnegative")
    if any(step is not None and step <= 0 for step in (p_step, q_step)):
        raise ValueError("grid steps must be positive")

    # Enumerate the shorter grid when both coordinates have legal steps.
    x_cap, y_cap = p_total, q_cap
    x_step, y_step = p_step, q_step
    swap = q_step is not None and (p_step is None or q_cap // q_step < p_total // p_step)
    if swap:
        x_cap, y_cap = y_cap, x_cap
        x_step, y_step = y_step, x_step

    # Each line gives y <= intercept - fall*x. Keep the tightest parallel line.
    lines = {Fraction(0): y_cap}
    for a, b, support in supports:
        a, b, support = Fraction(a), Fraction(b), Fraction(support)
        if a <= 0 or b <= 0 or support < 0:
            raise ValueError("support weights must be positive and the limit nonnegative")
        if swap:
            a, b = b, a
        x_cap = min(x_cap, support / b)
        fall, intercept = b / a, support / a
        lines[fall] = min(intercept, lines.get(fall, intercept))

    if not x_cap or not y_cap:
        return Fraction(0)
    if x_step is not None:
        return _grid_product(x_cap, lines, x_step, y_step) / p_total
    return _continuous_product(x_cap, lines) / p_total


def _grid_product(x_cap, lines, x_step, y_step):
    if y_step is not None:
        integers = []
        for fall, intercept in lines.items():
            a, c = fall * x_step / y_step, intercept / y_step
            den = lcm(a.denominator, c.denominator)
            integers.append((int(a * den), int(c * den), den))
        best = 0
        for i in range(1, x_cap // x_step + 1):
            j = min((c - a * i) // den for a, c, den in integers)
            if j <= 0:
                break
            best = max(best, i * j)
        return best * x_step * y_step
    best = Fraction(0)
    for index in range(1, x_cap // x_step + 1):
        x = index * x_step
        y = min(intercept - fall * x for fall, intercept in lines.items())
        best = max(best, x * y)
    return best


def _continuous_product(x_cap, lines):
    best = Fraction(0)
    for fall, intercept in lines.items():
        lo, hi = Fraction(0), x_cap
        # Clip to the interval where this line is the active upper bound.
        for other_fall, other_intercept in lines.items():
            if fall > other_fall:
                lo = max(lo, (intercept - other_intercept) / (fall - other_fall))
            elif fall < other_fall:
                hi = min(hi, (intercept - other_intercept) / (fall - other_fall))
            if lo > hi:
                break
        if lo <= hi:
            x = min(hi, max(lo, intercept / (2 * fall))) if fall else hi
            best = max(best, x * (intercept - fall * x))
    return best
