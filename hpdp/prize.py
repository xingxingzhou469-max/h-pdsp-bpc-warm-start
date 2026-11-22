"""A shared directed-time-budget upper bound for route-prefix completions."""

from bisect import bisect_right
from fractions import Fraction
from math import ceil


def _curve(items, budget, base=0):
    """Breakpoints of a fractional knapsack value as a function of budget."""
    value = base + sum(prize for cost, prize in items if cost == 0)
    points = [(0, value)]
    spent = 0
    priced = sorted(((cost, prize) for cost, prize in items if cost > 0),
                    key=lambda item: Fraction(item[1], item[0]), reverse=True)
    for cost, prize in priced:
        if spent + cost >= budget:
            if spent < budget:
                points.append((budget, value + Fraction((budget - spent) * prize, cost)))
            return points
        spent += cost
        value += prize
        points.append((spent, value))
    if spent < budget:
        points.append((budget, value))
    return points


def _reverse(points, budget):
    return [(budget - t, value) for t, value in reversed(points)]


def _interpolate(points, times, t):
    right = bisect_right(times, t)
    if t == points[right - 1][0] or right == len(points):
        return points[right - 1][1]
    a, y0 = points[right - 1]
    b, y1 = points[right]
    return y0 + Fraction(t - a, b - a) * (y1 - y0)


def _product_bound(supply, people, delivery_capacity):
    """Maximize min(supply(t), delivery_capacity(t))*people(t), exactly."""
    curves = (supply, people, delivery_capacity)
    times = [[t for t, _ in points] for points in curves]
    best, previous = 0, None
    for t in sorted(set(times[0]) | set(times[1]) | set(times[2])):
        f, q, cap = (_interpolate(points, knots, t) for points, knots in zip(curves, times))
        best = max(best, min(f, cap) * q)
        if previous is not None:
            f0, q0, cap0 = previous
            # On this interval the supply increase and population decrease are
            # affine. After supply crosses delivery capacity both factors fall.
            df, dq = f - f0, q - q0
            stop = Fraction(cap0 - f0, df + cap0 - cap) if f > cap else 1
            if df > 0 and dq < 0:
                vertex = Fraction(df * q0 + f0 * dq, -2 * df * dq)
                if 0 < vertex < stop:
                    best = max(best, (f0 + df * vertex) * (q0 + dq * vertex))
            if f > cap:
                best = max(best, (f0 + df * stop) * (q0 + dq * stop))
        if f >= cap:
            break
        previous = f, q, cap
    return best


def _budget_bound(budget, donors, agencies, initial, unloaded, covered, capacity):
    if budget < 0:
        return 0
    if not agencies:
        return unloaded * covered
    if budget >= sum(cost for cost, _ in donors) + sum(cost for cost, _ in agencies):
        delivered = min(initial + sum(amount for _, amount in donors), unloaded + capacity * len(agencies))
        return delivered * (covered + sum(people for _, people in agencies))
    people = _curve(agencies, budget, covered)
    cap = _curve([(cost, capacity) for cost, _ in agencies], budget, unloaded)
    if not donors:
        return min(initial, cap[-1][1]) * people[-1][1]
    supply = _curve(donors, budget, initial)
    return _product_bound(supply, _reverse(people, budget), _reverse(cap, budget))


def prize_bound(r, label):
    """Return an integer upper numerator, with the same denominator as prefix_bound.

    The caller must have invoked r.prepare_bounds(). A zero value bounds all
    completions, including when the prefix is infeasible, without classifying it.
    """
    remaining = r.limit - label.elapsed
    last = label.last
    if remaining < r.dist[last][0]:
        return 0
    eligible = [v for v in r.nodes if not label.mask & (1 << v)
                and r.dist[last][v] + r.dist[v][0] <= remaining]
    donors = [v for v in eligible if r.supply[v]]
    if not label.route and len(donors) == 1:
        # Every legal tour starts at a pickup. Charge that arc and service
        # exactly when the first stop is forced.
        return prize_bound(r, label.extend(r, donors[0]))
    agencies = [v for v in eligible if r.pop[v]]
    covered = sum(r.pop[v] for v in label.route)
    base = label.unloaded + label.onboard
    best = r.prior
    for outgoing, incoming, cost in r.charges:
        upper = _budget_bound(remaining - outgoing[last] - incoming[0],
                              [(cost[v], r.supply[v]) for v in donors],
                              [(cost[v], r.pop[v]) for v in agencies],
                              base, label.unloaded, covered, r.capacity)
        best = min(best, upper)
    return ceil(best)
