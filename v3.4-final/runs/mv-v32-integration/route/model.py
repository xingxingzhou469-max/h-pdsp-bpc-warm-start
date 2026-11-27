"""Exact input and conservative projections for the route-pricing bound."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import accumulate
from math import gcd, lcm


def rational(value):
    if type(value) is float:
        return Fraction(str(value))
    return Fraction(value)


def read_model(instance):
    supply, people, service = (
        tuple(map(rational, instance[name])) for name in ("supply", "population", "service")
    )
    travel = tuple(tuple(map(rational, row)) for row in instance["travel"])
    capacity, limit = map(rational, (instance["capacity"], instance["limit"]))
    size = len(supply)
    if not size or len(people) != size or len(service) != size or len(travel) != size:
        raise ValueError("inconsistent instance dimensions")
    if any(len(row) != size for row in travel):
        raise ValueError("inconsistent travel dimensions")
    if min(capacity, limit, *supply, *people, *service, *(min(row) for row in travel)) < 0:
        raise ValueError("negative model input")
    if supply[0] or people[0] or any(a and b for a, b in zip(supply, people)):
        raise ValueError("depot demand or supply, or mixed station")
    n = size - 1
    eligible_stock = tuple(min(s, capacity) for s in supply[1:])
    people = people[1:]
    increments = tuple(tuple(travel[i][j] + service[j] for j in range(1, size))
                       for i in range(size))
    home = tuple(travel[i][0] for i in range(1, size))
    out_cost = tuple(
        service[i] + min(travel[i][j] for j in range(size) if j != i)
        for i in range(1, size)
    )
    in_cost = tuple(
        service[i] + min(travel[j][i] for j in range(size) if j != i)
        for i in range(1, size)
    )
    return Model(n, supply[1:], eligible_stock, people, capacity, limit,
                 increments, home, out_cost, in_cost)


@dataclass(frozen=True)
class Model:
    n: int
    supply: tuple
    stock: tuple
    people: tuple
    capacity: Fraction
    limit: Fraction
    increments: tuple
    home: tuple
    out_cost: tuple
    in_cost: tuple

    @property
    def population(self):
        return sum(self.people, Fraction())

    @property
    def quantity(self):
        return min(sum(self.stock, Fraction()), self.capacity * sum(p > 0 for p in self.people))

    @property
    def reward_unit(self):
        return self.quantity / (1 << 22) if self.quantity else Fraction(1)

    def weights(self, beta):
        unit = self.reward_unit
        return tuple(-(-(s + beta * p) // unit)
                     for s, p in zip(self.stock, self.people))

    def boxes(self, vehicles):
        time = self.limit * vehicles
        max_visits = min(
            sum(length <= time for length in accumulate(sorted(costs)))
            for costs in (self.in_cost, self.out_cost)
        )
        stocks = [Fraction(), *accumulate(sorted((s for s in self.stock if s), reverse=True))]
        for d, pop in enumerate(accumulate(sorted((p for p in self.people if p), reverse=True)), 1):
            n_sources = min(len(stocks) - 1, max_visits - d)
            if n_sources <= 0:
                break
            yield min(stocks[n_sources], self.capacity * d), pop

    def project(self):
        # Prefix stations preceding the first supplier and suffix stations
        # following the last agency can still be useful on nonmetric inputs.
        start = list(self.increments[0])
        finish = list(self.home)
        for distance, gate, backward in ((start, self.supply, False), (finish, self.people, True)):
            remaining = {i for i in range(self.n) if not gate[i]}
            while remaining:
                j = min(remaining, key=distance.__getitem__)
                remaining.remove(j)
                for i in range(self.n):
                    arc = self.increments[i + 1][j] if backward else self.increments[j + 1][i]
                    distance[i] = min(distance[i], distance[j] + arc)
        internal = self.increments[1:]
        times = (self.limit, *start, *finish,
                 *(d for i, row in enumerate(internal) for j, d in enumerate(row) if i != j))
        denominator = lcm(*(time.denominator for time in times))
        unit = Fraction(gcd(*(int(time * denominator) for time in times)), denominator)
        if not unit:
            unit = Fraction(1)
        if self.limit // unit > 640:
            unit = self.limit / 640
        horizon = int(self.limit // unit)
        beyond = horizon + 1
        ticks = lambda cost: min(beyond, cost // unit)
        head = tuple(ticks(cost) if self.supply[i] else beyond for i, cost in enumerate(start))
        tail = tuple(ticks(cost) if self.people[i] else beyond for i, cost in enumerate(finish))
        arcs = tuple(tuple(ticks(d) if i != j else beyond for j, d in enumerate(row))
                     for i, row in enumerate(internal))
        return Projection(horizon, arcs, head, tail)


@dataclass(frozen=True)
class Projection:
    horizon: int
    arcs: tuple
    start: tuple
    finish: tuple

    @property
    def layered(self):
        return any(0 in row for row in self.arcs)

    def reversed(self):
        return Projection(self.horizon, tuple(zip(*self.arcs)), self.finish, self.start)


def product_bound(model, boxes, supports):
    """Exact upper envelope and one maximizing point, using AM–GM on each face."""
    if not model.population or not model.quantity:
        return Fraction(), (Fraction(), Fraction())
    best = Fraction()
    point = (Fraction(), Fraction())
    for q_cap, p_cap in boxes:
        faces = [(Fraction(), q_cap), *supports]
        for slope, intercept in faces:
            low, high = Fraction(), p_cap
            if slope:
                high = min(high, intercept / slope)
            for other_slope, other_rhs in faces:
                delta, right = other_slope - slope, other_rhs - intercept
                if delta > 0:
                    high = min(high, right / delta)
                elif delta < 0:
                    low = max(low, right / delta)
                elif right < 0:
                    high = -1
                    break
            if low > high:
                continue
            p = min(high, max(low, intercept / (2 * slope))) if slope else high
            q = intercept - slope * p
            product = q * p
            if product > best:
                best, point = product, (q, p)
    return best / model.population, point
