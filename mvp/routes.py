"""Route enumeration for small multi-vehicle instances.

A route is a tuple of distinct non-depot stops.  It is *structurally feasible*
when it starts at a supplier, ends at an agency, and its depot-to-depot
travel + service time fits the route limit.  Quantity feasibility depends on
the allocation and is checked by :mod:`mvp.fleet` when a column is built.
"""

import time
from numbers import Integral

from .fleet import number


def route_time(instance, route):
    travel, service = instance["travel"], instance["service"]
    total = sum(number(travel[i][j]) for i, j in zip((0, *route), (*route, 0)))
    total += sum(number(service[i]) for i in route)
    return total


def route_is_structurally_feasible(instance, route):
    if len(route) < 2 or len(set(route)) != len(route):
        return False
    supply, population = instance["supply"], instance["population"]
    if not all(isinstance(node, Integral) and not isinstance(node, bool)
               and 0 < node < len(supply) for node in route):
        return False
    if number(supply[route[0]]) <= 0 or number(population[route[-1]]) <= 0:
        return False
    if any((number(supply[n]) > 0) == (number(population[n]) > 0) for n in route):
        return False
    return route_time(instance, route) <= number(instance["limit"])


def return_bounds(instance):
    """Shortest completion time from each node to the depot, last stop agency.

    Travel can be directed and non-metric, so ``travel[node][0]`` is *not* a
    valid lower bound on the remaining time of a partial route.  This function
    relaxes the elementary constraint and is therefore a valid lower bound.
    ``service[j]`` is counted when the route continues through ``j``.
    """
    supply, population = instance["supply"], instance["population"]
    travel, service = instance["travel"], instance["service"]
    size = len(supply)
    active = [i for i in range(1, size)
              if number(supply[i]) > 0 or number(population[i]) > 0]
    agencies = [i for i in range(1, size) if number(population[i]) > 0]
    backward = {i: number(travel[i][0]) if i in agencies else None for i in active}
    for _ in range(len(active)):
        changed = False
        for i in active:
            best = backward[i]
            for j in active:
                if j == i or backward[j] is None:
                    continue
                value = number(travel[i][j]) + number(service[j]) + backward[j]
                if best is None or value < best:
                    best = value
            if best != backward[i]:
                backward[i] = best
                changed = True
        if not changed:
            break

    return backward


def enumerate_routes(instance, *, max_stops=None, deadline=None, max_routes=None):
    """Yield structurally feasible routes of increasing DFS order.

    ``max_stops`` caps the number of non-depot stops.  ``deadline`` is an
    absolute ``time.monotonic()`` value.  Routes are emitted whenever the
    current last stop is an agency; the DFS continues while the partial path
    can still reach an agency and return to the depot within the limit.
    """
    supply, population = instance["supply"], instance["population"]
    travel, service = instance["travel"], instance["service"]
    limit = number(instance["limit"])
    size = len(supply)
    suppliers = [i for i in range(1, size) if number(supply[i]) > 0]
    agencies = {i for i in range(1, size) if number(population[i]) > 0}
    allowed = [i for i in range(1, size) if i in agencies or number(supply[i]) > 0]
    stop_cap = max_stops if max_stops is not None else size - 1
    remaining = return_bounds(instance)
    emitted = 0

    def walk(path, elapsed, visited):
        nonlocal emitted
        last = path[-1]
        if last in agencies and elapsed + number(travel[last][0]) <= limit:
            yield tuple(path)
            emitted += 1
        if len(path) >= stop_cap:
            return
        for node in allowed:
            if node in visited or remaining[node] is None:
                continue
            arrival = elapsed + number(travel[last][node]) + number(service[node])
            if arrival + remaining[node] > limit:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                return
            if max_routes is not None and emitted >= max_routes:
                return
            visited.add(node)
            yield from walk(path + [node], arrival, visited)
            visited.remove(node)

    for start in suppliers:
        if remaining[start] is None:
            continue
        elapsed = number(travel[0][start]) + number(service[start])
        if elapsed + remaining[start] > limit:
            continue
        if deadline is not None and time.monotonic() >= deadline:
            return
        if max_routes is not None and emitted >= max_routes:
            return
        yield from walk([start], elapsed, {start})


def count_routes(instance, **kwargs):
    return sum(1 for _ in enumerate_routes(instance, **kwargs))


class RoutePool:
    """A set of routes with lazily computed canonical columns."""

    def __init__(self, instance):
        self.instance = instance
        self.routes = []
        self.index = {}
        self.columns = {}

    def add(self, route):
        route = tuple(route)
        if route in self.index:
            return False
        if not route_is_structurally_feasible(self.instance, route):
            return False
        self.index[route] = len(self.routes)
        self.routes.append(route)
        return True

    def column(self, route):
        route = tuple(route)
        if route not in self.columns:
            from .fleet import canonical_column

            self.columns[route] = canonical_column(self.instance, route)
        return self.columns[route]

    def extend(self, generator):
        for route in generator:
            self.add(route)
        return self

    def __len__(self):
        return len(self.routes)
