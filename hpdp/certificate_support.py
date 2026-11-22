"""Fixed-route verification and dense visit-set primitives used by V8."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import groupby
from math import lcm

import numpy as np

from .check import check


def rank_certificate(instance, route, amounts):
    """Verify a supplied solution and certify optimality on its fixed route."""
    feasible = check(instance, route, amounts, tol=0)
    capacity = Fraction(str(instance["capacity"]))
    pop = [Fraction(str(p)) for p in instance["population"]]
    supply = [Fraction(str(s)) for s in instance["supply"]]
    y = {i: Fraction(str(amount)) for i, amount in zip(route, amounts) if pop[i]}
    order = sorted(y, key=lambda i: y[i] / pop[i])
    selected = set()
    levels = []
    for rate, tied in groupby(order, lambda i: y[i] / pop[i]):
        selected.update(tied)
        # Between consecutive selected agencies only one load can arrive.
        # Sum the capped supplies in these intervals on the original route.
        interval_supply = rank = 0
        for i in route:
            interval_supply += supply[i]
            if i in selected:
                rank += min(capacity, interval_supply)
                interval_supply = 0
        received = sum(y[i] for i in selected)
        if received != rank:
            raise ValueError(f"Population-level set {sorted(selected)} receives {received}, rank {rank}")
        levels.append(dict(rate=rate, agencies=sorted(selected), received=received, rank=rank))
    return dict(objective=feasible["objective"], upper_bound=feasible["objective"],
                return_load=feasible["return_load"], levels=levels)


def _integer_units(values):
    fractions = [Fraction(str(value)) for value in values]
    scale = lcm(*(value.denominator for value in fractions))
    return [int(value * scale) for value in fractions], scale


def _dense_layout(n, max_arc, max_service):
    if not 2 <= n <= 20:
        raise ValueError("Dense certificate supports 2 to 20 nondepot sites")
    max_tour = (n + 1) * max_arc + max_service
    unreachable = max_tour + 1
    # Dense arithmetic includes the infinity sentinel and its outgoing arc.
    largest_addition = unreachable + max_arc + max_service
    if largest_addition > np.iinfo(np.int64).max:
        raise ValueError("Exact time scaling exceeds safe signed 64-bit arithmetic")
    dtype = np.int32 if largest_addition <= np.iinfo(np.int32).max else np.int64
    return max_tour, unreachable, dtype


@dataclass(slots=True)
class _VisitTimes:
    paths: np.ndarray
    arcs: np.ndarray
    end: np.ndarray
    trip_time: np.ndarray
    feasible: np.ndarray
    scale: int
    unreachable: int

    def shortest_route(self, mask):
        last = int(self.end[mask])
        remaining = mask
        reversed_route = []
        while remaining:
            reversed_route.append(last + 1)
            previous = remaining ^ (1 << last)
            if previous:
                elapsed = int(self.paths[remaining, last])
                last = next(
                    i for i in range(self.paths.shape[1])
                    if previous & (1 << i)
                    and int(self.paths[previous, i]) + int(self.arcs[i + 1, last + 1]) == elapsed
                )
            remaining = previous
        return reversed_route[::-1]


def _visit_times(instance):
    """Minimum original depot-to-last travel for all masks, without time pruning."""
    size = len(instance["travel"])
    n = size - 1
    if not 2 <= n <= 20:
        raise ValueError("Dense certificate supports 2 to 20 nondepot sites")

    all_time, scale = _integer_units(
        [t for row in instance["travel"] for t in row] + instance["service"] + [instance["limit"]]
    )
    arc_values, service, limit = all_time[:size**2], all_time[size**2:-1], all_time[-1]
    max_tour, unreachable, dtype = _dense_layout(n, max(arc_values), sum(service))
    arcs = np.array(arc_values, dtype=dtype).reshape((size, size))

    pickup_ends = [i for i, amount in enumerate(instance["supply"][1:]) if amount > 0]
    delivery_ends = [i for i, people in enumerate(instance["population"][1:]) if people > 0]
    pickup_mask = sum(1 << i for i in pickup_ends)
    mask_count = 1 << n
    paths = np.full((mask_count, n), unreachable, dtype=dtype)
    for i in pickup_ends:
        paths[1 << i, i] = arcs[0, i + 1]

    cardinality = np.zeros(mask_count, dtype=np.uint8)
    services = np.zeros(mask_count, dtype=dtype)
    for j in range(n):
        bit = 1 << j
        cardinality[bit:2 * bit] = cardinality[:bit] + 1
        services[bit:2 * bit] = services[:bit] + service[j + 1]
    with_pickup = np.arange(mask_count) & pickup_mask != 0
    for count in range(1, n):
        layer = np.flatnonzero((cardinality == count) & with_pickup)
        for j in range(n):
            bit = 1 << j
            prior = layer[(layer & bit) == 0]
            # The sentinel term can exceed unreachable if no feasible
            # predecessor exists. Clamp it back before storing next-layer costs.
            travel = (paths[prior] + arcs[1:, j + 1]).min(axis=1)
            paths[prior | bit, j] = np.minimum(travel, unreachable)

    end = np.full(mask_count, -1, dtype=np.int8)
    return_travel = np.full(mask_count, unreachable, dtype=dtype)
    for j in delivery_ends:
        closed = paths[:, j] + arcs[j + 1, 0]
        better = closed < return_travel
        end[better] = j
        return_travel[better] = closed[better]
    trip_time = return_travel + services
    feasible = np.flatnonzero((end >= 0) & (trip_time <= min(limit, max_tour)))
    return _VisitTimes(paths, arcs, end, trip_time, feasible, scale, unreachable)


def _box_numerator(sorted_people, capacity, flow, covered):
    """Evaluate all saturation-count bounds with exact integer arithmetic."""
    best = flow * covered
    prefix = 0
    tail = covered
    for k, population in enumerate(sorted_people, 1):
        prefix += capacity * (2 * k - 1) * population
        tail -= population
        best = min(best, prefix + tail * (flow + k * capacity))
    return best


def _reverse_tails(instance):
    # A reverse path starts with an original agency: its first arc is the
    # original depot-return arc. Its last node is the first node of the suffix.
    return _visit_times({
        **instance, "travel": list(map(list, zip(*instance["travel"]))),
        "supply": instance["population"], "population": instance["supply"],
    })


def _nodes(mask):
    while mask:
        bit = mask & -mask
        yield bit
        mask ^= bit
