"""Exact fleet verifier and fleet-level objective for multi-vehicle H-PDSP.

Independent of the single-vehicle checker: every constraint is re-derived from
the raw instance and every number is a :class:`fractions.Fraction`, so a zero
tolerance run is an exact certificate for the given witness.

Model conventions (locked for this project, see ``docs/MODEL.md``):

* ``K`` identical vehicles, capacity ``C``, common route-time limit ``L``.
* Every vehicle starts empty at the depot and returns to the depot.
* Each non-empty route begins at a supplier and ends at an agency; zero
  quantities are allowed at any visited stop and service time still counts.
* Each supplier and each agency is visited by the whole fleet at most once.
* Pickups and deliveries may alternate along a route; leftover load may be
  carried back to the depot.
* Deliveries are aggregated over the fleet and scored by the global
  proportional-fair objective.
"""

from fractions import Fraction
from itertools import combinations, pairwise
from numbers import Integral

from . import allocation as _parent_allocation


def number(value):
    """Exact rational reading of ints, Fractions, Decimals and float literals."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    return Fraction(str(value))


def json_number(value):
    return value.numerator if value.denominator == 1 else str(value)


def fleet_objective(instance, deliveries, *, check_zero=True):
    """Return the exact fleet objective for an agency delivery vector.

    ``deliveries`` may be a dict ``{agency: quantity}`` or a sequence indexed
    like the instance (entry 0 is ignored).  Agencies missing from a dict, or
    with zero entry in a sequence, count as unserved but stay in the population
    denominator.  The objective is

        Z = sum_i q_i - sum_{i<j} |p_i q_j - p_j q_i|,

    with ``p_i`` the population share over *all* agencies of the instance.
    """
    population = instance["population"]
    agencies = [i for i in range(1, len(population)) if number(population[i]) > 0]
    if isinstance(deliveries, dict):
        quantity = {i: number(deliveries.get(i, 0)) for i in agencies}
    else:
        if len(deliveries) != len(population):
            raise ValueError("delivery vector must match the instance length")
        quantity = {i: number(deliveries[i]) for i in agencies}
    for i, value in quantity.items():
        if value < 0:
            raise ValueError(f"Negative delivery at agency {i}")
    total_population = sum(number(population[i]) for i in agencies)
    if check_zero and total_population == 0:
        raise ValueError("instance has no agency population")
    weights = {i: number(population[i]) / total_population for i in agencies}
    total = sum(quantity.values())
    penalty = sum(abs(weights[i] * quantity[j] - weights[j] * quantity[i])
                  for i, j in combinations(agencies, 2))
    return total - penalty


def objective_layers(instance, deliveries):
    """Same objective through the survival-curve identity ``Z = int F^2``.

    ``F(t)`` is the population share of agencies whose per-capita delivery
    ``q_i / p_i`` is at least ``t``.  For a finite step function
    ``Z = sum_k (t_k - t_{k-1}) * F(t_k)^2``.  Used as an independent algebra
    check of :func:`fleet_objective` and as the basis of the decomposition
    argument that makes route columns exact.
    """
    population = instance["population"]
    agencies = [i for i in range(1, len(population)) if number(population[i]) > 0]
    if isinstance(deliveries, dict):
        quantity = {i: number(deliveries.get(i, 0)) for i in agencies}
    else:
        quantity = {i: number(deliveries[i]) for i in agencies}
    total_population = sum(number(population[i]) for i in agencies)
    weights = {i: number(population[i]) / total_population for i in agencies}
    # Survival curve F(t) = population share with per-capita delivery >= t.
    steps = sorted((quantity[i] / weights[i], number(population[i])) for i in agencies)
    score = Fraction(0)
    previous = Fraction(0)
    standing = total_population  # population with ratio >= previous (all at t=0)
    for ratio, people in steps:
        if ratio > previous:
            score += (ratio - previous) * (standing / total_population) ** 2
            previous = ratio
        standing -= people
    return score


def _route_detail(instance, route, amounts, tol):
    """Validate one route and return its exact time, loads and deliveries."""
    tolerance = number(tol)
    stops = tuple(route)
    quantities = tuple(number(value) for value in amounts)
    if len(stops) < 2:
        raise ValueError("a non-empty route needs at least two stops")
    if len(stops) != len(quantities):
        raise ValueError("one amount per stop is required")

    supply, population = instance["supply"], instance["population"]
    for node in stops:
        if isinstance(node, bool) or not isinstance(node, Integral) or not 0 < node < len(supply):
            raise ValueError(f"invalid non-depot stop {node!r}")
        if (number(supply[node]) > 0) == (number(population[node]) > 0):
            raise ValueError(f"stop {node} must be an active supplier or agency")
    if len(set(stops)) != len(stops):
        raise ValueError("a route may visit each stop at most once")
    if number(supply[stops[0]]) <= 0:
        raise ValueError("the first stop must be a supplier")
    if number(population[stops[-1]]) <= 0:
        raise ValueError("the last stop must be a delivery agency")

    travel, service = instance["travel"], instance["service"]
    elapsed = sum(number(travel[i][j]) for i, j in pairwise((0, *stops, 0)))
    elapsed += sum(number(service[i]) for i in stops)
    if elapsed > number(instance["limit"]):
        raise ValueError(f"route time {elapsed} exceeds limit {instance['limit']}")

    capacity = number(instance["capacity"])
    load = Fraction(0)
    loads, deliveries = [], {}
    for node, amount in zip(stops, quantities):
        if amount < -tolerance:
            raise ValueError(f"negative amount at stop {node}")
        if number(supply[node]) > 0:
            if amount > number(supply[node]) + tolerance:
                raise ValueError(f"pickup exceeds supply at stop {node}")
            load += amount
        else:
            deliveries[node] = amount
            load -= amount
        if not -tolerance <= load <= capacity + tolerance:
            raise ValueError(f"load {load} outside [0, {capacity}] after stop {node}")
        loads.append(load)
    return dict(route=list(stops), amounts=[json_number(v) for v in quantities],
                time=elapsed, loads=loads, return_load=load,
                deliveries={i: v for i, v in deliveries.items() if v},
                delivered=sum(deliveries.values()))


def verify_fleet(instance, routes, amounts, *, vehicles=None, tol=0):
    """Exactly verify a fleet witness.

    ``routes`` holds one stop sequence per used vehicle (the depot is implicit
    and must be omitted).  Trailing empty routes denote idle vehicles; the
    number of entries may not exceed ``vehicles``.  Returns a report with the
    exact fleet objective, per-route detail and the aggregated delivery vector.
    Raises ``ValueError`` on the first violated constraint.
    """
    tolerance = number(tol)
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    if len(routes) != len(amounts):
        raise ValueError("one amount sequence per route is required")
    if vehicles is not None and len(routes) > vehicles:
        raise ValueError(f"{len(routes)} routes exceed the fleet size {vehicles}")

    seen = {}
    details = []
    aggregated = {}
    for index, (route, sequence) in enumerate(zip(routes, amounts)):
        if not route:
            if sequence:
                raise ValueError(f"idle vehicle {index} has quantities")
            details.append(dict(vehicle=index, route=[], amounts=[], time=Fraction(0),
                                loads=[], deliveries={}, delivered=Fraction(0)))
            continue
        detail = _route_detail(instance, route, sequence, tolerance)
        for node in detail["route"]:
            if node in seen:
                raise ValueError(
                    f"stop {node} is visited by vehicles {seen[node]} and {index}")
            seen[node] = index
        for node, quantity in detail["deliveries"].items():
            aggregated[node] = aggregated.get(node, Fraction(0)) + quantity
        detail["vehicle"] = index
        details.append(detail)

    population = instance["population"]
    for node in aggregated:
        if number(population[node]) <= 0:
            raise ValueError(f"delivery at non-agency {node}")
    objective = fleet_objective(instance, aggregated)
    return dict(objective=objective, deliveries=aggregated, routes=details,
                used_vehicles=sum(1 for route in routes if route),
                visited=sum(len(route) for route in routes),
                total_delivered=sum(aggregated.values()))


def canonical_column(instance, route):
    """Canonical (exact optimal) allocation of a fixed route.

    Returns ``(score, amounts)`` where ``score`` is the route's own fair score
    against the global population denominator.  The decomposition theorem
    (verified by brute force in this project) says the fleet optimum for a set
    of disjoint routes is obtained by merging these canonical columns.
    """
    return _parent_allocation.allocate(instance, route)


def route_pool_value(instance, routes, columns=None):
    """Fleet objective of a set of disjoint routes using canonical columns."""
    deliveries = {}
    seen = set()
    for route in routes:
        if not route:
            continue
        if len(set(route)) != len(route) or seen & set(route):
            raise ValueError("routes in a pool must be non-empty and disjoint")
        seen.update(route)
        _, amounts = canonical_column(instance, route) if columns is None else columns[route]
        for node, quantity in zip(route, amounts):
            if number(instance["population"][node]) > 0:
                deliveries[node] = deliveries.get(node, Fraction(0)) + number(quantity)
    return fleet_objective(instance, deliveries), deliveries


def sorted_witness(instance, routes, amounts, *, vehicles=None):
    """Normalized JSON-friendly witness for a fleet solution."""
    report = verify_fleet(instance, routes, amounts, vehicles=vehicles, tol=0)
    return {
        "objective": json_number(report["objective"]),
        "used_vehicles": report["used_vehicles"],
        "vehicles": [
            {"vehicle": detail["vehicle"], "route": detail["route"],
             "amounts": detail["amounts"], "time": json_number(detail["time"])}
            for detail in report["routes"]
        ],
    }
