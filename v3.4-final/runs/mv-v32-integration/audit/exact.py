"""Exact small-instance oracle for the simple-route multi-vehicle H-PDSP."""

from dataclasses import dataclass
from fractions import Fraction as F
from itertools import combinations


def need(condition, message):
    if not condition:
        raise ValueError(message)


def rational(value):
    return F(str(value))


@dataclass(frozen=True)
class Model:
    capacity: F
    limit: F
    supply: tuple
    population: tuple
    service: tuple
    travel: tuple

    @classmethod
    def read(cls, instance):
        capacity, limit = map(rational, (instance["capacity"], instance["limit"]))
        supply, population, service = (
            tuple(map(rational, instance[key])) for key in ("supply", "population", "service")
        )
        travel = tuple(tuple(map(rational, row)) for row in instance["travel"])
        size = len(supply)
        need(size >= 1 and len(population) == len(service) == len(travel) == size
             and all(len(row) == size for row in travel), "inconsistent instance dimensions")
        need(min(capacity, limit, *supply, *population, *service, *(min(row) for row in travel)) >= 0,
             "negative model parameter")
        need(supply[0] == population[0] == 0, "depot supply or population")
        need(all(bool(s) != bool(p) for s, p in zip(supply[1:], population[1:])),
             "every station must be exclusively a supplier or an agency")
        return cls(capacity, limit, supply, population, service, travel)

    @property
    def stations(self):
        return range(1, len(self.supply))


def verify_witness(instance, k, witness):
    """Check the documented route domain and exact pairwise-dispersion objective."""
    model = Model.read(instance)
    routes, amounts = witness["routes"], witness["amounts"]
    need(type(k) is int and k >= 1, "invalid fleet size")
    need(len(routes) == len(amounts) <= k, "fleet size mismatch")
    used = set()
    delivered = [F(0)] * len(model.population)
    for route, route_amounts in zip(routes, amounts):
        need(len(route) == len(route_amounts), "route/amount size mismatch")
        if not route:
            continue
        previous, elapsed, load = 0, F(0), F(0)
        for station, raw_amount in zip(route, route_amounts):
            need(type(station) is int and station in model.stations, "invalid non-depot station")
            need(station not in used, "station used more than once")
            used.add(station)
            amount = rational(raw_amount)
            need(amount >= 0, "negative quantity")
            elapsed += model.travel[previous][station] + model.service[station]
            if model.supply[station]:
                need(amount <= model.supply[station], "oversized pickup")
                load += amount
            else:
                delivered[station] = amount
                load -= amount
            need(0 <= load <= model.capacity, "load outside [0, capacity]")
            previous = station
        need(elapsed + model.travel[previous][0] <= model.limit, "return exceeds route limit")
    total_population = sum(model.population)
    if not total_population:
        return F(0)
    agencies = [i for i in model.stations if model.population[i]]
    dispersion = sum(
        abs(model.population[j] * delivered[i] - model.population[i] * delivered[j])
        for i, j in combinations(agencies, 2)
    )
    return sum(delivered) - dispersion / total_population


def simple_routes(model):
    """Enumerate timely returns without requiring earlier prefixes to return directly."""
    def visit(route, mask, elapsed, previous):
        for station in model.stations:
            bit = 1 << station
            if mask & bit:
                continue
            new_time = elapsed + model.travel[previous][station] + model.service[station]
            if new_time > model.limit:
                continue
            path = (*route, station)
            new_mask = mask | bit
            if new_time + model.travel[station][0] <= model.limit:
                yield path, new_mask
            yield from visit(path, new_mask, new_time, station)

    yield from visit((), 0, F(0), 0)


def route_rank(model, route, subset):
    """Maximum total delivery to a fixed subset when all other drops are zero."""
    load, total = F(0), F(0)
    for station in route:
        if model.supply[station]:
            load = min(model.capacity, load + model.supply[station])
        elif station in subset:
            total += load
            load = F(0)
    return total


def route_constraints(model, route):
    agencies = sorted(
        station for station in route
        if model.population[station] and route_rank(model, route, (station,))
    )
    return tuple(
        (subset, route_rank(model, route, subset))
        for size in range(1, len(agencies) + 1)
        for subset in combinations(agencies, size)
    )


def endpoint_compatible(model, route):
    return not route or bool(model.supply[route[0]] and model.population[route[-1]])


def direct_route_key(model, route):
    return (not endpoint_compatible(model, route), len(route), route)


def intersection(boundaries, dimension):
    """Solve the selected equalities exactly and discard singular choices."""
    matrix = [
        [*map(F, coefficients), limit] for coefficients, limit in boundaries
    ]
    for col in range(dimension):
        pivot = next((row for row in range(col, dimension) if matrix[row][col]), None)
        if pivot is None:
            return None
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        divisor = matrix[col][col]
        matrix[col] = [value / divisor for value in matrix[col]]
        for row in range(dimension):
            ratio = matrix[row][col]
            if ratio and row != col:
                matrix[row] = [
                    value - ratio * ref for value, ref in zip(matrix[row], matrix[col])
                ]
    return tuple(row[-1] for row in matrix)


def allocation(model, ranks):
    """Optimize over continuous amounts by cutting the polytope at rate ties."""
    agencies = sorted({subset[0] for subset, _ in ranks if len(subset) == 1})
    dimension = len(agencies)
    populations = tuple(model.population[i] for i in agencies)
    total_population = sum(model.population)
    constraints = [
        (tuple(-int(index == col) for col in range(dimension)), F(0))
        for index in range(dimension)
    ]
    constraints.extend(
        (tuple(int(i in subset) for i in agencies), rank)
        for subset, rank in ranks
    )
    rate_ties = [
        (tuple((populations[j] if col == i else -populations[i] if col == j else 0)
               for col in range(dimension)), F(0))
        for i, j in combinations(range(dimension), 2)
    ]
    best, best_amounts = F(0), (F(0),) * dimension
    systems = 0
    for boundaries in combinations((*constraints, *rate_ties), dimension):
        systems += 1
        point = intersection(boundaries, dimension)
        if point is None or any(
            sum(coef * amount for coef, amount in zip(coefficients, point)) > limit
            for coefficients, limit in constraints
        ):
            continue
        rates = tuple(amount / people for amount, people in zip(point, populations))
        weighted_min = sum(
            (1 if i == j else 2) * populations[i] * populations[j] * min(rates[i], rates[j])
            for i in range(dimension) for j in range(i, dimension)
        )
        score = weighted_min / total_population
        if score > best:
            best, best_amounts = score, point
    return best, dict(zip(agencies, best_amounts)), systems


def pickup_amounts(model, route, deliveries):
    """Backfill the drops from later to earlier stations, returning with zero load."""
    needed = F(0)
    amounts = []
    for station in reversed(route):
        if model.supply[station]:
            quantity = min(needed, model.supply[station])
            needed -= quantity
        else:
            quantity = deliveries.get(station, F(0))
            needed += quantity
        need(needed <= model.capacity, "oracle exceeded capacity while recovering pickups")
        amounts.append(str(quantity))
    need(needed == 0, "oracle could not recover pickups for its delivery vector")
    return list(reversed(amounts))


def solve(instance, k):
    model = Model.read(instance)
    need(type(k) is int and k >= 1, "invalid fleet size")
    need(len(model.stations) <= 8 and sum(bool(p) for p in model.population) <= 4,
         "oracle size limit: eight stations and four agencies")
    timely_paths = 0
    profiles = {}
    for route, mask in simple_routes(model):
        timely_paths += 1
        ranks = route_constraints(model, route)
        if ranks:
            identity = mask, ranks
            if identity not in profiles or direct_route_key(model, route) < direct_route_key(
                model, profiles[identity]
            ):
                profiles[identity] = route
    options = [(mask, ranks, route) for (mask, ranks), route in profiles.items()]
    best, best_fleet, best_deliveries = F(0), (), {}
    cached = {}
    fleet_count, systems = 0, 0

    def fleet_key(fleet):
        return (sum(not endpoint_compatible(model, route) for route in fleet),
                sum(map(len, fleet)), len(fleet), fleet)

    def enumerate_fleets(start, used, fleet, ranks):
        nonlocal best, best_fleet, best_deliveries, fleet_count, systems
        if len(fleet) == k:
            return
        for index in range(start, len(options)):
            mask, additions, route = options[index]
            if used & mask:
                continue
            fleet_count += 1
            new_fleet = tuple(sorted((*fleet, route)))
            new_ranks = tuple(sorted((*ranks, *additions)))
            if new_ranks not in cached:
                score, deliveries, examined = allocation(model, new_ranks)
                cached[new_ranks] = score, deliveries
                systems += examined
            score, deliveries = cached[new_ranks]
            if score > best or (score == best and fleet_key(new_fleet) < fleet_key(best_fleet)):
                best, best_fleet, best_deliveries = score, new_fleet, deliveries
            enumerate_fleets(index + 1, used | mask, new_fleet, new_ranks)

    enumerate_fleets(0, 0, (), ())
    witness = {
        "routes": [list(route) for route in best_fleet],
        "amounts": [pickup_amounts(model, route, best_deliveries) for route in best_fleet],
    }
    need(verify_witness(instance, k, witness) == best, "oracle witness differs from optimum")
    return {
        "K": k,
        "optimum": str(best),
        "witness": witness,
        "timely_ordered_paths": timely_paths,
        "positive_route_profiles": len(options),
        "positive_fleet_combinations": fleet_count,
        "distinct_allocation_polytopes": len(cached),
        "linear_systems_examined": systems,
    }
