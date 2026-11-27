"""Exact small-graph gate for routing, allocation, projection and proof replay."""

from fractions import Fraction
from itertools import combinations, permutations
from random import Random
from time import monotonic

from check import replay
from model import product_bound, read_model
from solve import Pricer


def solve_square(rows):
    matrix = [[*map(Fraction, coefficients), Fraction(rhs)] for coefficients, rhs in rows]
    n = len(matrix)
    for i in range(n):
        pivot = next((j for j in range(i, n) if matrix[j][i]), None)
        if pivot is None:
            return None
        matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
        scale = matrix[i][i]
        matrix[i] = [x / scale for x in matrix[i]]
        for j in range(n):
            if i != j and matrix[j][i]:
                factor = matrix[j][i]
                matrix[j] = [x - factor * y for x, y in zip(matrix[j], matrix[i])]
    return [row[-1] for row in matrix]


def truth(instance, k):
    supply, people, service = (instance[x] for x in ("supply", "population", "service"))
    travel, capacity, limit = (instance[x] for x in ("travel", "capacity", "limit"))
    nodes = tuple(range(1, len(supply)))
    total = sum(people)
    if not total:
        return Fraction()
    routes = []
    for length in range(2, len(nodes) + 1):
        for route in permutations(nodes, length):
            if not any(supply[i] for i in route) or not any(people[i] for i in route):
                continue
            duration = (sum(service[i] for i in route)
                        + sum(travel[i][j] for i, j in zip((0, *route), (*route, 0))))
            if duration <= limit:
                routes.append((frozenset(route), route))

    def allocate(fleet):
        demands = tuple(i for route in fleet for i in route if people[i])
        if not demands:
            return Fraction()
        d = len(demands)
        unit = {
            i: tuple(int(i == j) for j in demands)
            for i in demands
        }
        rows = {(tuple(-x for x in unit[i]), 0) for i in demands}
        for route in fleet:
            for end, stop in enumerate(route, 1):
                if not people[stop]:
                    continue
                # Greedy pickup gives maximum available load. Expanding
                # min(capacity, load + supply) gives every interval below.
                for start in range(end):
                    if start and not supply[route[start - 1]]:
                        continue
                    segment = set(route[start:end])
                    rows.add((tuple(int(i in segment) for i in demands),
                              sum(supply[i] for i in segment) + (capacity if start else 0)))
        best = Fraction()
        for order in permutations(demands):
            inequalities = rows | {
                (tuple(people[i] * y - people[j] * x for x, y in zip(unit[i], unit[j])), 0)
                for i, j in zip(order, order[1:])
            }
            for active in combinations(inequalities, d):
                point = solve_square(active)
                if point is None or any(
                    sum(a * x for a, x in zip(row, point)) > b for row, b in inequalities
                ):
                    continue
                deliveries = dict(zip(demands, point))
                loss = sum(abs(people[i] * deliveries.get(j, 0)
                               - people[j] * deliveries.get(i, 0))
                           for j, i in combinations(nodes, 2))
                best = max(best, sum(point) - loss / total)
        return best

    best = Fraction()

    def fleets(start, used, fleet):
        nonlocal best
        best = max(best, allocate(fleet))
        if len(fleet) < k:
            for index in range(start, len(routes)):
                mask, route = routes[index]
                if not used & mask:
                    fleets(index + 1, used | mask, (*fleet, route))

    fleets(0, frozenset(), ())
    return best


def route_truth(graph, reward):
    n = len(reward)
    return max([0] + [
        sum(reward[i] for i in path)
        for length in range(1, n + 1)
        for path in permutations(range(n), length)
        if (graph.start[path[0]] + sum(graph.arcs[i][j] for i, j in zip(path, path[1:]))
            + graph.finish[path[-1]]) <= graph.horizon
    ])


def fixture(supply, people, travel, *, capacity=3, limit=6, service=None):
    return {
        "supply": supply,
        "population": people,
        "service": service if service is not None else [0] * len(supply),
        "travel": travel,
        "capacity": capacity,
        "limit": limit,
    }


def main():
    rng = Random(908418)
    n = 5
    zero = [[0] * n for _ in range(n)]
    directed = [[9] * n for _ in range(n)]
    for i, j in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0)):
        directed[i][j] = 1
    cases = [
        (fixture([0, 0, 3, 0, 2], [0, 7, 0, 5, 0], directed), 1),
        (fixture([0, 2, 3, 0, 0], [0, 0, 0, 1, 8], zero, limit=0), 2),
        (fixture([0, 2, 3, 0, 0], [0, 0, 0, 1, 8], zero, limit=0), 0),
        (fixture([0, Fraction(2, 1019), 3, 0, 0], [0, 0, 0, 1, Fraction(8, 7)],
                 directed, limit=Fraction(56001, 10000)), 2),
    ]
    for case in range(12):
        travel = [[rng.randrange(5) if i != j else 0 for j in range(n)] for i in range(n)]
        cases.append((fixture([0, 2, 3, 0, 0], [0, 0, 0, 3, 5], travel,
                              limit=5 + case % 4,
                              service=[0] + [rng.randrange(2) for _ in range(n - 1)]),
                      1 + case % 3))

    checks = 0
    for index, (instance, k) in enumerate(cases):
        exact = truth(instance, k)
        model = read_model(instance)
        graph = model.project()
        pricer = Pricer(graph)
        boxes = tuple(model.boxes(k)) if k else ()
        supports = []
        for beta in (Fraction(2, 7), Fraction(1), Fraction(3)):
            prices = tuple(rng.randrange(1 << 20) for _ in range(model.n))
            reward = tuple(w - p for w, p in zip(model.weights(beta), prices))
            ceiling = route_truth(graph, reward)
            outcomes = (
                ("walk", False, pricer.solve(reward, monotonic() + 4)),
                ("elementary", False, pricer.solve(reward, monotonic() + 4, state_limit=5000)),
                ("elementary", True, pricer.solve(reward, monotonic() + 4, True, state_limit=5000)),
            )
            for mode, reverse, result in outcomes:
                assert result is not None, (index, beta, mode, reverse)
                price, _ = result
                assert price >= ceiling and (mode == "walk" or price == ceiling), (
                    index, beta, mode, reverse, price, ceiling)
                rhs = model.reward_unit * (sum(prices) + k * price)
                upper = product_bound(model, boxes, [(beta, rhs)])[0]
                cert = {"method": "route-price-v1", "vehicles": k, "upper": str(upper),
                        "supports": [{"beta": str(beta), "prices": prices, "route_bound": price,
                                      "reverse": reverse, "mode": mode}]}
                assert exact <= replay(instance, k, cert) == upper, (index, exact, upper)
                checks += 1
                cert["upper"] = "-1"
                try:
                    replay(instance, k, cert)
                except ValueError:
                    pass
                else:
                    raise AssertionError("forged envelope accepted")
                if k and ceiling > 0:
                    cert["upper"] = str(upper)
                    cert["supports"][0]["route_bound"] = ceiling - 1
                    try:
                        replay(instance, k, cert)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("forged route ceiling accepted")
    print(f"{len(cases)} exact fleet oracles, {checks} replayed pricing supports")


if __name__ == "__main__":
    main()
