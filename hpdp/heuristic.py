"""Deterministic common incumbent: feasible two-stop starts and route insertion."""

from fractions import Fraction
from heapq import heappop, heappush
from operator import truediv
from time import monotonic

from .allocation import allocate
from .check import check


def throughput(r, route):
    delivered = load = 0
    for v in route:
        if r.supply[v]:
            load = min(r.capacity, load + r.supply[v])
        else:
            delivered += load
            load = 0
    return delivered


def shortest_nonempty(r):
    """Shortest first-pickup, last-delivery tour using separate depot copies."""
    sink = len(r.pop)
    parents = {}
    times = {0: 0}
    queue = [(0, 0)]
    while queue:
        elapsed, node = heappop(queue)
        if times[node] != elapsed:
            continue
        if node == sink:
            route = []
            while parents[node]:
                node = parents[node]
                route.append(node)
            return tuple(reversed(route))
        next_stops = r.p if node == 0 else r.nodes
        for v in next_stops:
            if v == node:
                continue
            time = elapsed + r.travel[node][v] + r.service[v]
            if time <= r.limit and time < times.get(v, r.limit+1):
                times[v], parents[v] = time, node
                heappush(queue, (time, v))
        if r.pop[node]:
            time = elapsed + r.travel[node][0]
            if time <= r.limit and time < times.get(sink, r.limit+1):
                times[sink], parents[sink] = time, node
                heappush(queue, (time, sink))
    return None


def shared_seed(r, deadline, *, insert=True, emit=None):
    """The same bounded insertion work is charged to every compared method."""
    pairs = []
    for p in r.p:
        for d in r.d:
            t = r.travel[0][p] + r.travel[p][d] + r.travel[d][0] + r.service[p] + r.service[d]
            if t <= r.limit:
                pairs.append((r.supply[p]*r.pop[d], t, p, d))
    if not pairs:
        route = shortest_nonempty(r) if insert and monotonic() < deadline else None
        if route is None:
            return None
        value, amounts = allocate(r.instance, route)
        if check(r.instance, route, amounts, tol=0)["objective"] != value:
            raise ArithmeticError("Shortest route failed independent verification")
        seed = {"route":route, "amounts":amounts}
        if emit:
            emit(seed, value)
        return seed
    pairs.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
    fast = max(pairs, key=lambda x: Fraction(x[0], max(1, x[1])))
    best = None
    best_value = None
    def evaluate(route):
        nonlocal best, best_value
        value, amounts = allocate(r.instance, route)
        if best_value is None or value > best_value:
            actual = check(r.instance, route, amounts, tol=0)["objective"]
            if value != actual:
                raise ArithmeticError("Heuristic witness disagrees with independent checker")
            best, best_value = {"route": tuple(route), "amounts": amounts}, actual
            if emit:
                emit(best, actual)
    evaluate(pairs[0][2:])
    if not insert:
        return best
    # Both proxy and score lie between zero and the integer numerator prior.
    # Normal benchmark inputs retain the original binary64 ordering. Fine
    # decimal units can make their difference too large to represent as a float.
    insertion_rate = Fraction if r.prior.bit_length() > 1023 else truediv
    for score, elapsed, p, d in dict.fromkeys((pairs[0], fast)):
        route = (p, d)
        available = set(r.nodes) - set(route)
        people = r.pop[d]
        for _ in range(min(32, len(available))):
            if monotonic() >= deadline:
                return best
            walk = (0,) + route + (0,)
            choice = None
            for v in sorted(available):
                # Keep the two least costly legal insertion positions of each site.
                lo = 0 if r.supply[v] else 1
                hi = len(route) if r.supply[v] else len(route)+1
                trials = []
                for pos in range(lo, hi):
                    i, j = walk[pos:pos+2]
                    delta = r.travel[i][v]+r.service[v]+r.travel[v][j]-r.travel[i][j]
                    if elapsed + delta <= r.limit:
                        trials.append((delta, pos))
                for delta, pos in sorted(trials)[:2]:
                    proposed = route[:pos] + (v,) + route[pos:]
                    proxy = throughput(r, proposed) * (people + r.pop[v])
                    # A zero-gain stop can usefully position the next supply or delivery.
                    rank = (insertion_rate(proxy-score, max(r.time_scale, delta)), proxy, -delta, -v, -pos)
                    if choice is None or rank > choice[0]:
                        choice = (rank, proposed, v, delta, proxy)
            if choice is None:
                break
            _, route, v, delta, score = choice
            elapsed += delta
            people += r.pop[v]
            available.remove(v)
            evaluate(route)
    return best
