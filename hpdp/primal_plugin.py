"""Propose routes with the independent study's product ALNS.

The product is a search heuristic. The caller receives only a route and
decides whether it improves the certified incumbent.
"""

from fractions import Fraction
from heapq import heappop, heappush
from itertools import pairwise
from math import lcm
from random import Random
from time import monotonic
from typing import Callable


class _DeadlineReached(Exception):
    pass


def _scaled_rows(rows, guard):
    """Convert written numeric values to an exact common integer unit."""
    scale = 1
    parsed = []
    for row in rows:
        guard()
        values = tuple(x if type(x) is int else Fraction(str(x)) for x in row)
        scale = lcm(scale, *(value.denominator for value in values))
        parsed.append(values)
    scaled = []
    for row in parsed:
        guard()
        scaled.append(tuple(x.numerator * (scale // x.denominator) for x in row))
    return scale, scaled


class _Search:
    def __init__(self, instance, deadline):
        self.deadline = deadline
        self.guard()
        self.n = len(instance["travel"])
        self.time_unit, travel = _scaled_rows(
            (*instance["travel"], instance["service"], (instance["limit"],)), self.guard,
        )
        self.travel = travel[:self.n]
        self.service = travel[self.n]
        self.limit = travel[-1][0]
        self.quantity_unit, quantities = _scaled_rows(
            (instance["supply"], instance["population"], (instance["capacity"],)), self.guard,
        )
        self.supply, self.people, (self.capacity,) = quantities
        self.pickups = tuple(i for i in range(1, self.n) if self.supply[i])
        self.agencies = tuple(i for i in range(1, self.n) if self.people[i])

    def guard(self):
        if monotonic() >= self.deadline:
            raise _DeadlineReached

    def duration(self, route):
        travel = self.travel
        return (sum(travel[u][v] for u, v in pairwise((0, *route, 0)))
                + sum(self.service[u] for u in route))

    def usable(self, route):
        return (len(route) >= 2 and all(type(u) is int and 0 < u < self.n for u in route)
                and len(set(route)) == len(route) and self.supply[route[0]]
                and self.people[route[-1]] and self.duration(route) <= self.limit)

    def rank(self, route):
        """Penalize the throughput product for early delivery shortages."""
        self.guard()
        people = sum(self.people[u] for u in route)
        load = flow = 0
        delivered_at = []
        for u in route:
            if self.supply[u]:
                load = min(self.capacity, load + self.supply[u])
            else:
                flow += load
                delivered_at.append((flow, self.people[u]))
                load = 0
        seen = shortage = 0
        for available, population in delivered_at:
            seen += population
            shortage = max(shortage, flow * seen - available * people)
        product = flow * people
        return product - shortage, product, -self.duration(route)

    def starting_routes(self):
        """Find feasible pairs or a shortest legal directed route."""
        travel, service = self.travel, self.service
        pairs = []
        for p in self.pickups:
            self.guard()
            start = travel[0][p] + service[p]
            for d in self.agencies:
                duration = start + travel[p][d] + service[d] + travel[d][0]
                if duration <= self.limit:
                    pairs.append((p, d))
        if pairs:
            return pairs

        # Separate depot copies enforce the first/last roles even in a
        # nonmetric instance where all two-stop walks exceed the limit.
        sink = self.n
        best = [0] + [self.limit + 1] * sink
        parent = [0] * (sink + 1)
        queue = [(0, 0)]
        while queue:
            self.guard()
            elapsed, u = heappop(queue)
            if elapsed != best[u]:
                continue
            if u == sink:
                route = []
                while parent[u]:
                    u = parent[u]
                    route.append(u)
                return [tuple(reversed(route))]
            stops = self.pickups if u == 0 else range(1, self.n)
            for v in stops:
                duration = elapsed + travel[u][v] + service[v]
                if duration < best[v]:
                    best[v], parent[v] = duration, u
                    heappush(queue, (duration, v))
            if self.people[u] and elapsed + travel[u][0] < best[sink]:
                best[sink], parent[sink] = elapsed + travel[u][0], u
                heappush(queue, (best[sink], sink))
        return []

    def two_opt(self, route):
        travel, supply, people = self.travel, self.supply, self.people
        n = len(route)
        for _ in range(60):
            self.guard()
            seq = (0, *route, 0)
            prefix = [0]
            for a, b in pairwise(route):
                prefix.append(prefix[-1] + travel[b][a] - travel[a][b])
            best_delta = 0
            move = None
            for i in range(n - 1):
                self.guard()
                a, b = seq[i], seq[i + 1]
                end = n if people[b] else n - 1
                for j in range(i + 1, end):
                    c, d = seq[j + 1], seq[j + 2]
                    if i == 0 and not supply[c]:
                        continue
                    # Reversing the block reverses all interior directed arcs.
                    delta = (travel[a][c] + travel[b][d] - travel[a][b] - travel[c][d]
                             + prefix[j] - prefix[i])
                    if delta < best_delta:
                        best_delta = delta
                        move = (i, j)
            if move is None:
                return
            i, j = move
            route[i:j + 1] = reversed(route[i:j + 1])

    def relocate(self, route):
        """Perform one improving Or-opt relocation of a one- or two-stop block."""
        travel, supply, people = self.travel, self.supply, self.people
        seq = (0, *route, 0)
        for length in (1, 2):
            if length >= len(route):
                break
            for i in range(len(route) - length + 1):
                self.guard()
                j = i + length
                a, first, last, b = seq[i], route[i], route[j - 1], seq[j + 1]
                removed = travel[a][first] + travel[last][b] - travel[a][b]
                rest = route[:i] + route[j:]
                rest_starts = bool(supply[rest[0]])
                rest_ends = bool(people[rest[-1]])
                first_slot = 0 if supply[first] and rest_ends else 1
                last_slot = len(rest) if people[last] and rest_starts else len(rest) - 1
                seq_rest = (0, *rest, 0)
                best_slot = None
                best_add = removed
                for pos in range(first_slot, last_slot + 1):
                    if 0 < pos < len(rest) and not (rest_starts and rest_ends):
                        continue
                    c, d = seq_rest[pos], seq_rest[pos + 1]
                    added = travel[c][first] + travel[last][d] - travel[c][d]
                    if added < best_add:
                        best_slot, best_add = pos, added
                if best_slot is not None:
                    route[:] = rest[:best_slot] + route[i:j] + rest[best_slot:]
                    return True
        return False

    def insert(self, route, rng, noise):
        travel, service, supply, people = self.travel, self.service, self.supply, self.people
        elapsed = self.duration(route)
        sigma = sum(supply[u] for u in route)
        population = sum(people[u] for u in route)
        unused = set(range(1, self.n)).difference(route)
        while unused:
            best = None
            seq = (0, *route, 0)
            arcs = tuple((a, b, travel[a][b]) for a, b in pairwise(seq))
            for u in sorted(unused):
                self.guard()
                first = 0 if supply[u] else 1
                end = len(arcs) if people[u] else len(arcs) - 1
                best_delta = None
                for pos in range(first, end):
                    a, b, old = arcs[pos]
                    delta = travel[a][u] + travel[u][b] - old + service[u]
                    if best_delta is None or delta < best_delta:
                        best_delta, slot = delta, pos
                if best_delta is None or elapsed + best_delta > self.limit:
                    continue
                gain = (supply[u] * (population + self.quantity_unit)
                        + people[u] * (sigma + self.quantity_unit))
                score = gain / max(best_delta, self.time_unit)
                if noise:
                    score *= 1 + noise * rng.random()
                if best is None or score > best[0]:
                    best = (score, u, slot, best_delta)
            if best is None:
                return
            _, u, slot, delta = best
            route.insert(slot, u)
            unused.remove(u)
            elapsed += delta
            sigma += supply[u]
            population += people[u]

    def swap(self, route):
        travel, service, supply, people = self.travel, self.service, self.supply, self.people
        elapsed = self.duration(route)
        while True:
            unused = set(range(1, self.n)).difference(route)
            seq = (0, *route, 0)
            sigma = sum(supply[u] for u in route)
            population = sum(people[u] for u in route)
            best_key = (0, 0)
            best = None
            for i, u in enumerate(route):
                self.guard()
                a, b = seq[i], seq[i + 2]
                old_cost = travel[a][u] + travel[u][b] + service[u]
                candidates = self.pickups if supply[u] else self.agencies
                for v in candidates:
                    if v not in unused:
                        continue
                    gain = ((supply[v] - supply[u]) * population
                            + (people[v] - people[u]) * sigma)
                    if gain < 0:
                        continue
                    delta = travel[a][v] + travel[v][b] + service[v] - old_cost
                    key = (gain, -delta)
                    if elapsed + delta <= self.limit and key > best_key:
                        best_key = key
                        best = (i, v, delta)
            if best is None:
                return
            i, v, delta = best
            route[i] = v
            elapsed += delta

    def repair(self, route, rng, noise=0):
        route = list(route)
        for _ in range(2):
            self.two_opt(route)
            for _ in range(30):
                if not self.relocate(route):
                    break
            self.insert(route, rng, noise)
            self.swap(route)
        return tuple(route)

    def destroy(self, route, rng):
        self.guard()
        if len(route) <= 3:
            return route
        k = min(len(route) - 2, rng.randint(2, max(3, min(10, len(route) // 4))))
        travel = self.travel
        ranked = []
        for a, u, b in zip(route, route[1:], route[2:]):
            cost = travel[a][u] + travel[u][b] - travel[a][b] + self.service[u]
            prize = self.supply[u] + self.people[u]
            ranked.append((prize / max(cost, self.time_unit), u))
        ranked.sort()
        drop = {u for _, u in ranked[:k // 2]}
        drop.update(rng.sample([u for _, u in ranked[k // 2:]], k - len(drop)))
        return tuple(u for u in route if u not in drop)


def propose(
    instance: dict,
    seed_route: tuple | None,
    deadline: float,
    *,
    emit: Callable[[tuple[int, ...]], bool],
    seed: int = 20261119,
) -> None:
    """Send successive private-rank improvements before the monotonic deadline."""
    try:
        search = _Search(instance, deadline)
        rng = Random(seed)
        best = seed_route if seed_route and search.usable(seed_route) else None
        best_rank = search.rank(best) if best is not None else None

        def consider(route):
            nonlocal best, best_rank
            search.guard()
            rank = search.rank(route)
            if best_rank is None or rank > best_rank:
                best, best_rank = route, rank
                search.guard()
                return emit(route)
            return True

        starts = search.starting_routes() if best is None else None
        if best is None:
            if not starts:
                return
            initial = max(starts, key=search.rank)
            if not consider(initial):
                return
        current = search.repair(best, rng)
        if not consider(current):
            return

        stalled = iteration = 0
        while stalled < 256:
            search.guard()
            iteration += 1
            if iteration % 12 == 0:
                if starts is None:
                    starts = search.starting_routes()
                if not starts:
                    starts = [best]
                base = rng.choice(starts)
            else:
                base = best if rng.random() < 0.6 else current
                base = search.destroy(base, rng)
            candidate = search.repair(base, rng, noise=0.35)
            if not search.usable(candidate):
                stalled += 1
                continue
            current = candidate
            previous = best
            if not consider(current):
                return
            stalled = 0 if best is not previous else stalled + 1
    except _DeadlineReached:
        return
