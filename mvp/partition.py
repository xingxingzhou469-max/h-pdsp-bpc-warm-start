"""Partition-based fleet heuristic.

For a fixed set of stops, the only freedom left is their sequence.  Because
travel is directed and non-metric, a set can be feasible in one order and
infeasible in another.  This module

* finds good time-feasible sequences for a node set (randomised nearest
  neighbour plus 2-opt / Or-opt);
* searches over partitions of the active nodes into at most K routes, moving
  nodes between routes and re-sequencing only the affected routes;
* records every evaluated route in a pool for the integer master.

The fleet objective is evaluated with the exact canonical columns (converted
to binary64 for speed); incumbents are re-verified exactly by the caller.
"""

import math
import random
import time

from .fleet import number
from .routes import RoutePool, route_is_structurally_feasible


class PartitionSearch:
    def __init__(self, instance, vehicles, seed=0, pool=None):
        self.instance = instance
        self.K = int(vehicles)
        self.rng = random.Random(seed)
        self.supply = [float(number(v)) for v in instance["supply"]]
        self.population = [float(number(v)) for v in instance["population"]]
        self.travel = [[float(number(v)) for v in row] for row in instance["travel"]]
        self.service = [float(number(v)) for v in instance["service"]]
        self.capacity = float(number(instance["capacity"]))
        self.suppliers = [i for i in range(1, len(self.supply)) if self.supply[i] > 0]
        self.agencies = [i for i in range(1, len(self.supply)) if self.population[i] > 0]
        self.active = [i for i in range(1, len(self.supply))
                       if self.supply[i] > 0 or self.population[i] > 0]
        self.total_people = sum(self.population[i] for i in self.agencies)
        self.weights = {i: self.population[i] / self.total_people for i in self.agencies}
        self.pool = RoutePool(instance) if pool is None else pool
        self._columns = {}
        self._orders = {}

    # -- sequencing ------------------------------------------------------
    def time_of(self, route):
        total = 0.0
        previous = 0
        for node in route:
            total += self.travel[previous][node] + self.service[node]
            previous = node
        return total + self.travel[previous][0]

    def _endpoints_ok(self, route):
        return (bool(route) and self.supply[route[0]] > 0
                and self.population[route[-1]] > 0)

    def _two_opt(self, route, deadline):
        best = list(route)
        best_time = self.time_of(best)
        improved = True
        while improved and time.monotonic() < deadline:
            improved = False
            size = len(best)
            for i in range(size - 1):
                if time.monotonic() >= deadline:
                    return best, best_time
                for j in range(i + 1, size):
                    candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    if not self._endpoints_ok(candidate):
                        continue
                    value = self.time_of(candidate)
                    if value < best_time - 1e-9:
                        best, best_time, improved = candidate, value, True
            for length in (1, 2, 3):
                for start in range(size - length + 1):
                    if time.monotonic() >= deadline:
                        return best, best_time
                    segment = best[start:start + length]
                    rest = best[:start] + best[start + length:]
                    for position in range(len(rest) + 1):
                        candidate = rest[:position] + segment + rest[position:]
                        if candidate == best or not self._endpoints_ok(candidate):
                            continue
                        value = self.time_of(candidate)
                        if value < best_time - 1e-9:
                            best, best_time, improved = candidate, value, True
        return best, best_time

    def remember_order(self, route):
        self._orders[frozenset(route)] = list(route)

    def best_order(self, nodes, *, restarts=4, deadline=None, register=True):
        """Return the best feasible sequence for ``nodes`` (or ``None``)."""
        if deadline is not None and time.monotonic() >= deadline:
            return None
        key = frozenset(nodes)
        if len(key) != len(nodes):
            return None
        cached = self._orders.get(key)
        if cached is not None:
            if register:
                self.pool.add(cached)
            return cached
        nodes = list(nodes)
        if not nodes:
            return []
        if not any(self.supply[n] > 0 for n in nodes) or not any(self.population[n] > 0 for n in nodes):
            self._orders[key] = None
            return None
        suppliers = [n for n in nodes if self.supply[n] > 0]
        agencies = [n for n in nodes if self.population[n] > 0]
        best, best_time = None, math.inf
        deadline = deadline if deadline is not None else time.monotonic() + 1.0
        for attempt in range(max(1, restarts)):
            if time.monotonic() >= deadline:
                break
            first = suppliers[attempt % len(suppliers)] if attempt < len(suppliers) else self.rng.choice(suppliers)
            remaining = [n for n in nodes if n != first]
            route = [first]
            # randomised nearest-neighbour over the remaining stops
            while remaining:
                if attempt == 0:
                    choice = min(remaining, key=lambda n: self.travel[route[-1]][n] + self.service[n])
                else:
                    ranked = sorted(remaining, key=lambda n: self.travel[route[-1]][n] + self.service[n])
                    choice = ranked[min(len(ranked) - 1, int(self.rng.expovariate(1.0)))]
                route.append(choice)
                remaining.remove(choice)
            # Move a supplier or agency to a valid endpoint when needed.
            if self.population[route[-1]] <= 0:
                for index in range(len(route) - 1, -1, -1):
                    if self.population[route[index]] > 0:
                        route.append(route.pop(index))
                        break
            if self.supply[route[0]] <= 0:
                for index in range(len(route)):
                    if self.supply[route[index]] > 0:
                        route.insert(0, route.pop(index))
                        break
            if not self._endpoints_ok(route):
                continue
            order, value = self._two_opt(route, min(deadline, time.monotonic() + 0.5))
            if value < best_time and route_is_structurally_feasible(self.instance, order):
                best, best_time = order, value
        if best is None:
            return None
        self._orders[key] = best
        if register:
            self.pool.add(best)
        return best

    # -- columns and value -----------------------------------------------
    def column(self, route):
        key = tuple(route)
        cached = self._columns.get(key)
        if cached is None:
            _, amounts = self.pool.column(key)
            deliveries = {}
            for node, quantity in zip(route, amounts):
                value = float(number(quantity))
                if self.population[node] > 0 and value > 0:
                    deliveries[node] = value
            cached = deliveries
            self._columns[key] = cached
        return cached

    def value(self, routes):
        merged = {}
        for route in routes:
            if not route:
                continue
            for node, quantity in self.column(route).items():
                merged[node] = merged.get(node, 0.0) + quantity
        steps = sorted((merged.get(i, 0.0) / self.weights[i], self.population[i])
                       for i in self.agencies)
        total = 0.0
        previous = 0.0
        standing = self.total_people
        for ratio, people in steps:
            if ratio > previous:
                total += (ratio - previous) * (standing / self.total_people) ** 2
                previous = ratio
            standing -= people
        return total

    # -- partition search ------------------------------------------------
    def balanced_partition(self):
        """Initial partition balancing supply per population across routes."""
        groups = [[] for _ in range(self.K)]
        supply_load = [0.0] * self.K
        people_load = [0.0] * self.K
        for node in sorted(self.suppliers, key=lambda n: -self.supply[n]):
            index = min(range(self.K), key=lambda k: supply_load[k])
            groups[index].append(node)
            supply_load[index] += min(self.capacity, self.supply[node])
        for node in sorted(self.agencies, key=lambda n: -self.population[n]):
            index = max(range(self.K),
                        key=lambda k: (supply_load[k] + 1e-9) / (people_load[k] + 1e-9))
            groups[index].append(node)
            people_load[index] += self.population[node]
        return groups

    def repair(self, groups, *, deadline):
        """Turn each group into a feasible route by moving offending nodes."""
        routes = [[] for _ in range(self.K)]
        groups = list(groups) + [[] for _ in range(max(0, self.K - len(groups)))]
        overflow = []
        for index, group in enumerate(groups):
            if time.monotonic() >= deadline:
                return routes
            if not group:
                routes[index] = []
                continue
            order = self.best_order(group, deadline=deadline)
            if order is None:
                overflow.extend(group)
                routes[index] = []
            else:
                routes[index] = order
        for node in overflow:
            if time.monotonic() >= deadline:
                break
            if any(node in (route or []) for route in routes):
                continue
            placed = False
            for index in range(self.K):
                if not routes[index]:
                    continue
                candidate = self.best_order(list(routes[index]) + [node], restarts=2, deadline=deadline)
                if candidate is not None:
                    routes[index] = candidate
                    placed = True
                    break
            if not placed:
                for index in range(self.K):
                    if not routes[index]:
                        matched = False
                        for other in self.active:
                            if time.monotonic() >= deadline:
                                break
                            if other == node or any(other in (r or []) for r in routes):
                                continue
                            if (self.supply[node] > 0) != (self.supply[other] > 0):
                                candidate = self.best_order([node, other], restarts=1, deadline=deadline)
                                if candidate is not None:
                                    routes[index] = candidate
                                    matched = True
                                    break
                        if matched:
                            placed = True
                            break
        return routes

    def search(self, *, seconds=60.0, seed=0, initial=None, temperature=0.01):
        deadline = time.monotonic() + seconds
        self.rng = random.Random(seed)
        raw = initial if initial else self.balanced_partition()
        groups = [list(group) for group in raw]
        groups += [[] for _ in range(max(0, self.K - len(groups)))]
        routes = self.repair(groups, deadline=deadline)
        current = self.value(routes)
        best_routes = [list(r) for r in routes]
        best_value = current
        iteration = 0
        total = max(1, int(seconds * 20))
        while time.monotonic() < deadline:
            iteration += 1
            served = [(index, node) for index, route in enumerate(routes)
                      for node in (route or [])]
            if not served:
                break
            proposal = [list(r) if r else [] for r in routes]
            kind = self.rng.random()
            if kind < 0.5:
                index, node = self.rng.choice(served)
                target = self.rng.randrange(self.K)
                if target == index:
                    continue
                proposal[index].remove(node)
                proposal[target].append(node)
            elif kind < 0.85:
                index, node = self.rng.choice(served)
                other = None
                pool = [n for n in self.active if not any(n in (r or []) for r in proposal)]
                if pool:
                    other = self.rng.choice(pool)
                    proposal[index].append(other)
                    if not self.rng.random() < 0.5:
                        proposal[index].remove(node)
                else:
                    continue
            else:
                used = [i for i in range(self.K) if proposal[i]]
                if len(used) < 2:
                    continue
                a, b = self.rng.sample(used, 2)
                if not proposal[a] or not proposal[b]:
                    continue
                ia = self.rng.randrange(len(proposal[a]))
                ib = self.rng.randrange(len(proposal[b]))
                proposal[a][ia], proposal[b][ib] = proposal[b][ib], proposal[a][ia]
            new_routes = self.repair(proposal, deadline=deadline)
            value = self.value(new_routes)
            heat = temperature * max(0.0, 1.0 - iteration / total)
            if value >= current or (heat > 0 and self.rng.random()
                                    < math.exp((value - current) / (heat * max(current, 1e-9)))):
                routes, current = new_routes, value
                if value > best_value + 1e-12:
                    best_value = value
                    best_routes = [list(r) for r in new_routes]
        self.routes = best_routes
        return best_routes, best_value
