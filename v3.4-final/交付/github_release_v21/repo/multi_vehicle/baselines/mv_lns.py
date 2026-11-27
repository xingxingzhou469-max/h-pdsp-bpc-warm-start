"""Multi-vehicle adaptation of Eisenhandler and Tzur (2019), §§5–6."""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import chain, pairwise
from math import exp, lcm
from random import Random
from time import monotonic


NEIGHBORS = 10
RESET_AFTER = 1000
BAD_LIMIT = 50
INITIAL_TEMPERATURE = 100_000


def _number(value):
    return value if isinstance(value, Fraction) else Fraction(str(value))


def _integerize(values):
    values = tuple(map(_number, values))
    scale = lcm(*(value.denominator for value in values))
    return tuple(int(value * scale) for value in values), scale


def _robin_hood(groups, capacity):
    """Exact reverse-suffix water filling; each group is (supply, population)."""
    size = len(groups)
    wealth = [Fraction(0)] * size
    room = [Fraction(0)] * size
    for first in range(size - 1, -1, -1):
        supply, people = groups[first]
        room[first] = Fraction(capacity - supply)
        local_level = Fraction(supply, people)
        following = first + 1
        if following == size or not room[following] or local_level <= wealth[following]:
            wealth[first] = local_level
            continue

        wealth[first] = wealth[following]
        remaining = supply - people * wealth[first]
        while remaining:
            # Only the equal-level prefix before the first full pickup can
            # share this raise. Later recipients use room at every earlier pickup.
            end = following
            while end < size and room[end] and wealth[end] == wealth[first]:
                end += 1
            suffix_people = 0
            by_capacity = None
            for index in range(end - 1, first, -1):
                suffix_people += groups[index][1]
                cap_step = room[index] / suffix_people
                by_capacity = cap_step if by_capacity is None else min(by_capacity, cap_step)
            active_people = people + suffix_people
            step = remaining / active_people
            if by_capacity is not None:
                step = min(step, by_capacity)
            if end < size and room[end]:
                step = min(step, wealth[end] - wealth[first])
            if step <= 0:
                raise ArithmeticError("Robin Hood raise must be positive")
            remaining -= step * active_people
            wealth[first] += step
            for index in range(following, end):
                wealth[index] += step
                room[index] -= step * suffix_people
                suffix_people -= groups[index][1]
    return tuple(wealth)


@dataclass(frozen=True, slots=True)
class _Value:
    segments: tuple
    wealth: tuple
    layers: tuple
    time: int


@dataclass(frozen=True, slots=True)
class _Segment:
    vehicle: int
    first: int
    cut: int
    end: int
    wealth: Fraction


class _Problem:
    def __init__(self, instance):
        quantities, self.quantity_scale = _integerize((instance["capacity"], *instance["supply"]))
        self.capacity, self.supply = quantities[0], quantities[1:]
        self.population, _ = _integerize(instance["population"])
        self.people = sum(self.population)
        self.size = len(self.supply)
        self.pickup = tuple(supply > 0 for supply in self.supply)
        self.nodes = tuple(range(1, self.size))
        self.attractiveness = tuple(supply + people for supply, people in
                                    zip(self.supply, self.population))
        self.removal_weight = tuple(1 / value if value else 0 for value in self.attractiveness)
        travel = tuple(tuple(map(_number, row)) for row in instance["travel"])
        service = tuple(map(_number, instance["service"]))
        ticks, _ = _integerize(chain((instance["limit"],), service, *travel))
        self.limit = ticks[0]
        service_ticks = ticks[1:self.size + 1]
        self.arc = tuple(
            tuple(ticks[1 + self.size + i * self.size + j] + (service_ticks[j] if j else 0)
                  for j in range(self.size))
            for i in range(self.size)
        )
        self.nearest = tuple(
            tuple(sorted(self.nodes, key=row.__getitem__)) for row in travel
        )
        self._levels = lru_cache(maxsize=8192)(_robin_hood)
        self.value = lru_cache(maxsize=16384)(self._value)

    def route_time(self, route):
        return sum(self.arc[left][right] for left, right in pairwise((0, *route, 0))) if route else 0

    def valid_shape(self, route):
        return not route or (self.pickup[route[0]] and not self.pickup[route[-1]])

    def _value(self, route):
        if not self.valid_shape(route):
            return None
        elapsed = self.route_time(route)
        if elapsed > self.limit:
            return None
        segments, groups = [], []
        first = 0
        while first < len(route):
            cut, supply = first, 0
            while cut < len(route) and self.pickup[route[cut]]:
                supply += self.supply[route[cut]]
                cut += 1
            end, people = cut, 0
            while end < len(route) and not self.pickup[route[end]]:
                people += self.population[route[end]]
                end += 1
            segments.append((first, cut, end))
            groups.append((min(supply, self.capacity), people))
            first = end
        levels = self._levels(tuple(groups), self.capacity)
        layers = tuple((rate, people) for rate, (_, people) in zip(levels, groups))
        return _Value(tuple(segments), levels, layers, elapsed)

    def score(self, values):
        """Merge the wealth profiles under the population of the whole instance."""
        layers = sorted(chain.from_iterable(value.layers for value in values), reverse=True)
        numerator, higher = Fraction(0), 0
        for rate, people in layers:
            numerator += rate * (people * (people + 2 * higher))
            higher += people
        return numerator / (self.people * self.quantity_scale)

    def greedy(self, vehicles, deadline):
        remaining = set(self.nodes)
        routes = []
        for _ in range(vehicles):
            route, last, elapsed = [], 0, 0
            while remaining and monotonic() < deadline:
                # A pickup extension leaves a feasible delivery and return.
                # The travel order itself is nearest neighbor as in §6.
                available_deliveries = [
                    node for node in remaining if not self.pickup[node]
                ]
                if not available_deliveries:
                    break
                node = next(
                    (node for node in self.nearest[last] if node in remaining
                     and (route or self.pickup[node])
                     and self.arc[last][node] + elapsed + (
                         min(self.arc[node][agency] + self.arc[agency][0]
                             for agency in available_deliveries)
                         if self.pickup[node] else self.arc[node][0]
                     ) <= self.limit),
                    None,
                )
                if node is None:
                    break
                route.append(node)
                remaining.remove(node)
                elapsed += self.arc[last][node]
                last = node
            # If time expires between selecting a pickup and its delivery,
            # retain the last completed depot-to-delivery prefix.
            while route and self.pickup[route[-1]]:
                remaining.add(route.pop())
            routes.append(tuple(route))
        return tuple(routes)

    def insert(self, route, block, positions, rng):
        elapsed = self.route_time(route)
        block_time = sum(self.arc[a][b] for a, b in pairwise(block))
        first, last = block[0], block[-1]
        feasible = []
        for position in positions:
            if (position == 0 and not self.pickup[first]
                    or position == len(route) and self.pickup[last]):
                continue
            before = route[position - 1] if position else 0
            after = route[position] if position < len(route) else 0
            old_arc = self.arc[before][after] if route else 0
            elapsed_new = elapsed + block_time + self.arc[before][first] + self.arc[last][after] - old_arc
            if elapsed_new <= self.limit:
                feasible.append(position)
        if not feasible:
            return None
        position = rng.choice(feasible)
        return route[:position] + block + route[position:]

    def retime(self, route, segments, deadline):
        """Directed 2-opt and one-stop relocation inside each same-role run."""
        arc, order = self.arc, list(route)
        for first, cut, end in segments:
            for lo, hi in ((first, cut), (cut, end)):
                while hi - lo >= 2 and monotonic() < deadline:
                    change, best_delta = None, 0
                    for left in range(lo, hi):
                        a = order[left - 1] if left else 0
                        u = order[left]
                        b = order[left + 1] if left + 1 < len(order) else 0
                        reversed_delta = 0
                        for right in range(left + 1, hi):
                            v, prev = order[right], order[right - 1]
                            c = order[right + 1] if right + 1 < len(order) else 0
                            reversed_delta += arc[v][prev] - arc[prev][v]
                            delta = reversed_delta + arc[a][v] + arc[u][c] - arc[a][u] - arc[v][c]
                            if delta < best_delta:
                                change, best_delta = (True, left, right), delta
                        drop_delta = arc[a][b] - arc[a][u] - arc[u][b]
                        for target in range(lo, hi + 1):
                            if target in (left, left + 1):
                                continue
                            prev = order[target - 1] if target else 0
                            nxt = order[target] if target < len(order) else 0
                            delta = drop_delta + arc[prev][u] + arc[u][nxt] - arc[prev][nxt]
                            if delta < best_delta:
                                change, best_delta = (False, left, target), delta
                    if change is None:
                        break
                    reverse, left, right = change
                    if reverse:
                        order[left:right + 1] = reversed(order[left:right + 1])
                    else:
                        order.insert(right - (right > left), order.pop(left))
        return tuple(order)


def _weighted(rng, candidates, weight):
    return rng.choices(candidates, weights=[weight(item) for item in candidates])[0]


class _Neighborhood:
    """Station/segment indices are rebuilt only on an accepted fleet."""

    def __init__(self, problem, routes, values, rng):
        self.problem, self.routes, self.rng = problem, routes, rng
        self.stops = tuple((i, j, node) for i, route in enumerate(routes)
                           for j, node in enumerate(route))
        self.stops_by_role = tuple(tuple(stop for stop in self.stops
                                         if problem.pickup[stop[2]] == role)
                                   for role in (False, True))
        self.used = tuple(i for i, route in enumerate(routes) if route)
        self.idle = tuple(i for i, route in enumerate(routes) if not route)
        visited = {node for _, _, node in self.stops}
        self.missing = tuple(node for node in problem.nodes if node not in visited)
        self.missing_role = tuple(tuple(node for node in self.missing if problem.pickup[node] == role)
                                  for role in (False, True))
        self.segments = tuple(_Segment(i, *segment, rate) for i, value in enumerate(values)
                              for segment, rate in zip(value.segments, value.wealth))

    def _available_role(self, groups):
        # Paper weights are conditional on role. The paper does not specify
        # the mixed-role prior for these fleet moves. Use equal probabilities
        # when both roles can move, and condition on the sole available role.
        delivery, pickup = groups
        if delivery and pickup:
            return groups[self.rng.getrandbits(1)]
        return delivery or pickup

    def _visit(self):
        role = self._available_role(self.stops_by_role)
        return _weighted(self.rng, role, lambda stop: self.problem.removal_weight[stop[2]])

    def _new(self, nodes):
        return _weighted(self.rng, nodes, self.problem.attractiveness.__getitem__)

    def _new_mixed(self):
        return self._new(self._available_role(self.missing_role))

    def _transfer(self, first, position, second, positions):
        routes = self.routes
        source = routes[first]
        node = source[position]
        shorter = source[:position] + source[position + 1:]
        if not self.problem.valid_shape(shorter):
            return None
        destination = shorter if first == second else routes[second]
        replacement = self.problem.insert(destination, (node,), positions, self.rng)
        if replacement is None:
            return None
        return {first: shorter, second: replacement}

    def relocate_station(self):
        if not self.stops:
            return None
        vehicle, position, _ = self.rng.choice(self.stops)
        second = self.rng.randrange(len(self.routes))
        size = len(self.routes[second]) - (vehicle == second)
        return self._transfer(vehicle, position, second, range(size + 1))

    def _swap(self, first, pos_a, second, pos_b):
        a, b = self.routes[first], self.routes[second]
        node_a, node_b = a[pos_a], b[pos_b]
        if first == second:
            route = list(a)
            route[pos_a], route[pos_b] = node_b, node_a
            return {first: tuple(route)}
        return {first: a[:pos_a] + (node_b,) + a[pos_a + 1:],
                second: b[:pos_b] + (node_a,) + b[pos_b + 1:]}

    def swap_stations(self):
        if len(self.stops) < 2:
            return None
        (first, a, _), (second, b, _) = self.rng.sample(self.stops, 2)
        return self._swap(first, a, second, b)

    def reverse_subsequence(self):
        if not self.used:
            return None
        vehicle = self.rng.choice(self.used)
        route = self.routes[vehicle]
        lo, hi = sorted(self.rng.sample(range(len(route)), 2))
        return {vehicle: route[:lo] + route[lo:hi + 1][::-1] + route[hi + 1:]}

    def insert_station(self):
        if not self.missing:
            return None
        vehicle = self.rng.randrange(len(self.routes))
        route = self.routes[vehicle]
        if route:
            block = (self._new_mixed(),)
        elif all(self.missing_role):
            block = (self._new(self.missing_role[1]), self._new(self.missing_role[0]))
        else:
            return None
        result = self.problem.insert(route, block, range(len(route) + 1), self.rng)
        return None if result is None else {vehicle: result}

    def remove_station(self):
        if not self.stops:
            return None
        vehicle, position, _ = self._visit()
        route = self.routes[vehicle]
        return {vehicle: route[:position] + route[position + 1:]}

    def replace_station(self):
        if not self.stops:
            return None
        vehicle, position, node = self._visit()
        choices = self.missing_role[self.problem.pickup[node]]
        if not choices:
            return None
        route = self.routes[vehicle]
        return {vehicle: route[:position] + (self._new(choices),) + route[position + 1:]}

    def _rich_poor(self):
        if len(self.segments) < 2:
            return None
        # Draw inexpensive biased order statistics. Equal levels still allow
        # a perturbation, with a random assignment of rich and poor.
        first, second = self.rng.sample(self.segments, 2)
        return (first, second) if first.wealth >= second.wealth else (second, first)

    def _position(self, segment, pickup, *, remove=False):
        start, end = ((segment.first, segment.cut) if pickup else (segment.cut, segment.end))
        route = self.routes[segment.vehicle]
        weights = self.problem.removal_weight if remove else self.problem.attractiveness
        return _weighted(self.rng, range(start, end), lambda position: weights[route[position]])

    def _wealth_transfer(self, source, destination, pickup):
        position = self._position(source, pickup)
        start, end = ((destination.first, destination.cut) if pickup
                      else (destination.cut, destination.end))
        offset = source.vehicle == destination.vehicle and position < start
        return self._transfer(source.vehicle, position, destination.vehicle,
                              range(start - offset, end - offset + 1))

    def move_poor_delivery_to_rich(self):
        pair = self._rich_poor()
        if pair is None:
            return None
        rich, poor = pair
        return self._wealth_transfer(poor, rich, False)

    def swap_poor_rich_deliveries(self):
        pair = self._rich_poor()
        if pair is None:
            return None
        rich, poor = pair
        return self._swap(poor.vehicle, self._position(poor, False),
                          rich.vehicle, self._position(rich, False, remove=True))

    def insert_at_wealth_segment(self):
        if not self.segments or not self.missing:
            return None
        node = self._new_mixed()
        pickup = self.problem.pickup[node]
        first, second = self.rng.choices(self.segments, k=2)
        chooser = min if pickup else max
        segment = chooser((first, second), key=lambda item: item.wealth)
        route = self.routes[segment.vehicle]
        positions = (range(segment.first, segment.cut + 1) if pickup
                     else range(segment.cut, segment.end + 1))
        result = self.problem.insert(route, (node,), positions, self.rng)
        return None if result is None else {segment.vehicle: result}

    def remove_from_wealth_segment(self):
        if not self.segments:
            return None
        pickup = bool(self.rng.getrandbits(1))
        first, second = self.rng.choices(self.segments, k=2)
        chooser = max if pickup else min
        segment = chooser((first, second), key=lambda item: item.wealth)
        vehicle = segment.vehicle
        position = self._position(segment, pickup, remove=True)
        route = self.routes[vehicle]
        return {vehicle: route[:position] + route[position + 1:]}

    def move_rich_pickup_to_poor(self):
        pair = self._rich_poor()
        if pair is None:
            return None
        rich, poor = pair
        return self._wealth_transfer(rich, poor, True)

    def move_poor_segment_after_rich(self):
        pair = self._rich_poor()
        if pair is None:
            return None
        rich, poor = pair
        source = self.routes[poor.vehicle]
        block = source[poor.first:poor.end]
        shorter = source[:poor.first] + source[poor.end:]
        # A whole segment can also activate an idle vehicle.
        if shorter and self.idle and self.rng.randrange(4) == 0:
            return {poor.vehicle: shorter, self.rng.choice(self.idle): block}
        same = rich.vehicle == poor.vehicle
        target = shorter if same else self.routes[rich.vehicle]
        position = rich.end - (len(block) if same and poor.first < rich.first else 0)
        return {poor.vehicle: shorter, rich.vehicle: target[:position] + block + target[position:]}


OPERATORS = (
    _Neighborhood.relocate_station,
    _Neighborhood.swap_stations,
    _Neighborhood.reverse_subsequence,
    _Neighborhood.insert_station,
    _Neighborhood.remove_station,
    _Neighborhood.replace_station,
    _Neighborhood.move_poor_delivery_to_rich,
    _Neighborhood.swap_poor_rich_deliveries,
    _Neighborhood.insert_at_wealth_segment,
    _Neighborhood.remove_from_wealth_segment,
    _Neighborhood.move_rich_pickup_to_poor,
    _Neighborhood.move_poor_segment_after_rich,
)


def _evaluate(problem, routes):
    values = tuple(map(problem.value, routes))
    if any(value is None for value in values):
        return None
    return problem.score(values), -sum(value.time for value in values), routes, values


def _retime(problem, state, deadline):
    score, _, routes, values = state
    paths = tuple(problem.retime(route, value.segments, deadline)
                  for route, value in zip(routes, values))
    if paths == routes:
        return state
    tightened = _evaluate(problem, paths)
    if tightened is None or tightened[0] != score or tightened[1] <= state[1]:
        raise ArithmeticError("intra-segment routing must shorten time and preserve the allocation")
    return tightened


def _checked_publish(state, source, publish):
    score, _, routes, _ = state
    checked = publish(routes, source)
    if checked != score:
        raise ArithmeticError(f"published fleet scores {checked}, expected {score}")


def solve(instance, vehicles, seed, deadline, publish, *, improve=None):
    """Search to a monotonic deadline and return the best routes, depot implicit.

    The publish callback independently verifies a candidate and returns its
    exact score. In the hybrid arm, improve publishes its own witness and
    returns its candidate routes (or None) before its subdeadline.
    """
    start = monotonic()
    if vehicles < 1 or start >= deadline:
        return ()
    problem = _Problem(instance)
    if not problem.people:
        return ((),) * vehicles
    rng = Random(seed)
    best = _evaluate(problem, problem.greedy(vehicles, deadline))
    if best[0] > 0 and monotonic() < deadline:
        _checked_publish(best, "lns-nearest-neighbor", publish)
    best = _retime(problem, best, deadline)
    current = best
    neighborhood = _Neighborhood(problem, *current[2:], rng)
    iteration = stagnant = bad = 0
    budget = deadline - start
    lns_interval = min(4.0, max(0.20, budget / 12))
    improve_slice = min(6.0, max(0.25, budget / 10))
    next_improvement = start + min(1.0, lns_interval)

    while (now := monotonic()) < deadline:
        if improve is not None and now >= next_improvement and deadline - now >= 0.08:
            candidate = improve(best[2], min(deadline, now + improve_slice))
            if candidate is not None:
                routes = tuple(map(tuple, candidate))
                stops = tuple(chain.from_iterable(routes))
                if (len(routes) > vehicles or len(set(stops)) != len(stops)
                        or any(type(node) is not int or not 0 < node < problem.size for node in stops)):
                    raise ValueError("segment improvement returned invalid fleet routes")
                improved = _evaluate(problem, routes + ((),) * (vehicles - len(routes)))
                if improved is None:
                    raise ValueError("segment improvement returned infeasible fleet routes")
                if improved[:2] > best[:2]:
                    best = current = _retime(problem, improved, deadline)
                    neighborhood = _Neighborhood(problem, *current[2:], rng)
                    stagnant = bad = 0
            next_improvement = monotonic() + lns_interval
            continue

        iteration += 1
        neighbor = None
        for _ in range(NEIGHBORS):
            if monotonic() >= deadline:
                break
            changes = rng.choice(OPERATORS)(neighborhood)
            if changes is None:
                continue
            candidate = tuple(changes.get(i, route) for i, route in enumerate(current[2]))
            if candidate == current[2]:
                continue
            scored = _evaluate(problem, candidate)
            if scored is not None and (neighbor is None or scored[:2] > neighbor[:2]):
                neighbor = scored

        stagnant += 1
        if neighbor is None or 5 * neighbor[0] < 4 * best[0]:
            bad += 1
        else:
            bad = 0
        if neighbor is not None:
            if neighbor[:2] > best[:2]:
                strict = neighbor[0] > best[0]
                best = neighbor
                if strict:
                    if monotonic() < deadline:
                        _checked_publish(best, "lns-neighborhood", publish)
                    stagnant = bad = 0
                best = _retime(problem, best, deadline)
                neighbor = best
            delta = neighbor[0] - current[0]
            if delta >= 0 or rng.random() < exp(float(delta) * iteration / INITIAL_TEMPERATURE):
                current = neighbor
                neighborhood = _Neighborhood(problem, *current[2:], rng)
        if stagnant >= RESET_AFTER or bad >= BAD_LIMIT:
            current = best
            neighborhood = _Neighborhood(problem, *current[2:], rng)
            stagnant = bad = 0
        elif iteration % 100 == 0:
            current = _retime(problem, current, deadline)
            neighborhood = _Neighborhood(problem, *current[2:], rng)
    return best[2]
