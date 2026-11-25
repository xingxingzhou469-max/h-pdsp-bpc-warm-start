"""Independent, exact verifier for ng-product-v1 certificates."""

import argparse
import json
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from heapq import heappop, heappush
from itertools import combinations
from math import ceil, gcd, lcm
from pathlib import Path


def _number(value):
    if type(value) is bool:
        raise ValueError("a Boolean is not a model number")
    if isinstance(value, (int, Fraction)):
        return Fraction(value)
    return Fraction(str(value))


def _rational(value):
    if not isinstance(value, str):
        raise ValueError("certificate rationals must be strings")
    return Fraction(value)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _rational_gcd(values):
    denominator = lcm(*(value.denominator for value in values))
    return Fraction(gcd(*(int(value * denominator) for value in values)), denominator)


@dataclass(frozen=True)
class _Model:
    nodes: tuple
    suppliers: tuple
    caps: tuple
    pops: tuple
    capacity: Fraction
    limit: Fraction
    increments: tuple
    returns: tuple
    total_population: Fraction


def _derive(instance):
    supply, population, service = (
        tuple(map(_number, instance[key])) for key in ("supply", "population", "service")
    )
    travel = tuple(tuple(map(_number, row)) for row in instance["travel"])
    capacity, limit = (_number(instance[key]) for key in ("capacity", "limit"))
    size = len(supply)
    _require(size > 0 and len(population) == len(service) == len(travel) == size
             and all(len(row) == size for row in travel), "inconsistent instance dimensions")
    _require(min(capacity, limit, *service, *supply, *population) >= 0
             and all(min(row) >= 0 for row in travel), "negative instance quantity or time")
    _require(supply[0] == population[0] == 0, "the depot cannot supply goods or have demand")
    _require(all((s > 0) != (p > 0) for s, p in zip(supply[1:], population[1:])),
             "every non-depot station must be a supplier or agency")
    nodes = tuple(range(1, size))
    suppliers = tuple(supply[i] > 0 for i in nodes)
    caps = tuple(min(supply[i], capacity) for i in nodes)
    pops = tuple(population[i] for i in nodes)
    increments = tuple(tuple(travel[i][j] + service[j] for j in nodes) for i in (*nodes, 0))
    returns = tuple(travel[i][0] for i in nodes)
    total = sum((p for p in population[1:] if p > 0), Fraction(0))
    return _Model(nodes, suppliers, caps, pops, capacity, limit, increments, returns, total)


@dataclass(frozen=True)
class _Projection:
    horizon: int
    arcs: tuple
    home: tuple
    layered: bool
    memory: tuple | None
    cycle2: bool
    ordered: bool
    suppliers: tuple
    pickup_caps: tuple
    load_limit: int


def _projection(model, vehicles, support):
    time_unit = _rational(support["time_unit"])
    load_unit = _rational(support["load_unit"])
    _require(time_unit > 0 and load_unit > 0, "projection units must be positive")
    mode = support["mode"]
    _require(mode in ("pooled", "per_vehicle"), "unknown time mode")
    pool_budget = vehicles * model.limit // time_unit
    horizon = pool_budget if mode == "pooled" else model.limit // time_unit
    _require(type(support["horizon"]) is int and support["horizon"] == horizon,
             "projected horizon does not match the raw input")
    cycle2 = support["cycle2"]
    ordered = support["ordered"]
    _require(type(cycle2) is bool and type(ordered) is bool, "invalid pricing flags")
    arcs = tuple(tuple(value // time_unit for value in row) for row in model.increments)
    home = tuple(value // time_unit for value in model.returns)
    layered = any(value == 0 for i, row in enumerate(arcs[:-1])
                  for j, value in enumerate(row) if i != j)
    memory = None
    if support["ng_sets"] is not None:
        ng_sets = support["ng_sets"]
        raw_to_bit = {raw: 1 << i for i, raw in enumerate(model.nodes)}
        _require(isinstance(ng_sets, (list, tuple)) and len(ng_sets) == len(model.nodes),
                 "ng sets must be aligned with eligible station IDs")
        memory = []
        for node, group in zip(model.nodes, ng_sets):
            _require(isinstance(group, (list, tuple))
                     and all(type(raw) is int and raw in raw_to_bit for raw in group)
                     and node in group and len(group) == len(set(group)), "invalid ng set")
            memory.append(sum(raw_to_bit[raw] for raw in group))
        memory = tuple(memory)
    pickup_caps = tuple(ceil(cap / load_unit) for cap in model.caps) if ordered else ()
    load_limit = ceil(model.capacity / load_unit) if ordered else 0
    return (_Projection(horizon, arcs, home, layered, memory, cycle2, ordered,
                        model.suppliers, pickup_caps, load_limit), pool_budget, load_unit)


def _initial(projection, to_home):
    for node, supplier in enumerate(projection.suppliers):
        if supplier and projection.arcs[-1][node] + to_home[node] <= projection.horizon:
            mask = 0 if projection.memory is None else 1 << node
            yield projection.arcs[-1][node], (node, mask)


def _time_to_exit(projection):
    """Lower bound on remaining time, ignoring memory, load and the visit count."""
    beyond = projection.horizon + 1
    distance = [beyond if supplier else min(beyond, home)
                for supplier, home in zip(projection.suppliers, projection.home)]
    pending = set(range(len(distance)))
    while pending:
        target = min(pending, key=distance.__getitem__)
        best = distance[target]
        if best == beyond:
            break
        pending.remove(target)
        for node in pending:
            distance[node] = min(distance[node], projection.arcs[node][target] + best)
    return distance


class _Successors:
    def __init__(self, projection):
        self.memory = projection.memory
        self.to_home = _time_to_exit(projection)
        self.by_station = [sorted((duration + self.to_home[target], duration, target, 1 << target)
                                  for target, duration in enumerate(row) if target != node)
                           for node, row in enumerate(projection.arcs[:-1])]
        self.by_state = {}

    def __getitem__(self, state):
        if state not in self.by_state:
            node, mask = state
            self.by_state[state] = tuple(
                (remaining, duration, target, (target, 0 if self.memory is None
                                               else (mask & self.memory[target]) | bit))
                for remaining, duration, target, bit in self.by_station[node] if not mask & bit
            )
        return self.by_state[state]


class _ClockBuckets:
    def __init__(self):
        self.pending = {}
        self.times = []

    def __getitem__(self, clock):
        if clock not in self.pending:
            heappush(self.times, clock)
            self.pending[clock] = {}
        return self.pending[clock]

    def __iter__(self):
        while self.times:
            clock = heappop(self.times)
            yield clock, self.pending.pop(clock)


def _merge_label(current, score, previous):
    # Only the best score from a different predecessor can help when the
    # best predecessor is the otherwise admissible next stop.
    if current is None:
        return score, previous, None
    best, best_previous, alternate = current
    if previous == best_previous:
        if score > best:
            return score, previous, alternate
    elif score > best:
        return score, previous, best
    elif alternate is None or score > alternate:
        return best, best_previous, score
    return current


def _unordered(projection, reward, tau, marker_bits=()):
    horizon = projection.horizon
    successors = _Successors(projection)
    clocks = _ClockBuckets()
    for clock, state in _initial(projection, successors.to_home):
        clocks[clock][state] = (reward[state[0]], -1, None)
    optimum = 0
    classes = [0] + [None] * ((1 << len(marker_bits)) - 1) if marker_bits else None
    layers = len(reward) if projection.layered else 1
    for layer in range(layers):
        next_clocks = _ClockBuckets() if projection.layered else clocks
        for clock, states in clocks:
            for state, (value, previous, alternate) in states.items():
                node, _ = state
                finish = clock + projection.home[node]
                if not projection.suppliers[node] and finish <= horizon:
                    score = value - tau * finish
                    optimum = max(optimum, score)
                    if classes is not None:
                        signature = sum(1 << bit for bit, flag in enumerate(marker_bits) if state[1] & flag)
                        if classes[signature] is None or score > classes[signature]:
                            classes[signature] = score
                if projection.layered and layer + 1 == layers:
                    continue
                predecessor = node if projection.cycle2 else -1
                for remaining, duration, target, next_state in successors[state]:
                    if clock + remaining > horizon:
                        break
                    score = alternate if target == previous else value
                    if score is not None:
                        bucket = next_clocks[clock + duration]
                        bucket[next_state] = _merge_label(
                            bucket.get(next_state), score + reward[target], predecessor
                        )
        clocks = next_clocks
    return optimum, classes


def _process_load(labels, pickup, load_reward, excluded_previous):
    """Maximize over the admissible arrival loads for each departure load."""
    scores = [None if label is None else label[2] if label[1] == excluded_previous
              else label[0] for label in labels]
    if pickup is not None:
        window = deque()
        for load, score in enumerate(scores):
            while window and window[0][0] < load - pickup:
                window.popleft()
            if score is not None:
                while window and window[-1][1] <= score:
                    window.pop()
                window.append((load, score))
            scores[load] = window[0][1] if window else None
    else:
        best = None
        for load in range(len(scores) - 1, -1, -1):
            score = scores[load]
            if score is not None:
                adjusted = score + load_reward * load
                if best is None or adjusted > best:
                    best = adjusted
            scores[load] = None if best is None else best - load_reward * load
    return scores


def _ordered(projection, reward, load_reward, tau):
    horizon = projection.horizon
    successors = _Successors(projection)
    load_count = projection.load_limit + 1
    clocks = _ClockBuckets()
    for clock, state in _initial(projection, successors.to_home):
        clocks[clock][state] = [(0, -1, None)] + [None] * (load_count - 1)
    optimum = 0
    layers = len(reward) if projection.layered else 1
    for layer in range(layers):
        next_clocks = _ClockBuckets() if projection.layered else clocks
        for clock, states in clocks:
            for state, labels in states.items():
                node, _ = state
                pickup = projection.pickup_caps[node] if projection.suppliers[node] else None
                finish = clock + projection.home[node]
                processed = _process_load(labels, pickup, load_reward, None)
                if pickup is None and finish <= horizon and processed[0] is not None:
                    optimum = max(optimum, processed[0] + reward[node] - tau * finish)
                if projection.layered and layer + 1 == layers:
                    continue
                excluded = ({entry[1] for entry in labels if entry is not None}
                            if projection.cycle2 else ())
                predecessor = node if projection.cycle2 else -1
                for remaining, duration, target, next_state in successors[state]:
                    if clock + remaining > horizon:
                        break
                    output = (_process_load(labels, pickup, load_reward, target)
                              if target in excluded else processed)
                    bucket = next_clocks[clock + duration]
                    if next_state not in bucket:
                        bucket[next_state] = [None] * load_count
                    target_loads = bucket[next_state]
                    for load, score in enumerate(output):
                        if score is not None:
                            target_loads[load] = _merge_label(
                                target_loads[load], score + reward[node], predecessor
                            )
        clocks = next_clocks
    return optimum


@lru_cache(maxsize=256)
def _route_value(projection, reward, load_reward, tau, marker_bits):
    scale = lcm(*(amount.denominator for amount in (*reward, load_reward, tau)))
    node_values = tuple(int(amount * scale) for amount in reward)
    time_value = int(tau * scale)
    if projection.ordered:
        best = _ordered(projection, node_values, int(load_reward * scale), time_value)
        classes = None
    else:
        best, classes = _unordered(projection, node_values, time_value, marker_bits)
    return (Fraction(best, scale), None if classes is None else
            tuple(None if x is None else Fraction(x, scale) for x in classes))


def _check_witness(model, projection, reward, load_reward, tau, best, witness):
    if witness is None:
        _require(best == 0, "idle witness has the wrong objective")
        return
    route = witness["route"]
    raw_to_node = {raw: idx for idx, raw in enumerate(model.nodes)}
    _require(isinstance(route, (list, tuple)) and route
             and all(type(raw) is int and raw in raw_to_node for raw in route), "invalid witness route")
    route = [raw_to_node[raw] for raw in route]
    _require(projection.suppliers[route[0]] and not projection.suppliers[route[-1]],
             "witness must start at a supplier and finish at an agency")
    _require(not projection.layered or len(route) <= len(model.nodes), "witness exceeds visit limit")
    amounts = witness["amounts"]
    if projection.ordered:
        _require(isinstance(amounts, (list, tuple)) and len(amounts) == len(route)
                 and all(type(x) is int and x >= 0 for x in amounts), "invalid witness load ticks")
    else:
        _require(amounts is None, "unordered witness must have null amounts")
    previous = len(model.nodes)
    previous_previous = -1
    mask = 0
    clock = 0
    load = 0
    value = Fraction(0)
    for index, node in enumerate(route):
        _require(node != previous and (not projection.cycle2 or node != previous_previous),
                 "witness uses a forbidden arc")
        if projection.memory is not None:
            bit = 1 << node
            _require(not mask & bit, "witness revisits a station in ng memory")
            mask = (mask & projection.memory[node]) | bit
        clock += projection.arcs[previous][node]
        value += reward[node]
        if projection.ordered:
            amount = amounts[index]
            if projection.suppliers[node]:
                _require(amount <= projection.pickup_caps[node], "witness exceeds pickup capacity")
                load += amount
            else:
                load -= amount
                value += amount * load_reward
            _require(0 <= load <= projection.load_limit, "witness violates vehicle load bounds")
        previous_previous, previous = previous, node
    _require(load == 0, "witness ends with residual load")
    clock += projection.home[route[-1]]
    _require(type(witness["time_ticks"]) is int and witness["time_ticks"] == clock
             and clock <= projection.horizon, "invalid witness travel time")
    _require(value - clock * tau == best, "witness does not attain the replayed route value")


def _replay_details(model, vehicles, support):
    projection, pool_budget, load_unit = _projection(model, vehicles, support)
    alpha, beta, tau = (_rational(support[key]) for key in ("alpha", "beta", "time_price"))
    prices = tuple(map(_rational, support["visit_prices"]))
    _require(alpha > 0 and beta > 0 and tau >= 0 and len(prices) == len(model.nodes)
             and all(price >= 0 for price in prices), "invalid support prices")
    supply_unit = support.get("supply_unit")
    if projection.ordered:
        _require(supply_unit is None, "ordered support must use its load grid")
        supply_credit = (0,) * len(model.nodes)
    elif supply_unit is None:
        supply_credit = model.caps
    else:
        step = _rational(supply_unit)
        _require(step > 0, "supplier credit unit must be positive")
        supply_credit = tuple(ceil(cap / step) * step for cap in model.caps)
    reward = tuple(alpha * quantity + beta * pop - price
                   for quantity, pop, price in zip(supply_credit, model.pops, prices))
    load_reward = alpha * load_unit if projection.ordered else Fraction(0)
    marker_bits = ()
    markers = support.get("markers")
    if markers is not None:
        _require(isinstance(markers, (tuple, list)) and 0 < len(markers) <= 8
                 and all(type(n) is int and n in model.nodes for n in markers)
                 and list(markers) == sorted(set(markers)) and not projection.ordered
                 and projection.memory is not None, "invalid marked support")
        marker_bits = tuple(1 << (n - 1) for n in markers)
        full = sum(marker_bits)
        _require(all(mask & full == full for mask in projection.memory), "marker is not permanent")
    best, classes = _route_value(projection, reward, load_reward, tau, marker_bits)
    if markers is not None:
        reported = tuple(None if x is None else _rational(x) for x in support["class_values"])
        _require(reported == classes, "marked route class values do not reproduce")
    else:
        _require("class_values" not in support, "unmarked support has route class values")
    linear = sum(prices, Fraction(0)) + vehicles * best + pool_budget * tau
    _require(best == _rational(support["route_value"]), "route value does not reproduce")
    _require(linear == _rational(support["linear_support"]), "linear support does not reproduce")
    if "witness" in support:
        _check_witness(model, projection, reward, load_reward, tau, best, support["witness"])
    return (alpha, beta, linear), classes


def _verify_support(model, vehicles, support):
    return _replay_details(model, vehicles, support)[0]


def _check_vehicles(instance, vehicles):
    _require(type(vehicles) is int and vehicles >= 0, "invalid vehicle count")
    if "vehicles" in instance:
        raw = instance["vehicles"]
        _require(type(raw) is int and raw == vehicles, "fleet size differs from raw input")


def replay_support(instance: dict, vehicles: int, support: dict) -> tuple[Fraction, Fraction, Fraction]:
    """Verify one support and return its (alpha, beta, linear support)."""
    _check_vehicles(instance, vehicles)
    return _verify_support(_derive(instance), vehicles, support)


def replay_marked(instance, vehicles, support):
    """Return exact maxima by permanent visit signature, including the idle class."""
    _check_vehicles(instance, vehicles)
    _, classes = _replay_details(_derive(instance), vehicles, support)
    _require(classes is not None, "expected a marked support")
    return classes


def _frontier(x_cap, y_cap, inequalities):
    end = min(x_cap, min((support / a for a, _, support in inequalities), default=x_cap))
    lines = [(Fraction(0), y_cap)]
    lines.extend((a / b, support / b) for a, b, support in inequalities)
    return end, lines


def _maximize_frontier(end, lines, step):
    # On each active segment the product is a concave quadratic. Endpoints
    # and stationary points also locate the adjacent candidates on one grid.
    critical = {Fraction(0), end}
    for slope, offset in lines:
        if slope:
            critical.add(offset / (2 * slope))
    for (m1, c1), (m2, c2) in combinations(lines, 2):
        if m1 != m2:
            critical.add((c1 - c2) / (m1 - m2))
    candidates = {min(end, max(Fraction(0), value)) for value in critical}
    if step is not None:
        candidates = {tick * step for value in candidates
                      for tick in (value // step, ceil(value / step)) if tick * step <= end}
    return max(x * min(offset - slope * x for slope, offset in lines) for x in candidates)


def _two_grids(q_cap, p_cap, inequalities, q_step, p_step):
    q_count, p_count = q_cap // q_step, p_cap // p_step
    if q_count <= p_count:
        width, height = q_count, p_count
        lines = [(a * q_step, b * p_step, support) for a, b, support in inequalities]
    else:
        width, height = p_count, q_count
        lines = [(b * p_step, a * q_step, support) for a, b, support in inequalities]
    exact = []
    for a, b, support in lines:
        scale = lcm(a.denominator, b.denominator, support.denominator)
        exact.append((int(a * scale), int(b * scale), int(support * scale)))
    if not exact:
        return width * height * q_step * p_step
    width = min(width, min(support // a for a, _, support in exact))
    best = 0
    for i in range(1, width + 1):
        j = height
        for a, b, support in exact:
            j = min(j, (support - a * i) // b)
        if j == 0:
            break
        best = max(best, i * j)
    return best * q_step * p_step


def _check_step(value, expected, active, name):
    if value is None:
        _require(not active or expected == 0, f"missing {name} step")
        return None
    step = _rational(value)
    _require(step == expected, f"{name} step does not match the raw input gcd")
    return step


def replay(instance: dict, vehicles: int, certificate: dict) -> Fraction:
    """Return the certifiable product bound after exact replay of all fields."""
    _check_vehicles(instance, vehicles)
    _require(certificate["method"] == "ng-product-v1"
             and type(certificate["vehicles"]) is int and certificate["vehicles"] == vehicles,
             "certificate method or vehicle count mismatch")
    model = _derive(instance)
    q_cap = min(sum(model.caps, Fraction(0)), model.capacity * sum(p > 0 for p in model.pops))
    p_cap = sum(model.pops, Fraction(0))
    _require(q_cap == _rational(certificate["quantity_cap"])
             and p_cap == _rational(certificate["p_cap"]), "caps do not reproduce")
    grid = certificate["grid"]
    _require(grid in ("continuous", "population", "both"), "unknown product grid")
    p_step = _check_step(certificate["p_step"], _rational_gcd(model.pops),
                         grid in ("population", "both"), "population")
    q_step = _check_step(certificate["q_step"], _rational_gcd((model.capacity, *model.caps)),
                         grid == "both", "quantity")
    _require(isinstance(certificate["supports"], list), "supports must be an array")
    inequalities = [_verify_support(model, vehicles, support) for support in certificate["supports"]]
    if not q_cap or not p_cap or not vehicles:
        bound = Fraction(0)
    elif grid == "both":
        bound = _two_grids(q_cap, p_cap, inequalities, q_step, p_step) / model.total_population
    elif grid == "population":
        swapped = [(b, a, support) for a, b, support in inequalities]
        bound = _maximize_frontier(*_frontier(p_cap, q_cap, swapped), p_step) / model.total_population
    else:
        bound = _maximize_frontier(*_frontier(q_cap, p_cap, inequalities), None) / model.total_population
    _require(bound == _rational(certificate["upper"]), "product bound does not reproduce")
    return bound


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--event", required=True, type=Path)
    args = parser.parse_args()
    try:
        raw = args.input.read_bytes()
        instance = json.loads(raw, parse_float=Fraction)
        event = json.loads(args.event.read_text(encoding="utf-8"), parse_float=Fraction)
        certificate = event["certificate"]
        input_hash = sha256(raw).hexdigest()
        for record in (event, certificate):
            if "input_sha256" in record:
                _require(record["input_sha256"] == input_hash, "input SHA-256 mismatch")
        vehicles = event.get("vehicles", certificate["vehicles"])
        if "vehicles" in instance:
            _require(vehicles == instance["vehicles"], "event fleet size differs from raw input")
        bound = replay(instance, vehicles, certificate)
        _require(_number(event["score"]) == bound, "event score does not reproduce")
    except (KeyError, OSError, TypeError, ValueError, ZeroDivisionError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps({"ok": True, "upper": str(bound), "input_sha256": input_hash}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
