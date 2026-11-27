"""Restricted TS 2019 supersegment matheuristic for an identical fleet."""

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, pairwise
from random import Random
from time import monotonic

import highspy
import numpy as np

try:
    from ..engine.primal_vendor.allocation import allocate
except ImportError:
    from primal_vendor.allocation import allocate


BASE_LIMIT = 30
POOL_LIMIT = 60
MAX_COMBINE = 4
VERIFY_RESERVE = 0.02


def _exact(values):
    return tuple(Fraction(str(value)) for value in values)


@dataclass(frozen=True)
class _Segment:
    stops: tuple
    sites: frozenset
    duration: Fraction
    deliveries: tuple


class _SegmentPool:
    def __init__(self, instance):
        self.instance = instance
        self.supply = _exact(instance["supply"])
        self.people = _exact(instance["population"])
        self.travel = tuple(map(_exact, instance["travel"]))
        self.service = _exact(instance["service"])
        self.limit = Fraction(str(instance["limit"]))

    def blocks(self, stops, isolated):
        steps = [0, *(i for i in range(1, len(stops))
                      if self.people[stops[i - 1]] and self.supply[stops[i]]), len(stops)]
        if isolated:
            _, amounts = allocate(self.instance, stops)
            boundaries = set(steps)
            cuts = [0]
            load = 0
            for i, (site, quantity) in enumerate(zip(stops, amounts), 1):
                load += quantity if self.supply[site] else -quantity
                if i in boundaries and load == 0:
                    cuts.append(i)
            steps = cuts
        return tuple(stops[left:right] for left, right in pairwise(steps))

    def segment(self, stops, *, require_transfer=False):
        duration = sum((self.service[i] for i in stops), Fraction(0))
        duration += sum(self.travel[i][j] for i, j in pairwise(stops))
        if duration > self.limit:
            return None
        _, amounts = allocate(self.instance, stops)
        if require_transfer:
            load = 0
            for i, (site, quantity) in enumerate(zip(stops, amounts), 1):
                load += quantity if self.supply[site] else -quantity
                if (i < len(stops) and self.people[site]
                        and self.supply[stops[i]] and not load):
                    return None
        return _Segment(stops, frozenset(stops), duration,
                        tuple((i, q) for i, q in zip(stops, amounts) if self.people[i]))

    def build(self, routes, history, rng, deadline):
        """Keep recent blocks and sample novel within-segment deletions/joins."""
        start = tuple(self.blocks(route, True) for route in routes if route)
        pool = {}
        for blocks in start:
            for stops in blocks:
                pool[stops] = self.segment(stops)
        source_routes = dict.fromkeys(route for fleet in history for route in fleet if route)
        source_routes.update(dict.fromkeys(route for route in routes if route))
        base = dict.fromkeys(stops for route in source_routes for stops in self.blocks(route, False))
        base = list(base)[:BASE_LIMIT]
        for _ in range(4 * BASE_LIMIT):
            if not base or len(base) >= BASE_LIMIT or monotonic() >= deadline:
                break
            stops = rng.choice(base)
            valid = [i for i in range(len(stops))
                     if (i or self.supply[stops[1]])
                     and (i < len(stops) - 1 or self.people[stops[-2]])]
            if valid:
                i = rng.choice(valid)
                shorter = stops[:i] + stops[i + 1:]
                if shorter not in base:
                    base.append(shorter)

        for stops in base:
            if len(pool) >= POOL_LIMIT or monotonic() >= deadline:
                break
            if stops not in pool:
                segment = self.segment(stops)
                if segment is not None:
                    pool[stops] = segment

        attempted = set(pool)
        for _ in range(4 * POOL_LIMIT):
            if len(base) < 2 or len(pool) >= POOL_LIMIT or monotonic() >= deadline:
                break
            blocks = rng.sample(base, rng.randint(2, min(MAX_COMBINE, len(base))))
            stops = tuple(i for block in blocks for i in block)
            if stops in attempted or len(stops) != len(set(stops)):
                continue
            attempted.add(stops)
            segment = self.segment(stops, require_transfer=True)
            if segment is not None:
                pool[stops] = segment
        return tuple(pool.values()), start


def _solve_pool(problem, segments, vehicles, seed, deadline, start=(), *, diagnostics=None):
    """Sequence isolated supersegments as at most K directed depot tours."""
    diagnostics = {} if diagnostics is None else diagnostics
    if not segments:
        diagnostics["status"] = "empty_pool"
        return None
    diagnostics["status"] = "model_build_budget"
    if monotonic() >= deadline - VERIFY_RESERVE:
        return None
    m = len(segments)
    depot = m
    limit = problem.limit
    travel = problem.travel
    active = sorted({i for segment in segments for i, _ in segment.deliveries})
    total_population = sum(problem.people)
    weight = {i: float(problem.people[i] / total_population) for i in active}

    arcs = {}
    incoming = [[] for _ in range(m + 1)]
    outgoing = [[] for _ in range(m + 1)]
    durations = {}

    def arc(left, right, duration):
        index = len(arcs)
        arcs[left, right] = index
        outgoing[left].append(index)
        incoming[right].append(index)
        durations[left, right] = duration

    for i, left in enumerate(segments):
        if monotonic() >= deadline - VERIFY_RESERVE:
            return None
        access = travel[0][left.stops[0]] + left.duration
        home = travel[left.stops[-1]][0]
        if access <= limit:
            arc(depot, i, access)
        if left.duration + home <= limit:
            arc(i, depot, home)
        for j, right in enumerate(segments):
            if not left.sites.isdisjoint(right.sites):
                continue
            link = travel[left.stops[-1]][right.stops[0]] + right.duration
            if left.duration + link <= limit:
                arc(i, j, link)
    if not incoming[depot] or not outgoing[depot]:
        diagnostics["status"] = "no_depot_arcs"
        return None

    binary_count = len(arcs)
    arrival = range(binary_count, binary_count + m)
    order = range(arrival.stop, arrival.stop + m)
    delivery = dict(zip(active, range(order.stop, order.stop + len(active))))
    pairs = tuple(combinations(active, 2))
    position = order.stop + len(active)
    n = position + len(pairs)
    lower = np.zeros(n)
    upper = np.full(n, np.inf)
    upper[:binary_count] = 1
    upper[arrival.start:arrival.stop] = float(limit)
    upper[order.start:order.stop] = m - 1
    cost = np.zeros(n)
    # Agencies absent from the pool receive zero. Their penalty with each
    # active agency can be collected directly into that agency's coefficient.
    active_population = sum(problem.people[i] for i in active)
    cost[order.stop:position] = -float(active_population / total_population)
    cost[position:] = 1

    columns, values, row_start = [], [], [0]
    row_lower, row_upper = [], []

    def row(terms, low=-np.inf, high=np.inf):
        for column, value in terms:
            columns.append(column)
            values.append(float(value))
        row_start.append(len(columns))
        row_lower.append(low)
        row_upper.append(high)

    row(((j, 1) for j in outgoing[depot]), high=vehicles)
    row(((j, 1) for j in incoming[depot]), high=vehicles)
    by_site = [[] for _ in problem.people]
    allocations = {i: [] for i in active}
    for s, segment in enumerate(segments):
        row([*((j, 1) for j in incoming[s]), *((j, -1) for j in outgoing[s])], 0, 0)
        for i in sorted(segment.sites):
            by_site[i].extend((j, 1) for j in outgoing[s])
        for i, q in segment.deliveries:
            allocations[i].extend((j, -q) for j in outgoing[s])
    for terms in by_site:
        if terms:
            row(terms, high=1)
    for i, terms in allocations.items():
        row([(delivery[i], 1), *terms], 0, 0)

    for (i, j), x in arcs.items():
        duration = durations[i, j]
        if i == depot:
            row([(arrival[j], 1), (x, -duration)], low=0)
        elif j == depot:
            row([(arrival[i], 1), (x, duration)], high=float(limit))
        else:
            row([(arrival[j], 1), (arrival[i], -1), (x, -(limit + duration))],
                low=-float(limit))
            row([(order[j], 1), (order[i], -1), (x, -m)], low=1 - m)
    for z, (i, j) in enumerate(pairs, position):
        row([(delivery[i], weight[j]), (delivery[j], -weight[i]), (z, -1)], high=0)
        row([(delivery[i], -weight[j]), (delivery[j], weight[i]), (z, -1)], high=0)
    if monotonic() >= deadline - VERIFY_RESERVE:
        return None

    solver = highspy.Highs()
    solver.setOptionValue("output_flag", False)
    solver.setOptionValue("threads", 1)
    solver.setOptionValue("random_seed", seed)
    solver.setOptionValue("mip_feasibility_tolerance", 1e-8)
    solver.addVars(n, lower, upper)
    solver.changeColsCost(n, np.arange(n, dtype=np.int32), cost)
    solver.changeColsIntegrality(binary_count, np.arange(binary_count, dtype=np.int32),
                                 np.ones(binary_count, dtype=np.uint8))
    solver.addRows(len(row_lower), np.asarray(row_lower), np.asarray(row_upper), len(columns),
                   np.asarray(row_start, dtype=np.int32), np.asarray(columns, dtype=np.int32),
                   np.asarray(values, dtype=np.float64))

    keys = {segment.stops: i for i, segment in enumerate(segments)}
    if start:
        warm = np.zeros(n)
        for route in start:
            ids = (depot, *(keys[stops] for stops in route), depot)
            elapsed = 0
            for a, (i, j) in enumerate(pairwise(ids)):
                warm[arcs[i, j]] = 1
                elapsed += durations[i, j]
                if j != depot:
                    warm[arrival[j]] = float(elapsed)
                    warm[order[j]] = a
                    for k, q in segments[j].deliveries:
                        warm[delivery[k]] = float(q)
        for z, (i, j) in enumerate(pairs, position):
            warm[z] = abs(weight[j] * warm[delivery[i]] - weight[i] * warm[delivery[j]])
        solution = highspy.HighsSolution()
        solution.col_value = warm
        solver.setSolution(solution)

    before_solver = monotonic()
    remaining = deadline - before_solver - VERIFY_RESERVE
    if remaining <= 0:
        return None
    solver.setOptionValue("time_limit", remaining)
    status = solver.run()
    diagnostics.update(status=str(solver.getModelStatus()), highs_version=solver.version(),
                       arcs=binary_count, columns=n, rows=len(row_lower),
                       solve_limit_s=remaining, solve_s=monotonic() - before_solver)
    if status == highspy.HighsStatus.kError:
        raise RuntimeError(f"HiGHS failed to solve the supersegment MILP: {solver.getModelStatus()}")
    solution = solver.getSolution()
    diagnostics["highs_incumbent"] = bool(solution.value_valid)
    if not solution.value_valid:
        return None

    chosen = [(i, j) for (i, j), x in arcs.items() if solution.col_value[x] > .5]
    successor = {i: j for i, j in chosen if i != depot}
    used_segments, used_sites, routes = set(), set(), []
    for i, j in chosen:
        if i != depot:
            continue
        stops = []
        elapsed = Fraction(0)
        while j != depot:
            if j in used_segments or j not in successor or used_sites & segments[j].sites:
                return None
            segment = segments[j]
            elapsed += durations[i, j]
            used_segments.add(j)
            used_sites.update(segment.sites)
            stops.extend(segment.stops)
            i, j = j, successor[j]
        if elapsed + durations[i, depot] > limit:
            return None
        routes.append(tuple(stops))
    if len(routes) > vehicles or len(used_segments) + len(routes) != len(chosen):
        return None
    return tuple(routes)


def solve(instance, vehicles, seed, deadline, publish):
    """Alternate the multi-vehicle LNS with the restricted HiGHS sequence model."""
    if __package__:
        from . import mv_lns
    else:
        import mv_lns

    history = deque(maxlen=5)
    best = Fraction(0)
    rng = Random(seed)
    started = monotonic()
    phases = []
    problem = _SegmentPool(instance)

    def record(routes, source):
        nonlocal best
        value = publish(routes, source)
        if value is not None and value > best:
            best = value
            history.appendleft(tuple(tuple(route) for route in routes if route))
        return value

    def improve(routes, phase_deadline):
        phase_deadline = min(deadline, phase_deadline)
        phase_start = monotonic()
        budget = phase_deadline - phase_start
        if budget <= VERIFY_RESERVE:
            return None
        routes = tuple(tuple(route) for route in routes if route)
        pool, start = problem.build(routes, history, rng, phase_start + budget * .25)
        phase = dict(start_s=phase_start - started, budget_s=budget, pool_size=len(pool),
                     pool_s=monotonic() - phase_start)
        candidate = _solve_pool(problem, pool, vehicles, seed, phase_deadline, start,
                                diagnostics=phase)
        previous_best = best
        accepted = False
        if candidate and monotonic() < phase_deadline:
            value = record(candidate, "segment-milp")
            accepted = value is not None and value > previous_best
        phase.update(candidate=bool(candidate), improved=accepted, elapsed_s=monotonic() - phase_start)
        phases.append(phase)
        return candidate if accepted else None

    routes = mv_lns.solve(instance, vehicles, seed, deadline, record, improve=improve)
    return {"routes": routes, "phases": phases}
