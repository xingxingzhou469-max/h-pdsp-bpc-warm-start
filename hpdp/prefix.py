"""Exact branch and bound on route prefixes with integer resource bounds."""

from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
from heapq import heappop, heappush
from itertools import accumulate, count
from math import lcm
from time import monotonic

from .allocation import allocate
from .check import check


def scaled(numbers):
    """Choose a common integer unit for the decimal or rational input as written."""
    numbers = [x if isinstance(x, int) else Fraction(str(x)) for x in numbers]
    scale = lcm(*(x.denominator for x in numbers))
    return [int(x * scale) for x in numbers], scale


class Resources:
    def __init__(self, instance):
        self.instance = instance
        n = len(instance["service"])
        self.nodes = range(1, n)
        times, self.time_scale = scaled(
            [x for row in instance["travel"] for x in row]
            + instance["service"] + [instance["limit"]]
        )
        self.travel = [times[i*n:(i+1)*n] for i in range(n)]
        self.service = times[n*n:-1]
        self.limit = times[-1]
        amounts, self.flow_scale = scaled(instance["supply"] + [instance["capacity"]])
        self.capacity = amounts[-1]
        self.supply = [min(self.capacity, s) for s in amounts[:-1]]
        self.pop, _ = scaled(instance["population"])
        self.people = sum(self.pop)
        self.units = self.people * self.flow_scale
        self.p = tuple(i for i in self.nodes if self.supply[i])
        self.d = tuple(i for i in self.nodes if self.pop[i])
        self.prior = min(sum(self.supply), len(self.d)*self.capacity)*self.people

    def prepare_bounds(self):
        # Cost of the arc includes service at its destination. Repeated visits
        # and the depot may be used by the relaxation only.
        distances = [[0 if i == j else t + self.service[j] for j, t in enumerate(row)]
                     for i, row in enumerate(self.travel)]
        for k, middle in enumerate(distances):
            for row in distances:
                step = row[k]
                for j, tail in enumerate(middle):
                    if step + tail < row[j]:
                        row[j] = step + tail
        self.dist = distances
        n = len(distances)
        # Two feasible sets of directed assignment dual potentials. Each
        # potential pair charges the actual first and last ends exactly once.
        def potentials(matrix):
            a = [min(t for j, t in enumerate(row) if i != j) for i, row in enumerate(matrix)]
            b = [min(matrix[i][j] - a[i] for i in range(n) if i != j) for j in range(n)]
            return a, b
        a, b = potentials(self.travel)
        rev_a, rev_b = potentials(list(zip(*self.travel)))
        self.charges = [(u, v, [s+x+y for s, x, y in zip(self.service, u, v)])
                        for u, v in ((a, b), (rev_b, rev_a))]


@dataclass(frozen=True, slots=True)
class Label:
    route: tuple = ()
    mask: int = 0
    last: int = 0
    elapsed: int = 0
    unloaded: int = 0  # maximum total unloaded if emptying at every delivery
    onboard: int = 0   # supply gathered since that last emptying
    blocks: tuple = ()  # alternating capped supply and aggregate population

    def extend(self, r, v):
        t = self.elapsed + r.travel[self.last][v] + r.service[v]
        load, unloaded = self.onboard, self.unloaded
        blocks = self.blocks
        if r.supply[v]:
            load = min(r.capacity, load + r.supply[v])
            if len(blocks) % 2:
                blocks = blocks[:-1] + (load,)
            else:
                blocks += (load,)
        else:
            unloaded += load
            load = 0
            if len(blocks) % 2:
                blocks += (r.pop[v],)
            else:
                blocks = blocks[:-1] + (blocks[-1]+r.pop[v],)
        return Label(self.route + (v,), self.mask | (1 << v), v, t, unloaded, load, blocks)


def box_bound(pops, flow, capacity):
    """Sharp bound times total population; pops must be sorted descending."""
    tail = sum(pops)
    result = flow * tail
    fixed = 0
    for h, n in enumerate(pops, 1):
        if n * (flow - (h-1)*capacity) <= capacity * tail:
            break
        tail -= n
        fixed += capacity * (2*h-1)*n
        result = fixed + tail * (flow + h*capacity)
    return result


def prefix_bound(r, label, joint):
    """Upper numerator, denominated in r.units. None proves time infeasibility."""
    remaining = r.limit - label.elapsed
    last = label.last
    if remaining < r.dist[last][0]:
        return None
    eligible = [v for v in r.nodes if not (label.mask >> v) & 1
                and r.dist[last][v] + r.dist[v][0] <= remaining]
    ps = [v for v in eligible if r.supply[v]]
    ds = [v for v in eligible if r.pop[v]]
    top_supply = list(accumulate(sorted((r.supply[v] for v in ps), reverse=True), initial=0))
    free_pops = sorted((r.pop[v] for v in ds), reverse=True)
    visited = sorted((r.pop[v] for v in label.route if r.pop[v]), reverse=True)
    costs = []
    for outgoing, incoming, cost in r.charges:
        budget = remaining - outgoing[last] - incoming[0]
        pc = list(accumulate(sorted(cost[v] for v in ps), initial=0))
        dc = list(accumulate(sorted(cost[v] for v in ds), initial=0))
        costs.append((budget, pc, dc))
    best = None
    q = sum(visited)
    for b in range(len(ds)+1):
        if b:
            q += free_pops[b-1]
        if b == 0 and not r.pop[last]:
            continue
        a = min(bisect_right(pc, budget-dc[b])-1 for budget, pc, dc in costs)
        if a < 0:
            break
        if not label.route and a == 0:
            continue
        # With no further deliveries, only immediate termination is legal.
        if b == 0 and label.elapsed + r.travel[last][0] > r.limit:
            continue
        supply_cap = label.unloaded + label.onboard + top_supply[a]
        flow = min(supply_cap, label.unloaded + b*r.capacity)
        value = q * flow
        if joint and (best is None or value > best):
            pop = sorted(visited + free_pops[:b], reverse=True)
            value = box_bound(pop, flow, r.capacity)
        best = value if best is None else max(best, value)
    return best


def solve_prefix(r, deadline, seed, *, joint=True, dominance=True, prize=False, emit=None):
    start = monotonic()
    seed_value = None if seed is None else check(r.instance, seed["route"], seed["amounts"], tol=0)["objective"]
    lb = seed_value
    route, amounts = (None, None) if seed is None else (seed["route"], seed["amounts"])
    metrics = dict(expanded=0, generated=0, pruned_bound=0, pruned_time=0, dominated=0,
                   allocations=0, max_frontier=1, bound_seconds=0.0, dominance=dominance, joint=joint,
                   prize_calls=0, prize_tighter=0, prize_seconds=0.0)
    def result(ub_num, exhausted=False):
        ub = Fraction(ub_num, r.units) if ub_num is not None else lb
        if lb is not None:
            ub = max(ub, lb)
        status = "infeasible" if exhausted and lb is None else "optimal" if lb is not None and ub == lb else "timeout"
        return {"status": status, "lb": lb, "ub": ub, "route": route, "amounts": amounts,
                "lb_exact": None if lb is None else str(lb), "ub_exact": None if ub is None else str(ub),
                "certificate": "rational_frontier", "metrics": metrics.copy()}

    if not r.p or not r.d:
        return result(None, True)
    if monotonic() >= deadline:
        return result(r.prior)
    r.prepare_bounds()
    if prize:
        from .prize import prize_bound
    def bound(label):
        tic = monotonic()
        val = prefix_bound(r, label, joint)
        if prize and val is not None and len(label.route) <= 1:
            started_prize = monotonic()
            extra = prize_bound(r, label)
            metrics["prize_seconds"] += monotonic() - started_prize
            metrics["prize_calls"] += 1
            metrics["prize_tighter"] += extra < val
            val = min(val, extra)
        metrics["bound_seconds"] += monotonic() - tic
        return val

    root = Label()
    upper = bound(root)
    if lb is not None and upper is not None and lb*r.units > upper:
        raise ArithmeticError("Root bound below verified incumbent")
    if upper is None:
        if lb is not None:
            raise ArithmeticError("Root time bound rejected a verified seed")
        return result(None, True)
    queue = [(-upper, 0, 0, root)]
    serial = count(1)
    shortest = {}
    next_save = start
    while queue:
        ub_num = -queue[0][0]
        if lb is not None and ub_num <= lb*r.units:
            return result(ub_num, True)
        now = monotonic()
        if now >= deadline:
            return result(ub_num)
        if emit and now >= next_save:
            emit(result(ub_num))
            next_save = now + 0.5
        _, _, _, label = heappop(queue)
        key = (label.mask, label.last, label.blocks)
        if dominance and label.elapsed > shortest.get(key, label.elapsed):
            metrics["dominated"] += 1
            continue
        metrics["expanded"] += 1
        if r.pop[label.last] and label.elapsed + r.travel[label.last][0] <= r.limit:
            value, candidate = allocate(r.instance, label.route)
            metrics["allocations"] += 1
            if lb is None or value > lb:
                checked = check(r.instance, label.route, candidate, tol=0)["objective"]
                if value != checked or value*r.units > ub_num:
                    raise ArithmeticError("Independent witness or active upper bound failed")
                lb, route, amounts = checked, label.route, candidate
                if emit:
                    emit(result(ub_num))
        for v in (r.p if not label.route else r.nodes):
            if label.mask & (1 << v):
                continue
            if monotonic() >= deadline:
                # The partially expanded parent still covers ungenerated children.
                return result(ub_num)
            child = label.extend(r, v)
            if child.elapsed + r.dist[v][0] > r.limit:
                metrics["pruned_time"] += 1
                continue
            metrics["generated"] += 1
            child_key = (child.mask, v, child.blocks)
            if dominance and shortest.get(child_key, child.elapsed + 1) <= child.elapsed:
                metrics["dominated"] += 1
                continue
            candidate_ub = bound(child)
            if candidate_ub is None:
                metrics["pruned_time"] += 1
                continue
            candidate_ub = min(ub_num, candidate_ub)
            if lb is not None and candidate_ub <= lb*r.units:
                metrics["pruned_bound"] += 1
                continue
            if dominance:
                shortest[child_key] = child.elapsed
            heappush(queue, (-candidate_ub, -len(child.route), next(serial), child))
        metrics["max_frontier"] = max(metrics["max_frontier"], len(queue))
    return result(None, True)
