"""Exact original-time elementary-route screen with a fleet fairness upper bound."""
from bisect import bisect_left
from dataclasses import dataclass, field
from fractions import Fraction as F
from functools import cmp_to_key, lru_cache
from math import isqrt, lcm
from time import monotonic_ns

from common import Data, verify_lower

METHOD = "exact-fleet-route-screen-v3"


class ScreenLimit(RuntimeError):
    """A resource limit prevented completion of the proof."""


class Work:
    def __init__(self, deadline_ns):
        self.deadline = deadline_ns
        self.counts = dict(prefixes=0, routes=0, suffix_states=0, fleet_steps=0,
                           rank_subsets=0, profile_checks=0, profile_fleets=0)

    def check(self):
        if self.deadline is not None and monotonic_ns() >= self.deadline:
            raise ScreenLimit("screen deadline")

    def spend(self, kind, quota, amount=1):
        count = self.counts[kind] + amount
        self.counts[kind] = count
        if count > quota:
            raise ScreenLimit(f"screen {kind} quota")
        if (count & 255) < amount:
            self.check()


def quantity_lattice(data):
    """A finite superset of potential delivery totals for elementary fleets."""
    if not data.q_step or not data.cap or not data.q_cap or data.q_cap // data.q_step > 30_000:
        return None
    capacity = int(data.cap / data.q_step)
    quantities = [int(q / data.q_step) for q in data.caps if q]
    cap_ticks = int(data.q_cap // data.q_step)
    mask = (1 << (cap_ticks + 1)) - 1
    subsets = 1
    for q in quantities:
        subsets = (subsets | (subsets << q)) & mask
    possible = 0
    for full_runs in range(1 + min(sum(quantities), cap_ticks) // capacity):
        possible |= (subsets << (full_runs * capacity)) & mask
    return possible


def _frontier(polygon, threshold, work, lattice=None):
    q_cap, p_cap = (F(polygon[key]) for key in ("quantity_cap", "p_cap"))
    q_step, p_step = (F(polygon[key]) if polygon[key] is not None else None
                      for key in ("q_step", "p_step"))
    cuts = [tuple(F(line[k]) for k in ("alpha", "beta", "linear_support"))
            for line in polygon["supports"]]
    swap = p_step is not None and (q_step is None or p_cap // p_step < q_cap // q_step)
    if swap:
        q_cap, p_cap = p_cap, q_cap
        q_step, p_step = p_step, q_step
        cuts = [(b, a, s) for a, b, s in cuts]
        lattice = None
    if q_step is None or q_cap // q_step > 30_000:
        return threshold, None, 0, (F(0), F(0), F(polygon["quantity_cap"]), F(polygon["p_cap"]))
    total = F(polygon["p_cap"])
    below, smallest = F(0), [None] * len(cuts)
    smallest_q, smallest_p, largest_q, largest_p = None, None, F(0), F(0)
    threats = 0
    for i in range(1, q_cap // q_step + 1):
        if (i & 255) == 0:
            work.check()
        if lattice is not None and not (lattice >> i & 1):
            continue
        q = i * q_step
        p_max = min([p_cap, *((s - a*q) / b for a, b, s in cuts)])
        if p_max <= 0:
            break
        if p_step is not None:
            p_max = (p_max // p_step) * p_step
        p_at_threshold = threshold * total / q
        if p_step is not None:
            p_safe = (p_at_threshold // p_step) * p_step
            p_need = p_safe + p_step
        else:
            p_safe = p_need = p_at_threshold
        below = max(below, q * min(p_max, p_safe) / total)
        if p_max <= p_at_threshold:
            continue
        threats += 1
        smallest_q = q if smallest_q is None else min(smallest_q, q)
        smallest_p = p_need if smallest_p is None else min(smallest_p, p_need)
        largest_q, largest_p = max(largest_q, q), max(largest_p, p_max)
        for j, (a, b, _) in enumerate(cuts):
            need = a*q + b*p_need
            if smallest[j] is None or need < smallest[j]:
                smallest[j] = need
    box = ((smallest_p, smallest_q, largest_p, largest_q) if swap
           else (smallest_q, smallest_p, largest_q, largest_p))
    return below, smallest, threats, box


def threat_width(data, polygon, threshold, deadline):
    lattice = quantity_lattice(data)
    if lattice is None or polygon["q_step"] is None:
        return None
    return _frontier(polygon, threshold, Work(deadline), lattice)[2]


@dataclass
class RouteCut:
    index: int
    slack: int
    scale: int
    best: int
    alpha: int
    beta: int
    price: tuple
    tau: int
    time_unit: F
    reward: tuple
    horizon: int
    cycle2: bool
    neighbors: tuple
    own: tuple


def _route_cuts(data, vehicles, polygon, minimum, q_unit, p_unit):
    cuts = []
    for index, (line, need) in enumerate(zip(polygon["supports"], minimum)):
        if line["ordered"] or line["mode"] not in ("per_vehicle", "pooled"):
            continue
        alpha, beta, tau, value, support = (F(line[k]) for k in (
            "alpha", "beta", "time_price", "route_value", "linear_support"))
        price = tuple(map(F, line["visit_prices"]))
        unit = F(line["time_unit"])
        total_ticks = data.limit * vehicles // unit
        if min(price, default=0) < 0 or support != sum(price) + vehicles * value + total_ticks * tau:
            continue
        supply = data.caps
        if line.get("supply_unit") is not None:
            grid = F(line["supply_unit"])
            supply = tuple(-(-x // grid) * grid for x in supply)
        slack = support - need
        if slack < 0:
            continue
        raw_supply = tuple(alpha * x for x in supply)
        raw_reward = tuple(s + beta*p - pi for s, p, pi in zip(raw_supply, data.people, price))
        scale = lcm(*(x.denominator for x in (
            slack, value, alpha*q_unit, beta*p_unit, tau, *price, *raw_supply, *raw_reward)))
        bits = {node: 1 << i for i, node in enumerate(data.nodes)}
        ng = line["ng_sets"] or ([()] * len(data.nodes))
        tracked = {node for own, group in zip(data.nodes, ng) for node in group if node != own}
        memory = tuple(bits[node] if node in tracked else 0 for node in data.nodes)
        neighbors = tuple(sum(bits[node] for node in group if node in tracked) for group in ng)
        cuts.append(RouteCut(index, int(scale*slack), scale, int(scale*value),
                             int(scale*alpha*q_unit), int(scale*beta*p_unit),
                             tuple(int(scale*x) for x in price), int(scale*tau), unit,
                             tuple(int(scale*x) for x in raw_reward),
                             data.limit // unit, line["cycle2"], neighbors, memory))
    # Small absolute relaxation slack in quantity units is a deterministic
    # preference only. Each selected inequality is checked independently.
    cuts.sort(key=lambda cut: (F(cut.slack, cut.alpha), cut.index))
    return cuts[:4]


@dataclass
class Bucket:
    mask: int
    population: int
    quantity: int
    gaps: tuple
    paid: tuple
    paths: dict = field(default_factory=dict)
    profiles: tuple | None = None


def _suffix(data, cut, allowed, work):
    n = len(data.nodes)
    time = tuple(tuple(x // cut.time_unit for x in row) for row in data.increments)
    home = tuple(x // cut.time_unit for x in data.home)
    destinations = tuple(j for j in range(n) if allowed & (1 << j))
    arcs = tuple(sorted((row[j], j) for j in destinations if j != i and row[j] <= cut.horizon)
                 for i, row in enumerate(time))
    # The walk uses the same projected memory rules and a conservative upper
    # on the number of visits when zero-time transitions are present.
    layered = any(cost == 0 for row in arcs[:-1] for cost, _ in row)
    cap = len(destinations) - 1 if layered else -1

    @lru_cache(maxsize=None)
    def suffix(previous, node, left, memory, steps):
        work.spend("suffix_states", 1_200_000)
        value = -cut.tau * home[node] if not data.sources[node] and home[node] <= left else None
        if steps == 0:
            return value
        for cost, target in arcs[node]:
            if cost > left:
                break
            if target == previous or cut.own[target] & memory:
                continue
            future = suffix(node if cut.cycle2 else -1, target, left-cost,
                            (memory & cut.neighbors[target]) | cut.own[target],
                            steps-1 if steps > 0 else steps)
            if future is not None:
                merit = cut.reward[target] - cut.tau*cost + future
                value = merit if value is None else max(value, merit)
        return value

    return suffix, time, home, cap


def _enumerate(data, cuts, forbidden, q_unit, p_unit, q_bound, p_bound, target_num, work):
    n = len(data.nodes)
    scale = lcm(*(x.denominator for x in (data.limit, *data.home, *(
        value for row in data.increments for value in row))))
    time = tuple(tuple(int(x*scale) for x in row) for row in data.increments)
    home = tuple(int(x*scale) for x in data.home)
    horizon = int(data.limit*scale)
    quantities = tuple(int(x/q_unit) for x in data.caps)
    populations = tuple(int(x/p_unit) for x in data.people)
    capacity = int(data.cap/q_unit)
    imbalance = q_bound * p_bound - target_num
    allowed = ((1 << n) - 1) ^ forbidden
    destinations = tuple(j for j in range(n) if allowed & (1 << j))
    arcs = tuple(sorted((row[j], j, 1 << j) for j in destinations if j != i and row[j] <= horizon)
                 for i, row in enumerate(time))
    tail = [home[i] if populations[i] else horizon+1 for i in range(n)]
    for _ in range(n):
        for i in destinations:
            tail[i] = min([tail[i], *(cost+tail[j] for cost, j, _ in arcs[i])])

    # On small graphs enumeration is already economical. On large graphs a
    # relaxed suffix makes a near-support threshold effective before closure.
    suffix_cut = cuts[0] if cuts and n > 25 else None
    suffix, projected, projected_home, steps = (_suffix(data, suffix_cut, allowed, work)
                                                 if suffix_cut else (None, None, None, None))
    secondary = tuple((cut, tuple(tuple(x // cut.time_unit for x in row) for row in data.increments),
                       tuple(x // cut.time_unit for x in data.home)) for cut in cuts)
    by_mask = {}
    path = []
    # If priced clocks are exactly proportional to the original clock (or
    # carry no price), a faster prefix gives no worse terminal reduced merit.
    project_exactly = all(not cut.tau or all(x % cut.time_unit == 0 for row in data.increments for x in row)
                          for cut in cuts)
    fastest = {} if n > 25 and project_exactly else None
    suppliers = tuple(j for j in destinations if quantities[j])
    source_mask = sum(1 << j for j in suppliers)
    if cuts and n > 25:
        shortest = [list(row) for row in time[:-1]]
        for k in destinations:
            for i in destinations:
                for j in destinations:
                    shortest[i][j] = min(shortest[i][j], shortest[i][k]+shortest[k][j])
        collectable = tuple(tuple((shortest[i][j]+tail[j], 1 << j) for j in suppliers if i != j)
                            for i in range(n))
    else:
        collectable = None

    @lru_cache(maxsize=65_536)
    def reachable(node, remaining):
        return sum(bit for cost, bit in collectable[node] if cost <= remaining)

    @lru_cache(maxsize=65_536)
    def supply_of(mask):
        if not mask:
            return 0
        first = mask & -mask
        return quantities[first.bit_length()-1] + supply_of(mask ^ first)

    def visit(previous, node, elapsed, mask, qty, stock, people,
              groups, pickup_run, delivery_run, ticks, price, memory, steps):
        work.spend("prefixes", 16_000_000)
        demand_credit = people*q_bound
        current_supply = qty+stock
        if current_supply > q_bound or people > p_bound:
            return
        if collectable is not None and demand_credit-current_supply*p_bound >= imbalance:
            unvisited = source_mask & ~mask
            total = current_supply + supply_of(unvisited & reachable(node, horizon-elapsed))
            if demand_credit-total*p_bound >= imbalance:
                return
        if suffix is not None:
            left = suffix_cut.horizon - ticks
            future = suffix(previous if suffix_cut.cycle2 else -1, node, left, memory, steps)
            merit = suffix_cut.alpha*current_supply + suffix_cut.beta*people - price - suffix_cut.tau*ticks
            if future is None or merit + future < suffix_cut.best - suffix_cut.slack:
                return
        if fastest is not None:
            key = (mask, node, groups, pickup_run, delivery_run)
            if fastest.get(key, horizon+1) <= elapsed:
                return
            if len(fastest) < 1_200_000:
                fastest[key] = elapsed
        if (populations[node] and elapsed+home[node] <= horizon and qty <= q_bound and people <= p_bound
            and abs(qty*p_bound - people*q_bound) < imbalance):
            deficits = []
            for cut, cut_times, cut_home in secondary:
                if cut is suffix_cut:
                    used = ticks + projected_home[node]
                    paid = price
                else:
                    used = cut_home[node] + cut_times[-1][path[0]]
                    used += sum(cut_times[a][b] for a, b in zip(path, path[1:]))
                    paid = sum(cut.price[v] for v in path)
                deficit = cut.best - (cut.alpha*qty + cut.beta*people - paid - cut.tau*used)
                if deficit > cut.slack:
                    break
                if deficit < 0:
                    raise ValueError("route merit exceeds the verified support maximum")
                deficits.append(deficit)
            else:
                work.spend("routes", 1_500_000)
                grouping = (*groups, (pickup_run, delivery_run))
                if mask in by_mask:
                    bucket = by_mask[mask]
                    bucket.quantity = max(bucket.quantity, qty)
                    if cuts:
                        bucket.gaps = tuple(map(min, zip(bucket.gaps, deficits)))
                else:
                    paid_prices = tuple(sum(cut.price[v] for v in path) for cut in cuts)
                    bucket = by_mask[mask] = Bucket(mask, people, qty, tuple(deficits), paid_prices)
                if grouping not in bucket.paths:
                    bucket.paths[grouping] = tuple(path)

        remaining = horizon - elapsed
        for cost, target, bit in arcs[node]:
            if cost > remaining:
                break
            if bit & mask or cost + tail[target] > remaining:
                continue
            path.append(target)
            is_pickup = data.sources[target]
            if suffix is not None:
                nt = ticks + projected[node][target]
                np = price + suffix_cut.price[target]
                nm = (memory & suffix_cut.neighbors[target]) | suffix_cut.own[target]
            else:
                nt = np = nm = 0
            if is_pickup and delivery_run:
                closed = (*groups, (pickup_run, delivery_run))
                picks, delivers = quantities[target], 0
            elif is_pickup:
                closed = groups
                picks, delivers = min(capacity, pickup_run+quantities[target]), 0
            else:
                closed = groups
                picks, delivers = pickup_run, delivery_run+populations[target]
            visit(node, target, elapsed+cost, mask|bit, qty if is_pickup else qty+stock,
                  min(capacity, stock+quantities[target]) if is_pickup else 0,
                  people+populations[target],
                  closed, picks, delivers, nt, np, nm, steps-1 if steps is not None and steps > 0 else steps)
            path.pop()

    for cost, target, bit in arcs[-1]:
        if data.sources[target] and cost + tail[target] <= horizon:
            path.append(target)
            visit(-1, target, cost, bit, 0, quantities[target], 0,
                  (), quantities[target], 0,
                  projected[-1][target] if suffix is not None else 0,
                  suffix_cut.price[target] if suffix_cut else 0,
                  suffix_cut.own[target] if suffix_cut else 0, steps)
            path.pop()
    if suffix is not None:
        suffix.cache_clear()
    return list(by_mask.values()), quantities, populations


def _rate_cmp(a, b):
    delta = a[0]*b[1] - b[0]*a[1]
    return (delta > 0) - (delta < 0)


def _rank_chain(groups, capacity, work, rates=None):
    """Each selected group can collect stock at its first delivery stop."""
    length = len(groups)
    if length == 1:
        if rates is not None:
            rates[0] = F(*groups[0])
        return groups
    full = (1 << length) - 1
    work.spend("rank_subsets", 10_000_000, full)
    rank = [0] * (full+1)
    people = [0] * (full+1)
    for mask in range(1, full+1):
        stock = amount = pop = 0
        for bit, (supply, demand) in enumerate(groups):
            stock = min(capacity, stock+supply)
            if mask & (1 << bit):
                amount += stock
                pop += demand
                stock = 0
        rank[mask], people[mask] = amount, pop

    fixed = 0
    chain = []
    while fixed != full:
        left = full ^ fixed
        chosen, q, p = 0, 0, 1
        add = left
        while add:
            candidate = rank[fixed | add] - rank[fixed]
            size = people[add]
            cross, best_cross = candidate*p, q*size
            if not chosen or cross < best_cross or (cross == best_cross and size > p):
                chosen, q, p = add, candidate, size
            add = (add - 1) & left
        fixed |= chosen
        if rates is not None:
            for j in range(length):
                if chosen & (1 << j):
                    rates[j] = F(q, p)
        if chain and _rate_cmp(chain[-1], (q, p)) == 0:
            earlier_q, earlier_p = chain.pop()
            q, p = q + earlier_q, p + earlier_p
        if chain and _rate_cmp(chain[-1], (q, p)) > 0:
            raise ValueError("nonmonotone rank chain")
        chain.append((q, p))
    return tuple(chain)


def _witness(data, paths, q_unit, p_unit, work):
    capacity = int(data.cap/q_unit)
    quantities = tuple(int(x/q_unit) for x in data.caps)
    populations = tuple(int(x/p_unit) for x in data.people)
    result = dict(routes=[], amounts=[])
    for path in paths:
        groups, demand_group = [], {}
        pending = pop = 0
        for node in path:
            if data.sources[node]:
                if pop:
                    groups.append((pending, pop))
                    pending = pop = 0
                pending = min(capacity, pending+quantities[node])
            else:
                pop += populations[node]
                demand_group[node] = len(groups)
        groups.append((pending, pop))
        rates = [None] * len(groups)
        _rank_chain(tuple(groups), capacity, work, rates)
        missing = F(0)
        amounts = []
        for node in reversed(path):
            if data.sources[node]:
                amount = min(missing, quantities[node])
                missing -= amount
            else:
                amount = rates[demand_group[node]] * populations[node]
                missing += amount
            if missing > capacity:
                raise ValueError("rank-chain allocation exceeds vehicle capacity")
            amounts.append(str(amount*q_unit))
        if missing:
            raise ValueError("rank-chain allocation starts with nonempty load")
        result["routes"].append([data.nodes[v] for v in path])
        result["amounts"].append(list(reversed(amounts)))
    return result


def _dominates(a, b, work):
    if a == b:
        return True
    work.spend("profile_checks", 3_000_000)
    # H(t) is piecewise linear, so checking all breakpoints and the plateau
    # suffices. Integrating H against any other profile preserves domination.
    for q, p in (*a, *b):
        av = sum(min(x*p, q*y) for x, y in a)
        bv = sum(min(x*p, q*y) for x, y in b)
        if av < bv:
            return False
    return True


def _profiles(bucket, capacity, chains, work):
    if bucket.profiles is not None:
        return bucket.profiles
    frontier = {}
    for groups, path in bucket.paths.items():
        chain = chains.get(groups)
        if chain is None:
            chain = chains[groups] = _rank_chain(groups, capacity, work)
        if chain in frontier or any(_dominates(old, chain, work) for old in frontier):
            continue
        frontier = {old: prev for old, prev in frontier.items() if not _dominates(chain, old, work)}
        frontier[chain] = path
    bucket.profiles = tuple(frontier.items())
    return bucket.profiles


def _fair(levels, population):
    value = 0
    for quantity, size in sorted(levels, key=cmp_to_key(_rate_cmp)):
        population -= size
        value += quantity * (size + 2*population)
    return value


class Levels:
    """Threshold to bitset for a route statistic, retaining canonical indices."""
    def __init__(self, values, work, max_levels=None):
        values = tuple(values)
        low = min(values, default=0)
        width = max(values, default=0) - low
        step = max(1, -(-width // max_levels)) if max_levels else 1
        groups = {}
        for i, value in enumerate(values):
            if (i & 1023) == 0:
                work.check()
            value = low + -(-(value-low)//step)*step
            groups[value] = groups.get(value, 0) | (1 << i)
        self.values = sorted(groups)
        self.sets = [0] * (len(groups) + 1)
        for i in range(len(self.values)-1, -1, -1):
            self.sets[i] = self.sets[i+1] | groups[self.values[i]]

    def at_least(self, value):
        return self.sets[bisect_left(self.values, value)]

    def top(self, mask, amount):
        """Maximum over at most amount members, overcounting the best for d > 1."""
        if not mask or not amount:
            return 0
        lo, hi = 0, len(self.values)
        while lo+1 < hi:
            mid = (lo+hi)//2
            if mask & self.sets[mid]:
                lo = mid
            else:
                hi = mid
        return self.values[lo] * amount


def _fleet(data, buckets, cuts, vehicles, required, quantities, populations, floor, needed_q, needed_p, work):
    if len(buckets) > 70_000:
        raise ScreenLimit("screen mask quota")
    best, paths = floor, None
    # The object is a symmetric fleet. A canonical order includes every active
    # subset once, with missing vehicles contributing the idle merit and zero flow.
    buckets.sort(key=lambda b: (-b.quantity*b.population, b.mask))
    q_all, p_all = sum(quantities), sum(populations)
    n = len(quantities)
    count = len(buckets)
    full = (1 << count) - 1
    outside = [full] * n
    inside = [0] * n
    for i, bucket in enumerate(buckets):
        if (i & 255) == 0:
            work.check()
        used = bucket.mask
        bit = 1 << i
        while used:
            station = used & -used
            used ^= station
            inside[station.bit_length()-1] |= bit
    for j in range(n):
        outside[j] ^= inside[j]
    pivots = sorted((j for j in range(n) if required & (1 << j)),
                    key=lambda j: inside[j].bit_count())
    bit_indices = {}
    for bucket in buckets:
        used = bucket.mask
        indices = []
        while used:
            bit = used & -used
            used ^= bit
            indices.append(bit.bit_length()-1)
        bit_indices[bucket.mask] = tuple(indices)

    q_levels = Levels((b.quantity for b in buckets), work)
    p_levels = Levels((b.population for b in buckets), work)
    # AM-GM gives necessary linear merit in several positive directions for a
    # candidate fleet whose (integer) product exceeds the best numerator.
    normals = ((max(1, p_all), max(1, q_all)), (max(1, p_all), max(1, 2*q_all)),
               (max(1, 2*p_all), max(1, q_all)))
    linear = [(a, b, Levels((a*r.quantity+b*r.population for r in buckets), work, 256)) for a, b in normals]
    marks = Levels(((b.mask & required).bit_count() for b in buckets), work) if required else None
    relaxed_gaps = [Levels((b.paid[j] - b.gaps[j] for b in buckets), work, 256) for j in range(len(cuts))]
    budgets = [sum(c.price) - c.slack for c in cuts]
    capacity = int(data.cap/(data.q_step or 1))
    chains = {}
    chosen = []

    def profiles(index, levels, selected, pop):
        nonlocal best, paths
        if index == len(chosen):
            work.spend("profile_fleets", 10_000_000)
            value = _fair(levels, pop)
            if value > best:
                best, paths = value, tuple(selected)
            return
        bucket = chosen[index]
        for chain, path in _profiles(bucket, capacity, chains, work):
            candidate = (*levels, *chain)
            optimistic = (*candidate, *((b.quantity, b.population) for b in chosen[index+1:]))
            if _fair(optimistic, pop) > best:
                selected.append(path)
                profiles(index+1, candidate, selected, pop)
                selected.pop()

    @lru_cache(maxsize=65_536)
    def covers(uncovered):
        if not uncovered:
            return full
        bit = uncovered & -uncovered
        return inside[bit.bit_length()-1] & covers(uncovered ^ bit)

    def search(available, used, left, qty, pop, supply_left, people_left, deficits, net, coarse):
        work.spend("fleet_steps", 30_000_000)
        uncovered = required & ~used
        if (coarse > best and not uncovered and qty >= needed_q and pop >= needed_p
            and all(d + left*c.best <= c.slack and e - left*c.best >= budget
                    for d, e, budget, c in zip(deficits, net, budgets, cuts))):
            profiles(0, (), [], pop)
        if not available or not left:
            return
        if left == 1:
            available &= covers(uncovered)
        elif uncovered:
            mark_need = uncovered.bit_count()
            top = marks.top(available, 1)
            if top*left < mark_need:
                return
            available &= marks.at_least(mark_need - top*(left-1))
        if not available:
            return
        for j, (cut, levels) in enumerate(zip(cuts, relaxed_gaps)):
            top = max(-cut.best, levels.top(available, 1))
            available &= levels.at_least(budgets[j] - net[j] - top*(left-1))
            if not available:
                return
        max_q, max_p = q_levels.top(available, 1), p_levels.top(available, 1)
        q_more, p_more = min(supply_left, max_q*left), min(people_left, max_p*left)
        needed = best - coarse
        if qty+q_more < needed_q or pop+p_more < needed_p:
            return
        if q_more*p_more + 2*sum(min(r.quantity*p_more, q_more*r.population) for r in chosen) <= needed:
            return
        q_needed = max(needed_q-qty, needed//(p_more+2*pop) + 1)
        p_needed = max(needed_p-pop, needed//(q_more+2*qty) + 1)
        available &= q_levels.at_least(q_needed - max_q*(left-1))
        available &= p_levels.at_least(p_needed - max_p*(left-1))
        for a, b, levels in linear:
            need = isqrt(4*a*b*(best+1)-1) + 1 - a*qty - b*pop
            top = levels.top(available, 1)
            if not available or top*left < need:
                return
            available &= levels.at_least(need - top*(left-1))
        if uncovered and left > 1:
            pivot = next(j for j in pivots if uncovered & (1 << j))
            picks = available & inside[pivot]
        else:
            picks = available
        while picks:
            work.spend("fleet_steps", 30_000_000)
            bit = picks & -picks
            picks ^= bit
            bucket = buckets[bit.bit_length()-1]
            next_deficits = tuple(a+b for a, b in zip(deficits, bucket.gaps))
            if any(d > c.slack for d, c in zip(next_deficits, cuts)):
                continue
            next_net = tuple(previous+paid-deficit for previous, paid, deficit in zip(net, bucket.paid, bucket.gaps))
            next_qty, next_pop = qty+bucket.quantity, pop+bucket.population
            next_coarse = coarse + bucket.quantity*bucket.population + 2*sum(
                min(r.quantity*bucket.population, bucket.quantity*r.population) for r in chosen)
            if left == 1 and next_coarse <= best:
                continue
            next_available = (available if uncovered else picks) if left > 1 else 0
            next_supply, next_people = supply_left, people_left
            if next_available:
                for j in bit_indices[bucket.mask]:
                    next_available &= outside[j]
                    next_supply -= quantities[j]
                    next_people -= populations[j]
            chosen.append(bucket)
            search(next_available, used|bucket.mask, left-1, next_qty, next_pop,
                   next_supply, next_people, next_deficits, next_net, next_coarse)
            chosen.pop()

    zero = (0,)*len(cuts)
    search(full, 0, vehicles, 0, 0, q_all, p_all, zero, zero, 0)
    return best, paths, len(chains)


def screen(instance, vehicles, polygon_or_none, threshold, deadline_ns, *,
           required=(), forbidden=(), stock_grid=False):
    """Return a deterministic proof, or raise ScreenLimit before any partial result.

    The caller verifies any supplied polygon/support certificate before invoking
    pure Python replay. Deadlines affect completion only, never proof contents.
    """
    work = Work(deadline_ns)
    work.check()
    data = Data.from_instance(instance)
    threshold = F(threshold)
    raw_vehicles = instance.get("vehicles", vehicles)
    if (type(vehicles) is not int or vehicles < 1 or threshold < 0
            or type(raw_vehicles) is not int or raw_vehicles != vehicles):
        raise ValueError("invalid fleet size or screen threshold")
    required, forbidden = tuple(sorted(set(required))), tuple(sorted(set(forbidden)))
    bits = {node: 1 << j for j, node in enumerate(data.nodes)}
    if any(type(node) is not int or node not in bits for node in (*required, *forbidden)):
        raise ValueError("unknown branch station")
    required_mask, forbidden_mask = (sum(bits[node] for node in group) for group in (required, forbidden))
    polygon = polygon_or_none
    base = dict(method=METHOD, threshold=str(threshold), required=list(required),
                forbidden=list(forbidden), polygon=polygon,
                maximizing_routes=None, maximizing_witness=None)
    lattice = None
    if stock_grid:
        if polygon is None or polygon["q_step"] is None or F(polygon["q_step"]) != data.q_step:
            raise ValueError("stock screening requires the original quantity grid")
        base["stock_grid"] = True
        lattice = quantity_lattice(data)
    if not data.total or not data.q_cap or required_mask & forbidden_mask:
        return dict(base, upper="0", stats=work.counts)
    if polygon is None and threshold >= data.q_cap:
        return dict(base, upper=str(data.q_cap), stats=work.counts)
    q_unit, p_unit = data.q_step, data.p_step
    objective_unit = q_unit * p_unit / data.total
    cuts, threat_rows = (), None
    below = (threshold // objective_unit) * objective_unit
    q_min = (threshold // (data.total/p_unit * objective_unit) + 1) * q_unit
    p_min = (threshold // (data.q_cap/q_unit * objective_unit) + 1) * p_unit
    q_max, p_max = data.q_cap, data.total
    if polygon is not None:
        below, minimum, threat_rows, (proof_q, proof_p, q_max, p_max) = _frontier(
            polygon, threshold, work, lattice)
        if minimum is not None:
            if not threat_rows:
                work.check()
                return dict(base, upper=str(below), stats=work.counts, threat_rows=0)
            cuts = _route_cuts(data, vehicles, polygon, minimum, q_unit, p_unit)
            q_min, p_min = max(q_min, proof_q), max(p_min, proof_p)
    supply_all = sum(data.caps)
    must_visit = sum(1 << j for j, (qty, people) in enumerate(zip(data.caps, data.people))
                     if supply_all-qty < q_min or data.total-people < p_min)
    if must_visit & forbidden_mask:
        work.check()
        return dict(base, upper=str(below), stats=work.counts, threat_rows=threat_rows)
    floor = int(below // objective_unit)
    # Inside the threat region a fleet's product is above threshold, but its
    # fairness score can lie between below and threshold. Only the proven
    # nonthreat ceiling can be used in fairness pruning.
    buckets, quantities, populations = _enumerate(data, cuts, forbidden_mask, q_unit, p_unit,
                                                   q_max//q_unit, p_max//p_unit, floor, work)
    bound, paths, chain_count = _fleet(data, buckets, cuts, vehicles, required_mask | must_visit,
                                       quantities, populations, floor, -(-q_min//q_unit), -(-p_min//p_unit), work)
    witness = _witness(data, paths, q_unit, p_unit, work) if paths else None
    if witness is not None and verify_lower(instance, vehicles, witness) != bound*objective_unit:
        raise ValueError("rank-chain allocation differs from the exact terminal bound")
    work.check()
    work.counts.update(masks=len(buckets), rank_chains=chain_count)
    return dict(base, upper=str(max(below, bound*objective_unit)), nonthreat_upper=str(below),
                threat_rows=threat_rows, support_indices=[cut.index for cut in cuts],
                slack=[str(F(cut.slack, cut.scale)) for cut in cuts], stats=work.counts,
                implied_required=[node for j, node in enumerate(data.nodes) if must_visit & (1 << j)],
                maximizing_routes=witness["routes"] if witness else None, maximizing_witness=witness)
