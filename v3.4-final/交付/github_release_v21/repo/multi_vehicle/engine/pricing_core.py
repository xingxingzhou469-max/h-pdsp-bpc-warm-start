"""C++ integer pricing, with restricted-master price hints kept outside verification."""
import ctypes as ct
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from math import lcm
from pathlib import Path
import os
import shutil
import subprocess
from time import monotonic_ns
from highspy import Highs, HighsModelStatus, HighsStatus, kHighsInf
import numpy as np

from common import NANO, ceil_ratio, digest, projection
from polygon import maximize

GRANULARITY = 1024
DENSE_STATE_CAP = 2_000_000
NG_STATE_CAP = 240_000
NG_ORDERED_STATE_CAP = 900_000
GRID_POINTS_CAP = 30_000
PHASE_ITERATIONS = 75
DSSR_ROUNDS = 9
POOL_CAP = 1800
SLOPE_FACTORS = (Fraction(3, 2), Fraction(3, 4), 2, Fraction(1, 2), 3,
                 Fraction(5, 4), Fraction(5, 2), 4, Fraction(1, 3), Fraction(7, 4),
                 8, Fraction(1, 8), 16)
MISSING = -(2**61)


class StateCap(Exception):
    pass


class Deadline(Exception):
    pass


@lru_cache(maxsize=1)
def library():
    source = Path(__file__).with_name("upper_src") / "price.cpp"
    build = Path(os.environ.get("NG_BB_BUILD", Path(__file__).resolve().parents[1] / "build"))
    flags = ["-std=c++17", "-O3", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared"]
    if os.environ.get("NG_BB_SANITIZE") == "1":
        flags += ["-g", "-pedantic", "-fsanitize=address,undefined", "-fno-sanitize-recover=all"]
    key = sha256((digest(source) + repr(flags)).encode()).hexdigest()[:16]
    target = build / f"price-{key}.so"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(f".{os.getpid()}.tmp")
        try:
            compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("clang++") or "c++"
            subprocess.run([compiler, *flags, str(source), "-o", str(temporary)], check=True, timeout=30)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    handle = ct.CDLL(str(target))
    price = handle.price
    ints = ct.POINTER(ct.c_int)
    values = ct.POINTER(ct.c_int64)
    masks = ct.POINTER(ct.c_uint64)
    price.argtypes = ([ct.c_int] * 7 + [ct.c_int64] + [ints] * 3 + [values] +
                      [ct.c_int64] * 2 + [masks] * 2 + [values] + [ints] * 5
                      + [ct.c_uint64, ct.c_int, values])
    price.restype = ct.c_int
    return price


@dataclass(frozen=True)
class Column:
    visits: tuple
    time: int
    quantity: Fraction
    population: Fraction
    value: Fraction
    route: tuple
    amounts: tuple | None


class Pricing:
    def __init__(self, data, vehicles, mode, *, cycle2=False, ordered=False,
                 neighborhoods=None, units=None, state_cap=None, markers=()):
        self.data, self.vehicles = data, vehicles
        self.mode, self.cycle2, self.ordered = mode, cycle2, ordered
        self.time_unit, self.load_unit = projection(data, vehicles) if units is None else units
        self.pool_budget = int(data.limit * vehicles // self.time_unit)
        self.horizon = self.pool_budget if mode == "pooled" else int(data.limit // self.time_unit)
        n = self.n = len(data.nodes)
        self.load_max = ceil_ratio(data.cap, self.load_unit) if ordered else 0
        if (n == 0 or n * (n + 1) >= 2**30
                or self.horizon < 0 or self.horizon > (2**30 - 2) // n
                or self.load_max >= 2**30):
            raise StateCap("projected dimensions exceed native limits")
        maximum = self.horizon + 1
        arcs = [min(maximum, int(t // self.time_unit)) for row in data.increments for t in row]
        self.layered = any(arcs[i * n + j] == 0 for i in range(n) for j in range(n) if i != j)
        self.max_steps = n if self.layered else self.horizon + 1
        self.arcs = (ct.c_int * ((n + 1) * n))(*arcs)
        self.home = (ct.c_int * n)(*(min(maximum, int(cost // self.time_unit)) for cost in data.home))
        self.pickups = (ct.c_int * n)(*(
            (ceil_ratio(cap, self.load_unit) if ordered else 0) if source else -1
            for cap, source in zip(data.caps, data.sources)
        ))
        # Preserve exact credits when the raw grid already has a small denominator.
        self.supply_unit = (data.q_step if data.q_step and data.q_step.denominator <= GRANULARITY
                            else Fraction(1, GRANULARITY))
        self.credits = tuple(ceil_ratio(cap, self.supply_unit) * self.supply_unit for cap in data.caps)
        self.markers = tuple(sorted(set(markers)))
        if self.markers:
            if not set(self.markers) <= set(data.nodes):
                raise ValueError("invalid marked station")
            neighborhoods = tuple(tuple(sorted(set(group) | set(self.markers)))
                                  for group in (neighborhoods or [(v,) for v in data.nodes]))
        self.neighborhoods = neighborhoods
        tracked = sorted({node for i, group in enumerate(neighborhoods or ())
                          for node in group if node != data.nodes[i]})
        if len(tracked) > 64:
            raise StateCap("more than 64 stations tracked by ng memory")
        bits = {node: 1 << i for i, node in enumerate(tracked)}
        self.marked_bits = sum(bits.get(node, 0) for node in self.markers)
        # With one station there is no nonempty route in this model. The
        # marked flag still needs its own memory bit for the class interface.
        if self.markers and not self.marked_bits:
            bits = {self.markers[0]: 1}
            self.marked_bits = 1
        self.ng_masks = (ct.c_uint64 * n)(*(
            sum(bits.get(node, 0) for node in group)
            for group in (neighborhoods or [()] * n)
        ))
        self.own_bits = (ct.c_uint64 * n)(*(bits.get(node, 0) for node in data.nodes))
        self.sparse = bool(bits)
        sparse_cap = NG_ORDERED_STATE_CAP if ordered else NG_STATE_CAP
        self.state_cap = (sparse_cap if self.sparse else DENSE_STATE_CAP) if state_cap is None else state_cap
        if not 0 < self.state_cap < 2**30:
            raise StateCap("invalid native state cap")
        base = (self.horizon + 1) * n * (n + 1 if self.layered else 1)
        if (base if self.sparse else base * (self.load_max + 1)) > self.state_cap:
            raise StateCap("projected DP exceeds the state cap")

    def solve(self, alpha, beta, prices, time_price, deadline_ns):
        raw = [beta * p - price + (0 if self.ordered else alpha * cap)
               for p, cap, price in zip(self.data.people, self.credits, prices)]
        load_reward = alpha * self.load_unit if self.ordered else Fraction(0)
        scale = lcm(*(v.denominator for v in (*raw, load_reward, time_price)))
        rewards = [int(scale * x) for x in raw]
        load_ticks = int(scale * load_reward)
        time_ticks = int(scale * time_price)
        magnitude = max(map(abs, rewards), default=0) + abs(load_ticks) * self.load_max
        if (max(abs(time_ticks), abs(load_ticks)) >= 2**60
                or magnitude * (self.max_steps + 2) + abs(time_ticks) * self.horizon >= 2**60):
            raise OverflowError("scaled reward exceeds the exact C++ signed arithmetic range")

        n = self.n
        values = (ct.c_int64 * n)()
        lengths = (ct.c_int * n)()
        used_times = (ct.c_int * n)()
        stops = (ct.c_int * (n * self.max_steps))()
        amounts = (ct.c_int * (n * self.max_steps))()
        states = ct.c_int()
        classes = (ct.c_int64 * (1 << len(self.markers)))() if self.markers else None
        engine = library()
        remaining = min(deadline_ns - monotonic_ns(), 2**62)
        if remaining <= 0:
            raise Deadline
        status = engine(n, self.horizon, self.load_max, int(self.layered), int(self.cycle2),
                        int(self.sparse), self.state_cap, remaining, self.arcs, self.home,
                        self.pickups, (ct.c_int64 * n)(*rewards), load_ticks, time_ticks,
                        self.ng_masks, self.own_bits, values, lengths, used_times, stops,
                        amounts, ct.byref(states), self.marked_bits, len(classes) if classes is not None else 0,
                        classes)
        if status == 1:
            raise StateCap(f"DP reached the {self.state_cap} state cap")
        if status == 2:
            raise Deadline
        if status != 0:
            raise RuntimeError(f"unexpected DP status {status}")

        columns = []
        for end, length in enumerate(lengths):
            if not length:
                continue
            start = end * self.max_steps
            route = tuple(reversed(stops[start:start + length]))
            qtys = tuple(reversed(amounts[start:start + length])) if self.ordered else None
            visits = Counter(route)
            quantity = (sum(q for i, q in zip(route, qtys) if self.data.people[i]) * self.load_unit
                        if self.ordered else sum(self.credits[i] for i in route))
            population = sum(self.data.people[i] for i in route)
            columns.append(Column(tuple(visits[i] for i in range(n)), used_times[end],
                                  quantity, population, Fraction(values[end], scale),
                                  tuple(self.data.nodes[i] for i in route), qtys))
        best = max(columns, default=None, key=lambda column: column.value)
        route_value = max(Fraction(0), best.value) if best else Fraction(0)
        support = sum(prices, Fraction(0)) + self.vehicles * route_value + self.pool_budget * time_price
        witness = None
        if best and route_value > 0:
            witness = dict(route=best.route, time_ticks=best.time, amounts=best.amounts)
        cert = dict(mode=self.mode, cycle2=self.cycle2, ordered=self.ordered,
                    ng_sets=self.neighborhoods, time_unit=str(self.time_unit), load_unit=str(self.load_unit),
                    alpha=str(alpha), beta=str(beta), visit_prices=tuple(map(str, prices)),
                    time_price=str(time_price), route_value=str(route_value), linear_support=str(support),
                    supply_unit=None if self.ordered else str(self.supply_unit),
                    horizon=self.horizon, witness=witness, reached_states=states.value,
                    state_cap=self.state_cap)
        if classes is not None:
            cert.update(markers=self.markers, class_values=[None if x == MISSING else str(Fraction(x, scale))
                                                          for x in classes])
        return cert, columns


class RestrictedDual:
    """Incremental hint LP. Only a completed integer DP can certify a price."""

    def __init__(self, pricing, beta):
        self.solver = Highs()
        self.solver.setOptionValue("log_to_console", False)
        self.solver.setOptionValue("output_flag", False)
        self.solver.setOptionValue("threads", 1)
        self.solver.setOptionValue("solver", "simplex")
        self.n, self.beta = pricing.n, beta
        count = self.n + 2
        self.solver.addCols(count, [1] * self.n + [pricing.pool_budget, pricing.vehicles],
                            np.zeros(count), np.full(count, kHighsInf), 0, [], [], [])
        self.seen = set()

    def add(self, columns):
        starts, indexes, values, rhs = [], [], [], []
        for col in columns:
            key = col.visits, col.time, col.quantity
            if key in self.seen:
                continue
            self.seen.add(key)
            starts.append(len(indexes))
            for node, count in enumerate(col.visits):
                if count:
                    indexes.append(node)
                    values.append(count)
            if col.time:
                indexes.append(self.n)
                values.append(col.time)
            indexes.append(self.n + 1)
            values.append(1)
            rhs.append(float(col.quantity + self.beta * col.population))
        if rhs:
            self.solver.addRows(len(rhs), rhs, np.full(len(rhs), kHighsInf),
                                len(indexes), starts, indexes, values)


def price_hint(pool, pricing, deadline_ns):
    remaining = (deadline_ns - monotonic_ns()) / NANO
    if not pool.seen or remaining <= 0:
        return None
    pool.solver.setOptionValue("time_limit", remaining)
    if (pool.solver.run() != HighsStatus.kOk
            or pool.solver.getModelStatus() != HighsModelStatus.kOptimal):
        return None
    values = pool.solver.getSolution().col_value[:-1]
    ticks = tuple(Fraction(max(0, round(x * GRANULARITY)), GRANULARITY) for x in values)
    return ticks[:-1], ticks[-1]


def _fits_memory(column, pricing):
    memory = 0
    for node in column.route:
        i = node - 1
        if memory & pricing.own_bits[i]:
            return False
        memory = (memory & pricing.ng_masks[i]) | pricing.own_bits[i]
    return True


def phase(pricing, beta, initial, deadline_ns, collect, shared=None, iterations=PHASE_ITERATIONS):
    alpha = Fraction(1)
    prices, tau = initial
    pool, seen = RestrictedDual(pricing, beta), set()
    if shared:
        pool.add(col for col in shared.values() if _fits_memory(col, pricing))
    best, best_cert, best_columns, best_hint = None, None, (), initial
    reason = "iteration_cap"
    for iteration in range(1, iterations + 1):
        if (prices, tau) in seen:
            reason = "repeated_prices"
            break
        seen.add((prices, tau))
        try:
            cert, columns = pricing.solve(alpha, beta, prices, tau, deadline_ns)
        except StateCap:
            reason = "state_cap"
            break
        except Deadline:
            reason = "deadline"
            break
        except OverflowError:
            reason = "overflow"
            break
        support = Fraction(cert["linear_support"])
        if best is None or support < best:
            best, best_cert, best_columns, best_hint = support, cert, columns, (prices, tau)
            collect(cert, iteration)
        try:
            pool.add(columns)
            if shared is not None:
                for column in columns:
                    key = column.visits, column.quantity
                    old = shared.get(key)
                    if old is None or old.time >= column.time:
                        shared[key] = column
                while len(shared) > POOL_CAP:
                    del shared[next(iter(shared))]
            hint = price_hint(pool, pricing, deadline_ns)
        except OverflowError:
            reason = "overflow"
            break
        if hint is None:
            reason = "deadline" if monotonic_ns() >= deadline_ns else "no_price_hint"
            break
        prices, tau = hint
    return best_cert, best_columns, best_hint, reason, len(seen)


def conflict_updates(columns):
    """Rank recurring shortest remembered segments from exact terminal maximizers."""
    votes = Counter()
    for column in sorted(columns, key=lambda col: -col.value)[:12]:
        last = {}
        for j, node in enumerate(column.route):
            if node in last:
                segment = tuple(sorted(set(column.route[last[node] + 1:j])))
                votes[node, segment] += 1
            last[node] = j
    return [update for update, _ in sorted(votes.items(),
                                          key=lambda item: (-item[1], len(item[0][1]), item[0]))]


def refine(data, neighborhoods, updates, count):
    sets = [set(group) for group in neighborhoods]
    indexes = {node: i for i, node in enumerate(data.nodes)}
    consumed = 0
    for node, interior in updates:
        targets = [indexes[j] for j in interior if node not in sets[indexes[j]]]
        if not targets:
            continue
        for target in targets:
            sets[target].add(node)
        consumed += 1
        if consumed == count:
            break
    return tuple(tuple(sorted(group)) for group in sets) if consumed else None


def envelope(data, vehicles, supports, grid="continuous"):
    if grid == "both" and (not data.p_step or not data.q_step
                            or min(data.total // data.p_step, data.q_cap // data.q_step) > GRID_POINTS_CAP):
        grid = "continuous"
    p_step = data.p_step if grid in ("both", "population") else None
    q_step = data.q_step if grid == "both" else None
    lines = [tuple(Fraction(s[k]) for k in ("alpha", "beta", "linear_support")) for s in supports]
    upper = (maximize(data.q_cap, data.total, lines, p_step=p_step, q_step=q_step)
             if vehicles else Fraction(0))
    return dict(method="ng-product-v1", vehicles=vehicles, supports=list(supports), grid=grid,
                quantity_cap=str(data.q_cap), p_cap=str(data.total),
                p_step=str(p_step) if p_step is not None else None,
                q_step=str(q_step) if q_step is not None else None, upper=str(upper))


ARMS = ("pooled", "per_vehicle", "cycle2", "ng", "polygon", "ordered_polygon")


def rational_slope(value):
    """Any positive slope is legal. Limit hint precision before integer pricing."""
    return value.limit_denominator(GRANULARITY) or Fraction(1, GRANULARITY)


@dataclass
class Slope:
    beta: Fraction
    sets: tuple | None
    hint: tuple
    certificate: dict | None
    columns: tuple
    attempts: int = 0
    exhausted: bool = False
    seconds: float = 0.0
    rate: Fraction = Fraction(0)


def repeats(cert):
    route = (cert["witness"] or {}).get("route", ()) if cert else ()
    return len(route) - len(set(route))


def run_upper(data, vehicles, arm, deadline_ns, emit, *, after_core=None, after_slice=None):
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    mode = "pooled" if arm == "pooled" else "per_vehicle"
    cycle2, ordered = arm not in ARMS[:2], arm == "ordered_polygon"
    dssr, multiple = arm in ARMS[3:], arm in ARMS[4:]
    progress, lines, slopes, shared = [], {}, {}, {}
    best_bound, emitted_lines = None, None
    started = monotonic_ns()
    budget = max(0, deadline_ns - started)
    grid = "both" if multiple else "continuous"

    def emit_envelope(iteration, *, snapshot=False):
        nonlocal best_bound, emitted_lines
        selected = [lines[key] for key in sorted(lines)]
        if selected == emitted_lines or monotonic_ns() >= deadline_ns:
            return
        answer = envelope(data, vehicles, selected, grid)
        value = Fraction(answer["upper"])
        if (best_bound is None or value < best_bound or snapshot and value == best_bound) and emit(
                answer, len(progress), iteration):
            best_bound, emitted_lines = value, selected

    def collect(cert, iteration):
        beta = Fraction(cert["beta"])
        old = lines.get(beta)
        if old:
            previous = Fraction(old["linear_support"]), repeats(old)
            candidate = Fraction(cert["linear_support"]), repeats(cert)
            if previous <= candidate:
                return
        lines[beta] = cert
        emit_envelope(iteration)

    if not data.q_cap or not data.total or vehicles == 0:
        emit(envelope(data, vehicles, ()), 0, 0)
        return progress

    ideal_beta = data.q_cap / data.total
    nil = ((Fraction(0),) * len(data.nodes), Fraction(0))
    neighborhoods = tuple((node,) for node in data.nodes) if dssr else None
    units = None
    if ordered:
        time_unit, load_unit = projection(data, vehicles)
        units = time_unit, max(load_unit, data.cap / 16)

    def search(beta, sets, initial, stop, stage):
        begin = monotonic_ns()
        try:
            pricing = Pricing(data, vehicles, mode, cycle2=cycle2, ordered=ordered,
                              neighborhoods=sets, units=units)
        except StateCap:
            result = None, (), initial, "state_cap", 0
        else:
            result = phase(pricing, beta, initial, stop, collect, shared=shared)
        cert, columns, prices, reason, iterations = result
        seconds = (monotonic_ns() - begin) / NANO
        progress.append(dict(stage=stage, beta=str(beta), memory=sum(len(g)-1 for g in sets) if sets else 0,
                             reason=reason, iterations=iterations, elapsed_s=seconds,
                             repeated_visits=repeats(cert),
                             support=cert["linear_support"] if cert else None,
                             reached_states=cert["reached_states"] if cert else None))
        return Slope(beta, sets, prices, cert, columns, seconds=seconds)

    def slice_until(ratio, stop=deadline_ns):
        return min(stop, monotonic_ns() + max(NANO // 50, int(ratio * budget)))

    def open_slope(factor, stop):
        beta = rational_slope(ideal_beta * factor)
        if beta in slopes or monotonic_ns() >= stop:
            return
        previous = [s for s in slopes.values() if s.certificate]
        if previous:
            neighbor = min(previous, key=lambda s: abs(beta - s.beta))
            scale = max(Fraction(1), beta / neighbor.beta)
            seed = tuple(x * scale for x in neighbor.hint[0]), neighbor.hint[1] * scale
            seed = tuple(Fraction(round(x * GRANULARITY), GRANULARITY) for x in seed[0]), Fraction(
                round(seed[1] * GRANULARITY), GRANULARITY)
        else:
            seed = nil
        slopes[beta] = search(beta, neighborhoods, seed, stop, "core" if len(slopes) < 4 else "new_slope")

    if not multiple:
        beta = rational_slope(ideal_beta)
        slopes[beta] = search(beta, neighborhoods, nil, slice_until(0.35) if dssr else deadline_ns, "core")
    else:
        # Each core slope gets a proportional first pass before any DSSR refinement.
        for factor in (1, Fraction(3, 2), Fraction(3, 4), 2):
            open_slope(factor, slice_until(0.095))
        emit_envelope(0, snapshot=True)
        if after_core is not None and monotonic_ns() < deadline_ns:
            after_core(envelope(data, vehicles, [lines[key] for key in sorted(lines)], grid))

    def deepen(slope, stop):
        # Try a block of recurring conflicts, then one when the larger memory caps.
        updates = conflict_updates(slope.columns)
        slope.attempts += 1
        for count in (3, 1):
            proposed = refine(data, slope.sets, updates, count)
            if proposed is None or monotonic_ns() >= stop:
                break
            candidate = search(slope.beta, proposed, slope.hint, stop, "dssr")
            if candidate.certificate is not None:
                candidate.attempts = slope.attempts
                gain = Fraction(slope.certificate["linear_support"]) - Fraction(candidate.certificate["linear_support"])
                candidate.rate = max(Fraction(0), gain) / max(1, round(1000*candidate.seconds))
                slopes[slope.beta] = candidate
                return
        slope.exhausted = True

    def priorities():
        active = [s for s in slopes.values() if s.certificate and not s.exhausted
                  and repeats(s.certificate) and s.attempts < (DSSR_ROUNDS * 2 if multiple else DSSR_ROUNDS)]
        selected = tuple(lines.values())
        base = Fraction(envelope(data, vehicles, selected, "continuous")["upper"])

        def importance(slope):
            rest = [s for s in selected if Fraction(s["beta"]) != slope.beta]
            loss = Fraction(envelope(data, vehicles, rest, "continuous")["upper"]) - base
            solo = Fraction(envelope(data, vehicles, (lines[slope.beta],), "continuous")["upper"])
            return -loss, -slope.rate, solo, slope.beta

        ranked = sorted(active, key=importance)
        return sorted(enumerate(ranked), key=lambda item: (
            item[0] + item[1].attempts / 2, -repeats(item[1].certificate), item[0]))

    factors = iter(f for f in SLOPE_FACTORS if f not in (Fraction(3, 2), Fraction(3, 4), 2))
    next_factor = next(factors, None) if multiple else None
    rounds = 0
    while dssr and monotonic_ns() < deadline_ns:
        candidates = priorities()
        # Alternate targeted memory and new slopes. Once the sweep is done,
        # spend every available slice on the surviving binding conflicts.
        if next_factor is not None and (not candidates or rounds % 3 == 2):
            open_slope(next_factor, slice_until(0.065))
            next_factor = next(factors, None)
        elif candidates:
            deepen(candidates[0][1], slice_until(0.072))
        else:
            break
        rounds += 1
        emit_envelope(0, snapshot=True)
        if after_slice is not None and monotonic_ns() < deadline_ns:
            after_slice()
    return progress
