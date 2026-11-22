"""Bounded multi-start prefix construction for strictly feasible incumbents."""

from heapq import heappop, heappush, heapreplace
from itertools import count
from time import monotonic
from typing import NamedTuple

from .allocation import allocate
from .check import check
from .neighborhood import improve_seed


# Time credits as fractions of the horizon. Small credits prioritize compact
# prefixes, while larger ones give supply and covered population more weight.
CREDITS = (1, 4, 12)
CREDIT_SCALE = 10
WIDTH = 28
PER_END = 2
COMPLETE_WIDTH = 18


class Prefix(NamedTuple):
    route: tuple
    used: int
    last: int
    time: int
    unloaded: int
    onboard: int
    people: int


def _completion_times(r, deadline):
    """Relaxed travel/service time to an agency-ending return, allowing repeats."""
    distances = [r.limit + 1] * len(r.travel)
    queue = []
    for d in r.d:
        direct = r.travel[d][0]
        if direct <= r.limit:
            distances[d] = direct
            heappush(queue, (direct, d))
    while queue:
        if monotonic() >= deadline:
            return None
        elapsed, v = heappop(queue)
        if elapsed != distances[v]:
            continue
        extra = elapsed + r.service[v]
        for u in r.nodes:
            if u != v and r.travel[u][v] + extra < distances[u]:
                distances[u] = r.travel[u][v] + extra
                heappush(queue, (distances[u], u))
    return distances


def _offer(heap, size, priority, serial, value):
    entry = (priority, serial, value)
    if len(heap) < size:
        heappush(heap, entry)
    elif entry[:2] > heap[0][:2]:
        heapreplace(heap, entry)


def _layer(r, beam, credit, tail, bias, floor, deadline, metrics, serial, width, per_end):
    by_end = [[] for _ in r.travel]
    complete = []
    for prefix in beam:
        if monotonic() >= deadline:
            break
        travel = r.travel[prefix.last]
        choices = r.nodes if prefix.last else r.p
        for v in choices:
            bit = 1 << v
            if prefix.used & bit:
                continue
            time = prefix.time + travel[v] + r.service[v]
            near = time + tail[v]
            if near > r.limit:
                continue
            if r.supply[v]:
                unloaded = prefix.unloaded
                load = min(r.capacity, prefix.onboard + r.supply[v])
                people = prefix.people
            else:
                unloaded = prefix.unloaded + prefix.onboard
                load = 0
                people = prefix.people + r.pop[v]
            # Both the rule for beam retention and the beam cap are heuristic.
            # The upper envelope for a completed route only filters that route.
            raw = unloaded * people
            potential = (unloaded + load) * (people + bias)
            rank = (potential * credit // (credit + near), raw, -near)
            route = prefix.route + (v,)
            new = Prefix(route, prefix.used | bit, v, time, unloaded, load, people)
            index = next(serial)
            _offer(by_end[v], per_end, rank, index, new)
            metrics["generated_prefixes"] += 1
            if r.pop[v] and raw > floor and time + r.travel[v][0] <= r.limit:
                _offer(complete, COMPLETE_WIDTH, (raw, -time), index, route)
    ranking = sorted((entry for pool in by_end for entry in pool), reverse=True)[:width]
    return [entry[2] for entry in ranking], [entry[2] for entry in sorted(complete, reverse=True)]


def improve(r, deadline, seed, *, emit=None, metrics=None, width=None, per_end=None):
    """Return only verified, strictly improving incumbents; the beam supplies no bound."""
    best = seed
    best_value = None if seed is None else check(r.instance, seed["route"], seed["amounts"], tol=0)["objective"]
    if best_value is not None and best_value * r.units >= r.prior:
        return best
    if not r.p or not r.d or monotonic() >= deadline:
        return best
    if metrics is None:
        metrics = {}
    width = WIDTH if width is None else width
    per_end = PER_END if per_end is None else per_end
    metrics.update(generated_prefixes=0, retained_prefixes=0, complete_allocations=0,
                   beam_layers=0, strict_improvements=0, polish_improvements=0)
    initial_floor = -1 if best_value is None else (best_value * r.units) // 1
    floor = initial_floor

    def publish(witness, value, *, polished=False):
        nonlocal best, best_value, floor
        if (best_value is not None and value <= best_value) or monotonic() >= deadline:
            return
        best, best_value, floor = witness, value, (value * r.units) // 1
        metrics["strict_improvements"] += 1
        metrics["polish_improvements"] += polished
        if emit is not None:
            emit(best, value)

    tail = _completion_times(r, deadline)
    if tail is None:
        return best
    bias = max(1, r.people // len(r.d))
    initial = Prefix((), 0, 0, 0, 0, 0, 0)
    beams = [[initial] for _ in CREDITS]
    credits = [max(r.time_scale, r.limit * fraction // CREDIT_SCALE)
               for fraction in CREDITS]
    evaluated = set()
    serial = count()
    while any(beams) and monotonic() < deadline:
        for i, beam in enumerate(beams):
            if not beam:
                continue
            next_beam, complete = _layer(r, beam, credits[i], tail, bias, floor,
                                         deadline, metrics, serial, width, per_end)
            metrics["retained_prefixes"] += len(next_beam)
            metrics["beam_layers"] += 1
            beams[i] = next_beam
            for route in complete:
                if monotonic() >= deadline:
                    return best
                if route in evaluated:
                    continue
                evaluated.add(route)
                value, amounts = allocate(r.instance, route)
                metrics["complete_allocations"] += 1
                if best_value is None or value > best_value:
                    exact = check(r.instance, route, amounts, tol=0)["objective"]
                    if value != exact:
                        raise ArithmeticError("Beam allocation disagrees with independent route check")
                    publish({"route": route, "amounts": amounts}, exact)
                    if floor >= r.prior:
                        return best
    if best is not None and floor > initial_floor and monotonic() < deadline:
        improve_seed(r, best, deadline, fixed_work=True,
                     emit=lambda witness, value: publish(witness, value, polished=True))
    return best
