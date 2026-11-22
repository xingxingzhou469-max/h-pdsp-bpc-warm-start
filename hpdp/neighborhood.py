"""Deterministic, time-limited neighborhood improvement of a shared seed."""

from heapq import heappush, heapreplace
from itertools import pairwise
from time import monotonic

from .allocation import allocate
from .check import check
from .heuristic import throughput


_MAX_SECONDS = 0.65
_MAX_SCANS = 7
_SHORTLIST = 4


def _time(r, route):
    return sum(r.travel[a][b] + r.service[b] for a, b in pairwise((0, *route, 0)))


def _moves(r, route, elapsed):
    travel = r.travel
    walk = (0, *route, 0)
    n = len(route)
    unused = sorted(set((*r.p, *r.d)) - set(route))
    # A reversal changes every internal directed arc. A prefix sum gives
    # the exact internal contribution for any segment.
    reverse_delta = [0]
    for a, b in pairwise(route):
        reverse_delta.append(reverse_delta[-1] + travel[b][a] - travel[a][b])
    for i in range(n - 1):
        for j in range(i + 1, n):
            before, a, b, after = walk[i], route[i], route[j], walk[j + 2]
            delta = (
                travel[before][b] + travel[a][after]
                - travel[before][a] - travel[b][after]
                + reverse_delta[j] - reverse_delta[i]
            )
            yield "reverse", route[:i] + route[i:j+1][::-1] + route[j+1:], elapsed + delta

    for i, node in enumerate(route):
        left, right = walk[i], walk[i + 2]
        removed = route[:i] + route[i+1:]
        reduced = elapsed - travel[left][node] - r.service[node] - travel[node][right]
        removed_time = reduced + travel[left][right]
        if n > 2:
            yield "remove", removed, removed_time
        for replacement in unused:
            replacement_time = (
                reduced + travel[left][replacement] + r.service[replacement] + travel[replacement][right]
            )
            yield "replace", route[:i] + (replacement,) + route[i+1:], replacement_time
        for j, (a, b) in enumerate(pairwise((0, *removed, 0))):
            if i == j:
                continue
            moved_time = (
                removed_time + travel[a][node] + r.service[node] + travel[node][b] - travel[a][b]
            )
            yield "relocate", removed[:j] + (node,) + removed[j:], moved_time

    for node in unused:
        for j, (a, b) in enumerate(pairwise(walk)):
            delta = travel[a][node] + r.service[node] + travel[node][b] - travel[a][b]
            yield "insert", route[:j] + (node,) + route[j:], elapsed + delta


def _full_route(r, start, alternating):
    """Try a long route with either nearest-neighbor or alternating role choices."""
    last = min(r.d, key=lambda d: (r.travel[d][0], -r.pop[d], d))
    remaining = set((*r.p, *r.d)) - {start, last}
    route = [start]
    load = r.supply[start]
    while remaining:
        candidates = remaining
        if alternating:
            preferred = {v for v in remaining if bool(r.pop[v]) == bool(load)}
            if preferred:
                candidates = preferred
        node = min(candidates, key=lambda v: (r.travel[route[-1]][v] + r.service[v], v))
        load = min(r.capacity, load + r.supply[node]) if r.supply[node] else 0
        route.append(node)
        remaining.remove(node)
    return (*route, last)


def _three_stop(r, deadline):
    active = sorted((*r.p, *r.d))
    for p in r.p:
        for d in r.d:
            for v in active:
                if monotonic() >= deadline:
                    return None
                if v != p and v != d and _time(r, (p, v, d)) <= r.limit:
                    return (p, v, d)
    return None


def _shortlist(r, route, elapsed, deadline, expanded):
    pools = {}
    bridge = None
    current_proxy = throughput(r, route) * sum(r.pop[v] for v in route)
    for kind, candidate, time in _moves(r, route, elapsed):
        if monotonic() >= deadline:
            break
        if time > r.limit or candidate in expanded:
            continue
        if not r.supply[candidate[0]] or not r.pop[candidate[-1]]:
            continue
        proxy = throughput(r, candidate) * sum(r.pop[v] for v in candidate)
        ranking = (proxy, -time, candidate)
        pool = pools.setdefault(kind, [])
        if len(pool) < _SHORTLIST:
            heappush(pool, ranking)
        elif ranking > pool[0]:
            heapreplace(pool, ranking)
        if time < elapsed and 5 * proxy >= 4 * current_proxy:
            entry = (-time, proxy, candidate)
            bridge = entry if bridge is None else max(bridge, entry)
    ranked = sorted((entry for pool in pools.values() for entry in pool), reverse=True)
    candidates = {candidate: -negative for _, negative, candidate in ranked}
    bridge_route = None if bridge is None else bridge[2]
    if bridge_route is not None:
        candidates[bridge_route] = -bridge[0]
    return candidates, bridge_route


def improve_seed(r, seed, deadline, *, emit=None, fixed_work=False):
    """Return the best strictly verified seed found before the shared deadline."""
    hard_stop = deadline if fixed_work else min(deadline, monotonic() + _MAX_SECONDS)
    if not r.p or not r.d or monotonic() >= hard_stop:
        return seed
    best = seed
    best_value = None if seed is None else check(
        r.instance, seed["route"], seed["amounts"], tol=0
    )["objective"]
    if best_value is not None and best_value * r.units >= r.prior:
        return seed
    expanded, scored = set(), {}

    def score(route):
        nonlocal best, best_value
        if route in scored:
            return scored[route]
        if monotonic() >= hard_stop:
            return None
        value, amounts = allocate(r.instance, route)
        if best_value is None or value > best_value:
            checked = check(r.instance, route, amounts, tol=0)["objective"]
            if value != checked:
                raise ArithmeticError("Neighborhood scoring disagrees with independent checker")
            if monotonic() >= hard_stop:
                return None
            best = {"route": route, "amounts": amounts}
            best_value = checked
            if emit:
                emit(best, checked)
        scored[route] = value
        return value

    starts = set()
    if seed is not None:
        initial = tuple(seed["route"])
        starts.add(initial)
        first = initial[0]
    else:
        first = min(r.p, key=lambda p: (r.travel[0][p] + r.service[p], -r.supply[p], p))
    for alternating in (False, True):
        starts.add(_full_route(r, first, alternating))

    rank = None
    for route in sorted(starts):
        elapsed = _time(r, route)
        if elapsed > r.limit:
            continue
        value = score(route)
        if value is not None:
            key = (value, -elapsed, route)
            rank = key if rank is None else max(rank, key)
    # If the pair-based seed found no route on a directed network, check
    # three-stop walks directly, including routes with a zero-amount transit.
    if rank is None and seed is None:
        route = _three_stop(r, hard_stop)
        if route is not None:
            value = score(route)
            if value is not None:
                rank = (value, -_time(r, route), route)

    for _ in range(_MAX_SCANS):
        if rank is None or monotonic() >= hard_stop:
            break
        value, negative_time, route = rank
        elapsed = -negative_time
        expanded.add(route)
        options, bridge = _shortlist(r, route, elapsed, hard_stop, expanded)
        step = None
        for candidate, time in options.items():
            estimated = score(candidate)
            if estimated is not None:
                key = (estimated, -time, candidate)
                if key[:2] > rank[:2]:
                    step = key if step is None else max(step, key)
        if step is None and bridge in scored:
            step = (scored[bridge], -options[bridge], bridge)
        rank = step
    return best
