"""Exact search by feasible visit set and, where necessary, its visit orders."""

from fractions import Fraction
from heapq import heappop, heappush
from time import monotonic

from .allocation import allocate
from .check import check
from .prefix import Label, box_bound, prefix_bound


_MAX_DP_RECORDS = 150_000


def _nodes(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def solve_subset(r, deadline, seed, *, emit=None, max_records=_MAX_DP_RECORDS):
    """Solve using shortest-time subset DP and complete search over unresolved sets."""
    next_emit = monotonic()
    route, amounts = (None, None) if seed is None else (seed["route"], seed["amounts"])
    lb = None if seed is None else check(r.instance, route, amounts, tol=0)["objective"]
    lb_num = None if lb is None else lb * r.units
    metrics = dict(
        phase="preprocessing", stop_reason=None, dp_expanded=0, dp_peak_records=0,
        dp_record_limit=max_records, feasible_sets=0, screened_sets=0,
        order_independent_sets=0, order_states=0, allocations=0,
        bound_seconds=0.0, dp_seconds=0.0, order_seconds=0.0,
    )
    pending = []

    def result(ub_num, *, exhausted=False):
        ub = None if ub_num is None else Fraction(ub_num, r.units)
        if lb is not None:
            ub = lb if ub is None else max(lb, ub)
        if exhausted and lb is None:
            status = "infeasible"
        elif lb is not None and ub == lb:
            status = "optimal"
        elif metrics["stop_reason"] == "dp_record_limit":
            status = "resource_limit"
        else:
            status = "timeout"
        return {
            "status": status, "lb": lb, "ub": ub, "route": route, "amounts": amounts,
            "lb_exact": None if lb is None else str(lb),
            "ub_exact": None if ub is None else str(ub),
            "certificate": "rational_subset_frontier",
            "metrics": {**metrics, "pending_sets": len(pending)},
        }

    def checkpoint(ub_num, *, force=False):
        nonlocal next_emit
        now = monotonic()
        if now >= deadline:
            metrics["stop_reason"] = "deadline"
            return False
        if emit and (force or now >= next_emit):
            emit(result(ub_num))
            next_emit = now + 0.5
        return True

    def room(records):
        if max_records is not None and records > max_records:
            metrics["stop_reason"] = "dp_record_limit"
            return False
        metrics["dp_peak_records"] = max(metrics["dp_peak_records"], records)
        return True

    def evaluate(stops, upper, frontier_bound):
        nonlocal route, amounts, lb, lb_num
        value, allocations = allocate(r.instance, stops)
        metrics["allocations"] += 1
        value_num = value * r.units
        if value_num > upper:
            raise ArithmeticError("A feasible route exceeds its subset upper bound")
        if lb is None or value > lb:
            verified = check(r.instance, stops, allocations, tol=0)["objective"]
            if verified != value:
                raise ArithmeticError("Independent check disagrees with the subset allocation")
            if not checkpoint(frontier_bound):
                return None
            route, amounts, lb, lb_num = stops, allocations, verified, value_num
            checkpoint(frontier_bound, force=True)
        return value_num

    if not r.p or not r.d:
        return result(None, exhausted=True)
    if not checkpoint(r.prior):
        return result(r.prior)
    tic = monotonic()
    r.prepare_bounds()
    global_bound = prefix_bound(r, Label(), joint=True)
    metrics["bound_seconds"] += monotonic() - tic
    if not checkpoint(r.prior):
        return result(r.prior)
    if global_bound is None:
        if lb is not None:
            raise ArithmeticError("The root time bound excludes the verified seed")
        return result(None, exhausted=True)
    if lb_num is not None and lb_num > global_bound:
        raise ArithmeticError("The root upper bound is lower than the verified seed")

    metrics["phase"] = "subsets"
    dp_start = monotonic()
    if not checkpoint(global_bound, force=True):
        return result(global_bound)
    pickup_mask = sum(1 << v for v in r.p)
    delivery_mask = sum(1 << v for v in r.d)
    active_mask = pickup_mask | delivery_mask
    layer = {}
    for v in r.p:
        elapsed = r.travel[0][v] + r.service[v]
        if elapsed + r.dist[v][0] <= r.limit:
            layer[1 << v] = {v: (elapsed, (v,))}
    layer_size = len(layer)

    def dp_result(*, exhausted=False):
        metrics["dp_seconds"] = monotonic() - dp_start
        return result(global_bound, exhausted=exhausted)

    if not room(layer_size):
        return dp_result()
    while layer:
        next_layer = {}
        next_size = 0
        for mask, endings in layer.items():
            if lb_num is not None and lb_num >= global_bound:
                return dp_result(exhausted=True)
            if not checkpoint(global_bound):
                return dp_result()

            fastest = min(
                ((elapsed + r.travel[last][0], stops)
                 for last, (elapsed, stops) in endings.items()
                 if r.pop[last] and elapsed + r.travel[last][0] <= r.limit),
                default=None,
            )
            if fastest is not None:
                metrics["feasible_sets"] += 1
                donors = list(_nodes(mask & pickup_mask))
                pops = sorted((r.pop[i] for i in _nodes(mask & delivery_mask)), reverse=True)
                tic = monotonic()
                flow = min(sum(r.supply[i] for i in donors), len(pops) * r.capacity)
                upper = min(global_bound, box_bound(pops, flow, r.capacity))
                metrics["bound_seconds"] += monotonic() - tic
                if lb_num is not None and upper <= lb_num:
                    metrics["screened_sets"] += 1
                else:
                    _, fastest_route = fastest
                    value_num = evaluate(fastest_route, upper, global_bound)
                    if value_num is None:
                        return dp_result()
                    if len(donors) == 1:
                        # The unique pickup must be first. The rest only empties the truck.
                        if value_num != upper:
                            raise ArithmeticError("Single-donor proportional bound did not close")
                        metrics["order_independent_sets"] += 1
                    elif upper > lb_num:
                        if not room(layer_size + next_size + len(pending) + 1):
                            return dp_result()
                        heappush(pending, (-upper, mask, fastest_route))

            remaining = active_mask ^ mask
            if not remaining & delivery_mask:
                continue
            for last, (elapsed, stops) in endings.items():
                metrics["dp_expanded"] += 1
                for v in _nodes(remaining):
                    if not checkpoint(global_bound):
                        return dp_result()
                    new_time = elapsed + r.travel[last][v] + r.service[v]
                    if new_time + r.dist[v][0] > r.limit:
                        continue
                    targets = next_layer.setdefault(mask | (1 << v), {})
                    old = targets.get(v)
                    if old is not None and old[0] <= new_time:
                        continue
                    if old is None:
                        if not room(layer_size + next_size + len(pending) + 1):
                            return dp_result()
                        next_size += 1
                    targets[v] = new_time, (*stops, v)
        layer, layer_size = next_layer, next_size

    metrics["dp_seconds"] = monotonic() - dp_start
    metrics["phase"] = "orders"
    order_start = monotonic()

    def order_result(upper, *, exhausted=False):
        metrics["order_seconds"] = monotonic() - order_start
        return result(upper, exhausted=exhausted)

    if pending and not checkpoint(-pending[0][0], force=True):
        return order_result(-pending[0][0])
    while pending:
        negative_upper, mask, fastest_route = pending[0]
        upper = -negative_upper
        if lb_num is not None and upper <= lb_num:
            return order_result(upper, exhausted=True)
        # Every permutation with the right endpoints is considered. The
        # shortest-time DP's discarded orders are restored in this search.
        stack = [(0, (), mask, 0)]
        while stack:
            if lb_num is not None and upper <= lb_num:
                break
            if not checkpoint(upper):
                return order_result(upper)
            last, stops, remaining, elapsed = stack.pop()
            metrics["order_states"] += 1
            if not remaining:
                if stops != fastest_route and elapsed + r.travel[last][0] <= r.limit:
                    if evaluate(stops, upper, upper) is None:
                        return order_result(upper)
                continue
            eligible = remaining if last else remaining & pickup_mask
            if remaining.bit_count() == 1:
                eligible &= delivery_mask
            for v in _nodes(eligible):
                new_time = elapsed + r.travel[last][v] + r.service[v]
                if new_time + r.dist[v][0] <= r.limit:
                    stack.append((v, (*stops, v), remaining ^ (1 << v), new_time))
        heappop(pending)
    return order_result(None, exhausted=True)
