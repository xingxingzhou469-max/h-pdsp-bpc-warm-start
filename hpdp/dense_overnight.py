"""Exact compact search using one reversed visit table and a continuous order audit."""

from fractions import Fraction
from time import monotonic

from hpdp.allocation import allocate
from hpdp.check import check
from .certificate_support import _box_numerator, _nodes, _reverse_tails, rank_certificate


def _mask_totals(r, sites):
    """Small subset-sum tables for capped supply, population and agency count."""
    totals = [(0, 0, 0)]
    for v in sites:
        totals.extend((s + r.supply[v], p + r.pop[v], d + int(r.pop[v] > 0))
                      for s, p, d in totals.copy())
    return totals


def _pending_sets(r, reverse, cutoff, deadline, metrics):
    split = len(r.nodes) // 2
    low = _mask_totals(r, r.nodes[:split])
    high = _mask_totals(r, r.nodes[split:])
    low_mask = len(low) - 1
    agencies = sorted(((1 << (v - 1), r.pop[v]) for v in r.d), key=lambda x: -x[1])
    pending = []
    for index, mask in enumerate(reverse.feasible.tolist()):
        if index % 1024 == 0 and monotonic() >= deadline:
            return None
        metrics["sets_scanned"] += 1
        low_supply, low_people, low_count = low[mask & low_mask]
        high_supply, high_people, high_count = high[mask >> split]
        covered = low_people + high_people
        flow = min(low_supply + high_supply, (low_count + high_count) * r.capacity)
        if flow * covered <= cutoff:
            continue
        metrics["box_evaluations"] += 1
        people = [p for bit, p in agencies if mask & bit]
        box = _box_numerator(people, r.capacity, flow, covered)
        if box > cutoff:
            pending.append((-box, mask))
    pending.sort()
    return pending


def _audit_orders(r, mask, upper, reverse, cutoff, publish, deadline, metrics):
    """Exhaust one set, publishing strict improvements without restarting its DFS."""
    sites = [bit.bit_length() for bit in _nodes(mask)]
    people = sorted((r.pop[v] for v in sites if r.pop[v]), reverse=True)
    covered = sum(people)
    pickups = sum(1 << (v - 1) for v in sites if r.supply[v])
    budget = r.limit - sum(r.service[v] for v in sites)
    by_flow = {}
    route = []

    def search(remaining, last, spent, unloaded, load, future_supply, future_deliveries):
        if monotonic() >= deadline:
            return False
        if upper <= cutoff():
            return True
        metrics["expanded_prefixes"] += 1
        flow = min(unloaded + load + future_supply, unloaded + r.capacity * future_deliveries)
        if flow * covered <= cutoff():
            metrics["quantity_pruned_prefixes"] += 1
            return True
        if flow not in by_flow:
            by_flow[flow] = _box_numerator(people, r.capacity, flow, covered)
        bound = by_flow[flow]
        if bound <= cutoff():
            metrics["quantity_pruned_prefixes"] += 1
            return True
        if not remaining:
            value, amounts = allocate(r.instance, route)
            audited = rank_certificate(r.instance, route, amounts)
            if audited["upper_bound"] != value or value * r.units > min(bound, upper):
                raise ArithmeticError("Fixed-route allocation exceeds its independently checked bound")
            metrics["audited_orders"] += 1
            if monotonic() >= deadline:
                return False
            publish(value, route, amounts)
            return True

        choices = remaining if last else remaining & pickups
        for bit in _nodes(choices):
            v = bit.bit_length()
            step = int(reverse.arcs[v, last])
            tail = int(reverse.paths[remaining, v - 1])
            if tail == reverse.unreachable or spent + step + tail > budget:
                metrics["time_pruned_children"] += 1
                continue
            route.append(v)
            if r.supply[v]:
                closed = search(remaining ^ bit, v, spent + step, unloaded,
                                min(r.capacity, load + r.supply[v]), future_supply - r.supply[v],
                                future_deliveries)
            else:
                closed = search(remaining ^ bit, v, spent + step, unloaded + load,
                                0, future_supply, future_deliveries - 1)
            route.pop()
            if not closed:
                return False
            if upper <= cutoff():
                return True
        return True

    return search(mask, 0, 0, 0, 0, sum(r.supply[v] for v in sites), len(people))


def solve_dense(resources, deadline, seed, *, emit=None):
    """Return an exact frontier for 2 to 20 nondepot sites in the compact interface."""
    r = resources
    witness = seed
    lb = None if seed is None else check(r.instance, seed["route"], seed["amounts"],
                                        tol=0)["objective"]
    floor = -1 if lb is None else (lb * r.units) // 1
    metrics = dict(phase="reverse", forward_builds=0, reverse_builds=0, sets_scanned=0,
                   box_evaluations=0, sets_audited=0, audit_calls=0, improvements=0,
                   expanded_prefixes=0, audited_orders=0, time_pruned_children=0,
                   quantity_pruned_prefixes=0)
    cert = "analytic_prior"

    def result(upper):
        ub = None if upper is None else Fraction(upper, r.units)
        if lb is not None and ub is not None:
            # With all remaining set bounds below the witness, the witness
            # itself gives the global optimum.
            ub = max(lb, ub)
        return dict(status="infeasible" if ub is None else "optimal" if lb == ub else "timeout",
                    lb_exact=lb, ub_exact=ub, route=None if witness is None else witness["route"],
                    amounts=None if witness is None else witness["amounts"], certificate=cert,
                    metrics=metrics.copy())

    def checkpoint(upper):
        if emit is not None and monotonic() < deadline:
            emit(result(upper))

    if lb is not None and lb * r.units > r.prior:
        raise ArithmeticError("Witness exceeds the analytic prior")
    if floor == r.prior or monotonic() >= deadline:
        return result(r.prior)
    checkpoint(r.prior)
    started = monotonic()
    reverse = _reverse_tails(r.instance)
    if reverse.scale != r.time_scale:
        raise ArithmeticError("Reversed and original travel times use different units")
    metrics.update(reverse_builds=1, reverse_seconds=monotonic()-started,
                   feasible_sets=len(reverse.feasible), table_cells=reverse.paths.size,
                   table_bytes=reverse.paths.nbytes, phase="set_bounds")
    checkpoint(r.prior)
    started = monotonic()
    pending = _pending_sets(r, reverse, floor, deadline, metrics)
    metrics["bound_seconds"] = monotonic() - started
    if pending is None or monotonic() >= deadline:
        return result(r.prior)
    if not pending and lb is None:
        cert = "rational_dense_reverse"
        return result(None)
    cert = "rational_dense_reverse"
    metrics.update(residual_sets=len(pending), sets_remaining=len(pending), phase="order_audit")
    orders_started = monotonic()

    def cutoff():
        return floor

    def publish(value, route, amounts):
        nonlocal lb, floor, witness
        if lb is not None and value <= lb:
            return
        lb, floor = value, (value * r.units) // 1
        witness = {"route": route.copy(), "amounts": amounts}
        metrics["improvements"] += 1
        metrics["order_seconds"] = monotonic() - orders_started
        if emit is not None:
            emit(result(upper))

    for i, (negative, mask) in enumerate(pending):
        upper = -negative
        metrics.update(sets_remaining=len(pending)-i, active_mask=mask)
        if upper <= floor:
            return result(upper)
        if i == 0:
            checkpoint(upper)
        metrics["audit_calls"] += 1
        closed = _audit_orders(r, mask, upper, reverse, cutoff, publish, deadline, metrics)
        metrics["order_seconds"] = monotonic() - orders_started
        if not closed:
            return result(upper)
        metrics.update(sets_audited=i+1, sets_remaining=len(pending)-i-1)
        checkpoint(-pending[i+1][0] if i+1 < len(pending) else floor)
    return result(floor)
