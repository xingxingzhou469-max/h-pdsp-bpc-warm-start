"""Order-audit experiments using alternating pickup and delivery blocks."""

from fractions import Fraction
from time import monotonic

from hpdp.allocation import allocate
from hpdp.check import check
from hpdp.dense_overnight import _pending_sets
from .certificate_support import _box_numerator, _nodes, _reverse_tails, rank_certificate


def _extend_blocks(r, blocks, v):
    """Positive entries are capped pickup totals; negative entries are populations."""
    if r.supply[v]:
        amount = r.supply[v]
        if blocks and blocks[-1] > 0:
            return (*blocks[:-1], min(r.capacity, blocks[-1] + amount))
        return (*blocks, amount)
    people = r.pop[v]
    if blocks[-1] < 0:
        return (*blocks[:-1], blocks[-1] - people)
    return (*blocks, -people)


def audit_block_orders(r, mask, upper, reverse, cutoff, publish, deadline, metrics,
                       *, merge_prefixes=True):
    """Exhaust one set, merging equivalent blocks and dominated prefixes."""
    sites = [bit.bit_length() for bit in _nodes(mask)]
    people = sorted((r.pop[v] for v in sites if r.pop[v]), reverse=True)
    covered = sum(people)
    pickups = sum(1 << (v - 1) for v in sites if r.supply[v])
    budget = r.limit - sum(r.service[v] for v in sites)
    by_flow = {}
    fastest = {}
    evaluated_blocks = set()
    for name in ("prefix_dominance_cuts", "leaf_equivalence_cuts",
                 "max_prefix_states", "max_leaf_signatures"):
        metrics.setdefault(name, 0)
    route = []

    def search(remaining, last, spent, unloaded, load, future_supply, future_deliveries, blocks):
        if monotonic() >= deadline:
            return False
        current_cutoff = cutoff()
        if upper <= current_cutoff:
            return True
        metrics["expanded_prefixes"] += 1
        flow = min(unloaded + load + future_supply, unloaded + r.capacity * future_deliveries)
        if flow * covered <= current_cutoff:
            metrics["quantity_pruned_prefixes"] += 1
            return True
        if flow not in by_flow:
            by_flow[flow] = _box_numerator(people, r.capacity, flow, covered)
        bound = by_flow[flow]
        if bound <= current_cutoff:
            metrics["quantity_pruned_prefixes"] += 1
            return True
        if not remaining:
            if blocks in evaluated_blocks:
                metrics["leaf_equivalence_cuts"] += 1
                return True
            value, amounts = allocate(r.instance, route)
            audited = rank_certificate(r.instance, route, amounts)
            if audited["upper_bound"] != value or value * r.units > min(bound, upper):
                raise ArithmeticError("Fixed-route allocation exceeds its independently checked bound")
            metrics["audited_orders"] += 1
            if monotonic() >= deadline:
                return False
            evaluated_blocks.add(blocks)
            metrics["max_leaf_signatures"] = max(metrics["max_leaf_signatures"], len(evaluated_blocks))
            publish(value, route, amounts)
            return True

        if merge_prefixes:
            state = (remaining, last, blocks)
            if fastest.get(state, spent+1) <= spent:
                metrics["prefix_dominance_cuts"] += 1
                return True
            fastest[state] = spent
            metrics["max_prefix_states"] = max(metrics["max_prefix_states"], len(fastest))

        choices = remaining if last else remaining & pickups
        for bit in _nodes(choices):
            v = bit.bit_length()
            step = int(reverse.arcs[v, last])
            tail = int(reverse.paths[remaining, v - 1])
            if tail == reverse.unreachable or spent + step + tail > budget:
                metrics["time_pruned_children"] += 1
                continue
            route.append(v)
            extended = _extend_blocks(r, blocks, v)
            if r.supply[v]:
                closed = search(remaining ^ bit, v, spent + step, unloaded,
                                min(r.capacity, load + r.supply[v]), future_supply - r.supply[v],
                                future_deliveries, extended)
            else:
                closed = search(remaining ^ bit, v, spent + step, unloaded + load,
                                0, future_supply, future_deliveries - 1, extended)
            route.pop()
            if not closed:
                return False
            if upper <= cutoff():
                return True
        return True

    return search(mask, 0, 0, 0, 0, sum(r.supply[v] for v in sites), len(people), ())


def solve_dense_blocks(resources, deadline, seed, *, emit=None):
    """Return an exact dense frontier using block-equivalent permutations."""
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
        cert = "rational_dense_blocks"
        return result(None)
    cert = "rational_dense_blocks"
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
        closed = audit_block_orders(r, mask, upper, reverse, cutoff, publish, deadline, metrics)
        metrics["order_seconds"] = monotonic() - orders_started
        if not closed:
            return result(upper)
        metrics.update(sets_audited=i+1, sets_remaining=len(pending)-i-1)
        checkpoint(-pending[i+1][0] if i+1 < len(pending) else floor)
    return result(floor)
