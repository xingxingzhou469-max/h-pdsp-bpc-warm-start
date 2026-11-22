"""Propose routing upper-bound certificates from numerical LPs."""

from dataclasses import dataclass
from math import gcd
from time import monotonic

from .cert_verifier import branch_row
from .cut_cert import cut_row
from .prefix import Resources
from .priced_routes import cap_product
from .routing_exact import _route_reward, flow_cap, rewards, routing
from .routing_lp_bound import _suggest_lp, matrix, separate


@dataclass
class Node:
    leaf: dict
    hint: object

    @property
    def bound(self):
        return self.leaf["cap_product_bound"]


def _slopes(r, model, seed_route):
    flow, people = flow_cap(r), r.people
    seed_flow = _route_reward(r, model, 1, 0, seed_route or ())
    seed_pop = sum(r.pop[v] for v in seed_route or ())
    relative = [
        (seed_pop, seed_flow), (3*people, 5*flow), (4*people, 5*flow),
        (3*seed_pop, 4*seed_flow), (people, 2*flow),
        (5*seed_pop, 4*seed_flow), (people, flow),
    ]
    slopes = []
    for a, b in relative:
        if min(a, b) <= 0:
            continue
        divisor = gcd(a, b)
        slope = (a//divisor, b//divisor)
        if slope not in slopes:
            slopes.append(slope)
    return slopes


def propose(instance, instance_sha256, seed_route, deadline, *, emit, cuts=(), branch_depth=2):
    """Emit improved full certificate documents until the deadline or callback stop."""
    r = Resources(instance)
    flow = flow_cap(r)
    model = routing(r, pickup_blocks=sum(r.supply)>r.capacity)
    eq, eq_rhs = matrix(model.degrees, model.columns)
    pool = [(spec, cut_row(r, model, spec)) for spec in cuts]
    known = set()
    base_rows = [model.time, *model.flow_rows]
    best = r.prior + 1
    running = True

    def publish(slope, nodes):
        nonlocal best, running
        bound = max(node.bound for node in nodes)
        if running and bound < best and monotonic() < deadline:
            best = bound
            running = emit(dict(version=1, instance_sha256=instance_sha256,
                                slope=list(slope), pickup_blocks=model.flow_ceiling is not None,
                                leaves=[node.leaf for node in nodes], cap_product_bound=bound))

    def strengthen(slope, path, end, rounds, on_bound):
        objective = rewards(r, model, *slope)
        branches = [branch_row(model, test) for test in path]
        incumbent = None
        for iteration in range(rounds):
            if not running or monotonic() >= end:
                break
            inequalities = base_rows + [row for _,row in pool] + branches
            info, hint = _suggest_lp(model, objective, inequalities, eq, eq_rhs, end, 10**8)
            if info.get("certificate_status") != "integer_replay":
                break
            reward = info["numerator"] // info["denominator"]
            bound = -1 if reward < 0 else cap_product(reward, *slope, flow, r.people)
            if incumbent is None or bound <= incumbent.bound:
                cert = info["certificate"]
                multipliers = cert["nonnegative"]
                fixed = len(base_rows)
                active = [(spec, y) for (spec,_),y in zip(pool,multipliers[fixed:fixed+len(pool)])
                          if y]
                leaf = dict(cuts=[spec for spec,_ in active], branch_rows=path,
                            integer_multipliers={
                                "nonnegative": [*multipliers[:fixed], *(y for _,y in active),
                                                *multipliers[fixed+len(pool):]],
                                "free": cert["free"]},
                            denominator=cert["denominator"], cap_product_bound=bound)
                incumbent = Node(leaf, hint)
                on_bound(incumbent)
            if not running or bound < 0 or iteration+1 >= rounds:
                break
            fresh = separate(model, hint, known, end, 96)
            if not fresh:
                break
            known.update(fresh)
            for side, target in fresh:
                spec = {"type":"SEC", "set":list(side), "target":target}
                pool.append((spec, cut_row(r, model, spec)))
        return incumbent

    start = monotonic()
    remaining = max(0.0, deadline-start)
    root_end = start + (0.68 if branch_depth else 1) * remaining
    top, top_slope = None, None
    slopes = _slopes(r, model, seed_route)
    for index, slope in enumerate(slopes):
        now = monotonic()
        if not running or now >= root_end:
            break
        # The first slope builds the shared pool. Later slopes reuse its rows
        # and spend their time on different exposed faces of the relaxation.
        budget = (root_end-now) * (0.52 if index == 0 else 1/(len(slopes)-index))
        node = strengthen(slope, [], now+budget, 30 if index == 0 else 12,
                          lambda candidate: publish(slope, [candidate]))
        if node is not None and (top is None or node.bound < top.bound):
            top, top_slope = node, slope
    if not running or top is None or top.bound < 0 or not branch_depth:
        return

    frontier = [top]
    for depth in range(branch_depth):
        level_end = deadline if depth+1 == branch_depth else monotonic() + (deadline-monotonic())*0.45
        candidates = sorted(frontier, key=lambda node:node.bound, reverse=True)
        for index, node in enumerate(candidates):
            if not running or node.hint is None or monotonic() >= level_end:
                break
            if node.bound < 0 or node.bound < max(leaf.bound for leaf in frontier):
                continue
            path = node.leaf["branch_rows"]
            fixed = {v for v,_ in path}
            weights = []
            for v, col in model.visits.items():
                y = node.hint[col]
                if v not in fixed and 1e-7 < y < 1-1e-7:
                    priority = max(r.supply[v]/flow, r.pop[v]/r.people)
                    weights.append((priority*min(y,1-y), v))
            if not weights:
                continue
            _, v = max(weights)
            children = []
            now = monotonic()
            split_end = now + (level_end-now)/(len(candidates)-index)
            for value in (0, 1):
                child_end = split_end if value else (monotonic()+split_end)/2
                child = strengthen(top_slope, [*path, [v,value]], child_end, 12, lambda _: None)
                if child is None:
                    break
                children.append(child)
            if len(children) == 2 and max(leaf.bound for leaf in children) <= node.bound:
                frontier = [leaf for leaf in frontier if leaf is not node] + children
                publish(top_slope, frontier)
