"""Search a fixed visit set with full pickups and proportional deliveries."""

from fractions import Fraction
from numbers import Integral
from time import monotonic

from .check import check


MAX_LABELS = 100_000


def _assignment_costs(r, nodes, budget, deadline):
    """One exact assignment dual, then the eligible reduced-cost arcs."""
    size = len(nodes)
    arcs = []
    for a, u in enumerate(nodes):
        if monotonic() >= deadline:
            return "timeout", None
        arcs.append([
            (b, r.travel[u][v]) for b, v in enumerate(nodes)
            if a != b and (a != 0 or r.supply[v]) and (b != 0 or r.pop[u])
            and r.travel[u][v] <= budget
        ])
    if any(not row for row in arcs):
        return "fixed_infeasible", None
    outgoing = [min(t for _, t in row) for row in arcs]
    incoming = [None]*size
    for a, row in enumerate(arcs):
        for b, t in row:
            value = t-outgoing[a]
            if incoming[b] is None or value<incoming[b]:
                incoming[b]=value
    if any(v is None for v in incoming):
        return "fixed_infeasible", None
    lower = sum(outgoing)+sum(incoming)
    if lower > budget:
        return "fixed_infeasible", lower
    rows = []
    columns = [[] for _ in nodes]
    for a, row in enumerate(arcs):
        reduced = sorted((t-outgoing[a]-incoming[b],t,b) for b,t in row)
        rows.append(reduced)
        for cost, _, b in reduced:
            columns[b].append((cost,a))
    for column in columns:
        column.sort()
    # A direct return is used only for a completed visit set.
    returns = [next((cost for cost, _, b in row if b==0), None) for row in rows]
    return "ready", (lower, rows, columns, returns)


def _can_finish(rows, columns, remaining, last, slack):
    """Necessary reduced travel from last through remaining sites to the depot."""
    sources = remaining | (1 << last)
    targets = remaining | 1
    cost = 0
    todo = sources
    while todo:
        bit = todo & -todo
        a = bit.bit_length()-1
        todo ^= bit
        for arc, _, b in rows[a]:
            if targets & (1<<b) and (a!=last or b!=0):
                cost += arc
                break
        else:
            return False
        if cost > slack:
            return False
    cost = 0
    todo = targets
    while todo:
        bit = todo & -todo
        b = bit.bit_length()-1
        todo ^= bit
        for arc, a in columns[b]:
            if sources & (1<<a) and (a!=last or b!=0):
                cost += arc
                break
        else:
            return False
        if cost > slack:
            return False
    return True


def find_order(r, sites, deadline):
    """Return a verified equal-bound order, or a result scoped to the fixed quantities.

    All selected donors contribute their clipped stock. Each selected agency
    receives its population share of that stock. Infeasibility never applies
    to other quantities on the same set. A malformed selection raises ValueError.
    """
    started = monotonic()
    selection = tuple(sites)
    if any(isinstance(v,bool) or not isinstance(v,Integral) or v not in r.nodes for v in selection):
        raise ValueError("Visit sites must be non-depot model indices")
    if len(set(selection))!=len(selection):
        raise ValueError("Each visit site may appear only once in the selection")
    nodes = (0,*sorted(map(int,selection)))
    people = sum(r.pop[v] for v in nodes)
    stock = sum(r.supply[v] for v in nodes)
    target = Fraction(stock*people,r.units)
    budget = r.limit-sum(r.service[v] for v in nodes)
    metrics = dict(states=0,expanded=0,edges=0,dominated=0,time_cuts=0,capacity_cuts=0,
                   memo_records=0,memo_omitted=0,assignment_travel_lower=None,travel_budget=budget)
    best = {}

    def result(status, reason=None):
        completed = monotonic()
        if completed>=deadline:
            status, reason = "timeout", "deadline"
        witness = best if status=="found" else {}
        value = witness.get("value")
        return {
            "status":status, "reason":reason, "target":target, "target_exact":str(target),
            "route":witness.get("route"), "amounts":witness.get("amounts"), "route_time":witness.get("time"),
            "lb":value, "lb_exact":None if value is None else str(value),
            "metrics":{**metrics,"seconds":completed-started},
        }

    if monotonic() >= deadline:
        return result("timeout","deadline")
    if not people or not stock:
        return result("fixed_infeasible","missing_role")
    if budget < 0:
        return result("fixed_infeasible","services_exceed_limit")
    if any(r.pop[v]*stock > r.capacity*people for v in nodes):
        return result("fixed_infeasible","single_delivery_exceeds_capacity")

    status, costs = _assignment_costs(r,nodes,budget,deadline)
    if status=="timeout" or monotonic()>=deadline:
        return result("timeout","deadline")
    if status=="fixed_infeasible":
        metrics["assignment_travel_lower"]=costs
        return result(status,"assignment_time")
    lower, rows, columns, returns = costs
    metrics["assignment_travel_lower"] = lower
    slack = budget-lower
    limit = r.capacity*people
    changes = [r.supply[v]*people-r.pop[v]*stock for v in nodes]
    all_sites = ((1<<len(nodes))-1)^1
    label_bits = len(nodes).bit_length()
    memo = {}
    route = []

    def viable(mask,last,spent):
        metrics["states"]+=1
        key=(mask<<label_bits)|last
        previous=memo.get(key)
        if previous is not None and previous<=spent:
            metrics["dominated"]+=1
            return False
        if previous is not None or len(memo)<MAX_LABELS:
            memo[key]=spent
            metrics["memo_records"]=len(memo)
        else:
            metrics["memo_omitted"]+=1
        if not _can_finish(rows,columns,mask,last,slack-spent):
            metrics["time_cuts"]+=1
            return False
        metrics["expanded"]+=1
        return True

    # A stack frame retains the candidate iterator at each depth. Only
    # completed frames can subsequently be revisited at the same depth.
    if not viable(all_sites,0,0):
        return result("timeout","deadline") if monotonic()>=deadline else result(
            "fixed_infeasible","residual_time")
    stack = [(all_sites,0,0,iter(rows[0]))]
    while stack:
        if monotonic()>=deadline:
            return result("timeout","deadline")
        remaining, spent, load, choices = stack[-1]
        arc = next(choices,None)
        if arc is None:
            stack.pop()
            if route:
                route.pop()
            continue
        charge, _, v = arc
        bit=1<<v
        if not (remaining&bit):
            continue
        metrics["edges"]+=1
        new_time=spent+charge
        if new_time>slack:
            metrics["time_cuts"]+=1
            continue
        new_load=load+changes[v]
        if not 0<=new_load<=limit:
            metrics["capacity_cuts"]+=1
            continue
        rest=remaining^bit
        if rest:
            if viable(rest,v,new_time):
                route.append(v)
                stack.append((rest,new_time,new_load,iter(rows[v])))
            continue

        back=returns[v]
        if back is None or new_time+back>slack:
            metrics["time_cuts"]+=1
            continue
        order=tuple(nodes[i] for i in (*route,v))
        amounts=[
            Fraction(r.supply[u],r.flow_scale) if r.supply[u]
            else Fraction(r.pop[u]*stock, people*r.flow_scale)
            for u in order
        ]
        verified=check(r.instance,order,amounts,tol=0)
        if verified["objective"]!=target:
            raise ArithmeticError("Fixed-order witness disagrees with the expected proportional value")
        if monotonic()>=deadline:
            return result("timeout","deadline")
        best.update(route=order,amounts=amounts,value=verified["objective"],time=verified["time"])
        return result("found")
    if monotonic()>=deadline:
        return result("timeout","deadline")
    return result("fixed_infeasible","exhausted")
