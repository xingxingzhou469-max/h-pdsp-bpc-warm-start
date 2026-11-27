"""Keep station integrality when closing a jointly priced fleet support."""
from bisect import bisect_right
from fractions import Fraction as F
from functools import reduce
from itertools import accumulate
from math import gcd, lcm
import os
from pathlib import Path
import sys

V32 = Path(os.environ.get("V33_V32", Path(__file__).resolve().parents[2] / "mv-v32-integration"))
sys.path.insert(0, str(V32))
from route.model import rational, read_model
from zone.check import replay as zone_replay


def ticks(values, limit):
    den = lcm(*(v.denominator for v in values))
    raw = [int(v * den) for v in values]
    div = reduce(gcd, raw, 0) or 1
    weights = [v // div for v in raw]
    return (weights, F(div, den)) if sum(weights) <= limit else (None, None)


def minimum_cost(sizes, costs):
    """For every exactly attainable attribute sum, least cost of a distinct station subset."""
    inf = 1 + sum(abs(x) for x in costs)
    best = [0] + [inf] * sum(sizes)
    used = 0
    for amount, cost in zip(sizes, costs):
        if not amount:
            continue
        for q in range(used, -1, -1):
            if best[q] != inf:
                best[q+amount] = min(best[q+amount], best[q]+cost)
        used += amount
    return [None if x==inf else x for x in best]


def bound(instance, k, parent, verify=True):
    model = read_model(instance)
    previous = zone_replay(instance,k,parent) if verify else F(parent["upper"])
    if not previous or not parent["supports"]:
        return previous
    st,su=ticks(model.stock,6000)
    pop,pu=ticks(model.people,12000)
    if st is None or pop is None:
        return previous
    max_p = [sum(pop)] * (sum(st)+1)
    constraints = []
    def tighten(costs, budget):
        qcost = minimum_cost(st,costs)
        pcost = minimum_cost(pop,costs)
        constraints.append((qcost,pcost,budget))
        by_cost = sorted((val,p) for p,val in enumerate(pcost) if val is not None)
        costs_sorted, populations = zip(*by_cost)
        reach = list(accumulate(populations, max))
        for q,c in enumerate(qcost):
            idx = -1 if c is None else bisect_right(costs_sorted, budget-c)-1
            max_p[q] = min(max_p[q], reach[idx] if idx>=0 else 0)
    supports = []
    for line in parent["supports"]:
        beta = F(line["beta"])
        weights = model.weights(beta)
        costs = [w-p for w,p in zip(weights,line["prices"])]
        limit = k*line["route_bound"]
        tighten(costs,limit)
        face = F((limit+sum(line["prices"]))**2,1)/beta
        supports.append((face,costs,limit))
    travel = [tuple(map(rational,row)) for row in instance["travel"]]
    site_count = len(travel)
    # Productive routes contain two distinct non-depot stations, so each vertex has two distinct neighbors.
    def visit_cost(i):
        incoming = sorted((travel[j][i],j) for j in range(site_count) if j!=i)[:2]
        outgoing = sorted((travel[i][j],j) for j in range(site_count) if j!=i)[:2]
        return min(x+y for x,h in incoming for y,j in outgoing if h!=j)/2
    service = list(map(rational,instance["service"]))
    incidence = tuple(visit_cost(i)+service[i] for i in range(1,site_count))
    depot = visit_cost(0)
    timespecs = [(incidence,depot), (model.in_cost,min(row[0] for row in travel[1:])),
                 (model.out_cost,min(travel[0][1:]))]
    for times,extra in timespecs:
        den=lcm(model.limit.denominator,extra.denominator,*(x.denominator for x in times))
        tighten([int(t*den) for t in times], int(k*max(0,model.limit-extra)*den))
    # Each nonnegative sum of two valid inequalities still uses the same selected stations.
    ordered = sorted(supports, key=lambda row:row[0])[:4]
    for i,(_,a,x) in enumerate(ordered):
        for _,b,y in ordered[:i]:
            tighten([v+w for v,w in zip(a,b)],x+y)
    product = 0
    for q in sorted(range(1,len(max_p)), key=lambda j:j*max_p[j], reverse=True):
        if q*max_p[q] <= product:
            break
        for p in range(max_p[q],product//q,-1):
            if all(s[q] is not None and d[p] is not None and s[q]+d[p]<=v for s,d,v in constraints):
                product = q*p
                break
    return min(previous, product*su*pu/model.population)


def replay(instance,k,doc):
    if doc.get("method")!="fleet-selection-v33" or doc.get("vehicles")!=k:
        raise ValueError("selection identity differs")
    val=bound(instance,k,doc["parent"],verify=True)
    if val!=F(doc["upper"]):
        raise ValueError("selection certificate differs")
    return val
