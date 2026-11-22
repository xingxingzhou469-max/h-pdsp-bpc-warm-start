"""Root-specific integer prices and chronological fairness bounds for a few supplier contexts."""

from collections import Counter
import ctypes
from fractions import Fraction
from functools import lru_cache
import json
from pathlib import Path
import sys
from time import monotonic
import warnings

from scipy.optimize import OptimizeWarning, linprog

from .allocation import allocate
from .check import check
from .mandatory_wide import MandatoryGraph, MandatoryTable


def active_graph(r, lower):
    if (lower is None or not 2 <= len(r.p) <= 5 or not 25 <= len(r.d) <= 63
            or sum(r.supply) <= r.capacity or max(sum(r.supply),r.capacity) >= 2**61
            or max(r.pop)>100000 or not 0<r.limit<=1600):
        return None
    graph=MandatoryGraph(r)
    if (graph.population_unit!=1 or not graph.fits(3) or graph.zeros
            or any(r.travel[u][v]+r.service[v]<=0 for u in r.nodes for v in r.nodes if u!=v)):
        return None
    open_contexts=sum(root.flow*r.people>lower*r.units for root in graph.roots)
    return graph if 1<=open_contexts<=12 else None


def fit(cuts,graph,support,prices,radius,deadline):
    n=graph.n
    rows=[]; rhs=[]
    for route in cuts:
        counts=Counter(route)
        rows.append([1-counts[v] if v in graph.r.d else 0 for v in range(n)]+[-1])
        rhs.append(-sum(graph.weights[v] for v in route))
    bounds=[(0,0) if v not in graph.r.d or v in support else
            (max(0,prices[v]-radius),min(graph.weights[v],prices[v]+radius)) for v in range(n)]
    remaining=deadline-monotonic()
    if remaining<=0:return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',OptimizeWarning)
        out=linprog([0]*n+[1],A_ub=rows,b_ub=rhs,bounds=bounds+[(None,None)],
                    options={'threads':1,'time_limit':remaining})
    if not out.success or monotonic()>=deadline:return None
    return tuple(max(lo,min(hi,round(x))) for (lo,hi),x in zip(bounds,out.x[:-1]))


def priced_root(graph,root,deadline,commit):
    ordered=sorted(graph.r.d,key=lambda v:-graph.r.pop[v])
    track=tuple(ordered[:3]); pivot=graph.weights[ordered[len(ordered)//4]]
    prices=tuple(0 if v in track else max(0,w-pivot) for v,w in enumerate(graph.weights))
    best=None; best_price=prices;cuts=[];seen=set()
    for it in range(35):
        if monotonic()>=deadline:break
        with MandatoryTable(graph,prices,track,True) as table:
            ok,cap=table.root(root,deadline)
            if not ok or cap is None:break
            route=table.route(root)
        if monotonic()>=deadline:break
        if best is None or cap<best:
            best,best_price=cap,prices
            commit(cap)
        sig=tuple(sorted(route))
        if sig not in seen:cuts.append(route);seen.add(sig)
        radius=100 if it<16 else 275 if it<55 else max(graph.weights)
        prop=fit(cuts,graph,track,best_price,radius,deadline)
        if prop is None:break
        if prop==prices:
            prop=fit(cuts,graph,track,best_price,max(graph.weights),deadline)
            if prop is None or prop==prices:break
        prices=prop
    return best,best_price,track,len(cuts)


Report=ctypes.CFUNCTYPE(None,ctypes.c_int,ctypes.POINTER(ctypes.c_int))


@lru_cache(maxsize=1)
def native_branch():
    suffix='dylib' if sys.platform=='darwin' else 'so'
    lib=ctypes.CDLL(str(Path(__file__).with_name(f'libmandatory_wide.{suffix}')))
    fn=lib.hpdp_priced_population_branch
    fn.restype=ctypes.c_int
    fn.argtypes=[ctypes.c_char_p,ctypes.c_double,Report,ctypes.c_char_p,ctypes.c_size_t]
    return fn


def branch(graph,root,prices,tracked,cutoff,deadline,offer):
    r=graph.r
    rows=[[graph.n,r.limit,root.first,root.supplier_mask,len(tracked),r.capacity,cutoff,-1],
          tracked,r.pop,r.supply,prices]
    rows.extend([min(r.limit+1,t+r.service[v]) for v,t in enumerate(row)] for row in r.travel)
    payload=''.join(' '.join(map(str,row))+'\n' for row in rows).encode()
    errors=[]
    @Report
    def candidate(size,nodes):
        if monotonic()>=deadline or errors:return
        try:offer(tuple(nodes[:size]))
        except BaseException as exc:errors.append(exc)
    out=ctypes.create_string_buffer(8192)
    fn=native_branch()
    if monotonic()>=deadline:return None
    status=fn(payload,deadline-monotonic(),candidate,out,len(out))
    if errors:raise errors[0]
    if status!=1:raise RuntimeError(f'integer population search error {status}')
    return json.loads(out.value)


def solve(r,graph,deadline,seed,upper,emit):
    started=monotonic()
    lower=check(r.instance,**seed,tol=0)['objective']; current=seed
    caps={root:Fraction(root.flow*r.people,r.units) for root in graph.roots}
    metrics={'phase':'priced_supplier_fairness','price_tables':0,'branches':0,'expanded':0,'contexts':len(caps)}
    def record():
        bound=min(upper,max(caps.values()))
        if lower>bound:raise ArithmeticError('local rigorous bound contradicts a verified route')
        return dict(status='optimal' if lower==bound else 'timeout',certificate='rational_supplier_fairness',
                    lb=lower,ub=bound,lb_exact=str(lower),ub_exact=str(bound),
                    route=current['route'],amounts=current['amounts'],metrics={**metrics,'phase_seconds':monotonic()-started})
    last_pair=(lower,upper)
    def publish():
        nonlocal last_pair
        if emit and monotonic()<deadline:
            row=record(); pair=(row['lb'],row['ub'])
            if pair!=last_pair:emit(row);last_pair=pair
    def offer(route):
        nonlocal lower,current
        if not route or monotonic()>=deadline:return
        value,amounts=allocate(r.instance,route)
        if value>lower:
            if check(r.instance,route,amounts,tol=0)['objective']!=value:raise ArithmeticError('invalid route')
            if monotonic()<deadline:
                lower=value;current=dict(route=list(route),amounts=amounts);publish()
    def priority(root):
        service=sum(r.service[v] for v in r.p if graph.donor_bits[v]&root.supplier_mask)
        return (-root.flow*(r.limit-service),root.supplier_mask,root.first)
    for root in sorted(caps,key=priority):
        if monotonic()>=deadline:break
        if caps[root]<=lower:continue
        def commit(pop):
            caps[root]=min(caps[root],Fraction(root.flow*min(pop,r.people),r.units));publish()
        best,prices,tracked,rounds=priced_root(graph,root,min(deadline,monotonic()+4),commit)
        metrics['price_tables']+=rounds
        if best is None or caps[root]<=lower or monotonic()>=deadline:continue
        cutoff=(lower*r.units).__floor__()
        res=branch(graph,root,prices,tracked,cutoff,deadline,offer)
        if res is None:continue
        if res['population_root']!=-1 and res['population_root']!=min(best,r.people):
            raise ArithmeticError('independent time tables disagree')
        if monotonic()>=deadline:break
        offer(res['route'])
        metrics['branches']+=1;metrics['expanded']+=res['expanded']
        if res['complete']:
            refined=Fraction(max(cutoff,int(res['largest_leaf_upper_numerator'])),r.units)
            caps[root]=min(caps[root],refined);publish()
    return record()
