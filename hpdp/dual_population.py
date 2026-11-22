"""Choose verifiable integer walk prices by a restricted master LP on one-supplier cases."""
from collections import Counter
import ctypes
from fractions import Fraction
from time import monotonic
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, linprog

from .allocation import allocate
from .check import check
from .priced_routes import Graph, Int, Long, ProgressFn, Table, WitnessFn, _repair, array
from .priced_routes import solve as solve_walks, supported


def _propose(cuts, population, tracked, scale):
    """A numerical master suggests prices. Only the integer walk DP can certify a bound."""
    n=len(population)
    rows=[]
    rhs=[]
    for visits in cuts:
        rows.append([1-count for count in visits]+[-1])
        rhs.append(-sum(p*c for p,c in zip(population,visits)))
    bounds=[(0,0 if i in tracked else p) for i,p in enumerate(population)]+[(0,None)]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=OptimizeWarning,message="Unrecognized options.*")
        lp=linprog([0]*n+[1],A_ub=np.asarray(rows,dtype=float),b_ub=rhs,bounds=bounds,
                   method="highs",options={"threads":1})
    if not lp.success or not np.isfinite(lp.x[:-1]).all():
        return None
    return [max(0,min(p*scale,round(scale*x))) for p,x in zip(population,lp.x)]


def solve(r,deadline,seed,*,emit=None,method="master",tracked_count=4,rounds=36,price_seconds=5):
    if len(r.p)!=1 or len(r.travel)>64 or not supported(r) or any(r.service[v]<=0 for v in r.d):
        return solve_walks(r,deadline,seed,emit=emit)
    # Keep total corrected populations, every reward-accumulating walk and
    # the original crossing products inside the verified native integer range.
    scale=min(32,(2**29-1)//((r.limit+2)*r.people))
    if scale<1:
        return solve_walks(r,deadline,seed,emit=emit)
    started=monotonic()
    graph=Graph(r,True)
    factor=Fraction(r.supply[r.p[0]],r.units)
    witness=seed
    lb=None if seed is None else check(r.instance,seed["route"],seed["amounts"],tol=0)["objective"]
    ub=Fraction(r.prior,r.units)
    stats=dict(phase="dual_prices",price_method=method,scale=scale,tables=0,master_calls=0,
               master_seconds=0.0,improvements=0,expanded=0)
    def record():
        if lb is not None and (ub is None or lb>ub):
            raise ArithmeticError("An integral population certificate excludes a verified route")
        return dict(status="infeasible" if ub is None else "optimal" if ub==lb else "timeout",
                    lb=lb,ub=ub,lb_exact=None if lb is None else str(lb),
                    ub_exact=None if ub is None else str(ub),certificate="rational_elementary_population",
                    route=None if witness is None else witness["route"],
                    amounts=None if witness is None else witness["amounts"],metrics=stats.copy())
    def publish():
        if emit and monotonic()<deadline:
            emit(record())
    def accept(route):
        nonlocal witness,lb
        value,amounts=allocate(r.instance,route)
        if lb is None or value>lb:
            if check(r.instance,route,amounts,tol=0)["objective"]!=value:
                raise ArithmeticError("Invalid walk-derived allocation")
            if monotonic()<deadline:
                lb=value
                witness=dict(route=route,amounts=amounts)
                stats["improvements"]+=1
                publish()
        return value
    if lb==ub or started>=deadline:
        return record()
    if not graph.starts:
        ub=None
        return record()
    first=graph.starts[0]
    if seed:
        accept(seed["route"])
    weights=[p*scale for p in r.pop]
    ordered=sorted(r.d,key=lambda j:(-r.pop[j],j))
    track=ordered[:tracked_count]
    while graph.n*(r.limit+1)*(1<<len(track))*12>100_000_000:
        track.pop()
    pivot=r.pop[ordered[len(ordered)//4]]
    prices=[max(0,w-pivot*scale) if j not in track else 0 for j,w in enumerate(weights)]
    stats["tracked_vertices"]=track
    stop=min(deadline,started+min(price_seconds,0.36*(deadline-started)))
    cuts=[]
    tried=set()
    best=None
    best_upper=None
    try:
        for i in range(rounds):
            if monotonic()>=stop or tuple(prices) in tried:
                break
            tried.add(tuple(prices))
            table=Table(graph,weights,prices,track,True,stop)
            if not table.handle:
                break
            stats["tables"]+=1
            score=table.root(first,weights)
            if score is None:
                table.close()
                ub=None
                return record()
            walk=table.route(first)
            if best_upper is None or score<best_upper:
                if best: best.close()
                best_upper,best=score,table
                stats["best_prices_scaled"]=prices[:]
                stats["scaled_root_bound"]=score
                ub=min(ub,factor*(score//scale))
                publish()
            else:
                table.close()
            for unique in (tuple(dict.fromkeys(walk)),tuple(reversed(dict.fromkeys(reversed(walk))))):
                route=_repair(r,(first.first,*unique))
                if route:
                    accept(route)
            if lb==ub or monotonic()>=deadline:
                return record()
            counts=Counter(walk)
            visits=tuple(counts[j] for j in range(graph.n))
            if visits not in cuts: cuts.append(visits)
            if method=="master":
                tic=monotonic()
                suggested=_propose(cuts,r.pop,track,scale)
                stats["master_seconds"]+=monotonic()-tic
                stats["master_calls"]+=1
                if suggested is None: break
                prices=suggested
            elif method=="subgradient":
                step=max(1,scale*pivot//(2+i//3))
                prices=[min(weights[j],max(0,prices[j]+step*(counts[j]-1)))
                        if j in r.d and j not in track else 0 for j in range(graph.n)]
            else:
                raise ValueError("Unknown price-selection method")
        stats["relaxation_seconds"]=monotonic()-started
        if not best or monotonic()>=deadline:
            if not best and monotonic()<deadline:
                return solve_walks(r,deadline,witness,emit=emit)
            return record()
        stats["phase"]="elementary_search"
        errors=[]
        @WitnessFn
        def offer(length,nodes,scaled_people,reply):
            try:
                if monotonic()>=deadline:
                    return
                value=accept(tuple(nodes[:length]))
                if value!=factor*Fraction(scaled_people,scale):
                    raise ArithmeticError("Native cumulative population and witness disagree")
                if monotonic()<deadline:
                    reply[0]=scaled_people
            except BaseException as exc:
                errors.append(exc)
        @ProgressFn
        def progress(bound,expanded,proceed):
            nonlocal ub
            try:
                if monotonic()>=deadline:
                    return
                if bound>=0:
                    ub=min(ub,max(lb or 0,factor*(bound//scale)))
                    stats["expanded"]=expanded
                    publish()
                if monotonic()<deadline:
                    proceed[0]=1
            except BaseException as exc:
                errors.append(exc)
        upper,expanded=Int(),Long()
        cutoff=-1 if lb is None else scale*int(lb//factor)
        result=best.lib.elementary(best.handle,array(weights),array(best.prices),first.first,
                   first.remaining,cutoff,scale*int(ub//factor),max(0,deadline-monotonic()),
                   offer,progress,ctypes.byref(upper),ctypes.byref(expanded))
        if errors: raise errors[0]
        if result<0: raise RuntimeError("Integer elementary search failed")
        stats["expanded"]=expanded.value
        if result==1:
            if upper.value!=(-1 if lb is None else scale*int(lb//factor)):
                raise ArithmeticError("Population search and verified witness disagree")
            ub=lb
        elif upper.value>=0:
            ub=min(ub,max(lb or 0,factor*(upper.value//scale)))
        return record()
    finally:
        if best: best.close()
