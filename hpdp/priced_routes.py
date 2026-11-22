"""Exact bounds from time-indexed walks with priced and explicitly tracked sites."""

import ctypes
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from math import gcd
from pathlib import Path
import sys
from time import monotonic

from .allocation import allocate
from .check import check
from .portfolio import solve_portfolio


Int=ctypes.c_int
Ptr=ctypes.POINTER(Int)
Long=ctypes.c_int64
WitnessFn=ctypes.CFUNCTYPE(None,Int,Ptr,Int,Ptr)
ProgressFn=ctypes.CFUNCTYPE(None,Int,Long,Ptr)
IMPOSSIBLE=-1000000000


@lru_cache(maxsize=2)
def library(path=None):
    suffix="dylib" if sys.platform=="darwin" else "so"
    lib=ctypes.CDLL(str(path or Path(__file__).with_name(f"libpopulation_walks.{suffix}")))
    methods=[
        ("build",ctypes.c_void_p,[Int,Int,Int,Int,Ptr,Ptr,Ptr,Ptr,Ptr,ctypes.c_double,Ptr]),
        ("free",None,[ctypes.c_void_p]),
        ("value",Int,[ctypes.c_void_p,Int,Int,Int,Int]),
        ("trace",Int,[ctypes.c_void_p,Int,Int,Int,Int,Ptr]),
        ("elementary",Int,[ctypes.c_void_p,Ptr,Ptr,Int,Int,Int,Int,ctypes.c_double,
                           WitnessFn,ProgressFn,Ptr,ctypes.POINTER(Long)]),
    ]
    for name,out_type,arg_types in methods:
        entry="elementary_v2" if name=="elementary" else name
        fn=getattr(lib,"hpdp_walk_"+entry)
        fn.restype=out_type
        fn.argtypes=arg_types
        setattr(lib,name,fn)
    return lib


def array(values):
    return (Int*len(values))(*values)


def supported(r):
    return (0<r.limit<=1600 and 2<=len(r.nodes)<=110 and
            all(r.travel[i][j]+r.service[j]>0 for i in r.nodes for j in r.nodes if i!=j))


def cap_product(total,a,b,flow_cap,pop_cap):
    """Integer upper bound on flow * covered population under aF + bP <= total."""
    upper=min(flow_cap*pop_cap,total*total//(4*a*b))
    if 2*a*flow_cap<=total:
        upper=min(upper,flow_cap*(total-a*flow_cap)//b)
    if 2*b*pop_cap<=total:
        upper=min(upper,pop_cap*(total-b*pop_cap)//a)
    return upper


@dataclass(frozen=True)
class Start:
    first: int
    remaining: int
    donors: tuple
    flow: int


class Graph:
    def __init__(self,r,project):
        self.r=r
        self.n=len(r.travel)
        self.project=project
        travel=[row.copy() for row in r.travel]
        if project and len(r.p)>1:
            for k in range(self.n):
                for row in travel:
                    step=row[k]
                    for j,tail in enumerate(travel[k]):
                        row[j]=min(row[j],step+tail)
        self.times=array([min(r.limit+1,travel[i][j]+r.service[j])
                          for i in range(self.n) for j in range(self.n)])
        self.inner=array([int(i>0 and (not project or r.pop[i]>0)) for i in range(self.n)])
        self.finish=array([min(r.limit+1,travel[i][0]) if r.pop[i] else r.limit+1
                           for i in range(self.n)])
        self.starts=[]
        if project:
            for count in range(1,len(r.p)+1):
                for donors in combinations(r.p,count):
                    cost=sum(r.service[i] for i in donors)
                    flow=min(sum(r.supply[i] for i in donors),len(r.d)*r.capacity)
                    for first in donors:
                        remaining=r.limit-r.travel[0][first]-cost
                        if remaining>=0:
                            self.starts.append(Start(first,remaining,donors,flow))
        else:
            for first in r.p:
                remaining=r.limit-r.travel[0][first]-r.service[first]
                if remaining>=0:
                    self.starts.append(Start(first,remaining,(first,),0))


class Table:
    def __init__(self,graph,weights,prices,tracked,no_return,deadline):
        self.lib=library()
        self.graph=graph
        self.prices=tuple(prices)
        self.price_sum=sum(prices)
        self.bits=[0]*graph.n
        for k,v in enumerate(tracked):
            self.bits[v]=1<<k
        self.full=(1<<len(tracked))-1
        reduced=array([w-p for w,p in zip(weights,prices)])
        status=Int()
        self.handle=self.lib.build(graph.n,graph.r.limit,len(tracked),int(no_return),
                                   graph.times,reduced,array(self.bits),graph.inner,graph.finish,
                                   max(0,deadline-monotonic()),ctypes.byref(status))
        if status.value<0:
            raise RuntimeError(f"Walk table construction failed ({status.value})")

    def root(self,start,weights):
        mask=self.full^self.bits[start.first]
        reduced=self.lib.value(self.handle,mask,start.remaining,start.first,0)
        if reduced==IMPOSSIBLE:
            return None
        return weights[start.first]+reduced+self.price_sum-self.prices[start.first]

    def route(self,start):
        buf=(Int*(start.remaining+1))()
        length=self.lib.trace(self.handle,self.full^self.bits[start.first],
                              start.remaining,start.first,0,buf)
        if length<0:
            raise ArithmeticError("Priced walk could not reconstruct a finite start")
        return tuple(buf[:length])

    def close(self):
        if self.handle:
            self.lib.free(self.handle)
            self.handle=None


def _repair(r,route):
    route=list(route)
    while route and not r.pop[route[-1]]:
        route.pop()
    if len(route)<2 or not r.supply[route[0]]:
        return None
    time=sum(r.travel[i][j]+r.service[j] for i,j in zip([0,*route],[*route,0]))
    while time>r.limit and len(route)>2:
        candidates=[]
        for k in range(1,len(route)):
            if k==len(route)-1 and not r.pop[route[k-1]]:
                continue
            i,v=route[k-1:k+1]
            j=route[k+1] if k+1<len(route) else 0
            saving=r.travel[i][v]+r.service[v]+r.travel[v][j]-r.travel[i][j]
            candidates.append((saving,k))
        if not candidates:
            return None
        saving,k=max(candidates)
        route.pop(k)
        time-=saving
    return tuple(route) if time<=r.limit else None


def solve(r,deadline,seed,*,emit=None,tracked=4,no_return=True,
          relaxation_budget=4.0,library_search=True,bound_only=False):
    """A strict anytime solver. Every incomplete price table is discarded."""
    if not supported(r):
        return solve_portfolio(r,deadline,seed,emit=emit)
    started=monotonic()
    prior=Fraction(r.prior,r.units)
    upper=prior
    witness=seed
    value=None if seed is None else check(r.instance,seed["route"],seed["amounts"],tol=0)["objective"]
    metrics=dict(phase="walk_tables",tables=0,improvements=0,expanded=0,tracked=0,
                 forbid_immediate_return=no_return)
    def row():
        if value is not None and (upper is None or value>upper):
            raise ArithmeticError("Verified route contradicts the priced upper bound")
        return dict(status="infeasible" if upper is None else "optimal" if value==upper else "timeout",
                    lb=value,ub=upper,lb_exact=None if value is None else str(value),
                    ub_exact=None if upper is None else str(upper),certificate="rational_priced_walks",
                    route=None if witness is None else witness["route"],
                    amounts=None if witness is None else witness["amounts"],metrics=metrics.copy())
    def publish():
        if emit and monotonic()<deadline:
            emit(row())
    def consider(route):
        nonlocal value,witness
        v,amounts=allocate(r.instance,route)
        if value is None or v>value:
            if check(r.instance,route,amounts,tol=0)["objective"]!=v:
                raise ArithmeticError("Invalid reconstructed elementary route")
            if monotonic()<deadline:
                value=v
                witness=dict(route=route,amounts=amounts)
                metrics["improvements"]+=1
                publish()
        return v
    if value==upper or monotonic()>=deadline:
        return row()
    project=len(r.p)<=5 and all(r.service[v]>0 for v in r.d) and r.people*(r.limit+1)<2**29
    graph=Graph(r,project)
    if not graph.starts:
        upper=None
        return row()
    exact_single=project and len(r.p)==1 and graph.n<=64
    metrics["projected_suppliers"]=project
    if seed is not None and exact_single:
        consider(seed["route"])
    best_table=None
    best_root=None
    end=min(deadline,started+min(relaxation_budget,(1 if bound_only else 0.4)*(deadline-started)))
    flow_cap=min(sum(r.supply),len(r.d)*r.capacity)
    if seed:
        covered=sum(r.pop[v] for v in seed["route"])
        harvested=sum(r.supply[v] for v in seed["route"])
    else:
        covered,harvested=r.people,flow_cap
    # The coefficients and rounding choose the relaxation, never a pruning rule.
    ratios=[(1,1)] if project else [(covered,harvested),(2*covered,harvested),(covered,2*harvested)]
    try:
        for ratio_idx,(a,b) in enumerate(ratios):
            divisor=gcd(a,b)
            a,b=a//divisor,b//divisor
            raw=list(r.pop) if project else [a*s+b*p for s,p in zip(r.supply,r.pop)]
            unit=1 if project else max(1,(max(raw)+1023)//1024)
            weights=[(w+unit-1)//unit for w in raw]
            free=sorted((v for v in r.nodes if graph.inner[v]),key=lambda v:-weights[v])
            pivot=weights[free[len(free)//4]]
            prices=[max(0,w-pivot) if graph.inner[i] else 0 for i,w in enumerate(weights)]
            first_prices=prices[:]
            bounds=[r.prior]*len(graph.starts)
            rank_counts=[0]*graph.n
            sub_end=monotonic()+max(0,end-monotonic())/(len(ratios)-ratio_idx)
            for memory in dict.fromkeys((0,tracked)):
                if monotonic()>=sub_end:
                    break
                track=sorted(free,key=lambda v:(-rank_counts[v],-weights[v],v))[:memory]
                while track and graph.n*(r.limit+1)*(1<<len(track))*12>100_000_000:
                    track.pop()
                for v in track:
                    prices[v]=0
                seen=set()
                for it in range(9 if memory==0 else 4):
                    key=tuple(prices)
                    if key in seen or monotonic()>=sub_end: break
                    seen.add(key)
                    tab=Table(graph,weights,prices,track,no_return,sub_end)
                    if not tab.handle: break
                    metrics["tables"]+=1
                    metrics["tracked"]=max(metrics["tracked"],len(track))
                    active=[]
                    for i,context in enumerate(graph.starts):
                        additive=tab.root(context,weights)
                        if additive is None:
                            bounds[i]=-1
                            continue
                        unrounded=additive*unit
                        cap=(context.flow*min(r.people,unrounded) if project else
                             cap_product(unrounded,a,b,flow_cap,r.people))
                        bounds[i]=min(bounds[i],cap)
                        active.append((bounds[i],i,additive))
                    if not active:
                        tab.close()
                        upper=None
                        return row()
                    best,i,additive=max(active)
                    upper=min(upper,Fraction(best,r.units))
                    publish()
                    context=graph.starts[i]
                    walk=tab.route(context)
                    for unique in (tuple(dict.fromkeys(walk)),tuple(reversed(dict.fromkeys(reversed(walk))))):
                        ordered=[context.first,*(v for v in context.donors if v!=context.first),
                                 *(v for v in unique if v not in context.donors)]
                        repaired=_repair(r,ordered)
                        if repaired:
                            consider(repaired)
                    if exact_single and (best_root is None or additive<best_root):
                        if best_table: best_table.close()
                        best_table,best_root=tab,additive
                    else:
                        tab.close()
                    if value==upper or monotonic()>=deadline:
                        return row()
                    counts=[0]*graph.n
                    for v in (context.first,*walk): counts[v]+=1
                    step=max(1,pivot//(2+it//2))
                    for v in free:
                        rank_counts[v]+=max(0,counts[v]-1)
                        prices[v]=(0 if v in track else
                                   max(0,min(weights[v],prices[v]+step*(counts[v]-1))))
                if memory==0:
                    prices=[(old+new)//2 for old,new in zip(first_prices,prices)]
            if monotonic()>=end: break

        metrics["relaxation_seconds"]=monotonic()-started
        if bound_only:
            return row()
        if exact_single and best_table and library_search and monotonic()<deadline:
            metrics["phase"]="elementary_walk_search"
            first=graph.starts[0]
            factor=Fraction(first.flow,r.units)
            threshold=-1 if value is None else int(value//factor)
            root_ub=int(upper//factor)
            error=[]
            @WitnessFn
            def found(length,nodes,people,reply):
                try:
                    if monotonic()>=deadline:
                        return
                    v=consider(tuple(nodes[:length]))
                    if v!=factor*people:
                        raise ArithmeticError("One-supplier proportional allocation disagrees with native population")
                    if monotonic()<deadline:
                        reply[0]=people
                except BaseException as exc:
                    error.append(exc)
            @ProgressFn
            def update(pop,expanded,proceed):
                nonlocal upper
                try:
                    if monotonic()>=deadline:
                        return
                    if pop>=0:
                        upper=min(upper,max(value or 0,factor*pop))
                        metrics["expanded"]=expanded
                        publish()
                    if monotonic()<deadline:
                        proceed[0]=1
                except BaseException as exc:
                    error.append(exc)
            result=Int()
            count=Long()
            status=best_table.lib.elementary(best_table.handle,array(r.pop),array(best_table.prices),
                        first.first,first.remaining,threshold,root_ub,max(0,deadline-monotonic()),
                        found,update,ctypes.byref(result),ctypes.byref(count))
            if error: raise error[0]
            if status<0: raise RuntimeError(f"Elementary walker failed ({status})")
            metrics["expanded"]=count.value
            if status==1:
                if result.value!=(-1 if value is None else int(value//factor)):
                    raise ArithmeticError("Native result and verified incumbent differ")
                upper=value
            elif result.value>=0:
                upper=min(upper,max(value or 0,factor*result.value))
            return row()
    finally:
        if best_table: best_table.close()
    if value==upper or monotonic()>=deadline: return row()
    metrics["phase"]="portfolio_with_walk_bound"
    def combine(other):
        nonlocal upper,value,witness
        if other["lb_exact"] is not None:
            incoming=Fraction(other["lb_exact"])
            if value is None or incoming>value:
                value=incoming
                witness=other
        upper=None if other["ub_exact"] is None else min(upper,Fraction(other["ub_exact"]))
        metrics["portfolio"]=other["metrics"]
        return row()
    result=solve_portfolio(r,deadline,witness,emit=(lambda other:emit(combine(other))) if emit else None)
    return combine(result)
