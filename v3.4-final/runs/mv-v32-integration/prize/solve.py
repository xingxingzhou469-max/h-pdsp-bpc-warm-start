"""Balanced prize-collecting tours, rescored and repaired for the original humanitarian objective."""
from time import monotonic,monotonic_ns
START=monotonic()
import argparse
from fractions import Fraction as F
from functools import lru_cache
import json
import math
import platform
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
if sys.platform=='darwin' and platform.machine()=='arm64' and sys.version_info[:2]==(3,13):
    sys.path.insert(0,str(HERE/'vendor'))
from levels.solve import Publisher
from reference.common import number

def run(raw,k,out,seconds,warm,seed):
    end=START+seconds
    out.mkdir(parents=True,exist_ok=True)
    publish=Publisher(raw,k,out,int(end*1e9))
    best_routes=publish.seed(warm) if warm else ()
    if not warm:
        publish.consider((),'prize-idle')
    supply=[float(min(number(s),number(raw['capacity']))) for s in raw['supply']]
    people=[float(number(x)) for x in raw['population']]
    total_p=sum(people)
    stock=sum(supply)
    n=len(supply)
    if not k or not total_p or not stock or monotonic()>=end:
        return
    import numpy as np
    import pyvrp as pv
    duration=float(number(raw['limit']))
    tr=[[float(number(c)) for c in row] for row in raw['travel']]
    srv=[float(number(t)) for t in raw['service']]
    horizon=100000
    scale=horizon/duration if duration>0 else 100000
    ticks=np.asarray([[min(horizon+1,round(t*scale)) for t in row] for row in tr],dtype=np.int64)
    svc=[min(horizon+1,round(t*scale)) for t in srv]
    pos=raw.get('positions',[(0,0)]*n)
    loc=[pv.Location(float(x),float(y)) for x,y in pos]
    depot=[pv.Depot(0)]
    unit=100000
    amount=[round(x/stock*unit) for x in supply]
    head=[round(x/total_p*unit) for x in people]
    travel_cost=np.rint(ticks*(0.05/k)).astype(np.int64)
    best_float=float(publish.best)
    seen=set()

    @lru_cache(maxsize=8000)
    def variants(seq):
        if not seq:
            return [((),())]
        options=[]
        for orient in (seq,seq[::-1]):
            ring=sum(tr[i][j] for i,j in zip(orient,orient[1:]+orient[:1]))+sum(srv[i] for i in orient)
            for offset,first in enumerate(orient):
                last=orient[offset-1]
                if not supply[first] or not people[last]:
                    continue
                used=ring-tr[last][first]+tr[0][first]+tr[last][0]
                if used>duration+1e-8:
                    continue
                path=orient[offset:]+orient[:offset]
                # Prefix supply is a one-directional flow. With nonbinding
                # capacity, pooling adjacent inverted rates gives the levels.
                groups=[]
                q=h=0.
                def append(q,h):
                    while groups and groups[-1][0]*h>=q*groups[-1][1]:
                        a,b=groups.pop();q+=a;h+=b
                    groups.append((q,h))
                for i in path:
                    if supply[i]:
                        if h:
                            append(q,h);q=h=0.
                        q+=supply[i]
                    else:
                        h+=people[i]
                if h: append(q,h)
                levels=tuple((q/h,h) for q,h in groups)
                options.append((path,levels))
        return options

    def merit(levels):
        mass=value=0.
        for rate,p in sorted(levels,reverse=True):
            value+=rate*p*(p+2*mass);mass+=p
        return value/total_p

    def observe(solution):
        nonlocal best_routes,best_float
        if monotonic()>=end or not solution.is_feasible():
            return
        routes=tuple(tuple(act.idx+1 for act in route if act.is_client()) for route in solution.routes())
        if not routes or routes in seen:
            return
        if len(seen)>5000: seen.clear()
        seen.add(routes)
        q=sum(supply[i] for route in routes for i in route)
        h=sum(people[i] for route in routes for i in route)
        if q*h/total_p<=best_float-1e-7:
            return
        options=[variants(route) for route in routes]
        if any(not group for group in options):
            return
        chosen=[max(group,key=lambda item:merit(item[1])) for group in options]
        for slot,group in enumerate(options):
            other=tuple(x for j,(_,levels) in enumerate(chosen) if slot!=j for x in levels)
            chosen[slot]=max(group,key=lambda item:merit(other+item[1]))
        estimate=merit(tuple(x for _,levels in chosen for x in levels))
        if estimate+1e-7<best_float: return
        cand=tuple(path for path,_ in chosen)
        if publish.consider(cand,'balanced-prize-waterfill'):
            best_routes=cand
            best_float=float(publish.best)

    class Capture(pv.IteratedLocalSearchCallbacks):
        count=0
        def on_best(self,best):
            observe(best)
        def on_iteration(self,current,candidate,best,cost_evaluator):
            self.count+=1
            if self.count%47==0:
                observe(current)

    iteration=0
    while monotonic()<end-.08:
        if best_routes:
            qbase=sum(supply[i] for route in best_routes for i in route)/stock
            pbase=sum(people[i] for route in best_routes for i in route)/total_p
        else:
            qbase=pbase=min(.95,.30+.14*k)
        qbase=max(.1,qbase); pbase=max(.1,pbase)
        balance=(1.13,1.28,1.07,2.0,1.18)[iteration%5]
        tilt=(1,.92,1.09,1.02,.98)[iteration%5]
        clients=[pv.Client(location=i,delivery=[amount[i],head[i],amount[i]],
                           pickup=[0,0,round(head[i]*qbase/pbase)],service_duration=svc[i],
                           prize=max(1,round(amount[i]/qbase+tilt*head[i]/pbase)),required=False)
                 for i in range(1,n)]
        caps=[max(max(amount),round(unit*qbase*balance/k)),max(max(head),round(unit*pbase*balance/k))]
        caps.append(max(max(amount),max(round(h*qbase/pbase) for h in head),round(unit*qbase*(1.06,1.16,1.08,1.50,1.11)[iteration%5]/k)))
        vt=[pv.VehicleType(num_available=k,capacity=caps,shift_duration=horizon if duration else 0,
                           unit_distance_cost=1)]
        matrix=ticks.copy()
        # Closing the cycle after an agency, opening it at a supplier, matches
        # the witness verifier. Only the heuristic subproblem uses these gates.
        for i in range(1,n):
            if supply[i]:
                matrix[i,0]=horizon+1
            else:
                matrix[0,i]=horizon+1
        data=pv.ProblemData(loc,clients,depot,vt,[travel_cost],[matrix])
        initial=None
        if best_routes and iteration%4!=3:
            initial=pv.Solution(data,[[i-1 for i in route] for route in best_routes])
        params=pv.SolveParams(ils=pv.IteratedLocalSearchParams(callbacks=Capture()))
        stop=min(end-.03,monotonic()+3.0+min(iteration,5)*1.6)
        result=pv.solve(data,lambda _:monotonic()>=stop,seed+iteration,False,False,params,initial_solution=initial)
        observe(result.best)
        iteration+=1

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('instance',type=Path)
    p.add_argument('--vehicles',type=int,required=True)
    p.add_argument('--seconds',type=float,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--warm',type=Path)
    p.add_argument('--seed',type=int,default=7)
    a=p.parse_args()
    run(json.loads(a.instance.read_bytes(),parse_float=F),a.vehicles,a.out,a.seconds,a.warm,a.seed)
