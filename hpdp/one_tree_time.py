"""Certified travel lower bound for visiting a specified set once, with legal depot roles."""
from time import monotonic


def _indices(resources, sites):
    sites=tuple(sorted(sites))
    if len(sites)!=len(set(sites)) or any(type(v) is not int or v not in resources.nodes for v in sites):
        raise ValueError("The subset must contain unique nondepot sites")
    donors=[i for i,v in enumerate(sites) if resources.supply[v]]
    agencies=[i for i,v in enumerate(sites) if resources.pop[v]]
    if not donors or not agencies:
        raise ValueError("A complete visit set requires a supplier and an agency")
    return sites,donors,agencies


def _prim(base, prices):
    n=len(base)
    used=[False]*n
    best=[None]*n
    parents=[-1]*n
    degrees=[0]*n
    best[0]=0
    total=0
    for _ in range(n):
        selected=None
        for i,cost in enumerate(best):
            if not used[i] and cost is not None and (selected is None or cost<best[selected]):
                selected=i
        used[selected]=True
        total+=best[selected]
        parent=parents[selected]
        if parent>=0:
            degrees[parent]+=1
            degrees[selected]+=1
        for j,raw in enumerate(base[selected]):
            if used[j]:
                continue
            cost=raw+prices[selected]+prices[j]
            if best[j] is None or cost<best[j]:
                best[j]=cost
                parents[j]=selected
    return total,degrees


def lower_bound(resources, sites, deadline, *, iterations=48,denominator=32):
    """Return the best fully constructed integer-priced tree, or None before the first commit."""
    if type(iterations) is not int or iterations<1 or type(denominator) is not int or denominator<1:
        raise ValueError("Positive integer iteration cap and price denominator required")
    if monotonic()>=deadline:
        return None
    sites,donors,agencies=_indices(resources,sites)
    travel=resources.travel
    base=[[denominator*min(travel[i][j],travel[j][i]) for j in sites] for i in sites]
    depart=[denominator*travel[0][i] for i in sites]
    arrive=[denominator*travel[i][0] for i in sites]
    service=sum(resources.service[i] for i in sites)
    budget=resources.limit-service
    prices=[0]*len(sites)
    best=None
    stale=0
    damping=1
    for it in range(1,iterations+1):
        if monotonic()>=deadline:
            break
        tree,degrees=_prim(base,prices)
        p=min(donors,key=lambda i:depart[i]+prices[i])
        d=min(agencies,key=lambda i:arrive[i]+prices[i])
        degrees[p]+=1
        degrees[d]+=1
        numerator=tree+depart[p]+prices[p]+arrive[d]+prices[d]-2*sum(prices)
        if monotonic()>=deadline:
            break
        if best is None or numerator>best["travel_numerator"]:
            best=dict(sites=sites,prices=tuple(prices),denominator=denominator,
                      travel_numerator=numerator,service_scaled=service,time_scale=resources.time_scale,
                      lower_scaled=service-(-numerator//denominator),best_iteration=it)
            stale=0
        else:
            stale+=1
        best["iterations"]=it
        if best["lower_scaled"]>resources.limit:
            break
        direction=[degree-2 for degree in degrees]
        norm=sum(g*g for g in direction)
        if not norm:
            break
        if stale>=8:
            damping*=2
            stale=0
        step=max(1,2*(denominator*(budget+1)-numerator)//(damping*norm))
        prices=[v+step*g for v,g in zip(prices,direction,strict=True)]
    return best


def replay(resources,sites,certificate):
    """Rebuild a minimum tree by Kruskal and check the proposed prices against the original arcs."""
    sites,donors,agencies=_indices(resources,sites)
    D=certificate["denominator"]
    prices=certificate["prices"]
    if (type(D) is not int or D<1 or any(type(x) is not int for x in prices)
            or len(prices)!=len(sites) or tuple(certificate["sites"])!=sites):
        raise ValueError("Tree certificate does not have the requested domain")
    parent=list(range(len(sites)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    edges=sorted((D*min(resources.travel[a][b],resources.travel[b][a])+prices[i]+prices[j],i,j)
                 for i,a in enumerate(sites) for j,b in enumerate(sites) if i<j)
    total=0
    count=0
    for cost,i,j in edges:
        a,b=root(i),root(j)
        if a!=b:
            parent[a]=b
            total+=cost
            count+=1
        if count==len(sites)-1:
            break
    total+=min(D*resources.travel[0][sites[i]]+prices[i] for i in donors)
    total+=min(D*resources.travel[sites[j]][0]+prices[j] for j in agencies)
    total-=2*sum(prices)
    service=sum(resources.service[i] for i in sites)
    lower=service-(-total//D)
    if (certificate["time_scale"]!=resources.time_scale or certificate["service_scaled"]!=service
            or certificate["travel_numerator"]!=total or certificate["lower_scaled"]!=lower):
        raise ValueError("Claimed travel or service bound differs from integer reconstruction")
    return lower
