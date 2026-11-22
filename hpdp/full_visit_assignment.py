"""Reproducible integer assignment branches for full-visit time feasibility."""

from fractions import Fraction
from time import monotonic

from .certificate_support import rank_certificate

from .allocation import allocate
from .check import check
from .prefix import box_bound


def arc_costs(r):
    """Original travel only; full-visit service is one constant."""
    return tuple(tuple(
        None if i==j or (i==0 and j not in r.p) or (j==0 and i not in r.d) else value
        for j,value in enumerate(row)) for i,row in enumerate(r.travel))


def integer_assignment(costs, deadline):
    """Return an exact Hungarian primal-dual pair, a Hall set, or None on expiry."""
    n = len(costs)
    u,v = [0]*(n+1),[0]*(n+1)
    matching,way = [0]*(n+1),[0]*(n+1)
    for source in range(1,n+1):
        matching[0]=source
        used=bytearray(n+1)
        gap=[None]*(n+1)
        j0=0
        while True:
            if monotonic()>=deadline:
                return None
            used[j0]=1
            i0=matching[j0]
            row=costs[i0-1]
            delta,j1=None,0
            for j,price in enumerate(row,1):
                if used[j]:
                    continue
                if price is not None:
                    reduced=price-u[i0]-v[j]
                    if gap[j] is None or reduced<gap[j]:
                        gap[j],way[j]=reduced,j0
                if gap[j] is not None and (delta is None or gap[j]<delta):
                    delta,j1=gap[j],j
            if delta is None:
                hall=sorted(matching[j]-1 for j in range(n+1) if used[j])
                neighbors=sorted({j for i in hall for j,c in enumerate(costs[i]) if c is not None})
                return dict(hall=hall,neighbors=neighbors)
            for j in range(n+1):
                if used[j]:
                    u[matching[j]]+=delta
                    if j:
                        v[j]-=delta
                elif gap[j] is not None:
                    gap[j]-=delta
            j0=j1
            if not matching[j0]:
                break
        while j0:
            previous=way[j0]
            matching[j0]=matching[previous]
            j0=previous
    next_node=[-1]*n
    for j in range(1,n+1):
        next_node[matching[j]-1]=j-1
    return dict(next=next_node,u=u[1:],v=v[1:],
                cost=sum(costs[i][j] for i,j in enumerate(next_node)))


def verify_assignment(costs, proof):
    """Check using only integer equalities and inequalities."""
    n=len(costs)
    if "hall" in proof:
        hall=proof["hall"]
        if (not hall or any(type(i) is not int or not 0<=i<n for i in hall)
                or len(set(hall))!=len(hall)):
            raise ValueError("Invalid Hall rows")
        expected={j for i in hall for j,c in enumerate(costs[i]) if c is not None}
        if sorted(expected)!=list(proof["neighbors"]) or len(expected)>=len(hall):
            raise ValueError("Invalid Hall neighborhood")
        return False
    nxt,u,v=proof["next"],proof["u"],proof["v"]
    if (len(nxt)!=n or len(u)!=n or len(v)!=n
            or any(type(x) is not int for x in (*nxt,*u,*v,proof["cost"]))
            or set(nxt)!=set(range(n))):
        raise ValueError("Invalid integer matching or potentials")
    chosen=[costs[i][j] for i,j in enumerate(nxt)]
    if (None in chosen or sum(chosen)!=proof["cost"] or sum(u)+sum(v)!=proof["cost"]
            or any(u[i]+v[j]>c for i,row in enumerate(costs)
                   for j,c in enumerate(row) if c is not None)):
        raise ValueError("Invalid assignment primal-dual certificate")
    return True


def _restricted(base, fixed, forbidden):
    by_row=dict(fixed)
    by_column={j:i for i,j in fixed}
    return tuple(tuple(
        c if (i,j) not in forbidden and by_row.get(i,j)==j and by_column.get(j,i)==i else None
        for j,c in enumerate(row)) for i,row in enumerate(base))


def _cycles(nxt):
    unseen=set(range(len(nxt)))
    result=[]
    while unseen:
        first=min(unseen)
        cycle=[]
        site=first
        while site in unseen:
            unseen.remove(site)
            cycle.append(site)
            site=nxt[site]
        result.append(tuple(cycle))
    return result


def _edges(cycle):
    return tuple(zip(cycle,cycle[1:]+cycle[:1]))


def _partition(path,fixed,forbidden,cut):
    """Disjoint branches by the first missing edge of one proper cycle."""
    return [(path+(i,),fixed+cut[:i],forbidden|{edge}) for i,edge in enumerate(cut)]


def check_full_visit(r, deadline, *, branch=True, trace=None):
    """Find one exact full-visit time witness or exclude every full-visit route."""
    base=arc_costs(r)
    service=sum(r.service)
    stack=[((),(),frozenset())]
    metrics=dict(nodes=0,branches=0,hall_cuts=0,time_cuts=0,forced_cuts=0,
                 root_lb_scaled=None,time_scale=r.time_scale,service_scaled=service)

    def commit(row):
        if trace is not None:
            trace(row)

    def finish(status,route=None,elapsed=None):
        commit(dict(event="end",status=status,pending=len(stack)))
        return dict(full_visit=status,full_route=route,full_time_scaled=elapsed,
                    metrics={**metrics,"pending":len(stack)})

    while stack:
        path,fixed,forbidden=stack[-1]
        costs=_restricted(base,fixed,forbidden)
        proof=integer_assignment(costs,deadline)
        if proof is None or monotonic()>=deadline:
            return finish("unknown")
        stack.pop()
        metrics["nodes"]+=1
        row=dict(event="node",path=path,assignment=proof)
        if not verify_assignment(costs,proof):
            metrics["hall_cuts"]+=1
            commit({**row,"kind":"hall"})
            continue
        elapsed=proof["cost"]+service
        if not path:
            metrics["root_lb_scaled"]=elapsed
        if elapsed>r.limit:
            metrics["time_cuts"]+=1
            commit({**row,"kind":"time"})
            continue
        cycles=_cycles(proof["next"])
        if len(cycles)==1:
            route=cycles[0][1:]
            measured=check(r.instance,route,[0]*len(route),tol=0)["time"]
            if measured!=Fraction(elapsed,r.time_scale):
                raise ArithmeticError("Time witness differs from the original arcs and service")
            commit({**row,"kind":"tour"})
            return finish("feasible",route,elapsed)
        cycle=min(cycles,key=lambda c:(sum(e not in fixed for e in _edges(c)),len(c),c))
        cut=tuple(sorted((e for e in _edges(cycle) if e not in fixed),
                         key=lambda e:(-base[e[0]][e[1]],e)))
        commit({**row,"kind":"branch" if cut else "forced","cycle":cycle,"cut":cut})
        if cut:
            metrics["branches"]+=1
            stack.extend(reversed(_partition(path,fixed,forbidden,cut)))
            if not branch:
                return finish("unknown")
        else:
            metrics["forced_cuts"]+=1
    return finish("infeasible")


def replay(r, records):
    """Verify a streamed proof without running a second assignment optimizer."""
    base=arc_costs(r)
    service=sum(r.service)
    stack=[((),(),frozenset())]
    witness=None
    footer=None
    count=0
    for row in records:
        if footer is not None:
            raise ValueError("Rows after proof footer")
        if row["event"]=="end":
            wanted=row["status"]
            if (row["pending"]!=len(stack) or wanted not in ("unknown","infeasible","feasible")
                    or (wanted=="infeasible" and (stack or witness is not None))
                    or (wanted=="feasible" and witness is None)
                    or (wanted=="unknown" and witness is not None)):
                raise ValueError("Unsupported proof footer")
            footer=wanted
            continue
        if row["event"]!="node" or not stack or witness is not None:
            raise ValueError("Unexpected proof node")
        path,fixed,forbidden=stack.pop()
        if tuple(row["path"])!=path:
            raise ValueError("Missing, duplicated or misordered branch")
        count+=1
        proof=row["assignment"]
        assigned=verify_assignment(_restricted(base,fixed,forbidden),proof)
        if not assigned:
            possible="hall"
        elif proof["cost"]+service>r.limit:
            possible="time"
        else:
            cycles=_cycles(proof["next"])
            if len(cycles)==1:
                possible="tour"
                witness=cycles[0][1:]
                measured=check(r.instance,witness,[0]*len(witness),tol=0)["time"]
                if measured!=Fraction(proof["cost"]+service,r.time_scale):
                    raise ValueError("Decoded time witness disagrees with the certificate")
            else:
                cycle=tuple(row["cycle"])
                if (len(set(cycle))!=len(cycle) or not 1<len(cycle)<len(base)
                        or not set(cycle)<=set(range(len(base)))
                        or any(proof["next"][i]!=j for i,j in _edges(cycle))):
                    raise ValueError("Claimed subtour is absent from the matching")
                cut=tuple(tuple(e) for e in row["cut"])
                expected=set(_edges(cycle))-set(fixed)
                if set(cut)!=expected or len(set(cut))!=len(cut):
                    raise ValueError("Subtour partition does not cover all first missing edges")
                possible="branch" if cut else "forced"
                stack.extend(reversed(_partition(path,fixed,forbidden,cut)))
        if row["kind"]!=possible:
            raise ValueError("Invalid closure at this assignment node")
    # A truncated trace can show checked nodes but has no successful footer.
    return dict(full_visit=footer or "unknown",complete=footer is not None,
                nodes=count,pending=len(stack),full_route=witness)


def omitted_bounds(r):
    """Upper numerators for routes omitting each specified site."""
    stock=sum(r.supply)
    out={}
    agencies=set(r.d)
    for omit in r.nodes:
        people=sorted((r.pop[v] for v in agencies if v!=omit),reverse=True)
        flow=min(stock-r.supply[omit],len(people)*r.capacity)
        out[omit]=box_bound(people,flow,r.capacity)
    return out


def solve_full_visit(r, deadline, seed, *, emit=None, trace=None, branch=True, upper=None):
    """Use only a closed full-visit exclusion to lower the existing global bound."""
    upper=Fraction(r.prior,r.units) if upper is None else Fraction(upper)
    value=None if seed is None else check(r.instance,seed["route"],seed["amounts"],tol=0)["objective"]
    witness=seed
    full_witness=None
    state=check_full_visit(r,deadline,branch=branch,trace=trace)
    new_cert="inherited_strict_upper"
    metrics=dict(state["metrics"],full_visit=state["full_visit"],allocation="not_applicable")
    if state["full_visit"]=="infeasible" and monotonic()<deadline:
        bounds=omitted_bounds(r)
        if monotonic()<deadline:
            upper=min(upper,Fraction(max(bounds.values()),r.units))
            metrics["omitted_bounds_scaled"]=bounds
            new_cert="rational_one_or_more_omitted"
    elif state["full_visit"]=="feasible":
        metrics["allocation"]="timeout"
        if monotonic()<deadline:
            route=state["full_route"]
            score,amounts=allocate(r.instance,route)
            checked=check(r.instance,route,amounts,tol=0)
            rank=rank_certificate(r.instance,route,amounts)
            if (score!=checked["objective"] or score!=rank["upper_bound"]
                    or checked["time"]!=Fraction(state["full_time_scaled"],r.time_scale)):
                raise ArithmeticError("Full-visit witness did not pass independent exact checks")
            if monotonic()<deadline:
                full_witness=dict(route=route,amounts=amounts,objective_exact=str(score))
                metrics.update(allocation="verified",full_score_exact=str(score),
                               full_time_exact=str(checked["time"]))
                if value is None or score>value:
                    value,witness=score,dict(route=route,amounts=amounts)
    if value is not None and value>upper:
        raise ArithmeticError("New upper bound contradicts a verified witness")
    result=dict(status="optimal" if value==upper else "timeout",lb=value,ub=upper,
                lb_exact=None if value is None else str(value),ub_exact=str(upper),
                route=None if witness is None else witness["route"],
                amounts=None if witness is None else witness["amounts"],certificate=new_cert,
                full_visit=state["full_visit"],full_route=state["full_route"],
                full_time_scaled=state["full_time_scaled"],full_witness=full_witness,metrics=metrics)
    if emit is not None and monotonic()<deadline:
        emit(result)
    return result
