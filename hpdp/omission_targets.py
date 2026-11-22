"""Exact time exclusions and fixed-allocation witnesses for high-coverage visit sets."""
from fractions import Fraction
from heapq import heappop, heappush
from time import monotonic

from .check import check
from .fixed_delivery_order import find_order
from .full_visit_assignment import arc_costs, integer_assignment, verify_assignment
from .one_tree_time import lower_bound, replay as replay_tree


def _cap(r,sites):
    stock=sum(r.supply[v] for v in sites)
    agencies=sum(1 for v in sites if r.pop[v])
    return min(stock,agencies*r.capacity)*sum(r.pop[v] for v in sites)


def _matrix(base,sites):
    ids=(0,*sites)
    return tuple(tuple(base[i][j] for j in ids) for i in ids)


def solve(r,deadline,seed,*,emit=None,tree=True,fixed=True,max_sets=64,target_seconds=1.4):
    """A bound-only stop leaves every unproved visit set in the upper envelope."""
    if type(max_sets) is not int or max_sets<1 or not 0<target_seconds<=60:
        raise ValueError("Invalid visit-set or construction limits")
    nodes=tuple(r.nodes)
    value=None if seed is None else check(r.instance,seed["route"],seed["amounts"],tol=0)["objective"]
    witness=seed
    base=arc_costs(r)
    full=(1<<len(nodes))-1
    frontier=[(-r.prior,full,0)]
    unresolved=-1
    proofs=[]
    stats=dict(examined=0,assignment_excluded=0,tree_excluded=0,service_excluded=0,
               target_attempts=0,target_found=0,unresolved_sets=0,tree_seconds=0.0,
               construction_seconds=0.0,search="role-constrained time and fixed-load orders")

    def state(reason):
        covered=max(unresolved,-frontier[0][0] if frontier else -1)
        upper=None if covered<0 else Fraction(covered,r.units)
        if value is not None:
            upper=value if upper is None else max(value,upper)
        status="infeasible" if upper is None else ("optimal" if upper==value else "timeout")
        return dict(status=status,lb=value,ub=upper,lb_exact=None if value is None else str(value),
                    ub_exact=None if upper is None else str(upper),
                    route=None if witness is None else witness["route"],
                    amounts=None if witness is None else witness["amounts"],certificate="rational_omission_time_targets",
                    metrics={**stats,"stop_reason":reason,"pending_groups":len(frontier),
                             "unresolved_upper_scaled":unresolved,"time_exclusions":list(proofs)})

    last=None
    while frontier and monotonic()<deadline and stats["examined"]<max_sets:
        neg,mask,start=frontier[0]
        cap=-neg
        if value is not None and cap<=value*r.units:
            break
        sites=tuple(v for i,v in enumerate(nodes) if mask&(1<<i))
        service=sum(r.service[v] for v in sites)
        exclusion=None
        if service>r.limit:
            exclusion=dict(sites=sites,kind="service",bound=service)
        else:
            costs=_matrix(base,sites)
            assign=integer_assignment(costs,deadline)
            if assign is None or monotonic()>=deadline:
                break
            has_matching=verify_assignment(costs,assign)
            if not has_matching or assign["cost"]+service>r.limit:
                exclusion=dict(sites=sites,kind="assignment",proof=assign)
            elif tree:
                at=monotonic()
                cert=lower_bound(r,sites,deadline)
                stats["tree_seconds"]+=monotonic()-at
                if cert is not None and cert["lower_scaled"]>r.limit:
                    if replay_tree(r,sites,cert)<=r.limit:
                        raise ArithmeticError("Tree exclusion did not survive integer replay")
                    exclusion=dict(sites=sites,kind="tree",proof=cert)
        if monotonic()>=deadline:
            break
        if exclusion is None and fixed:
            at=monotonic()
            stats["target_attempts"]+=1
            # Reserve some time for the other still-live masks even if this target is difficult.
            local_end=min(deadline,at+min(target_seconds,.45*(deadline-at)))
            result=find_order(r,sites,local_end)
            stats["construction_seconds"]+=monotonic()-at
            if result["status"]=="found" and monotonic()<deadline:
                score=check(r.instance,result["route"],result["amounts"],tol=0)["objective"]
                if set(result["route"])!=set(sites) or score!=Fraction(result["lb_exact"]) or score*r.units>cap:
                    raise ArithmeticError("Unverified fixed-load solution")
                if monotonic()>=deadline:
                    break
                stats["target_found"]+=1
                if value is None or score>value:
                    value,witness=score,{"route":result["route"],"amounts":result["amounts"]}
                    if emit is not None and monotonic()<deadline:
                        emit(state("verified_fixed_load_witness"))
        # The parent remains in the heap until every still-promising child is ready.
        children=[]
        for i in range(start,len(nodes)):
            if mask&(1<<i):
                child=mask^(1<<i)
                child_sites=tuple(v for j,v in enumerate(nodes) if child&(1<<j))
                bound=_cap(r,child_sites)
                if bound>0 and (value is None or bound>value*r.units):
                    children.append((-bound,child,i+1))
        if monotonic()>=deadline:
            break
        heappop(frontier)
        for child in children:
            heappush(frontier,child)
        stats["examined"]+=1
        if exclusion is not None:
            proofs.append(exclusion)
            stats[exclusion["kind"]+"_excluded"]+=1
        elif value is None or cap>value*r.units:
            # Failure to achieve one fixed allocation has no implication for other allocations.
            unresolved=max(unresolved,cap)
            stats["unresolved_sets"]+=1
        row=state("visited_set_frontier")
        signature=(row["lb_exact"],row["ub_exact"])
        if emit is not None and signature!=last and monotonic()<deadline:
            emit(row)
        last=signature
        if row["status"] in ("optimal","infeasible"):
            return row
    return state("deadline" if monotonic()>=deadline else "set_cap" if stats["examined"]>=max_sets
                 else "remaining_allocations" if unresolved>=0 else "covered")


def certify_upper(r,upper,exclusions):
    """Check just the exclusions actually needed to prove a reported bound; enumerate no route orders."""
    bound=Fraction(upper)*r.units
    base=arc_costs(r)
    nodes=tuple(r.nodes)
    proof_map={tuple(p["sites"]):p for p in exclusions}
    if len(proof_map)!=len(exclusions):
        raise ValueError("Repeated visit-set certificate")
    used=[]
    def scan(sites,start):
        if _cap(r,sites)<=bound:
            return
        cert=proof_map.get(sites)
        if cert is None:
            raise ValueError("Uncovered potentially improving exact set")
        service=sum(r.service[v] for v in sites)
        if cert["kind"]=="tree":
            time_bound=replay_tree(r,sites,cert["proof"])
        elif cert["kind"]=="assignment":
            match=verify_assignment(_matrix(base,sites),cert["proof"])
            time_bound=service+cert["proof"]["cost"] if match else r.limit+1
        elif cert["kind"]=="service" and service==cert["bound"]:
            time_bound=service
        else:
            raise ValueError("Unsupported time exclusion")
        if time_bound<=r.limit:
            raise ValueError("Time exclusion does not exceed the horizon")
        used.append(sites)
        for i in range(start,len(nodes)):
            if nodes[i] in sites:
                scan(tuple(v for v in sites if v!=nodes[i]),i+1)
    scan(nodes,0)
    return len(used)
