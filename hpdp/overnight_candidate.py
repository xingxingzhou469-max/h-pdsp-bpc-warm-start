"""Structure-dependent exact H-PDSP search; the seed is shared by all methods."""

from fractions import Fraction
from time import monotonic

from .check import check
from .chain_oracle import is_supported as chain_supported, solve_chain
from .dense_witness import solve_dense_blocks
from .dual_population import solve as solve_dual
from .hfb_research import improve
from .fragment import improve as fragment_incumbent
from .large_incumbent import improve as beam_incumbent
from .large_sets import solve as solve_sets, supported as sets_supported
from .mandatory_walks import MandatoryGraph, solve_bound as mandatory_bound, solve_exact as mandatory_exact
from . import population_sections
from .mandatory_wide import solve_exact as mandatory_wide_exact, supported as wide_supported
from .master_weighted import solve as solve_master_prices
from .omission_targets import solve as solve_omission_targets
from .order_frontier import solve as solve_order_frontier, supported as order_supported
from .portfolio import solve_portfolio
from .prefix import Label, prefix_bound
from .prize import prize_bound
from .priced_routes import solve as solve_walks, supported as walks_supported
from .v8_plugins import certify as propose_dual, improve as propose_primal
from .sparse_sets import solve as solve_sparse, supported as sparse_supported
from .subset import solve_subset
from .weighted_prefix import solve as solve_weighted, supported as weighted_supported
from .certificate_support import _dense_layout


def solve_candidate(r, deadline, seed, *, emit=None):
    n = len(r.nodes)
    prior = Fraction(r.prior, r.units)
    value = None if seed is None else check(r.instance, seed["route"], seed["amounts"], tol=0)["objective"]
    current = seed
    upper = prior
    dual = {}
    started = monotonic()
    metrics = {"selection": None, "quick_subset_seconds": 0.0, "incumbent_seconds": 0.0}

    def pack(witness, score, bound, status, certificate, detail):
        return dict(status=status, certificate=certificate, lb=score, ub=bound,
                    lb_exact=None if score is None else str(score),
                    ub_exact=None if bound is None else str(bound),
                    route=None if witness is None else witness["route"],
                    amounts=None if witness is None else witness["amounts"],
                    metrics={**metrics, "search":detail}, **dual)

    def merge(row):
        nonlocal current, value, upper
        lower = None if row["lb_exact"] is None else Fraction(row["lb_exact"])
        bound = None if row["ub_exact"] is None else Fraction(row["ub_exact"])
        if lower is not None and (value is None or lower > value):
            value = lower
            current = {"route":row["route"], "amounts":row["amounts"]}
        if bound is None:
            if row["status"] != "infeasible" or value is not None:
                raise ArithmeticError("Missing global upper bound in an active feasible search")
            return pack(None,None,None,"infeasible",row["certificate"],row["metrics"])
        upper = min(upper, bound)
        if value is not None and value > upper:
            raise ArithmeticError("Search phases have incompatible exact bounds")
        return pack(current,value,upper,"optimal" if value == upper else "timeout",
                    row["certificate"],row["metrics"])

    def observe(row):
        merged = merge(row)
        if emit and monotonic() < deadline:
            emit(merged)

    def incumbent(witness, score):
        nonlocal current, value
        if value is None or score > value:
            current, value = witness, score
            if value > upper:
                raise ArithmeticError("Improved route exceeds the active strict upper bound")
            if emit and monotonic() < deadline:
                emit(pack(current,value,upper,"optimal" if value==upper else "timeout",
                          "rational_candidate_frontier",{"phase":"incumbent"}))

    if value == prior or started >= deadline:
        metrics["selection"] = "analytic_prior"
        return pack(current,value,prior,"optimal" if value==prior else "timeout","analytic_prior",{})
    # All components start from the same publicly generated seed. Preceding
    # phases may hand a better verified witness and a tighter global bound on.
    r.prepare_bounds()
    empty=Label()
    root=prefix_bound(r,empty,joint=True)
    if n>=24 and root is not None and monotonic()<deadline:
        root=min(root,prize_bound(r,empty))
        children=[]
        phase_limit=min(deadline,monotonic()+0.28)
        for supplier in r.p:
            if monotonic()>=phase_limit:
                break
            child=empty.extend(r,supplier)
            child_upper=prefix_bound(r,child,joint=True)
            if child_upper is not None:
                children.append(min(child_upper,prize_bound(r,child)))
        else:
            # Every legal nonempty route has exactly one first supplier. The
            # maximum can replace the root only after every alternative was seen.
            root=min(root,max(children)) if children else None
    if root is None:
        if value is not None:
            raise ArithmeticError("Root time bound contradicts a verified witness")
        return pack(None,None,None,"infeasible","rational_prefix_root",{})
    upper=min(upper,Fraction(root,r.units))
    if value is not None and value>upper:
        raise ArithmeticError("Root upper bound below the common seed")
    root_result=pack(current,value,upper,"optimal" if value==upper else "timeout","rational_prefix_root",{})
    if emit and monotonic()<deadline:
        emit(root_result)
    if value==upper or monotonic()>=deadline:
        return root_result
    if 22<=n<=28 and 10*sum(r.service)<9*r.limit and sparse_supported(r):
        metrics["selection"]="target_load_construction"
        now=monotonic()
        probe=merge(solve_sparse(r,min(deadline,now+min(.9,.15*max(0,deadline-now))),current,emit=observe))
        if emit and monotonic()<deadline: emit(probe)
        if probe["status"] in ("optimal","infeasible"): return probe
        if current is not None and value is not None and 4*value*r.units >= 3*r.prior and monotonic()<deadline:
            metrics["selection"]="omission_time_and_fixed_targets"
            started_probe=monotonic()
            probe_budget=min(2.0,.25*max(0,deadline-started_probe))
            metrics["omission_probe_budget"]=probe_budget
            row=solve_omission_targets(r,started_probe+probe_budget,current,emit=observe)
            metrics["omission_probe_seconds"]=monotonic()-started_probe
            routed=merge(row)
            if emit and monotonic()<deadline: emit(routed)
            if routed["status"] in ("optimal","infeasible") or monotonic()>=deadline: return routed
    if sets_supported(r) and (n<=22 or 10*sum(r.service)>=9*r.limit):
        if n<=20 and order_supported(r):
            metrics["selection"]="sets_with_required_deliveries"
            return merge(solve_order_frontier(r,deadline,current,emit=observe,cache_slots=0,
                                               prefix_rank_bound=False,minimum_delivery=True))
        metrics["selection"]="threshold_sets"
        return merge(solve_sets(r,deadline,current,emit=observe))
    if n<=20:
        try:
            _dense_layout(n,max(map(max,r.travel)),sum(r.service))
        except ValueError:
            pass
        else:
            metrics["selection"]="reverse_sets_with_block_dominance"
            return merge(solve_dense_blocks(r,deadline,current,emit=observe))
    if 25<=n<=48 and seed and len(seed["route"])<=6:
        metrics["selection"] = "small_subset_probe"
        probe = solve_subset(r,min(deadline, monotonic()+0.12),current,emit=observe,max_records=10000)
        metrics["quick_subset_seconds"] = monotonic()-started
        resolved = merge(probe)
        if emit and monotonic() < deadline:
            emit(resolved)
        if resolved["status"] in ("optimal","infeasible"):
            return resolved

    few_suppliers=(len(r.d)>=25 and 2<=len(r.p)<=5 and MandatoryGraph.supported(r) and walks_supported(r))
    exact_range=(len(r.d)<=63 and r.people*sum(r.supply)<2**63 and r.capacity<2**63)
    if few_suppliers and sum(r.supply)<=r.capacity and (exact_range or wide_supported(r)):
        if chain_supported(r):
            metrics["selection"]="chronological_supply_chain"
            now=monotonic()
            trial=merge(solve_chain(r,min(deadline,now+min(12,.45*max(0,deadline-now))),current,emit=observe))
            if emit and monotonic()<deadline: emit(trial)
            if trial["status"] in ("optimal","infeasible") or monotonic()>=deadline: return trial
        # Here even all the declared supply fits in the truck. The chronological
        # cumulative-supply bound is particularly effective with the common seed.
        metrics["selection"]="nonbinding_capacity_mandatory_suppliers"
        search=mandatory_exact if exact_range else mandatory_wide_exact
        now=monotonic()
        proof_end=deadline-min(.85,.025*max(0,deadline-now))
        trial=merge(search(r,proof_end,current,emit=observe))
        if emit and monotonic()<deadline: emit(trial)
        if trial["status"] in ("optimal","infeasible") or monotonic()>=deadline: return trial
        metrics["selection"]="remaining_projected_bound"
        return merge(solve_walks(r,deadline,current,emit=observe,bound_only=monotonic()>=proof_end-.1))
    t=monotonic()
    # Few-supplier instances that fit entirely in the truck have already
    # returned above. A capacity-binding route can still benefit from a beam.
    if n>=35 and len(r.p)>1 and current:
        short=32<=n<=48 and len(current["route"])<=8
        budget=min(0.24 if short else 2.2,max(0,deadline-t)*0.22)
        metrics["selection"]="beam_incumbent"
        beam_incumbent(r,min(deadline,t+budget),current,emit=incumbent,
                       width=256 if short else 28,per_end=16 if short else 2)
    if 35 <= n < 128 and current and value != upper and len(r.p) >= 2:
        now = monotonic()
        metrics["selection"] = "fragment_incumbent"
        metrics["fragment"] = fragment_incumbent(
            r, min(deadline, now + min(3.0, 0.10 * max(0.0, deadline - now))),
            current, emit=incumbent,
        )
        metrics["fragment_seconds"] = monotonic() - now
    if n >= 38 and len(r.p) >= 2 and value != upper:
        now = monotonic()
        budget = min(5.0, 0.09 * max(0, deadline-now))
        if budget >= 0.08:
            metrics["selection"] = "product_alns"
            metrics["primal_rejected"] = propose_primal(
                r, now+budget, current, emit=incumbent, should_stop=lambda: value==upper)
            metrics["primal_seconds"] = monotonic()-now
    if current and value!=upper and len(r.p)>=2 and len(current["route"])>=max(7,(n+2)//3):
        metrics["selection"] = "neighborhood"
        now=monotonic()
        improve(r,min(deadline,now+min(0.65,max(0,deadline-now)*0.12)),current,emit=incumbent)
    metrics["incumbent_seconds"]=monotonic()-t
    if value==upper:
        return pack(current,value,upper,"optimal","rational_candidate_frontier",{})
    if sets_supported(r):
        # On 23–24 sites with short service, the first full suffix table may
        # dominate the limit. Preserve a budget for the alternative proof.
        t=monotonic()
        metrics["selection"]="limited_threshold_sets"
        trial=merge(solve_sets(r,min(deadline,t+min(8,max(0,deadline-t)*0.22)),current,emit=observe))
        if emit and monotonic()<deadline: emit(trial)
        if trial["status"] in ("optimal","infeasible"): return trial
    if walks_supported(r) and n>=25 and len(r.p)==1:
        metrics["selection"]="single_supplier_walks"
        now=monotonic()
        first=merge(solve_walks(r,min(deadline,now+min(1.5,.25*max(0,deadline-now))),current,emit=observe))
        if emit and monotonic()<deadline: emit(first)
        if first["status"] in ("optimal","infeasible"): return first
        metrics["selection"]="single_supplier_dual_prices"
        return merge(solve_dual(r,deadline,current,emit=observe))
    if few_suppliers and current:
        graph=population_sections.active_graph(r,value)
        if graph is not None:
            now=monotonic()
            metrics["selection"]="priced_supplier_fairness"
            end=min(deadline,now+min(18,.30*max(0,deadline-now)))
            probe=merge(population_sections.solve(r,graph,end,current,upper,observe))
            if emit and monotonic()<deadline: emit(probe)
            if probe["status"]=="optimal" or monotonic()>=deadline: return probe
    if few_suppliers:
        now=monotonic()
        metrics["selection"]="projected_supplier_bound"
        merge(solve_walks(r,min(deadline,now+min(1.0,.11*max(0,deadline-now))),current,emit=observe,bound_only=True))
        if value==upper: return pack(current,value,upper,"optimal","rational_candidate_frontier",{})
        if exact_range or wide_supported(r):
            metrics["selection"]="mandatory_suppliers_and_prefix_equity"
            search=mandatory_exact if exact_range else mandatory_wide_exact
            trial=merge(search(r,deadline,current,emit=observe))
            if trial["status"] in ("optimal","infeasible") or monotonic()>=deadline: return trial
        else:
            metrics["selection"]="mandatory_supplier_bound"
            now=monotonic()
            trial=merge(mandatory_bound(r,min(deadline,now+min(2.5,.20*max(0,deadline-now))),current,emit=observe))
            if emit and monotonic()<deadline: emit(trial)
            if trial["status"] in ("optimal","infeasible") or monotonic()>=deadline: return trial
    if weighted_supported(r):
        metrics["selection"]="weighted_prefix_probe"
        now=monotonic()
        probe=merge(solve_weighted(r,min(deadline,now+min(2.0,.25*max(0,deadline-now))),current,emit=observe))
        if emit and monotonic()<deadline: emit(probe)
        if probe["status"] in ("optimal","infeasible") or monotonic()>=deadline: return probe
        metrics["selection"]="joint_reward_root"
        now=monotonic()
        root=merge(solve_walks(r,min(deadline,now+min(3.8,.22*max(0,deadline-now))),current,emit=observe,bound_only=True))
        if root["status"] in ("optimal","infeasible") or monotonic()>=deadline: return root
        proof_schedule_upper = upper
        if n>=40:
            now = monotonic()
            end = now + min(9.0, 0.17*max(0, deadline-now))
            metrics["selection"] = "deep_routing_dual"

            def accept_bound(bound, document):
                nonlocal upper
                if value is not None and (bound is None or bound < value):
                    raise ArithmeticError("Replayed dual certificate excludes a verified incumbent")
                if bound is None or bound < upper:
                    upper = bound
                    dual.update(dual_certificate=document,
                                dual_upper_exact=None if bound is None else str(bound))
                    status = "infeasible" if bound is None else "optimal" if bound == value else "timeout"
                    if emit and monotonic() < deadline:
                        emit(pack(current,value,upper,status,"verified_routing_dual",{}))
                return upper is not None and upper != value and monotonic() < end

            metrics["dual_rejected"] = propose_dual(
                r, getattr(r,"input_sha256",None), end, current, emit=accept_bound)
            metrics["dual_seconds"] = monotonic()-now
            if upper is None or value == upper or monotonic() >= deadline:
                status = "infeasible" if upper is None else "optimal" if value == upper else "timeout"
                certificate = "verified_routing_dual" if dual else "rational_candidate_frontier"
                return pack(current,value,upper,status,certificate,{})
        metrics["selection"]="master_priced_prefix"
        priced=merge(solve_master_prices(r,deadline,current,emit=observe,current_global_upper=proof_schedule_upper))
        if emit and monotonic()<deadline: emit(priced)
        if priced["status"] in ("optimal","infeasible") or monotonic()>=deadline: return priced
        metrics["selection"]="weighted_elementary_prefix"
        return merge(solve_weighted(r,deadline,current,emit=observe))
    if walks_supported(r):
        metrics["selection"]="priced_walks"
        return merge(solve_walks(r,deadline,current,emit=observe))
    metrics["selection"]="two_stage"
    return merge(solve_portfolio(r,deadline,current,emit=observe))
