"""Exact weighted-reward certificates from a floating directed routing LP."""
from fractions import Fraction
from itertools import islice
from math import gcd, isfinite
from time import monotonic

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .check import check
from .model import LP_OPTIONS, _run_highs, _violated_cuts
from .priced_routes import cap_product
from .routing_exact import Row, Routing, _route_reward, exact_upper, flow_cap, rewards, routing


def slopes(r, seed, count):
    if not 1 <= count <= 3:
        raise ValueError("Choose one to three fixed slope multipliers")
    if seed is not None:
        route = seed["route"]
        population = sum(r.pop[v] for v in route)
        flow = min(sum(r.supply[v] for v in route),r.capacity*sum(r.pop[v]>0 for v in route))
    else:
        population = r.people
        flow = flow_cap(r)
    values = []
    for x,y in ((1,1),(2,1),(1,2))[:count]:
        a,b = x*population,y*flow
        common = gcd(a,b)
        values.append((a//common,b//common))
    return values


def matrix(rows, columns):
    i, j, values, rhs = [], [], [], []
    for k, row in enumerate(rows):
        rhs.append(row.rhs/row.float_scale)
        for column, coefficient in row.terms:
            i.append(k)
            j.append(column)
            values.append(coefficient/row.float_scale)
    return coo_matrix((values,(i,j)),shape=(len(rows),columns)).tocsc(), rhs


def quantize(inequalities, equalities, raw_ub, raw_eq, objective_scale, grid=10**8):
    """Invert SciPy's minimization marginals and round on a reproducible grid."""
    if type(grid) is not int or grid <= 0:
        raise ValueError("The integer grid factor must be positive")
    if (raw_ub is None or raw_eq is None or len(raw_ub)!=len(inequalities)
            or len(raw_eq)!=len(equalities) or any(not isfinite(x) for x in (*raw_ub,*raw_eq))):
        return None
    denominator = grid*max(row.float_scale for row in (*inequalities,*equalities))
    def convert(marginal, row):
        return round(Fraction.from_float(-float(marginal))*objective_scale*denominator/row.float_scale)
    nonnegative = [max(0,convert(x,row)) for x,row in zip(raw_ub,inequalities,strict=True)]
    free = [convert(x,row) for x,row in zip(raw_eq,equalities,strict=True)]
    return dict(denominator=denominator,nonnegative=nonnegative,free=free)


def _finite(x):
    return float(x) if x is not None and isfinite(x) else None


def _suggest_lp(model, objective, inequalities, eq_matrix, eq_rhs, deadline, grid):
    started = monotonic()
    ineq_matrix, ineq_rhs = matrix(inequalities,model.columns)
    remaining = deadline-monotonic()
    observation = {"status":"deadline_before_lp","elapsed":monotonic()-started}
    if remaining <= 0:
        return observation,None
    scale = max(1,*objective)
    lp_started = monotonic()
    try:
        raw = _run_highs(linprog,c=[-x/scale for x in objective],A_ub=ineq_matrix,
                         b_ub=ineq_rhs,A_eq=eq_matrix,b_eq=eq_rhs,bounds=(0,1),
                         method="highs-ds",options={**LP_OPTIONS,"time_limit":remaining})
    except (RuntimeError,ValueError) as exc:
        return dict(status="solver_error",message=str(exc),
                    certificate_status="unavailable",scipy_seconds=monotonic()-lp_started,
                    elapsed=monotonic()-started),None
    info = dict(status=int(raw.status),message=str(raw.message),
                numerical_reward_normalized=None if raw.fun is None else _finite(-raw.fun),
                objective_scale=scale, grid_factor=grid, scipy_seconds=monotonic()-lp_started,
                matrix_seconds=lp_started-started)
    if monotonic() >= deadline:
        info.update(certificate_status="deadline_after_lp",elapsed=monotonic()-started)
        return info,None
    raw_ub = (raw.get("ineqlin") or {}).get("marginals")
    raw_eq = (raw.get("eqlin") or {}).get("marginals")
    cert = quantize(inequalities,model.degrees,raw_ub,raw_eq,scale,grid)
    info["certificate_status"] = "no_finite_multipliers"
    if cert is not None:
        info["raw_inequality_marginals"] = list(map(float,raw_ub))
        info["raw_equality_marginals"] = list(map(float,raw_eq))
        bound = exact_upper(objective,inequalities,model.degrees,**cert)
        if monotonic() < deadline:
            info.update(certificate=cert,certificate_status="integer_replay",**bound)
        else:
            info["certificate_status"] = "deadline_after_integer_replay"
    info["elapsed"] = monotonic()-started
    return info,raw.get("x")


def separate(model, hint, known, deadline, limit):
    """Generate a bounded number of valid outward cuts using the frozen separator."""
    if hint is None or len(hint)!=model.columns or limit<=0 or monotonic()>=deadline:
        return []
    if not all(isfinite(x) for x in hint) or model.nodes*10_000_000 >= 2**31:
        return []
    x = np.zeros((model.nodes,model.nodes))
    for arc,col in model.arcs.items():
        x[arc] = hint[col]
    y = [0.0]+[hint[col] for col in model.visits.values()]
    candidates = ((tuple(sorted(side)),k) for side,k in _violated_cuts(x,y,deadline))
    return list(islice((cut for cut in candidates if cut not in known),limit))


def observation_upper(r, model, report):
    """Replay a saved LP observation without invoking SciPy or trusting its status."""
    if report.get("certificate_status") != "integer_replay":
        return None
    if report.get("pickup_blocks", False) != (model.flow_ceiling is not None):
        raise ValueError("Recorded LP variant does not match its replay model")
    a,b = report["slope"]
    objective = rewards(r,model,a,b)
    inequalities = model.inequalities(report["cuts"])
    if "grid_factor" in report:
        rounded = quantize(inequalities,model.degrees,report["raw_inequality_marginals"],
                           report["raw_equality_marginals"],max(1,*objective),report["grid_factor"])
        if rounded!=report["certificate"]:
            raise ValueError("The stored integer multipliers differ from the declared grid recovery")
    replay = exact_upper(objective,inequalities,model.degrees,**report["certificate"])
    if any(report[key]!=value for key,value in replay.items()):
        raise ValueError("The recorded strict LP bound differs from exact replay")
    weighted = replay["numerator"]//replay["denominator"]
    if weighted != report["weighted_integer_upper"]:
        raise ValueError("The recorded rounded reward bound differs from exact replay")
    upper = -1 if weighted < 0 else cap_product(weighted,a,b,flow_cap(r),r.people)
    expected = None if upper < 0 else str(Fraction(upper,r.units))
    if report["fairness_upper_exact"]!=expected:
        raise ValueError("The recorded fair objective bound differs from exact replay")
    return upper


def solve(r, deadline, seed, *, connected=False, slope_count=3, cut_rounds=2, cuts_per_round=8,
          grid=10**8, emit=None, pickup_blocks=False):
    """LP root-bound side experiment. Retain every strictly verified bound until cutoff."""
    if not 1 <= slope_count <= 3 or cut_rounds<0 or cuts_per_round<1 or grid<1:
        raise ValueError("Invalid slope, cut or certificate-grid configuration")
    witness = seed
    lower = None if seed is None else check(r.instance,seed["route"],seed["amounts"],tol=0)["objective"]
    upper = r.prior
    stats = dict(variant="cuts" if connected else "plain",phase="build",lp_calls=0,
                 accepted_certificates=0,cut_sets=0,max_certified_cuts=0,latest_lp=None,
                 oracle_failures=0,scipy_seconds=0.0,separation_seconds=0.0)
    zero_flow = pickup_blocks and flow_cap(r) == 0
    if pickup_blocks:
        stats.update(pickup_blocks=True,flow_ceiling=flow_cap(r))

    def row():
        ub = Fraction(0) if zero_flow else None if upper<0 else Fraction(upper,r.units)
        if lower is not None and (ub is None or lower>ub):
            raise ArithmeticError("The exact LP certificate excludes the verified common seed")
        status = "infeasible" if ub is None else "optimal" if ub==lower else "timeout"
        return dict(status=status,lb=lower,ub=ub,lb_exact=None if lower is None else str(lower),
                    ub_exact=None if ub is None else str(ub),certificate="rational_routing_lp",
                    route=None if witness is None else witness["route"],
                    amounts=None if witness is None else witness["amounts"],metrics=stats.copy())

    def publish():
        if emit and monotonic()<deadline:
            emit(row())

    if zero_flow:
        stats["phase"] = "zero_flow"
        return row()
    if monotonic()>=deadline or lower == Fraction(upper,r.units):
        return row()
    start = monotonic()
    model = routing(r,pickup_blocks=pickup_blocks)
    a_eq,b_eq = matrix(model.degrees,model.columns)
    weights = slopes(r,seed,slope_count)
    stats.update(phase="roots",build_seconds=monotonic()-start,slopes=weights,
                 columns=model.columns,equalities=len(model.degrees))
    for index,(a,b) in enumerate(weights):
        now = monotonic()
        if now>=deadline:
            break
        slope_end = now+(deadline-now)/(len(weights)-index)
        objective = rewards(r,model,a,b)
        cuts, known = [], set()
        for iteration in range(1+cut_rounds if connected else 1):
            if monotonic()>=slope_end:
                break
            inequalities=model.inequalities(cuts)
            info,hint=_suggest_lp(model,objective,inequalities,a_eq,b_eq,slope_end,grid)
            stats["lp_calls"]+=1
            stats["scipy_seconds"]+=info.get("scipy_seconds",0.0)
            info.update(slope=(a,b),round=iteration,cuts=list(cuts),ineq_rows=len(inequalities))
            if pickup_blocks:
                info["pickup_blocks"] = True
            if info.get("certificate_status")=="integer_replay":
                weighted=info["numerator"]//info["denominator"]
                candidate=-1 if weighted<0 else cap_product(weighted,a,b,flow_cap(r),r.people)
                info.update(weighted_integer_upper=weighted,
                            fairness_upper_exact=None if candidate<0 else str(Fraction(candidate,r.units)))
                if seed is not None:
                    reward=_route_reward(r,model,a,b,seed["route"])
                    if reward>weighted:
                        raise ArithmeticError("The exact LP reward bound excludes the common route")
                stats["accepted_certificates"]+=1
                stats["max_certified_cuts"]=max(stats["max_certified_cuts"],len(cuts))
                upper=min(upper,candidate)
            else:
                stats["oracle_failures"]+=1
            stats["latest_lp"]=info
            publish()
            if upper<0 or lower==Fraction(upper,r.units):
                return row()
            if not connected or iteration==cut_rounds:
                break
            separation_start=monotonic()
            new=separate(model,hint,known,slope_end,cuts_per_round)
            stats["separation_seconds"]+=monotonic()-separation_start
            if not new:
                break
            known.update(new)
            cuts.extend(new)
            stats["cut_sets"]+=len(new)
    stats["phase"]="done"
    return row()
