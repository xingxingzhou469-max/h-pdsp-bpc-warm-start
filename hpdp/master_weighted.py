"""Experimental restricted-master prices for the native weighted prefix search."""
import ctypes
from fractions import Fraction
from hashlib import sha256
from math import gcd, isfinite
from time import monotonic
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, linprog

from . import priced_routes as pr
from . import weighted_prefix as wp
from .allocation import allocate
from .check import check


def propose(cuts, weights, deadline):
    """Suggest clipped integral prices. The numerical epigraph value is discarded."""
    remaining = deadline-monotonic()
    if not cuts or remaining <= 0:
        return None
    visits = np.asarray(cuts, dtype=np.float64)
    columns = np.column_stack((1-visits, -np.ones(len(visits))))
    rhs = -(visits @ weights)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=OptimizeWarning, message="Unrecognized options.*")
        result = linprog([0]*len(weights)+[1], A_ub=columns, b_ub=rhs,
                         bounds=[(0, w) for w in weights]+[(0, None)], method="highs",
                         options={"threads": 1, "time_limit": remaining})
    if monotonic() >= deadline or not result.success or not np.isfinite(result.x[:-1]).all():
        return None
    return tuple(max(0, min(w, round(x))) for w, x in zip(weights, result.x[:-1]))


def separate(table, weights, max_cuts):
    """Exact maximum over all donor starts, with valid walks for the master."""
    scores = [(value, -index, start) for index, start in enumerate(table.graph.starts)
              if (value := table.root(start, weights)) is not None]
    if not scores:
        return None, []
    cuts = []
    for value, _, start in sorted(scores, reverse=True)[:max_cuts]:
        frequencies = [0]*table.graph.n
        for stop in (start.first, *table.route(start)):
            frequencies[stop] += 1
        visits = tuple(frequencies)
        reconstructed = sum(w*c+p*(1-c) for w, p, c in zip(weights, table.prices, visits))
        if reconstructed != value:
            raise ArithmeticError("An integer root walk does not reproduce its separated cut")
        cuts.append(visits)
    return max(scores)[0], cuts


def prepare_master(r, graph, deadline, seed, tracked, rounds, stats):
    first_pop = r.people if seed is None else sum(r.pop[v] for v in seed["route"])
    first_supply = sum(r.supply) if seed is None else sum(r.supply[v] for v in seed["route"])
    divisor = gcd(first_pop, first_supply)
    a, b = first_pop//divisor, first_supply//divisor
    weights, unit = wp.rounded_weights(r, a, b)
    ordered = sorted(r.nodes, key=lambda v: (-weights[v], v))
    pivot = weights[ordered[len(ordered)//4]]
    initial = tuple(max(0, w-pivot) for w in weights)
    while tracked and graph.n*(r.limit+1)*(1 << tracked)*12 > 64_000_000:
        tracked -= 1

    now = monotonic()
    # Leave at least half of the shared table budget for the more expensive
    # tracked table. Zero-bit candidate timings and incomplete work still count.
    scout_end = (now+deadline)/2 if tracked else deadline
    initial_table = pr.Table(graph, weights, initial, (), True, deadline)
    if not initial_table.handle:
        return []
    best = initial_table
    tracked_table = None
    try:
        initial_value, initial_cuts = separate(initial_table, weights, 6)
        stats["initial_untracked_reward"] = initial_value
        stats["integer_rounds"] = 1
        if initial_value is None:
            return [(best, a, b, unit)]
        visits = initial_cuts[0]
        track_order = sorted(ordered, key=lambda v: -max(0, visits[v]-1))
        explicit = tuple(sorted(track_order[:tracked]))
        stats["tracked_vertices"] = explicit
        cuts = dict.fromkeys(initial_cuts)
        seen = {initial}
        best_value = initial_value
        for _ in range(rounds):
            if monotonic() >= scout_end:
                break
            started = monotonic()
            proposed = propose(list(cuts), weights, scout_end)
            stats["master_seconds"] += monotonic()-started
            stats["master_calls"] += 1
            if proposed is None or proposed in seen:
                break
            seen.add(proposed)
            candidate = pr.Table(graph, weights, proposed, (), True, scout_end)
            if not candidate.handle:
                break
            try:
                stats["integer_rounds"] += 1
                reward, more = separate(candidate, weights, 6)
                if reward is None:
                    raise ArithmeticError("Pricing changed the feasible relaxation topology")
                cuts.update(dict.fromkeys(more))
                if reward < best_value:
                    best.close()
                    best, best_value = candidate, reward
                    candidate = None
            finally:
                if candidate is not None:
                    candidate.close()
        stats["selected_untracked_reward"] = best_value
        stats["master_cuts"] = len(cuts)
        if explicit and monotonic() < deadline:
            proposed = best.prices if best_value < initial_value else tuple(
                max(0, min(w, p+max(1,pivot//2)*(visits[v]-1)))
                for v, (w,p) in enumerate(zip(weights, initial)))
            prices = tuple(0 if v in explicit else p for v,p in enumerate(proposed))
            tracked_table = pr.Table(graph, weights, prices, explicit, True, deadline)
            if tracked_table.handle:
                return [(best, a, b, unit), (tracked_table, a, b, unit)]
        return [(best, a, b, unit)]
    except BaseException:
        best.close()
        if tracked_table is not None:
            tracked_table.close()
        raise


def table_summary(r, graph, tables):
    descriptions = []
    for table, a, b, unit in tables:
        weights, _ = wp.rounded_weights(r, a, b, unit)
        roots = [value for start in graph.starts if (value := table.root(start, weights)) is not None]
        descriptions.append(dict(
            supply_weight=a, population_weight=b, unit=unit,
            tracked=[v for v in r.nodes if table.bits[v]],
            prices_sha256=sha256(str(table.prices).encode()).hexdigest(),
            root_reward=max(roots, default=None),
        ))
    return descriptions


def solve(r, deadline, seed, *, strategy="master", emit=None, roots_only=False,
          tracked=4, rounds=12, table_seconds=4.0, heap_limit=100_000, binary=None,
          current_global_upper=None):
    """Compare two price strategies on the same strict search ABI and wall clock."""
    if not wp.supported(r):
        raise ValueError("Master-priced prefix study needs the weighted search's integer domain")
    if (strategy not in ("static", "master") or not 0 <= tracked <= 8 or not 0 <= rounds <= 128
            or not isfinite(table_seconds) or table_seconds < 0 or not 1 <= heap_limit < 2**31):
        raise ValueError("Invalid price, table, or heap configuration")
    start = monotonic()
    witness = seed
    lb = None if seed is None else check(r.instance, seed["route"], seed["amounts"], tol=0)["objective"]
    upper = r.prior
    stats = dict(strategy=strategy, phase="table_build", tables=[], master_calls=0, master_seconds=0.0,
                 master_cuts=0, integer_rounds=0, strict_root_exact=None)
    errors = []

    def numerator(value):
        if value is None:
            return -1
        num = value*r.units
        if num.denominator != 1:
            raise ArithmeticError("Study expects a route-optimal common seed and integral route scores")
        return num.numerator

    def record():
        bound = None if upper < 0 else Fraction(upper, r.units)
        if lb is not None and (bound is None or bound < lb):
            raise ArithmeticError("An integer certificate contradicts a verified route")
        status = "infeasible" if bound is None else "optimal" if bound == lb else "timeout"
        return dict(status=status, lb=lb, ub=bound, lb_exact=None if lb is None else str(lb),
                    ub_exact=None if bound is None else str(bound), certificate="rational_master_weighted_frontier",
                    route=None if witness is None else witness["route"],
                    amounts=None if witness is None else witness["amounts"], metrics=stats.copy())

    if monotonic() >= deadline:
        return record()
    if lb is not None and (lb*r.units).denominator!=1:
        improved,amounts=allocate(r.instance,witness["route"])
        if improved<lb or improved!=check(r.instance,witness["route"],amounts,tol=0)["objective"]:
            raise ArithmeticError("Supplied witness failed fixed-route reallocation")
        if monotonic()>=deadline: return record()
        lb=improved
        witness={"route":witness["route"],"amounts":amounts}
        if emit: emit(record())
    if numerator(lb)==upper: return record()
    graph = pr.Graph(r, False)
    stop = min(deadline, start+min(table_seconds, 0.4*(deadline-start)))
    try:
        if strategy == "static":
            tables = wp.prepare_tables(r, graph, stop, witness, tracked)
        elif monotonic() < stop:
            tables = prepare_master(r, graph, stop, witness, tracked, rounds, stats)
        else:
            tables = []
    except OverflowError:
        tables=[]
        stats["walk_price_domain"]="reward range exceeded"
    try:
        stats["tables"] = table_summary(r, graph, tables)
        stats["prep_seconds"] = monotonic()-start
        if monotonic() >= deadline:
            return record()

        @wp.Witness
        def offer(length, nodes, prefix_bound, reply):
            nonlocal lb, witness
            try:
                if monotonic() >= deadline:
                    return
                route = tuple(nodes[:length])
                value, amounts = allocate(r.instance, route)
                if lb is None or value > lb:
                    if check(r.instance, route, amounts, tol=0)["objective"] != value or numerator(value) > prefix_bound:
                        raise ArithmeticError("Verified route contradicts an active prefix certificate")
                    if monotonic() >= deadline:
                        return
                    lb, witness = value, dict(route=route, amounts=amounts)
                if monotonic() < deadline:
                    reply[0] = numerator(lb)
            except BaseException as exc:
                errors.append(exc)

        @wp.Checkpoint
        def checkpoint(bound, counters, proceed):
            nonlocal upper
            try:
                if monotonic() >= deadline:
                    return
                upper = min(upper, bound)
                stats.update(zip(wp.COUNTERS, counters[:len(wp.COUNTERS)]))
                if emit:
                    emit(record())
                if monotonic() < deadline:
                    proceed[0] = 1
            except BaseException as exc:
                errors.append(exc)

        with wp.Context(r, graph, tables, binary=binary) as context:
            root_bounds = [bound for v in r.p if (bound := context.bound((v,))) is not None]
            upper = min(upper, max(root_bounds, default=-1))
            stats["phase"] = "root"
            stats["strict_root_exact"] = None if upper < 0 else str(Fraction(upper,r.units))
            no_gain=(current_global_upper is not None and upper>=0 and
                     Fraction(upper,r.units)>=current_global_upper)
            stats["kept_existing_frontier"]=no_gain
            if emit and monotonic() < deadline:
                emit(record())
            if roots_only or no_gain or upper <= numerator(lb) or monotonic() >= deadline:
                return record()
            stats["phase"] = "search"
            native_upper = wp.Long(upper)
            counts = (wp.Long*len(wp.COUNTERS))()
            status = context.lib.search(context.handle, 1, heap_limit, numerator(lb),
                                        max(0, deadline-monotonic()), offer, checkpoint,
                                        ctypes.byref(native_upper), counts)
        if errors:
            raise errors[0]
        if status < 0:
            raise RuntimeError(f"Native search failed with status {status}")
        upper = min(upper, native_upper.value)
        stats.update(zip(wp.COUNTERS, counts))
        return record()
    finally:
        for table, *_ in tables:
            table.close()
