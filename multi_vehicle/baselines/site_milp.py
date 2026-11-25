"""Cold-start three-index site MILP adapted from DeepSeek's FleetModel."""

from fractions import Fraction
from heapq import heapify, heappop, heappush
from math import inf, isfinite
from time import monotonic


class _Deadline(Exception):
    pass


def _exact(value):
    return value if isinstance(value, (int, Fraction)) else Fraction(str(value))


def _arc_set(instance, suppliers, agencies, check):
    """Prune by exact directed time, allowing paths through other stops."""
    sites = sorted(suppliers | agencies)
    nodes = [0, *sites]
    raw_travel = instance["travel"]
    travel = {}
    for i in nodes:
        check()
        travel[i] = {j: _exact(raw_travel[i][j]) for j in nodes if i != j}
    service = {i: _exact(instance["service"][i]) for i in sites}
    limit = _exact(instance["limit"])

    def shortest(initial, step):
        distances = dict(initial)
        queue = [(cost, node) for node, cost in initial.items()]
        heapify(queue)
        while queue:
            check()
            cost, i = heappop(queue)
            if cost != distances[i]:
                continue
            for j in sites:
                if i == j:
                    continue
                candidate = cost + step(i, j)
                if candidate < distances.get(j, inf):
                    distances[j] = candidate
                    heappush(queue, (candidate, j))
        return distances

    # Forward times include service at i. Reverse times exclude service at i.
    forward = shortest({i: travel[0][i] + service[i] for i in suppliers},
                       lambda i, j: travel[i][j] + service[j])
    backward = shortest({i: travel[i][0] for i in agencies},
                        lambda i, j: travel[j][i] + service[i])
    reachable = [i for i in sites if i in forward and i in backward
                 and forward[i] + backward[i] <= limit]
    arcs = [(0, i) for i in reachable if i in suppliers
            and travel[0][i] + service[i] + backward[i] <= limit]
    for i in reachable:
        check()
        arcs.extend((i, j) for j in reachable if i != j
                    and forward[i] + travel[i][j] + service[j] + backward[j] <= limit)
    arcs.extend((i, 0) for i in reachable if i in agencies
                and forward[i] + travel[i][0] <= limit)
    return reachable, arcs, travel, service, limit


def _build_lp(instance, vehicles, check, np, highspy):
    supply = [_exact(value) for value in instance["supply"]]
    population = [_exact(value) for value in instance["population"]]
    suppliers = {i for i in range(1, len(supply)) if supply[i] > 0}
    agencies = {i for i in range(1, len(population)) if population[i] > 0}
    sites, arcs, travel, service, limit = _arc_set(instance, suppliers, agencies, check)
    capacity = float(_exact(instance["capacity"]))
    sites_per_vehicle = len(sites)
    big = sites_per_vehicle + 1
    fleet = range(vehicles)

    cost, upper = [], []

    def col(hi, objective=0.0):
        index = len(cost)
        upper.append(hi)
        cost.append(objective)
        return index

    x = [{arc: col(1.0) for arc in arcs} for _ in fleet]
    y = [{i: col(1.0) for i in sites} for _ in fleet]
    binary = len(cost)
    load = [{(i, j): col(capacity) for i, j in arcs if i} for _ in fleet]
    order = [{i: col(sites_per_vehicle) for i in sites} for _ in fleet]
    pick = [{i: col(min(capacity, float(supply[i]))) for i in sites if i in suppliers}
            for _ in fleet]
    drop = [{i: col(capacity) for i in sites if i in agencies} for _ in fleet]
    quantity = {i: col(capacity, -1.0) for i in sorted(agencies)}
    check()

    outgoing = {i: [] for i in [0, *sites]}
    incoming = {i: [] for i in [0, *sites]}
    for i, j in arcs:
        outgoing[i].append((i, j))
        incoming[j].append((i, j))

    indices, coefficients, offsets = [], [], [0]
    row_lower, row_upper = [], []

    def row(terms, lo=-inf, hi=inf):
        for index, coefficient in terms:
            if coefficient:
                indices.append(index)
                coefficients.append(coefficient)
        offsets.append(len(indices))
        row_lower.append(lo)
        row_upper.append(hi)

    work_time = {arc: float(travel[arc[0]][arc[1]]) for arc in arcs}
    stop_time = {i: float(service[i]) for i in sites}
    for k in fleet:
        check()
        row([(x[k][arc], 1.0) for arc in outgoing[0]], hi=1.0)
        row([(x[k][arc], work_time[arc]) for arc in arcs]
            + [(y[k][i], stop_time[i]) for i in sites], hi=float(limit))
        for i in sites:
            check()
            departs = [(x[k][arc], -1.0) for arc in outgoing[i]]
            arrives = [(x[k][arc], -1.0) for arc in incoming[i]]
            row([(y[k][i], 1.0), *departs], 0.0, 0.0)
            row([(y[k][i], 1.0), *arrives], 0.0, 0.0)
            # Load after i goes out, including on a possible return to depot.
            # Load before i comes in, with depot departure fixed at zero.
            balance = [(load[k][arc], 1.0) for arc in outgoing[i]]
            balance += [(load[k][arc], -1.0) for arc in incoming[i] if arc[0]]
            if i in pick[k]:
                balance.append((pick[k][i], -1.0))
                row([(pick[k][i], 1.0), (y[k][i], -upper[pick[k][i]])], hi=0.0)
            if i in drop[k]:
                balance.append((drop[k][i], 1.0))
                row([(drop[k][i], 1.0), (y[k][i], -capacity)], hi=0.0)
            row(balance, 0.0, 0.0)
            row([(order[k][i], 1.0), (y[k][i], -sites_per_vehicle)], hi=0.0)
        for arc, index in load[k].items():
            row([(index, 1.0), (x[k][arc], -capacity)], hi=0.0)
        for i, j in arcs:
            if i and j:
                row([(order[k][i], 1.0), (order[k][j], -1.0), (x[k][i, j], big)],
                    hi=big - 1)

    for i in sites:
        row([(y[k][i], 1.0) for k in fleet], hi=1.0)
    for i, index in quantity.items():
        row([(index, 1.0)] + [(drop[k][i], -1.0) for k in fleet if i in drop[k]],
            0.0, 0.0)
    total_people = sum(population[i] for i in agencies)
    share = {i: float(population[i] / total_people) for i in quantity}
    ordered_agencies = list(quantity)
    for pos, i in enumerate(ordered_agencies):
        check()
        for j in ordered_agencies[pos + 1:]:
            gap = col(inf, 1.0)
            row([(quantity[i], -share[j]), (quantity[j], share[i]), (gap, -1.0)],
                hi=0.0)
            row([(quantity[i], share[j]), (quantity[j], -share[i]), (gap, -1.0)],
                hi=0.0)

    check()
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = len(cost), len(row_lower)
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = cost, np.zeros(len(cost)), upper
    lp.row_lower_, lp.row_upper_ = row_lower, row_upper
    lp.integrality_ = ([highspy.HighsVarType.kInteger] * binary
                       + [highspy.HighsVarType.kContinuous] * (len(cost) - binary))
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = (offsets, indices,
                                                                    coefficients)
    check()
    return lp, arcs


def _decode_routes(values, vehicles, arcs, np):
    if len(values) < vehicles * len(arcs):
        return None
    routes = []
    for k in range(vehicles):
        start = k * len(arcs)
        selected = np.flatnonzero(values[start:start + len(arcs)] > 0.5)
        successor = {}
        for index in selected:
            i, j = arcs[index]
            successor[i] = j
        if not selected.size:
            continue
        if len(successor) != selected.size or 0 not in successor:
            return None
        route, seen = [], set()
        node = successor[0]
        while node != 0:
            if node in seen or node not in successor:
                return None
            route.append(node)
            seen.add(node)
            node = successor[node]
        if len(route) + 1 != selected.size:
            return None
        routes.append(tuple(route))
    return routes


def solve(instance, vehicles, seed, deadline, publish):
    """Run native HiGHS. The publisher owns exact projection and checkpointing."""
    started = monotonic()
    result = dict(status="deadline_before_solve", seed=int(seed), threads=1,
                  objective=None, dual_bound=None, projected_objective=None,
                  mip_gap=None, nodes=None, incumbent_callbacks=0, decoded=0)
    projected = None
    published_routes = set()
    publish_numeric = getattr(publish, "numeric", None)

    def check():
        if monotonic() >= deadline:
            raise _Deadline

    try:
        check()
        import highspy
        import numpy as np

        check()
        lp, arcs = _build_lp(instance, vehicles, check, np, highspy)
        highs = highspy.Highs()
        settings = {
            "log_to_console": False,
            "threads": 1,
            "random_seed": int(seed),
            "mip_rel_gap": 0.0,
            "mip_abs_gap": 0.0,
            "mip_feasibility_tolerance": 1e-8,
            "primal_feasibility_tolerance": 1e-8,
            "dual_feasibility_tolerance": 1e-8,
            "presolve": "on",
        }
        for name, value in settings.items():
            if highs.setOptionValue(name, value) != highspy.HighsStatus.kOk:
                raise RuntimeError(f"HiGHS rejected {name}={value}")
        check()
        if highs.passModel(lp) == highspy.HighsStatus.kError:
            raise RuntimeError("HiGHS model copy failed")
        check()
    except _Deadline:
        result["elapsed"] = monotonic() - started
        return result

    result.update(highs_version=highs.version(), columns=lp.num_col_, rows=lp.num_row_,
                  arcs_per_vehicle=len(arcs))

    def bound_from_solver(dual, source, nodes, solver_time):
        upper = -float(dual)
        if monotonic() >= deadline or not isfinite(upper):
            return
        previous = result["dual_bound"]
        if previous is None or upper < previous:
            if publish_numeric is not None:
                publish_numeric(upper, source, nodes=int(nodes), solver_time=float(solver_time))
            result["dual_bound"] = upper

    def consider_solution(values, source):
        nonlocal projected
        if monotonic() >= deadline:
            return
        routes = _decode_routes(np.asarray(values), vehicles, arcs, np)
        if routes is None:
            return
        result["decoded"] += 1
        key = tuple(sorted(routes))
        if key not in published_routes and monotonic() < deadline:
            published_routes.add(key)
            value = publish(routes, source)
            if value is not None and (projected is None or value > projected):
                projected = value

    def live_bound(event):
        data = event.data_out
        bound_from_solver(data.mip_dual_bound, "site_milp_dual_live",
                          data.mip_node_count, data.running_time)

    def incumbent(event):
        if monotonic() >= deadline:
            return
        result["incumbent_callbacks"] += 1
        primal = -float(event.data_out.mip_primal_bound)
        if isfinite(primal):
            result["objective"] = primal
        live_bound(event)
        consider_solution(event.data_out.mip_solution, "site_milp_incumbent")

    def interrupt(event):
        if monotonic() >= deadline:
            event.interrupt()

    highs.cbMipImprovingSolution.subscribe(incumbent)
    highs.cbMipLogging.subscribe(live_bound)
    highs.cbMipInterrupt.subscribe(interrupt)

    solving = monotonic()
    result["build_seconds"] = solving - started
    if solving >= deadline:
        result["elapsed"] = solving - started
        return result
    if highs.setOptionValue("time_limit", deadline - solving) != highspy.HighsStatus.kOk:
        raise RuntimeError("HiGHS rejected the remaining time limit")
    if highs.run() == highspy.HighsStatus.kError:
        raise RuntimeError("HiGHS solve failed")

    finished = monotonic()
    result.update(status=str(highs.getModelStatus()), solver_seconds=finished - solving)
    info = highs.getInfo()
    result["nodes"] = int(info.mip_node_count)
    if finished < deadline:
        bound_from_solver(info.mip_dual_bound, "site_milp_dual_final",
                          info.mip_node_count, highs.getRunTime())
        if isfinite(info.mip_gap):
            result["mip_gap"] = float(info.mip_gap)
        if info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible:
            result["objective"] = -float(info.objective_function_value)
            if monotonic() < deadline:
                consider_solution(highs.getSolution().col_value, "site_milp_final")

    result.update(projected_objective=str(projected) if projected is not None else None,
                  distinct_fleets=len(published_routes),
                  elapsed=monotonic() - started)
    return result
