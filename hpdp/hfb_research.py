"""An incumbent search that crosses time and objective barriers before proof."""

from fractions import Fraction
from math import exp
from random import Random
from time import monotonic

from .allocation import _principal_levels, allocate
from .check import check
from .neighborhood import _time
from .portfolio import solve_portfolio


def _profile(r, route):
    groups = []
    for v in route:
        if r.supply[v]:
            supply = r.supply[v]
            if groups and groups[-1][1] == 0:
                supply = min(r.capacity, groups.pop()[0] + supply)
            groups.append((supply, 0))
        elif groups[-1][1]:
            supply, people = groups.pop()
            groups.append((supply, people + r.pop[v]))
        else:
            groups[-1] = (groups[-1][0], r.pop[v])
    return tuple(groups)


def _numerator(r, profile):
    groups = [(1 << i, supply, people) for i, (supply, people) in enumerate(profile)]
    remaining = sum(people for _, _, people in groups)
    value = 0
    for _, rate, people in _principal_levels(groups, r.capacity):
        remaining -= people
        value += rate * people * (people + 2 * remaining)
    return value


def _perturb(rng, route, active):
    """Propose a visit-set or order change using no historical solution data."""
    n = len(route)
    i, j = sorted(rng.sample(range(n), 2))
    move = rng.randrange(7)
    if move == 0:
        return route[:i] + route[i:j + 1][::-1] + route[j + 1:]
    if move == 1:
        return route[:i] + (route[j],) + route[i + 1:j] + (route[i],) + route[j + 1:]
    if move == 2:
        first = rng.randrange(n)
        last = min(n, first + rng.randrange(1, 4))
        subroute = route[first:last]
        rest = route[:first] + route[last:]
        gap = rng.randrange(len(rest) + 1)
        return rest[:gap] + subroute + rest[gap:]
    if move == 3 and n > 2:
        return route[:i] + route[i + 1:]
    unused = tuple(v for v in active if v not in route)
    if not unused:
        return route
    node = rng.choice(unused)
    if move == 4:
        return route[:i] + (node,) + route[i + 1:]
    if move == 5:
        return route[:j] + (node,) + route[j:]
    gap = rng.randrange(n + 1)
    return route[:gap] + (node,) + route[gap:]


def improve(r, deadline, seed, *, emit=None, metrics=None):
    """Return a strictly checked incumbent; exploratory routes can be over time."""
    if seed is None or len(r.p) < 2 or len(r.d) < 2:
        return seed
    start_route = tuple(seed["route"])
    best = seed
    best_value = check(r.instance, start_route, seed["amounts"], tol=0)["objective"]
    best_num = best_value * r.units
    if best_num >= r.prior or monotonic() >= deadline:
        return best
    rng = Random(121190)
    active = tuple(r.nodes)
    profiles = {}
    for restart in range(18):
        route = start_route if restart % 5 == 4 else tuple(best["route"])
        # Use integer route resources in the proposal scoring. Floating point
        # appears only in the selection of speculative search moves.
        profile = _profile(r, route)
        route_num = profiles.get(profile)
        if route_num is None:
            route_num = profiles[profile] = _numerator(r, profile)
        pressure = (1, 2, 4)[restart % 3]
        ranking = float(route_num / r.prior)
        temperature = 0.018 if restart % 2 else 0.035
        for step in range(3600):
            if step % 16 == 0 and monotonic() >= deadline:
                return best
            trial = _perturb(rng, route, active)
            if trial == route or not r.supply[trial[0]] or not r.pop[trial[-1]]:
                continue
            trial_time = _time(r, trial)
            if trial_time * 3 > r.limit * 4:
                continue
            trial_profile = _profile(r, trial)
            value = profiles.get(trial_profile)
            if value is None:
                value = profiles[trial_profile] = _numerator(r, trial_profile)
            if metrics is not None:
                metrics["proposals_scored"] += 1
            if value > best_num and trial_time <= r.limit:
                exact_value, amounts = allocate(r.instance, trial)
                verified = check(r.instance, trial, amounts, tol=0)["objective"]
                if verified != exact_value or value != verified * r.units:
                    raise ArithmeticError("Search profile disagrees with independently checked route")
                if monotonic() >= deadline:
                    return best
                best_num = value
                best = {"route": trial, "amounts": amounts}
                if metrics is not None:
                    metrics["improvements"] += 1
                if emit:
                    emit(best, verified)
                if best_num == r.prior:
                    return best
            excess = max(0, trial_time - r.limit) / max(1, r.limit)
            trial_rank = float(value / r.prior) - pressure * excess
            delta = trial_rank - ranking
            cooling = temperature * (1.0 - 0.97 * step / 3600)
            if delta >= 0 or rng.random() < exp(delta / cooling):
                route, ranking = trial, trial_rank
    return best


def solve(r, deadline, seed, *, emit=None):
    """Improve a common seed, then retain the certified portfolio residual bound."""
    t = monotonic()
    metrics = {"improvements": 0, "proposals_scored": 0}
    prior = check(r.instance, seed["route"], seed["amounts"], tol=0)["objective"] if seed else None
    upper = None if not r.units else Fraction(r.prior, r.units)

    def incumbent_row(new_seed, value):
        return dict(
            status="optimal" if value is not None and value == upper else "timeout",
            certificate="rational_hfb_frontier",
            lb=value, ub=upper,
            lb_exact=None if value is None else str(value),
            ub_exact=None if upper is None else str(upper),
            route=None if new_seed is None else new_seed["route"],
            amounts=None if new_seed is None else new_seed["amounts"],
            metrics={"phase": "incumbent", "hfb": {**metrics, "seconds": monotonic() - t}},
        )

    if emit and t < deadline:
        emit(incumbent_row(seed, prior))
    if prior is not None and prior == upper:
        return incumbent_row(seed, prior)
    heuristic_deadline = min(deadline, t + min(4, max(0, (deadline - t) / 2)))
    new_seed = improve(
        r, heuristic_deadline, seed,
        emit=(lambda best, value: emit(incumbent_row(best, value))) if emit else None,
        metrics=metrics,
    )
    seconds = monotonic() - t

    def proof_row(row):
        return dict(row, certificate="rational_hfb_frontier",
                    metrics={"phase": "proof", "hfb": {**metrics, "seconds": seconds},
                             "portfolio": row["metrics"]})

    result = solve_portfolio(r, deadline, new_seed, emit=(lambda row: emit(proof_row(row))) if emit else None)
    return proof_row(result)
