"""Exact fixed-route H-PDSP allocation by weighted polymatroid decomposition."""

from fractions import Fraction


def _number(value):
    if isinstance(value, (int, Fraction)):
        return value
    return Fraction(str(value))


def _rank(groups, capacity, mask):
    """Maximum total delivered to the selected delivery runs."""
    load = delivered = 0
    for bit, supply, _ in groups:
        load = min(capacity, load + supply)
        if bit & mask:
            delivered += load
            load = 0
    return delivered


def _chord_cut(groups, capacity, fixed, allowed, rate):
    """Minimize rank(A) + rate * pop(allowed - A), fixed <= A <= allowed.

    Costs are multiplied by the rate denominator. Each group has two
    vertices in the cut, one for its pickup run and one for its delivery run.
    Keep one minimum-cost partial cut per side of the latest vertex.
    """
    scale, price = rate.denominator, rate.numerator
    carry = capacity * scale
    # Putting every route vertex outside is always feasible. A larger cost
    # can represent a forbidden state using only exact arithmetic.
    forbidden = 1 + sum(supply for _, supply, _ in groups) * scale
    out_cost = in_cost = 0  # outside / inside the network source side
    out_mask = in_mask = 0
    for bit, supply, people in groups:
        # A pickup pays its supply arc outside the source side. A delivery
        # pays its terminal arc inside. Either can cut the carry arc.
        for supply_cost, sink_bit in ((supply * scale, 0), (0, bit)):
            if in_cost + carry < out_cost:
                next_out_cost, next_out_mask = in_cost + carry, in_mask
            else:
                next_out_cost, next_out_mask = out_cost, out_mask
            if out_cost < in_cost:
                in_cost, in_mask = out_cost, out_mask
            if sink_bit & fixed:
                in_cost = forbidden
            elif sink_bit & allowed:
                in_cost += price * people
            out_cost = next_out_cost + supply_cost
            out_mask = next_out_mask | (sink_bit & allowed)

    return (out_cost, out_mask) if out_cost <= in_cost else (in_cost, in_mask)


def _principal_levels(groups, capacity):
    """Yield (run mask, per-person quantity, population) in increasing rate."""
    all_runs = (1 << len(groups)) - 1
    total_people = sum(people for _, _, people in groups)
    total_flow = sum(supply for _, supply, _ in groups)
    pending = [(0, all_runs, 0, total_flow, 0, total_people)]
    while pending:
        left, right, rank_l, rank_r, pop_l, pop_r = pending.pop()
        block = right ^ left
        rate = Fraction(rank_r - rank_l, pop_r - pop_l)
        if block.bit_count() > 1:
            cost, middle = _chord_cut(groups, capacity, left, right, rate)
            if cost < rank_r * rate.denominator:
                rank_m = _rank(groups, capacity, middle)
                pop_m = sum(people for bit, _, people in groups if bit & middle)
                pending.append((middle, right, rank_m, rank_r, pop_m, pop_r))
                pending.append((left, middle, rank_l, rank_m, pop_l, pop_m))
                continue
        yield block, rate, pop_r - pop_l


def allocate(instance, route):
    """Return (exact optimal score, nonnegative amounts in route order).

    Nonempty routes start at a supplier, end at an agency and omit the depot.
    Travel feasibility belongs to the caller. Empty routes score zero.
    """
    route = tuple(route)
    if not route:
        return Fraction(0), []
    supplies, population = instance["supply"], instance["population"]
    if len(route) != len(set(route)) or any(n <= 0 or n >= len(supplies) for n in route):
        raise ValueError("route must contain distinct non-depot node indices")
    stops = [(_number(supplies[n]), _number(population[n])) for n in route]
    if stops[0][0] <= 0 or stops[-1][1] <= 0:
        raise ValueError("route must begin at a supplier and end at an agency")
    capacity = _number(instance["capacity"])
    total_population = sum(map(_number, population))

    # Aggregate each pickup run and its following consecutive delivery run.
    # The time-line of take/drop decisions within each run is restored below.
    groups = []
    group_at_stop = []
    pending_supply = 0
    for supply, people in stops:
        if supply > 0 and people == 0:
            pending_supply += supply
        elif people > 0 and supply == 0:
            if pending_supply:
                groups.append([1 << len(groups), min(capacity, pending_supply), 0])
                pending_supply = 0
            groups[-1][2] += people
        else:
            raise ValueError("each visited site must have exactly one positive role")
        group_at_stop.append(len(groups) - 1)

    rates = [None] * len(groups)
    remaining_people = sum(people for _, _, people in groups)
    score_numerator = 0
    for mask, rate, people in _principal_levels(groups, capacity):
        remaining_people -= people
        score_numerator += rate * people * (people + 2 * remaining_people)
        while mask:
            bit = mask & -mask
            rates[bit.bit_length() - 1] = rate
            mask ^= bit

    # Working backward from empty return, take the latest available supply.
    need = Fraction(0)
    amounts = [Fraction(0)] * len(route)
    for i in reversed(range(len(route))):
        supply, people = stops[i]
        if supply:
            amounts[i] = Fraction(min(supply, need))
            need -= amounts[i]
        else:
            amounts[i] = rates[group_at_stop[i]] * people
            need += amounts[i]
        if need > capacity:
            raise ArithmeticError("route allocation exceeds carrying capacity")
    if need:
        raise ArithmeticError("route allocation would require initial inventory")
    return Fraction(score_numerator, total_population), amounts
