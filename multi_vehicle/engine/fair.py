"""Exact aggregate fairness envelope. This module needs only the standard library."""
from fractions import Fraction as F
from math import lcm

from polygon import maximize

POP_GRID_CAP = 30_000


def capped_numerator(quantity, population, capped, capacity):
    """Maximal Gini numerator for capped agencies and one uncapped group.

    Capped agency populations are sorted in descending order.
    """
    numerator = F(0)
    for people in capped:
        if quantity * people <= capacity * population:
            break
        population -= people
        quantity -= capacity
        numerator += capacity * (people + 2 * population)
    return numerator + quantity * population


def _subset_sums(ticks):
    reachable = 1
    for step in ticks:
        reachable |= reachable << step
    return reachable


def node_caps(data, forbidden=()):
    population = [(v, p) for v, p in zip(data.nodes, data.people) if p and v not in forbidden]
    stock = sum((c for v, c in zip(data.nodes, data.caps) if v not in forbidden), F(0))
    return min(stock, data.cap * len(population)), population


def evaluate(data, lines, *, required=(), forbidden=(), parent=None):
    """Maximize at the largest allowable Q* on each reachable population grid point.

    The optional parent is an independently certified objective bound. It does
    not constrain the product or remove any population grid points.
    """
    qcap, population = node_caps(data, forbidden)
    pcap = sum((p for _, p in population), F(0))
    if not qcap or not pcap or parent == 0 or any(s < 0 for _, _, s in lines):
        return F(0), F(0)
    if data.total // data.p_step > POP_GRID_CAP:
        product = maximize(qcap, pcap, lines) * pcap / data.total
        return product, min(product, parent) if parent is not None else product

    fixed = tuple(sorted((p for v, p in population if v in required), reverse=True))
    optional = sorted(((int(p / data.p_step), p) for v, p in population if v not in required), reverse=True)
    steps = tuple(x for x, _ in optional)
    fixed_ticks = int(sum(fixed, F(0)) / data.p_step)
    reachable = _subset_sums(steps)
    # Equal-population optional agencies have the same exclusion test.
    without = {step: _subset_sums(steps[:i] + steps[i + 1:])
               for i, step in enumerate(steps) if i == 0 or step != steps[i - 1]}

    integer_lines = []
    for a, b, support in lines:
        x, y = a * data.q_step, b * data.p_step
        scale = lcm(x.denominator, y.denominator, support.denominator)
        integer_lines.append((int(x * scale), int(y * scale), int(support * scale)))
    q_ticks_cap = qcap // data.q_step
    best_product = 0
    best_capped = F(0)  # In units q_step*p_step/P0.
    unit = data.q_step * data.p_step / data.total
    want = None if parent is None else parent / unit
    for additional in range(sum(steps), -1, -1):
        if not reachable >> additional & 1:
            continue
        p_tick = fixed_ticks + additional
        if not p_tick:
            continue
        q_tick = min(q_ticks_cap, min(((s - y * p_tick) // x for x, y, s in integer_lines),
                                     default=q_ticks_cap))
        if q_tick <= 0:
            continue
        product = q_tick * p_tick
        best_product = max(best_product, product)
        if product <= best_capped or (want is not None and best_capped >= want):
            continue
        necessary = [people for step, people in optional if not without[step] >> additional & 1]
        mandatory = sorted((*fixed, *necessary), reverse=True)
        if not mandatory or q_tick * data.q_step * mandatory[0] <= data.cap * p_tick * data.p_step:
            best_capped = product
        else:
            capped = capped_numerator(q_tick * data.q_step, p_tick * data.p_step, mandatory, data.cap)
            best_capped = max(best_capped, capped / (data.q_step * data.p_step))
    ceiling = best_capped * unit
    return best_product * unit, min(ceiling, parent) if parent is not None else ceiling
