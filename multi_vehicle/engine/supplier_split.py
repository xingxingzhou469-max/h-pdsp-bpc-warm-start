"""Proof arithmetic for a cover of fleets by active count and singleton supplier."""
from fractions import Fraction as F
from itertools import combinations

from common import Data, ceil_ratio
from verify import replay, replay_marked

METHOD = "active-supplier-cover-v2"
BIN_COUNT = 256


def cover(data, vehicles):
    sources = sorted((stock, node) for stock, node, is_source
                     in zip(data.caps, data.nodes, data.sources) if is_source)
    max_active = min(vehicles, len(sources))
    if not sources or max_active <= len(sources) // 2:
        return None
    low = max_active - 1
    count = len(sources) - (2 * max_active - len(sources)) + 1
    return low, tuple(sources[:count]), tuple(sources)


def without_supplier(instance, station, vehicles):
    keep = [i for i in range(len(instance["supply"])) if i != station]
    return dict(instance, vehicles=vehicles,
                **{key: [instance[key][i] for i in keep]
                   for key in ("population", "supply", "service")},
                travel=[[instance["travel"][i][j] for j in keep] for i in keep])


def population_caps(data, support):
    stocks = {node: q for node, q, source in zip(data.nodes, data.caps, data.sources) if source}
    caps = dict.fromkeys(stocks, data.total)
    if support is None:
        return caps
    # Exactly one remembered supplier contributes this known rounded credit.
    unit = F(support["supply_unit"]) if support["supply_unit"] else None
    alpha, beta = F(support["alpha"]), F(support["beta"])
    for i, node in enumerate(support["markers"]):
        maximum = support["class_values"][1 << i]
        rounded = stocks[node] if unit is None else ceil_ratio(stocks[node], unit) * unit
        caps[node] = None if maximum is None else min(data.total, (F(maximum) - alpha * rounded) / beta)
    return caps


def singleton_upper(data, stock, population, min_other_stock, other_stock, supports, other):
    if population is None or not data.total or not data.p_step:
        return F(0)
    p_max = min([data.total, *((s - a * min_other_stock) / b for a, b, s in supports)])
    if p_max < 0:
        return F(0)
    q_cap = min(other_stock, F(other["quantity_cap"]))
    solo_numerator = data.total * F(other["upper"])
    if not other["vehicles"]:
        q_cap, p_max = F(0), F(0)
    # Each rectangle rounds population upwards and uses the decreasing
    # quantity frontier at the left edge. All comparisons stay rational.
    population_step = data.p_step
    ticks = p_max // population_step
    stride = max(1, ceil_ratio(ticks + 1, BIN_COUNT))
    result = F(0)
    for start in range(0, ticks + 1, stride):
        lo = start * population_step
        hi = min(ticks, start + stride - 1) * population_step
        q = min([q_cap, *((s - b * lo) / a for a, b, s in supports)])
        if q < 0:
            break
        p = min(population, data.total - lo)
        upper = min(solo_numerator, q * hi) + stock * p + 2 * min(stock * hi, q * p)
        result = max(result, upper)
    return result / data.total


def residual_cases(data, stock, population, low, residual, other):
    supports = [(F(s["alpha"]), F(s["beta"]), F(s["linear_support"]))
                for s in other["supports"]]
    cases = []
    # The data order makes amounts within each subset increasing.
    for count in range(low, len(residual) + 1):
        for group in combinations(residual, count):
            stock_cap = sum((q for q, _ in group), F(0))
            floor = min(stock_cap, data.cap + sum((q for q, _ in group[:max(0, low - 1)]), F(0)))
            upper = singleton_upper(data, stock, population, floor, stock_cap, supports, other)
            cases.append(dict(sources=sorted(node for _, node in group), stock_cap=str(stock_cap),
                              potential_stock_floor=str(floor), upper=str(upper)))
    return cases


def build(data, vehicles, active, singleton, others):
    low, options, all_sources = cover(data, vehicles)
    caps = population_caps(data, singleton)
    leaves = [dict(active_at_most=low, proof=active, upper=active["upper"])]
    for (stock, node), other in zip(options, others, strict=True):
        residual = tuple(item for item in all_sources if item[1] != node)
        cases = residual_cases(data, stock, caps[node], low, residual, other)
        upper = max(F(item["upper"]) for item in cases)
        leaves.append(dict(singleton=node, other_active_at_most=low,
                           singleton_population=None if caps[node] is None else str(caps[node]),
                           residual_source_cases=cases, proof=other, upper=str(upper)))
    return dict(method=METHOD, vehicles=vehicles, singleton_support=singleton, leaves=leaves,
                upper=str(max(F(leaf["upper"]) for leaf in leaves)))


def replay_split(instance, vehicles, certificate):
    data = Data.from_instance(instance)
    case = cover(data, vehicles)
    if (case is None or certificate["method"] != METHOD or type(certificate["vehicles"]) is not int
            or type(instance.get("vehicles", vehicles)) is not int
            or certificate["vehicles"] != vehicles or instance.get("vehicles", vehicles) != vehicles):
        raise ValueError("active supplier proof has the wrong fleet or scope")
    low, options, all_sources = case
    leaves = certificate["leaves"]
    if not isinstance(leaves, list) or len(leaves) != 1 + len(options):
        raise ValueError("active supplier proof omits a leaf")
    if (type(leaves[0]["active_at_most"]) is not int
            or any(type(leaf["singleton"]) is not int or type(leaf["other_active_at_most"]) is not int
                   for leaf in leaves[1:])):
        raise ValueError("active supplier leaves require integer labels")
    support = certificate["singleton_support"]
    if support is not None:
        if (support["mode"] != "per_vehicle" or
                list(support["markers"]) != sorted(v for _, v in all_sources)
                or F(support["time_price"]) or any(map(F, support["visit_prices"]))):
            raise ValueError("singleton population requires all supplier markers and free prices")
        replay_marked(instance, vehicles, support)
    active = leaves[0]["proof"]
    replay(dict(instance, vehicles=low), low, active)
    others = []
    for (_, station), leaf in zip(options, leaves[1:], strict=True):
        proof = leaf["proof"]
        replay(without_supplier(instance, station, low), low, proof)
        others.append(proof)
    rebuilt = build(data, vehicles, active, support, others)
    if rebuilt != certificate:
        raise ValueError("supplier cover, conditional bound or endpoint differs")
    return F(rebuilt["upper"])
