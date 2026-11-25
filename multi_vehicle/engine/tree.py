"""Finite visit-branch certificates and exact fleet convolution."""
from fractions import Fraction as F
from functools import lru_cache

from common import Data, ceil_ratio
from fair import evaluate
from verify import replay as replay_product, replay_marked

FAIR = "fair-capacity-v1"
TREE = "visit-branch-v1"


def inequalities(supports):
    return [tuple(F(s[k]) for k in ("alpha", "beta", "linear_support")) for s in supports]


def fair_certificate(data, vehicles, root):
    population_product, upper = evaluate(
        data, inequalities(root["supports"]), parent=F(root["upper"]))
    if not vehicles:
        population_product = upper = F(0)
    return dict(method=FAIR, vehicles=vehicles, root=root,
                population_product=str(population_product), upper=str(upper))


@lru_cache(maxsize=128)
def _convolve(classes, vehicles):
    """Max over exactly K mutually disjoint class signatures, with idle."""
    current = [F(0)] + [None] * (len(classes) - 1)
    for _ in range(vehicles):
        previous, current = current, [None] * len(classes)
        for union in range(len(classes)):
            sub = union
            while True:
                left, right = previous[union ^ sub], classes[sub]
                if left is not None and right is not None:
                    value = left + right
                    if current[union] is None or current[union] < value:
                        current[union] = value
                if not sub:
                    break
                sub = (sub - 1) & union
    return tuple(current)


def _parsed(support):
    return tuple(None if value is None else F(value) for value in support["class_values"])


def _conditional(data, vehicles, markers, support):
    values = _convolve(_parsed(support), vehicles)
    prices = dict(zip(data.nodes, map(F, support["visit_prices"])))
    pool_ticks = data.limit * vehicles // F(support["time_unit"])
    constant = sum((price for v, price in prices.items() if v not in markers), F(0))
    constant += pool_ticks * F(support["time_price"])
    return tuple(None if value is None else constant + value + sum(
        (prices[v] for i, v in enumerate(markers) if mask >> i & 1), F(0))
        for mask, value in enumerate(values))


def _partitions(mask, slots):
    """Unordered partitions of the required suppliers into at most K nonidle routes."""
    if not mask:
        yield ()
    elif slots == 1:
        yield (mask,)
    elif slots:
        pivot = mask & -mask
        rest = mask ^ pivot
        sub = rest
        while True:
            for later in _partitions(rest ^ sub, slots - 1):
                yield (sub | pivot, *later)
            if not sub:
                break
            sub = (sub - 1) & rest


def _group_numerator(groups, quantity, population):
    """Route-group water filling with fixed optimistic stocks and a population budget."""
    score = F(0)
    for stock, cap in groups:
        if population * stock <= quantity * cap:
            break
        population -= cap
        quantity -= stock
        score += cap * (stock + 2 * quantity)
    return score + population * quantity


def _class_boxes(caps, total, unit, alpha, beta, values):
    quantities, populations = [F(0)] * len(values), [None] * len(values)
    populations[0] = F(0)
    for mask, value in enumerate(values[1:], 1):
        amounts = [caps[i] for i in range(len(caps)) if mask >> i & 1]
        quantities[mask] = sum(amounts, F(0))
        if value is None:
            continue
        rounded = sum((ceil_ratio(x, unit) * unit for x in amounts), F(0)) if unit else quantities[mask]
        populations[mask] = min(total, max(F(0), (value - alpha * rounded) / beta))
    return quantities, populations


def _partition_limits(data, vehicles, markers, supports):
    sources = tuple(v for v, is_source in zip(data.nodes, data.sources) if is_source)
    if not data.cap or not data.total or markers != sources:
        return None
    for support in supports:
        if F(support["time_price"]) or any(map(F, support["visit_prices"])):
            continue
        caps = tuple(data.caps[v - 1] for v in markers)
        return _class_boxes(caps, data.total,
                            F(support["supply_unit"]) if support.get("supply_unit") else None,
                            F(support["alpha"]), F(support["beta"]), _parsed(support))
    return None


def _partition_lines(data, vehicles, markers, supports, required):
    result = []
    for s in supports:
        prices = tuple(map(F, s["visit_prices"]))
        total = sum((price for v, price in zip(data.nodes, prices)
                     if v not in markers or v in required), F(0))
        total += (data.limit * vehicles // F(s["time_unit"])) * F(s["time_price"])
        result.append((F(s["alpha"]), F(s["beta"]), total, _parsed(s)))
    return result


def _pack_upper(data, vehicles, mask, boxes, class_lines, root_lines, ceiling):
    """Bound each possible source partition before taking their maximum.

    With a short original Q grid, check every point. Otherwise use outward
    boxes: the left end limits population and the right end limits quantity.
    """
    quantities, populations = boxes
    stock = quantities[mask]
    total = data.total
    if not stock or not ceiling:
        return F(0)
    count = stock // data.q_step
    grid = count <= 80
    step = data.q_step if grid else stock / 40
    count = count if grid else 40
    best = F(0)
    for parts in _partitions(mask, vehicles):
        if not all(populations[s] is not None and quantities[s] for s in parts):
            continue
        groups = sorted(((quantities[s], populations[s]) for s in parts), key=lambda x: x[1]/x[0])
        p_box = min(total, sum((p for _, p in groups), F(0)))
        rough = _group_numerator(groups, stock, p_box) / total
        if rough <= best:
            continue
        if any(any(values[s] is None for s in parts) for _, _, _, values in class_lines):
            continue
        lines = [*root_lines, *((a, b, c + sum((values[s] for s in parts), F(0))
                                + (vehicles-len(parts))*values[0])
                               for a, b, c, values in class_lines)]
        # Convert each constraint once to p <= intercept - slope*q.
        shifted = [(a / b, s / b) for a, b, s in lines]
        local = F(0)
        for i in range(count, 0, -1):
            q = i * step
            q_lo = q if grid else q - step
            if q * p_box / total <= max(local, best):
                break
            p = min(p_box, min((c - a * q_lo for a, c in shifted), default=p_box))
            p = (p // data.p_step) * data.p_step
            if p <= 0 or q * p / total <= max(local, best):
                continue
            value = min(q*p, _group_numerator(groups, stock, p)) / total
            local = max(local, value)
            if min(ceiling, rough) <= local:
                break
        best = max(best, min(ceiling, rough, local))
        if best == ceiling:
            break
    return best


def tree_certificate(data, vehicles, root, markers, marked):
    """Every binary visit leaf is present in numeric order of its required mask."""
    markers = tuple(markers)
    original_lines = inequalities(root["supports"])
    root_fair = F(fair_certificate(data, vehicles, root)["upper"])
    conditional = [(F(s["alpha"]), F(s["beta"]), _conditional(data, vehicles, markers, s))
                   for s in marked]
    partition = _partition_limits(data, vehicles, markers, marked)
    leaves = []
    for mask in range(1 << len(markers)):
        required = tuple(v for i, v in enumerate(markers) if mask >> i & 1)
        forbidden = tuple(v for v in markers if v not in required)
        infeasible = any(s[mask] is None or s[mask] < 0 for _, _, s in conditional)
        group = None
        if infeasible or not vehicles:
            product = fair = F(0)
        else:
            lines = [*original_lines, *((a, b, s[mask]) for a, b, s in conditional)]
            product, fair = evaluate(data, lines, required=required, forbidden=forbidden,
                                     parent=root_fair)
            if partition is not None:
                part_lines = _partition_lines(data, vehicles, markers, marked, required)
                group = _pack_upper(data, vehicles, mask, partition, part_lines, original_lines, fair)
        upper = fair if group is None else min(fair, group)
        leaves.append(dict(mask=mask, required=list(required), forbidden=list(forbidden),
                           infeasible=infeasible, population_product=str(product), fair=str(fair),
                           supplier_partition=None if group is None else str(group), upper=str(upper)))
    return dict(method=TREE, vehicles=vehicles, root=root, markers=list(markers),
                marked=list(marked), root_fair=str(root_fair), leaves=leaves,
                upper=str(max(F(leaf["upper"]) for leaf in leaves)))


def replay_new(instance, vehicles, cert):
    """Recompute every leaf and validate all pricing supports in pure Python."""
    root = cert["root"]
    replay_product(instance, vehicles, root)
    data = Data.from_instance(instance)
    if type(cert["vehicles"]) is not int or cert["vehicles"] != vehicles:
        raise ValueError("certificate fleet size differs")
    if cert["method"] == FAIR:
        observed = fair_certificate(data, vehicles, root)
    elif cert["method"] == TREE:
        markers = cert["markers"]
        if (not isinstance(markers, list) or not 1 <= len(markers) <= 8
                or any(type(v) is not int or v not in data.nodes for v in markers)
                or markers != sorted(set(markers)) or not isinstance(cert["marked"], list)):
            raise ValueError("invalid branch stations")
        for support in cert["marked"]:
            if list(support["markers"]) != markers or support["mode"] != "per_vehicle":
                raise ValueError("marked support has a different visit partition")
            replay_marked(instance, vehicles, support)
        observed = tree_certificate(data, vehicles, root, markers, cert["marked"])
    else:
        raise ValueError("unknown candidate proof method")
    if observed != cert:
        raise ValueError("fairness/tree endpoint, leaf coverage or transcript does not reproduce")
    return F(observed["upper"])
