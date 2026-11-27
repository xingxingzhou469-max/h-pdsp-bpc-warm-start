"""Independent standard-library replay of a two-vehicle reduced-cost screen."""

import argparse
from array import array
from bisect import bisect_right
from fractions import Fraction
from hashlib import sha256
import json
from math import isqrt, lcm
from pathlib import Path
import sys
from time import monotonic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from route.model import product_bound, rational, read_model
from zone.check import CELL_LIMIT, MISSING, layout, replay as replay_zone

METHOD = "two-vehicle-zone-screen-v1"


def identity(instance, k):
    def canonical(value):
        return [canonical(x) for x in value] if isinstance(value, list) else str(rational(value))
    fields = ("capacity", "limit", "supply", "population", "service", "travel")
    return sha256(json.dumps([k, *(canonical(instance[key]) for key in fields)],
                             separators=(",", ":")).encode()).hexdigest()


def threshold_data(model, line, theta):
    beta = Fraction(line["beta"])
    prices, ceiling = tuple(line["prices"]), line["route_bound"]
    square = 4 * theta * beta * model.population / model.reward_unit**2
    total_weight = isqrt(square.numerator // square.denominator) + 1
    cutoff = total_weight - sum(prices) - ceiling
    weights = model.weights(beta)
    rewards = tuple(w - p for w, p in zip(weights, prices))
    return total_weight, cutoff, weights, rewards


def mask_digest(masks):
    result = sha256()
    for mask in sorted(masks):
        result.update(mask.to_bytes(16, "little"))
    return result.hexdigest()


def suffix_table(graph, reward, partition):
    """Independent bottom-up int64 suffix bounds, including local region memory."""
    n = len(reward)
    layered = graph.layered
    layers = n if layered else 1
    horizon = graph.horizon
    frame, offsets, inside, outside = layout(partition, graph, reward)
    rows = layers * (horizon + 1)
    if rows * (frame + 2 * n) > CELL_LIMIT:
        raise ValueError("screen replay table limit exceeded")
    table = array("q", [MISSING]) * (rows * frame)
    alternatives = array("q", [MISSING]) * (rows * n)
    blocked = array("h", [-1]) * (rows * n)
    layer_shift = (horizon + 1) if layered else 0
    outside = [tuple((t, j, -t * n + j, -t * frame + offsets[j], w)
                     for t, j, w in row) for row in outside]
    inside = [tuple(tuple((t, off - t * frame, w) for t, off, w in moves)
                    for moves in states) for states in inside]
    for slot in range(rows):
        step, time = divmod(slot, horizon + 1)
        base, short = slot * frame, slot * n
        previous, previous_short = base - layer_shift * frame, short - layer_shift * n
        for i, cases in enumerate(inside):
            index = base + offsets[i]
            best = 0 if graph.finish[i] <= time else MISSING
            second, skipped = MISSING, -1
            if layered and not step:
                for local in range(len(cases)):
                    table[index + local] = best
                continue
            for duration, j, child_short, child, weight in outside[i]:
                if duration > time:
                    break
                child_short += previous_short
                value = (alternatives[child_short] if blocked[child_short] == i
                         else table[previous + child])
                if value == MISSING:
                    continue
                value += weight
                if value > best:
                    second, best, skipped = best, value, j
                elif value > second:
                    second = value
            for local, moves in enumerate(cases):
                best_local = MISSING
                for duration, child, weight in moves:
                    if duration > time:
                        break
                    value = table[previous + child]
                    if value != MISSING:
                        best_local = max(best_local, value + weight)
                table[index + local] = max(best_local, best)
                if not local:
                    alternatives[short + i] = max(best_local, second)
                    blocked[short + i] = skipped
    return table, alternatives, blocked, frame, offsets


def enumerate_masks(graph, rewards, partition, cutoff):
    n, horizon = len(rewards), graph.horizon
    layered = int(graph.layered)
    table, alternatives, blocked, frame, offsets = suffix_table(graph, rewards, partition)
    groups = {}
    for i, zone in enumerate(partition):
        groups.setdefault(zone, []).append(i)
    regional = {label: {i: 1 << bit for bit, i in enumerate(group)}
                for label, group in groups.items()}
    local = [regional[zone] for zone in partition]
    own = [bits[i] for i, bits in enumerate(local)]
    move = [tuple((duration, j, 1 << j)
                  for j, duration in enumerate(row) if duration <= horizon)
            for row in graph.arcs]
    masks = {0} if cutoff <= 0 else set()
    dominant = [{} for _ in rewards]
    initial_layer = n - 1 if layered else 0

    def visit(i, previous, remaining, layer, visited, memory, score):
        row = layer * (horizon + 1) + remaining
        bit = own[i]
        compact = (memory & (bit - 1)) | ((memory >> 1) & ~(bit - 1))
        suffix = table[row * frame + offsets[i] + compact]
        if not compact and previous >= 0 and blocked[row * n + i] == previous:
            suffix = alternatives[row * n + i]
        if suffix == MISSING or score + suffix < cutoff:
            return
        known = dominant[i]
        if known.get(visited, -1) >= remaining:
            return
        known[visited] = remaining
        if graph.finish[i] <= remaining and score >= cutoff:
            masks.add(visited)
        if layer or not layered:
            for duration, j, marker in move[i]:
                if duration <= remaining and visited & marker == 0:
                    updated = memory | local[i][j] if partition[i] == partition[j] else own[j]
                    visit(j, i, remaining - duration, layer - layered,
                          visited | marker, updated, score + rewards[j])

    for i in range(n - 1, -1, -1):
        elapsed = graph.start[i]
        if elapsed <= horizon:
            visit(i, -1, horizon - elapsed, initial_layer, 1 << i, own[i], rewards[i])
    return masks


def excludes_fleet(model, masks, weights, total_weight, theta, deadline=None):
    """Test all pairs that are disjoint and pass the exact integer support floor."""
    stock_scale = lcm(model.capacity.denominator, *(value.denominator for value in model.stock))
    people_scale = lcm(*(value.denominator for value in model.people))
    stock = [int(value * stock_scale) for value in model.stock]
    people = [int(value * people_scale) for value in model.people]
    capacity = int(model.capacity * stock_scale)
    target = theta * model.population * stock_scale * people_scale
    numerator, denominator = target.numerator, target.denominator
    rows = []
    for mask in masks:
        bits = mask
        weight = q = p = agencies = 0
        while bits:
            bit = bits & -bits
            i = bit.bit_length() - 1
            weight += weights[i]
            q += stock[i]
            p += people[i]
            agencies += people[i] > 0
            bits ^= bit
        rows.append((-weight, min(q, capacity * agencies), p, mask))
        if deadline is not None and len(rows) % 2048 == 0 and monotonic() >= deadline:
            return None
    rows.sort()
    negative_weights = [row[0] for row in rows]
    count = len(rows)
    if not count:
        return True
    forbidden = [bytearray((count + 7) // 8) for _ in stock]
    for index, (_, _, _, mask) in enumerate(rows):
        byte, bit_value = index >> 3, 1 << (index & 7)
        while mask:
            bit = mask & -mask
            forbidden[bit.bit_length() - 1][byte] |= bit_value
            mask ^= bit
        if deadline is not None and index % 2048 == 0 and monotonic() >= deadline:
            return None
    forbidden = [int.from_bytes(items, "little") for items in forbidden]
    touched = 0
    for index, (negative, q, p, mask) in enumerate(rows):
        if deadline is not None and monotonic() >= deadline:
            return None
        end = bisect_right(negative_weights, -total_weight - negative)
        if end <= index:
            # With lower subsequent weights no later ordered pair can qualify.
            break
        conflicts = 0
        while mask:
            bit = mask & -mask
            conflicts |= forbidden[bit.bit_length() - 1]
            mask ^= bit
        candidates = (((1 << end) - 1) ^ ((1 << index) - 1)) & ~conflicts
        while candidates:
            bit = candidates & -candidates
            j = bit.bit_length() - 1
            _, other_q, other_p, _ = rows[j]
            group_bound = q*p + other_q*other_p + 2*min(q*other_p, other_q*p)
            if group_bound * denominator > numerator:
                return False
            candidates ^= bit
            touched += 1
            if deadline is not None and touched % 4096 == 0 and monotonic() >= deadline:
                return None
    return True


def replay(instance, k, cert):
    if (cert.get("method") != METHOD or cert.get("vehicles") != k
            or cert.get("instance_digest") != identity(instance, k)
            or type(cert.get("upper")) is not str):
        raise ValueError("invalid screen certificate identity")
    model = read_model(instance)
    parent = cert["parent"]
    inherited = (replay_zone(instance, k, parent) if parent is not None else
                 product_bound(model, tuple(model.boxes(k)) if k else (), [])[0])
    upper = Fraction(cert["upper"])
    probe = cert.get("screen")
    if probe is None:
        if upper != inherited:
            raise ValueError("wrong inherited upper")
        return upper
    if (k != 2 or parent is None or not 0 <= upper < inherited or not 0 < model.n <= 128
            or not model.quantity or not model.population):
        raise ValueError("invalid screened bounds")
    index = probe["line_index"]
    if type(index) is not int or not 0 <= index < len(parent["supports"]):
        raise ValueError("invalid screen support index")
    line = parent["supports"][index]
    integer, cutoff, weights, rewards = threshold_data(model, line, upper)
    graph = model.project()
    # Enumerate in the reverse orientation to the producer.
    if not line["reverse"]:
        graph = graph.reversed()
    masks = enumerate_masks(graph, rewards, tuple(line["partition"]), cutoff)
    if (len(masks) != probe["mask_count"] or mask_digest(masks) != probe["mask_sha256"]
            or not excludes_fleet(model, masks, weights, integer, upper)):
        raise ValueError("the complete mask screen did not replay")
    return upper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--vehicles", type=int, required=True)
    args = parser.parse_args()
    instance = json.loads(args.instance.read_text(), parse_float=Fraction)
    certificate = json.loads(args.certificate.read_text())
    print(replay(instance, args.vehicles, certificate))


if __name__ == "__main__":
    main()
