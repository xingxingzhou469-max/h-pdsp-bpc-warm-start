"""Pure-standard-library replay for the zone route upper bound."""

import argparse
from array import array
from fractions import Fraction
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from route.model import product_bound, read_model

MISSING = -(1 << 61)
CELL_LIMIT = 18_500_000


def layout(partition, graph, rewards):
    cells = {}
    for i, name in enumerate(partition):
        cells.setdefault(name, []).append(i)
    if max(map(len, cells.values())) > 12:
        raise ValueError("zone exceeds 12 stations")
    local = [{j: 1 << b for b, j in enumerate(cells[name])} for name in partition]
    own = [local[i][i] for i in range(len(partition))]
    offsets, cursor = [], 0
    for row in local:
        offsets.append(cursor)
        cursor += 1 << (len(row) - 1)

    in_zone, outside = [], []
    for i, label in enumerate(partition):
        outs = sorted((t, j, rewards[j]) for j, t in enumerate(graph.arcs[i])
                      if partition[j] != label and t <= graph.horizon)
        outside.append(outs)
        options = []
        low = own[i] - 1
        for mask in range(1 << (len(local[i]) - 1)):
            visited = (mask & low) | ((mask & ~low) << 1) | own[i]
            moves = []
            for j, bit in local[i].items():
                time = graph.arcs[i][j]
                if visited & bit or time > graph.horizon:
                    continue
                enlarged = visited | bit
                tail_low = own[j] - 1
                child = (enlarged & tail_low) | ((enlarged >> 1) & ~tail_low)
                moves.append((time, offsets[j] + child, rewards[j]))
            options.append(sorted(moves))
        in_zone.append(options)
    return cursor, offsets, in_zone, outside


def priced(graph, reward, partition):
    n = len(reward)
    if not n:
        return 0
    horizon, layered = graph.horizon, graph.layered
    layers = n if layered else 1
    frame, offsets, inside, outside = layout(partition, graph, reward)
    rows = layers * (horizon + 1)
    if rows * (frame + 2 * n) > CELL_LIMIT:
        raise ValueError("zone replay table limit exceeded")
    if max(map(abs, reward)) * ((n if layered else horizon + 1) + 2) >= 2**59:
        raise ValueError("zone pricing integer range exceeded")
    product = array("q", [MISSING]) * (rows * frame)
    without = array("q", [MISSING]) * (rows * n)
    blocked = array("h", [-1]) * (rows * n)
    previous_layer = (horizon + 1) * layered

    outside = [tuple((t, -t * n + j, -t * frame + offsets[j], w) for t, j, w in row)
               for row in outside]
    inside = [tuple(tuple((t, off - t * frame, w) for t, off, w in moves) for moves in states)
              for states in inside]
    for step in range(layers):
        for time in range(horizon + 1):
            row = step * (horizon + 1) + time
            source, destination = row * frame, row * n
            tail, tail_single = source - previous_layer * frame, destination - previous_layer * n
            for i, moves_by_state in enumerate(inside):
                turn = offsets[i] + source
                best = 0 if graph.finish[i] <= time else MISSING
                runner, forbid = MISSING, -1
                if layered and not step:
                    for index in range(len(moves_by_state)):
                        product[turn + index] = best
                    without[destination + i] = best
                    continue
                for duration, label, child, w in outside[i]:
                    if duration > time:
                        break
                    value = (without[tail_single + label] if blocked[tail_single + label] == i
                             else product[tail + child])
                    if value == MISSING:
                        continue
                    value += w
                    if value > best:
                        runner, best, forbid = best, value, label + duration * n
                    elif value > runner:
                        runner = value
                for mask, moves in enumerate(moves_by_state):
                    value = MISSING
                    for duration, child, w in moves:
                        if duration > time:
                            break
                        sub = product[tail + child]
                        if sub != MISSING and sub + w > value:
                            value = sub + w
                    product[turn + mask] = max(value, best)
                    if not mask:
                        without[destination + i] = max(value, runner)
                        blocked[destination + i] = forbid
    bound = 0
    for i, travel in enumerate(graph.start):
        if travel <= horizon:
            value = product[(rows - 1 - travel) * frame + offsets[i]]
            if value != MISSING:
                bound = max(bound, reward[i] + value)
    return bound


def replay(instance: dict, k: int, cert: dict) -> Fraction:
    if type(k) is not int or k < 0 or cert.get("vehicles") != k:
        raise ValueError("invalid or inconsistent vehicle count")
    if cert.get("method") != "zone-price-v1" or type(cert.get("upper")) is not str:
        raise ValueError("invalid zone pricing certificate")
    model = read_model(instance)
    supports = []
    if cert["supports"]:
        forward = model.project()
        back = forward.reversed()
        for line in cert["supports"]:
            price, beta = line["prices"], Fraction(line["beta"])
            part, rev, cap = line["partition"], line["reverse"], line["route_bound"]
            if (beta <= 0 or len(price) != model.n or len(part) != model.n
                    or any(type(x) is not int or x < 0 for x in price)
                    or any(type(x) is not int or not 0 <= x < model.n for x in part)
                    or type(rev) is not bool or type(cap) is not int or cap < 0):
                raise ValueError("invalid zone support")
            reward = tuple(w - p for w, p in zip(model.weights(beta), price))
            if priced(back if rev else forward, reward, part) != cap:
                raise ValueError("zone pricing value does not replay")
            supports.append((beta, model.reward_unit * (k * cap + sum(price))))
    boxes = tuple(model.boxes(k)) if k else ()
    upper = product_bound(model, boxes, supports)[0]
    if upper != Fraction(cert["upper"]):
        raise ValueError("upper bound does not match exact envelope")
    return upper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--vehicles", type=int, required=True)
    args = parser.parse_args()
    instance = json.loads(args.instance.read_text(), parse_float=Fraction)
    cert = json.loads(args.certificate.read_text())
    print(replay(instance, args.vehicles, cert))


if __name__ == "__main__":
    main()
