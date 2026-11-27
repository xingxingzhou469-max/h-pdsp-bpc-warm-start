"""Exact pure-standard-library replay of the marked fleet master."""

import argparse
from array import array
from fractions import Fraction
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from route.model import product_bound, read_model
from zone.check import layout
from maskfleet.core import CELL_LIMIT, MISSING, merge


def classes(graph, reward, partition, markers):
    n, horizon, layered = len(reward), graph.horizon, graph.layered
    count = 1 << len(markers)
    depth = n if layered else 1
    frame, offsets, inside, outside = layout(partition, graph, reward)
    clock = (horizon + 1) * count
    rows = depth * clock
    if rows * (frame + 2 * n) > CELL_LIMIT:
        raise ValueError("masked pricing exceeds the DP memory limit")
    if max(map(abs, reward)) * ((n if layered else horizon + 1) + 2) >= 2**59:
        raise ValueError("masked pricing integer range exceeded")
    tag = [0] * n
    for bit, i in enumerate(markers):
        tag[i] = 1 << bit
    dp = array("q", [MISSING]) * (rows * frame)
    second = array("q", [MISSING]) * (rows * n)
    forbidden = array("h", [-1]) * (rows * n)
    shift = layered * clock
    outside = [
        tuple((t, j, j - t * count * n, offsets[j] - t * count * frame, w)
              for t, j, w in row)
        for row in outside
    ]
    inside = [
        tuple(tuple((t, index - t * count * frame, w) for t, index, w in row)
              for row in states)
        for states in inside
    ]
    for step in range(depth):
        for time in range(horizon + 1):
            base = step * clock + time * count
            for mask in range(count):
                row = base + mask
                for i, states in enumerate(inside):
                    mark = tag[i]
                    if mark and not mask & mark:
                        continue
                    rest = mask ^ mark
                    start = row * frame + offsets[i]
                    terminal = 0 if not rest and graph.finish[i] <= time else MISSING
                    if layered and not step:
                        if terminal != MISSING:
                            for index in range(len(states)):
                                dp[start + index] = terminal
                            second[row * n + i] = terminal
                        continue
                    pred = base + rest - shift
                    a, b = pred * n, pred * frame
                    hi, lo, arg = terminal, MISSING, -1
                    for duration, j, alt, index, w in outside[i]:
                        if duration > time:
                            break
                        child = second[a + alt] if forbidden[a + alt] == i else dp[b + index]
                        if child == MISSING:
                            continue
                        value = child + w
                        if value > hi:
                            hi, lo, arg = value, hi, j
                        elif value > lo:
                            lo = value
                    for index, choices in enumerate(states):
                        best = MISSING
                        for duration, child, w in choices:
                            if duration > time:
                                break
                            v = dp[b + child]
                            if v != MISSING and v + w > best:
                                best = v + w
                        dp[start + index] = max(best, hi)
                        if not index:
                            second[row * n + i] = max(best, lo)
                            forbidden[row * n + i] = arg
    result = []
    for mask in range(count):
        top = MISSING if mask else 0
        for i, time in enumerate(graph.start):
            if time <= horizon:
                row = (depth - 1) * clock + (horizon - time) * count + mask
                val = dp[row * frame + offsets[i]]
                if val != MISSING:
                    top = max(top, val + reward[i])
        result.append(None if top == MISSING else top)
    return result


def replay(instance: dict, k: int, cert: dict) -> Fraction:
    if type(k) is not int or k < 0 or cert.get("vehicles") != k:
        raise ValueError("invalid or inconsistent vehicle count")
    if cert.get("method") != "mask-fleet-v1" or type(cert.get("upper")) is not str:
        raise ValueError("invalid marked fleet certificate")
    model = read_model(instance)
    lines = []
    if cert["supports"]:
        fwd = model.project()
        rev = fwd.reversed()
        for line in cert["supports"]:
            beta, price = Fraction(line["beta"]), line["prices"]
            part, marks, back = line["partition"], line["markers"], line["reverse"]
            if (beta <= 0 or len(price) != model.n or len(part) != model.n
                    or any(type(x) is not int or x < 0 for x in price)
                    or any(type(x) is not int or not 0 <= x < model.n for x in part)
                    or type(back) is not bool or not 1 <= len(marks) <= 4
                    or len(set(marks)) != len(marks)
                    or any(type(x) is not int or not 0 <= x < model.n or price[x] for x in marks)):
                raise ValueError("invalid marked fleet support")
            reward = tuple(a - b for a, b in zip(model.weights(beta), price))
            exact = classes(rev if back else fwd, reward, part, marks)
            if exact != list(line["class_values"]):
                raise ValueError("marked route classes do not replay")
            combined = merge(exact, min(k, model.n))[0][0]
            if type(line["fleet_bound"]) is not int or combined != line["fleet_bound"]:
                raise ValueError("marked fleet cap does not replay")
            lines.append((beta, model.reward_unit * (sum(price) + combined)))
    boxes = tuple(model.boxes(k)) if k else ()
    upper = product_bound(model, boxes, lines)[0]
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
    certificate = json.loads(args.certificate.read_text())
    print(replay(instance, args.vehicles, certificate))


if __name__ == "__main__":
    main()
