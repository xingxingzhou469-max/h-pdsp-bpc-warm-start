"""Solver-free replay of route-price-v1 using only exact integer and Fraction arithmetic."""

import argparse
from fractions import Fraction
import json
from pathlib import Path

if __package__:
    from .model import product_bound, read_model
else:
    from model import product_bound, read_model


def _branch_coverage(graph, reward, ceiling, elementary):
    """Prove all branches are capped, without using a production search trace."""
    n = len(reward)
    horizon = graph.horizon
    layered = graph.layered
    layers = n if layered else 1
    if not n:
        return
    plane = n * (horizon + 1)
    if plane * layers > 7_500_000:
        raise ValueError("pricing proof exceeds the replay table limit")
    outgoing = [
        sorted((cost, j) for j, cost in enumerate(row) if cost <= horizon)
        for row in graph.arcs
    ]
    best = [None] * (layers * plane)
    second = [None] * (layers * plane)
    next_stop = [-1] * (layers * plane)

    for layer in range(layers):
        origin = layer * plane
        loop = not layered or layer
        child_origin = (layer - 1) * plane if layered else origin
        for time in range(horizon + 1):
            for i, arcs in enumerate(outgoing):
                hi, lo, arg = (0 if graph.finish[i] <= time else None), None, -1
                if loop:
                    for cost, j in arcs:
                        if cost > time:
                            break
                        child = child_origin + (time - cost) * n + j
                        v = second[child] if next_stop[child] == i else best[child]
                        if v is None:
                            continue
                        v += reward[j]
                        if hi is None or v > hi:
                            hi, lo, arg = v, hi, j
                        elif lo is None or v > lo:
                            lo = v
                cell = origin + time * n + i
                best[cell], second[cell], next_stop[cell] = hi, lo, arg

    bits = [1 << i for i in range(n)]
    covered = {}

    def visit(i, previous, time, left, mask, score):
        cell = left * plane + time * n + i
        tail = second[cell] if next_stop[cell] == previous and previous >= 0 else best[cell]
        if tail is None or score + tail <= ceiling:
            return
        if not elementary:
            raise ValueError("walk ceiling violated")
        key = (i, mask)
        if covered.get(key, -1) >= time:
            return
        covered[key] = time
        if score > ceiling and graph.finish[i] <= time:
            raise ValueError("elementary route ceiling violated")
        if layered and not left:
            return
        for cost, j in outgoing[i]:
            if cost > time:
                break
            if not mask & bits[j]:
                visit(j, i, time - cost, left - layered, mask | bits[j], score + reward[j])

    for i, time in enumerate(graph.start):
        if time <= horizon:
            visit(i, -1, horizon - time, layers - 1, bits[i], reward[i])


def replay(instance: dict, k: int, cert: dict) -> Fraction:
    if type(k) is not int or k < 0 or cert.get("vehicles") != k:
        raise ValueError("invalid or inconsistent vehicle count")
    if cert.get("method") != "route-price-v1" or type(cert.get("upper")) is not str:
        raise ValueError("invalid route pricing certificate")
    model = read_model(instance)
    proof_lines = cert["supports"]
    supports = []
    if proof_lines:
        graph = model.project()
        for line in proof_lines:
            beta = Fraction(line["beta"])
            prices = line["prices"]
            ceiling = line["route_bound"]
            mode = line["mode"]
            reverse = line["reverse"]
            if (beta <= 0 or len(prices) != model.n
                    or any(type(p) is not int or p < 0 for p in prices)
                    or type(ceiling) is not int or ceiling < 0
                    or mode not in ("walk", "elementary") or type(reverse) is not bool):
                raise ValueError("invalid support line")
            reward = tuple(w - p for w, p in zip(model.weights(beta), prices))
            _branch_coverage(graph.reversed() if reverse else graph, reward, ceiling,
                             mode == "elementary")
            rhs = model.reward_unit * (k * ceiling + sum(prices))
            supports.append((beta, rhs))
    upper = product_bound(model, tuple(model.boxes(k)), supports)[0] if k else Fraction()
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
