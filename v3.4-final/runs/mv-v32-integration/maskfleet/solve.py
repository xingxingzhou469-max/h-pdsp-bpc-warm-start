"""Reoptimize station prices against exact marker-disjoint fleet pricing."""

from time import monotonic, monotonic_ns

START = monotonic()

import argparse
from collections import Counter
import ctypes as ct
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
import json
from math import isfinite
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from route.model import product_bound, read_model
from route.solve import atomically
from zone.solve import cluster, knapsack_price, legal_zone, repeated_segments
from maskfleet.core import CELL_LIMIT, MISSING, merge


class Native:
    def __init__(self, graph):
        lib = ct.CDLL(str(Path(__file__).with_name("mask_price.so")))
        self.fn = lib.mask_price
        i, v, ptr = ct.c_int, ct.c_int64, ct.POINTER
        self.fn.argtypes = [i, i, i, i, v, ptr(i), ptr(i), ptr(i), ptr(i), ptr(i),
                            ptr(v), ptr(v), ptr(i), ptr(i)]
        self.fn.restype = i
        n = self.n = len(graph.start)
        self.h, self.layered = graph.horizon, int(graph.layered)
        self.steps = n if self.layered else self.h + 1
        self.data = [
            ((i * (n * n))(*(cost for row in g.arcs for cost in row)),
             (i * n)(*g.start), (i * n)(*g.finish))
            for g in (graph, graph.reversed())
        ]
        self.paths = (i * (16 * self.steps))()
        self.lengths = (i * 16)()

    def price(self, reward, partition, marks, reverse, until):
        remaining = min(1 << 60, int((until - monotonic()) * 1e9))
        if remaining <= 0:
            return None
        if max(map(abs, reward), default=0) * (self.steps + 2) >= 2**59:
            raise OverflowError("master price is outside exact integer range")
        n = self.n
        k = 1 << len(marks)
        types = [0] * n
        for bit, marker in enumerate(marks):
            types[marker] = 1 << bit
        values = (ct.c_int64 * k)()
        status = self.fn(
            n, self.h, self.layered, len(marks), remaining, *self.data[reverse],
            (ct.c_int * n)(*partition), (ct.c_int * n)(*types), (ct.c_int64 * n)(*reward),
            values, self.lengths, self.paths,
        )
        if status in (2, 3):
            return None
        if status:
            raise RuntimeError(f"masked pricing failed with status {status}")
        routes = [
            tuple(self.paths[i * self.steps:i * self.steps + self.lengths[i]])
            for i in range(k)
        ]
        return [None if v == MISSING else v for v in values], (
            [path[::-1] for path in routes] if reverse else routes)


class Master:
    def __init__(self, highspy, np, model, weights, marks, k=1, per_route=False):
        self.highspy = highspy
        self.weights, self.model = weights, model
        excluded = set(marks)
        self.nodes = tuple(i for i in range(len(weights)) if i not in excluded)
        self.index = {i: j for j, i in enumerate(self.nodes)}
        self.last = len(self.nodes)
        self.solver = highspy.Highs()
        self.solver.setOptionValue("output_flag", False)
        self.solver.setOptionValue("threads", 1)
        self.solver.setOptionValue("solver", "simplex")
        m = self.last + 1
        self.solver.addCols(m, [1.] * self.last + [float(k if per_route else 1)],
                            np.zeros(m), np.full(m, highspy.kHighsInf), 0, [], [], [])
        self.rows, self.seen = [], set()

    def add(self, fleets):
        starts, columns, coeffs, lower = [], [], [], []
        for fleet in fleets:
            visits = Counter(i for path in fleet for i in path)
            pattern = tuple(sorted(visits.items()))
            if not pattern or pattern in self.seen:
                continue
            self.seen.add(pattern)
            self.rows.append(pattern)
            starts.append(len(columns))
            for i, times in pattern:
                if i in self.index:
                    columns.append(self.index[i])
                    coeffs.append(float(times))
            columns.append(self.last)
            coeffs.append(1.)
            lower.append(float(sum(self.weights[i] * times for i, times in pattern)))
        if lower:
            self.solver.addRows(len(lower), lower, [self.highspy.kHighsInf] * len(lower),
                                len(columns), starts, columns, coeffs)
        return len(lower)

    def optimize(self, until):
        duration = until - monotonic()
        if duration <= 0 or not self.rows:
            return None
        self.solver.setOptionValue("time_limit", duration)
        if self.solver.run() != self.highspy.HighsStatus.kOk:
            return None
        if self.solver.getModelStatus() != self.highspy.HighsModelStatus.kOptimal:
            return None
        solution = self.solver.getSolution()
        p = [0] * len(self.weights)
        for i, value in zip(self.nodes, solution.col_value):
            p[i] = max(0, round(value))
        fractional = [(pattern, mix) for pattern, mix in zip(self.rows, solution.row_dual)
                      if mix > 1e-8]
        q = sum(mix * count * float(self.model.stock[i])
                for pattern, mix in fractional for i, count in pattern)
        pop = sum(mix * count * float(self.model.people[i])
                  for pattern, mix in fractional for i, count in pattern)
        ratio = q / pop if q > 0 and pop > 0 else 0.
        return tuple(p), self.solver.getObjectiveValue(), fractional, ratio


def mark_candidates(weights, prices, paths, fractional, n):
    visits = Counter(i for path in paths[:500] for i in set(path))
    active = Counter()
    for pattern, mass in fractional:
        if mass < 1 - 1e-8:
            for i, _ in pattern:
                active[i] += mass
    scale = max(weights) or 1
    frequent = max(visits.values(), default=1)
    ordered = sorted(range(n), key=lambda i: (
        prices[i] / scale + weights[i] / (5 * scale) + visits[i] / (3 * frequent) + active[i] / 3, -i),
        reverse=True)
    if n < 3 or not fractional:
        return tuple(ordered[:4]), None
    test = set(ordered[:26])
    pairs, triples = Counter(), Counter()
    for row, mass in fractional:
        selected = sorted(i for i, _ in row if i in test)
        for key in combinations(selected, 2):
            pairs[key] += mass
        for key in combinations(selected, 3):
            triples[key] += mass
    subset = max(combinations(sorted(test), 3), key=lambda t: (
        pairs[t[:2]] + pairs[(t[0], t[2])] + pairs[t[1:]] - 2 * triples[t],
        sum(prices[i] for i in t)))
    score = pairs[subset[:2]] + pairs[(subset[0], subset[2])] + pairs[subset[1:]] - 2 * triples[subset]
    if score > 1.00001:
        return tuple([*(i for i in ordered if i in subset),
                      *(i for i in ordered if i not in subset)][:4]), score
    return tuple(ordered[:4]), None


def load_parent(certificate, columns, k, n):
    hints, paths, digest = {}, [], None
    if certificate:
        raw = certificate.read_bytes()
        digest = sha256(raw).hexdigest()
        cert = json.loads(raw)
        if cert.get("method") not in ("zone-price-v1", "route-price-v1") or cert.get("vehicles") != k:
            raise ValueError("parent must be a zone or route certificate for the same fleet")
        for line in cert["supports"]:
            slope, price = Fraction(line["beta"]), tuple(line["prices"])
            if slope <= 0 or len(price) != n or any(type(i) is not int or i < 0 for i in price):
                raise ValueError("malformed parent price hint")
            part = line.get("partition")
            if part and (len(part) != n or any(type(i) is not int or not 0 <= i < n for i in part)):
                raise ValueError("invalid parent partition")
            hints[slope] = price, tuple(part) if part else None
    if columns:
        for path in json.loads(columns.read_bytes()).get("routes", ()):
            if path and all(type(i) is int and 1 <= i <= n for i in path):
                paths.append(tuple(i - 1 for i in path))
    return hints, paths, digest


def graph_legal(path, graph):
    return (all(a != b for a, b in zip(path, path[1:]))
            and all(path[i] != path[i + 2] for i in range(len(path) - 2))
            and (not graph.layered or len(path) <= len(graph.start))
            and (graph.start[path[0]] + graph.finish[path[-1]]
                 + sum(graph.arcs[i][j] for i, j in zip(path, path[1:]))) <= graph.horizon)


def fleet_columns(paths, price, partition, marks, k):
    tag = {i: 1 << bit for bit, i in enumerate(marks)}
    count = 1 << len(marks)
    values, routes = [0] + [None] * (count - 1), [()] * count
    for path in paths:
        if not legal_zone(path, partition):
            continue
        mask = 0
        for i in path:
            bit = tag.get(i, 0)
            if mask & bit:
                break
            mask |= bit
        else:
            score = sum(price[i] for i in path)
            if values[mask] is None or score > values[mask]:
                values[mask], routes[mask] = score, path
    return [tuple(routes[mask] for mask in seq) for _, seq in merge(values, k)]


def solve(instance, k, out, deadline, warm, columns):
    model = read_model(instance)
    boxes = tuple(model.boxes(k)) if k else ()
    seed, routes, digest = load_parent(warm, columns, k, model.n)
    best, point, lines, proof = None, None, {}, {}

    def publish(meta):
        nonlocal best, point
        upper, point = product_bound(model, boxes, list(lines.items()))
        if (best is None or upper < best) and monotonic() < deadline:
            atomically(out / "best.json", {
                "method": "mask-fleet-v1", "upper": str(upper), "vehicles": k,
                "supports": list(proof.values()), "warm_parent_sha256": digest,
                "label_selection": meta, "emitted_monotonic_ns": monotonic_ns(),
            })
            best = upper

    publish(None)
    if not best or not model.n or model.n > 256 or monotonic() >= deadline:
        return
    graph = model.project()
    n = model.n
    rows = (graph.horizon + 1) * (n if graph.layered else 1)
    if rows * n * 6 > CELL_LIMIT:
        return
    routes = [path for path in dict.fromkeys(routes) if graph_legal(path, graph)]
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "HIGHS_NUM_THREADS"):
        os.environ[name] = "1"
    import highspy
    import numpy as np

    native = Native(graph)
    work = max(0., deadline - monotonic())
    stop = deadline - min(.05, work / 30)
    active_k = min(k, n)
    pool = dict.fromkeys(routes)
    target = (point[0] / point[1]).limit_denominator(8192) or model.quantity / model.population
    centers = {beta: price for beta, (price, _) in seed.items()}
    beta0 = target
    if seed:
        # A limiting parent support need not have beta closest to the box.
        parent = json.loads(warm.read_bytes())
        scaled = [(Fraction(line["beta"]), model.reward_unit * (
            sum(line["prices"]) + k * line["route_bound"])) for line in parent["supports"]]
        _, parent_point = product_bound(model, boxes, scaled)
        beta0 = min(scaled, key=lambda s: abs(s[1] - parent_point[0] - s[0] * parent_point[1]))[0]
    weights0 = model.weights(beta0)
    start_prices = centers.get(beta0, knapsack_price(model, weights0, k))
    fractional = []
    if pool:
        vanilla = Master(highspy, np, model, weights0, (), active_k, per_route=True)
        vanilla.add((path,) for path in pool)
        lp = vanilla.optimize(min(stop, monotonic() + min(2.0, .12 * work)))
        if lp:
            fractional = lp[2]
    labels, violation = mark_candidates(weights0, start_prices, routes, fractional, n)
    selection = {"subset_row_violation": violation, "ranked_markers": labels}
    pairs = sorted(
        (min(graph.arcs[i][j], graph.arcs[j][i]), graph.arcs[i][j] + graph.arcs[j][i], i, j)
        for i in range(n) for j in range(i + 1, n)
    )
    gradients, votes = {}, repeated_segments(routes, weights0)

    def choose_part(beta, marks, width):
        def fits(part):
            frame = sum(size * (1 << (size - 1)) for size in Counter(part).values())
            return rows * (1 << len(marks)) * (frame + 2 * n) <= CELL_LIMIT

        if beta in seed and seed[beta][1] and fits(seed[beta][1]):
            return seed[beta][1]
        for g in (width, 4, 2, 1):
            part = cluster(graph, g, votes, pairs)
            if fits(part):
                return part
        return None

    def phase(beta, marks, width, end, reverse_start=False):
        part = choose_part(beta, marks, width)
        if part is None:
            return
        weights = model.weights(beta)
        price = centers.get(beta)
        if price is None:
            near = min(centers, key=lambda b: abs(b - beta), default=None)
            price = (tuple(round(p * (beta / near if pop else 1))
                           for p, pop in zip(centers[near], model.people))
                     if near else knapsack_price(model, weights, k))
        anchor = price = tuple(p if i not in marks else 0 for i, p in enumerate(price))
        master = Master(highspy, np, model, weights, marks)
        residual = tuple(w - p for w, p in zip(weights, price))
        master.add(fleet_columns(pool, residual, part, marks, active_k))
        used = set()
        iteration = 0
        while monotonic() < end:
            reverse = bool(iteration % 2) ^ reverse_start
            if (price, reverse) in used:
                reverse = not reverse
            if (price, reverse) in used:
                break
            used.add((price, reverse))
            residual = tuple(w - p for w, p in zip(weights, price))
            oracle = native.price(residual, part, marks, reverse, end)
            if oracle is None:
                break
            values, paths = oracle
            states = merge(values, active_k)
            cap = states[0][0]
            total = sum(price) + cap
            rhs = total * model.reward_unit
            if beta not in lines or rhs < lines[beta]:
                lines[beta] = rhs
                anchor = centers[beta] = price
                proof[beta] = {
                    "beta": str(beta), "prices": price, "partition": part, "markers": marks,
                    "reverse": reverse, "class_values": values, "fleet_bound": cap,
                }
                publish(selection)
            new = master.add(tuple(paths[mask] for mask in sequence) for _, sequence in states)
            for path in paths:
                if path:
                    pool[path] = None
            votes.update(repeated_segments(paths, residual))
            suggestion = master.optimize(end)
            if suggestion is None:
                break
            proposed, restricted, _, gradient = suggestion
            if gradient > 0 and isfinite(gradient):
                gradients[beta] = Fraction(gradient).limit_denominator(8192) or beta
            if total - restricted <= max(n, restricted * .000001):
                break
            iteration += 1
            price = (proposed if iteration % 3 == 0 or not new
                     else tuple((3 * p + a + 2) // 4 for p, a in zip(proposed, anchor)))

    def until(share):
        return min(stop, monotonic() + work * share)

    def binding():
        return min(lines, key=lambda b: abs(lines[b] - point[0] - b * point[1]), default=beta0)

    orders = [(1, .12), (2, .19), (3, .28), (4, .19)]
    if violation:
        orders = [(3, .30), (2, .16), (4, .18), (1, .08)]
    for m, share in orders:
        if m > len(labels) or monotonic() >= stop:
            continue
        phase(binding(), labels[:m], 6, until(share))
    center = binding()
    adaptive = [(3 * center + gradients.get(center, center)) / 4,
                4 * center / 5, 5 * center / 4]
    for beta in dict.fromkeys(adaptive):
        if monotonic() >= stop:
            break
        phase(beta, labels[:min(3, len(labels))], 4, until(.095), reverse_start=True)
    if monotonic() < stop:
        phase(binding(), labels[:min(3, len(labels))], 4, stop)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--vehicles", type=int, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warm-zone", "--warm-route", dest="warm", type=Path)
    parser.add_argument("--columns", type=Path)
    args = parser.parse_args()
    if args.vehicles < 0 or not isfinite(args.seconds) or args.seconds <= 0:
        parser.error("vehicles must be nonnegative and seconds must be finite and positive")
    args.out.mkdir(parents=True, exist_ok=True)
    instance = json.loads(args.instance.read_text(), parse_float=Fraction)
    solve(instance, args.vehicles, args.out, START + args.seconds, args.warm, args.columns)


if __name__ == "__main__":
    main()
