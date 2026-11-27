"""Exact zone pricing with continuous reoptimization of the route master."""

from time import monotonic, monotonic_ns

START = monotonic()

import argparse
from collections import Counter
import ctypes as ct
from fractions import Fraction
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from route.model import product_bound, read_model
from route.solve import RestrictedMaster, atomically

CELL_LIMIT = 18_500_000


class Native:
    def __init__(self, graph):
        lib = ct.CDLL(str(Path(__file__).with_name("zone_price.so")))
        self.fn = lib.zone_price
        i, v, ptr = ct.c_int, ct.c_int64, ct.POINTER
        self.fn.argtypes = [
            i, i, i, v, ptr(i), ptr(i), ptr(i), ptr(i), ptr(v), ptr(v), ptr(i), ptr(i),
        ]
        self.fn.restype = i
        n = self.n = len(graph.start)
        self.h, self.layered = graph.horizon, int(graph.layered)
        self.steps = n if self.layered else self.h + 1
        self.data = [
            ((i * (n * n))(*(cost for row in g.arcs for cost in row)),
             (i * n)(*g.start), (i * n)(*g.finish))
            for g in (graph, graph.reversed())
        ]
        self.paths = (i * (n * self.steps))()
        self.lengths = (i * n)()

    def price(self, reward, partition, reverse, end):
        remaining = min(1 << 60, int((end - monotonic()) * 1e9))
        if remaining <= 0:
            return None
        if max(map(abs, reward), default=0) * (self.steps + 2) >= 2**59:
            raise OverflowError("zone price outside signed integer range")
        out = ct.c_int64()
        status = self.fn(
            self.n, self.h, self.layered, remaining, *self.data[reverse],
            (ct.c_int * self.n)(*partition), (ct.c_int64 * self.n)(*reward),
            ct.byref(out), self.lengths, self.paths,
        )
        if status in (2, 3):
            return None
        if status:
            raise RuntimeError(f"zone pricing status {status}")
        paths = [
            tuple(self.paths[i * self.steps:i * self.steps + length])
            for i, length in enumerate(self.lengths) if length
        ]
        return out.value, [path[::-1] for path in paths] if reverse else paths


def legal_zone(path, partition):
    seen, zone = set(), -1
    for i in path:
        if partition[i] != zone:
            seen.clear()
            zone = partition[i]
        if i in seen:
            return False
        seen.add(i)
    return True


def repeated_segments(paths, prices):
    moves = Counter()
    for index, path in enumerate(sorted(paths, key=lambda p: sum(prices[i] for i in p),
                                        reverse=True)[:16]):
        last = {}
        for j, i in enumerate(path):
            if i in last:
                moves[tuple(sorted(set(path[last[i]:j])))] += 16 - index
            last[i] = j
    return moves


def cluster(graph, width, votes, pairs):
    n = len(graph.arcs)
    if width == 1:
        return tuple(range(n))
    roots = list(range(n))
    groups = {i: [i] for i in range(n)}

    def merge(candidates):
        labels = sorted({roots[i] for i in candidates})
        if len(labels) <= 1 or sum(len(groups[label]) for label in labels) > width:
            return
        label = labels[0]
        for other in labels[1:]:
            group = groups.pop(other)
            for j in group:
                roots[j] = label
            groups[label].extend(group)

    for nodes, _ in sorted(votes.items(), key=lambda a: (-a[1] / len(a[0]), a[0])):
        if len(nodes) <= width:
            merge(nodes)
    for _, _, i, j in pairs:
        merge((i, j))
    return tuple(roots)


class Master(RestrictedMaster):
    def __init__(self, highspy, np, weights, k, model, partition):
        super().__init__(highspy, np, weights, k, False)
        self.model = model
        self.partition = partition
        self.resources = []

    def add(self, paths):
        fresh = []
        pending = set()
        stats = []
        for path in paths:
            if not legal_zone(path, self.partition):
                continue
            key = tuple(sorted(Counter(path).items()))
            if key in self.keys or key in pending:
                continue
            pending.add(key)
            fresh.append(path)
            stats.append((
                sum(float(self.model.stock[i]) * number for i, number in key),
                sum(float(self.model.people[i]) * number for i, number in key),
            ))
        added = super().add(fresh)
        self.resources.extend(stats)
        return added

    def suggest(self, end):
        hint = self.prices(end)
        if hint is None:
            return None, None, None
        lp = self.solver.getObjectiveValue()
        vals = self.solver.getSolution().row_dual
        q = sum(x * pair[0] for x, pair in zip(vals, self.resources))
        p = sum(x * pair[1] for x, pair in zip(vals, self.resources))
        slope = q / p if p > 0 and q > 0 else None
        if slope is not None and not isfinite(slope):
            slope = None
        return hint, lp, slope


def knapsack_price(model, weights, k):
    cost = tuple((a + b) / 2 for a, b in zip(model.in_cost, model.out_cost))
    left = model.limit * k
    rate = Fraction()
    for ratio, i in sorted(((w / c, i) for i, (w, c) in enumerate(zip(weights, cost)) if c),
                           reverse=True):
        if left < cost[i]:
            rate = ratio
            break
        left -= cost[i]
    return tuple(max(0, round(w - rate * c)) for w, c in zip(weights, cost))


class Archive:
    def __init__(self, model, out, deadline):
        self.model, self.out, self.deadline = model, out, deadline
        self.good, self.walks, self.seen = {}, {}, set()
        self.last = 0

    def add(self, paths, weights):
        for path in paths:
            if path in self.seen:
                continue
            self.seen.add(path)
            unique = frozenset(path)
            value = sum(weights[i] for i in unique)
            direct = len(unique) == len(path)
            if direct:
                m = self.model
                duration = (m.increments[0][path[0]] + m.home[path[-1]]
                            + sum(m.increments[i + 1][j] for i, j in zip(path, path[1:])))
                direct = duration <= m.limit
                if direct:
                    old = self.good.get(unique)
                    if old is None or duration < old[1]:
                        self.good[unique] = (value, duration, path)
            if not direct:
                old = self.walks.get(unique)
                if old is None or len(path) < len(old[2]):
                    self.walks[unique] = (value, 0, path)
        for archive in (self.good, self.walks):
            if len(archive) > 1800:
                top = sorted(archive, key=lambda key: archive[key], reverse=True)[:1200]
                keep = {key: archive[key] for key in top}
                archive.clear()
                archive.update(keep)
        self.flush()

    def flush(self, force=False):
        now = monotonic()
        if now >= self.deadline or (not force and now - self.last < .30):
            return
        if not self.good and not self.walks:
            return
        self.last = now
        good = sorted(self.good.values(), reverse=True)[:1400]
        flexible = sorted(self.walks.values(), reverse=True)[:1000]
        atomically(self.out / "columns.json", {
            "routes": [[i + 1 for i in row[2]] for row in (*good, *flexible)],
            "direct_count": len(good),
        })


def load_warm(path, k, n):
    raw = path.read_bytes()
    obj = json.loads(raw)
    if obj.get("method") not in ("route-price-v1", "zone-price-v1") or obj.get("vehicles") != k:
        raise ValueError("incompatible route price suggestion")
    prices = {}
    for line in obj["supports"]:
        beta, values = Fraction(line["beta"]), tuple(line["prices"])
        if beta <= 0 or len(values) != n or any(type(x) is not int or x < 0 for x in values):
            raise ValueError("invalid route price suggestion")
        prices[beta] = values
    return prices, sha256(raw).hexdigest()


def solve(instance, k, out, deadline, warm=None):
    model = read_model(instance)
    boxes = tuple(model.boxes(k)) if k else ()
    centers, source = load_warm(warm, k, model.n) if warm else ({}, None)
    upper, point = None, (Fraction(), Fraction())
    lines, proofs = {}, {}

    def publish():
        nonlocal upper, point
        candidate, point = product_bound(model, boxes, list(lines.items()))
        if (upper is None or candidate < upper) and monotonic() < deadline:
            atomically(out / "best.json", {
                "method": "zone-price-v1",
                "vehicles": k,
                "upper": str(candidate),
                "supports": list(proofs.values()),
                "warm_route_sha256": source,
                "emitted_monotonic_ns": monotonic_ns(),
            })
            upper = candidate

    publish()
    if not upper or model.n > 256 or monotonic() >= deadline:
        return
    graph = model.project()
    table_rows = (graph.horizon + 1) * (model.n if graph.layered else 1)
    if table_rows * model.n * 3 > CELL_LIMIT:
        return
    native = Native(graph)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "HIGHS_NUM_THREADS"):
        os.environ[name] = "1"
    import highspy
    import numpy as np

    arch = Archive(model, out, deadline)
    work = max(0, deadline - monotonic())
    stop = deadline - min(.04, work * .05)
    pool, vote, gradients = {}, Counter(), {}
    ref = model.quantity / model.population
    initial = (point[0] / point[1]).limit_denominator(8192) or ref
    pairs = sorted(
        (min(graph.arcs[i][j], graph.arcs[j][i]), graph.arcs[i][j] + graph.arcs[j][i], i, j)
        for i in range(model.n) for j in range(i + 1, model.n)
    )

    def partition_for(width):
        part = cluster(graph, width, vote, pairs)
        frame = sum(size * (1 << (size - 1)) for size in Counter(part).values())
        return part if table_rows * (2 * model.n + frame) <= CELL_LIMIT else None

    def initial_price(beta, weights):
        if beta in centers:
            return centers[beta]
        if centers:
            near = min(centers, key=lambda b: abs(beta - b))
            return tuple(round(p * (beta / near if person else 1))
                         for p, person in zip(centers[near], model.people))
        return knapsack_price(model, weights, k)

    def phase(beta, part, end):
        if part is None or monotonic() >= end:
            return
        weights = model.weights(beta)
        prices = anchor = initial_price(beta, weights)
        solver = Master(highspy, np, weights, k, model, part)
        solver.add(pool)
        seen = set()
        iteration = 0
        while monotonic() < end:
            reverse = bool(iteration % 2)
            if (prices, reverse) in seen:
                reverse = not reverse
            if (prices, reverse) in seen:
                break
            seen.add((prices, reverse))
            rewards = tuple(w - p for w, p in zip(weights, prices))
            result = native.price(rewards, part, reverse, end)
            if result is None:
                break
            bound, paths = result
            total = k * bound + sum(prices)
            line = model.reward_unit * total
            if beta not in lines or line < lines[beta]:
                anchor = centers[beta] = prices
                lines[beta] = line
                proofs[beta] = {
                    "beta": str(beta), "prices": prices, "partition": part,
                    "route_bound": bound, "reverse": reverse,
                }
                publish()
            vote.update(repeated_segments(paths, rewards))
            for path in paths:
                pool[path] = None
            while len(pool) > 9000:
                del pool[next(iter(pool))]
            added = solver.add(paths)
            arch.add(paths, weights)
            hint, restricted, slope = solver.suggest(end)
            if slope is not None:
                gradients[beta] = Fraction(slope).limit_denominator(8192) or beta
            if hint is None:
                break
            if total - restricted <= max(k * model.n, restricted * 0.000001):
                break
            iteration += 1
            prices = (hint if iteration % 3 == 0 or not added
                      else tuple((3 * p + a + 2) // 4 for p, a in zip(hint, anchor)))

    def until(share):
        return min(stop, monotonic() + work * share)

    def limiting():
        return min(lines, key=lambda b: abs(lines[b] - point[0] - b * point[1]), default=initial)

    def slope_proposals(beta):
        target = gradients.get(beta, beta)
        smooth = (3 * beta + target) / 4
        return [
            smooth.limit_denominator(8192) or beta, beta * Fraction(4, 5),
            beta * Fraction(5, 4), beta * Fraction(2, 3), beta * Fraction(3, 2),
        ]

    phase(initial, partition_for(1), until(.16))
    part4 = partition_for(4)
    phase(limiting(), part4, until(.18))
    for beta in list(dict.fromkeys(slope_proposals(limiting())))[:3]:
        if beta not in lines:
            phase(beta, part4, until(.075))
    for width, share in ((6, .15), (8, .14), (10, .10), (12, .095)):
        if monotonic() >= stop:
            break
        part = partition_for(width)
        if part is not None:
            phase(limiting(), part, until(share))
            part4 = part

    marked = set()
    while monotonic() < stop:
        bind = limiting()
        directions = [bind, *slope_proposals(bind)]
        beta = next((b for b in directions if b not in marked and b > 0), None)
        if beta is None:
            break
        marked.add(beta)
        phase(beta, part4 or partition_for(1), until(.18))
    arch.flush(force=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--vehicles", type=int, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warm-route", type=Path)
    args = parser.parse_args()
    if args.vehicles < 0 or not isfinite(args.seconds) or args.seconds <= 0:
        parser.error("vehicles must be nonnegative and seconds must be finite and positive")
    args.out.mkdir(parents=True, exist_ok=True)
    instance = json.loads(args.instance.read_text(), parse_float=Fraction)
    solve(instance, args.vehicles, args.out, START + args.seconds, args.warm_route)


if __name__ == "__main__":
    main()
