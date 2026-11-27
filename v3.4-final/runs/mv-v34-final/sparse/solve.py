#!/usr/bin/env python3
"""Anytime fair upper bound by sparse supplier partition and route prices."""
from time import monotonic_ns
START = monotonic_ns()

import argparse
from collections import Counter
import ctypes as ct
from fractions import Fraction as F
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import os
import warnings

from model import METHOD, SCALE, Model, clusters

HERE = Path(__file__).resolve().parent
NS = 10**9


class Native:
    def __init__(self, model, partition):
        self.model = model
        m = self.model
        self.fn = ct.CDLL(str(HERE / "price.so")).sparse_price
        i, v = ct.c_int, ct.c_int64
        self.fn.argtypes = [i, i, i, v, ct.POINTER(i), ct.POINTER(i), ct.POINTER(i),
                            ct.POINTER(v), ct.POINTER(v), ct.POINTER(v), ct.POINTER(i), ct.POINTER(i)]
        self.fn.restype = i
        self.args = ((i * (m.n * m.n))(*(c for row in m.costs for c in row)),
                     (i * m.n)(*m.bits), (i * m.n)(*partition))
        self.path = (i * (m.horizon + 1))()

    def price(self, group, rewards, prices, deadline):
        m = self.model
        remaining = deadline - monotonic_ns()
        if remaining <= 0 or max(max(map(abs, rewards)), max(prices)) * (m.horizon + 1) >= (1 << 59):
            return None
        bound, length = ct.c_int64(), ct.c_int()
        status = self.fn(m.n, m.horizon, group, remaining, *self.args,
                         (ct.c_int64 * m.n)(*rewards), (ct.c_int64 * m.n)(*(-p for p in prices)),
                         ct.byref(bound), ct.byref(length), self.path)
        if status in (2, 3):
            return None
        if status:
            raise RuntimeError(f"pricing failed with status {status}")
        route = tuple(self.path[:length.value])
        if not route:
            return bound.value, None
        first = next(i for i, node in enumerate(route) if m.bits[node])
        population = sum((m.p[node] for node in route[first + 1:]), F())
        signature = (population, tuple(sorted(Counter(route).items())))
        return bound.value, signature


class Leaf:
    def __init__(self, groups, model, caps):
        self.groups = groups
        self.upper = model.box(groups, caps)
        self.alpha = (F(1, 2),) * (len(groups) * (len(groups) - 1) // 2)
        self.prices = (0,) * model.n
        self.columns = [set() for _ in groups]
        self.line = None
        self.trials = 0
        self.done = not groups or self.upper == 0

    def export(self):
        return {"groups": list(self.groups), "upper": str(self.upper), "line": self.line}

    def parameters(self, model, linprog, np):
        groups = self.groups
        dimension = len(groups) + model.n + len(self.alpha)
        offset = len(groups) + model.n
        if not any(self.columns):
            return
        rows, rhs = [], []
        pair_indices = tuple(combinations(range(len(groups)), 2))
        base = model.supports(groups, (F(),) * len(self.alpha))
        stock = tuple(model.stock[g] for g in groups)
        for slot, columns in enumerate(self.columns):
            for population, path in sorted(columns):
                objective_mass = float(population / model.total)
                row = np.zeros(dimension)
                row[slot] = -1
                for i, visits in path:
                    if model.p[i]:
                        row[len(groups) + i] = -visits
                for j, (a, b) in enumerate(pair_indices):
                    if slot == a:
                        row[offset + j] = float(2 * stock[b]) * objective_mass
                    if slot == b:
                        row[offset + j] = -float(2 * stock[a]) * objective_mass
                rows.append(row)
                rhs.append(-float(base[slot]) * objective_mass)
        objective = np.ones(dimension)
        objective[offset:] = 0
        bounds = [(0, None)] * len(groups) + [
            (0, None) if model.p[i] else (0, 0) for i in range(model.n)] + [(0, 1)] * len(self.alpha)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected")
            result = linprog(objective, A_ub=rows, b_ub=rhs, bounds=bounds,
                             method="highs-ds", options={"threads": 1})
        if not result.success:
            return
        self.prices = tuple(max(0, round(SCALE * x)) for x in result.x[len(groups):offset])
        self.alpha = tuple(F(min(SCALE, max(0, round(SCALE * x))), SCALE) for x in result.x[offset:])

    def iterate(self, native, deadline, linprog, np):
        model = native.model
        self.parameters(model, linprog, np)
        coefficients = model.supports(self.groups, self.alpha)
        priors = tuple(len(x) for x in self.columns)
        bounds = []
        for i, (group, coefficient) in enumerate(zip(self.groups, coefficients)):
            result = native.price(group, model.rewards(coefficient, self.prices), self.prices, deadline)
            if result is None:
                self.done = True
                return
            bound, path = result
            bounds.append(bound)
            if path:
                self.columns[i].add(path)
        endpoint = F(sum(self.prices) + sum(bounds), SCALE)
        self.trials += 1
        if endpoint < self.upper:
            self.upper = endpoint
            self.line = {"alpha": list(map(str, self.alpha)), "prices": list(self.prices),
                         "route_bounds": bounds}
        if priors == tuple(len(x) for x in self.columns):
            self.done = True


def publish(leaves, partition, width, k, path, deadline):
    now = monotonic_ns()
    if now >= deadline:
        return
    document = {"method": METHOD, "vehicles": k, "partition": partition, "cell_width": width,
                "leaves": [leaf.export() for leaf in leaves],
                "upper": str(max(x.upper for x in leaves)),
                "verified_elapsed_ns": now - START}
    tmp = path.with_suffix(".tmp")
    try:
        with tmp.open("w") as file:
            json.dump(document, file, separators=(",", ":"))
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        if monotonic_ns() < deadline:
            tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def save_warm(path, fingerprint, leaves):
    # Keep pricing signatures as LP proposals. Each reported bound is repriced.
    columns = []
    for leaf in leaves:
        if any(leaf.columns):
            slots = [[[str(population), route] for population, route in sorted(slot)[:512]]
                     for slot in leaf.columns]
            columns.append([list(leaf.groups), slots])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"input_sha256": fingerprint, "columns": columns},
                              separators=(",", ":")) + "\n")
    tmp.replace(path)


def load_warm(path, fingerprint, leaves):
    data = json.loads(path.read_bytes())
    if data["input_sha256"] != fingerprint:
        raise ValueError("sparse warm columns belong to a different input")
    by_leaf = {tuple(groups): columns for groups, columns in data["columns"]}
    for leaf in leaves:
        for slot, columns in zip(leaf.columns, by_leaf.get(leaf.groups, [])):
            slot.update((F(population), tuple(map(tuple, route))) for population, route in columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--vehicles", required=True, type=int)
    parser.add_argument("--seconds", required=True, type=float)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cell-width", type=int, choices=range(1, 7), default=5)
    parser.add_argument("--warm", type=Path)
    parser.add_argument("--save-warm", action="store_true")
    args = parser.parse_args()
    if not 0 < args.seconds <= 86400 or args.vehicles < 0:
        parser.error("require a nonnegative fleet and positive finite budget")
    end = START + int(args.seconds * NS)
    payload = args.instance.read_bytes()
    model = Model(json.loads(payload, parse_float=F), args.vehicles)
    fingerprint = sha256(payload).hexdigest() if args.warm or args.save_warm else None
    caps = model.cheap_caps()
    leaves = [Leaf(parts, model, caps) for parts in model.covers()]
    partition = clusters(model, args.cell_width)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "best.json"
    if path.exists():
        raise FileExistsError(path)
    publish(leaves, partition, args.cell_width, args.vehicles, path, end)
    if args.warm:
        load_warm(args.warm, fingerprint, leaves)

    if not model.positive or not model.total or not model.capacity:
        return
    import numpy as np
    from scipy.optimize import linprog
    native = Native(model, partition)
    last_save = START
    while monotonic_ns() + 15_000_000 < end:
        candidates = [x for x in leaves if not x.done]
        if not candidates:
            break
        leaf = max(candidates, key=lambda x: (x.upper, -x.trials))
        if any(x.done and x.upper >= leaf.upper for x in leaves):
            break
        before = max(x.upper for x in leaves)
        leaf.iterate(native, end - 12_000_000, linprog, np)
        if max(x.upper for x in leaves) < before:
            publish(leaves, partition, args.cell_width, args.vehicles, path, end)
        if args.save_warm and last_save + NS // 2 < monotonic_ns() < end - 15_000_000:
            save_warm(args.out / "warm.json", fingerprint, leaves)
            last_save = monotonic_ns()
    publish(leaves, partition, args.cell_width, args.vehicles, path, end)
    if args.save_warm and monotonic_ns() < end - 15_000_000:
        save_warm(args.out / "warm.json", fingerprint, leaves)
    print(json.dumps({"caps": {str(g): str(p) for g, p in caps.items()},
                      "leaves": [{"groups": x.groups, "upper": str(x.upper), "trials": x.trials,
                                  "columns": list(map(len, x.columns))} for x in leaves]}), flush=True)


if __name__ == "__main__":
    main()
