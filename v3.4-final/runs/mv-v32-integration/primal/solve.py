#!/usr/bin/env python3
"""Single-threaded anytime lower bound for the multi-vehicle H-PDSP."""

import time

START_NS = time.monotonic_ns()

import argparse
import ctypes as ct
from fractions import Fraction
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "reference"
sys.path.insert(0, str(FROZEN))
from common import Data, number, verify_lower
from primal_vendor.fleet import canonical_column
from primal_vendor.routes import route_time

NANO = 1_000_000_000
DOUBLES = ct.POINTER(ct.c_double)
INTS = ct.POINTER(ct.c_int)
PUBLISH = ct.CFUNCTYPE(ct.c_int, ct.c_char_p, INTS, ct.c_int, ct.c_double)


def unpack(payload, length):
    if length < 1 or payload[0] < 0:
        raise ValueError("invalid native fleet")
    routes = []
    offset = 1
    for _ in range(payload[0]):
        if offset >= length:
            raise ValueError("missing native route length")
        size = payload[offset]
        offset += 1
        if size < 2 or offset + size > length:
            raise ValueError("invalid native route length")
        routes.append(tuple(payload[offset:offset + size]))
        offset += size
    if offset != length:
        raise ValueError("invalid native fleet length")
    return tuple(routes)


class Publisher:
    def __init__(self, instance, vehicles, directory, deadline_ns):
        self.instance = instance
        self.vehicles = vehicles
        self.path = directory / "best.json"
        self.temporary = directory / f".best.{os.getpid()}.tmp"
        self.deadline_ns = deadline_ns
        self.best = None
        self.column = lru_cache(maxsize=1024)(self._column)

    def _column(self, route):
        if route_time(self.instance, route) > number(self.instance["limit"]):
            return None
        _, amounts = canonical_column(self.instance, route)
        return [str(quantity) for quantity in amounts]

    def consider(self, routes, source):
        if time.monotonic_ns() >= self.deadline_ns:
            return 0
        columns = [self.column(route) for route in routes]
        if any(column is None for column in columns):
            return 0
        witness = {"routes": routes, "amounts": columns}
        score = verify_lower(self.instance, self.vehicles, witness)
        if self.best is not None and score <= self.best:
            return 0
        now = time.monotonic_ns()
        if now >= self.deadline_ns:
            return 0
        record = {"lower": str(score), "witness": witness, "source": source,
                  "verified_elapsed_ns": now - START_NS}
        try:
            with self.temporary.open("w", encoding="utf-8") as output:
                json.dump(record, output, separators=(",", ":"))
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            if time.monotonic_ns() >= self.deadline_ns:
                return 0
            self.temporary.replace(self.path)
        finally:
            self.temporary.unlink(missing_ok=True)
        self.best = score
        return 1


def native_arguments(data):
    """Scale each numerical axis.  None requests the exact directed seed search."""
    size = len(data.nodes) + 1
    if size > 513:
        return None
    stock_scale = max(data.caps)
    population_scale = max(data.people)
    capacity = float(min(data.cap / stock_scale, len(data.nodes)))
    stock = [0.0] + [float(q / stock_scale) for q in data.caps]
    people = [0.0] + [float(p / population_scale) for p in data.people]
    # Squared and divided rates must stay well within the native numeric range.
    if any(0 < q and x < 1e-70 for q, x in zip(data.caps, stock[1:])):
        return None
    if any(0 < p and x < 1e-70 for p, x in zip(data.people, people[1:])):
        return None

    def duration(value):
        if value > data.limit:
            return 2.0
        return float(value / data.limit) if data.limit else 0.0

    costs = [duration(value)
             for before, row in ((0, data.increments[-1]),
                                 *zip(data.nodes, data.increments[:-1]))
             for value in ((data.home[before - 1] if before else Fraction(0)), *row)]
    return (capacity, float(data.limit > 0), (ct.c_double * size)(*stock),
            (ct.c_double * size)(*people), (ct.c_double * (size * size))(*costs))


def exact_seed(data, used, deadline_ns):
    """Find a shortest eligible directed supplier-to-agency path with Fraction."""
    remaining = [node for node in data.nodes if node not in used]
    distance = {node: data.increments[-1][node - 1] for node in remaining
                if data.sources[node - 1]}
    predecessors = {node: 0 for node in distance}
    settled = set()
    tail = 0
    best_time = data.limit
    while distance and time.monotonic_ns() < deadline_ns:
        node = min(distance, key=distance.__getitem__)
        cost = distance.pop(node)
        if cost > best_time:
            break
        settled.add(node)
        if not data.sources[node - 1] and cost + data.home[node - 1] <= best_time:
            tail, best_time = node, cost + data.home[node - 1]
        for target in remaining:
            if target in settled:
                continue
            arrival = cost + data.increments[node - 1][target - 1]
            if arrival < distance.get(target, math.inf) and arrival <= best_time:
                distance[target] = arrival
                predecessors[target] = node
    route = []
    while tail:
        route.append(tail)
        tail = predecessors[tail]
    return tuple(reversed(route))


def exact_seeds(data, publisher):
    """Lightweight legal search for data outside the native numeric range."""
    routes, used = [], set()
    for _ in range(publisher.vehicles):
        if time.monotonic_ns() >= publisher.deadline_ns:
            break
        route = exact_seed(data, used, publisher.deadline_ns)
        if not route:
            break
        routes.append(route)
        used.update(route)
        publisher.consider(tuple(routes), "exact-directed-seed")


def search(data, publisher, seed):
    if not publisher.vehicles or not data.cap or not data.total or not any(data.caps):
        return
    arguments = native_arguments(data)
    if arguments is None:
        exact_seeds(data, publisher)
        return
    artifact = HERE / "build/fleet_primal.so"
    if not artifact.is_file():
        raise FileNotFoundError(f"{artifact} is missing; run {HERE / 'build.sh'} before timing")
    library = ct.CDLL(str(artifact))
    engine = library.astra_primal
    engine.argtypes = (ct.c_int, ct.c_int, ct.c_uint, ct.c_double, ct.c_double,
                       ct.c_double, DOUBLES, DOUBLES, DOUBLES, PUBLISH)
    engine.restype = ct.c_int
    errors = []

    @PUBLISH
    def receive(source, payload, length, _approximate):
        try:
            return publisher.consider(unpack(payload, length), source.decode("ascii"))
        except Exception as exc:
            errors.append(exc)
            return -1

    remaining = (publisher.deadline_ns - time.monotonic_ns()) / NANO - 0.003
    if remaining <= 0:
        return
    status = engine(len(data.nodes), min(publisher.vehicles, len(data.nodes)), seed,
                    remaining, *arguments, receive)
    if errors:
        raise errors[0]
    if status:
        raise RuntimeError(f"native primal returned {status}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--vehicles", type=int, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not 0 < args.seconds <= 86_400 or args.vehicles < 0 or not 0 <= args.seed < 2**32:
        parser.error("require 0 < seconds <= 86400, vehicles >= 0, and an unsigned 32-bit seed")
    deadline = START_NS + int(args.seconds * NANO)
    instance = json.loads(args.instance.read_bytes(), parse_float=Fraction)
    data = Data.from_instance(instance)
    args.out.mkdir(parents=True, exist_ok=True)
    publisher = Publisher(instance, args.vehicles, args.out, deadline)
    publisher.consider((), "idle")
    search(data, publisher, args.seed)


if __name__ == "__main__":
    main()
