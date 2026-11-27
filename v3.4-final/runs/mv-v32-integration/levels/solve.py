#!/usr/bin/env python3
"""Anytime H-PDSP search using discretized delivery-rate integrals."""

import time

START_NS = time.monotonic_ns()

import argparse
import ctypes as ct
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
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

    def seed(self, path):
        record = json.loads(path.read_bytes(), parse_float=Fraction)
        witness = record.get("witness", record.get("certificate", record))
        score = verify_lower(self.instance, self.vehicles, witness)
        if ("lower" in record and score != number(record["lower"])
                or "score" in record and score != number(record["score"])):
            raise ValueError("warm witness score differs")
        routes = tuple(tuple(route) for route in witness["routes"] if route)
        normalized = {"routes": witness["routes"],
                      "amounts": [[str(number(q)) for q in row] for row in witness["amounts"]]}
        self.write(normalized, score, "levels-warm")
        self.consider(routes, "levels-warm-waterfill")
        return routes

    def consider(self, routes, source):
        if time.monotonic_ns() >= self.deadline_ns:
            return 0
        columns = [self.column(route) for route in routes]
        if any(column is None for column in columns):
            return 0
        witness = {"routes": routes, "amounts": columns}
        score = verify_lower(self.instance, self.vehicles, witness)
        return self.write(witness, score, source)

    def write(self, witness, score, source):
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


def guide_digest(instance, vehicles):
    """Serialize the model using the root-LP guide's canonical field order."""
    def canonical(value):
        if isinstance(value, list):
            return [canonical(item) for item in value]
        return str(number(value))

    fields = ("supply", "population", "service", "travel", "limit", "capacity")
    model = [vehicles, *(canonical(instance[name]) for name in fields)]
    return sha256(json.dumps(model, separators=(",", ":")).encode()).hexdigest()


def native_guide(path, payload, publisher, size):
    if path is None:
        return None, None
    document = json.loads(path.read_bytes(), parse_float=Fraction)
    if "input_sha256" in document and document["input_sha256"] != sha256(payload).hexdigest():
        raise ValueError("guide input SHA-256 differs")
    if "vehicles" in document and document["vehicles"] != publisher.vehicles:
        raise ValueError("guide fleet size differs")
    if ("instance_digest" in document
            and document["instance_digest"] != guide_digest(publisher.instance, publisher.vehicles)):
        raise ValueError("guide model digest differs")
    if len(document["y"]) != size:
        raise ValueError("guide station vector has the wrong length")
    visits = [float(min(1, max(0, number(value)))) for value in document["y"]]
    flows = [0.0] * (size * size)
    for previous, target, value in document["arcs"]:
        if (type(previous) is not int or type(target) is not int
                or not 0 <= min(previous, target) <= max(previous, target) < size):
            raise ValueError("guide arc has an invalid station")
        index = previous * size + target
        flows[index] = min(1.0, flows[index] + float(min(1, max(0, number(value)))))
    return (ct.c_double * size)(*visits), (ct.c_double * len(flows))(*flows)


def search(data, publisher, seed, warm, guide, payload):
    if not publisher.vehicles or not data.cap or not data.total or not any(data.caps):
        return
    arguments = native_arguments(data)
    if arguments is None:
        exact_seeds(data, publisher)
        return
    artifact = HERE / "build/levels.so"
    if not artifact.is_file():
        raise FileNotFoundError(f"{artifact} is missing; run {HERE / 'build.sh'} before timing")
    library = ct.CDLL(str(artifact))
    engine = library.astra_levels
    engine.argtypes = (ct.c_int, ct.c_int, ct.c_uint, ct.c_double, ct.c_double,
                       ct.c_double, DOUBLES, DOUBLES, DOUBLES, INTS, ct.c_int,
                       DOUBLES, DOUBLES, PUBLISH)
    engine.restype = ct.c_int
    errors = []

    @PUBLISH
    def receive(source, payload, length, _approximate):
        try:
            return publisher.consider(unpack(payload, length), source.decode("ascii"))
        except Exception as exc:
            errors.append(exc)
            return -1

    packed = [len(warm)]
    for route in warm:
        packed.extend((len(route), *route))
    native_warm = (ct.c_int * len(packed))(*packed)
    guidance = native_guide(guide, payload, publisher, len(data.nodes) + 1)
    remaining = (publisher.deadline_ns - time.monotonic_ns()) / NANO - 0.003
    if remaining <= 0:
        return
    status = engine(len(data.nodes), min(publisher.vehicles, len(data.nodes)), seed,
                    remaining, *arguments, native_warm, len(packed), *guidance, receive)
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
    parser.add_argument("--warm", type=Path, help="previous best.json or exact fleet witness")
    parser.add_argument("--guide", type=Path, help="station fractions and directed arc flow hints")
    args = parser.parse_args()
    if not 0 < args.seconds <= 86_400 or args.vehicles < 0 or not 0 <= args.seed < 2**32:
        parser.error("require 0 < seconds <= 86400, vehicles >= 0, and an unsigned 32-bit seed")
    deadline = START_NS + int(args.seconds * NANO)
    payload = args.instance.read_bytes()
    instance = json.loads(payload, parse_float=Fraction)
    data = Data.from_instance(instance)
    args.out.mkdir(parents=True, exist_ok=True)
    publisher = Publisher(instance, args.vehicles, args.out, deadline)
    if args.warm is not None:
        warm = publisher.seed(args.warm)
    else:
        publisher.consider((), "levels-idle")
        warm = ()
    search(data, publisher, args.seed, warm, args.guide, payload)


if __name__ == "__main__":
    main()
