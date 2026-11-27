#!/usr/bin/env python3
"""Fixed-other-vehicles directed giant-tour subsequence search."""

import time

START_NS = time.monotonic_ns()

import argparse
import ctypes as ct
from fractions import Fraction
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
V32 = Path(os.environ.get("V33_V32", HERE.parents[1] / "mv-v32-integration"))
spec = spec_from_file_location("frozen_levels", V32 / "levels" / "solve.py")
frozen = module_from_spec(spec)
spec.loader.exec_module(frozen)
frozen.START_NS = START_NS


def search(data, publisher, seed, warm):
    if not warm or not data.cap or not data.total:
        return
    values = frozen.native_arguments(data)
    if values is None:
        return
    library = ct.CDLL(str(HERE / "build/subsequence.so"))
    engine = library.astra_subsequence
    engine.argtypes = (ct.c_int, ct.c_int, ct.c_uint, ct.c_double, ct.c_double,
                       ct.c_double, frozen.DOUBLES, frozen.DOUBLES, frozen.DOUBLES,
                       ct.c_int, frozen.INTS, ct.c_int, frozen.PUBLISH)
    engine.restype = ct.c_int
    errors = []

    @frozen.PUBLISH
    def receive(source, payload, length, _estimate):
        try:
            return publisher.consider(frozen.unpack(payload, length), source.decode("ascii"))
        except Exception as error:
            errors.append(error)
            return -1

    packed = [len(warm)]
    for route in warm:
        packed.extend((len(route), *route))
    native = (ct.c_int * len(packed))(*packed)

    # Integer input times fit in a moderate grid without rounding. For other
    # inputs the native objective is a search heuristic, with exact replay.
    if 0 < data.limit <= 800 and data.limit.denominator == 1 and all(
            x.denominator == 1 for row in (*data.increments, data.home) for x in row):
        ticks = int(data.limit)
    else:
        ticks = 400
    remaining = (publisher.deadline_ns - time.monotonic_ns()) / frozen.NANO - 0.009
    if remaining <= 0:
        return
    status = engine(len(data.nodes), min(publisher.vehicles, len(data.nodes)), seed,
                    remaining, *values, ticks, native, len(packed), receive)
    if errors:
        raise errors[0]
    if status:
        raise RuntimeError(f"native subsequence search returned {status}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--vehicles", type=int, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warm", type=Path, required=True)
    args = parser.parse_args()
    if args.vehicles < 0 or not 0 < args.seconds <= 86400 or not 0 <= args.seed < 2**32:
        parser.error("invalid fleet, budget or seed")
    raw = json.loads(args.instance.read_bytes(), parse_float=Fraction)
    data = frozen.Data.from_instance(raw)
    args.out.mkdir(parents=True, exist_ok=True)
    deadline = START_NS + int(args.seconds * frozen.NANO)
    publisher = frozen.Publisher(raw, args.vehicles, args.out, deadline)
    warm = publisher.seed(args.warm)
    search(data, publisher, args.seed, warm)


if __name__ == "__main__":
    main()
