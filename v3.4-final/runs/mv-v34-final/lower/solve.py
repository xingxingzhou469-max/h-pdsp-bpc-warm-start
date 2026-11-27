#!/usr/bin/env python3
"""Directed route surgery for the multi-vehicle H-PDSP."""

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
FROZEN = Path(os.environ.get("V33_V32", HERE.parents[1] / "mv-v32-integration"))
spec = spec_from_file_location("frozen_levels", FROZEN / "levels/solve.py")
frozen = module_from_spec(spec)
spec.loader.exec_module(frozen)
frozen.START_NS = START_NS


def search(data, publisher, seed, warm, guide, payload):
    if not publisher.vehicles or not data.cap or not data.total or not any(data.caps):
        return
    arguments = frozen.native_arguments(data)
    if arguments is None:
        frozen.exact_seeds(data, publisher)
        return

    library = ct.CDLL(str(HERE / "build/surgery.so"))
    engine = library.astra_surgery
    engine.argtypes = (ct.c_int, ct.c_int, ct.c_uint, ct.c_double, ct.c_double,
                       ct.c_double, frozen.DOUBLES, frozen.DOUBLES, frozen.DOUBLES,
                       frozen.INTS, ct.c_int, frozen.DOUBLES, frozen.DOUBLES, frozen.PUBLISH)
    engine.restype = ct.c_int
    errors = []

    @frozen.PUBLISH
    def receive(source, route_payload, length, _score):
        try:
            return publisher.consider(frozen.unpack(route_payload, length), source.decode("ascii"))
        except Exception as error:
            errors.append(error)
            return -1

    packed = [len(warm)]
    for route in warm:
        packed.extend((len(route), *route))
    native_warm = (ct.c_int * len(packed))(*packed)
    guidance = frozen.native_guide(guide, payload, publisher, len(data.nodes) + 1)
    remaining = (publisher.deadline_ns - time.monotonic_ns()) / frozen.NANO - 0.006
    if remaining > 0:
        status = engine(len(data.nodes), min(publisher.vehicles, len(data.nodes)),
                        seed, remaining, *arguments, native_warm, len(packed), *guidance, receive)
        if errors:
            raise errors[0]
        if status:
            raise RuntimeError(f"native surgery returned {status}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--vehicles", type=int, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warm", type=Path)
    parser.add_argument("--guide", type=Path)
    args = parser.parse_args()
    if not 0 < args.seconds <= 86_400 or args.vehicles < 0 or not 0 <= args.seed < 2**32:
        parser.error("require 0 < seconds <= 86400, vehicles >= 0, and an unsigned 32-bit seed")

    payload = args.instance.read_bytes()
    instance = json.loads(payload, parse_float=Fraction)
    data = frozen.Data.from_instance(instance)
    args.out.mkdir(parents=True, exist_ok=True)
    deadline = START_NS + int(args.seconds * frozen.NANO)
    publisher = frozen.Publisher(instance, args.vehicles, args.out, deadline)
    if args.warm:
        warm = publisher.seed(args.warm)
    else:
        publisher.consider((), "surgery-idle")
        warm = ()
    search(data, publisher, args.seed, warm, args.guide, payload)


if __name__ == "__main__":
    main()
