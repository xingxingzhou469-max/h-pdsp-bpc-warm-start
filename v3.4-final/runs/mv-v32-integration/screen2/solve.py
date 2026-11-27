"""Anytime exact two-vehicle mask screening around a zone support."""

from time import monotonic, monotonic_ns

START = monotonic()

import argparse
import ctypes as ct
from fractions import Fraction
import json
from math import isfinite
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from route.model import product_bound, read_model
from route.solve import atomically
from screen2.check import METHOD, excludes_fleet, identity, mask_digest, threshold_data
from zone.solve import Native


class Screen:
    def __init__(self, graph):
        lib = ct.CDLL(str(Path(__file__).with_name("screen_kernel.so")))
        self.fn = lib.zone_screen
        i, v, u, ptr = ct.c_int, ct.c_int64, ct.c_uint64, ct.POINTER
        self.fn.argtypes = [
            i, i, i, v, ptr(i), ptr(i), ptr(i), ptr(i), ptr(v), v, i, ptr(i), ptr(u), ptr(u),
        ]
        self.fn.restype = i
        n = self.n = len(graph.start)
        self.h, self.layered = graph.horizon, int(graph.layered)
        self.data = [
            ((i * (n * n))(*(cost for row in g.arcs for cost in row)), (i * n)(*g.start),
             (i * n)(*g.finish))
            for g in (graph, graph.reversed())
        ]
        self.max_masks = 300000
        self.masks = (u * (2 * self.max_masks))()

    def enumerate(self, reward, part, cutoff, reverse, deadline):
        limit = int((deadline - monotonic()) * 1e9)
        if limit <= 0 or not -(1 << 61) <= cutoff <= (1 << 61):
            return None
        steps = self.n if self.layered else self.h + 1
        if max(map(abs, reward), default=0) * (steps + 2) >= 2**59:
            return None
        count, explored = ct.c_int(), ct.c_uint64()
        status = self.fn(
            self.n, self.h, self.layered, limit, *self.data[reverse],
            (ct.c_int * self.n)(*part), (ct.c_int64 * self.n)(*reward), cutoff,
            self.max_masks, ct.byref(count), self.masks, ct.byref(explored),
        )
        if status in (2, 3):
            return None
        if status:
            raise RuntimeError(f"mask enumeration status {status}")
        return ({int(self.masks[2 * i]) | int(self.masks[2 * i + 1]) << 64 for i in range(count.value)},
                explored.value)


def validate_parent(model, parent, k, deadline):
    if (parent.get("method") != "zone-price-v1" or parent.get("vehicles") != k
            or type(parent.get("upper")) is not str):
        raise ValueError("screening warm proof must be a zone proof for the same fleet")
    lines = []
    native = Native(model.project()) if parent["supports"] else None
    for line in parent["supports"]:
        beta = Fraction(line["beta"])
        prices, part = tuple(line["prices"]), tuple(line["partition"])
        reverse, cap = line["reverse"], line["route_bound"]
        if (beta <= 0 or len(prices) != model.n or len(part) != model.n
                or any(type(p) is not int or p < 0 for p in prices)
                or any(type(x) is not int or not 0 <= x < model.n for x in part)
                or type(reverse) is not bool or type(cap) is not int or cap < 0):
            raise ValueError("invalid zone warm support")
        rewards = tuple(w - p for w, p in zip(model.weights(beta), prices))
        result = native.price(rewards, part, reverse, deadline)
        if result is None:
            return None
        if cap != result[0]:
            raise ValueError("warm cap does not match the current input")
        lines.append((beta, model.reward_unit * (k * cap + sum(prices))))
    upper = product_bound(model, tuple(model.boxes(k)) if k else (), lines)[0]
    if upper != Fraction(parent["upper"]):
        raise ValueError("warm proof does not reconstruct its stated upper")
    return upper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--vehicles", type=int, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warm-route", type=Path)
    parser.add_argument("--target", type=Fraction)
    args = parser.parse_args()
    if args.vehicles < 0 or not isfinite(args.seconds) or args.seconds <= 0:
        parser.error("expected nonnegative vehicles and positive finite seconds")
    instance = json.loads(args.instance.read_text(), parse_float=Fraction)
    model = read_model(instance)
    args.out.mkdir(parents=True, exist_ok=True)
    deadline = START + args.seconds
    key = identity(instance, args.vehicles)

    def emit(upper, parent, screen=None):
        atomically(args.out / "best.json", {
            "method": METHOD, "instance_digest": key, "vehicles": args.vehicles,
            "upper": str(upper), "parent": parent, "screen": screen,
            "emitted_monotonic_ns": monotonic_ns(),
        })

    initial = product_bound(model, tuple(model.boxes(args.vehicles)) if args.vehicles else (), [])[0]
    emit(initial, None)
    if args.warm_route is None or not initial or monotonic() >= deadline:
        return
    parent = json.loads(args.warm_route.read_text())
    upper = validate_parent(model, parent, args.vehicles, deadline)
    if upper is None or monotonic() >= deadline:
        return
    emit(upper, parent)
    if (args.vehicles != 2 or not 0 < model.n <= 128 or not upper
            or not parent["supports"] or monotonic() >= deadline):
        return
    worker = Screen(model.project())
    steps = [Fraction(p, 1000) for p in (1, 3, 10, 20, 35, 50)]
    targets = ([args.target] if args.target is not None else [upper * (1 - step) for step in steps])
    if any(not 0 <= target < upper for target in targets):
        parser.error("target must be nonnegative and strictly below the warm upper")
    best_proved=upper
    failed_floor=None
    for _ in range(16):
        if targets:
            theta=targets.pop(0)
        elif failed_floor is not None and best_proved-failed_floor>upper/20000:
            theta=(best_proved+failed_floor)/2
        else:
            break
        if monotonic() >= deadline:
            break
        indexed = [
            (index, line, threshold_data(model, line, theta))
            for index, line in enumerate(parent["supports"])
        ]
        indexed.sort(key=lambda row: row[1]["route_bound"] - row[2][1])
        completed = False
        for index, line, (integer, cutoff, weights, rewards) in indexed[:3]:
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            local = min(deadline, monotonic() + max(0.5, remaining * 0.7))
            print(f"{monotonic() - START:.3f}s try U={float(theta):.8f} "
                  f"slack={line['route_bound'] - cutoff} line={index}", flush=True)
            result = worker.enumerate(rewards, line["partition"], cutoff, line["reverse"], local)
            if result is None:
                continue
            masks, explored = result
            excluded = excludes_fleet(model, masks, weights, integer, theta, deadline)
            print(f"{monotonic() - START:.3f}s examined={explored} "
                  f"masks={len(masks)} excluded={excluded}", flush=True)
            if excluded and monotonic() < deadline:
                emit(theta, parent, {"line_index": index, "mask_count": len(masks),
                                     "mask_sha256": mask_digest(masks), "explored": explored})
                best_proved=theta
                completed = True
                break
        if not completed:
            if args.target is not None:
                break
            failed_floor=theta
            targets.clear()


if __name__ == "__main__":
    main()
