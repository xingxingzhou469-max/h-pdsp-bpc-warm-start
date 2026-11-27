"""Column generation and anytime elementary pricing for a strict fleet bound."""

from time import monotonic, monotonic_ns

START = monotonic()

import argparse
from collections import Counter
import ctypes
from fractions import Fraction
import json
from math import isfinite
import os
from pathlib import Path

if __package__:
    from .model import product_bound, read_model
else:
    from model import product_bound, read_model


def atomically(path, certificate):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(certificate, stream, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Pricer:
    def __init__(self, graph):
        library = ctypes.CDLL(str(Path(__file__).with_name("route_price.so")))
        self.function = library.route_price
        int_t, int64_t = ctypes.c_int, ctypes.c_int64
        ptr = ctypes.POINTER
        self.function.argtypes = [
            int_t, int_t, int_t, int_t, int64_t, ptr(int_t), ptr(int_t), ptr(int_t), ptr(int64_t),
            ptr(int64_t), ptr(int_t), ptr(int_t), ptr(int_t),
        ]
        self.function.restype = int_t
        self.graphs = []
        n = self.n = len(graph.start)
        self.horizon = horizon = graph.horizon
        self.layered = int(graph.layered)
        self.width = width = n if self.layered else horizon + 1
        for orientation in (graph, graph.reversed()):
            self.graphs.append((
                (int_t * (n * n))(*(cost for row in orientation.arcs for cost in row)),
                (int_t * n)(*orientation.start),
                (int_t * n)(*orientation.finish),
            ))
        self.lengths = (int_t * (n + 1))()
        self.paths = (int_t * ((n + 1) * width))()

    def solve(self, reward, deadline, reverse=False, state_limit=0):
        if max(map(abs, reward), default=0) * (self.width + 2) >= 2**59:
            raise OverflowError("pricing rewards exceed the native integer range")
        remaining = int((deadline - monotonic()) * 1e9)
        if remaining <= 0:
            return None
        records, expanded = ctypes.c_int64(), ctypes.c_int()
        status = self.function(
            self.n, self.horizon, self.layered, state_limit, remaining,
            *self.graphs[reverse], (ctypes.c_int64 * self.n)(*reward),
            ctypes.byref(records), self.lengths, self.paths, ctypes.byref(expanded),
        )
        if status in (2, 3):
            return None
        if status != 0:
            raise RuntimeError(f"unexpected pricing status {status}")
        paths = []
        for i, length in enumerate(self.lengths):
            if length:
                path = tuple(self.paths[i * self.width:i * self.width + length])
                paths.append(path[::-1] if reverse else path)
        return records.value, paths


class RestrictedMaster:
    """Dual to the station- and fleet-constrained route packing LP."""

    def __init__(self, highspy, np, weights, vehicles, elementary):
        self.highspy = highspy
        self.solver = problem = highspy.Highs()
        self.weights = weights
        self.n = len(weights)
        self.elementary = elementary
        self.keys = set()
        problem.setOptionValue("output_flag", False)
        problem.setOptionValue("threads", 1)
        problem.setOptionValue("solver", "simplex")
        cols = self.n + 1
        problem.addCols(cols, [1.] * self.n + [float(vehicles)],
                        np.zeros(cols), np.full(cols, highspy.kHighsInf),
                        0, [], [], [])

    def add(self, paths):
        starts, indexes, values, bounds = [], [], [], []
        for route in paths:
            counts = Counter(route)
            if self.elementary and len(counts) != len(route):
                continue
            key = tuple(sorted(counts.items()))
            if key in self.keys:
                continue
            self.keys.add(key)
            starts.append(len(indexes))
            for i, count in key:
                indexes.append(i)
                values.append(float(count))
            indexes.append(self.n)
            values.append(1.)
            bounds.append(float(sum(self.weights[i] * count for i, count in key)))
        if bounds:
            self.solver.addRows(len(bounds), bounds, [self.highspy.kHighsInf] * len(bounds),
                                len(indexes), starts, indexes, values)
        return len(bounds)

    def prices(self, deadline):
        remaining = (deadline - monotonic())
        if remaining <= 0 or not self.keys:
            return None
        self.solver.setOptionValue("time_limit", remaining)
        if self.solver.run() != self.highspy.HighsStatus.kOk:
            return None
        if self.solver.getModelStatus() != self.highspy.HighsModelStatus.kOptimal:
            return None
        return tuple(max(0, round(x)) for x in self.solver.getSolution().col_value[:-1])


def solve(instance, k, out, deadline):
    model = read_model(instance)
    boxes = tuple(model.boxes(k)) if k else ()
    best = None
    point = (Fraction(0), Fraction(0))
    lines = {}

    def publish():
        nonlocal best, point
        rhs = [(beta, line[0]) for beta, line in lines.items()]
        upper, point = product_bound(model, boxes, rhs)
        if (best is None or upper < best) and monotonic() < deadline:
            atomically(out / "best.json", {
                "method": "route-price-v1",
                "vehicles": k,
                "upper": str(upper),
                "supports": [line[1] for line in lines.values()],
                "emitted_monotonic_ns": monotonic_ns(),
            })
            best = upper

    publish()
    if not best or model.n > 256 or monotonic() >= deadline:
        return

    # These dependencies are used to select prices. Proof replay needs neither.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "HIGHS_NUM_THREADS"):
        os.environ[name] = "1"
    import highspy
    import numpy as np

    graph = model.project()
    if (model.n * (graph.horizon + 1) * (model.n if graph.layered else 1) > 7_500_000):
        return
    pricer = Pricer(graph)
    pool = {}

    def phase(beta, center, stop, state_limit=0, iterations=28):
        elementary = bool(state_limit)
        mode = "elementary" if elementary else "walk"
        weights = model.weights(beta)
        master = RestrictedMaster(highspy, np, weights, k, elementary)
        master.add(pool)
        prices = center
        seen = set()
        for step in range(iterations):
            if monotonic() >= stop:
                break
            reverse = bool(step % 2)
            key = (prices, reverse)
            if key in seen:
                break
            seen.add(key)
            reward = tuple(w - p for w, p in zip(weights, prices))
            result = pricer.solve(reward, stop, reverse, state_limit)
            if result is None:
                break
            ceiling, paths = result
            rhs = model.reward_unit * (sum(prices) + k * ceiling)
            old = lines.get(beta)
            if old is None or rhs < old[0]:
                lines[beta] = (rhs, {
                    "beta": str(beta),
                    "prices": prices,
                    "route_bound": ceiling,
                    "mode": mode,
                    "reverse": reverse,
                })
                center = prices
                publish()
            added = master.add(paths)
            for path in paths:
                pool[path] = None
            hint = master.prices(stop)
            if hint is None:
                break
            if not added or step % 4 == 3:
                prices = hint
            elif elementary:
                prices = tuple((x + 3 * c + 2) // 4 for x, c in zip(hint, center))
            else:
                prices = tuple((3 * x + c + 2) // 4 for x, c in zip(hint, center))
        return center

    work = max(0, deadline - monotonic())
    time_buffer = min(0.03, work / 20)
    stop = deadline - time_buffer
    ref_beta = model.quantity / model.population
    initial = ref_beta * Fraction(3, 2)
    centers = {}

    def beta_at_point():
        if not all(point):
            return initial
        ratio = point[0] / point[1]
        return ratio.limit_denominator(4096) or ratio

    def short_slice(seconds):
        return min(stop, monotonic() + seconds)

    for factor in (None, Fraction(2, 3), Fraction(4, 3)):
        beta = beta_at_point() if factor is None else beta_at_point() * factor
        if not beta or beta in centers:
            beta = initial
        if beta in centers:
            continue
        if centers:
            neighbor = min(centers, key=lambda a: abs(a - beta))
            seed = tuple(round(p * max(1, beta / neighbor)) for p in centers[neighbor])
        else:
            seed = (0,) * model.n
        centers[beta] = phase(beta, seed, short_slice(work * .15))

    # Refine the directions actually limiting the product, and learn prices
    # on distinct-station columns once the full-route frontier is available.
    if not lines:
        return
    for level, share in ((45_000, .08), (180_000, .13), (480_000, .24)):
        if monotonic() >= stop:
            break
        target = beta_at_point()
        assert target > 0
        order = sorted(lines, key=lambda beta: (
            abs(lines[beta][0] - point[0] - beta * point[1]), abs(beta - target)))
        for beta in order[:2]:
            centers[beta] = phase(beta, centers[beta], short_slice(work * share), level, 5)

    beta = beta_at_point()
    if beta > 0 and beta not in lines and monotonic() < stop:
        neighbor = min(lines, key=lambda a: abs(a - beta))
        center = phase(beta, centers[neighbor], short_slice(work * .12))
        phase(beta, center, stop, 480_000, 8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--vehicles", type=int, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.vehicles < 0 or not isfinite(args.seconds) or args.seconds <= 0:
        parser.error("vehicles must be nonnegative and seconds must be finite and positive")
    args.out.mkdir(parents=True, exist_ok=True)
    instance = json.loads(args.instance.read_text(), parse_float=Fraction)
    solve(instance, args.vehicles, args.out, START + args.seconds)


if __name__ == "__main__":
    main()
