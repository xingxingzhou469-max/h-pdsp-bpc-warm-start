"""Recheck a single frozen non-Latet benchmark upper bound without calling V8."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from warm_start import load_start
from proofs.closure.verify_blind05 import verify as tree_bound


def require(condition, message):
    if not condition:
        raise ValueError(message)


def native_binary(method, build_dir):
    source = HERE / (method + ".cpp")
    inputs = [source, HERE / "reverse.cpp"] if method == "window" else [source]
    fingerprint = hashlib.sha256(b"".join(p.read_bytes() for p in inputs)).hexdigest()
    directory = Path(build_dir)
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / (method + "-" + fingerprint[:16])
    if not binary.exists():
        with tempfile.TemporaryDirectory(dir=directory) as scratch:
            temporary = Path(scratch) / method
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O3",
                            str(source), "-o", str(temporary)], check=True)
            os.replace(temporary, binary)
    return binary


def verify(instance_path, solution_path, *, build_dir=None, seconds=120):
    """Return a verified rational upper bound; incomplete searches fail closed.

    build_dir may be reused across calls to cache the locally compiled binaries.
    """
    if build_dir is None:
        with tempfile.TemporaryDirectory() as directory:
            return verify(instance_path, solution_path, build_dir=directory, seconds=seconds)
    require(math.isfinite(seconds) and seconds > 0, "seconds must be finite and positive")
    instance_path, solution_path = Path(instance_path), Path(solution_path)
    model, seed = load_start(instance_path, solution_path, vehicles=1)
    name = model["id"]
    value = Fraction(seed["objective_exact"])
    stock, pop, service = (model[k] for k in ("supply", "population", "service"))
    n, capacity, limit = len(pop), model["capacity"], model["limit"]
    total_stock = sum(min(x, capacity) for x in stock)
    ceiling = min(total_stock, sum(x > 0 for x in pop) * capacity)
    base = dict(instance_id=name, input_sha256=seed["input_sha256"],
                solution_sha256=hashlib.sha256(solution_path.read_bytes()).hexdigest(), lb_exact=str(value))
    if ceiling == value:
        return dict(base, method="total_supply_capacity", ub_exact=str(ceiling))
    if name == "blind-05":
        proof = tree_bound(instance_path, HERE / "blind-05-compact-certificate.json")
        upper = Fraction(proof["strict_optimum"])
        require(upper == value, "Stored tree threshold differs from the feasible witness")
        return dict(base, method="omission_one_tree", ub_exact=str(upper),
                    checked_time_sets=len(proof["checked_time_certificates"]))

    # The native procedures below use fixed-width integer arithmetic. The
    # registered SHA list limits them to the audited integer benchmark inputs.
    registered = json.loads((HERE / "integer_inputs.json").read_text())
    require(registered.get(name) == seed["input_sha256"], "No registered integer proof for input")
    goal = value * sum(pop)
    require(goal.denominator == 1, "Witness optimum numerator must be integral")
    goal = goal.numerator
    if n <= 25:
        method = "visit_sets"
        rows = [(n - 1, capacity, limit, goal), *zip(stock, pop, service), *model["travel"]]
        args = []
    else:
        method = "window" if n <= 64 else "reverse"
        route = seed["routes"][0]
        weight = sum(pop[v] for v in route)
        delivered = sum(Fraction(q) for v, q in zip(route, seed["amounts"][0]) if pop[v])
        require(delivered.denominator == 1 and weight > 0 and delivered > 0,
                "Expected positive integral pricing weights")
        rows = [(n, capacity, limit, goal, weight, delivered), stock, pop, service, *model["travel"]]
        args = [str(seconds), "8", "5"] if method == "window" else [str(seconds), "4"]
        if name == "blind-09":
            prices = json.loads((HERE / "blind-09-prices.json").read_text())
            rows.append([len(prices)])
            for table in prices:
                tracked, multipliers = table["tracked"], table["visit_prices"]
                require(len(multipliers) == n - 1, "Missing visit prices")
                rows.append([table["supply_weight"], table["population_weight"], len(tracked),
                             *tracked, *multipliers])
            args.append("prices")

    stream = "\n".join(" ".join(map(str, row)) for row in rows) + "\n"
    process = subprocess.run([str(native_binary(method, build_dir)), *args], input=stream,
                             text=True, capture_output=True, timeout=seconds + 2)
    counts = dict(words for line in process.stdout.splitlines() if len(words := line.split()) == 2)
    require(process.returncode == 0,
            f"{name} search did not close (exit {process.returncode}, {counts.get('status', '')}): "
            + process.stderr[-400:])
    if method == "visit_sets":
        exclusions = ["time_filter", "commodity_filter", "water_filter", "infeasible_sets", "residual_sets"]
        require(int(counts["all_sets"]) == (1 << (n - 1)) - 1 == sum(int(counts[k]) for k in exclusions)
                and int(counts["max_scored_numerator"]) <= goal, "Incomplete subset coverage")
        evidence = dict(checked_sets=int(counts["all_sets"]), residual_sets=int(counts["residual_sets"]),
                        complete_orders_scored=int(counts["scored"]))
    else:
        require(counts["status"] == "strict_upper_bound" and int(counts["unresolved_roots"]) == 0
                and int(counts["goal"]) == goal == int(counts["upper"]), "Incomplete reverse route cover")
        expected_roots = sum(p > 0 and model["travel"][v][0] + service[v] <= limit
                             for v, p in enumerate(pop))
        require(int(counts["closed_roots"]) == expected_roots, "Missing a reachable last-agency root")
        if method == "window":
            require(int(counts["unresolved_tasks"]) == 0
                    and int(counts["closed_tasks"]) == int(counts["formed_tasks"]), "Open reverse tasks")
        evidence = dict(closed_roots=int(counts["closed_roots"]), expanded=int(counts["expanded"]),
                        complete_orders_scored=int(counts["route_audits"]), unresolved_roots=0)
        if method == "window":
            evidence.update(closed_tasks=int(counts["closed_tasks"]), unresolved_tasks=0)
    return dict(base, method=method, ub_exact=str(value), **evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("solution", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--seconds", type=float, default=120)
    args = parser.parse_args()
    print(json.dumps(verify(args.instance, args.solution, build_dir=args.build_dir, seconds=args.seconds)))


if __name__ == "__main__":
    main()
