"""Recheck the normalized Latet 2708 solution and its exhaustive upper bound."""

from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
from itertools import combinations, pairwise
import json
from math import lcm
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUT_SHA256 = "563596f6f1cc32d7fd6bcab6568dbad8718b6dc05e196cb5d15038333d5b5338"
VALUE = Fraction(53628831, 3266)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_witness(model, witness):
    route = witness["route"]
    amounts = list(map(Fraction, witness["amounts"]))
    supply = list(map(Fraction, model["supply"]))
    people = model["population"]
    require(witness["instance_id"] == model["id"]
            and witness["input_sha256"] == INPUT_SHA256, "Witness identity changed")
    require(len(route) == len(amounts) == len(set(route)) and route
            and all(type(v) is int and 0 < v < len(people) for v in route),
            "Invalid witness route")
    require(supply[route[0]] > 0 and people[route[-1]] > 0, "Invalid witness endpoints")
    elapsed = sum(model["travel"][u][v] + model["service"][v]
                  for u, v in pairwise([0, *route, 0]))
    require(elapsed <= model["limit"], "Witness exceeds time limit")
    load = Fraction(0)
    deliveries = dict.fromkeys((v for v, p in enumerate(people) if p), Fraction(0))
    for v, amount in zip(route, amounts, strict=True):
        require(amount >= 0, "Negative amount")
        if supply[v] > 0:
            require(amount <= supply[v], "Witness exceeds supplier stock")
            load += amount
        else:
            deliveries[v] = amount
            load -= amount
        require(0 <= load <= model["capacity"], "Invalid witness vehicle load")
    delivered = sum(deliveries.values())
    penalty = sum(abs(people[u] * deliveries[v] - people[v] * deliveries[u])
                  for u, v in combinations(deliveries, 2)) / sum(people)
    require(delivered - penalty == Fraction(witness["objective_exact"]) == VALUE,
            "Exact witness objective differs")
    return {"time": elapsed, "delivered": str(delivered), "objective_exact": str(VALUE)}


def verify_upper(model, executable):
    raw_supply = list(map(Fraction, model["supply"]))
    capacity = Fraction(model["capacity"])
    scale = lcm(capacity.denominator, *(v.denominator for v in raw_supply))
    stock = [int(min(q, capacity) * scale) for q in raw_supply]
    people = model["population"]
    donors = tuple(v for v, q in enumerate(stock) if q)
    require(donors == (1, 2, 3, 4, 5), "Unexpected supplier roles")
    limit = model["limit"]
    duration = [[min(limit + 1, t + model["service"][v]) for v, t in enumerate(row)]
                for row in model["travel"]]
    require(all(type(t) is int for row in duration for t in row), "Expected integer time")
    threshold = (VALUE * sum(people) * scale).__floor__()

    # Population order and a nonnegative price vector selected from the input.
    # Soundness only requires that each price lie in [0, population].
    agencies = sorted((v for v, p in enumerate(people) if p), key=lambda v: (-people[v], v))
    tracked = agencies[:3]
    pivot = people[agencies[len(agencies) // 4]]
    prices = [0 if v in tracked else max(0, p - pivot) for v, p in enumerate(people)]
    contexts = []
    for mask in range(1, 1 << len(donors)):
        selected = [v for v in donors if mask & (1 << (v - 1))]
        total_stock = sum(stock[v] for v in selected)
        for first in selected:
            context = {"supplier_mask": mask, "first_supplier": first}
            if total_stock * sum(people) <= threshold:
                context["method"] = "supplier stock"
            else:
                header = [len(people), limit, first, mask, len(tracked), int(capacity * scale), threshold]
                lines = [header, people, prices, stock, *duration, tracked]
                payload = "\n".join(" ".join(map(str, line)) for line in lines) + "\n"
                result = subprocess.run([str(executable)], input=payload, text=True,
                                        check=True, capture_output=True)
                population, states, prefixes, unresolved = map(int, result.stdout.split())
                require(unresolved == 0, f"Upper bound still open for mask {mask}, first {first}")
                context.update(method="population and elementary prefix bounds", population_upper=population,
                               tail_states=states, expanded_prefixes=prefixes,
                               unresolved_terminals=unresolved)
            contexts.append(context)
    require(len(contexts) == len(donors) * (1 << (len(donors) - 1)), "Incomplete supplier contexts")
    return contexts


def main():
    raw = (ROOT / "instances/latet-2708.json").read_bytes()
    require(sha256(raw).hexdigest() == INPUT_SHA256, "Published input SHA-256 differs")
    model = json.loads(raw, parse_float=Decimal)
    witness = check_witness(model, json.loads((ROOT / "solutions/latet-2708.json").read_bytes()))
    with TemporaryDirectory(prefix="hpdp-latet2708-") as directory:
        executable = Path(directory) / "check"
        subprocess.run([os.environ.get("CXX", "c++"), "-std=c++20", "-O3", "-Wall", "-Wextra",
                        "-Wpedantic", str(HERE / "check.cpp"), "-o", str(executable)], check=True)
        contexts = verify_upper(model, executable)
    print(json.dumps({"instance_id": model["id"], "input_sha256": INPUT_SHA256,
                      "lower_exact": str(VALUE), "upper_exact": str(VALUE), "witness": witness,
                      "verified": True, "contexts": contexts}, indent=2))


if __name__ == "__main__":
    main()
