"""Validate a route and its allocations directly against the H-PDSP data."""

import argparse
import json
from decimal import Decimal
from fractions import Fraction
from itertools import combinations, pairwise
from numbers import Integral
from pathlib import Path


def load_witness(path):
    """Load a witness, preserving any decimal quantity or objective literals."""
    return json.loads(Path(path).read_bytes(), parse_float=Decimal)


def _number(value):
    """Interpret decimal data as written, including solver output and JSON numbers."""
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Expected a finite number, got {value!r}") from exc


def check(instance, route, amounts, *, tol=1e-7):
    """Return exact objective, travel-and-service time, and loads after each stop.

    The depot is implicit at both ends. Route time is checked exactly.
    Set tol=0 to also check all quantities without tolerance.
    A nonempty route and zero initial load are required. Leftovers may be
    carried back to the depot, as permitted by the original formulation.
    """
    tolerance = _number(tol)
    if tolerance < 0:
        raise ValueError("Tolerance must be nonnegative")
    try:
        stops = tuple(route)
        quantities = tuple(map(_number, amounts))
    except TypeError as exc:
        raise ValueError("Expected a route and an allocation sequence") from exc
    if len(stops) < 2 or len(stops) != len(quantities):
        raise ValueError("A nonempty route needs at least two stops and one amount per stop")

    population = tuple(map(_number, instance["population"]))
    supply = tuple(map(_number, instance["supply"]))
    for node in stops:
        if (
            isinstance(node, bool)
            or not isinstance(node, Integral)
            or not 0 < node < len(supply)
        ):
            raise ValueError(f"Invalid nondepot stop {node!r}")
        if (supply[node] > 0) == (population[node] > 0):
            raise ValueError(f"Stop {node} must be an active supplier or delivery agency")
    if len(set(stops)) != len(stops):
        raise ValueError("A route may visit each stop at most once")
    if supply[stops[0]] <= 0 or population[stops[-1]] <= 0:
        raise ValueError("The first stop must be a supplier and the last a delivery agency")

    travel = instance["travel"]
    service = instance["service"]
    elapsed = sum(_number(travel[i][j]) for i, j in pairwise((0, *stops, 0)))
    elapsed += sum(_number(service[i]) for i in stops)
    if elapsed > _number(instance["limit"]):
        raise ValueError(f"Route time {elapsed} exceeds limit {instance['limit']}")

    capacity = _number(instance["capacity"])
    remaining_supply = {i: amount for i, amount in enumerate(supply[1:], 1) if amount > 0}
    deliveries = {i: 0 for i, people in enumerate(population[1:], 1) if people > 0}
    loads = []
    load = Fraction(0)
    for node, amount in zip(stops, quantities):
        if amount < -tolerance:
            raise ValueError(f"Negative amount at stop {node}")
        if node in remaining_supply:
            remaining_supply[node] -= amount
            if remaining_supply[node] < -tolerance:
                raise ValueError(f"Pickup exceeds available supply at stop {node}")
            load += amount
        else:
            if amount > capacity + tolerance:
                raise ValueError(f"Delivery exceeds vehicle capacity at stop {node}")
            deliveries[node] = amount
            load -= amount
        if not -tolerance <= load <= capacity + tolerance:
            raise ValueError(f"Load {load} is outside [0, {capacity}] after stop {node}")
        loads.append(load)

    total = sum(deliveries.values())
    penalty = sum(
        abs(population[i] * deliveries[j] - population[j] * deliveries[i])
        for i, j in combinations(deliveries, 2)
    ) / sum(population[i] for i in deliveries)
    return {
        "objective": total - penalty,
        "time": elapsed,
        "loads": loads,
        "total_delivered": total,
        "return_load": load,
        "remaining_supply": remaining_supply,
    }


def _json_number(value):
    return value.numerator if value.denominator == 1 else str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path, help="Normalized instance JSON")
    parser.add_argument("witness", type=Path, help="Witness JSON with route and amounts")
    parser.add_argument(
        "--tol", default="1e-7",
        help="Absolute tolerance for quantities and the legacy objective field (default: 1e-7). "
             "The route time limit and objective_exact are checked exactly.",
    )
    args = parser.parse_args()

    try:
        instance = json.loads(args.instance.read_bytes(), parse_float=Decimal)
        witness = load_witness(args.witness)
        result = check(instance, witness["route"], witness["amounts"], tol=args.tol)
        if "objective_exact" in witness and _number(witness["objective_exact"]) != result["objective"]:
            raise ValueError(f"Claimed objective_exact differs from {result['objective']}")
        if "objective" in witness:
            error = abs(_number(witness["objective"]) - result["objective"])
            if error > _number(args.tol):
                raise ValueError(f"Claimed objective differs from {result['objective']}")
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"Invalid witness: {exc}\n")
    print(json.dumps(result, default=_json_number))


if __name__ == "__main__":
    main()
