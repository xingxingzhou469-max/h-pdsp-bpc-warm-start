"""Read a single-vehicle witness and start the fleet H-PDSP solver from it."""

import argparse
import hashlib
import json
import math
import os
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from hpdp.check import check
from hpdp.instances import decode_instance
from mvp.fleet import canonical_column, json_number, verify_fleet


def load_start(instance_path, solution_path, *, vehicles):
    """Verify the original bytes, quantities and objective. Return instance and fleet start."""
    if type(vehicles) is not int or vehicles < 1:
        raise ValueError("vehicles must be a positive integer")
    raw = Path(instance_path).read_bytes()
    instance = decode_instance(raw)
    witness = json.loads(Path(solution_path).read_bytes(), parse_float=Decimal)
    fingerprint = hashlib.sha256(raw).hexdigest()
    if witness["instance_id"] != instance["id"] or witness["input_sha256"] != fingerprint:
        raise ValueError("the witness belongs to a different instance ID or input SHA-256")
    if witness.get("status") == "no_witness" or witness["route"] is None:
        raise ValueError("the single-vehicle run produced no feasible witness")
    if not isinstance(witness["route"], list) or not witness["route"]:
        raise ValueError("a single-vehicle witness needs a nonempty route")
    if not isinstance(witness["amounts"], list):
        raise ValueError("the witness must provide one amount per stop in a JSON array")
    single = check(instance, witness["route"], witness["amounts"], tol=0)
    report = verify_fleet(instance, [witness["route"]], [witness["amounts"]],
                          vehicles=vehicles, tol=0)
    if single["objective"] != report["objective"]:
        raise ArithmeticError("single-vehicle and fleet scoring disagree")
    if report["objective"] != Fraction(str(witness["objective_exact"])):
        raise ValueError("recorded objective differs from the exact fleet objective")
    detail = report["routes"][0]
    start = {
        "instance_id": instance["id"],
        "input_sha256": fingerprint,
        "vehicles": vehicles,
        "used_vehicles": report["used_vehicles"],
        "routes": [detail["route"]],
        "amounts": [detail["amounts"]],
        "objective_exact": str(report["objective"]),
        "route_time_exact": str(detail["time"]),
    }
    return instance, start


def run_bpc(instance, start, seconds):
    """Run pricing and the restricted master, then replay the returned fleet exactly."""
    from mvp.bpc import BpcSolver

    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("the BPC time budget must be finite and positive")
    solver = BpcSolver(instance, start["vehicles"])
    routes, _, record = solver.solve(seconds=seconds, initial_routes=start["routes"])
    routes = [list(route) for route in routes if route]
    amounts = [[json_number(value) for value in canonical_column(instance, route)[1]]
               for route in routes]
    checked = verify_fleet(instance, routes, amounts, vehicles=start["vehicles"], tol=0)
    if checked["objective"] < Fraction(start["objective_exact"]):
        raise ArithmeticError("the BPC incumbent fell below its verified initial solution")
    master = record.get("master") or {}
    return {
        "initial": start,
        "final": {"routes": routes, "amounts": amounts,
                  "used_vehicles": checked["used_vehicles"],
                  "objective_exact": str(checked["objective"])},
        "pool_routes": record["pool_routes"],
        "master_status": master.get("status"),
        "mip_start": master.get("mip_start"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("solution", type=Path)
    parser.add_argument("--vehicles", type=int, required=True)
    parser.add_argument("--run-seconds", type=float, help="also run the fleet BPC heuristic")
    args = parser.parse_args()
    try:
        instance, start = load_start(args.instance, args.solution, vehicles=args.vehicles)
        result = start
        if args.run_seconds is not None:
            # HiGHS can write its startup banner directly to fd 1 before logging is configurable.
            sys.stdout.flush()
            with os.fdopen(os.dup(sys.stdout.fileno()), "w") as original_stdout:
                try:
                    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
                    result = run_bpc(instance, start, args.run_seconds)
                finally:
                    sys.stdout.flush()
                    os.dup2(original_stdout.fileno(), sys.stdout.fileno())
    except (KeyError, OSError, TypeError, ValueError, ZeroDivisionError) as exc:
        parser.exit(1, f"Invalid warm start: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
