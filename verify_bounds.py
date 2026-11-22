"""Verify archived single-vehicle bounds from the published inputs and witnesses."""
import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from warm_start import load_start

ROOT = Path(__file__).resolve().parent


def verify(row, build):
    name = row["instance_id"]
    model = ROOT / row["input_file"]
    solution = ROOT / row["solution_file"]
    _, start = load_start(model, solution, vehicles=1)
    lower = Fraction(start["objective_exact"])
    if start["input_sha256"] != row["input_sha256"] or lower != Fraction(row["objective_exact"]):
        raise ValueError(f"{name}: lower bound or input differs from the index")
    method = row["upper_proof_kind"]
    if method == "integer_dual_frontiers":
        from proofs.open_bounds.replay import replay
        upper = Fraction(replay(name)["upper_exact"])
    elif method == "integer_dual":
        from hpdp.cert_verifier import verify_certificate
        upper = verify_certificate(model.read_bytes(), json.loads((ROOT / row["upper_proof_file"]).read_bytes()))
    elif method == "latet_2708_supply_partition":
        result = subprocess.run([sys.executable, str(ROOT / row["upper_proof_file"])],
                                check=True, capture_output=True, text=True)
        upper = Fraction(json.loads(result.stdout)["upper_exact"])
    elif method == "latet_supplier_contexts":
        from proofs.latet.replay import replay
        upper = replay(name)
    elif method == "independent_integer_check":
        from proofs.closure.verify import verify as verify_closure
        upper = Fraction(verify_closure(model, solution, build_dir=build)["ub_exact"])
    else:
        raise ValueError(f"{name}: unknown upper-bound method")
    if upper is None or upper < lower or upper != Fraction(row["upper_bound_exact"]):
        raise ValueError(f"{name}: replay disagrees with the archived interval")
    status = "strict_optimal" if upper == lower else "feasible"
    if status != row["single_vehicle_status"]:
        raise ValueError(f"{name}: replay disagrees with the archived status")
    return dict(instance_id=name, lower_exact=str(lower), upper_exact=str(upper),
                status=status, method=method)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instances", nargs="*", help="instance IDs; by default check the entire index")
    args = parser.parse_args()
    with (ROOT / "index.csv").open(encoding="utf-8", newline="") as stream:
        entries = list(csv.DictReader(stream))
    rows = {row["instance_id"]: row for row in entries}
    if len(rows) != 60 or len(entries) != len(rows):
        parser.error("the release index must contain 60 distinct instances")
    selected = args.instances or list(rows)
    if set(selected) - rows.keys():
        parser.error("unknown instance in the requested list")
    with tempfile.TemporaryDirectory() as build:
        results = []
        for name in selected:
            result = verify(rows[name], build)
            results.append(result)
            print(json.dumps(result), flush=True)
    optimal = sum(r["status"] == "strict_optimal" for r in results)
    print(f"{len(results)} intervals verified; {optimal} closed; {len(results)-optimal} open", file=sys.stderr)


if __name__ == "__main__":
    main()
