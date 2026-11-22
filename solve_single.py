"""Run frozen V8's single-vehicle search and export a verified warm-start witness."""

import argparse
from fractions import Fraction
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic

from single_vehicle.build_native import require_native


def solve_instance(payload, seconds):
    from hpdp.check import check
    from hpdp.heuristic import shared_seed
    from hpdp.instances import decode_instance
    from hpdp.neighborhood import improve_seed
    from hpdp.overnight_candidate import solve_candidate
    from hpdp.prefix import Resources

    start = monotonic()
    hard_deadline = start + seconds
    deadline = hard_deadline - min(0.45, 0.08 * seconds)
    instance = decode_instance(payload)
    instance_id = instance["id"]
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("Instance id must be a nonempty string")
    input_digest = sha256(payload).hexdigest()
    resources = Resources(instance)
    resources.input_sha256 = input_digest

    witness = None
    objective = None

    def capture(candidate, value):
        nonlocal witness, objective
        if monotonic() > hard_deadline or (objective is not None and value <= objective):
            return
        route, amounts = list(candidate["route"]), list(candidate["amounts"])
        exact = check(instance, route, amounts, tol=0)["objective"]
        if exact != value:
            raise ArithmeticError("V8 witness disagrees with exact independent scoring")
        if monotonic() <= hard_deadline:
            witness = route, amounts
            objective = exact

    def observe(row):
        if row["lb_exact"] is not None:
            capture(row, Fraction(row["lb_exact"]))

    seed = shared_seed(resources, deadline, insert=True, emit=capture)
    seed = improve_seed(resources, seed, deadline, emit=capture, fixed_work=True)
    observe(solve_candidate(resources, deadline, seed, emit=observe))

    if witness is None:
        return dict(status="no_witness", instance_id=instance_id, input_sha256=input_digest,
                    route=None, amounts=None, objective_exact=None)
    route, amounts = witness
    exact_amounts = [Fraction(str(amount)) for amount in amounts]
    return dict(status="feasible", instance_id=instance_id, input_sha256=input_digest,
                route=route, amounts=[amount.numerator if amount.denominator == 1 else str(amount)
                                      for amount in exact_amounts],
                objective_exact=str(objective))


def write_result(output, document):
    encoded = json.dumps(document, ensure_ascii=False, allow_nan=False)
    if output is None:
        print(encoded)
        return
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(encoded + "\n")
            stream.close()
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path, help="Normalized single-vehicle JSON")
    parser.add_argument("--seconds", type=float, default=60, help="Search budget in seconds (default: 60)")
    parser.add_argument("--output", type=Path, help="Output JSON path. Defaults to stdout.")
    args = parser.parse_args()
    if not isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    if args.output is not None and args.output.resolve() == args.instance.resolve():
        parser.error("--output must differ from the input path")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        require_native()
        result = solve_instance(args.instance.read_bytes(), args.seconds)
        write_result(args.output, result)
    except (OSError, RuntimeError, ValueError, KeyError, ArithmeticError, ImportError) as error:
        parser.exit(1, f"V8 failed: {error}\n")


if __name__ == "__main__":
    main()
