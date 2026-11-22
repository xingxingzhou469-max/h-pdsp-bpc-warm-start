"""Run V8 single-vehicle search, export warm starts, and replay proof reports."""

import argparse
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
import json
from math import isfinite
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic

from single_vehicle.build_native import require_native


def _bound_status(lower, upper):
    if lower is not None and (upper is None or lower > upper):
        raise ValueError("Global upper bound excludes a verified witness")
    if upper is None:
        return "infeasible"
    return "optimal" if lower == upper else "bounded"


def _replay_bounds(resources, input_digest, lower, dual_document):
    from hpdp.cert_verifier import CertificateVerifier

    prior = Fraction(resources.prior, resources.units)
    upper = prior
    method = "analytic_prior"
    if dual_document is not None:
        dual_upper = CertificateVerifier(resources, input_digest).replay(dual_document)
        if dual_upper is None or dual_upper <= prior:
            upper = dual_upper
            method = "routing_dual"
    return dict(status=_bound_status(lower, upper),
                upper_bound_exact=None if upper is None else str(upper),
                analytic_prior_exact=str(prior), method=method, dual_certificate=dual_document)


def verify_proof(payload, document):
    """Independently check the exported feasible route and replayable global bound."""
    from hpdp.check import check
    from hpdp.instances import decode_instance
    from hpdp.prefix import Resources

    instance = decode_instance(payload)
    input_digest = sha256(payload).hexdigest()
    if (type(document) is not dict or type(document.get("version")) is not int
            or document["version"] != 1 or document.get("input_sha256") != input_digest
            or document.get("instance_id") != instance["id"]):
        raise ValueError("Proof version, instance ID or input SHA-256 mismatch")
    witness = document["witness"]
    if witness["input_sha256"] != input_digest or witness["instance_id"] != instance["id"]:
        raise ValueError("Proof witness instance ID or input SHA-256 mismatch")
    if witness["status"] == "no_witness":
        if any(witness[key] is not None for key in ("route", "amounts", "objective_exact")):
            raise ValueError("A no_witness report must have null route, amounts and objective")
        lower = None
    elif witness["status"] == "feasible":
        if not isinstance(witness["route"], list) or not isinstance(witness["amounts"], list):
            raise ValueError("A feasible witness requires route and amount arrays")
        lower = check(instance, witness["route"], witness["amounts"], tol=0)["objective"]
        if witness["objective_exact"] != str(lower):
            raise ValueError("Claimed witness objective differs from exact scoring")
    else:
        raise ValueError("Unknown witness status")
    claimed = document["replayable"]
    replayed = _replay_bounds(Resources(instance), input_digest, lower, claimed["dual_certificate"])
    if replayed != claimed:
        raise ValueError("Claimed replayable result differs from exact replay")
    return {**{key: replayed[key] for key in ("status", "upper_bound_exact", "method")},
            "instance_id": instance["id"], "input_sha256": input_digest,
            "lower_bound_exact": None if lower is None else str(lower)}


def solve_instance(payload, seconds, *, include_proof=False):
    """Return the warm-start document and, if requested, a separate proof report."""
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
    best_dual = None

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

    def collect_dual(bound, document):
        nonlocal best_dual
        if (best_dual is None or (best_dual[0] is not None
                                  and (bound is None or bound < best_dual[0]))):
            best_dual = bound, document

    seed = shared_seed(resources, deadline, insert=True, emit=capture)
    seed = improve_seed(resources, seed, deadline, emit=capture, fixed_work=True)
    search = solve_candidate(resources, deadline, seed, emit=observe,
                             emit_dual=collect_dual if include_proof else None)
    observe(search)

    if witness is None:
        result = dict(status="no_witness", instance_id=instance_id, input_sha256=input_digest,
                      route=None, amounts=None, objective_exact=None)
    else:
        route, amounts = witness
        exact_amounts = [Fraction(str(amount)) for amount in amounts]
        result = dict(status="feasible", instance_id=instance_id, input_sha256=input_digest,
                      route=route, amounts=[amount.numerator if amount.denominator == 1 else str(amount)
                                            for amount in exact_amounts],
                      objective_exact=str(objective))
    if not include_proof:
        return result

    trusted_upper = search["ub_exact"]
    if trusted_upper is None and search["status"] != "infeasible":
        raise ArithmeticError("Search returned no global upper bound or infeasibility ruling")
    replayable = _replay_bounds(resources, input_digest, objective,
                                None if best_dual is None else best_dual[1])
    trusted_status = _bound_status(objective, None if trusted_upper is None else Fraction(trusted_upper))
    proof = dict(version=1, instance_id=instance_id, input_sha256=input_digest, witness=result,
                 replayable=replayable,
                 trusted_search=dict(status=trusted_status, upper_bound_exact=trusted_upper,
                                     report_certificate=search["certificate"]))
    return result, proof


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
    proof_options = parser.add_mutually_exclusive_group()
    proof_options.add_argument("--proof-output", type=Path, help="Export a single-vehicle proof report.")
    proof_options.add_argument("--verify-proof", type=Path, help="Replay a proof report without running V8.")
    args = parser.parse_args()
    if not isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    try:
        paths = {"instance": args.instance, "--output": args.output,
                 "--proof-output": args.proof_output, "--verify-proof": args.verify_proof}
        for (left_name, left), (right_name, right) in combinations(paths.items(), 2):
            if left is not None and right is not None and (
                left.resolve() == right.resolve() or (left.exists() and right.exists() and left.samefile(right))
            ):
                parser.error(f"{left_name} and {right_name} must use different paths")

        payload = args.instance.read_bytes()
        if args.verify_proof is not None:
            document = json.loads(args.verify_proof.read_bytes(), parse_float=Decimal)
            write_result(args.output, verify_proof(payload, document))
            return
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
            os.environ[name] = "1"
        require_native()
        if args.proof_output is None:
            result = solve_instance(payload, args.seconds)
        else:
            result, proof = solve_instance(payload, args.seconds, include_proof=True)
            write_result(args.proof_output, proof)
        write_result(args.output, result)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, ArithmeticError, ImportError) as error:
        parser.exit(1, f"V8 failed: {error}\n")


if __name__ == "__main__":
    main()
