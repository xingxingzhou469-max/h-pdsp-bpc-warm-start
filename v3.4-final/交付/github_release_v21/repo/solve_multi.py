"""Run multi-vehicle H-PDSP v2.1 (exact lower witness + replayable strict upper bound) or paper baselines."""

import argparse
from fractions import Fraction as F
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from time import monotonic, monotonic_ns


ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "multi_vehicle" / "engine"


def _set_single_thread():
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"


def _write_json(path: Path | None, payload: dict):
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    if path is None:
        print(encoded)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(encoded + "\n")
            stream.close()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def solve_v21(instance: dict, input_sha256: str, vehicles: int, seconds: float, seed: int, primal_share: float):
    from multi_vehicle import ensure_engine_on_path
    from multi_vehicle.build_native import require_native

    require_native()
    ensure_engine_on_path()

    import primal as mv_primal
    from common import Data, NANO, verify_lower
    from pricing_core import envelope, library
    from upper import run as run_upper

    # Compile/load native libraries before starting the timed budget.
    native_primal = mv_primal.prepare(None, None, "upgrade")
    library()

    old_seed = mv_primal.SEED
    mv_primal.SEED = int(seed)
    try:
        start_ns = monotonic_ns()
        stop_ns = start_ns + int(seconds * NANO)
        primal_stop_ns = start_ns + int(seconds * primal_share * NANO)

        best_l = F(0)
        best_l_source = "idle_fleet"
        best_witness = {"routes": [], "amounts": []}

        def accept_witness(witness: dict, source: str):
            nonlocal best_l, best_l_source, best_witness
            val = verify_lower(instance, vehicles, witness)
            if val > best_l:
                best_l = val
                best_l_source = source
                best_witness = witness
            return val

        data = Data.from_instance(instance)
        root_cert = envelope(data, vehicles, ())
        best_u = F(root_cert["upper"])
        best_u_source = "input_box"
        best_u_cert = root_cert

        if primal_share > 0:
            mv_primal.run_primal(
                instance,
                vehicles,
                "upgrade",
                primal_stop_ns,
                accept_witness,
                native_primal,
            )

        if primal_share < 1.0 and monotonic_ns() < stop_ns:
            def submit_upper(cert: dict, source: str):
                nonlocal best_u, best_u_source, best_u_cert
                val = F(cert["upper"])
                if (witness := cert.get("maximizing_witness")) is not None:
                    accept_witness(witness, source + "_witness")
                if val < best_u:
                    best_u = val
                    best_u_source = source
                    best_u_cert = cert
                    return True
                return False

            run_upper(instance, vehicles, "joint", stop_ns, submit_upper, fixed_lower=best_l)
    finally:
        mv_primal.SEED = old_seed

    # Final independent exact re-check of L and L <= U.
    checked_l = verify_lower(instance, vehicles, best_witness)
    if checked_l != best_l or checked_l > best_u:
        raise ArithmeticError(f"Exact sanity check failed: L={checked_l}, U={best_u}")

    gap = (best_u - best_l) / best_l if best_l > 0 else (F(0) if best_u == 0 else None)
    status = "strict_optimal" if best_l == best_u else "bounded"
    solution_doc = {
        "method": "v2.1",
        "status": status,
        "input_sha256": input_sha256,
        "vehicles": vehicles,
        "score": str(best_l),
        "source": best_l_source,
        "upper_bound_exact": str(best_u),
        "upper_method": best_u_cert["method"],
        "gap_exact": None if gap is None else str(gap),
        "gap_pct": None if gap is None else round(float(gap) * 100, 3),
        "certificate": best_witness,
    }
    proof_doc = {
        "kind": "U",
        "input_sha256": input_sha256,
        "vehicles": vehicles,
        "score": str(best_u),
        "source": best_u_source,
        "certificate": best_u_cert,
    }
    return solution_doc, proof_doc


def solve_baseline(method: str, instance: dict, input_sha256: str, vehicles: int, seconds: float, seed: int):
    from multi_vehicle import ensure_engine_on_path

    ensure_engine_on_path()
    from common import verify_lower
    from primal_vendor.fleet import canonical_column

    if method == "site":
        from multi_vehicle.baselines import site_milp as mod
    elif method == "mv_lns":
        from multi_vehicle.baselines import mv_lns as mod
    elif method == "segment":
        from multi_vehicle.baselines import segment as mod
    else:
        raise ValueError(f"unknown baseline method: {method!r}")

    deadline = monotonic() + seconds
    best_l = F(0)
    best_source = "idle_fleet"
    best_witness = {"routes": [], "amounts": []}
    best_numeric_u = None

    class _Publisher:
        def __call__(self, routes, source):
            nonlocal best_l, best_source, best_witness
            if monotonic() >= deadline:
                return None
            clean_routes = [list(route) for route in routes if route]
            try:
                amounts = [list(map(str, canonical_column(instance, route)[1])) for route in clean_routes]
                witness = {"routes": clean_routes, "amounts": amounts}
                val = verify_lower(instance, vehicles, witness)
            except (ValueError, TypeError, IndexError):
                return None
            if val > best_l:
                best_l = val
                best_source = source
                best_witness = witness
            return val

        def numeric(self, upper, _source, **_meta):
            nonlocal best_numeric_u
            if isfinite(upper) and upper >= 0:
                if best_numeric_u is None or upper < best_numeric_u:
                    best_numeric_u = float(upper)

    mod.solve(instance, vehicles, int(seed), deadline, _Publisher())
    checked_l = verify_lower(instance, vehicles, best_witness)
    if checked_l != best_l:
        raise ArithmeticError("Baseline witness disagreed with exact verifier")

    solution_doc = {
        "method": method,
        "status": "feasible",
        "input_sha256": input_sha256,
        "vehicles": vehicles,
        "score": str(best_l),
        "source": best_source,
        "numeric_upper_bound": best_numeric_u,
        "certificate": best_witness,
    }
    return solution_doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path, help="Canonical multi-vehicle H-PDSP JSON.")
    parser.add_argument(
        "--vehicles",
        type=int,
        help="Fleet size K (defaults to instance['vehicles']).",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=60.0,
        help="Total wall-clock budget in seconds (default: 60).",
    )
    parser.add_argument(
        "--method",
        choices=("v2.1", "site", "mv_lns", "segment"),
        default="v2.1",
        help="Solver method (default: v2.1).",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed (default: 7).")
    parser.add_argument(
        "--primal-share",
        type=float,
        default=0.5,
        help="Fraction of budget allocated to primal search in v2.1 (default: 0.5; 0..1).",
    )
    parser.add_argument("--output", type=Path, help="Output solution JSON path (defaults to stdout).")
    parser.add_argument(
        "--proof-output",
        type=Path,
        help="Export independently replayable strict upper-bound proof JSON (v2.1 only).",
    )
    args = parser.parse_args()

    if not isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    if not (0.0 <= args.primal_share <= 1.0):
        parser.error("--primal-share must lie in [0, 1]")
    if args.proof_output is not None and args.method != "v2.1":
        parser.error("--proof-output is only supported for --method v2.1")

    _set_single_thread()
    raw_bytes = args.instance.read_bytes()
    input_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    instance = json.loads(raw_bytes, parse_float=F)
    vehicles = args.vehicles if args.vehicles is not None else instance.get("vehicles")
    if type(vehicles) is not int or vehicles < 1:
        parser.error("Provide --vehicles K (>= 1) or an instance JSON containing integer 'vehicles'")

    if args.method == "v2.1":
        solution, proof = solve_v21(
            instance, input_sha256, vehicles, args.seconds, args.seed, args.primal_share
        )
        if args.proof_output is not None:
            _write_json(args.proof_output, proof)
    else:
        solution = solve_baseline(
            args.method, instance, input_sha256, vehicles, args.seconds, args.seed
        )
    _write_json(args.output, solution)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
