"""Independent pure-stdlib verifier for multi-vehicle H-PDSP v2.1 solutions and strict upper bounds.

Run under `python3 -B -S verify_multi.py` without numpy, scipy, highspy, or compiled C++ code.
Every input SHA-256, feasible fleet witness L, and strict upper-bound proof U is re-derived
from the raw instance JSON using exact rational arithmetic (fractions.Fraction).
"""

import argparse
import csv
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
from time import monotonic


ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "multi_vehicle" / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from common import verify_lower  # noqa: E402
from screen import METHOD as SCREEN, screen  # noqa: E402
from supplier_split import METHOD as SPLIT, replay_split  # noqa: E402
from tree import FAIR, TREE, replay_new  # noqa: E402
from verify import replay  # noqa: E402


FORBIDDEN_MODULES = frozenset({
    "numpy",
    "scipy",
    "highspy",
    "upper",
    "pricing_core",
    "primal",
    "ctypes",
})

# Fast curated subset covering all scales (19/40/60/100/Latet), all 4 strict U proof
# mechanisms (exact-fleet-route-screen-v3, ng-product-v1, active-supplier-cover-v2,
# visit-branch-v1), and multi-method formal replay.
SAMPLE_STAGE_B_IDS = (
    "r1-strA-19-k2",
    "r1-strA-40-k2",
    "r1-lgA-60-k2",
    "r2-lgB-100-k3",
    "latet-1908-k2",
    "latet-2809-k3",
)
SAMPLE_FORMAL_IDS = ("strB-19-k2",)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_bytes(), parse_float=F)


def need(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def replay_upper_certificate(instance: dict, vehicles: int, certificate: dict):
    """Independently replay a strict multi-vehicle upper-bound certificate."""
    method = certificate["method"]
    if method == "ng-product-v1":
        return replay(instance, vehicles, certificate), None
    if method == SPLIT:
        return replay_split(instance, vehicles, certificate), None
    if method in (FAIR, TREE):
        return replay_new(instance, vehicles, certificate), None
    if method == SCREEN:
        need(
            certificate["required"] == certificate["forbidden"] == [],
            "branch screen offered as a root proof",
        )
        polygon = certificate["polygon"]
        if polygon is not None:
            replay(instance, vehicles, polygon)
        stock_grid = certificate.get("stock_grid", False)
        need(type(stock_grid) is bool, "invalid stock-screen flag")
        exact = screen(
            instance,
            vehicles,
            polygon,
            F(certificate["threshold"]),
            None,
            stock_grid=stock_grid,
        )
        need(exact == certificate, "route-screen transcript differs")
        witness = exact["maximizing_witness"]
        return (
            F(exact["upper"]),
            verify_lower(instance, vehicles, witness) if witness is not None else None,
        )
    raise ValueError(f"unknown upper proof method: {method!r}")


def check_forbidden_modules():
    loaded = sorted(
        name for name in sys.modules if name.split(".")[0] in FORBIDDEN_MODULES
    )
    need(not loaded, f"forbidden solver/native modules loaded during pure replay: {loaded}")


def _verify_witness_file(instance: dict, file_hash: str, vehicles: int, path: Path) -> F:
    doc = read_json(path)
    if "certificate" in doc:
        need(doc.get("input_sha256") == file_hash, f"{path}: input SHA-256 mismatch")
        need(doc.get("vehicles") == vehicles, f"{path}: vehicles mismatch")
        score = verify_lower(instance, vehicles, doc["certificate"])
        if "score" in doc and doc["score"] is not None:
            need(score == F(doc["score"]), f"{path}: claimed L score differs from exact replay")
        return score
    return verify_lower(instance, vehicles, doc)


def _verify_upper_file(instance: dict, file_hash: str, vehicles: int, path: Path):
    doc = read_json(path)
    if "certificate" in doc:
        need(doc.get("input_sha256") == file_hash, f"{path}: input SHA-256 mismatch")
        need(doc.get("vehicles") == vehicles, f"{path}: vehicles mismatch")
        cert = doc["certificate"]
        upper, witness_val = replay_upper_certificate(instance, vehicles, cert)
        if "score" in doc and doc["score"] is not None:
            need(upper == F(doc["score"]), f"{path}: claimed U score differs from exact replay")
        return upper, cert["method"], witness_val
    upper, witness_val = replay_upper_certificate(instance, vehicles, doc)
    return upper, doc["method"], witness_val


def verify_stage_b_csv(csv_path: Path, selected_ids=None, quiet=False):
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if selected_ids is not None:
        rows = [r for r in rows if r["id"] in selected_ids]

    checked = 0
    optimal = 0
    cert5 = 0
    methods = {}
    for row in rows:
        cid = row["id"]
        in_path = ROOT / row["input_file"]
        file_hash = digest(in_path)
        need(file_hash == row["input_sha256"], f"{cid}: input SHA-256 mismatch")
        instance = read_json(in_path)
        vehicles = int(row["K"])
        need(
            type(instance["vehicles"]) is int and instance["vehicles"] == vehicles,
            f"{cid}: vehicles mismatch in raw instance",
        )

        lower = _verify_witness_file(instance, file_hash, vehicles, ROOT / row["L_certificate"])
        need(lower == F(row["L_exact"]), f"{cid}: replayed L != CSV L_exact")

        upper, method, witness_val = _verify_upper_file(
            instance, file_hash, vehicles, ROOT / row["U_certificate"]
        )
        need(upper == F(row["U_v21_exact"]), f"{cid}: replayed U != CSV U_v21_exact")
        need(method == row["U_v21_method"], f"{cid}: proof method mismatch")
        need(lower <= upper, f"{cid}: INVALID INTERVAL L={lower} > U={upper}")
        if witness_val is not None:
            need(witness_val <= upper, f"{cid}: screen witness exceeds upper bound")

        gap = (upper - lower) / lower if lower > 0 else F(0)
        need(gap == F(row["gap_v21_exact"]), f"{cid}: exact gap mismatch")
        is_c5 = gap <= F(1, 20) if lower > 0 else (upper == 0)
        is_opt = lower == upper
        need(str(is_c5) == row["cert5_v21"], f"{cid}: cert5 mismatch")
        need(str(is_opt) == row["strict_optimal"], f"{cid}: strict_optimal mismatch")

        checked += 1
        optimal += int(is_opt)
        cert5 += int(is_c5)
        methods[method] = methods.get(method, 0) + 1
        if not quiet:
            print(
                json.dumps({
                    "suite": csv_path.stem,
                    "id": cid,
                    "K": vehicles,
                    "L_exact": str(lower),
                    "U_exact": str(upper),
                    "gap_pct": round(float(gap) * 100, 3),
                    "method": method,
                    "status": "strict_optimal" if is_opt else ("cert5" if is_c5 else "open"),
                }),
                flush=True,
            )
    return {
        "suite": csv_path.stem,
        "intervals": checked,
        "witnesses": checked,
        "strict_uppers": checked,
        "cert5": cert5,
        "strict_optimal": optimal,
        "methods": methods,
    }


def verify_formal_csv(csv_path: Path, selected_ids=None, quiet=False):
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if selected_ids is not None:
        rows = [r for r in rows if r["case"] in selected_ids]

    by_case = {}
    for row in rows:
        by_case.setdefault(row["case"], []).append(row)

    intervals = 0
    witnesses = 0
    strict_uppers = 0
    cert5 = 0
    optimal = 0
    methods = {}

    for cid, case_rows in by_case.items():
        first = case_rows[0]
        in_path = ROOT / first["input_file"]
        file_hash = digest(in_path)
        need(file_hash == first["input_sha256"], f"{cid}: formal input SHA-256 mismatch")
        instance = read_json(in_path)
        vehicles = int(first["K"])
        need(instance["vehicles"] == vehicles, f"{cid}: formal vehicles mismatch")

        v21_upper = None
        lowers_by_method = {}
        for row in case_rows:
            need(row["input_sha256"] == file_hash, f"{cid}: inconsistent SHA across methods")
            m = row["method"]
            lower = _verify_witness_file(instance, file_hash, vehicles, ROOT / row["L_certificate"])
            need(lower == F(row["L_exact"]), f"{cid}/{m}: replayed L != CSV L_exact")
            lowers_by_method[m] = lower
            witnesses += 1

            if row["U_certificate"]:
                upper, u_method, _ = _verify_upper_file(
                    instance, file_hash, vehicles, ROOT / row["U_certificate"]
                )
                need(upper == F(row["strict_U_exact"]), f"{cid}/{m}: replayed U != CSV strict_U_exact")
                need(u_method == row["strict_U_method"], f"{cid}/{m}: strict_U_method mismatch")
                need(lower <= upper, f"{cid}/{m}: INVALID INTERVAL L={lower} > U={upper}")
                gap = (upper - lower) / lower if lower > 0 else F(0)
                need(gap == F(row["gap_exact"]), f"{cid}/{m}: formal gap mismatch")
                is_c5 = gap <= F(1, 20) if lower > 0 else (upper == 0)
                is_opt = lower == upper
                need(str(is_c5) == row["cert5"], f"{cid}/{m}: cert5 mismatch")
                need(str(is_opt) == row["strict_optimal"], f"{cid}/{m}: strict_optimal mismatch")
                v21_upper = upper
                intervals += 1
                strict_uppers += 1
                cert5 += int(is_c5)
                optimal += int(is_opt)
                methods[u_method] = methods.get(u_method, 0) + 1

        # Cross-method soundness check: EVERY baseline feasible L must satisfy L <= v2.1 strict U!
        if v21_upper is not None:
            for m, lv in lowers_by_method.items():
                need(
                    lv <= v21_upper,
                    f"{cid}: cross-method bound conflict! {m} L={lv} > v2.1 strict U={v21_upper}",
                )
            if not quiet:
                v21_l = lowers_by_method["v2.1"]
                gap_pct = round(float((v21_upper - v21_l) / v21_l) * 100, 3) if v21_l > 0 else 0.0
                print(
                    json.dumps({
                        "suite": csv_path.stem,
                        "case": cid,
                        "K": vehicles,
                        "methods_L_verified": sorted(lowers_by_method),
                        "v21_L_exact": str(v21_l),
                        "v21_U_exact": str(v21_upper),
                        "gap_pct": gap_pct,
                        "status": "strict_optimal" if v21_l == v21_upper else "bounded",
                    }),
                    flush=True,
                )

    return {
        "suite": csv_path.stem,
        "intervals": intervals,
        "witnesses": witnesses,
        "strict_uppers": strict_uppers,
        "cert5": cert5,
        "strict_optimal": optimal,
        "methods": methods,
    }


def verify_single(instance_path: Path, solution_path: Path | None, proof_path: Path | None, vehicles: int | None):
    file_hash = digest(instance_path)
    instance = read_json(instance_path)
    k = vehicles if vehicles is not None else instance.get("vehicles")
    need(type(k) is int and k >= 1, "provide --vehicles K or an instance JSON with integer vehicles")

    lower = None
    upper = None
    method = None
    if solution_path is not None:
        lower = _verify_witness_file(instance, file_hash, k, solution_path)
    if proof_path is not None:
        upper, method, _ = _verify_upper_file(instance, file_hash, k, proof_path)
    if lower is not None and upper is not None:
        need(lower <= upper, f"INVALID INTERVAL: L={lower} > U={upper}")

    gap = None
    if lower is not None and upper is not None and lower > 0:
        gap = (upper - lower) / lower
    status = (
        "strict_optimal"
        if lower is not None and upper is not None and lower == upper
        else ("bounded" if upper is not None else "feasible")
    )
    check_forbidden_modules()
    print(
        json.dumps({
            "instance": str(instance_path),
            "input_sha256": file_hash,
            "vehicles": k,
            "L_exact": None if lower is None else str(lower),
            "U_exact": None if upper is None else str(upper),
            "U_method": method,
            "gap_exact": None if gap is None else str(gap),
            "gap_pct": None if gap is None else round(float(gap) * 100, 3),
            "status": status,
            "forbidden_modules": [],
        }),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("all", "stage_b", "blind", "latet", "formal"),
        default="all",
        help="Archived multi-vehicle benchmark suite to verify (default: all).",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Fast curated subset covering all scales and all 4 strict U proof mechanisms.",
    )
    parser.add_argument("--ids", nargs="+", help="Verify only specific instance/case IDs.")
    parser.add_argument("--quiet", action="store_true", help="Print only the final summary.")
    parser.add_argument("--single", type=Path, help="Verify a standalone instance JSON.")
    parser.add_argument("--solution", type=Path, help="Feasible fleet witness JSON for --single.")
    parser.add_argument("--proof", type=Path, help="Strict upper-bound proof JSON for --single.")
    parser.add_argument("--vehicles", type=int, help="Fleet size K override for --single.")
    args = parser.parse_args()

    started = monotonic()
    if args.single is not None:
        if args.solution is None and args.proof is None:
            parser.error("--single requires at least one of --solution or --proof")
        verify_single(args.single, args.solution, args.proof, args.vehicles)
        return 0

    results_dir = ROOT / "multi_vehicle" / "results"
    summaries = []

    sb_filter = set(args.ids) if args.ids else (set(SAMPLE_STAGE_B_IDS) if args.sample else None)
    fm_filter = set(args.ids) if args.ids else (set(SAMPLE_FORMAL_IDS) if args.sample else None)

    if args.suite in ("all", "stage_b", "blind"):
        summaries.append(verify_stage_b_csv(results_dir / "stage_b_96_blind.csv", sb_filter, args.quiet))
    if args.suite in ("all", "stage_b", "latet"):
        summaries.append(verify_stage_b_csv(results_dir / "stage_b_20_latet.csv", sb_filter, args.quiet))
    if args.suite in ("all", "formal"):
        summaries.append(verify_formal_csv(results_dir / "formal_25_baselines.csv", fm_filter, args.quiet))

    check_forbidden_modules()
    total_intervals = sum(s["intervals"] for s in summaries)
    total_witnesses = sum(s["witnesses"] for s in summaries)
    total_uppers = sum(s["strict_uppers"] for s in summaries)
    total_opt = sum(s["strict_optimal"] for s in summaries)
    need(total_intervals > 0 or total_witnesses > 0, "no matching instances were verified")

    elapsed = monotonic() - started
    print(
        f"PASS: verified {total_intervals} strict intervals "
        f"({total_witnesses} L witnesses, {total_uppers} strict U certificates, "
        f"{total_opt} strict_optimal, 0 L>U, 0 forbidden modules) in {elapsed:.2f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
