"""Independently replay the two published examples with Python's standard library."""
import argparse
from fractions import Fraction
from hashlib import sha256
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
EXAMPLES = {"directed-13": (8, "1", False), "latet-1908-k2": (60, "11575200/1633", True)}


def verify_manifest():
    for row in (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        expected, filename = row.split("  ", 1)
        path = (ROOT / filename).resolve()
        if not path.is_relative_to(ROOT) or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Package digest differs for {filename}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=EXAMPLES)
    args = parser.parse_args()
    if not sys.flags.no_site or not sys.dont_write_bytecode:
        parser.error("Run with python3 -B -S")
    verify_manifest()
    os.environ["V33_V32"] = str(ROOT / "runs/mv-v32-integration")
    os.environ["V33_V21"] = str(ROOT / "交付/github_release_v21/repo")
    sys.path.insert(0, str(ROOT / "runs/mv-v34-final/audit"))
    from replay import verify_arm
    for case in (args.case,) if args.case else EXAMPLES:
        seconds, answer, external_clock = EXAMPLES[case]
        result = verify_arm(ROOT / "evidence" / case, ROOT / "examples" / f"{case}.json",
                            seconds, 7, case=case, arm="v34", clock_required=external_clock,
                            require_lattice=True, expected_checkpoints=[seconds])
        endpoint = result["endpoints"][str(seconds)]
        if (Fraction(endpoint["lower"]) != Fraction(answer) or
                Fraction(endpoint["upper"]) != Fraction(answer) or result["optimal_event"] is None):
            raise ValueError(f"Exact score or terminal proof differs for {case}")
        print(f"{case}: L = U = {answer}; {result['replayed_lowers']} lower and "
              f"{result['replayed_uppers']} upper certificates replayed")


if __name__ == "__main__":
    main()
