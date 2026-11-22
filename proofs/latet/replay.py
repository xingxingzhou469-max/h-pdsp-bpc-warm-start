"""Independent upper-bound audits of the normalized Latet instances, using C++20."""
import functools
import hashlib
import json
import math
import os
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


@functools.cache
def _build(method):
    if method not in ("nonbinding", "two_supplier", "priced"):
        raise ValueError("unknown Latet proof type")
    work = tempfile.TemporaryDirectory()
    binary = Path(work.name) / method
    subprocess.run([os.environ.get("CXX", "c++"), "-O3", "-std=c++20",
                    str(HERE / (method + ".cpp")), "-o", str(binary)], check=True)
    return work, binary


def _payload(rows):
    return "\n".join(" ".join(map(str, row)) for row in rows) + "\n"


def replay(name, root=ROOT):
    spec = json.loads((HERE / "certificates.json").read_bytes())[name]
    raw = (Path(root) / "instances" / (name + ".json")).read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["instance_sha256"]:
        raise ValueError("Latet input fingerprint mismatch")
    data = json.loads(raw, parse_float=Fraction)
    upper = Fraction(spec["upper_exact"])
    scale = math.lcm(*(Fraction(s).denominator for s in data["supply"]))
    stock = [int(s * scale) for s in data["supply"]]
    people, travel = data["population"], data["travel"]
    capacity, limit = scale * data["capacity"], data["limit"]
    target = upper * scale * sum(people)
    if target.denominator != 1:
        raise ValueError("threshold must lie on the certificate's integer scale")
    size = len(people)
    method = spec["method"]
    _, binary = _build(method)
    if method != "priced":
        rows = [[size, capacity, limit, int(target)], data["source_nodes"], stock,
                people, data["service"], *travel]
        result = subprocess.run([str(binary), str(spec["tracked"]), "300"], input=_payload(rows),
                                capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        if "status certified" not in lines or f"strict_upper_numerator {target}" not in lines:
            raise ValueError(f"{name}: upper bound was not established: {lines[-8:]}")
        return upper

    # Every feasible route has a nonempty supplier set and a unique first supplier.
    if name != "latet-2808" or spec["instance_sha256"] != "40fec9c0309e95fb2bb136c6e704020bd1cc2dd723a68a24d84a06bf5abcca8a":
        raise ValueError("unexpected input for the fixed 2808 verifier")
    edges = [[min(limit + 1, t + data["service"][j]) for j, t in enumerate(row)]
             for row in travel]
    stock = [min(s, capacity) for s in stock]
    terms = {(c["mask"], c["first"]): c for c in spec["prices"]}
    for mask in range(1, 8):
        donors = [v for v in (1, 2, 3) if mask & (1 << (v - 1))]
        if Fraction(sum(stock[v] for v in donors), scale) <= upper:
            continue
        for first in donors:
            c = terms.pop((mask, first))
            header = [size, limit, first, mask, len(c["tracked"]), capacity, int(target)]
            result = subprocess.run([str(binary)], input=_payload(
                [header, people, c["prices"], stock, *edges, c["tracked"]]),
                capture_output=True, text=True, check=True)
            pop, _, _, unresolved = map(int, result.stdout.split())
            if pop != c["population_ub"] or unresolved:
                raise ValueError(f"{name}: incomplete proof for {mask}, {first}")
    if terms:
        raise ValueError("unexpected supplier context")
    return upper


if __name__ == "__main__":
    for case in json.loads((HERE / "certificates.json").read_bytes()):
        print(case, replay(case), flush=True)
