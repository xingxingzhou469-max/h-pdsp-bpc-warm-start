"""Verify an online screen2 equality using the exact native zone pricing DP."""
from hashlib import sha256
from pathlib import Path
import sys

from proofs import V32, decode, fraction, lattice_projection, lower, need, oracle


def check(original, lower_path, projected_path, stop_ns):
    payload = original.read_bytes()
    raw = decode(payload)
    k = raw["vehicles"]
    witness = decode(lower_path.read_bytes())
    low = lower(raw, k, witness["witness"])
    need(low == fraction(witness["lower"]), "wrong live lower score")
    kind, parent, projected = lattice_projection(oracle().Model.read(raw),
                                                  decode(projected_path.read_bytes()), sha256(payload).hexdigest())
    need(low == projected and kind == "screen2", "wrong live screen equality")
    sys.path.insert(0, str(V32))
    from route.model import read_model
    from screen2 import check as audit
    from screen2.solve import validate_parent

    def native_zone(instance, vehicles, zone):
        value = validate_parent(read_model(instance), zone, vehicles, stop_ns / 1e9)
        need(value is not None, "native zone replay missed the live budget")
        return value

    old = audit.replay_zone
    audit.replay_zone = native_zone
    try:
        # The complete mask digest and fleet exclusion are still replayed in Python.
        need(audit.replay(raw, k, parent) == fraction(parent["upper"]), "wrong live parent upper score")
    finally:
        audit.replay_zone = old


if __name__ == "__main__":
    source, witness, projection = map(Path, sys.argv[1:4])
    check(source, witness, projection, int(sys.argv[4]))
