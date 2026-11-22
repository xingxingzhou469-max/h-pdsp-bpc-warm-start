"""Replay archived single-vehicle upper bounds using integer certificates."""

import argparse
from hashlib import sha256
import json
import lzma
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if not __package__:
    sys.path.insert(0, str(ROOT))

from hpdp.cert_verifier import CertificateVerifier
from hpdp.instances import decode_instance
from hpdp.prefix import Resources


CASES = (
    "blind-12", "random-60-4", "random-60-5", "random-60-6",
    "random-100-2", "random-100-3", "random-100-4", "random-100-5",
)
EXTRA_CASES = ("blind-09",)


def replay(name, root=ROOT):
    """Replay every packaged source for an instance and return the tightest bound."""
    if name not in CASES + EXTRA_CASES:
        raise ValueError(f"No archived upper-bound certificate for {name}")

    root = Path(root)
    certificate = Path("proofs") / "open_bounds" / f"{name}.json.xz"
    payload = (root / certificate).read_bytes()
    documents = json.loads(lzma.decompress(payload))
    expected = {"historical_dual", "independent_frontier"} if name in CASES else {"independent_frontier"}
    if type(documents) is not dict or set(documents) != expected:
        raise ValueError(f"{name}: unexpected certificate sources")

    raw = (root / "instances" / f"{name}.json").read_bytes()
    instance = decode_instance(raw)
    if instance["id"] != name:
        raise ValueError(f"{name}: instance identifier mismatch")
    instance_sha256 = sha256(raw).hexdigest()
    verifier = CertificateVerifier(Resources(instance), instance_sha256)
    bounds = {source: verifier.replay(doc) for source, doc in documents.items()}
    if any(bound is None for bound in bounds.values()):
        raise ValueError(f"{name}: expected a finite upper bound per source")
    counts = {source: len(doc["leaves"]) for source, doc in documents.items()}
    winner = min(bounds, key=lambda source: (bounds[source], counts[source], source))
    return {
        "upper_exact": str(bounds[winner]),
        "source": winner,
        "leaf_count": counts[winner],
        "input_sha256": instance_sha256,
        "certificate": certificate.as_posix(),
        "certificate_sha256": sha256(payload).hexdigest(),
        "source_bounds": {source: str(bound) for source, bound in bounds.items()},
        "source_leaf_counts": counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instances", nargs="*", help="instance IDs, defaulting to the original eight")
    args = parser.parse_args()
    try:
        results = {name: replay(name) for name in args.instances or CASES}
    except (ValueError, OSError, EOFError, lzma.LZMAError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
