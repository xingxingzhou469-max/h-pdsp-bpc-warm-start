"""Replay portable routing dual certificates using the exact kernel only."""

import argparse
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path

from .cut_cert import CutEncoder
from .instances import decode_instance
from .prefix import Resources
from .priced_routes import cap_product
from .routing_exact import Row, exact_upper, flow_cap, rewards, routing


def branch_row(model, branch):
    if type(branch) is not list or len(branch) != 2:
        raise ValueError("A branch must specify [site, value]")
    v, value = branch
    if type(v) is not int or v not in model.visits or type(value) is not int or value not in (0, 1):
        raise ValueError("A branch must fix a nondepot visit to 0 or 1")
    return Row(((model.visits[v], 1 - 2*value),), -value)


def _check_partition(paths):
    """Collapse sibling leaves until exactly the root remains."""
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate leaf path")
    frontier = set(paths)
    while frontier != {()}:
        path = max(frontier, key=len)
        if not path:
            raise ValueError("Missing branch root")
        prefix, (v, value) = path[:-1], path[-1]
        sibling = (*prefix, (v, 1-value))
        if sibling not in frontier or prefix in frontier:
            raise ValueError("Overlapping or incomplete branch partition")
        frontier.remove(path)
        frontier.remove(sibling)
        frontier.add(prefix)


class CertificateVerifier:
    """Cache trusted integer models for a single immutable input."""

    def __init__(self, resources, instance_sha256):
        self.r = resources
        self.instance_sha256 = instance_sha256
        self.models = {}

    def replay(self, document):
        if (type(document) is not dict
                or set(document) != {"version", "instance_sha256", "slope", "pickup_blocks",
                                     "leaves", "cap_product_bound"}
                or type(document["version"]) is not int or document["version"] != 1
                or type(document["instance_sha256"]) is not str
                or len(document["instance_sha256"]) != 64
                or document["instance_sha256"] != self.instance_sha256
                or any(c not in "0123456789abcdef" for c in document["instance_sha256"])
                or type(document["pickup_blocks"]) is not bool):
            raise ValueError("Certificate version, fields or instance hash mismatch")
        slope, leaves = document["slope"], document["leaves"]
        if (type(slope) is not list or len(slope) != 2
                or any(type(a) is not int or a <= 0 for a in slope)
                or type(leaves) is not list or not leaves):
            raise ValueError("A certificate needs two positive slopes and at least one leaf")
        a, b = slope
        variant = document["pickup_blocks"]
        if variant not in self.models:
            model = routing(self.r, pickup_blocks=variant)
            self.models[variant] = model, CutEncoder(self.r, model)
        model, encoder = self.models[variant]
        paths, rows = [], []
        for leaf in leaves:
            if (type(leaf) is not dict
                    or set(leaf) != {"cuts", "branch_rows", "integer_multipliers",
                                    "denominator", "cap_product_bound"}
                    or type(leaf["cuts"]) is not list or type(leaf["branch_rows"]) is not list):
                raise ValueError("Invalid leaf fields")
            branches = [branch_row(model, branch) for branch in leaf["branch_rows"]]
            path = tuple(map(tuple, leaf["branch_rows"]))
            if len({v for v, _ in path}) != len(path):
                raise ValueError("A path splits on the same visit twice")
            paths.append(path)
            rows.append(branches)
        _check_partition(paths)

        objective = rewards(self.r, model, a, b)
        ceiling = flow_cap(self.r)
        upper = -1
        for leaf, branches in zip(leaves, rows, strict=True):
            inequalities = [model.time, *model.flow_rows]
            inequalities.extend(encoder.row(spec) for spec in leaf["cuts"])
            inequalities.extend(branches)
            multipliers = leaf["integer_multipliers"]
            if (type(multipliers) is not dict or set(multipliers) != {"nonnegative", "free"}
                    or any(type(vector) is not list for vector in multipliers.values())):
                raise ValueError("Invalid multiplier vectors")
            exact = exact_upper(objective, inequalities, model.degrees, leaf["denominator"],
                                **multipliers)
            reward = exact["numerator"] // exact["denominator"]
            bound = -1 if reward < 0 else cap_product(reward, a, b, ceiling, self.r.people)
            if type(leaf["cap_product_bound"]) is not int or leaf["cap_product_bound"] != bound:
                raise ValueError("Leaf product bound differs from integer replay")
            upper = max(upper, bound)
        if type(document["cap_product_bound"]) is not int or document["cap_product_bound"] != upper:
            raise ValueError("Global product bound differs from the maximum leaf bound")
        return None if upper < 0 else Fraction(upper, self.r.units)


def verify_certificate(instance_bytes, document):
    """Return a strict Fraction upper bound, or None for certified infeasibility."""
    verifier = CertificateVerifier(Resources(decode_instance(instance_bytes)),
                                   sha256(instance_bytes).hexdigest())
    return verifier.replay(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args()
    try:
        upper = verify_certificate(args.instance.read_bytes(), json.loads(args.certificate.read_bytes()))
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print("infeasible" if upper is None else str(upper))


if __name__ == "__main__":
    main()
