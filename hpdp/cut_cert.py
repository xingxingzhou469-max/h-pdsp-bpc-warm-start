"""Encode verified cut specifications as integer rows of the routing model."""

from .prefix import Resources
from .routing_exact import Row, routing


def _site_set(values, model, *, allow_depot=False):
    if (type(values) is not list or not values
            or any(type(v) is not int or not (0 if allow_depot else 1) <= v < model.nodes
                   for v in values) or values != sorted(set(values))):
        raise ValueError("Invalid cut site set")
    return set(values)


def cut_row(r, model, spec):
    if type(spec) is not dict or set(spec) != {"type", "set", "target"} or spec["type"] != "SEC":
        raise ValueError("Unrecognized cut specification")
    side = _site_set(spec["set"], model)
    k = spec["target"]
    if type(k) is not int or k not in side:
        raise ValueError("A connectivity cut must contain its nondepot target")
    outside = set(range(model.nodes)) - side
    # Equal degrees make both directions of the cut identical. Encode the
    # internal arcs of its smaller side using the degree equalities.
    inside = side if len(side) <= len(outside) else outside
    terms = [(model.visits[v], -1) for v in sorted(inside - {0, k})]
    if k not in inside:
        terms.append((model.visits[k], 1))
    for i in sorted(inside):
        for j in sorted(inside):
            col = model.arcs.get((i, j))
            if col is not None:
                terms.append((col, 1))
    return Row(tuple(terms), int(0 in inside))


def validate_cut(instance, cut_spec):
    """Check a cut's schema and model-dependent legality."""
    r = Resources(instance)
    try:
        cut_row(r, routing(r, pickup_blocks=True), cut_spec)
    except ValueError:
        return False
    return True
