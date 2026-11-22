"""Trusted adapters for route proposals and portable dual certificates."""

from copy import deepcopy
from hashlib import sha256
from time import monotonic

from . import dual_plugin, primal_plugin
from .allocation import allocate
from .cert_verifier import CertificateVerifier
from .check import check
from .instances import canonical_json


def checked_route(instance, route):
    """Reallocate a raw proposal and check the original directed route at tol=0."""
    if type(route) not in (tuple, list) or len(route) < 2 or any(type(v) is not int for v in route):
        raise ValueError("Expected raw integer route indices")
    route = tuple(route)
    value, amounts = allocate(instance, route)
    if check(instance, route, amounts, tol=0)["objective"] != value:
        raise ArithmeticError("Fixed-route allocation disagrees with the strict checker")
    return {"route":route, "amounts":amounts}, value


def improve(r, deadline, seed, *, emit, should_stop):
    """Publish strictly checked proposals through the trusted incumbent callback."""
    rejected = 0

    def offer(route):
        nonlocal rejected
        if monotonic() >= deadline or should_stop():
            return False
        try:
            witness, value = checked_route(r.instance, route)
        except ValueError:
            rejected += 1
        else:
            if monotonic() < deadline:
                emit(witness, value)
        return monotonic() < deadline and not should_stop()

    primal_plugin.propose(deepcopy(r.instance), None if seed is None else tuple(seed["route"]),
                          deadline, emit=offer)
    return rejected


def certify(r, instance_sha256, deadline, seed, *, emit, cuts=(), branch_depth=2):
    """Emit verified (bound, document) pairs. Invalid documents have no effect."""
    if instance_sha256 is None:
        instance_sha256 = sha256((canonical_json(r.instance)+"\n").encode()).hexdigest()
    verifier = CertificateVerifier(r, instance_sha256)
    rejected = 0

    def offer(document):
        nonlocal rejected
        if monotonic() >= deadline:
            return False
        try:
            upper = verifier.replay(document)
        except ValueError:
            rejected += 1
            return True
        return monotonic() < deadline and emit(upper, deepcopy(document))

    dual_plugin.propose(deepcopy(r.instance), instance_sha256,
                        None if seed is None else tuple(seed["route"]), deadline,
                        emit=offer, cuts=deepcopy(cuts), branch_depth=branch_depth)
    return rejected
