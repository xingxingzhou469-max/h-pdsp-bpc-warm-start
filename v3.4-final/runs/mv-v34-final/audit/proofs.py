"""Exact verification for the H-PDSP documented, simple-route domain."""

from fractions import Fraction as F
from functools import cache
from hashlib import sha256
import heapq
import importlib
import importlib.util
import json
from math import gcd, lcm
import os
from pathlib import Path
import sys
from types import ModuleType


RUN = Path(__file__).resolve().parents[1]
PROJECT = RUN.parents[1]
V32 = Path(os.environ.get("V33_V32", PROJECT / "runs/mv-v32-integration")).resolve()
V21 = Path(os.environ.get("V33_V21", PROJECT / "交付/github_release_v21/repo")).resolve()
LEGACY_CHECKERS = ("graph", "route", "regions", "screen2", "branchprice",
                   "maskfleet", "zone", "fair", "ng")
STRICT_PARENT_KINDS = frozenset((*LEGACY_CHECKERS, "box", "v21", "fleet-selection-v33",
                                "sparse-supplier-fair-v2", "fleet-cut-dual-v33"))
FORBIDDEN = {"ctypes", "highspy", "numpy", "scipy", "ortools", "pulp", "gurobipy"}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = dict(pairs)
    need(len(pairs) == len(result), "duplicated JSON key")
    return result


def reject_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def decode(payload):
    return json.loads(payload, parse_float=F, parse_constant=reject_constant,
                      object_pairs_hook=unique_object)


def stdlib_only():
    need(sys.flags.no_site and sys.dont_write_bytecode, "run replay with Python -B -S")
    loaded = FORBIDDEN.intersection(name.split(".")[0] for name in sys.modules)
    need(not loaded, f"solver or native FFI imported by replay: {sorted(loaded)}")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@cache
def oracle():
    return load(V32 / "audit/exact.py", "_v33_audit_v32_exact")


def fraction(value):
    need(type(value) in (int, F, str), "quantity must be an exact rational")
    return F(value)


def lattice_parameters(model):
    def unit(values):
        if not values:
            return F(0)
        return F(gcd(*(q.numerator for q in values)), lcm(*(q.denominator for q in values)))

    capped = [min(q, model.capacity) for q in model.supply if q and model.capacity]
    if sum(capped) > model.capacity:
        capped.append(model.capacity)
    population = sum(model.population)
    delta = unit(capped)
    rho = unit([p for p in model.population if p])
    return {"delta": delta, "rho": rho, "P": population,
            "g": delta * rho / population if population else F(0)}


def lattice_parent(doc):
    need(doc["schema"] == "mv-optimal-value-lattice-1", "unsupported lattice certificate schema")
    parent = doc["parent"]
    kind = parent["kind"]
    need(type(kind) is str and kind in STRICT_PARENT_KINDS, "invalid lattice parent upper kind")
    need(type(parent["bytes"]) is str, "lattice parent must contain the complete UTF-8 proof bytes")
    payload = parent["bytes"].encode("utf-8")
    need(sha256(payload).hexdigest() == parent["sha256"], "lattice parent SHA-256 differs")
    proof = decode(payload)
    # JSON decimal numbers become Fraction on input. Upper claims must explicitly
    # use integers or rational strings to distinguish them from floating output.
    need(type(proof["upper"]) in (int, str), "lattice parent upper is not an exact rational claim")
    need(fraction(proof["upper"]) >= 0, "lattice parent upper is negative")
    return kind, proof


def lattice_projection(model, doc, input_sha):
    need(type(input_sha) is str and doc["input_sha256"] == input_sha,
         "lattice certificate input SHA-256 differs")
    need(type(doc["upper"]) is str, "lattice endpoint must be a rational string")
    params = lattice_parameters(model)
    claims = doc["parameters"]
    need(type(claims) is dict and claims.keys() == params.keys()
         and all(type(claims[key]) is str and F(claims[key]) == value
                 for key, value in params.items()), "lattice parameters differ from the original input")
    kind, parent = lattice_parent(doc)
    step, bound = params["g"], fraction(parent["upper"])
    projected = (bound // step) * step if step else F(0)
    need(projected == F(doc["upper"]), "lattice upper differs from exact floor projection")
    return kind, parent, projected


def lower(instance, vehicles, witness):
    routes, amounts = witness["routes"], witness["amounts"]
    need(type(routes) is list and type(amounts) is list, "route and amount tables must be arrays")
    need(all(type(row) is list for row in (*routes, *amounts)), "route and amount rows must be arrays")
    for row in amounts:
        for value in row:
            fraction(value)
    return oracle().verify_witness(instance, vehicles, witness)


def endpoint_shortcuts(model):
    """Prove that dropping an agency prefix and a supplier suffix preserves feasibility.

    A prefix before the first supplier collects no stock and must deliver zero.
    A suffix after the last agency delivers nothing. Shortest paths through only
    the relevant role certify that the direct depot arc is at least as fast.
    """
    size = len(model.supply)
    suppliers = [i for i in range(1, size) if model.supply[i]]
    agencies = [i for i in range(1, size) if model.population[i]]
    travel, service = model.travel, model.service

    def distances(intermediate, terminal, cost):
        nodes = intermediate + terminal
        best = dict.fromkeys(nodes)
        best[0] = F(0)
        queue = [(F(0), 0)]
        can_extend = {0, *intermediate}
        while queue:
            length, i = heapq.heappop(queue)
            if i not in can_extend or length != best[i]:
                continue
            for j in nodes:
                if i == j:
                    continue
                candidate = length + cost(i, j)
                if best[j] is None or candidate < best[j]:
                    best[j] = candidate
                    heapq.heappush(queue, (candidate, j))
        return best

    head = distances(agencies, suppliers, lambda i, j: travel[i][j] + service[j])
    # Reverse the original suffix arcs; an intermediate supplier pays its service.
    tail = distances(suppliers, agencies, lambda i, j: travel[j][i] + (service[i] if i else 0))
    return {
        "prefix": [i for i in suppliers if head[i] < travel[0][i] + service[i]],
        "suffix": [i for i in agencies if tail[i] < travel[i][0]],
    }


@cache
def v21_verifier():
    return load(V21 / "verify_multi.py", "_v33_audit_v21_verify")


@cache
def checker(kind):
    if kind in LEGACY_CHECKERS:
        sys.path.insert(0, str(V32))
        return importlib.import_module(f"{kind}.check").replay
    if kind == "fleet-selection-v33":
        return load(RUN / "selection/check.py", "_v33_audit_selection").replay
    if kind == "sparse-supplier-fair-v2":
        path = RUN / "sparse"
        previous = sys.modules.get("model")
        # The standalone checker imports its sibling as a top-level module.
        # Isolate that binding from any other standalone checker.
        sys.modules["model"] = load(path / "model.py", "_v33_audit_sparse_model")
        try:
            return load(path / "check.py", "_v33_audit_sparse").replay
        finally:
            if previous is None:
                del sys.modules["model"]
            else:
                sys.modules["model"] = previous
    if kind == "fleet-cut-dual-v33":
        path = RUN / "upper/check.py"
        package = ModuleType("_v33_audit_upper")
        package.__path__ = [str(path.parent)]
        sys.modules[package.__name__] = package
        return load(path, f"{package.__name__}.check").replay
    raise ValueError(f"unsupported proof kind: {kind}")


def replay_certificate(instance, vehicles, kind, doc, model=None, input_sha=None):
    """Return the claimed endpoint only after mathematical verification."""
    key = "lower" if kind == "L" else "upper"
    advertised = fraction(doc[key])
    if kind == "L":
        exact = lower(instance, vehicles, doc["witness"])
    elif kind == "lattice":
        model = model or oracle().Model.read(instance)
        parent_kind, parent_doc, exact = lattice_projection(model, doc, input_sha)
        replay_certificate(instance, vehicles, parent_kind, parent_doc, model, input_sha)
    elif kind == "box":
        model = model or oracle().Model.read(instance)
        exact = min(sum(model.supply), model.capacity * sum(p > 0 for p in model.population))
    elif kind == "v21":
        shortcuts = endpoint_shortcuts(model or oracle().Model.read(instance))
        need(not any(shortcuts.values()), f"v2.1 endpoint shortcut would increase travel time: {shortcuts}")
        proof = doc["proof"]
        need(proof["kind"] == "U" and proof["input_sha256"] == input_sha
             and type(proof["vehicles"]) is int and proof["vehicles"] == vehicles
             and type(proof["score"]) is str and F(proof["score"]) == advertised,
             "v2.1 proof identity or exact score differs")
        exact, witnessed = v21_verifier().replay_upper_certificate(instance, vehicles, proof["certificate"])
        need(witnessed is None or witnessed <= exact, "legacy upper also contains an inconsistent witness")
    else:
        exact = checker(kind)(instance, vehicles, doc)
    need(exact == advertised, f"{kind} certificate value {advertised} differs from exact replay {exact}")
    stdlib_only()
    return exact
