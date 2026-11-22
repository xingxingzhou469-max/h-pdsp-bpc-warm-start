"""Load and validate normalized H-PDSP instance JSON."""

import json
import math
from decimal import Decimal
from pathlib import Path


def canonical_json(value):
    """Serialize normalized data, retaining exact JSON numbers on a round trip."""
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Nonfinite decimal is not a JSON number")
        # Reuse the established spelling when it represents the exact input.
        # This keeps the canonical bytes of the existing generated inputs.
        usual = json.dumps(float(value))
        if Decimal(usual) == value:
            return usual
        return str(value)
    if isinstance(value, dict):
        members = []
        for key in sorted(value):
            if isinstance(key, str):
                name = key
            elif key is None or isinstance(key, (int, float)):
                name = json.dumps(key)
            else:
                raise TypeError(f"Invalid JSON object key of type {type(key).__name__}")
            members.append(json.dumps(name, ensure_ascii=False) + ":" + canonical_json(value[key]))
        return "{" + ",".join(members) + "}"
    if isinstance(value, (tuple, list)):
        return "[" + ",".join(map(canonical_json, value)) + "]"
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_instance(data):
    """Check the depot and numeric schema. Travel can be asymmetric and nonmetric."""
    size = len(data["travel"])
    for field in ("travel", "service", "supply", "population"):
        if len(data[field]) != size:
            raise ValueError(f"{data['id']}: wrong size for {field}")
    for index, row in enumerate(data["travel"]):
        if len(row) != size or row[index] != 0:
            raise ValueError(f"{data['id']}: invalid travel row {index}")
    for field in ("service", "supply", "population"):
        if data[field][0] != 0:
            raise ValueError(f"{data['id']}: nonzero depot {field}")
    numbers = [data["limit"], data["capacity"]]
    for field in ("service", "supply", "population"):
        numbers.extend(data[field])
    numbers.extend(value for row in data["travel"] for value in row)
    def valid_number(value):
        if type(value) is Decimal:
            return value.is_finite() and value >= 0
        if type(value) is float:
            return math.isfinite(value) and value >= 0
        return type(value) is int and value >= 0

    if not all(map(valid_number, numbers)):
        raise ValueError(f"{data['id']}: times and quantities must be finite and nonnegative")
    if data["capacity"] == 0 or not any(data["supply"]) or not any(data["population"]):
        raise ValueError(f"{data['id']}: need positive capacity, supply and population")
    for node, (supply, population) in enumerate(zip(data["supply"], data["population"])):
        if node and (supply > 0) == (population > 0):
            raise ValueError(f"{data['id']}: node {node} lacks a unique role")


def decode_instance(raw):
    """Read normalized JSON without passing decimal literals through binary64."""
    data = json.loads(raw, parse_float=Decimal)
    validate_instance(data)
    return data


def load_instance(path):
    return decode_instance(Path(path).read_bytes())
