"""Publish exact lattice projections and terminate on independently replayed equality."""

from fractions import Fraction as F
from hashlib import sha256
import json
from math import gcd, lcm
from pathlib import Path
import subprocess
import sys
from time import monotonic_ns, time_ns


NS = 10**9
AUDITOR = Path(__file__).resolve().parent / "audit/replay.py"


def rational_gcd(values):
    if not values:
        return F(0)
    scale = lcm(*(value.denominator for value in values))
    return F(gcd(*(value.numerator * (scale // value.denominator) for value in values)), scale)


def parameters(raw):
    capacity = F(raw["capacity"])
    stock = [min(capacity, value) for value in map(F, raw["supply"]) if value > 0]
    people = [value for value in map(F, raw["population"]) if value > 0]
    if sum(stock) > capacity:
        stock.append(capacity)
    delta, rho, population = rational_gcd(stock), rational_gcd(people), sum(people, F(0))
    return {"delta": delta, "rho": rho, "P": population,
            "g": delta * rho / population if population else F(0)}


def encode(doc):
    return (json.dumps(doc, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


class LatticeTimeline:
    """Intercept the timeline's strict U publication path in every search mode."""

    def __init__(self, *args, **kwargs):
        self.proved_optimal = False
        self._lattice_grid = None
        super().__init__(*args, **kwargs)

    def admit(self, kind, payload):
        if self.proved_optimal:
            return
        old_count = self.counter
        if kind == "L":
            super().admit(kind, payload)
        else:
            doc = json.loads(payload)
            if type(doc["upper"]) not in (int, str):
                raise ValueError("a strict parent upper must be an exact rational")
            bound = F(doc["upper"])
            if bound < 0:
                raise ValueError("a strict parent upper must be nonnegative")
            if self._lattice_grid is None:
                self._lattice_grid = parameters(json.loads((self.out / "input.json").read_bytes(),
                                                          parse_float=str))
            step = self._lattice_grid["g"]
            projected = (bound // step) * step if step else F(0)
            if "U" in self.keep and projected >= self.keep["U"][0]:
                return
            super().admit("lattice", encode({
                "schema": "mv-optimal-value-lattice-1", "input_sha256": self.fingerprint,
                "upper": str(projected),
                "parameters": {key: str(value) for key, value in self._lattice_grid.items()},
                "parent": {"kind": kind, "sha256": sha256(payload).hexdigest(),
                           "bytes": payload.decode("utf-8")},
            }))
        if (self.counter != old_count and "L" in self.keep and "U" in self.keep
                and self.keep["L"][0] == self.keep["U"][0]):
            self._confirm_optimal()

    def _confirm_optimal(self):
        lower, upper = self.keep["L"][1], self.keep["U"][1]
        remaining = self.limit * NS - max(monotonic_ns() - self.tick, time_ns() - self.wall)
        if remaining <= 0:
            return
        source = self.out / "input.json"
        upper_doc = json.loads(upper.read_bytes())
        if upper_doc["parent"]["kind"] == "screen2":
            args = ["-B", str(AUDITOR.with_name("live_screen.py")), str(source), str(lower), str(upper),
                    str(monotonic_ns() + remaining)]
        else:
            args = ["-B", "-S", str(AUDITOR), "check-optimal", str(source), str(lower), str(upper)]
        try:
            subprocess.run([sys.executable, *args], env=self.env, check=True, timeout=remaining / NS)
        except subprocess.TimeoutExpired:
            return

        def reference(path):
            return {"file": path.relative_to(self.out).as_posix(),
                    "sha256": sha256(path.read_bytes()).hexdigest()}

        score = str(self.keep["L"][0])
        payload = encode({"schema": "mv-lattice-optimal-1", "input_sha256": self.fingerprint,
                          "value": score, "lower": reference(lower), "upper": reference(upper)})
        path = self.out / "events/optimal.json"
        temp = path.with_suffix(".part")
        temp.write_bytes(payload)
        temp.replace(path)
        mono, wall = monotonic_ns() - self.tick, time_ns() - self.wall
        if max(mono, wall) > self.limit * NS:
            path.unlink()
            return
        self.ledger.write(encode({"kind": "optimal", "score": score, "file": "events/optimal.json",
                                  "sha256": sha256(payload).hexdigest(), "elapsed_ns": mono,
                                  "wall_ns": wall}).decode())
        self.counter += 1
        self.proved_optimal = True
