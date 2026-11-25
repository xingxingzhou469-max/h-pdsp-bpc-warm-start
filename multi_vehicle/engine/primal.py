"""Cold PartitionSearch control and tight-budget rebuild-first primal arm."""
import ctypes as ct
from hashlib import sha256
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

if __package__:
    from .common import number
    from .primal_vendor.fleet import canonical_column
    from .primal_vendor.partition import PartitionSearch
    from .primal_vendor.routes import route_time
else:
    from common import number
    from primal_vendor.fleet import canonical_column
    from primal_vendor.partition import PartitionSearch
    from primal_vendor.routes import route_time


NANO = 10**9
SEED = 7
VERIFY_RESERVE_NS = 100_000_000
SOURCE = Path(__file__).resolve().parent / "primal_vendor" / "rebuild.cpp"
BUILD = Path(__file__).resolve().parents[1] / "build"
FLAGS = ("-std=c++17", "-O3", "-Wall", "-Wextra", "-pedantic", "-fPIC", "-shared")
PUBLISH = ct.CFUNCTYPE(None, ct.c_char_p, ct.c_char_p)


def prepare(instance, input_path, arm):
    """Prepare code only. Each measured run rebuilds its own routes and pool."""
    if arm == "bare":
        return None
    if arm != "upgrade":
        raise ValueError(f"unknown primal arm {arm!r}")
    build = Path(os.environ.get("NG_BB_BUILD", BUILD))
    build.mkdir(parents=True, exist_ok=True)
    flags = (*FLAGS, "-Werror")
    if os.environ.get("NG_BB_SANITIZE") == "1":
        flags += ("-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all")
    fingerprint = sha256(SOURCE.read_bytes() + repr(flags).encode()).hexdigest()
    binary = build / f"rebuild-{fingerprint[:20]}.so"
    if not binary.is_file():
        with tempfile.TemporaryDirectory(dir=build, prefix="compile-") as output:
            temporary = Path(output) / "rebuild"
            compiler = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("c++") or "c++"
            subprocess.run([compiler, *flags, str(SOURCE), "-o", str(temporary)], check=True)
            os.replace(temporary, binary)
    run = ct.CDLL(str(binary)).rebuild_fleet
    run.argtypes = (ct.c_char_p, PUBLISH)
    run.restype = ct.c_int
    return run


def _publish(instance, routes, deadline_ns, emit, source):
    if time.monotonic_ns() >= deadline_ns - VERIFY_RESERVE_NS // 2:
        return
    routes = [list(route) for route in routes if route]
    if not routes or any(route_time(instance, route) > number(instance["limit"]) for route in routes):
        return
    amounts = [list(map(str, canonical_column(instance, route)[1])) for route in routes]
    if time.monotonic_ns() < deadline_ns - VERIFY_RESERVE_NS // 2:
        emit({"routes": routes, "amounts": amounts}, source)


def _payload(instance, vehicles, seconds):
    supply, people = instance["supply"], instance["population"]
    service, travel = instance["service"], instance["travel"]
    size = len(supply)
    rows = [
        [size - 1, vehicles, SEED, seconds, float(number(instance["capacity"])), float(number(instance["limit"]))],
        [float(number(n)) for n in supply],
        [float(number(n)) for n in people],
        [float(number(row[0])) for row in travel],
    ]
    services = tuple(map(number, service))
    rows.extend([float(number(cost) + services[j]) for j, cost in enumerate(row)] for row in travel)
    return ("\n".join(" ".join(map(str, row)) for row in rows) + "\n").encode()


def _decode(payload, vehicles):
    stops = iter(map(int, payload.split()))
    routes = [[next(stops) for _ in range(next(stops))] for _ in range(vehicles)]
    if next(stops, None) is not None:
        raise ValueError("unexpected trailing stops in native primal witness")
    return routes


def _run_native(instance, vehicles, deadline_ns, emit, native):
    seconds = (deadline_ns - time.monotonic_ns() - VERIFY_RESERVE_NS) / NANO
    if seconds <= 0:
        return
    errors = []

    @PUBLISH
    def publish(source, payload):
        if errors:
            return
        try:
            _publish(instance, _decode(payload, vehicles), deadline_ns, emit, "S-" + source.decode("ascii"))
        except Exception as error:
            errors.append(error)

    status = native(_payload(instance, vehicles, seconds), publish)
    if errors:
        raise errors[0]
    if status:
        raise RuntimeError(f"native primal failed with status {status}")


def run_primal(instance, vehicles, arm, deadline_ns, emit, prepared=None):
    """Publish exact-quantity candidate witnesses by an absolute monotonic deadline."""
    if arm == "bare":
        deadline = (deadline_ns - VERIFY_RESERVE_NS) / NANO
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            return
        search = PartitionSearch(instance, vehicles, seed=SEED)
        search.search(
            seconds=seconds, seed=SEED, deadline=deadline,
            publish=lambda routes: _publish(instance, routes, deadline_ns, emit, "PartitionSearch"),
        )
    elif arm == "upgrade":
        if prepared is None:
            raise ValueError("prepare the native primal worker before starting the budget")
        _run_native(instance, vehicles, deadline_ns, emit, prepared)
    else:
        raise ValueError(f"unknown primal arm {arm!r}")
