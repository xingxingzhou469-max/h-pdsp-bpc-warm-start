"""Verify and supervise the exact few-supplier subprocess."""

from fractions import Fraction
from math import isfinite
import os
from pathlib import Path
import selectors
import subprocess
from tempfile import TemporaryDirectory
from time import monotonic

from .allocation import allocate
from .check import check


_INT = (1 << 63) - 1
_WIDE_LIMIT = 10**36
_STATS = ("expanded", "contexts_started", "table_cut", "time_population_cut", "equity_cut",
          "dominance_cut", "route_allocations")


def _domain(resources):
    r = resources
    if not (1 <= len(r.p) <= 4 and 1 <= len(r.d) <= 62 and len(r.nodes) <= 66):
        return None, "supplier_or_agency_count"
    if sum(Fraction(str(x)) for x in r.instance["supply"]) > Fraction(str(r.instance["capacity"])):
        return None, "binding_capacity"
    quantum = max(1, (r.limit+749)//750)
    # The grid may relax resource constraints. The subprocess traverses and
    # tests return feasibility with the original exact, wide integer times.
    resolution = r.limit//quantum
    if (r.limit >= _WIDE_LIMIT or r.capacity > _INT or r.people*(resolution+2) > _INT//4
            or r.prior >= 10*_WIDE_LIMIT):
        return None, "wide_numeric_range"
    return quantum, "supported"


def is_supported(resources):
    """Return whether the exact domain includes this instance."""
    return _domain(resources)[0] is not None


def _model(resources, lower):
    r = resources
    quantum, reason = _domain(r)
    if quantum is None:
        return None, reason
    cutoff = 0 if lower is None else (lower*r.units)//1
    clip = r.limit+1
    payload = [[len(r.travel), r.capacity, r.limit, quantum, cutoff],
               r.supply, r.pop, [min(t, clip) for t in r.service],
               *[[min(t, clip) for t in row] for row in r.travel]]
    return "\n".join(" ".join(map(str, r)) for r in payload)+"\n", quantum


def solve_chain(resources, deadline, seed, *, executable=None, emit=None, cancelled=None, tracked=4):
    """Solve the original instance, acknowledging only freshly checked incumbents.

    An emit callback can return False to stop work. An optional cancelled
    predicate is polled while the isolated worker has outstanding work.
    """
    r = resources
    if not isfinite(deadline):
        raise ValueError("Use a finite monotonic deadline")
    start = monotonic()
    lower = None if seed is None else check(r.instance, seed["route"], seed["amounts"], tol=0)["objective"]
    upper = Fraction(r.prior, r.units)
    witness = seed
    metrics = dict(phase="analytic_prior", support="supported", improvements=0, stop_reason=None)

    def row(lb, ub, solution):
        if ub is None:
            status = "infeasible"
        elif lb == ub:
            status = "optimal"
        elif metrics["stop_reason"] in ("native_worker_exited", "worker_ended_during_ack"):
            status = "interrupted"
        else:
            status = "timeout"
        return dict(
            status=status,
            lb=lb, ub=ub, lb_exact=None if lb is None else str(lb), ub_exact=None if ub is None else str(ub),
            route=None if solution is None else solution["route"],
            amounts=None if solution is None else solution["amounts"],
            certificate="rational_chain_frontier", metrics={**metrics, "seconds": monotonic()-start},
        )

    def stopped():
        if monotonic() >= deadline:
            metrics["stop_reason"] = "deadline"
        elif cancelled is not None and cancelled():
            metrics["stop_reason"] = "cancelled"
        return metrics["stop_reason"] is not None

    def publish(lb, ub, route):
        if stopped():
            return False
        if emit is not None and emit(row(lb, ub, route)) is False:
            metrics["stop_reason"] = "callback_cancelled"
        return not stopped()

    if lower is not None and lower > upper:
        raise ArithmeticError("Verified seed exceeds the analytic prior")
    if lower == upper or stopped():
        return row(lower, upper, witness)
    if type(tracked) is not int or not 0 <= tracked <= 7:
        raise ValueError("tracked must be an integer between zero and seven")
    payload, detail = _model(r, lower)
    if payload is None:
        metrics["support"] = detail
        return row(lower, upper, witness)
    metrics["grid_quantum"] = detail
    if executable is None:
        executable = Path(__file__).with_name("chain-worker")

    with TemporaryDirectory(prefix="hpdp-chain-") as work:
        staged = Path(work, "instance.txt")
        staged.write_text(payload, encoding="ascii")
        if stopped():
            return row(lower, upper, witness)
        command = [str(Path(executable).resolve()), str(staged), str(tracked),
                   str(min(3600.0, max(1e-9, deadline-monotonic())))]
        with (Path(work) / "stderr").open("w+b") as errors:
            launched = monotonic()
            with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                                  bufsize=0, cwd=work) as worker:
                finished = natural = False
                metrics["native_pid"] = worker.pid
                try:
                    with selectors.DefaultSelector() as events:
                        events.register(worker.stdout, selectors.EVENT_READ)
                        buffer = b""
                        eof = False
                        while not stopped() and not eof and not finished:
                            wait = max(0, deadline-monotonic())
                            if cancelled is not None:
                                wait = min(wait, 0.025)
                            if not events.select(wait):
                                continue
                            data = os.read(worker.stdout.fileno(), 16384)
                            if not data:
                                eof = True
                            buffer += data
                            while b"\n" in buffer and not stopped():
                                raw, buffer = buffer.split(b"\n", 1)
                                kind, *fields = raw.decode("ascii").split()
                                if kind == "domain":
                                    metrics.update(tracked=int(fields[0]), walk_table=bool(int(fields[1])))
                                elif kind == "candidate":
                                    native, *route = list(map(int, fields))
                                    value, amounts = allocate(r.instance, route)
                                    exact = check(r.instance, route, amounts, tol=0)["objective"]
                                    if value != exact or value*r.units != native or value > upper:
                                        raise ArithmeticError("Native incumbent fails original-input verification")
                                    if lower is not None and value <= lower:
                                        raise ArithmeticError("Native incumbent does not strictly improve")
                                    candidate = {"route": route, "amounts": amounts}
                                    if stopped():
                                        break
                                    metrics["improvements"] += 1
                                    metrics["phase"] = "incumbent"
                                    go = publish(value, upper, candidate)
                                    if not go and metrics["stop_reason"] != "callback_cancelled":
                                        metrics["improvements"] -= 1
                                        break
                                    lower, witness = value, candidate
                                    if not go or lower == upper:
                                        finished = True
                                        break
                                    try:
                                        worker.stdin.write(f"approve {native}\n".encode("ascii"))
                                    except BrokenPipeError:
                                        metrics["stop_reason"] = "worker_ended_during_ack"
                                elif kind in ("bound", "done"):
                                    score, *stats = map(int, fields)
                                    updated = min(upper, Fraction(score, r.units))
                                    if lower is not None and updated < lower:
                                        raise ArithmeticError("Native frontier contradicts verified lower bound")
                                    metrics.update(zip(_STATS, stats, strict=True))
                                    metrics["phase"] = "closed" if kind == "done" else "frontier"
                                    if kind == "done":
                                        if lower is None:
                                            if updated:
                                                raise ArithmeticError("Unresolved frontier reported complete")
                                            updated = None
                                        elif lower != updated:
                                            raise ArithmeticError("Complete search left a nonzero strict interval")
                                    go = publish(lower, updated, witness)
                                    if go or metrics["stop_reason"] == "callback_cancelled":
                                        upper = updated
                                    if not go or kind == "done" or lower == upper:
                                        natural = kind == "done"
                                        finished = True
                                        break
                                elif kind == "timeout":
                                    natural = True
                                    metrics["stop_reason"] = "native_deadline"
                                else:
                                    raise ArithmeticError(f"Unexpected chain solver event {kind!r}")
                        if eof and not finished and metrics["stop_reason"] is None and not stopped():
                            errors.seek(0)
                            reason = errors.read(1000).decode(errors="replace").strip()
                            metrics.update(stop_reason="native_worker_exited", native_error=reason)
                finally:
                    if natural and worker.poll() is None:
                        try:
                            worker.wait(timeout=min(0.010, max(0, deadline-monotonic())))
                        except subprocess.TimeoutExpired:
                            pass
                    if worker.poll() is None:
                        worker.kill()
                    metrics["native_returncode"] = worker.wait()
                    metrics["native_process_seconds"] = monotonic()-launched
    return row(lower, upper, witness)
