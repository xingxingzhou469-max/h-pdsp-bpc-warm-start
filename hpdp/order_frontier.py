"""Native residual-order dominance and rank-bound experiments for 2–20 sites."""

import ctypes
import sys
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from time import monotonic

from .allocation import allocate
from .check import check
from .hfb_research import _numerator, _profile

TIME_INF = 65534
LIBRARY_NAME = "liborder_frontier.dylib" if sys.platform == "darwin" else "liborder_frontier.so"
COUNTERS = (
    "set_nodes", "set_objective_cuts", "set_local_time_cuts", "set_assignment_cuts",
    "tail_states", "order_states", "order_time_cuts", "order_flow_cuts", "scored_orders",
    "sets_ruled_out_by_tail", "order_prefix_fairness_cuts",
    "target_states", "target_closures",
    "order_truncated_prefix_calls", "order_truncated_prefix_cuts",
    "dominated_prefixes", "equivalent_leaves", "memo_checks", "memo_replacements",
    "prefix_rank_calls", "prefix_rank_cuts",
    "minimum_delivery_cuts", "minimum_remaining_stock_cuts",
)

_int = ctypes.c_int
_bound = ctypes.c_int64
_ctx = ctypes.c_void_p
_callback = ctypes.CFUNCTYPE(None, _int, ctypes.POINTER(_int), ctypes.POINTER(_bound))


@lru_cache(maxsize=1)
def _bind(path):
    lib = ctypes.CDLL(str(path))
    counter_count = getattr(lib, "hpdp_order_frontier_counter_count", None)
    if counter_count is None:
        raise ValueError("The native library predates this wrapper. Rebuild it from the matching order_frontier.cpp.")
    counter_count.argtypes = ()
    counter_count.restype = _int
    if counter_count() != len(COUNTERS):
        raise ValueError("The native library and wrapper use different counter layouts")
    signatures = (
        ("create", _ctx, (_int, _int, _bound, ctypes.POINTER(_int), ctypes.POINTER(_int),
                         ctypes.POINTER(_bound), ctypes.POINTER(_bound), ctypes.c_size_t,
                         ctypes.c_bool, ctypes.c_bool)),
        ("destroy", None, (_ctx,)),
        ("collect", _int, (_ctx, _bound, ctypes.c_double)),
        ("upper", _bound, (_ctx, ctypes.c_size_t)),
        ("mask", ctypes.c_uint32, (_ctx, ctypes.c_size_t)),
        ("inspect", _int, (_ctx, ctypes.c_size_t, _bound, ctypes.c_double, _callback)),
        ("count", _bound, (_ctx, _int)),
    )
    for name, restype, argtypes in signatures:
        entry = "inspect_v2" if name == "inspect" else name
        function = getattr(lib, f"hpdp_order_frontier_{entry}")
        function.restype = restype
        function.argtypes = argtypes
        setattr(lib, name, function)
    return lib


def supported(r):
    """Native time and quantity limits, independent of any instance identifier."""
    n = len(r.nodes)
    largest_flow = max(sum(r.supply), 2 * n * r.capacity)
    return 2 <= n <= 20 and r.limit < TIME_INF and largest_flow * r.people < 2**60


def _context(lib, r, cache_slots, prefix_rank_bound, minimum_delivery=False):
    n = len(r.nodes)
    ints = _int * (n + 1)
    bounds = _bound * (n + 1)
    times = _int * ((n + 1) ** 2)
    return lib.create(
        n, r.limit, r.capacity,
        times(*(min(v, TIME_INF) for row in r.travel for v in row)),
        ints(*(min(v, TIME_INF) for v in r.service)),
        bounds(*r.supply), bounds(*r.pop), cache_slots, prefix_rank_bound, minimum_delivery,
    )


def solve(r, deadline, seed, *, library=None, emit=None, cache_slots=1 << 16,
          prefix_rank_bound=False, minimum_delivery=False):
    """Search all sets that could beat the verified seed with exact residual bounds."""
    if not supported(r):
        raise ValueError("Native visit-set solver needs 2–20 sites and bounded integer resources")
    if cache_slots < 0 or cache_slots > (1 << 18) or cache_slots & (cache_slots - 1):
        raise ValueError("Cache slots must be zero or a power of two up to 262144")
    if minimum_delivery and (cache_slots or prefix_rank_bound):
        raise ValueError("Minimum-delivery ablation requires both earlier variants disabled")
    witness = seed
    lb = None if seed is None else check(r.instance, seed["route"], seed["amounts"], tol=0)["objective"]
    lb_num = -1 if lb is None else lb * r.units
    prior = Fraction(r.prior, r.units)
    started = monotonic()
    metrics = {"phase": "sets", "collect_seconds": 0.0, "audit_seconds": 0.0,
               "sets_to_audit": None, "sets_audited": 0, "improvements": 0,
               "cache_slots": cache_slots, "prefix_rank_bound": prefix_rank_bound,
               "minimum_delivery": minimum_delivery}
    exception = None

    def row(ub, *, exhausted=False):
        if lb is not None:
            ub = lb if ub is None else max(lb, ub)
        status = "timeout"
        if exhausted and lb is None:
            status = "infeasible"
        elif lb is not None and lb == ub:
            status = "optimal"
        return {
            "status": status, "lb": lb, "ub": ub,
            "lb_exact": None if lb is None else str(lb), "ub_exact": None if ub is None else str(ub),
            "route": None if witness is None else witness["route"],
            "amounts": None if witness is None else witness["amounts"],
            "certificate": "rational_order_block_frontier", "metrics": dict(metrics),
        }

    def checkpoint(ub):
        if emit is not None and monotonic() < deadline:
            emit(row(ub))

    if lb == prior or monotonic() >= deadline:
        return row(prior)
    if not r.p or not r.d:
        return row(None, exhausted=True)
    # Only the configured verified build is loaded. Build time is managed by
    # the run supervisor and no alternative solver is called on a load error.
    configured = Path(library) if library is not None else Path(__file__).with_name(LIBRARY_NAME)
    lib = _bind(configured.resolve())
    context = _context(lib, r, cache_slots, prefix_rank_bound, minimum_delivery)
    if not context:
        raise MemoryError("Unable to allocate native visit-set context")

    def measurements():
        metrics.update({key: lib.count(context, i) for i, key in enumerate(COUNTERS)})

    score = lru_cache(maxsize=50_000)(lambda profile: _numerator(r, profile))
    active_upper = prior

    @_callback
    def on_order(length, stops, reply):
        nonlocal lb, lb_num, witness, exception
        try:
            if monotonic() >= deadline:
                return
            route = tuple(stops[i] for i in range(length))
            value_num = score(_profile(r, route))
            if value_num <= lb_num:
                if monotonic() < deadline:
                    reply[0] = lb_num // 1
                return
            value, amounts = allocate(r.instance, route)
            verified = check(r.instance, route, amounts, tol=0)["objective"]
            if verified != value or verified * r.units != value_num or value > active_upper:
                raise ArithmeticError("Order evaluation disagrees with the visit-set certificate")
            if monotonic() >= deadline:
                return
            lb, lb_num = verified, value_num
            witness = {"route": route, "amounts": amounts}
            metrics["improvements"] += 1
            metrics["audit_seconds"] = monotonic() - audit_started
            measurements()
            checkpoint(active_upper)
            if monotonic() < deadline:
                reply[0] = lb_num // 1
        except BaseException as exc:
            exception = exc

    try:
        checkpoint(prior)
        size = lib.collect(context, lb_num // 1, deadline - monotonic())
        metrics["collect_seconds"] = monotonic() - started
        measurements()
        if size == -2:
            raise MemoryError("Native visit-set enumeration failed")
        if size == -1:
            return row(prior)
        metrics.update(phase="orders", sets_to_audit=size)
        audit_started = monotonic()
        next_save = audit_started
        for i in range(size):
            upper_num = lib.upper(context, i)
            active_upper = Fraction(upper_num, r.units)
            if upper_num <= lb_num:
                return row(lb, exhausted=True)
            metrics["active_mask"] = lib.mask(context, i)
            if monotonic() >= next_save:
                checkpoint(active_upper)
                next_save = monotonic() + 0.5
            status = lib.inspect(context, i, lb_num // 1, deadline - monotonic(), on_order)
            if exception is not None:
                raise exception
            metrics["audit_seconds"] = monotonic() - audit_started
            measurements()
            if status == -2:
                raise MemoryError("Native visit-set order audit failed")
            if status == -1:
                return row(active_upper)
            metrics["sets_audited"] += 1
        return row(lb, exhausted=True)
    finally:
        lib.destroy(context)
