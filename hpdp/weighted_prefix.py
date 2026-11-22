"""Native elementary prefix search with the existing priced-walk relaxation."""
import ctypes
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from math import gcd, isfinite
from pathlib import Path
import sys
from time import monotonic

from .allocation import allocate
from .check import check
from . import priced_routes as pr


Int = ctypes.c_int
Long = ctypes.c_int64
Byte = ctypes.c_uint8
Handle = ctypes.c_void_p
IntPtr = ctypes.POINTER(Int)
LongPtr = ctypes.POINTER(Long)
BytePtr = ctypes.POINTER(Byte)
Witness = ctypes.CFUNCTYPE(None, Int, BytePtr, Long, LongPtr)
Checkpoint = ctypes.CFUNCTYPE(None, Long, LongPtr, IntPtr)
COUNTERS = ("expanded", "generated", "pruned_time", "pruned_bound", "bound_calls",
            "walk_calls", "walk_tighter", "allocations", "max_heap", "max_dfs", "dfs_switches")


class Walk(ctypes.Structure):
    _fields_ = [("handle", Handle), ("prices", IntPtr), ("bits", IntPtr), ("full", Int),
                ("supply_weight", Long), ("population_weight", Long), ("unit", Long)]


@lru_cache(maxsize=2)
def library(path=None):
    suffix = "dylib" if sys.platform == "darwin" else "so"
    lib = ctypes.CDLL(str(path or Path(__file__).with_name(f"libweighted_prefix.{suffix}")))
    signatures = (
        ("create", Handle, [Int, Int, Long, Long, IntPtr, LongPtr, LongPtr, ctypes.POINTER(Walk),
                            Int, Handle, Int]),
        ("free", None, [Handle]),
        ("bound", Long, [Handle, Int, BytePtr]),
        ("search", Int, [Handle, Int, Int, Long, ctypes.c_double, Witness, Checkpoint, LongPtr, LongPtr]),
    )
    for name, result, arguments in signatures:
        entry = "search_v2" if name == "search" else name
        fn = getattr(lib, "hpdp_weighted_" + entry)
        fn.restype = result
        fn.argtypes = arguments
        setattr(lib, name, fn)
    return lib


def supported(r):
    return (len(r.p) > 5 and pr.supported(r) and 0 < r.prior < 2**60
            and sum(r.supply) + (len(r.nodes)+1) * r.capacity < 2**60
            and 0 < r.people < 2**60)


def rounded_weights(r, supply_weight, population_weight, unit=None):
    raw = [supply_weight*s + population_weight*p for s, p in zip(r.supply, r.pop)]
    unit = max(1, (max(raw)+1023)//1024) if unit is None else unit
    if min(supply_weight, population_weight, unit) <= 0:
        raise ValueError("Walk coefficients and the rounding unit must be positive")
    weights = [(value+unit-1)//unit for value in raw]
    prefix_limit = supply_weight*sum(r.supply) + population_weight*r.people
    tail_limit = max(weights)*unit*(r.limit+len(r.nodes))
    if max(weights) > 1024 or prefix_limit + tail_limit >= 2**62:
        raise OverflowError("Walk rewards exceed the native integer range")
    return weights, unit


class Context:
    def __init__(self, r, graph, tables, *, binary=None, fairness=True):
        self.lib = library(binary)
        descriptions = (Walk*len(tables))()
        keepalive = []
        for i, (table, a, b, unit) in enumerate(tables):
            prices, bits = pr.array(table.prices), pr.array(table.bits)
            keepalive.extend((table, prices, bits))
            descriptions[i] = Walk(table.handle, prices, bits, table.full, a, b, unit)
        self.keepalive = keepalive
        walking = None if not tables else ctypes.cast(tables[0][0].lib.value, Handle)
        supplies = (Long*graph.n)(*r.supply)
        people = (Long*graph.n)(*r.pop)
        self.handle = self.lib.create(graph.n, r.limit, r.capacity, r.prior, graph.times, supplies,
                                      people, descriptions, len(tables), walking, int(fairness))
        if not self.handle:
            raise RuntimeError("Unable to create native weighted prefix context")

    def bound(self, route):
        values = (Byte*len(route))(*route)
        value = self.lib.bound(self.handle, len(route), values)
        return None if value < 0 else value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.lib.free(self.handle)
        self.handle = None


def prepare_tables(r, graph, deadline, seed, tracked):
    # One common slope pair and up to two deterministic price plans. The first
    # root witness of the untracked relaxation selects explicit visits in the second.
    if seed is None:
        total_pop, total_supply = r.people, sum(r.supply)
    else:
        total_pop = sum(r.pop[v] for v in seed["route"])
        total_supply = sum(r.supply[v] for v in seed["route"])
    common = gcd(total_pop, total_supply)
    a, b = total_pop//common, total_supply//common
    weights, unit = rounded_weights(r, a, b)
    ordered = sorted(r.nodes, key=lambda v: (-weights[v], v))
    pivot = weights[ordered[len(ordered)//4]]
    prices = [max(0, weight-pivot) for weight in weights]
    tables = []

    def add(active_prices, active_tracked):
        if monotonic() >= deadline:
            return None
        table = pr.Table(graph, weights, active_prices, active_tracked, True, deadline)
        if table.handle:
            tables.append((table, a, b, unit))
            return table
        return None

    try:
        first = add(prices, ())
        while tracked and graph.n*(r.limit+1)*(1 << tracked)*12 > 64_000_000:
            tracked -= 1
        if first is None or tracked == 0 or monotonic() >= deadline:
            return tables
        scores = [(value, -i, start) for i, start in enumerate(graph.starts)
                  if (value := first.root(start, weights)) is not None]
        counts = [0]*graph.n
        if scores:
            _, _, start = max(scores)
            for v in (start.first, *first.route(start)):
                counts[v] += 1
        selected = set(sorted(ordered, key=lambda v: -max(0, counts[v]-1))[:tracked])
        second = [0 if v in selected else
                  max(0, min(weight, prices[v]+max(1, pivot//2)*(counts[v]-1)))
                  for v, weight in enumerate(weights)]
        add(second, sorted(selected))
        return tables
    except BaseException:
        for table, *_ in tables:
            table.close()
        raise


def solve(r, deadline, seed, *, emit=None, per_prefix=True, tracked=4, fairness=True,
          table_seconds=4.0, heap_limit=100_000, binary=None):
    """Return rational bounds and independent witnesses within a single budget."""
    if not supported(r):
        raise ValueError("Weighted prefix search requires at least six donors and supported integer resources")
    if not 0 <= tracked <= 8 or not 1 <= heap_limit < 2**31 or not isfinite(table_seconds) or table_seconds < 0:
        raise ValueError("Invalid tracked-site, table-time or frontier budget")
    start = monotonic()
    witness = seed
    lb = None if seed is None else check(r.instance, seed["route"], seed["amounts"], tol=0)["objective"]
    upper = r.prior
    stats = dict(variant="per_prefix" if per_prefix else "root_only", tables=[], phase="table_build",
                 fairness=fairness, heap_limit=heap_limit)
    failures = []

    def result():
        ub = None if upper < 0 else Fraction(upper, r.units)
        if lb is not None and (ub is None or lb > ub):
            raise ArithmeticError("Native prefix frontier excludes its independently checked witness")
        status = "infeasible" if ub is None else "optimal" if ub == lb else "timeout"
        return dict(status=status, lb=lb, ub=ub, lb_exact=None if lb is None else str(lb),
                    ub_exact=None if ub is None else str(ub), certificate="rational_weighted_prefix",
                    route=None if witness is None else witness["route"],
                    amounts=None if witness is None else witness["amounts"], metrics=stats.copy())

    def scaled(score):
        if score is None:
            return -1
        numerator = score*r.units
        if numerator.denominator != 1:
            raise ArithmeticError("Exact route objective does not have an integral native numerator")
        return numerator.numerator

    if monotonic() >= deadline:
        return result()
    if lb is not None and (lb*r.units).denominator!=1:
        # Fixed-route optima lie on the integer numerator lattice. An arbitrary
        # feasible allocation supplied by a caller need not be route-optimal.
        value,amounts=allocate(r.instance,witness["route"])
        if check(r.instance,witness["route"],amounts,tol=0)["objective"]!=value or value<lb:
            raise ArithmeticError("Reallocation of the supplied witness failed")
        if monotonic()>=deadline:
            return result()
        witness={"route":witness["route"],"amounts":amounts}
        lb=value
        if emit: emit(result())
    if scaled(lb) > upper:
        raise ArithmeticError("Seed exceeds the analytic upper bound")
    if scaled(lb) == upper or monotonic() >= deadline:
        return result()
    graph = pr.Graph(r, False)
    try:
        tables = prepare_tables(r, graph, min(deadline, start+min(table_seconds, (deadline-start)*0.4)),
                                witness, tracked)
    except OverflowError:
        tables=[]
        stats["walk_price_domain"]="reward range exceeded; integer resource bounds only"
    try:
        stats["table_build_seconds"] = monotonic()-start
        stats["tables"] = [dict(supply_weight=a, population_weight=b, unit=unit,
                                tracked=[v for v in r.nodes if table.bits[v]],
                                prices_sha256=sha256(str(table.prices).encode()).hexdigest())
                           for table, a, b, unit in tables]
        if monotonic() >= deadline:
            return result()
        stats["phase"] = "search"

        @Witness
        def offer(length, stops, parent_upper, reply):
            nonlocal lb, witness
            try:
                if monotonic() >= deadline:
                    return
                route = tuple(stops[:length])
                value, amounts = allocate(r.instance, route)
                if lb is None or value > lb:
                    verified = check(r.instance, route, amounts, tol=0)["objective"]
                    if verified != value or scaled(verified) > parent_upper:
                        raise ArithmeticError("Exact allocation contradicts the native active prefix")
                    if monotonic() >= deadline:
                        return
                    lb, witness = verified, {"route": route, "amounts": amounts}
                if monotonic() < deadline:
                    reply[0] = scaled(lb)
            except BaseException as exc:
                failures.append(exc)

        @Checkpoint
        def progress(bound, counters, proceed):
            nonlocal upper
            try:
                if monotonic() >= deadline:
                    return
                upper = min(upper, bound)
                stats.update(zip(COUNTERS, counters[:len(COUNTERS)]))
                if emit:
                    emit(result())
                if monotonic() < deadline:
                    proceed[0] = 1
            except BaseException as exc:
                failures.append(exc)

        native_upper = Long(upper)
        counters = (Long*len(COUNTERS))()
        with Context(r, graph, tables, binary=binary, fairness=fairness) as context:
            status = context.lib.search(context.handle, int(per_prefix), heap_limit, scaled(lb),
                                        max(0, deadline-monotonic()), offer, progress,
                                        ctypes.byref(native_upper), counters)
        if failures:
            raise failures[0]
        if status < 0:
            raise RuntimeError(f"Native weighted prefix search failed ({status})")
        stats.update(zip(COUNTERS, counters))
        stats["search_complete"] = status == 1
        upper = min(upper, native_upper.value)
        return result()
    finally:
        for table, *_ in tables:
            table.close()
