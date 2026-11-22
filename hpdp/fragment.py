"""Bounded V5 incumbent phase using the native fragment repair study."""

import ctypes as c
from pathlib import Path
from time import monotonic

from .allocation import allocate
from .check import check

PUBLISH = c.CFUNCTYPE(c.c_int, c.POINTER(c.c_int), c.c_int, c.c_double)


def improve(r, deadline, seed, *, emit):
    n = len(r.pop)
    # The native ranking uses double; fixed-route allocation stays exact.
    # Operate only on integer quantities and bounded normalized populations.
    if r.flow_scale != 1 or r.people * sum(r.supply) >= 2**53 or r.capacity >= 2**53:
        return {}
    arc = [r.travel[i][j] + r.service[j] for i in range(n) for j in range(n)]
    if not 36 <= n <= 128 or len(r.p) < 2 or max(r.limit, *arc) * n >= 2**62:
        return {}
    if r.service[0] != 0 or not 0 < r.capacity or monotonic() >= deadline:
        return {}

    lib = c.CDLL(str(Path(__file__).with_name("libfragment.so")))
    func = lib.route_search
    func.argtypes = (
        c.c_int, c.c_int64, c.POINTER(c.c_int64), c.POINTER(c.c_double),
        c.POINTER(c.c_double), c.c_double, c.POINTER(c.c_int), c.c_int,
        c.c_uint64, c.c_int, c.c_double, PUBLISH, c.POINTER(c.c_int64),
    )
    func.restype = None
    arc = (c.c_int64 * (n * n))(*arc)
    supply = (c.c_double * n)(*(v / r.flow_scale for v in r.supply))
    pop = (c.c_double * n)(*r.pop)
    length = len(seed["route"])
    route = (c.c_int * length)(*seed["route"])
    counters = (c.c_int64 * 5)()
    failure = None

    @PUBLISH
    def report(pointer, size, _estimate):
        nonlocal failure
        try:
            order = tuple(pointer[:size])
            exact, amounts = allocate(r.instance, order)
            checked = check(r.instance, order, amounts, tol=0)["objective"]
            if exact != checked:
                raise ArithmeticError("Fragment repair witness failed exact scoring")
            if monotonic() >= deadline:
                return 0
            emit({"route": order, "amounts": amounts}, checked)
            return 1
        except BaseException as error:
            failure = error
            return 0

    remaining = deadline - monotonic()
    if remaining > 0:
        func(n, r.limit, arc, supply, pop, r.capacity / r.flow_scale, route, length,
             20261119, 1, remaining, report, counters)
    if failure is not None:
        raise failure
    return dict(zip(("iterations", "feasible_sampled_moves", "bridges",
                     "completed_bridges", "resets"), counters))
