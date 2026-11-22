"""SciPy HiGHS and connectivity cut primitives used by the V8 routing LP."""

import time
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, milp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow


HIGHS_OPTIONS = {"threads": 1, "random_seed": 0, "presolve": True}


LP_OPTIONS = {
    **HIGHS_OPTIONS,
    "primal_feasibility_tolerance": 1e-8,
    "dual_feasibility_tolerance": 1e-8,
}


def _run_highs(solver, **kwargs):
    # The public SciPy wrappers explicitly forward these native HiGHS options.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected:.*",
                                category=RuntimeWarning if solver is milp else OptimizeWarning)
        return solver(**kwargs)


def _violated_cuts(x, v, deadline):
    """Yield a violated outward directed cut for each separated agency or supplier."""
    # Integer capacities select candidate sets. Test violation with the original x.
    graph = csr_matrix(np.rint(np.clip(x, 0, 1) * 10_000_000).astype(np.int32))
    for i in range(1, len(v)):
        if time.monotonic() >= deadline:
            return
        if v[i] <= 1e-7:
            continue
        flow = maximum_flow(graph, i, 0).flow
        residual = graph - flow
        residual.eliminate_zeros()
        reachable = breadth_first_order(residual, i, directed=True, return_predecessors=False)
        if 0 in reachable:
            continue
        outgoing = np.ones(len(v), dtype=bool)
        outgoing[reachable] = False
        if np.sum(x[np.ix_(reachable, outgoing)]) < v[i] - 1e-7:
            yield frozenset(map(int, reachable)), i
