"""Bounded subset phase followed by exact prefix search, carrying both certificates."""
from fractions import Fraction
from time import monotonic

from .prefix import solve_prefix
from .subset import solve_subset


def solve_portfolio(r, deadline, seed, *, emit=None):
    t = monotonic()
    dp_end = min(deadline, t + min(4.0, max(0,deadline-t)/2))
    def subset_checkpoint(row):
        if emit:
            emit(dict(row, certificate="rational_portfolio",
                      metrics={"phase":"subset","subset":row["metrics"]}))
    first = solve_subset(r, dp_end, seed, emit=subset_checkpoint)
    if first["status"] in ("optimal","infeasible"):
        result = dict(first, certificate="rational_portfolio")
        result["metrics"] = {"closed_by":"subset","subset":first["metrics"],"subset_seconds":monotonic()-t}
        return result

    initial = None if first["route"] is None else {"route":first["route"],"amounts":first["amounts"]}
    dp_seconds = monotonic()-t
    def combined(prefix):
        row = dict(prefix)
        # Both UBs cover the whole original instance. The second phase starts
        # with the strictly feasible incumbent exported by the first phase.
        if row["ub"] is not None and first["ub"] is not None:
            row["ub"] = min(Fraction(row["ub"]),Fraction(first["ub"]))
            row["ub_exact"] = str(row["ub"])
            if row["lb"] is not None:
                if row["lb"] > row["ub"]:
                    raise ArithmeticError("Contradictory phase bounds")
                if row["lb"] == row["ub"]:
                    row["status"] = "optimal"
        row["certificate"] = "rational_portfolio"
        row["metrics"] = {"closed_by":"prefix" if row["status"] in ("optimal","infeasible") else None,
                          "subset":first["metrics"],"subset_status":first["status"],
                          "subset_seconds":dp_seconds,"prefix":prefix["metrics"]}
        return row
    def checkpoint(prefix):
        if emit:
            emit(combined(prefix))
    return combined(solve_prefix(r,deadline,initial,prize=True,emit=checkpoint))
