"""Exact small-mask fleet merging shared by producer and certificate checker."""

MISSING = -(1 << 61)
CELL_LIMIT = 18_000_000


def merge(values, vehicles):
    """Return one maximizing class sequence for each reachable total mask."""
    current = {0: (0, ())}
    available = [(m, v) for m, v in enumerate(values) if v is not None]
    for _ in range(vehicles):
        following = {}
        for mask, (score, used) in current.items():
            for visit, value in available:
                if mask & visit:
                    continue
                full = mask | visit
                candidate = score + value
                if full not in following or candidate > following[full][0]:
                    following[full] = (candidate, (*used, visit))
        current = following
    return sorted(current.values(), reverse=True)
