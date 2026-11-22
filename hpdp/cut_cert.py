"""Encode verified cut specifications as integer rows of the routing model."""

from heapq import heappop, heappush

from .prefix import Resources
from .routing_exact import Row, routing


def _site_set(values, model, *, allow_depot=False):
    if (type(values) is not list or not values
            or any(type(v) is not int or not (0 if allow_depot else 1) <= v < model.nodes
                   for v in values) or values != sorted(set(values))):
        raise ValueError("Invalid cut site set")
    return set(values)


def _distances(graph, source, forbidden=()):
    """Shortest arc-and-service distances with the depot excluded as transit."""
    best = [None] * len(graph)
    best[source] = 0
    frontier = [(0, source)]
    while frontier:
        elapsed, i = heappop(frontier)
        if elapsed != best[i]:
            continue
        for j, length in graph[i]:
            if j == 0 or j in forbidden:
                continue
            distance = elapsed + length
            if best[j] is None or distance < best[j]:
                best[j] = distance
                heappush(frontier, (distance, j))
    return best


class CutEncoder:
    """Reconstruct valid cut rows from a single immutable routing model."""

    def __init__(self, r, model):
        self.r = r
        self.model = model
        self.outgoing = self.incoming = self.return_time = None
        self.before_entry = {}
        self.to_target = {}

    def _site_target(self, spec):
        side = _site_set(spec["set"], self.model)
        k = spec["target"]
        if type(k) is not int or k not in side:
            raise ValueError("A connectivity cut must contain its nondepot target")
        return side, k

    def _sec(self, spec):
        model = self.model
        side, k = self._site_target(spec)
        outside = set(range(model.nodes)) - side
        # Equal degrees make both directions of the cut identical. Encode the
        # internal arcs of its smaller side using the degree equalities.
        inside = side if len(side) <= len(outside) else outside
        terms = [(model.visits[v], -1) for v in sorted(inside - {0, k})]
        if k not in inside:
            terms.append((model.visits[k], 1))
        for i in sorted(inside):
            for j in sorted(inside):
                col = model.arcs.get((i, j))
                if col is not None:
                    terms.append((col, 1))
        return Row(tuple(terms), int(0 in inside))

    def _blossom(self, spec):
        model = self.model
        side = _site_set(spec["handle"], model, allow_depot=True)
        teeth = spec["teeth"]
        if (len(side) == model.nodes or type(teeth) is not list or len(teeth) < 3
                or len(teeth) % 2 == 0
                or any(type(edge) is not list or len(edge) != 2
                       or any(type(v) is not int for v in edge)
                       or not 0 <= edge[0] < edge[1] < model.nodes for edge in teeth)
                or teeth != sorted(teeth)):
            raise ValueError("A blossom needs a proper handle and an odd list of teeth")
        edges = set()
        used = set()
        for i, j in teeth:
            if ((i in side) == (j in side) or i in used or j in used
                    or ((i, j) not in model.arcs and (j, i) not in model.arcs)):
                raise ValueError("Blossom teeth must be disjoint edges crossing the handle")
            edges.add((i, j))
            used.update((i, j))
        # A directed cycle crosses the handle an even number of times. A
        # route with two opposing arcs on one edge has no other tooth visits.
        terms = tuple((col, 1 if (min(i, j), max(i, j)) in edges else -1)
                      for (i, j), col in model.arcs.items() if (i in side) != (j in side))
        return Row(terms, len(edges) - 1)

    def _precedence_entry(self, spec):
        r, model = self.r, self.model
        side, k = self._site_target(spec)
        if self.outgoing is None:
            self.outgoing = [[] for _ in range(model.nodes)]
            self.incoming = [[] for _ in range(model.nodes)]
            for i, j in model.arcs:
                length = r.travel[i][j] + r.service[j]
                self.outgoing[i].append((j, length))
                self.incoming[j].append((i, length))
            self.return_time = _distances(self.incoming, 0)
        if self.return_time[k] is None:
            return Row(((model.visits[k], 1),), 0)
        key = tuple(spec["set"])
        if key not in self.before_entry:
            self.before_entry[key] = _distances(self.outgoing, 0, side)
        if k not in self.to_target:
            self.to_target[k] = _distances(self.incoming, k)
        start, to_target = self.before_entry[key], self.to_target[k]
        interval = r.limit - self.return_time[k]
        # Every route that visits k must cross a first entry arc into the set.
        # The shortest paths below lower bound its prefix and time to k.
        terms = [(model.visits[k], 1)]
        for (i, j), col in model.arcs.items():
            if (i not in side and j in side and start[i] is not None and to_target[j] is not None
                    and start[i] + r.travel[i][j] + r.service[j] + to_target[j] <= interval):
                terms.append((col, -1))
        return Row(tuple(terms), 0)

    def row(self, spec):
        if type(spec) is dict and type(spec.get("type")) is str:
            kind = spec["type"]
            if kind in ("SEC", "B_PRECEDENCE_ENTRY") and set(spec) == {"type", "set", "target"}:
                return self._sec(spec) if kind == "SEC" else self._precedence_entry(spec)
            if kind == "A_BLOSSOM" and set(spec) == {"type", "handle", "teeth"}:
                return self._blossom(spec)
        raise ValueError("Unrecognized cut specification")


def cut_row(r, model, spec):
    return CutEncoder(r, model).row(spec)


def validate_cut(instance, cut_spec):
    """Check a cut's schema and model-dependent legality."""
    r = Resources(instance)
    try:
        cut_row(r, routing(r, pickup_blocks=True), cut_spec)
    except ValueError:
        return False
    return True
