"""Exhaustive directed-walk cross-check for the zone recurrence."""

from fractions import Fraction
from random import Random
from time import monotonic

from check import priced, replay
from solve import Native, legal_zone
from route.model import Projection, product_bound, read_model


def enum(graph, rewards, zones):
    n, limit = len(rewards), graph.horizon
    best = 0

    def dfs(path, cost, score):
        nonlocal best
        i = path[-1]
        if cost + graph.finish[i] <= limit:
            best = max(best, score)
        if graph.layered and len(path) == n:
            return
        for j, a in enumerate(graph.arcs[i]):
            if cost + a > limit or (len(path) > 1 and j == path[-2]):
                continue
            trail = (*path, j)
            if legal_zone(trail, zones):
                dfs(trail, cost + a, score + rewards[j])

    for i, cost in enumerate(graph.start):
        if cost <= limit:
            dfs((i,), cost, rewards[i])
    return best


def main():
    rng, n = Random(10483), 5
    cheap = [[8] * n for _ in range(n)]
    for i, j in ((0, 1), (1, 2), (2, 0), (2, 3)):
        cheap[i][j] = 1
    graphs = [Projection(7, cheap, (0, 8, 8, 8, 8), (8, 8, 8, 0, 8))]
    for sample in range(18):
        h = 0 if sample == 0 else 6
        low = int(sample % 2 == 0 and h != 0)
        arcs = tuple(tuple(rng.randint(low, h + 1) if i != j else h + 1
                           for j in range(n)) for i in range(n))
        graphs.append(Projection(h, arcs, (0, h + 1, h + 1, 0, h + 1),
                                 (h + 1, h + 1, 0, h + 1, 0)))
    checked = 0
    partitions = [tuple(range(n)), (0, 0, 0, 3, 3), (0,) * n]
    for index, graph in enumerate(graphs):
        scores = tuple(rng.randrange(-11, 15) for _ in range(n)) if index else (7, 8, 9, 6, 1)
        native = Native(graph)
        for zones in partitions:
            for reverse in (False, True):
                oriented = graph.reversed() if reverse else graph
                truth = enum(oriented, scores, zones)
                result = native.price(scores, zones, reverse, monotonic() + 4)
                assert result is not None and result[0] == truth == priced(oriented, scores, zones), (
                    index, zones, reverse, result, truth)
                for path in result[1]:
                    assert legal_zone(path, zones)
                    assert all(path[i] != path[i + 2] for i in range(len(path) - 2))
                    length = (graph.start[path[0]] + graph.finish[path[-1]]
                              + sum(graph.arcs[i][j] for i, j in zip(path, path[1:])))
                    assert length <= graph.horizon
                checked += 1
        if not index:
            assert enum(graph, scores, partitions[1]) < enum(graph, scores, partitions[0])
    instance = {
        "capacity": "4/7", "limit": 0, "travel": [[0] * n for _ in range(n)],
        "service": [0] * n, "population": [0, 0, "3/2", 0, 5],
        "supply": [0, "2/5", 0, "5/9", 0],
    }
    model = read_model(instance)
    beta = Fraction(2, 7)
    part = [0] * model.n
    cap = enum(model.project(), model.weights(beta), part)
    upper = product_bound(model, tuple(model.boxes(2)), [(beta, 2 * cap * model.reward_unit)])[0]
    certificate = {
        "method": "zone-price-v1", "vehicles": 2, "upper": str(upper),
        "supports": [{"prices": [0] * model.n, "beta": str(beta), "route_bound": cap,
                      "partition": part, "reverse": False}],
    }
    assert replay(instance, 2, certificate) == upper
    certificate["supports"][0]["route_bound"] -= 1
    try:
        replay(instance, 2, certificate)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered price accepted")
    print(f"{checked} exact native/enumeration/replay pricing triples and a rational envelope")


if __name__ == "__main__":
    main()
