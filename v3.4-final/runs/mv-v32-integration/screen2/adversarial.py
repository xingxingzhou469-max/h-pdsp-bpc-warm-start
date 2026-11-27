"""Small exhaustive checks of elementary masks, idle coverage and threshold ties."""

from fractions import Fraction as F
from itertools import permutations
from time import monotonic

from check import (METHOD, enumerate_masks, excludes_fleet, identity, mask_digest,
                   read_model, replay, threshold_data)
from screen2.solve import Screen


def brute_masks(graph, reward, threshold):
    nodes = range(len(reward))
    result = {0} if threshold <= 0 else set()
    for size in range(1, len(reward) + 1):
        for path in permutations(nodes, size):
            duration = graph.start[path[0]] + graph.finish[path[-1]]
            duration += sum(graph.arcs[i][j] for i, j in zip(path, path[1:]))
            if duration <= graph.horizon and sum(reward[i] for i in path) >= threshold:
                result.add(sum(1 << i for i in path))
    return result


def main():
    for internal in (F(1, 2), F(0)):
        half = F(1, 2)
        instance = {
            "capacity": 2, "limit": 3 * half + internal,
            "supply": [0, 0, 2, 0], "population": [0, 1, 0, 1],
            "service": [0, 0, 0, 0],
            "travel": [[0, half, 10, 10], [10, 0, half, 10],
                       [10, 10, 0, internal], [half, 10, 10, 0]],
        }
        model = read_model(instance)
        graph = model.project()
        native = Screen(graph)
        line = {"beta": "1", "prices": [0, 0, 0], "partition": [0, 1, 1],
                "reverse": False, "route_bound": 3 * (1 << 21)}
        for target in (F(1), F(99, 100)):
            required, cutoff, weights, rewards = threshold_data(model, line, target)
            brute = brute_masks(graph, rewards, cutoff)
            compiled, _ = native.enumerate(rewards, line["partition"], cutoff, False,
                                           monotonic() + 10)
            independent = enumerate_masks(graph.reversed(), rewards, line["partition"], cutoff)
            assert brute == compiled == independent == {0, 0b110}
            assert excludes_fleet(model, brute, weights, required, target) == (target == 1)
            if target == 1:
                parent = {"method": "zone-price-v1", "vehicles": 2,
                          "upper": "2", "supports": [line]}
                proof = {"method": METHOD, "vehicles": 2, "instance_digest": identity(instance, 2),
                         "upper": "1", "parent": parent,
                         "screen": {"line_index": 0, "mask_count": len(brute),
                                    "mask_sha256": mask_digest(brute)}}
                assert replay(instance, 2, proof) == 1
    print("brute force, native enumeration, and independent replay agree")


if __name__ == "__main__":
    main()
