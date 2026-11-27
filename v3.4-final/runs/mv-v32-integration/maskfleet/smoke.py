"""Exhaustive marked pricing and full-model two-car support checks."""

from fractions import Fraction
from itertools import product
from pathlib import Path
from random import Random
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from maskfleet.check import classes, replay
from maskfleet.core import merge
from maskfleet.solve import Native
from route.model import product_bound, read_model

sys.path.insert(0, str(ROOT / "route"))
from route.adversarial import truth


def brute_classes(graph, reward, partition, marked):
    cap = graph.horizon
    tag = {i: 1 << j for j, i in enumerate(marked)}
    best = [0] + [None] * ((1 << len(marked)) - 1)

    def dfs(path, time, current_cell, visited, mask, score):
        i = path[-1]
        if time + graph.finish[i] <= cap and (best[mask] is None or score > best[mask]):
            best[mask] = score
        if graph.layered and len(path) == len(partition):
            return
        for j, t in enumerate(graph.arcs[i]):
            if time + t > cap or (len(path) > 1 and j == path[-2]):
                continue
            m = tag.get(j, 0)
            if mask & m:
                continue
            same = partition[j] == current_cell
            if same and j in visited:
                continue
            dfs((*path, j), time + t, partition[j], visited | {j} if same else {j},
                mask | m, score + reward[j])

    for i, time in enumerate(graph.start):
        if time <= cap:
            dfs((i,), time, partition[i], {i}, tag.get(i, 0), reward[i])
    return best


def main():
    rng, n = Random(10403), 5
    fixtures = []
    for i in range(8):
        fixtures.append({
            "capacity": Fraction(2 + (i % 2), 2),
            "limit": 6 if i > 1 else 0,
            "supply": [0, Fraction(7, 3), Fraction((i > 0) * 5, 2), 0, 0],
            "population": [0, 0, 0, Fraction(5, 2), 8],
            "service": [0] * n if i < 2 else [0, 1, 0, 1, 0],
            "travel": [[0 if i < 2 or j == k else rng.randrange(1, 6)
                        for j in range(n)] for k in range(n)],
        })
    checked = 0
    for index, instance in enumerate(fixtures):
        exact = truth(instance, 2)
        model = read_model(instance)
        graph = model.project()
        native = Native(graph)
        beta = Fraction(4, 13)
        for part in ((0, 1, 2, 3), (0, 0, 2, 2)):
            for marks in ((0,), (0, 2), (0, 1, 3), (0, 1, 2, 3)):
                prices = tuple(rng.randrange(1 << 19) if i not in marks and index else 0
                               for i in range(model.n))
                reward = tuple(w - p for w, p in zip(model.weights(beta), prices))
                reverse = bool(checked % 2)
                oriented = graph.reversed() if reverse else graph
                values = brute_classes(oriented, reward, part, marks)
                priced = native.price(reward, part, marks, reverse, monotonic() + 4)
                assert priced is not None and priced[0] == values == classes(
                    oriented, reward, part, marks), (index, values, priced)
                for mask, path in enumerate(priced[1]):
                    observed = sum(1 << j for j, marker in enumerate(marks) if marker in path)
                    if path:
                        assert observed == mask and sum(reward[i] for i in path) == values[mask]
                    else:
                        assert values[mask] in (0, None)
                brute_fleet = max(a + b for (i, a), (j, b)
                                  in product(enumerate(values), repeat=2)
                                  if a is not None and b is not None and not i & j)
                assert merge(values, 2)[0][0] == brute_fleet
                if not index and marks == (0,):
                    assert brute_fleet < 2 * values[1]
                upper = product_bound(model, tuple(model.boxes(2)), [
                    (beta, model.reward_unit * (sum(prices) + brute_fleet))
                ])[0]
                certificate = {
                    "method": "mask-fleet-v1", "vehicles": 2, "upper": str(upper),
                    "supports": [{"beta": str(beta), "prices": prices, "reverse": reverse,
                                  "partition": part, "markers": marks, "class_values": values,
                                  "fleet_bound": brute_fleet}],
                }
                assert exact <= replay(instance, 2, certificate) == upper
                certificate["supports"][0]["fleet_bound"] -= 1
                try:
                    replay(instance, 2, certificate)
                except ValueError:
                    pass
                else:
                    raise AssertionError("tampered fleet bound accepted")
                checked += 1
    print(f"{checked} masked price/fleet checks, {len(fixtures)} exact two-vehicle inputs")


if __name__ == "__main__":
    main()
