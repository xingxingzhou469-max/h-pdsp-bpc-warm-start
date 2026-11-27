"""Find a short simple one-supplier route through directed transit stops."""

from fractions import Fraction as F
from heapq import heappop, heappush
from time import monotonic_ns


SEED_NS = 250_000_000


class Expired(Exception):
    pass


def seed(raw, stop_mono_ns):
    """Offer an exact feasible lower witness or return None within the seed budget."""
    stop = min(stop_mono_ns, monotonic_ns() + SEED_NS)

    def check_time():
        if monotonic_ns() >= stop:
            raise Expired

    def exact(value):
        return value if type(value) in (int, F) else F(str(value))

    nodes = range(len(raw["supply"]))
    travel = raw["travel"]

    def arc(i, j):
        return exact(travel[i][j]) + (service[j] if j else 0)

    def shortest(root, allowance, blocked, reverse=False):
        """Label paths on the original arcs or their reversal, with service included."""
        distance = {root: 0}
        parent = {}
        settled = set(blocked)
        pending = [(0, root)]
        while pending:
            check_time()
            elapsed, i = heappop(pending)
            if i in settled:
                continue
            settled.add(i)
            for j in nodes:
                check_time()
                if j in settled:
                    continue
                length = elapsed + (arc(j, i) if reverse else arc(i, j))
                prior = distance.get(j)
                if length <= allowance and (prior is None or length < prior):
                    distance[j] = length
                    parent[j] = i
                    heappush(pending, (length, j))
        return distance, parent

    def path_to_root(i, parent):
        path = [i]
        while i in parent:
            check_time()
            i = parent[i]
            path.append(i)
        return path

    def offer(prefix, middle, tail, quantity):
        nonlocal best_numerator, witness
        path = prefix + middle + tail
        if len(set(path)) != len(path):
            return
        served = sum(people[i] for i in path[len(prefix):])
        numerator = quantity * served
        if numerator <= best_numerator:
            return
        # The sole pickup finances every later delivery at the same rate.
        # Earlier agencies and the other suppliers have zero quantity.
        rate = F(quantity) / served
        amounts = ["0"] * (len(prefix) - 1) + [str(quantity)]
        for i in path[len(prefix):]:
            check_time()
            amounts.append(str(rate * people[i]))
        candidate = {"lower": str(F(numerator) / total),
                     "witness": {"routes": [path], "amounts": [amounts]},
                     "source": "directed-one-donor"}
        check_time()
        best_numerator, witness = numerator, candidate

    def try_prefix(prefix, elapsed):
        donor = prefix[-1]
        remaining = limit - elapsed
        backward, reverse_parent = shortest(0, remaining, prefix, reverse=True)
        # Try the single arc first, then a path that itself can use transit stops.
        for i, back_time in backward.items():
            check_time()
            if people[i] and arc(donor, i) + back_time <= remaining:
                offer(prefix, [], path_to_root(i, reverse_parent)[:-1], stock[donor])
        forward, donor_parent = shortest(donor, remaining, [0, *prefix[:-1]])
        for i, back_time in backward.items():
            check_time()
            if people[i] and i in forward and forward[i] + back_time <= remaining:
                middle = path_to_root(i, donor_parent)[1:-1][::-1]
                offer(prefix, middle, path_to_root(i, reverse_parent)[:-1], stock[donor])

    best_numerator, witness = 0, None
    try:
        check_time()
        cap, limit = exact(raw["capacity"]), exact(raw["limit"])
        if not cap:
            return None
        service, people, stock = [], [], []
        for i in nodes:
            check_time()
            service.append(exact(raw["service"][i]))
            people.append(exact(raw["population"][i]))
            stock.append(min(cap, exact(raw["supply"][i])))
        total = sum(people)
        if not total or not any(stock):
            return None

        outward, parent = shortest(0, limit, ())
        donors = sorted((i for i in outward if stock[i]), key=lambda i: (-stock[i], outward[i], i))
        for donor in donors:
            check_time()
            if stock[donor] * total <= best_numerator:
                break
            prefix = path_to_root(donor, parent)[:-1][::-1]
            try_prefix(prefix, outward[donor])
            direct = arc(0, donor)
            if len(prefix) > 1 and direct <= limit:
                try_prefix([donor], direct)

        if witness is None:
            # A shortest prefix can consume an agency needed after pickup.
            # Reserve each such agency in turn by excluding it from the prefix.
            for donor in donors:
                for agency in path_to_root(donor, parent)[1:-1]:
                    check_time()
                    if not people[agency]:
                        continue
                    alternative, alternative_parent = shortest(0, limit, (agency,))
                    if donor in alternative:
                        prefix = path_to_root(donor, alternative_parent)[:-1][::-1]
                        try_prefix(prefix, alternative[donor])
                        if witness is not None:
                            return witness
    except Expired:
        pass
    return witness
