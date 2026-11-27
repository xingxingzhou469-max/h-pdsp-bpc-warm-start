"""Exact input, visit-count start bound and shared experiment I/O."""
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from itertools import accumulate
from math import gcd, lcm
from pathlib import Path
import json
import os

THREAD_ENV = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "HIGHS_NUM_THREADS",
)
CHECKPOINTS = (1, 5, 15, 30, 60)
NANO = 10**9


def require_single_thread():
    bad = [name for name in THREAD_ENV if os.environ.get(name) != "1"]
    if os.environ.get("PYTHONHASHSEED") != "0":
        bad.append("PYTHONHASHSEED")
    if bad:
        raise ValueError(f"incorrect thread or hash environment: {bad}")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes(), parse_float=Fraction)


def atomic(path, data):
    path = Path(path)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=1, default=str, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def number(value):
    if isinstance(value, Fraction):
        return value
    return Fraction(str(value))


def rational_gcd(values):
    values = tuple(map(number, values))
    denominator = lcm(*(x.denominator for x in values))
    return Fraction(gcd(*(int(x * denominator) for x in values)), denominator)


def ceil_ratio(value, unit):
    return int(-(-value // unit))


@dataclass(frozen=True)
class Data:
    nodes: tuple
    sources: tuple
    caps: tuple
    people: tuple
    cap: Fraction
    total: Fraction
    q_cap: Fraction
    p_step: Fraction
    q_step: Fraction
    limit: Fraction
    increments: tuple
    home: tuple

    @classmethod
    def from_instance(cls, instance):
        supply = tuple(map(number, instance["supply"]))
        population = tuple(map(number, instance["population"]))
        service = tuple(map(number, instance["service"]))
        travel = tuple(tuple(map(number, row)) for row in instance["travel"])
        capacity, limit = map(number, [instance["capacity"], instance["limit"]])
        size = len(supply)
        if not size or not (len(population) == len(service) == len(travel) == size
                            and all(len(row) == size for row in travel)):
            raise ValueError("inconsistent instance dimensions")
        if min(capacity, limit, *supply, *population, *service, *(min(row) for row in travel)) < 0:
            raise ValueError("negative instance quantity, travel, or limit")
        if supply[0] or population[0] or any(supply[i] and population[i] for i in range(1, size)):
            raise ValueError("depot or mixed supplier/agency unsupported")
        nodes = tuple(i for i in range(1, size) if supply[i] or population[i])
        if len(nodes) != size - 1:
            raise ValueError("non-depot nodes must be suppliers or agencies")
        sources = tuple(supply[i] > 0 for i in nodes)
        caps = tuple(min(capacity, supply[i]) for i in nodes)
        people = tuple(population[i] for i in nodes)
        total = sum(people, Fraction(0))
        q_cap = min(sum(caps, Fraction(0)), capacity * sum(p > 0 for p in people))
        increments = tuple(tuple(travel[i][j] + service[j] for j in nodes) for i in (*nodes, 0))
        home = tuple(travel[i][0] for i in nodes)
        return cls(nodes, sources, caps, people, capacity, total, q_cap,
                   rational_gcd(people), rational_gcd((capacity, *caps)), limit, increments, home)


def projection(data, vehicles):
    times = (data.limit, *data.home, *(
        cost for i, row in enumerate(data.increments) for j, cost in enumerate(row) if i != j
    ))
    time_unit = rational_gcd(times) or Fraction(1)
    if data.limit * vehicles // time_unit > 800:
        time_unit = data.limit * vehicles / 800
    load_unit = data.q_step or Fraction(1)
    if data.cap / load_unit > 350:
        load_unit = data.cap / 70
    return time_unit, load_unit


def start_bound(instance, vehicles):
    """Count visits via service and outgoing arcs, with exact top-k prefix sums."""
    data = Data.from_instance(instance)
    supply = {i: cap for i, cap, is_source in zip(data.nodes, data.caps, data.sources) if is_source}
    people = {i: p for i, p in zip(data.nodes, data.people) if p}
    suppliers = sorted(supply, key=supply.__getitem__, reverse=True)
    agencies = sorted(people, key=people.__getitem__, reverse=True)
    service = list(map(number, instance["service"]))
    travel = [list(map(number, row)) for row in instance["travel"]]
    fleet_time = data.limit * vehicles
    costs = sorted(service[i] for i in data.nodes)
    max_visits = sum(t <= fleet_time for t in accumulate(costs))
    all_cost = sum(service[i] + min(t for j, t in enumerate(travel[i]) if j != i) for i in data.nodes)
    max_visits = min(max_visits, len(data.nodes) - int(all_cost > fleet_time))
    best, used = Fraction(0), (0, 0)
    supply_prefix = list(accumulate(supply[i] for i in suppliers))
    for demand_count, population in enumerate(accumulate(people[i] for i in agencies), 1):
        supply_count = min(len(suppliers), max_visits - demand_count)
        if supply_count <= 0:
            break
        qty = min(supply_prefix[supply_count - 1], data.cap * demand_count)
        objective = qty * population / data.total
        if objective > best:
            best, used = objective, (supply_count, demand_count)
    return best, {"method": "exact_visit_count", "max_visits": max_visits,
                  "optimistic_suppliers": suppliers[:used[0]], "optimistic_agencies": agencies[:used[1]]}


def verify_lower(instance, vehicles, witness):
    """Strictly check a fixed fleet and independently compute its pairwise objective."""
    data = Data.from_instance(instance)
    routes, amounts = witness["routes"], witness["amounts"]
    if len(routes) != len(amounts) or len(routes) > vehicles:
        raise ValueError("fleet size mismatch")
    supply = list(map(number, instance["supply"]))
    population = list(map(number, instance["population"]))
    travel = [list(map(number, row)) for row in instance["travel"]]
    service = list(map(number, instance["service"]))
    seen, deliveries = set(), {}
    for route, qtys in zip(routes, amounts):
        if len(qtys) != len(route):
            raise ValueError("amount length mismatch")
        if not route:
            continue
        if any(type(node) != int or node not in data.nodes for node in route):
            raise ValueError("invalid station")
        if not supply[route[0]] or not population[route[-1]]:
            raise ValueError("invalid route endpoints")
        time, load, prev = Fraction(0), Fraction(0), 0
        for node, amount in zip(route, map(number, qtys)):
            if node in seen or amount < 0:
                raise ValueError("repeated station or negative amount")
            seen.add(node)
            time += travel[prev][node] + service[node]
            if supply[node]:
                if amount > supply[node]:
                    raise ValueError("supply exceeded")
                load += amount
            else:
                load -= amount
                deliveries[node] = amount
            if not 0 <= load <= data.cap:
                raise ValueError("load exceeded")
            prev = node
        if time + travel[prev][0] > data.limit:
            raise ValueError("route time exceeded")
    if not data.total:
        return Fraction(0)
    amounts_by_agency = [(p, deliveries.get(i, Fraction(0))) for i, p in enumerate(population) if p]
    dispersion = sum(abs(p * b - q * a)
                     for j, (p, a) in enumerate(amounts_by_agency)
                     for q, b in amounts_by_agency[j + 1:])
    return sum(deliveries.values(), Fraction(0)) - dispersion / data.total
