"""Exact preprocessing and arithmetic for the sparse supplier cover."""
from fractions import Fraction as F
from itertools import combinations, product
from math import gcd, lcm

METHOD = "sparse-supplier-fair-v2"
SCALE = 100000
NEG = -(1 << 61)


def num(x):
    return x if isinstance(x, F) else F(str(x))


class Model:
    def __init__(self, instance, k):
        self.n = n = len(instance["supply"])
        self.q = tuple(map(num, instance["supply"]))
        self.p = tuple(map(num, instance["population"]))
        self.service = tuple(map(num, instance["service"]))
        self.travel = tuple(tuple(map(num, row)) for row in instance["travel"])
        self.limit = num(instance["limit"])
        self.capacity = num(instance["capacity"])
        if (not 1 <= n <= 200 or type(k) is not int or k < 0 or
                len(self.p) != n or len(self.service) != n or len(self.travel) != n or
                any(len(row) != n for row in self.travel)):
            raise ValueError("invalid sparse model dimensions")
        if (any(x < 0 for x in (*self.q, *self.p, *self.service, self.limit, self.capacity)) or
                any(min(row) < 0 for row in self.travel) or self.q[0] or self.p[0] or
                any((q > 0) == (p > 0) for q, p in zip(self.q[1:], self.p[1:]))):
            raise ValueError("invalid supplier or agency data")
        self.suppliers = tuple(i for i in range(1, n) if self.q[i])
        self.agencies = tuple(i for i in range(1, n) if self.p[i])
        if len(self.suppliers) > 4:
            raise ValueError("sparse cover requires at most four suppliers")
        self.bits = tuple(1 << self.suppliers.index(i) if i in self.suppliers else 0 for i in range(n))
        self.total = sum(self.p, F())
        self.k = min(k, len(self.suppliers))
        self.stock = {mask: sum((min(self.q[i], self.capacity) for i in self.suppliers
                                 if self.bits[i] & mask), F())
                      for mask in range(1, 1 << len(self.suppliers))}
        # Rational increments are projected downward to retain all original tours.
        costs = [[x + (self.service[j] if j else 0) for j, x in enumerate(row)]
                 for row in self.travel]
        vals = [self.limit, *(costs[i][j] for i in range(n) for j in range(n) if i != j)]
        denominator = lcm(*(x.denominator for x in vals))
        step = F(gcd(*(int(x * denominator) for x in vals)), denominator) or F(1)
        unit = max(step, self.limit / 540)
        self.horizon = int(self.limit // unit)
        self.costs = tuple(tuple(min(self.horizon + 1, int(x // unit)) for x in row) for row in costs)
        self.positive = all(self.costs[i][j] for i in range(1, n)
                            for j in range(1, n) if i != j)

    def covers(self):
        def cover(mask, slots):
            if not mask:
                yield ()
            elif slots:
                pivot = mask & -mask
                rest = mask ^ pivot
                sub = rest
                while True:
                    for suffix in cover(rest ^ sub, slots - 1):
                        yield (sub | pivot, *suffix)
                    if not sub:
                        return
                    sub = (sub - 1) & rest
        if not self.total or not self.capacity:
            return [()]
        return [leaf for mask in range(1 << len(self.suppliers)) for leaf in cover(mask, self.k)]

    def cheap_caps(self):
        result = {}
        cost = self.costs
        for mask in self.stock:
            required = (0, *(i for i in self.suppliers if self.bits[i] & mask))
            allowed = (*required, *self.agencies)
            # Charge minimum incoming and minimum outgoing separately to
            # every visited vertex, so the entire route budget is doubled.
            charge = {i: min(cost[i][j] for j in allowed if j != i) +
                         min(cost[j][i] for j in allowed if j != i) for i in allowed}
            left = 2 * self.horizon - sum(charge[i] for i in required)
            if left < 0:
                result[mask] = F()
                continue
            dp = [F()] * (left + 1)
            for i in self.agencies:
                c, p = charge[i], self.p[i]
                for b in range(left, c - 1, -1):
                    dp[b] = max(dp[b], dp[b - c] + p)
            result[mask] = dp[left]
        return result

    def supports(self, groups, alpha):
        q = [self.stock[m] for m in groups]
        coeffs = q[:]
        for (i, j), x in zip(combinations(range(len(groups)), 2), alpha, strict=True):
            coeffs[i] += 2 * x * q[j]
            coeffs[j] += 2 * (1 - x) * q[i]
        return coeffs

    def box(self, groups, caps):
        # Population pooling and pairwise fairness linearizations.
        d = len(groups) * (len(groups) - 1) // 2
        best = sum((self.stock[g] for g in groups), F())
        for alpha in product((F(0), F(1, 2), F(1)), repeat=d):
            coeffs = self.supports(groups, alpha)
            left, value = self.total, F()
            for a, group in sorted(zip(coeffs, groups), reverse=True):
                covered = min(left, caps[group])
                value += a * covered
                left -= covered
            if self.total:
                best = min(best, value / self.total)
        return best

    def rewards(self, coefficient, prices):
        if not self.total:
            return (0,) * self.n
        return tuple(-(-(coefficient * p * SCALE / self.total) // 1) - price
                     for p, price in zip(self.p, prices))


def clusters(model, width=4):
    region = list(range(model.n))
    cities = {i: [i] for i in model.agencies}
    pairs = sorted((model.costs[i][j] + model.costs[j][i], i, j)
                   for i, j in combinations(model.agencies, 2))
    for _, i, j in pairs:
        a, b = region[i], region[j]
        if a == b or len(cities[a]) + len(cities[b]) > width:
            continue
        cities[a].extend(cities.pop(b))
        for node in cities[a]:
            region[node] = a
    return region
