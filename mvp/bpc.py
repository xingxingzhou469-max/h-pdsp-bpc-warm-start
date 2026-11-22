"""Restricted-master pricing and diving heuristic for multi-vehicle H-PDSP.

A validated starting fleet is also the incumbent. Its routes become columns.
Pricing and diving find additional candidates; the final integer master gets
the best verified fleet as a complete MIP start on the unrestricted pool.
"""

import math
import time
from numbers import Integral

from .fleet import exact_verify, number
from .master import PoolMaster
from .partition import PartitionSearch


class BpcSolver:
    def __init__(self, instance, vehicles, *, seed=0, pool=None):
        if isinstance(vehicles, bool) or not isinstance(vehicles, Integral) or vehicles < 1:
            raise ValueError("vehicles must be a positive integer")
        if pool is not None and pool.instance != instance:
            raise ValueError("the route pool must belong to this instance")
        self.instance = instance
        self.K = int(vehicles)
        self.partition = PartitionSearch(instance, self.K, seed=seed, pool=pool)
        self.pool = self.partition.pool

    def _reduced_cost(self, route, prices, node_costs, vehicle_cost):
        _, amounts = self.pool.column(route)
        revenue = sum(prices.get(node, 0.0) * float(number(amount))
                      for node, amount in zip(route, amounts)
                      if self.partition.population[node] > 0)
        return revenue - sum(node_costs.get(node, 0.0) for node in route) - vehicle_cost

    def price(self, prices, node_costs, vehicle_cost, *, trials=12, deadline=None, top=12):
        """Grow feasible supply-agency pairs using the master's reduced costs."""
        agencies = sorted(self.partition.agencies, key=lambda node: -prices.get(node, 0.0))
        suppliers = sorted(self.partition.suppliers, key=lambda node: -self.partition.supply[node])
        agencies = agencies[:top]
        found = []
        if not agencies or not suppliers:
            return found
        for trial in range(trials):
            if deadline is not None and time.monotonic() >= deadline:
                break
            start = (suppliers[trial] if trial < min(3, len(suppliers))
                     else self.partition.rng.choice(suppliers))
            options = self._sequence_moves(
                (start,), agencies, prices, node_costs, vehicle_cost, deadline)
            if not options:
                continue
            current, order = options[trial % min(2, len(options))]
            while deadline is None or time.monotonic() < deadline:
                extra_suppliers = [node for node in suppliers if node not in order][:3]
                options = self._sequence_moves(
                    order, agencies + extra_suppliers, prices, node_costs,
                    vehicle_cost, deadline)
                improving = [entry for entry in options if entry[0] > current + 1e-9]
                if not improving:
                    break
                current, order = improving[trial % min(2, len(improving))]
            if current > 1e-9:
                found.append((current, tuple(order)))
        return found

    def _sequence_moves(self, route, nodes, prices, node_costs, vehicle_cost, deadline):
        options = []
        for node in nodes:
            if deadline is not None and time.monotonic() >= deadline:
                break
            if node not in route:
                order = self.partition.best_order(
                    [*route, node], restarts=1, deadline=deadline, register=False)
                if order is not None:
                    options.append((self._reduced_cost(order, prices, node_costs, vehicle_cost),
                                    order))
        return sorted(options, key=lambda entry: -entry[0])

    def local_price(self, prices, node_costs, vehicle_cost, *, deadline, top_routes=12):
        """Try add/drop moves around promising existing routes."""
        routes = self.pool.routes
        if len(routes) > 1500:
            routes = routes[-1200:] + routes[::max(1, len(routes) // 300)]
        scored = []
        for route in routes:
            if time.monotonic() >= deadline:
                break
            scored.append((self._reduced_cost(route, prices, node_costs, vehicle_cost), route))
        scored.sort(key=lambda entry: -entry[0])
        found = []
        for _, route in scored[:top_routes]:
            found.extend(self._sequence_moves(route, self.partition.active, prices,
                                             node_costs, vehicle_cost, deadline))
            for node in route[:5]:
                if time.monotonic() >= deadline:
                    return found
                order = self.partition.best_order(
                    [other for other in route if other != node], restarts=1,
                    deadline=deadline, register=False)
                if order is not None:
                    found.append((self._reduced_cost(order, prices, node_costs, vehicle_cost),
                                  order))
        return found

    def price_round(self, *, deadline, price_trials=8):
        master = PoolMaster(self.instance, self.K, self.pool)
        left = deadline - time.monotonic()
        if left <= 0:
            return None
        lp = master.solve(time_limit=min(60.0, left * 0.25))
        if lp["status"] != "HighsModelStatus.kOptimal" or lp["row_dual"] is None:
            return None
        dual = lp["row_dual"]
        # HiGHS minimizes -Z. q_i - sum_r d_ir*x_r == 0 has a free dual.
        # Negating the column's minimization reduced cost gives the score below.
        prices = {node: -float(dual[row]) for node, row in master.q_row.items()}
        node_costs = {node: -float(dual[row]) for node, row in master.node_row.items()}
        vehicle_cost = -float(dual[master.fleet_row])
        found = self.price(prices, node_costs, vehicle_cost, trials=price_trials,
                           deadline=deadline)
        found += self.local_price(prices, node_costs, vehicle_cost, deadline=deadline)
        added = []
        for score, route in sorted(found, key=lambda entry: -entry[0]):
            if score > 1e-9 and self.pool.add(route):
                added.append(score)
        return dict(lp_objective=float(lp["objective"]), added=len(added),
                    columns=len(self.pool), max_added_reduced_cost=max(added, default=None))

    def dive(self, *, deadline, depth=6):
        """Fix promising columns in the full fleet LP, including their deliveries."""
        if not self.pool.routes:
            return [], []
        master = PoolMaster(self.instance, self.K, self.pool)
        fixed, history = [], []
        for level in range(min(depth, self.K)):
            left = deadline - time.monotonic()
            if left <= 0:
                break
            lp = master.solve(time_limit=min(30.0, left), fixed=fixed)
            if lp["values"] is None:
                break
            remaining = [(float(lp["values"][column]), index)
                         for index, column in enumerate(master.x) if index not in fixed]
            score, index = max(remaining, default=(0.0, -1))
            if score < 0.15:
                break
            fixed.append(index)
            history.append(dict(depth=level, x=score, route=list(self.pool.routes[index])))
        return [self.pool.routes[index] for index in fixed], history

    def solve(self, *, seconds=600.0, dive_depth=6, master_seconds=120.0,
              price_trials=12, seed=0, record=None, seed_fraction=0.40,
              initial_routes=None, skip_seed=False, skip_pricing=False,
              skip_dive=False, skip_search=False):
        """Return (routes, float score, record) with an exact witness score in record."""
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("seconds must be finite and nonnegative")
        if not math.isfinite(master_seconds) or master_seconds < 0:
            raise ValueError("master_seconds must be finite and nonnegative")
        if not 0 <= seed_fraction <= 1:
            raise ValueError("seed_fraction must be in [0, 1]")
        started = time.monotonic()
        deadline = started + seconds
        master_window = min(master_seconds, seconds * 0.2)
        work_deadline = deadline - master_window
        price_deadline = min(started + max(seed_fraction, 0.75) * seconds, work_deadline)
        record = record if record is not None else {}
        history = []
        self.partition.rng.seed(seed + 17)

        supplied = [] if initial_routes is None else [tuple(route) for route in initial_routes]
        # Validate the complete fleet before inserting any of its columns.
        seed_report = exact_verify(self.instance, supplied, self.K)
        best_routes = [route for route in supplied if route]
        best_value = seed_report["objective"]
        best_source = "initial_routes" if supplied else "empty_fleet"
        for route in best_routes:
            self.pool.add(route)
            # Use the supplied sequence as the first choice for this stop set.
            self.partition.remember_order(route)
        seed_indices = [self.pool.index[route] for route in best_routes]
        record["seed"] = dict(routes=[list(route) for route in best_routes],
                              column_indices=seed_indices, verified=True,
                              value=float(best_value), value_exact=str(best_value))
        if initial_routes is not None:
            history.append(dict(stage="seed_given", value=float(best_value),
                                seconds=time.monotonic() - started))

        def consider(routes, source):
            nonlocal best_routes, best_value, best_source
            report = exact_verify(self.instance, routes, self.K)
            value = report["objective"]
            if value > best_value:
                best_routes = [tuple(route) for route in routes if route]
                best_value, best_source = value, source
            return float(value)

        def search_until(cutoff, source, random_seed):
            remaining = cutoff - time.monotonic()
            if remaining > 0:
                routes, _ = self.partition.search(seconds=remaining, seed=random_seed,
                                                  initial=best_routes)
                history.append(dict(stage=source, value=consider(routes, source),
                                    seconds=time.monotonic() - started))

        if not skip_seed:
            seed_deadline = min(started + seed_fraction * seconds, work_deadline)
            search_until(seed_deadline, "seed", seed)

        iteration = 0
        while not skip_pricing and time.monotonic() < price_deadline:
            iteration += 1
            left = price_deadline - time.monotonic()
            priced = self.price_round(deadline=time.monotonic() + left * 0.5,
                                      price_trials=price_trials)
            if priced is None:
                break
            history.append(dict(stage="pricing", round=iteration, **priced,
                                seconds=time.monotonic() - started))
            if not priced["added"]:
                break

        if not skip_dive and time.monotonic() < work_deadline:
            fixed_routes, dive_history = self.dive(deadline=work_deadline, depth=dive_depth)
            for entry in dive_history:
                history.append(dict(stage="dive", seconds=time.monotonic() - started,
                                    **entry))
            if fixed_routes:
                consider(fixed_routes, "dive")

        if not skip_search:
            search_until(work_deadline, "topup_search", seed + 7000)

        master_record = None
        if self.pool.routes and master_seconds > 0 and time.monotonic() < deadline:
            master = PoolMaster(self.instance, self.K, self.pool)
            left = min(master_seconds, deadline - time.monotonic())
            if left > 0:
                start = [self.pool.index[route] for route in best_routes]
                mip = master.solve(integer=True, time_limit=left, start=start or None)
                master_record = dict(status=mip["status"], objective=mip["objective"],
                                     elapsed=mip["elapsed"], pool_gap=mip["mip_gap"],
                                     mip_start=mip["mip_start"])
                consider([self.pool.routes[index] for index in mip["selection"]],
                         "integer_master")

        routes = [list(route) for route in best_routes]
        record.update(best_value=float(best_value), best_value_exact=str(best_value),
                      best_routes=routes, best_source=best_source, pool_routes=len(self.pool),
                      iterations=iteration, history=history, master=master_record,
                      seconds=time.monotonic() - started)
        return routes, float(best_value), record
