import time
import unittest
from fractions import Fraction
from unittest.mock import patch

from mvp.bpc import BpcSolver
from mvp.fleet import exact_verify
from mvp.master import PoolMaster
from mvp.routes import RoutePool, enumerate_routes


def instance(supply, population, capacity, limit, *, arc=1):
    size = len(supply)
    return dict(id="micro", family_id="micro", capacity=capacity, limit=limit,
                supply=supply, population=population, service=[0] * size,
                travel=[[0 if i == j else arc for j in range(size)] for i in range(size)])


def complementary_pair():
    return instance([0, 1, 1, 0, 0], [0, 0, 0, 1, 1], 1, 3)


class BpcWarmStartTests(unittest.TestCase):
    def test_illegal_fleet_never_changes_the_route_pool(self):
        data = complementary_pair()
        data["travel"][2][4] = 5
        cases = [
            (1, [[1, 3], [2, 4]], "exceed the fleet size"),
            (2, [[1, 3], [2, 4]], "exceeds limit"),
            (2, [[1, 3], [1, 4]], "visited by vehicles"),
            (2, [[0, 1, 3, 0]], "distinct non-depot"),
            (2, [[True, 3]], "invalid non-depot stop"),
            (2, [[3, 1]], "begin at a supplier"),
            (2, [[1, 3, 3]], "distinct non-depot"),
        ]
        for k, routes, message in cases:
            with self.subTest(routes=routes, vehicles=k):
                pool = RoutePool(data).extend([(2, 3)])
                solver = BpcSolver(data, k, pool=pool)
                with self.assertRaisesRegex(ValueError, message):
                    solver.solve(initial_routes=routes, seconds=0)
                self.assertEqual(pool.routes, [(2, 3)])

    def test_forward_and_return_decimal_equality(self):
        data = instance([0, 1, 0], [0, 0, 1], 1, "0.3", arc="0.1")
        self.assertEqual(list(enumerate_routes(data)), [(1, 2)])
        solver = BpcSolver(data, 1)
        self.assertEqual(solver.partition.best_order([1, 2]), [1, 2])
        self.assertEqual(exact_verify(data, [[1, 2]], 1)["routes"][0]["time"],
                         Fraction(3, 10))
        tighter = dict(data, limit="0.299999999")
        self.assertEqual(list(enumerate_routes(tighter)), [])
        with self.assertRaisesRegex(ValueError, "exceeds limit"):
            BpcSolver(tighter, 1).solve(initial_routes=[[1, 2]], seconds=0)

    def test_pricing_from_supply_agency_pair_preserves_signed_multiplier(self):
        data = complementary_pair()
        solver = BpcSolver(data, 2)
        prices, station_cost, vehicle_cost = {3: 2.0, 4: -1.0}, {1: 0.25}, 0.5
        self.assertEqual(solver._reduced_cost([1, 4], prices, station_cost, vehicle_cost),
                         -1.75)
        results = solver.price(prices, station_cost, vehicle_cost, trials=2)
        self.assertIn((1.25, (1, 3)), results)
        self.assertTrue(all(route[-1] == 3 for _, route in results))
        self.assertEqual(solver.pool.routes, [])

    def test_restricted_lp_positive_equality_dual_is_not_clipped(self):
        data = instance([0, 1, 2, 3, 0, 0, 0], [0, 0, 0, 0, 3, 2, 5], 3, 9)
        pool = RoutePool(data).extend([(2, 6, 5), (3, 6, 2, 5)])
        master = PoolMaster(data, 1, pool)
        lp = master.solve()
        q6_dual = lp["row_dual"][master.q_row[6]]
        self.assertAlmostEqual(q6_dual, 0.08)
        solver = BpcSolver(data, 1, pool=pool)
        with patch.object(solver, "price", return_value=[]) as pricing, \
                patch.object(solver, "local_price", return_value=[]):
            solver.price_round(deadline=time.monotonic() + 1)
        prices = pricing.call_args.args[0]
        self.assertAlmostEqual(prices[6], -q6_dual)

    def test_initial_column_reaches_highs_and_bpc_improves_it(self):
        data = complementary_pair()
        pool = RoutePool(data)
        solver = BpcSolver(data, 2, pool=pool)
        self.assertIs(solver.partition.pool, pool)
        result, _, record = solver.solve(
            initial_routes=[[1, 3]], seconds=1, skip_seed=True, skip_dive=True,
            skip_search=True)
        start = record["master"]["mip_start"]
        self.assertEqual(record["seed"]["value_exact"], "1/2")
        self.assertEqual(start["routes"], [[1, 3]])
        self.assertEqual(start["column_indices"], record["seed"]["column_indices"])
        self.assertEqual(start["vector_length"], len(pool) + 3)
        self.assertEqual(start["set_solution_status"], "HighsStatus.kOk")
        self.assertTrue(start["recognized_before_run"])
        self.assertTrue(start["accepted"])
        self.assertTrue(start["assessment"].endswith(": feasible"))
        self.assertTrue(any(step["stage"] == "pricing" and step["added"] > 0
                            for step in record["history"]))
        self.assertEqual(record["best_source"], "integer_master")
        self.assertEqual(record["best_value_exact"], "2")
        self.assertEqual(exact_verify(data, result, 2)["objective"], 2)

    def test_dive_uses_fixed_column_deliveries_and_remaining_vehicle(self):
        data = instance([0, 10, 3, 4, 0, 0, 0], [0, 0, 0, 0, 5, 3, 2], 10, 3)
        rows = [(1, 4), (2, 5), (3, 6)]
        pool = RoutePool(data).extend(rows)
        master = PoolMaster(data, 2, pool)
        with self.assertRaisesRegex(ValueError, "unrestricted integer master"):
            master.solve(integer=True, fixed=[0], start=[0, 1])
        fixed_solution = master.solve(fixed=[0])
        vector = fixed_solution["values"]
        self.assertEqual(vector[master.x[0]], 1)
        self.assertAlmostEqual(sum(vector[column] for column in master.x), 2)
        self.assertEqual(vector[master.q[master.agency_index[4]]], 10)
        self.assertAlmostEqual(fixed_solution["objective"], 10)
        fixed_integer = master.solve(integer=True, fixed=[0])
        self.assertEqual(fixed_integer["selection"], [0, 2])
        self.assertAlmostEqual(fixed_integer["objective"], 9.8)
        routes, history = BpcSolver(data, 2, pool=pool).dive(
            deadline=time.monotonic() + 2)
        self.assertEqual(len(history), 2)
        self.assertEqual(exact_verify(data, routes, 2)["used_vehicles"], 2)

    def test_short_and_zero_budgets_preserve_a_verified_incumbent(self):
        data = complementary_pair()
        routes, _, record = BpcSolver(data, 2).solve(initial_routes=[[1, 3]], seconds=0)
        self.assertEqual(routes, [[1, 3]])
        self.assertEqual(record["best_value_exact"], "1/2")
        self.assertIsNone(record["master"])

        started = time.monotonic()
        routes, _, record = BpcSolver(data, 2).solve(initial_routes=[[1, 3]], seconds=0.5)
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertGreaterEqual(exact_verify(data, routes, 2)["objective"], Fraction(1, 2))
        self.assertTrue(any(step["stage"] == "dive" for step in record["history"]))
        self.assertTrue(record["master"]["mip_start"]["accepted"])


if __name__ == "__main__":
    unittest.main()
