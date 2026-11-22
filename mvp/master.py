"""Restricted master for a pool of feasible multi-vehicle H-PDSP routes.

Columns specify a station order and its canonical deliveries. The LP supplies
pricing duals. The integer master selects disjoint routes and evaluates the
fairness objective on deliveries across the fleet. Solver bounds refer to
the current pool.
"""

import time

import numpy as np
from scipy.sparse import csr_matrix

import highspy

from .fleet import number, verify_fleet


class PoolMaster:
    def __init__(self, instance, vehicles, pool):
        self.instance = instance
        self.K = vehicles
        self.pool = pool
        self.columns = [pool.column(route) for route in pool.routes]
        self._build()

    def _build(self):
        instance = self.instance
        population, supply = instance["population"], instance["supply"]
        agencies = [i for i in range(1, len(supply)) if number(population[i]) > 0]
        agency_index = {node: index for index, node in enumerate(agencies)}
        total_people = sum(number(population[i]) for i in agencies)
        weights = {i: float(number(population[i]) / total_people) for i in agencies}
        self.weights = weights
        routes = self.pool.routes

        lower, upper, cost = [], [], []

        def var(lb, ub, coefficient=0.0):
            position = len(lower)
            lower.append(float(lb))
            upper.append(float(ub))
            cost.append(float(coefficient))
            return position

        self.x = [var(0.0, 1.0) for _ in routes]
        self.q = [var(0.0, float(instance["capacity"])) for _ in agencies]
        self.gaps = {}
        for index, i in enumerate(agencies):
            for j in agencies[index + 1:]:
                self.gaps[(i, j)] = var(0.0, np.inf)
        rows, row_lower, row_upper = [], [], []

        def row(terms, lo=-np.inf, hi=np.inf):
            rows.append(terms)
            row_lower.append(float(lo))
            row_upper.append(float(hi))

        # q_i = sum_r d_ir x_r.  Build a node -> (column, quantity) index once;
        # scanning every route for every agency is O(|D| |R|) and dominates
        # 100-node master builds.
        node_columns = {}
        for index, route in enumerate(routes):
            _, amounts = self.columns[index]
            for node, quantity in zip(route, amounts):
                value = float(number(quantity))
                if value:
                    node_columns.setdefault(node, []).append((index, value))
        self.q_row = {}
        for node in agencies:
            terms = [(self.q[agency_index[node]], 1.0)]
            terms += [(self.x[index], -value)
                      for index, value in node_columns.get(node, [])]
            self.q_row[node] = len(rows)
            row(terms, 0.0, 0.0)
        # Fairness epigraph: gap_ij >= |w_i q_j - w_j q_i|.
        for (i, j), gap in self.gaps.items():
            qi, qj = self.q[agency_index[i]], self.q[agency_index[j]]
            row([(qi, weights[j]), (qj, -weights[i]), (gap, -1.0)], hi=0.0)
            row([(qi, -weights[j]), (qj, weights[i]), (gap, -1.0)], hi=0.0)
        # Fleet size and single visits.
        self.fleet_row = len(rows)
        row([(self.x[index], 1.0) for index in range(len(routes))], hi=float(self.K))
        self.node_row = {}
        for node in range(1, len(supply)):
            if number(supply[node]) > 0 or number(population[node]) > 0:
                terms = [(self.x[index], 1.0) for index, route in enumerate(routes)
                         if node in route]
                if terms:
                    self.node_row[node] = len(rows)
                    row(terms, hi=1.0)
        # HiGHS minimizes -sum(q) + sum(gap).
        for node in agencies:
            cost[self.q[agency_index[node]]] = -1.0
        for gap in self.gaps.values():
            cost[gap] = 1.0

        row_index, column_index, data = [], [], []
        for index, terms in enumerate(rows):
            for column, coefficient in terms:
                if coefficient:
                    row_index.append(index)
                    column_index.append(column)
                    data.append(coefficient)
        matrix = csr_matrix((np.asarray(data, dtype=float),
                             (np.asarray(row_index, dtype=np.int32),
                              np.asarray(column_index, dtype=np.int32))),
                            shape=(len(rows), len(lower)))
        matrix.sum_duplicates()
        self.matrix = matrix.tocsr()
        self.row_lower = np.asarray(row_lower)
        self.row_upper = np.asarray(row_upper)
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.cost = np.asarray(cost)
        self.agencies = agencies
        self.agency_index = agency_index

    def _warm_start(self, highs, indices):
        """Hand HiGHS a complete feasible solution built from route indices.

        Without a start the integer master is free to return an incumbent worse
        than the fleet the caller already holds, because the pool is large and
        the time limit is short.  The x values are fixed by ``indices``; the
        deliveries are the merged canonical columns and the gap variables take
        their exact absolute differences.
        """
        if len(set(indices)) != len(indices) or any(
                index < 0 or index >= len(self.x) for index in indices):
            raise ValueError("MIP start indices must select distinct pool routes")
        routes = [self.pool.routes[index] for index in indices]
        amounts = [self.columns[index][1] for index in indices]
        witness = verify_fleet(self.instance, routes, amounts, vehicles=self.K, tol=0)
        values = np.zeros(len(self.lower))
        for index in indices:
            values[self.x[index]] = 1.0
        quantity = {node: float(value) for node, value in witness["deliveries"].items()}
        for node, position in self.agency_index.items():
            values[self.q[position]] = quantity.get(node, 0.0)
        for (i, j), gap in self.gaps.items():
            values[gap] = abs(self.weights[i] * quantity.get(j, 0.0)
                              - self.weights[j] * quantity.get(i, 0.0))
        count = len(self.lower)
        status = highs.setSolution(count, np.arange(count, dtype=np.int32), values)
        stored = highs.getSolution()
        recognized = stored.value_valid and np.array_equal(stored.col_value, values)
        return dict(column_indices=list(indices), routes=[list(route) for route in routes],
                    vector_length=count, set_solution_status=str(status),
                    recognized_before_run=bool(recognized), accepted=None,
                    assessment=None, value_exact=str(witness["objective"]))

    def solve(self, *, integer=False, time_limit=60.0, verbose=False, fixed=(), start=None):
        if start is not None and (not integer or fixed):
            raise ValueError("a complete MIP start requires the unrestricted integer master")
        highs = highspy.Highs()
        highs.setOptionValue("output_flag", bool(verbose or start is not None))
        highs.setOptionValue("log_to_console", bool(verbose))
        highs.setOptionValue("time_limit", float(time_limit))
        highs.setOptionValue("threads", 1)
        highs.setOptionValue("random_seed", 0)
        highs.setOptionValue("mip_rel_gap", 0.0)
        highs.setOptionValue("mip_abs_gap", 0.0)
        highs.setOptionValue("presolve", "on")
        lower = self.lower
        if fixed:
            lower = lower.copy()
            for index in fixed:
                lower[self.x[index]] = 1.0
        highs.addVars(len(lower), lower, self.upper)
        highs.changeColsCost(len(self.cost), np.arange(len(self.cost), dtype=np.int32),
                             self.cost)
        if integer:
            highs.changeColsIntegrality(len(self.x), np.arange(len(self.x), dtype=np.int32),
                                        np.full(len(self.x), highspy.HighsVarType.kInteger))
        matrix = self.matrix
        highs.addRows(matrix.shape[0], self.row_lower, self.row_upper, matrix.nnz,
                      matrix.indptr.astype(np.int32), matrix.indices.astype(np.int32),
                      matrix.data)
        mip_start = None
        if start is not None:
            mip_start = self._warm_start(highs, start)

            def capture_start_assessment(event):
                message = event.message.strip()
                if message.startswith("Assessing feasibility of MIP solution"):
                    mip_start["assessment"] = message
                    mip_start["accepted"] = message.endswith(": feasible")

            highs.cbLogging.subscribe(capture_start_assessment)
        started = time.monotonic()
        highs.run()
        elapsed = time.monotonic() - started
        solution = highs.getSolution()
        values = np.asarray(solution.col_value) if solution.value_valid else None
        row_dual = np.asarray(solution.row_dual) if solution.dual_valid else None
        selection = []
        if values is not None:
            selection = [index for index, column in enumerate(self.x) if values[column] > 0.5]
        return dict(status=str(highs.getModelStatus()), elapsed=elapsed,
                    objective=-float(highs.getInfo().objective_function_value),
                    mip_gap=float(highs.getInfo().mip_gap) if integer else 0.0,
                    values=values, row_dual=row_dual, selection=selection,
                    mip_start=mip_start)
