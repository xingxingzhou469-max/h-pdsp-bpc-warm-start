import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mvp.bpc import BpcSolver
from warm_start import load_start


ROOT = Path(__file__).resolve().parents[1]


class DeliveryTests(unittest.TestCase):
    def test_all_60_starts_reach_the_integer_master(self):
        with (ROOT / "index.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            with self.subTest(instance=row["instance_id"]):
                instance, start = load_start(ROOT / row["input_file"], ROOT / row["solution_file"],
                                             vehicles=2)
                solver = BpcSolver(instance, 2)
                routes, _, record = solver.solve(
                    initial_routes=start["routes"], seconds=30, skip_seed=True,
                    skip_dive=True, skip_pricing=True, skip_search=True)
                self.assertEqual(record["seed"]["column_indices"], [0])
                accepted = record["master"]["mip_start"]
                self.assertTrue(accepted["recognized_before_run"])
                self.assertTrue(accepted["accepted"])
                self.assertEqual(accepted["routes"], start["routes"])
                self.assertEqual(record["best_value_exact"], start["objective_exact"])
                self.assertEqual(routes, start["routes"])

    def test_new_instance_through_both_clis(self):
        # Only 0-1-2-3-4-0 fits the time limit. With capacity 3 and supply (1,3),
        # deliveries (1,3) maximize x+y-|9y-x|/10, giving exactly 7/5.
        travel = [[0 if i == j else 6 for j in range(5)] for i in range(5)]
        for i, j in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0)):
            travel[i][j] = 1
        instance = {"id": "directed-warm-start", "family_id": "test", "capacity": 3,
                    "limit": 5, "supply": [0, 1, 0, 3, 0], "population": [0, 0, 9, 0, 1],
                    "travel": travel, "service": [0]*5}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "instance.json"
            witness_path = Path(directory) / "solution.json"
            path.write_text(json.dumps(instance), encoding="utf-8")
            subprocess.run([sys.executable, str(ROOT / "solve_single.py"), str(path),
                            "--seconds", "5", "--output", str(witness_path)], check=True)
            witness = json.loads(witness_path.read_bytes())
            self.assertEqual(witness["input_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(witness["route"], [1, 2, 3, 4])
            self.assertEqual(witness["amounts"], [1, 1, 3, 3])
            self.assertEqual(witness["objective_exact"], "7/5")
            completed = subprocess.run(
                [sys.executable, str(ROOT / "warm_start.py"), str(path), str(witness_path),
                 "--vehicles", "2", "--run-seconds", "2"],
                check=True, capture_output=True, text=True)
            result = json.loads(completed.stdout)
            self.assertEqual(result["initial"]["objective_exact"], "7/5")
            self.assertEqual(result["final"]["objective_exact"], "7/5")
            self.assertTrue(result["mip_start"]["accepted"])
            # No nonempty route can finish a unit earlier.
            instance["limit"] = 4
            path.write_text(json.dumps(instance), encoding="utf-8")
            completed = subprocess.run([sys.executable, str(ROOT / "solve_single.py"), str(path),
                                        "--seconds", "5"], check=True, capture_output=True)
            report = json.loads(completed.stdout)
            self.assertEqual(report["status"], "no_witness")
            self.assertIsNone(report["route"])
