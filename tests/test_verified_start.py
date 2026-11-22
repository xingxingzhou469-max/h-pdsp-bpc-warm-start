import csv
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from fractions import Fraction
from pathlib import Path

from hpdp.allocation import allocate
from mvp.fleet import verify_fleet
from warm_start import load_start


ROOT = Path(__file__).resolve().parents[1]


class VerifiedStartTests(unittest.TestCase):
    def test_all_historical_witnesses(self):
        with (ROOT / "index.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len({row["instance_id"] for row in rows}), 60)
        self.assertEqual(Counter(row["split"] for row in rows), {"public": 48, "blind": 12})
        for row in rows:
            with self.subTest(instance=row["instance_id"]):
                raw = ROOT / row["input_file"]
                solution = ROOT / row["solution_file"]
                self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(), row["input_sha256"])
                self.assertEqual(hashlib.sha256(solution.read_bytes()).hexdigest(), row["solution_sha256"])
                instance, start = load_start(raw, solution, vehicles=2)
                expected = Fraction(row["objective_exact"])
                self.assertEqual(Fraction(start["objective_exact"]), expected)
                score, amounts = allocate(instance, start["routes"][0])
                self.assertEqual(score, expected)
                self.assertEqual(amounts, [Fraction(v) for v in start["amounts"][0]])
                self.assertEqual(start["used_vehicles"], 1)

    def test_mismatched_raw_input_is_rejected_even_if_id_matches(self):
        instance = ROOT / "instances/random-20-1.json"
        changed = json.loads(instance.read_bytes())
        changed["limit"] += 1
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "same-id.json"
            target.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_start(target, ROOT / "solutions/random-20-1.json", vehicles=2)

    def test_invalid_single_vehicle_witnesses(self):
        path = ROOT / "solutions/random-20-1.json"
        original = json.loads(path.read_bytes())
        invalid = [
            {"objective_exact": "0"},
            {"amounts": [1000000, *original["amounts"][1:]]},
            {"route": [*original["route"][:-1], original["route"][0]]},
            {"status": "no_witness", "route": None, "amounts": None, "objective_exact": None},
        ]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "witness.json"
            for patch in invalid:
                with self.subTest(patch=patch):
                    target.write_text(json.dumps({**original, **patch}))
                    with self.assertRaises(ValueError):
                        load_start(ROOT / "instances/random-20-1.json", target, vehicles=2)
        with self.assertRaisesRegex(ValueError, "positive"):
            load_start(ROOT / "instances/random-20-1.json", path, vehicles=0)

    def test_fleet_scores_use_all_deliveries_together(self):
        instance = {"capacity": 1, "limit": 3, "travel": [[int(i != j) for j in range(5)]
                                                            for i in range(5)],
                    "service": [0]*5, "supply": [0, 1, 1, 0, 0], "population": [0, 0, 0, 1, 1]}
        first, second = [1, 3], [2, 4]
        for route in (first, second):
            self.assertEqual(verify_fleet(instance, [route], [[1, 1]])["objective"], Fraction(1, 2))
        both = verify_fleet(instance, [first, second], [[1, 1], [1, 1]], vehicles=2)
        self.assertEqual(both["objective"], 2)
