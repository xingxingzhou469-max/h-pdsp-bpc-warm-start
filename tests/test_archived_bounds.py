import csv
from pathlib import Path
import tempfile
import unittest

from verify_bounds import ROOT, verify


class ArchiveProofTests(unittest.TestCase):
    def test_indexed_upper_and_status_must_follow_replay(self):
        with (ROOT / "index.csv").open(encoding="utf-8") as stream:
            row = next(r for r in csv.DictReader(stream) if r["instance_id"] == "blind-08")
        with tempfile.TemporaryDirectory() as cache:
            self.assertEqual(verify(row, Path(cache))["upper_exact"], "1000")
            for change in ({"upper_bound_exact": "999"}, {"single_vehicle_status": "feasible"}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    verify({**row, **change}, Path(cache))
