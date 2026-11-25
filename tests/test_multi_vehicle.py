"""Regression and end-to-end verification tests for multi-vehicle H-PDSP v2.1 and baselines."""

from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MultiVehicleTests(unittest.TestCase):
    def test_pure_stdlib_sample_replay(self):
        """verify_multi.py --sample must run cleanly under `python3 -B -S` with 0 forbidden modules."""
        proc = subprocess.run(
            [sys.executable, "-B", "-S", str(ROOT / "verify_multi.py"), "--sample", "--quiet"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        self.assertIn("0 forbidden modules", proc.stderr)
        self.assertIn("0 L>U", proc.stderr)

    def test_tampered_lower_and_upper_proofs_rejected(self):
        """Tampering with a claimed L score or U bound must be rejected by verify_multi.py."""
        instance = ROOT / "multi_vehicle/data/stage_b_blind/r1-strA-19-k2.json"
        l_cert = ROOT / "multi_vehicle/results/certificates/stage_b/r1-strA-19-k2.L.json"
        u_cert = ROOT / "multi_vehicle/results/certificates/stage_b/r1-strA-19-k2.U.json"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            # Tamper L score upward
            bad_l = json.loads(l_cert.read_text(encoding="utf-8"))
            bad_l["score"] = str(F(bad_l["score"]) + 1)
            bad_l_path = tmp_dir / "bad.L.json"
            bad_l_path.write_text(json.dumps(bad_l), encoding="utf-8")
            proc_l = subprocess.run(
                [sys.executable, "-B", "-S", str(ROOT / "verify_multi.py"),
                 "--single", str(instance), "--solution", str(bad_l_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc_l.returncode, 0)

            # Tamper U score downward
            bad_u = json.loads(u_cert.read_text(encoding="utf-8"))
            bad_u["score"] = str(F(bad_u["score"]) - 1)
            bad_u_path = tmp_dir / "bad.U.json"
            bad_u_path.write_text(json.dumps(bad_u), encoding="utf-8")
            proc_u = subprocess.run(
                [sys.executable, "-B", "-S", str(ROOT / "verify_multi.py"),
                 "--single", str(instance), "--proof", str(bad_u_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc_u.returncode, 0)

    def test_live_v21_closes_strb19_and_replays_strict_proof(self):
        """Live solve_multi.py on strB-19-k2 must close optimal 490699/689 and replay under -B -S."""
        instance = ROOT / "multi_vehicle/data/formal_panel/strB-19-k2.json"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            sol_path = tmp_dir / "sol.json"
            proof_path = tmp_dir / "proof.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "solve_multi.py"),
                    str(instance),
                    "--seconds",
                    "4",
                    "--output",
                    str(sol_path),
                    "--proof-output",
                    str(proof_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}")
            sol = json.loads(sol_path.read_text(encoding="utf-8"))
            self.assertEqual(F(sol["score"]), F(490699, 689))
            self.assertEqual(F(sol["upper_bound_exact"]), F(490699, 689))
            self.assertEqual(sol["status"], "strict_optimal")

            replay = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-S",
                    str(ROOT / "verify_multi.py"),
                    "--single",
                    str(instance),
                    "--solution",
                    str(sol_path),
                    "--proof",
                    str(proof_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(replay.returncode, 0, msg=f"stderr:\n{replay.stderr}")
            out = json.loads(replay.stdout)
            self.assertEqual(out["status"], "strict_optimal")
            self.assertEqual(out["L_exact"], "490699/689")
            self.assertEqual(out["U_exact"], "490699/689")
            self.assertEqual(out["forbidden_modules"], [])

    def test_baselines_smoke_and_exact_replay(self):
        """All three paper baselines (mv_lns, segment, site) run cleanly and pass pure replay."""
        instance = ROOT / "multi_vehicle/data/formal_panel/strB-19-k2.json"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for method in ("mv_lns", "segment", "site"):
                sol_path = tmp_dir / f"{method}.json"
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "solve_multi.py"),
                        str(instance),
                        "--method",
                        method,
                        "--seconds",
                        "1.5",
                        "--output",
                        str(sol_path),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(proc.returncode, 0, msg=f"{method} stderr:\n{proc.stderr}")
                sol = json.loads(sol_path.read_text(encoding="utf-8"))
                self.assertGreater(F(sol["score"]), 0)
                self.assertLessEqual(F(sol["score"]), F(490699, 689))

                replay = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-S",
                        str(ROOT / "verify_multi.py"),
                        "--single",
                        str(instance),
                        "--solution",
                        str(sol_path),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(replay.returncode, 0, msg=f"{method} replay stderr:\n{replay.stderr}")


if __name__ == "__main__":
    unittest.main()
