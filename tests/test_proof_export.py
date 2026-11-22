import copy
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from hpdp.cert_verifier import verify_certificate
from solve_single import solve_instance, verify_proof


ROOT = Path(__file__).resolve().parents[1]


def instance_bytes(limit=3):
    # The only possible nonempty route is 0-1-2-0. Supplier 3 is unreachable.
    return json.dumps(dict(id="directed-proof", capacity=3, limit=limit,
                           supply=[0, 1, 0, 2], population=[0, 0, 1, 0], service=[0]*4,
                           travel=[[0, 1, 4, 4], [4, 0, 1, 4], [1, 4, 0, 4], [4, 4, 4, 0]])).encode()


def routing_dual(payload, bound, supplier_penalty=0, depot_penalty=0):
    # An isolated visit is fixed to zero by its degree equation. With no arcs
    # leaving the depot, its degree equation also certifies infeasibility.
    leaf = dict(cuts=[], branch_rows=[], denominator=1, cap_product_bound=bound,
                integer_multipliers=dict(nonnegative=[0],
                                         free=[depot_penalty, 0, 0, 0, 0, 0, supplier_penalty, 0]))
    return dict(version=1, instance_sha256=sha256(payload).hexdigest(), slope=[1, 1],
                pickup_blocks=False, leaves=[leaf], cap_product_bound=bound)


class ProofExportTests(unittest.TestCase):
    def run_cli(self, *args, standard_library_only=False):
        interpreter = [sys.executable, "-S"] if standard_library_only else [sys.executable]
        return subprocess.run([*interpreter, str(ROOT / "solve_single.py"), *map(str, args)],
                              capture_output=True, text=True)

    def test_search_report_and_independent_replay_use_distinct_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            instance, proof, witness = (folder / name for name in ("in.json", "proof.json", "solution.json"))
            instance.write_bytes(instance_bytes())
            complete = self.run_cli(instance, "--seconds", 3, "--output", witness, "--proof-output", proof)
            self.assertEqual(complete.returncode, 0, complete.stderr)
            self.assertEqual(complete.stdout, "")
            document = json.loads(proof.read_bytes())
            self.assertEqual(document["witness"], json.loads(witness.read_bytes()))
            self.assertEqual(document["witness"]["input_sha256"], sha256(instance.read_bytes()).hexdigest())
            self.assertEqual(document["witness"]["objective_exact"], "1")
            self.assertEqual(document["trusted_search"]["status"], "optimal")
            self.assertEqual(document["trusted_search"]["upper_bound_exact"], "1")
            self.assertEqual(document["replayable"]["status"], "bounded")
            self.assertEqual(document["replayable"]["upper_bound_exact"], "3")
            self.assertIsNone(document["replayable"]["dual_certificate"])
            default = self.run_cli(instance, "--seconds", 3)
            self.assertEqual(default.returncode, 0, default.stderr)
            self.assertEqual(json.loads(default.stdout), document["witness"])

            replay = self.run_cli(instance, "--verify-proof", proof, standard_library_only=True)
            self.assertEqual(replay.returncode, 0, replay.stderr)
            checked = json.loads(replay.stdout)
            self.assertEqual((checked["status"], checked["lower_bound_exact"], checked["upper_bound_exact"]),
                             ("bounded", "1", "3"))
            self.assertNotIn("trusted_search", checked)
            instance.write_bytes(instance.read_bytes() + b"\n")
            replay = self.run_cli(instance, "--verify-proof", proof, standard_library_only=True)
            self.assertNotEqual(replay.returncode, 0)
            self.assertIn("SHA-256", replay.stderr)

    def test_best_dual_is_exported_and_replayed_even_if_search_bound_is_tighter(self):
        payload = instance_bytes()
        dual = routing_dual(payload, 2, supplier_penalty=-1)
        worse = routing_dual(payload, 3)
        self.assertEqual(verify_certificate(payload, dual), Fraction(2))
        self.assertEqual(verify_certificate(payload, worse), Fraction(3))

        def candidate(resources, deadline, seed, *, emit, emit_dual):
            emit_dual(Fraction(2), dual)
            emit_dual(Fraction(3), worse)
            return dict(seed, status="optimal", lb_exact="1", ub_exact="1", certificate="kernel_test")

        with patch("hpdp.overnight_candidate.solve_candidate", side_effect=candidate):
            _, proof = solve_instance(payload, 3, include_proof=True)
        self.assertEqual(proof["replayable"]["dual_certificate"], dual)
        self.assertEqual(proof["trusted_search"]["upper_bound_exact"], "1")
        self.assertEqual(verify_proof(payload, proof)["upper_bound_exact"], "2")

        tighter = routing_dual(payload, 1, supplier_penalty=-2)
        proof["replayable"].update(status="optimal", upper_bound_exact="1", dual_certificate=tighter)
        self.assertEqual(verify_proof(payload, proof)["status"], "optimal")

        claimed_wrong = copy.deepcopy(proof)
        claimed_wrong["replayable"]["upper_bound_exact"] = "0"
        with self.assertRaisesRegex(ValueError, "differs from exact replay"):
            verify_proof(payload, claimed_wrong)
        claimed_wrong = copy.deepcopy(proof)
        claimed_wrong["replayable"]["dual_certificate"]["leaves"][0]["branch_rows"] = [[3, 0]]
        with self.assertRaisesRegex(ValueError, "incomplete branch partition"):
            verify_proof(payload, claimed_wrong)
        claimed_wrong = copy.deepcopy(proof)
        claimed_wrong["witness"]["objective_exact"] = "0"
        with self.assertRaisesRegex(ValueError, "differs from exact scoring"):
            verify_proof(payload, claimed_wrong)

    def test_no_witness_distinguishes_open_search_and_infeasibility(self):
        witness, proof = solve_instance(instance_bytes(), 1e-12, include_proof=True)
        self.assertEqual(witness["status"], "no_witness")
        self.assertEqual(proof["trusted_search"]["status"], "bounded")
        self.assertIsNone(verify_proof(instance_bytes(), proof)["lower_bound_exact"])

        payload = instance_bytes(limit=0)
        witness, proof = solve_instance(payload, 3, include_proof=True)
        self.assertEqual(witness["status"], "no_witness")
        self.assertEqual(proof["trusted_search"]["status"], "infeasible")
        self.assertIsNone(proof["trusted_search"]["upper_bound_exact"])
        self.assertEqual(verify_proof(payload, proof)["status"], "bounded")
        impossible = routing_dual(payload, -1, depot_penalty=-10)
        proof["replayable"].update(status="infeasible", upper_bound_exact=None,
                                   method="routing_dual", dual_certificate=impossible)
        self.assertEqual(verify_proof(payload, proof)["status"], "infeasible")

    def test_cli_rejects_output_aliases_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            instance, alias, output = (folder / name for name in ("in.json", "alias.json", "out.json"))
            original = instance_bytes()
            instance.write_bytes(original)
            alias.hardlink_to(instance)
            for args in (("--proof-output", instance), ("--proof-output", alias),
                         ("--output", output, "--proof-output", output),
                         ("--verify-proof", output, "--output", output)):
                with self.subTest(args=args):
                    completed = self.run_cli(instance, *args)
                    self.assertEqual(completed.returncode, 2, completed.stderr)
                    self.assertIn("must use different paths", completed.stderr)
                    self.assertEqual(instance.read_bytes(), original)
                    self.assertEqual(alias.read_bytes(), original)
                    self.assertFalse(output.exists())

    def test_source_fingerprints_match_the_delivery(self):
        source = json.loads((ROOT / "single_vehicle/SOURCE.json").read_bytes())
        hashes = source["source_hashes"]
        for filename, digest in hashes.items():
            with self.subTest(filename=filename):
                self.assertEqual(digest, sha256((ROOT / filename).read_bytes()).hexdigest())
        self.assertEqual(source["delivered_source_id"],
                         sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest())


if __name__ == "__main__":
    unittest.main()
