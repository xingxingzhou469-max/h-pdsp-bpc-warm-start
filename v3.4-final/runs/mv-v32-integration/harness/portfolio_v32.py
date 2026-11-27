#!/usr/bin/env python3
"""v3.2 portfolio entry.

Real-world Latet instances with K>=3 are routed to the frozen v2.1 engine: the
astra K3 region bound is systematically loose on that family. Every other graph
uses the astra v3.2 harness with the selected per-K policy. The routing decision
and both engine receipts are recorded in the arm directory.
"""
import argparse
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
PYTHON = "/opt/anaconda3/bin/python3"
V21 = PROJECT / "交付/github_release_v21/repo/solve_multi.py"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def fallback_reason(raw):
    vehicles = int(raw.get("vehicles", 0))
    family = str(raw.get("family_id", "")).lower()
    source = raw.get("source") or {}
    if (vehicles >= 3
            and (family.startswith("latet") or source.get("table_dataset") == "Latet")):
        return "latet-k3"
    sites = len(raw.get("supply", [])) - 1
    if vehicles == 2 and sites <= 20:
        return "small-k2"
    if vehicles >= 3 and sites <= 40:
        return "small-k3plus"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoints", type=int, nargs="*", default=[60])
    args = parser.parse_args()
    if not os.environ.get("ASTRA_LOCK_OWNER"):
        raise SystemExit("invoke via harness/locked.py so the M4 lock is held")
    if not 0 <= args.seed < 2**32 or args.seconds <= 0:
        raise SystemExit("invalid seed or budget")
    source = args.instance.resolve()
    raw = json.loads(source.read_bytes())
    out = args.out.resolve()
    reason = fallback_reason(raw)
    fallback = reason is not None
    started = datetime.now().astimezone().isoformat()
    began = time.monotonic()
    if fallback:
        if out.exists():
            raise SystemExit(f"output already exists: {out}")
        out.mkdir(parents=True)
        command = [PYTHON, "-B", str(V21), str(source), "--method", "v2.1",
                   "--seconds", str(args.seconds), "--seed", str(args.seed),
                   "--output", str(out / "solution.json"),
                   "--proof-output", str(out / "proof.json")]
        with (out / "worker.log").open("wb") as stream:
            proc = subprocess.run(command, cwd=PROJECT, stdout=stream,
                                  stderr=subprocess.STDOUT, timeout=args.seconds + 180)
        record = dict(schema="v32-portfolio-1", method="v3.2", engine="v2.1",
                      fallback=True, fallback_reason=reason,
                      input_sha256=digest(source), vehicles=raw.get("vehicles"),
                      seconds=args.seconds, seed=args.seed, command=command,
                      exit=proc.returncode, started=started,
                      ended=datetime.now().astimezone().isoformat(),
                      elapsed_s=round(time.monotonic() - began, 3),
                      engine_sha256=digest(V21))
    else:
        command = [PYTHON, "-B", str(ROOT / "harness/solve.py"), str(source),
                   "--out", str(out), "--seconds", str(args.seconds),
                   "--seed", str(args.seed), "--checkpoints", *map(str, args.checkpoints)]
        proc = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              timeout=args.seconds + 180)
        record = dict(schema="v32-portfolio-1", method="v3.2", engine="astra",
                      fallback=False, fallback_reason=None,
                      input_sha256=digest(source), vehicles=raw.get("vehicles"),
                      seconds=args.seconds, seed=args.seed, command=command,
                      exit=proc.returncode, started=started,
                      ended=datetime.now().astimezone().isoformat(),
                      elapsed_s=round(time.monotonic() - began, 3))
        if out.is_dir():
            (out / "router.log").write_text(proc.stdout)
            run = out / "run.json"
            if run.is_file():
                manifest = json.loads(run.read_text())
                record["policy"] = manifest.get("plan")
                record["harness_seed"] = manifest.get("seed")
    if not out.exists():
        out.mkdir(parents=True)
    (out / "portfolio.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: record[key] for key in
                      ("engine", "fallback", "seed", "seconds", "exit", "elapsed_s")},
                     ensure_ascii=False), flush=True)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
