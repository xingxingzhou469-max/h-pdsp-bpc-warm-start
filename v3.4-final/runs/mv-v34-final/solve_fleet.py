"""Single-core anytime H-PDSP search with exact lattice endpoints."""
from time import monotonic_ns, time_ns
import os

os.environ.setdefault("ASTRA_COLD_MONO_NS", str(monotonic_ns()))
os.environ.setdefault("ASTRA_COLD_WALL_NS", str(time_ns()))

import argparse
from fractions import Fraction as F
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
from time import sleep

from lattice import LatticeTimeline

ROOT = Path(__file__).resolve().parent
V32 = Path(os.environ.get("V33_V32", ROOT.parent / "mv-v32-integration")).resolve()
V21 = Path(os.environ.get("V33_V21", ROOT.parents[1] / "交付/github_release_v21/repo")).resolve()
SPEC = importlib.util.spec_from_file_location("frozen_timeline", V32 / "harness/solve.py")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

MODES = {"fleet-search", "compact", "supplier-fairness"}
LOCAL_STAGES = {"upper", "lower", "selection", "sparse", "subsequence"}
WORKER_KIND = {"upper": "fleet-cut-dual-v33", "selection": "fleet-selection-v33",
               "sparse": "sparse-supplier-fair-v2", "subsequence": "primal", "lower": "primal"}
REALLOCATE_K2_100 = True


def exact(x):
    return x if type(x) in (int, F) else F(x)


def endpoint_dominance(raw):
    """Test the two directed depot shortcuts with only zero-operation transit roles."""
    supply, people, service = (list(map(exact, raw[key])) for key in ("supply", "population", "service"))
    travel = raw["travel"]
    n = len(supply)
    for reverse, intermediate, targets in ((False, people, supply), (True, supply, people)):
        arcs = lambda i, j: exact(travel[j][i] if reverse else travel[i][j])
        direct = [arcs(0, i) for i in range(n)]
        best = direct.copy()
        todo = {i for i in range(1, n) if intermediate[i] > 0}
        while todo:
            j = min(todo, key=best.__getitem__)
            todo.remove(j)
            head = best[j] + service[j]
            for i in range(1, n):
                if targets[i] > 0 or i in todo:
                    length = head + arcs(j, i)
                    if length < best[i]:
                        if targets[i] > 0:
                            return False
                        best[i] = length
    return True


def decision(raw, dominates=None):
    s = list(map(exact, raw["supply"][1:]))
    cap, k = exact(raw["capacity"]), raw["vehicles"]
    n, ns = len(s), sum(q > 0 for q in s)
    if n <= 12:
        return "compact"
    sparse = k >= 3 and ns <= k
    previous_small = k == 2 and n <= 20
    if n <= 126 and (sparse or previous_small):
        if dominates is None:
            dominates = endpoint_dominance(raw)
        if dominates:
            return "supplier-fairness" if sparse and ns <= 3 else "supplier-cover"
    if n < 64 or sum(min(q, cap) for q in s) > 2 * cap:
        return "route-portfolio"
    return "fleet-search"


def cycle_seed(seed, cycle):
    return (seed + cycle * 1009) % (1 << 32)


def phase_lengths(k, mode, seconds, n=0, cycle=0, remaining=None, grid_screen=False):
    """An unchanged short opening followed by progressively longer proof/search epochs."""
    if mode == "supplier-fairness":
        if cycle == 0:
            opening = min(seconds, 60)
            return [("v21-prefix", opening * .79), ("sparse", opening * .21)]
        lengths = [("sparse", 18), ("v21-primal", 36), ("lower", 6)]
    elif mode == "compact":
        lengths = [("lower", 24), ("upper", 36)]
    elif grid_screen:
        lengths = [("primal", 14), ("zone", 16), ("screen2", 29), ("prize", 1)]
    elif k == 2:
        if REALLOCATE_K2_100 and n == 100 and cycle == 0:
            lengths = [("lower", 7), ("zone", 17), ("screen2", 7), ("prize", 13),
                       ("lower", 14), ("subsequence", 2)]
        else:
            lengths = [("lower", 7), ("zone", 10), ("selection", 1), ("screen2", 5),
                       ("prize", 13), ("lower", 14), ("upper", 8), ("subsequence", 2)]
    else:
        lengths = [("lower", 7), ("zone", 12), ("selection", 1), ("prize", 13),
                   ("lower", 15), ("upper", 10), ("subsequence", 2)]
    scale = min(seconds / 60, 1) if cycle == 0 else min(1 << min(cycle, 12), remaining / 60)
    return [(kind, length * scale) for kind, length in lengths]


def plan_for(raw, seconds, seed):
    dominates = endpoint_dominance(raw)
    mode = decision(raw, dominates)
    n, k = len(raw["supply"]) - 1, raw["vehicles"]
    plan = dict(mode=mode, vehicles=k, stations=n, seed=seed, endpoint_dominance=dominates)
    eligible = (mode == "route-portfolio" and dominates and k == 2 and seconds >= 30
                and sum(exact(value) > 0 for value in raw["supply"]) == 1)
    if eligible:
        plan["grid_screen"] = True
    if mode in MODES or eligible:
        plan["stages"] = phase_lengths(k, mode, seconds, n, grid_screen=eligible)
    if mode == "route-portfolio" and not dominates and n <= 20:
        plan["zone_first"] = True
    return plan


class Search(LatticeTimeline, base.Timeline):
    def __init__(self, out, instance, k, points, plan, seed):
        super().__init__(out, instance, k, points, plan, seed)
        self.previous_zone = None
        self.previous_columns = None
        self.previous_upper = None
        self.previous_sparse = None
        self.detour_path = None
        self.env.update(V33_V32=str(V32), V33_V21=str(V21))
        if not self.proved_optimal and k and not plan["endpoint_dominance"]:
            from detours import seed as directed_seed
            stop = min(self.tick + self.limit * base.NS, monotonic_ns() + 250_000_000)
            event = directed_seed(json.loads(instance.read_bytes(), parse_float=F), stop)
            if event and F(event["lower"]) > 0:
                self.admit("L", base.encode(event))
                if self.keep["L"][0] > 0:
                    self.detour_path = self.keep["L"][1]

    def active(self, stop):
        return (not self.proved_optimal and monotonic_ns() < stop
                and time_ns() < self.wall + self.limit * base.NS)

    def legacy(self, folder, stop, kind, cycle=0):
        """Relay complete, atomically published child ledger rows into the single outer ledger."""
        prefix = kind == "v21-prefix"
        source = folder / "engine"
        folder.mkdir(parents=True)
        if kind == "route-portfolio":
            entry = V32 / "harness/solve.py"
        else:
            entry = Path(self.env.get("V33_V21_OBSERVER", ROOT / "cloud/v21_observer.py"))
        seconds = min(self.limit, 60) if prefix else self.limit
        points = self.points if kind == "route-portfolio" else [seconds]
        args = [sys.executable, "-B", str(entry), str(self.source), "--out", str(source),
                "--seconds", str(seconds), "--seed", str(cycle_seed(self.seed, cycle)),
                "--checkpoints", *map(str, points)]
        if kind == "v21-primal":
            args += ["--primal-budget", str(max(1, (stop - monotonic_ns()) // base.NS))]
        if self.plan.get("zone_first"):
            policy_name = "policy_k2.json" if self.k == 2 else "policy_k3p.json"
            policy = base.read(V32 / "harness" / policy_name)
            policy["opening"] = sorted(policy["opening"], key=lambda row: row[0] != "zone")
            policy_path = folder / "policy.json"
            base.atomic(policy_path, base.encode(policy))
            args += ["--plan", str(policy_path)]
        position = 0

        def relay():
            nonlocal position
            journal = source / "events.jsonl"
            if not journal.is_file():
                return
            with journal.open("rb") as records:
                records.seek(position)
                for line in records:
                    if self.proved_optimal or not line.endswith(b"\n"):
                        break
                    row = json.loads(line)
                    self.admit(row["kind"], (source / row["file"]).read_bytes())
                    position += len(line)

        with (folder / "worker.log").open("wb") as log:
            proc = subprocess.Popen(args, env=self.env, stdout=log, stderr=subprocess.STDOUT)
            try:
                while self.active(stop):
                    relay()
                    code = proc.poll()
                    if code is not None:
                        if code:
                            raise RuntimeError(f"{kind} exited {code}: {folder / 'worker.log'}")
                        break
                    sleep(.012 if self.limit > 60 else .018)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=.12)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                proc.wait()
            relay()

    def save_zone(self, worker):
        if worker.path.is_file():
            if (self.previous_zone is None or self.limit <= 60
                    or F(base.read(worker.path)["upper"]) < F(base.read(self.previous_zone)["upper"])):
                self.previous_zone = worker.path
        columns = worker.path.with_name("columns.json")
        if not columns.is_file():
            return
        if self.limit <= 60 or self.previous_columns is None:
            self.previous_columns = columns
            return
        # Keep both pools as proposal hints. The next pricing stage certifies every support afresh.
        fresh = base.read(columns)["routes"]
        prior = base.read(self.previous_columns)["routes"]
        pooled = dict.fromkeys(tuple(route) for route in (*fresh[:1200], *prior, *fresh[1200:]))
        path = self.out / "zone-columns.json"
        base.atomic(path, base.encode({"routes": list(pooled)[:1800]}))
        self.previous_columns = path

    def run_stage(self, kind, remaining, stop, folder, cycle):
        warm_score, warm_path = self.keep["L"]
        warm = warm_score > 0 and warm_path != self.detour_path
        if kind == "selection" and self.previous_zone is None or kind == "subsequence" and not warm:
            return None
        module = ROOT if kind in LOCAL_STAGES else V32
        argv = [sys.executable, "-B", module / kind / "solve.py", self.source,
                "--vehicles", self.k, "--out", folder, "--seconds", max(.02, remaining - .045)]
        if kind in ("lower", "prize", "subsequence"):
            argv += ["--seed", cycle_seed(self.seed, cycle)]
            if warm and (self.plan["mode"] != "compact" or cycle):
                argv += ["--warm", warm_path]
        if kind == "primal":
            argv += ["--seed", cycle_seed(self.seed, cycle)]
        if self.previous_zone and kind in ("selection", "screen2", "upper", "zone"):
            if kind != "zone" or cycle:
                flag = "--warm-route" if kind in ("zone", "screen2") else "--warm-zone"
                argv += [flag, self.previous_zone]
        if kind == "screen2" and self.previous_zone and self.plan.get("grid_screen"):
            step = self._lattice_grid["g"]
            if step and warm_score > 0 and warm_score % step == 0:
                target = warm_score + F(999, 1000) * step
                if target < F(base.read(self.previous_zone)["upper"]):
                    argv += ["--target", str(target)]
        if kind == "upper":
            if self.previous_columns:
                argv += ["--columns", self.previous_columns]
            if self.limit > 60:
                argv += ["--deep" if cycle else "--save-upper"]
                if cycle and self.previous_upper:
                    argv += ["--warm-upper", self.previous_upper]
        if kind == "sparse" and self.limit > 60:
            argv += ["--save-warm"]
            if self.previous_sparse:
                argv += ["--warm", self.previous_sparse]

        worker = base.Worker(WORKER_KIND.get(kind, kind), folder, argv, self.env)
        try:
            while self.active(stop):
                self.inspect(worker)
                if self.proved_optimal:
                    break
                code = worker.proc.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f"{kind} exited {code}: {folder / 'worker.log'}")
                    break
                sleep(.012)
        finally:
            worker.close()
        self.inspect(worker)
        if kind == "zone":
            self.save_zone(worker)
        if kind == "upper" and (folder / "warm.json").is_file():
            self.previous_upper = folder
        if kind == "sparse" and (folder / "warm.json").is_file():
            self.previous_sparse = folder / "warm.json"
        return worker.cut

    def search(self, cutoff):
        mode = self.plan["mode"]
        if "stages" not in self.plan:
            now = monotonic_ns()
            self.legacy(self.out / f"work-00-{mode}", cutoff, mode)
            self.processes.append(dict(kind=mode, began_s=(now - self.tick) / base.NS,
                                       ended_s=(monotonic_ns() - self.tick) / base.NS))
            return
        index = 0
        cycle = 0
        while self.active(cutoff):
            available = (cutoff - monotonic_ns()) / base.NS
            if available <= .08:
                break
            search_cycle = cycle if self.limit > 60 else 0
            stages = phase_lengths(self.k, mode, self.limit, self.plan["stations"], search_cycle,
                                   available, self.plan.get("grid_screen", False))
            for kind, length in stages:
                if not self.active(cutoff):
                    break
                now = monotonic_ns()
                stop = min(cutoff, now + round(length * base.NS))
                remaining = (stop - now) / base.NS
                if remaining <= .02:
                    continue
                folder = self.out / f"work-{index:02d}-{kind}"
                if kind in ("v21-prefix", "v21-primal"):
                    self.legacy(folder, stop, kind, search_cycle)
                    cut = False
                else:
                    cut = self.run_stage(kind, remaining, stop, folder, search_cycle)
                    if cut is None:
                        continue
                self.processes.append(dict(kind=kind, cycle=cycle, seed=cycle_seed(self.seed, search_cycle),
                                           allocated_s=remaining, began_s=(now - self.tick) / base.NS,
                                           ended_s=(monotonic_ns() - self.tick) / base.NS,
                                           stopped_by_scheduler=cut))
                index += 1
            if mode == "supplier-fairness" and self.limit <= 60:
                break
            cycle += 1

    def run(self):
        try:
            cutoff = self.tick + self.limit * base.NS - 40_000_000
            if self.active(cutoff):
                self.search(cutoff)
        finally:
            self.ledger.close()
        ledger = (self.out / "events.jsonl").read_bytes()
        manifest = dict(case=self.source.stem, vehicles=self.k, seed=self.seed, checkpoints=self.points,
                        input_sha256=self.fingerprint, events_sha256=sha256(ledger).hexdigest(),
                        event_count=self.counter, plan=self.plan, started_mono_ns=self.tick, started_wall_ns=self.wall,
                        wall_elapsed_s=(time_ns() - self.wall) / base.NS, epochs=self.processes, load_samples=[])
        base.atomic(self.out / "run.json", base.encode(manifest))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("instance", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=int, required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--checkpoints", nargs="*", type=int, default=[])
    args = ap.parse_args()
    if args.seconds < 1 or not 0 <= args.seed < 2**32:
        ap.error("a positive integer budget and unsigned 32-bit seed are required")
    if sys.platform == "darwin" and not os.environ.get("ASTRA_LOCK_OWNER"):
        ap.error("use the project's M4 CPU lock on macOS")
    source, out = args.instance.resolve(), args.out.resolve()
    raw = json.loads(source.read_bytes(), parse_float=F)
    plan = plan_for(raw, args.seconds, args.seed)
    points = sorted({p for p in args.checkpoints if 0 < p < args.seconds} | {args.seconds})
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    base.atomic(out.with_name(out.name + ".decision.json"), base.encode(plan))

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    Search(out, source, plan["vehicles"], points, plan, args.seed).run()


if __name__ == "__main__":
    main()
