"""Replay every timed L and U for a cold arm, or a frozen public schedule."""

import argparse
from copy import deepcopy
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
import re
import signal

from legacy_cache import Reuse
from proofs import decode, fraction, lattice_parent, need, oracle, replay_certificate, stdlib_only


NS = 10**9
CLOCK_TOLERANCE_NS = NS // 4
CASE_NAME = re.compile(r"[A-Za-z0-9_.-]+")
KIND_NAME = re.compile(r"[A-Za-z0-9-]+")
REJECTIONS = (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError, OverflowError)
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "HIGHS_NUM_THREADS",
           "BLIS_NUM_THREADS", "RAYON_NUM_THREADS")


def digest(payload):
    return sha256(payload).hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def checked_payload(data, registered, label):
    need(digest(data) == registered, f"{label} SHA-256 differs")
    return data


def checked_clocks(mono, wall, label):
    need(type(mono) is int and type(wall) is int and min(mono, wall) >= 0,
         f"{label} clocks must be nonnegative integer nanoseconds")
    need(abs(mono - wall) < CLOCK_TOLERANCE_NS, f"{label} clocks drifted")
    return mono, wall


def checked_run_manifest(manifest, payload, vehicles, seconds, seed):
    checked_payload(payload, manifest["input_sha256"], "manifest input")
    need(type(manifest["vehicles"]) is int and manifest["vehicles"] == vehicles
         and type(manifest["seed"]) is int and manifest["seed"] == seed,
         "run fleet or seed differs from registered arm")
    points = manifest["checkpoints"]
    need(type(points) is list and points and points[-1] == seconds
         and all(type(value) is int and value > 0 for value in points)
         and points == sorted(set(points)), "invalid run checkpoint schedule")
    need(type(manifest["started_wall_ns"]) is int and manifest["started_wall_ns"] > 0,
         "invalid run wall-clock start")
    need(fraction(manifest["wall_elapsed_s"]) >= 0, "negative run duration")
    return points


def check_timed_row(row, index, previous, deadline):
    kind = row["kind"]
    need(type(kind) is str and KIND_NAME.fullmatch(kind), "invalid artifact kind")
    name = "events/optimal.json" if kind == "optimal" else f"events/{index:04d}-{kind}.json"
    need(row["file"] == name, "unexpected publication path or event index")
    clocks = checked_clocks(row["elapsed_ns"], row["wall_ns"], "publication")
    need(all(a <= b <= deadline for a, b in zip(previous, clocks)),
         "publication precedes previous event or exceeds deadline")
    need(type(row["score"]) is str and F(row["score"]) >= 0, "invalid event score")
    return clocks


def check_receipt(folder, manifest, seconds, seed, case, arm):
    """Validate the external supervisor's start, deadline and end record."""
    path = folder / "clock.json"
    payload = path.read_bytes()
    clock = decode(payload)
    need(clock["schema"] == "v33-cloud-cold-clock-1"
         and clock["input_sha256"] == manifest["input_sha256"]
         and (case is None or clock["case"] == case)
         and (arm is None or clock["method"] == arm), "cold supervisor identity differs")
    start_mono = clock["started_mono_ns"]
    start_wall = clock["started_wall_ns"]
    need(type(start_mono) is int and start_mono > 0 and type(start_wall) is int
         and start_wall == manifest["started_wall_ns"], "run start differs from cold supervisor")
    if "started_mono_ns" in manifest:
        need(manifest["started_mono_ns"] == start_mono, "run monotone start differs from cold supervisor")
    need(clock["deadline_ns"] == seconds * NS and clock["seed"] == seed
         and type(clock["deadline_ns"]) is int and type(clock["seed"]) is int,
         "cold supervisor budget or seed differs")
    if "checkpoints" in clock:
        need(clock["checkpoints"] == manifest["checkpoints"]
             and all(type(point) is int for point in clock["checkpoints"]),
             "cold supervisor checkpoints differ from run")
    need(type(clock["ended_mono_ns"]) is int and type(clock["ended_wall_ns"]) is int,
         "cold supervisor end clocks must be integer nanoseconds")
    duration = checked_clocks(clock["ended_mono_ns"] - start_mono,
                              clock["ended_wall_ns"] - start_wall, "supervisor duration")
    command = clock["command"]
    need(type(command) is list and len(command) == 11 + len(manifest["checkpoints"])
         and all(type(part) is str for part in command) and command[1] == "-B"
         and Path(command[3]).stem == clock["case"] and command[4] == "--out"
         and command[6:] == ["--seconds", str(seconds), "--seed", str(seed), "--checkpoints",
                             *map(str, manifest["checkpoints"])],
         "supervisor invocation differs from the registered arm")
    cpu = clock["cpu"]
    need(type(cpu) is list and len(cpu) == 1 and type(cpu[0]) is int and cpu[0] >= 0
         and clock["thread_env"] == dict.fromkeys(THREADS, "1"),
         "cold arm was not restricted to a single CPU and thread")
    need(type(clock["cutoff_reached"]) is bool and type(clock["exit_code"]) is int,
         "malformed process completion status")
    if clock["cutoff_reached"]:
        need(type(clock["kill_mono_ns"]) is int and type(clock["kill_wall_ns"]) is int,
             "missing supervisor kill clocks")
        kill = checked_clocks(clock["kill_mono_ns"] - start_mono,
                              clock["kill_wall_ns"] - start_wall, "supervisor kill")
        need(max(kill) >= seconds * NS and all(a <= b for a, b in zip(kill, duration))
             and clock["exit_code"] in (0, -signal.SIGKILL),
             "worker terminated before the cutoff or had an unexpected exit")
    else:
        need(clock["exit_code"] == 0 and clock["kill_mono_ns"] is None
             and clock["kill_wall_ns"] is None
             and manifest.get("manifest_origin") != "deadline-supervisor",
             "worker exit and supervisor evidence disagree")
    # Manifest durations are measured just before publishing run.json.
    need(fraction(manifest["wall_elapsed_s"]) * NS <= duration[1] + 1,
         "run completion time exceeds the supervisor lifetime")
    return clock, duration, digest(payload)


def check_observation(observation, row, length, clock, line=None):
    elapsed = observation["artifact_mtime_ns"] - clock["started_wall_ns"]
    need(observation["file"] == row["file"] and observation["sha256"] == row["sha256"]
         and type(observation["size"]) is int and observation["size"] == length
         and type(observation["artifact_mtime_ns"]) is int and 0 <= elapsed <= row["wall_ns"],
         "external artifact observation disagrees with publication")
    if clock.get("checks") == "dual-clock-live-prefix":
        mono, wall = observation["observed_mono_ns"], observation["observed_wall_ns"]
        need(type(mono) is int and type(wall) is int, "invalid external observation clock")
        observed = checked_clocks(mono - clock["started_mono_ns"], wall - clock["started_wall_ns"],
                                  "external observation")
        need(row["elapsed_ns"] <= observed[0] <= clock["deadline_ns"] and
             row["wall_ns"] <= observed[1] <= clock["deadline_ns"] and
             observation["ledger_line_sha256"] == digest(line),
             "external observation occurred after cutoff or observed different ledger bytes")
        return max(observed)
    return max(row["elapsed_ns"], row["wall_ns"])


def rejection(name, action):
    try:
        action()
    except REJECTIONS as error:
        return {"mutation": name, "rejected": True, "detail": str(error)}
    raise ValueError(f"tamper accepted: {name}")


def semantic_attacks(instance, k, kind, doc, model, input_sha):
    # an internally consistent score claim with SHA and ledger recomputed must still fail
    changed = deepcopy(doc)
    key = "lower" if kind == "L" else "upper"
    changed[key] = str(fraction(changed[key]) + F(1, 10**6) * (1 if kind == "L" else -1))
    if kind == "v21":
        changed["proof"]["score"] = changed[key]
        changed["proof"]["certificate"]["upper"] = changed[key]
    mutations = [("falsified exact score", changed)]
    if kind == "screen2":
        # The screen proves a requested threshold. A nearby threshold may
        # legitimately share the same complete masks and replay successfully.
        mutations = [
            ("negative bound threshold", {**doc, "upper": "-1"}),
            ("wrong graph digest", {**doc, "instance_digest": "0" * 64}),
        ]
        if doc["screen"] is not None:
            probe = {**doc["screen"], "mask_count": doc["screen"]["mask_count"] + 1}
            mutations.append(("forged complete mask cardinality", {**doc, "screen": probe}))
    if kind == "fleet-selection-v33":
        parent = doc["parent"]
        mutations.append(("wrong parent bound", {**doc, "parent": {
            **parent, "upper": str(F(parent["upper"]) + 1)}}))
    if kind == "sparse-supplier-fair-v2":
        mutations.append(("omitted supplier cover leaf", {**doc, "leaves": doc["leaves"][:-1]}))
        for i, leaf in enumerate(doc["leaves"]):
            if leaf["line"] is not None and leaf["line"]["route_bounds"]:
                invalid = deepcopy(doc)
                invalid["leaves"][i]["line"]["route_bounds"][0] += 1
                mutations.append(("forged route price", invalid))
                break
    if kind == "fleet-cut-dual-v33":
        mutations.append(("wrong graph digest", {**doc, "instance_sha256": "0" * 64}))
        support = doc["support"]
        if support is not None:
            dual_mutations = [
                ("connection cut with invalid anchor", {"cuts": [["connect", 2, 0, 1]]}),
                ("blossom cut with an even number of teeth", {"cuts": [["blossom", 2, [], 1]]}),
            ]
            if "station" in support:
                dual_mutations.extend((
                    ("negative station multiplier", {"station": [0, -1, *support["station"][2:]]}),
                    ("optimistic route bound", {"route_bound": support["route_bound"] - 1}),
                    ("odd station rank cut contains depot", {"parity": [[7, 1]]}),
                    ("zero slope tilt", {"root_tilts": ["0"]}),
                ))
            else:
                dual_mutations.append(("negative time multiplier", {"time": -1}))
            for label, alteration in dual_mutations:
                mutations.append((label, {**doc, "support": {**support, **alteration}}))
    if kind == "lattice":
        altered = deepcopy(doc)
        altered["parameters"]["delta"] = str(F(doc["parameters"]["delta"]) + 1)
        parent_kind, parent = lattice_parent(doc)
        bad_parent = json.dumps({"upper": "-1", "tampered": True}) + "\n"
        mutations.extend((
            ("falsified grid parameters", altered),
            ("wrong lattice input digest", {**doc, "input_sha256": "0" * 64}),
            ("changed parent bytes", {**doc, "parent": {**doc["parent"],
                                                       "bytes": doc["parent"]["bytes"] + " "}}),
            ("rehash of invalid parent", {**doc, "parent": {**doc["parent"], "bytes": bad_parent,
                                                          "sha256": digest(bad_parent.encode())}}),
        ))
    results = [rejection(f"{kind}: {label}",
                         lambda: replay_certificate(instance, k, kind, changed, model, input_sha))
               for label, changed in mutations]
    if kind == "lattice":
        results.extend(semantic_attacks(instance, k, parent_kind, parent, model, input_sha))
    return results


def proof_variant(kind, doc):
    if kind == "lattice":
        parent_kind, parent = lattice_parent(doc)
        return kind, proof_variant(parent_kind, parent)
    if kind == "screen2":
        return kind, "inherited" if doc["screen"] is None else "screened"
    if kind != "fleet-cut-dual-v33":
        return kind
    support = doc["support"]
    return kind, "box" if support is None else "route" if "station" in support else "arc"


def check_interval(low, high):
    need(low <= high, "verified L exceeds verified U")


def check_optimal(doc, row, lower, upper, input_sha):
    need(doc["schema"] == "mv-lattice-optimal-1" and doc["input_sha256"] == input_sha,
         "wrong optimality proof format or input")
    need(lower is not None and upper is not None and upper["kind"] == "lattice",
         "optimality proof requires preceding L and lattice U")
    for name, endpoint in (("lower", lower), ("upper", upper)):
        need(doc[name] == {"file": endpoint["file"], "sha256": endpoint["sha256"]},
             f"optimality {name} reference differs from the current endpoint")
    need(type(doc["value"]) is str and F(doc["value"]) == F(row["score"])
         == F(lower["score"]) == F(upper["score"]), "optimality proof endpoints differ")


def check_live_equality(original, lower, upper):
    """Independently replay the timely candidates before the producer stops."""
    stdlib_only()
    payload = original.read_bytes()
    instance = decode(payload)
    model = oracle().Model.read(instance)
    vehicles = instance["vehicles"]
    input_sha = digest(payload)
    low = replay_certificate(instance, vehicles, "L", decode(lower.read_bytes()), model, input_sha)
    high = replay_certificate(instance, vehicles, "lattice", decode(upper.read_bytes()), model, input_sha)
    need(low == high, "live optimality candidates do not agree")


def verify_arm(folder, original, seconds, seed=7, *, reuse=True, **rules):
    """Check all files and return the replayed anytime endpoints."""
    with Reuse(reuse) as cache:
        return _verify_arm(folder, original, seconds, seed, cache, **rules)


def _verify_arm(folder, original, seconds, seed, cache, *, clock_required=True, attacks=None,
                registered_sha=None, expected_checkpoints=None, case=None, arm=None,
                require_lattice=False, require_live=False):
    stdlib_only()
    folder, original = Path(folder), Path(original)
    original_payload = original.read_bytes()
    if registered_sha is not None:
        checked_payload(original_payload, registered_sha, "registered original input")
    copied = (folder / "input.json").read_bytes()
    need(copied == original_payload, "timed arm input differs from original bytes")
    instance = decode(original_payload)
    model = oracle().Model.read(instance)
    vehicles = instance["vehicles"]
    need(type(vehicles) is int and vehicles > 0, "invalid input fleet size")

    manifest_payload = (folder / "run.json").read_bytes()
    manifest = decode(manifest_payload)
    points = checked_run_manifest(manifest, copied, vehicles, seconds, seed)
    if expected_checkpoints is not None:
        need(points == expected_checkpoints, "run checkpoints differ from registered schedule")
    if clock_required:
        clock, duration, clock_digest = check_receipt(folder, manifest, seconds, seed, case, arm)
        need(not require_live or clock.get("checks") == "dual-clock-live-prefix",
             "current campaign is missing external live observations")
        observations_payload = (folder / "artifact_observations.json").read_bytes()
        observations = decode(observations_payload)
        observations_digest = digest(observations_payload)
    else:
        need(not require_live, "live observations require an external clock receipt")
        duration = (seconds * NS, seconds * NS)
        clock_digest = observations_digest = None

    ledger = checked_payload((folder / "events.jsonl").read_bytes(),
                             manifest["events_sha256"], "publication ledger")
    need(ledger.endswith(b"\n"), "publication ledger is truncated")
    lines = ledger.splitlines(keepends=True)
    rows = [decode(line) for line in lines]
    need(type(manifest["event_count"]) is int and manifest["event_count"] == len(rows),
         "publication ledger size differs")
    if clock_required:
        need(type(observations) is list and len(observations) == len(rows),
             "missing or extra external artifact observations")
    inventory = {path.name for path in (folder / "events").iterdir()}
    need(len(inventory) == len(rows), "orphan or missing certificate file")

    clocks = (0, 0)
    replay_counts = {}
    audit_attacks = []
    low = high = latest_l = latest_u = terminal = None
    endpoint_rows = []
    observation_times = {}
    for i, row in enumerate(rows):
        clocks = check_timed_row(row, i, clocks, seconds * NS)
        need(all(instant <= end for instant, end in zip(clocks, duration)),
             "publication falls outside cold supervisor lifetime")
        path = folder / row["file"]
        need(path.is_file() and not path.is_symlink(), "publication is missing or a symlink")
        observation_times[row["file"]] = (check_observation(observations[i], row, path.stat().st_size,
                                               clock, lines[i]) if clock_required else max(clocks))
        if row["kind"] == "optimal":
            need(i == len(rows) - 1, "optimality publication must be terminal")
            doc = decode(checked_payload(path.read_bytes(), row["sha256"], "optimality certificate"))
            check_optimal(doc, row, latest_l, latest_u, manifest["input_sha256"])
            terminal = row
            continue
        endpoint_rows.append(row)
        claim = F(row["score"])
        if row["kind"] == "L":
            need(low is None or claim > low, "published lower endpoint does not improve")
            low = claim
            latest_l = row
        else:
            need(not require_lattice or row["kind"] == "lattice", "candidate upper was not projected")
            need(high is None or claim < high, "published upper endpoint does not improve")
            high = claim
            latest_u = row
        if low is not None and high is not None:
            check_interval(low, high)
    need(low is not None and high is not None, "missing exact L or U")

    # Where the same support occurs in several screens, replaying stronger
    # upper claims first yields complete supersets for the weaker screens.
    for row in reversed(endpoint_rows):
        data = checked_payload((folder / row["file"]).read_bytes(), row["sha256"], "publication certificate")
        kind, doc = row["kind"], decode(data)
        cache.inspect(doc["parent"]["kind"] if kind == "lattice" else kind)
        exact = replay_certificate(instance, vehicles, kind, doc, model, manifest["input_sha256"])
        need(exact == F(row["score"]), "logged score differs from certificate replay")
        replay_counts[kind] = replay_counts.get(kind, 0) + 1
        if attacks is not None:
            attack_key = proof_variant(kind, doc)
            if attack_key not in attacks:
                audit_attacks.extend(semantic_attacks(instance, vehicles, kind, doc, model,
                                                      manifest["input_sha256"]))
                attacks.add(attack_key)
    checked = {}
    for at in points:
        timely = [row for row in endpoint_rows if observation_times[row["file"]] <= at * NS]
        lows = [row for row in timely if row["kind"] == "L"]
        highs = [row for row in timely if row["kind"] != "L"]
        need(lows and highs, f"missing exact L or U at {at} seconds")
        p, d = lows[-1], highs[-1]
        lv, uv = F(p["score"]), F(d["score"])
        check_interval(lv, uv)
        checked[str(at)] = {"lower": str(lv), "upper": str(uv),
                            "gap": str((uv - lv) / lv) if lv else None,
                            "lower_event": p, "upper_event": d,
                            "lower_observed_ns": observation_times[p["file"]],
                            "upper_observed_ns": observation_times[d["file"]]}

    if attacks is not None:
        first = rows[0]
        late = {**first, "elapsed_ns": seconds * NS + 1, "wall_ns": seconds * NS + 1}
        audit_attacks.extend((
            rejection("original input bytes", lambda: checked_run_manifest(
                manifest, copied + b" ", vehicles, seconds, seed)),
            rejection("publication ledger bytes", lambda: checked_payload(
                ledger + b" ", manifest["events_sha256"], "publication ledger")),
            rejection("publication artifact bytes", lambda: checked_payload(
                (folder / first["file"]).read_bytes() + b" ", first["sha256"], "publication certificate")),
            rejection("late publication", lambda: check_timed_row(late, 0, (0, 0), seconds * NS)),
            rejection("reversed interval", lambda: check_interval(high + 1, high)),
        ))

    return {
        "input_sha256": digest(original_payload), "vehicles": vehicles, "seed": seed,
        "seconds": seconds, "manifest_sha256": digest(manifest_payload),
        "ledger_sha256": digest(ledger), "clock_sha256": clock_digest,
        "observations_sha256": observations_digest,
        "replayed_lowers": replay_counts.get("L", 0),
        "replayed_uppers": len(endpoint_rows) - replay_counts.get("L", 0),
        "replayed_by_kind": replay_counts, "l_gt_u": 0, "all_logged_uppers": True,
        "endpoints": checked, "optimal_event": terminal,
        "optimal_observed_ns": observation_times[terminal["file"]] if terminal else None,
        "tamper": audit_attacks, "dp_cache": cache.stats(),
    }


def verify_archive(folder, schedule_path, pairs, *, selected=None, seconds_override=None, reuse=True):
    payload = schedule_path.read_bytes()
    schedule = decode(payload)
    seconds, seed = schedule["seconds"], schedule["seed"]
    need(type(seconds) is int and seconds > 0 and type(seed) is int and 0 <= seed < 2**32,
         "invalid public schedule metadata")
    cases = schedule["cases"]
    if selected is not None:
        need(len(set(selected)) == len(selected)
             and set(selected) <= {row["case"] for row in cases}, "unknown or repeated shard case")
        cases = [row for row in cases if row["case"] in selected]
    if seconds_override is not None:
        need(type(seconds_override) is int and 0 < seconds_override < seconds, "invalid smoke budget")
        seconds = seconds_override
    names = [row["case"] for row in cases]
    need(names and len(set(names)) == len(names), "duplicate or empty public schedule")
    cases_dir = folder / "arms"
    need({path.name for path in cases_dir.iterdir()} == set(names),
         "run package does not contain exactly the scheduled cases")
    seen = set()
    arm_results = {}
    for row in cases:
        name = row["case"]
        need(CASE_NAME.fullmatch(name) and name not in (".", "..") and
             row["path"] == f"inputs/{name}.json", "invalid public schedule case path")
        source = schedule_path.parent / row["path"]
        need({path.name for path in (cases_dir / name).iterdir() if path.is_dir()} == set(pairs),
             f"{name}: run package does not contain the requested arms")
        arm_results[name] = {}
        for arm in pairs:
            result = verify_arm(cases_dir / name / arm, source, seconds, seed, attacks=seen,
                                registered_sha=row["sha256"],
                                expected_checkpoints=[seconds], case=name, arm=arm, reuse=reuse)
            need(result["vehicles"] == row["vehicles"], f"{name}: registered fleet differs")
            arm_results[name][arm] = result
            print(name, arm, result["replayed_lowers"], result["replayed_uppers"], flush=True)
    results = [result for arms in arm_results.values() for result in arms.values()]
    return {
        "schedule_sha256": digest(payload), "cases": len(cases), "arms": len(results),
        "replayed_lowers": sum(row["replayed_lowers"] for row in results),
        "replayed_uppers": sum(row["replayed_uppers"] for row in results),
        "tamper_rejected": sum(len(row["tamper"]) for row in results),
        "l_gt_u": 0, "all_logged_uppers": True, "runs": arm_results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    run = commands.add_parser("run")
    run.add_argument("folder", type=Path)
    run.add_argument("--input", required=True, type=Path)
    run.add_argument("--seconds", required=True, type=int)
    run.add_argument("--seed", default=7, type=int)
    run.add_argument("--no-clock", action="store_true", help="for local truth and workbook gates")
    run.add_argument("--checkpoints", nargs="+", type=int, help="registered checkpoint schedule")
    run.add_argument("--require-lattice", action="store_true", help="require every upper to be projected")
    run.add_argument("--require-live", action="store_true", help="require live dual-clock observations")
    run.add_argument("--tamper", action="store_true")
    run.add_argument("--no-cache", action="store_true")
    run.add_argument("--output", type=Path)
    archive = commands.add_parser("archive")
    archive.add_argument("folder", type=Path)
    archive.add_argument("--schedule", required=True, type=Path)
    archive.add_argument("--output", required=True, type=Path)
    archive.add_argument("--arms", choices=("baseline", "candidate"), nargs="+",
                         default=["baseline", "candidate"])
    archive.add_argument("--cases", nargs="+", help="explicit case IDs for a known shard")
    archive.add_argument("--smoke-seconds", type=int, help="explicit shorter budget for a smoke run")
    archive.add_argument("--no-cache", action="store_true")
    live = commands.add_parser("check-optimal")
    live.add_argument("input", type=Path)
    live.add_argument("lower", type=Path)
    live.add_argument("upper", type=Path)
    args = parser.parse_args()
    if args.mode == "check-optimal":
        check_live_equality(args.input, args.lower, args.upper)
        return
    if args.mode == "run":
        summary = verify_arm(args.folder, args.input, args.seconds, args.seed,
                             clock_required=not args.no_clock,
                             attacks=set() if args.tamper else None, reuse=not args.no_cache,
                             expected_checkpoints=args.checkpoints, require_lattice=args.require_lattice,
                             require_live=args.require_live)
    else:
        summary = verify_archive(args.folder, args.schedule, args.arms,
                                 selected=args.cases, seconds_override=args.smoke_seconds,
                                 reuse=not args.no_cache)
    if args.output is not None:
        args.output.write_text(encode(summary), encoding="utf-8")
    else:
        print(encode(summary), end="")


if __name__ == "__main__":
    main()
