"""Observe the frozen v2.1 solver under the shared cold clock and retain timed certificates."""
from argparse import ArgumentParser
from fractions import Fraction
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import monotonic_ns, time_ns


def encoded(doc):
    return (json.dumps(doc, ensure_ascii=False, separators=(',', ':')) + '\n').encode()


class Ledger:
    def __init__(self, folder, source, seconds, seed, checkpoints):
        self.folder = folder
        self.start = int(os.environ['ASTRA_COLD_MONO_NS'])
        self.wall = int(os.environ['ASTRA_COLD_WALL_NS'])
        self.limit = int(seconds * 1_000_000_000)
        self.hash = sha256(source).hexdigest()
        self.seed = seed
        self.checkpoints = checkpoints
        self.count = 0
        self.lower = None
        self.upper = None
        folder.mkdir(parents=True, exist_ok=False)
        (folder / 'input.json').write_bytes(source)
        (folder / 'events').mkdir()
        self.stream = (folder / 'events.jsonl').open('ab', buffering=0)

    def add(self, kind, value, document):
        value = Fraction(value)
        if kind == 'L' and self.lower is not None and value <= self.lower:
            return
        if kind != 'L' and self.upper is not None and value >= self.upper:
            return
        payload = encoded(document)
        path = self.folder / 'events' / f'{self.count:04d}-{kind}.json'
        temporary = path.with_suffix('.part')
        temporary.write_bytes(payload)
        os.replace(temporary, path)
        elapsed, wall = monotonic_ns() - self.start, time_ns() - self.wall
        if not 0 <= min(elapsed, wall) <= max(elapsed, wall) <= self.limit:
            path.unlink()
            return
        row = dict(kind=kind, score=str(value), file=path.relative_to(self.folder).as_posix(),
                   sha256=sha256(payload).hexdigest(), elapsed_ns=elapsed, wall_ns=wall)
        self.stream.write(encoded(row))
        self.count += 1
        if kind == 'L':
            self.lower = value
        else:
            self.upper = value

    def complete(self, case, vehicles):
        self.stream.close()
        run = self.folder / 'run.json'
        temp = run.with_suffix('.part')
        temp.write_bytes(encoded(dict(case=case, hardware='kaggle-cpu', vehicles=vehicles,
            seed=self.seed, checkpoints=self.checkpoints, input_sha256=self.hash,
            events_sha256=sha256((self.folder / 'events.jsonl').read_bytes()).hexdigest(),
            event_count=self.count, started_mono_ns=self.start, started_wall_ns=self.wall,
            clock_limit_ns=self.limit, wall_elapsed_s=(time_ns()-self.wall)/1e9,
            solver='frozen-v2.1-observed', epochs=[], load_samples=[])))
        os.replace(temp, run)


def solve(args):
    engine = Path(os.environ['V33_V21']).resolve()
    raw = args.instance.read_bytes()
    instance = json.loads(raw, parse_float=Fraction)
    vehicles = instance['vehicles']
    ledger = Ledger(args.out, raw, args.seconds, args.seed, args.checkpoints)

    def lower(witness, source, score):
        ledger.add('L', score, dict(lower=str(score), witness=witness, source=source))

    def upper(cert, source):
        score = str(cert['upper'])
        proof = dict(kind='U', input_sha256=ledger.hash, vehicles=vehicles, score=score,
                     source=source, certificate=cert)
        ledger.add('v21', score, dict(upper=score, proof=proof))

    lower(dict(routes=[], amounts=[]), 'idle', 0)
    box = min(sum(Fraction(str(x)) for x in instance['supply']),
              Fraction(str(instance['capacity'])) * sum(Fraction(str(x)) > 0 for x in instance['population']))
    ledger.add('box', box, dict(upper=str(box)))

    sys.path.insert(0, str(engine))
    from multi_vehicle import ensure_engine_on_path
    ensure_engine_on_path()
    import primal
    import pricing_core
    import upper as upper_engine
    from common import verify_lower

    spec = importlib.util.spec_from_file_location('_frozen_solve_multi', engine / 'solve_multi.py')
    solver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver)
    epoch = monotonic_ns() if args.primal_budget else ledger.start
    solver.monotonic_ns = lambda: epoch
    run_primal = primal.run_primal
    envelope = pricing_core.envelope
    run_upper = upper_engine.run

    def observe_primal(graph, k, arm, deadline, publish, prepared):
        def observed(witness, source):
            score = publish(witness, source)
            lower(witness, source, score)
            return score
        return run_primal(graph, k, arm, deadline, observed, prepared)

    def observe_envelope(data, k, exclusions):
        pricing_core.envelope = envelope
        certificate = envelope(data, k, exclusions)
        upper(certificate, 'input_box')
        return certificate

    def observe_upper(graph, k, arm, deadline, publish, **options):
        def observed(certificate, source):
            accepted = publish(certificate, source)
            if (witness := certificate.get('maximizing_witness')) is not None:
                lower(witness, source + '_witness', verify_lower(graph, k, witness))
            if accepted:
                upper(certificate, source)
            return accepted
        return run_upper(graph, k, arm, deadline, observed, **options)

    primal.run_primal = observe_primal
    pricing_core.envelope = observe_envelope
    upper_engine.run = observe_upper
    solve_budget = args.primal_budget or args.seconds
    primal_share = 1.0 if args.primal_budget else 0.5
    solver.solve_v21(instance, ledger.hash, vehicles, solve_budget, args.seed, primal_share)
    ledger.complete(args.instance.stem, vehicles)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('instance', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seconds', required=True, type=int)
    parser.add_argument('--seed', default=7, type=int)
    parser.add_argument('--checkpoints', nargs='+', type=int, default=[60])
    parser.add_argument('--primal-budget', type=int, help='fresh primal epoch, with proof times on the outer cold clock')
    args = parser.parse_args()
    if args.seconds < 1 or args.checkpoints != [args.seconds]:
        parser.error('one checkpoint at the cutoff is required')
    if args.primal_budget is not None and not 0 < args.primal_budget <= args.seconds:
        parser.error('primal epoch must fit the overall run budget')
    solve(args)


if __name__ == '__main__':
    main()
