# H-PDSP v3.4-final

A single-core anytime solver for the multi-vehicle humanitarian pickup and distribution problem. The incumbent is a feasible fleet and its upper bound has an independently replayable certificate. When the endpoints coincide, the final certificate proves the optimum.

This is the frozen research candidate evaluated on Apple M4 on 2026-11-27, following the v2.1 release in the same repository. See [benchmark results](BENCHMARKS.md) for both improvements and regressions.

## Verify the included results

Python 3.11 or later is sufficient for verification. From this directory run:

```sh
python3 -B -S verify_examples.py
```

This checks the SHA-256 manifest, reconstructs the original input and certificate chain in the Python standard library, and verifies the two exact results. The Latet-1908 input is byte-identical to the instance in v2.1.0. The included Latet clock record uses portable command paths and retains the original timestamps and proof hashes. `directed-13` is a synthetic directed transit example.

## Build and run

Solving uses a C++20 compiler, Python and the dependencies in `requirements.txt`. The numerical versions are those used by the M4 evaluation.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python build_native.py
```

On macOS, cold starts use the project CPU lock:

```sh
python -B runs/mv-v33-astra/local_lock.py python -B runs/mv-v34-final/solve_fleet.py \
  examples/latet-1908-k2.json --out output/latet-1908-k2 --seconds 60 --seed 7
```

On Linux, start the solver directly:

```sh
python -B runs/mv-v34-final/solve_fleet.py examples/latet-1908-k2.json \
  --out output/latet-1908-k2 --seconds 60 --seed 7
```

The output directory must be new. The JSON format is illustrated by the examples and specifies a depot at index 0, disjoint supplier and agency sites, service and directed travel times, capacity, time limit, and fleet size. Every published endpoint references its exact proof in the event ledger. Replay a run made directly through the solver with the command below.

```sh
python -B -S runs/mv-v34-final/audit/replay.py run output/latet-1908-k2 \
  --input examples/latet-1908-k2.json --seconds 60 --seed 7 --no-clock --require-lattice
```

`runs/mv-v34-final/freeze.json` is the original experiment freeze. The 29 distributed runtime source files match its recorded SHA-256 values. The release manifest covers every distributed file. The experiment freeze SHA-256 is `b3a51e5bfd7a0fa524c94744b8508f8ce9432deb4716a2348c805b5a9eb176cb`.
