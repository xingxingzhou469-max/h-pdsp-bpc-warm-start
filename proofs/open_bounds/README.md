# Archived upper-bound certificates

Run this command from the repository root to replay all eight historical intervals' upper endpoints. The verifier uses Python's standard library.

```sh
python3 -B -S proofs/open_bounds/replay.py
```

Pass one or more instance IDs to select a subset. The optional `blind-09` certificate can be replayed explicitly with `python3 -B -S proofs/open_bounds/replay.py blind-09`. It supplies a finite upper bound of `1632000/2353`.

Each xz file contains JSON that maps a source name to a version-1 certificate. Replay checks the published instance SHA-256, reconstructs the routing rows from the instance and replays all leaves of a complete binary partition on visits. The maximum over the leaves bounds the full instance. The CLI returns both source bounds where available and selects their smaller value. Ties favor fewer leaves and then the source name. The result includes the SHA-256 of the instance and the published xz file. [Exact verifier](../../hpdp/cert_verifier.py)

## Archive provenance

The provenance table records the original artifact, SHA-256 and expected replay value for each source. Paths are relative to the respective research roots. `historical_dual` is the archived final candidate certificate or the minimum among the replayed certificates from the earlier sprint. Sprint streams were checked against their archived result digests. `independent_frontier` comes from the visit-branching frontier after 600 seconds, except for the 300-second `random-100-5` frontier. The eight paired results agree with the archived cross-run intervals. [PROVENANCE.csv](PROVENANCE.csv)

The independent frontiers used a connectivity cut of the form `y_k − Σ_(i∈S,j∉S) x_ij ≤ 0`. For release, the row was represented using the arcs within the smaller side of the cut. Subtracting the sum of the outgoing degree equalities over `S` or the incoming equalities over its complement recovers the original row from the compact row. The free dual multipliers were adjusted by the cut multiplier times each corresponding degree equality. This preserves the rational bound. Zero-weight cuts were removed. Where an archived ancestor covered another archived node, the ancestor represented the full subtree. The converted source bounds agree exactly with the original frontier maxima.

## Bound calculation

Let `N` be the scaled total population, `P` the population of visited delivery sites, and `F` the maximum total delivery on a fixed route in the scaled flow units. Taking as much as possible at each pickup and emptying the truck at each delivery attains this integer maximum. If `F_0` is the actual total delivered, the inequity penalty against unvisited sites is at least `F_0(N−P)/N`. The route score is therefore at most `FP/N` in the scaled flow units. The relaxation uses either the sum of visited pickups, each capped at vehicle capacity, or a flow variable bounded by that sum and by one truckload per pickup-to-delivery transition.

For each leaf the model has inequalities `Ax ≤ b`, equalities `Ex = d` and a box `0 ≤ x ≤ 1`. Its objective vector `c` represents a positive weighted reward on flow and population. For integer multipliers `lambda ≥ 0`, free integer multipliers `mu`, and positive integer denominator `q`, the verifier computes

```text
T = floor((lambda · b + mu · d
           + sum_j max(0, q*c_j - (A^T lambda)_j - (E^T mu)_j)) / q).
```

The integer route reward `aF+bP` is at most `T`. For the independent frontier, the archived slope `(a_0, b_0)` was encoded as `(a_0*N, b_0*Fmax)` and reduced by its greatest common divisor. The historical program already records the integer slope. With global ceilings `Fmax` and `N`, the product is bounded by

```text
min(Fmax*N, floor(T*T/(4*a*b)),
    floor(Fmax*(T-a*Fmax)/b) when 2*a*Fmax <= T,
    floor(N*(T-b*N)/a) when 2*b*N <= T).
```

Replay divides the leaf product bound by `N` times the flow scale to obtain the bound in the original objective units. [Routing kernel](../../hpdp/routing_exact.py) [Cut encoder](../../hpdp/cut_cert.py)
