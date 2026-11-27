# Apple M4 validation, 2026-11-27

One cold start per registered arm, seed 7 and a 60-second budget. Latet-2209 appears in both the diagnostic and public panels and was run separately in each. The v3.4 candidate was evaluated serially with the same input bytes as the historical v3.3 controls. The public historical controls also required the supervisor to read the complete artifact by the cutoff and independently verified only the final endpoints.

| Panel | Mean gap, v3.3 → v3.4 | Median gap, v3.3 → v3.4 | Gap ≤5%, v3.3 → v3.4 |
|---|---:|---:|---:|
| Seven diagnostic cases | 5.1048% → 5.0457% | 5.3406% → 4.7535% | 2/7 → 4/7 |
| 45 held-out cases | 3.2739% → 3.1912% | 4.0561% → 3.5026% | 34/45 → 36/45 |
| 150-station subset of the held-out cases | 4.2363% → 4.2724% | 4.5927% → 4.6566% | 5/9 → 5/9 |
| 26 public cases | 3.1947% → 3.2815% | 3.5390% → 3.9932% | 20/26 → 20/26 |

All 78 v3.4 runs passed standard-library replay, covering 2,754 lower-bound and 2,535 upper-bound certificates. Two public cases reached exact equality. The included Latet-1908 repair closed at `11575200/1633`, with its terminal certificate at 24.182 seconds and a normal early exit. The included `directed-13` synthetic gate closed at `1`, and all 48 small intervals contained their exact reference values.

The routing repair moved the four targeted 40-station cases into the route portfolio. Two crossed below 5% and one showed a clear regression. The held-out aggregate improved. The public panel and the 150-station subset regressed, so this version was retained as a research candidate.
