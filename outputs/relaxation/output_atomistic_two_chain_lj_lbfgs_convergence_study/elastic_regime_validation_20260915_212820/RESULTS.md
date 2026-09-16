# Elastic-dominance implementation and results

Implemented the explicit interaction multiplier, conservative finite-cutoff dominance bounds, independent two-start sweep, reference refinement, and failure-aware reporting. Existing research artifacts were preserved.

## Verified baseline

- 20/20 focused checks passed, including gradients, Hessians, interval extrema, weighted-Jacobian bounds, and cutoff sensitivity.
- The new lambda=1 run exactly matches 693 original values and all 19 original check outcomes.
- All 42 original CSV columns and 88 original NPZ keys are retained with unchanged original values.
- SHA-256 verification confirms all four original research artifacts are unchanged.

## Sweep outcome

| lambda | Continuum cell bound | Maximum atomistic bound | Accepted atomistic endpoints | Final reference points | Comparison status |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 0.230540 | 0.352198 | 12/12 | 800 | Validated |
| 2 | 0.461081 | 0.703843 | 12/12 | 3200 | Unresolved; no convergence fit claimed |
| 3 | 0.691621 | 1.057961 | 12/12 | 800 | Validated |
| 4 | 0.922162 | 1.403497 | 11/12 | 3200 | Unresolved; no convergence fit claimed |
| 5 | 1.152702 | 1.761948 | 11/12 | 3200 | Unresolved; no convergence fit claimed |
| 8 | 1.844323 | unavailable | unavailable | unavailable | timed_out |

The continuum cell bound can be computed without a solved profile; the lambda=8 entry is not evidence that its atomistic calculation completed.

## Interpretation

- The baseline satisfies the implemented sufficient-condition diagnostics: continuum 0.230540 and maximum atomistic 0.352198.
- At lambda=3, the conservative atomistic bound reaches 1.057961. Nevertheless, the validated phase-matched H2 slopes are -1.000695 (warm) and -1.000939 (seed 3). The condition is no longer established by this bound; this does not show that the true best Lipschitz constant exceeds the threshold.
- Direct evaluation gives abs(W_lambda_second(0))=1.152330 at lambda=5 and 1.843728 at lambda=8. Thus these settings violate the continuum smallness condition for the implemented finite sum independently of upper-bound conservatism.
- Lambda=2,4,5 retain rejected continuum candidates through 3200 points. Their residual/stationarity checks remain unresolved; no tolerance was relaxed and no convergence fit is claimed.
- The warm endpoint at lambda=4,N=320 and at lambda=5,N=20 also failed stationarity and had resolved negative raw curvature. These are rejected optimizer candidates, not evidence that no stable solution exists.
- Lambda=8 was stopped by the wall-clock cap. Its first attempt had also failed the coarse continuum acceptance check. It needs a later complete run before profile/convergence conclusions.
- Increasing the multiplier visibly increases the displacement amplitude in the saved profiles. Panels with unresolved comparisons are explicitly marked; their apparent agreement is not accepted convergence evidence.

The initial sweep used 102.672 seconds and the final sweep used 797.046 seconds, totaling 899.718 seconds of the approved 900-second exploration budget. Baseline capture, focused checks, and final plot generation are separate.

## Outputs

- [Every requested case and status](final_sweep/case_summary.csv)
- [Dominance bounds](final_sweep/regime_bounds.png)
- [Convergence comparisons](final_sweep/convergence_comparison.png)
- [Profile comparisons](final_sweep/profile_comparison.png)
- [Slope table](final_sweep/slope_summary.csv)
- [Detailed case metadata](final_sweep/sweep_status.json)
- [Focused validation](final_checks_retained/elastic_regime_validation.json)
- [Artifact compatibility](artifact_compatibility.json)

## Limits and next scientific step

These are floating-point bounds for finite image sums and tested grids, not a uniform-in-N proof, a positive full-space stability gap, or a global-minimum certificate. The Poincare argument controls layer fluctuations; layer means remain separate.

Before interpreting convergence at lambda=2,4,5 or completing lambda=8, address continuum stationarity at the requested resolutions without loosening acceptance. The present implementation deliberately retains the original optimizer settings and correction scope.

## Running the implementation

Use the documented relax interpreter and run the main study with `--interaction-scales 1 2 3 4 5 8 --time-budget-seconds 900`. Without that argument it runs only lambda=1. Each invocation creates a fresh output directory; `--output-dir` must name a new directory.

The implementation guide is `docs/elastic_regime_study.md` in the repository. No optimization or plotting runs at import time.
