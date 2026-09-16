# Tunable elastic-dominance study

The [two-chain LJ convergence study](../experiments/relaxation/atomistic_two_chain_lj_lbfgs_convergence_study.py)
uses a positive, dimensionless interaction multiplier

\[
E_\lambda=E_{\mathrm{elastic}}+\lambda E_{\mathrm{interaction}},
\qquad V_\lambda=\lambda V,\qquad W_\lambda=\lambda W.
\]

At `interaction_scale=1.0` this is the original model. Decreasing the multiplier
strengthens elasticity relative to the nonlinear interaction; increasing it
explores beyond the sufficient elastic-dominance condition. LJ shape, lattice
geometry, seeds, normalization, and optimizer tolerances remain fixed. The directional-gradient verification step
scales as `1e-6*max(1, interaction_scale)` to reduce subtraction cancellation
from larger energy offsets; its `1e-8` pass tolerance is unchanged. The
atomistic algorithm retains one L-BFGS-B call and its conditional correction.

## Running the sweep

From the repository root, in PowerShell:

```powershell
& 'C:\Users\lh201\anaconda3\envs\relax\python.exe' `
  experiments/relaxation/atomistic_two_chain_lj_lbfgs_convergence_study.py `
  --interaction-scales 1 2 3 4 5 8 `
  --time-budget-seconds 900
```

The default atomistic grids are `20, 40, 80, 160, 320, 640`, with warm and seed-3
starts. Each scale starts a fresh continuum solve from zero, samples its warm
start, and reuses the original seed-3 field. There is no continuation between
scales. Python callers can set the study's `interaction_scale`, `n_values`, and
`continuum_points` explicitly instead of changing module constants.

By default, the sweep creates a new
`outputs/relaxation/output_atomistic_two_chain_lj_lbfgs_convergence_study/elastic_regime_<timestamp>/`
directory. Running without arguments computes only lambda=1. To choose another
new directory, append `--output-dir` followed by its path; this selects the exact
directory supplied, which must not already exist. Existing runs are preserved.

The sweep container also holds the earlier timestamped runs, their recovery
copy, and `elastic_regime_validation_20260915_212820/`. The validation workspace
was created during implementation; normal runs do not create another one.
`output_migration.json` records the former paths and file hashes. Historical
reports and metadata retain their original provenance paths.

The restored fixed-parameter script writes its four outputs to
`outputs/relaxation/output_atomistic_two_chain_lj_lbfgs_elastic_dominance_convergence_study/`.
Repeated runs of that script replace those four files. The Fourier-continuum
postprocessor reads `profiles.npz` from this fixed-study directory.

The 900-second limit covers case computation, including case outputs. Completed
cases are checkpointed. At the limit, an active case is terminated and labeled
`timed_out`; later cases are labeled `not_run_budget_exhausted`. Combined tables
and plots are generated from completed cases after the computation budget.
An interrupted case is not a completed scientific result.

## What the dominance bounds measure

The [bound helper](../experiments/relaxation/lj_elastic_dominance.py) computes
conservative numerical bounds for the model with its finite image cutoff.

**Continuum:** report

\[
\rho_c=\lambda L_W,\qquad
L_W\geq\sup_{s\in[-\pi,\pi]}|W''(s)|,
\qquad W(s)=4\sum_{m=-M}^{M}V(s-2\pi m).
\]

Divide the phase cell into 2048 intervals. On each interval, sum signed lower
and upper bounds for the image contributions before taking absolute values.
The profile arguments must lie in the covered phase cell. Repeat the bound
with 4096 intervals and compare image cutoffs 80 and 160.

**Atomistic:** let `u_c` be the fixed-phase sampled continuum reference and
`u_d` the final normalized/reflected atomistic endpoint, each with `N` and
`N+1` layer entries. Bound the nonlinear force Jacobian on the whole segment
`(1-t)*u_c + t*u_d`, for `0 <= t <= 1`, and report

\[
\rho_a=C_P L_{\mathrm{segment}},\qquad
C_P=\frac{h_1^2}{4\sin^2(h_1/2)},\quad h_1=2\pi/N.
\]

Every pair argument varies affinely along this segment. Bound `abs(V''(A))`
on its argument interval and accumulate absolute Hessian row sums after
scaling by `D_h^(-1/2)` on each side, where
`D_h = diag(h1*I_N, h2*I_(N+1))`. This bounds the force Jacobian in the
mass-weighted Euclidean norm without creating a dense matrix.

Both bounds use interval endpoints and all enclosed stationary points of
`V''`. With `z=(c*s/L)^2`, `c=a/(2*pi)`, and `p=(sigma/L)^6`, nonzero
stationary phases follow from

\[
14z^4+36z^3+24z^2-(4+91p)z-6+21p=0.
\]

Numerical roots and outward floating-point padding are used. These bounds are
not certificates from validated interval arithmetic and do not bound the
infinite-image model's tail. Cutoff comparisons measure sensitivity separately.

The reported gap is `1-rho`. A bound below one supports the sufficient
dominance condition on the stated cell or comparison segment. A bound at or
above one means that this test does not establish the condition; it does not
establish instability or failed convergence. Bounds on finitely many grids
do not prove a uniform gap for all `N`.

The mean-sum gauge permits a relative constant mode that the elastic energy
does not control. The Poincare bound applies to separately mean-zero layer
components, so individual layer means and mean differences are reported
separately. A small `rho` alone is not a full stability certificate.

## References, acceptance, and outputs

Continuum references start at 400/800 points. For each start, compare the last
two reference grids in both fixed-phase and phase-matched comparisons, using
both `H_h^2` and maximum errors at the finest atomistic grid. If a reference
difference exceeds 5% of that error, refine to 1600 and then at most 3200
points. Refinement updates comparisons against the existing atomistic
endpoints; the warm initialization remains the original 800-point reference.
The actual initialization and final reference resolutions are recorded.
Finite continuum candidates that miss the existing tolerances are retained
with `accepted=False`. They are never relabeled as accepted. A failed coarse
reference also triggers refinement: the final comparison needs two accepted
reference resolutions and the same 5% accuracy test. Atomistic initial guesses
may use a retained candidate; endpoint acceptance remains independent and strict.

Numerical endpoint acceptance, reference/cutoff accuracy, dominance conditions,
and observed convergence are separate diagnostics. Crossing the dominance
threshold or finding different energies between starts does not end the
sweep. Rejected or unresolved comparisons are retained and excluded from
claimed fits; a fit is unavailable when its required grids are not valid.
The original baseline expectations remain separate checks at scale one.

Each completed `scale_<value>/` case contains `convergence_summary.csv`,
`profiles.npz`, `check_report.txt`, `study.pkl`, and `case_status.json`, together
with case plots and its run log. Existing profile keys and CSV columns remain
available; additional fields record the scale and diagnostics.

The sweep directory contains:

| Artifact | Meaning |
| --- | --- |
| `sweep_status.json` | Configuration, completed/failed/unfinished cases, timing |
| `case_summary.csv` | Every requested scale, status, phase-cell bound, accepted endpoint count |
| `regime_summary.csv` | Continuum and atomistic bounds, validity, errors divided by `h` |
| `slope_summary.csv` | Strict all-grid/finest-three slopes, plus valid-subset fits with their actual N values |
| `regime_bounds.png` | Dominance bounds against the sufficient-condition threshold |
| `convergence_comparison.png` | Fixed-phase and phase-matched error curves |
| `profile_comparison.png` | Both atomistic layers and their matched continuum references |

Interpret changes in slopes together with errors divided by `h`, reference
resolution, stationarity, curvature, and agreement between starts. Increased
error constants can change the appearance of a convergence curve without
changing its asymptotic rate. Multistart agreement is numerical evidence, not
a proof of global minimality. This document specifies the workflow and makes
no claim that a particular sweep completed or exhibited instability.
