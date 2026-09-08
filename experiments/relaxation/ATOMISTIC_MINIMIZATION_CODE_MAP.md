# Atomistic minimization code map

This note records the implemented numerical minimizer of the two-chain
atomistic model in `Hao_OralExam_Continuum_Limit.pdf` and the reusable code that
connects its pair potential to the matched continuum problem.

## Reusable code

| Existing file / functions | Step where it helps | How it can be used | Limitation |
|---|---|---|---|
| `clr/potentials/lj_periodic.py`: `EffectiveLJParameters`, pair derivatives, real-image sums, and analytic Fourier coefficients | Define one matched pair/continuum model | Use the same effective `V` in the atomistic energy and either the real-image or Fourier representation of `W=4*sum_m V`. | The Fourier cutoff must be rechecked when `L/a` or other parameters change. |
| `clr/relaxation/odd_periodic.py`: `build_odd_periodic`, `reduce_odd_gradient` | Define phase-fixed continuum optimization coordinates | Reconstruct `[0,a,0,-a[::-1]]` and apply its transpose-Jacobian gradient reduction in each continuum solver. | This construction assumes the established even periodic grid and is not the atomistic mean gauge. |
| `experiments/relaxation/atomistic_two_chain_lj_lbfgs_convergence_study.py` | Solve and validate the full two-chain problem | Implements Equations (12)--(13), the mean gauge, Appendix-B normalization, two fixed initializations, cutoff checks, and the paper's convergence norms. | The finite image cutoff and numerical stationarity are verified computationally; the two tracks are not selected against each other and do not prove global minimality or the uniform stability gap. |
| `experiments/potentials/lj_stacking_potential_exploration.py`: `phi_lj`, `dphi_lj_dr`, `V_gsfe_delta`, `dV_gsfe_ddelta` | Define a concrete pair potential | Use the effective even interaction `V(A) = phi_lj(sqrt(A**2 + L**2))` and `V'(A) = dphi_lj_dr(r) * A/r`. | It sums one registry point against a rigid lattice; it does not construct two relaxed chains. |
| `experiments/potentials/lj_lattice_sum_parameter_sweep.py`: `LatticeSumParams`, `LatticeSummedPotential.W`, `Wprime` | Build and differentiate the matched continuum lattice sum | Reuse the vectorized integer-image sum, analytic chain rule, and `image_count` convention `m=-M,...,M`. | The default `prefactor_W=2` is not the paper's reduced `W`; the class is an experiment-local rigid-lattice model. |
| `experiments/potentials/lj_registry_diagnostic_parameter_sweep.py`: `registry_summary`, `assumption_report` | Check the chosen continuum potential | Check phase location, evenness, and sampled monotonicity before solving. | These are numerical diagnostics, not proofs of the paper's assumptions or atomistic stability. |
| `experiments/potentials/lj_potential_convention_comparison.py`: `run_consistency_checks` | Verify scaling and derivatives | Adapt its independent `W`, `W'`, and factor checks to the selected pair potential and cutoff. | It currently compares two lattice-sum implementations only, not the atomistic energy. |
| `experiments/relaxation/jv_phase_fixed_finite_difference_lbfgs_single_case.py`: imported `build_u`, imported `reduced_gradient`, `energy_full`, `grad_full`, `solve_from_initial_guess` | Produce a readable phase-fixed continuum reference and illustrate reduced-coordinate L-BFGS | Use the shared reconstruction/transpose-chain-rule functions with the script's strict gradient acceptance. | Its `u` is the continuum correction `q=v-x`, its odd parametrization is not the atomistic gauge, and `W=2*cos` is hard-coded. |
| `experiments/relaxation/jv_unrestricted_finite_difference_lbfgs_mesh_study.py`: `solve_lbfgs`, multistarts, `periodic_u_spline`, `align_profile`, `run_and_save` | Unrestricted checks, continuum interpolation, and comparison workflow | Reuse deterministic multistart, stationarity checks, periodic spline sampling, and explicit `W`, `W'` callables. | Its default entry point intentionally remains the illustrative `W=2*cos`; its continuous zero-crossing alignment is not the atomistic discrete symmetry in Equation (15). |
| `experiments/relaxation/jv_unrestricted_lj_fourier_finite_difference_lbfgs_mesh_study.py` | Unrestricted pair-derived continuum check | Supplies the analytic K=5 LJ `W`, `W'` to the established mesh workflow and checks cutoff, shape, and `Lip(W')`. | It is a continuum minimizer study, not the two-chain atomistic convergence calculation. |
| `experiments/relaxation/atomistic_two_chain_lj_fourier_continuum_convergence_comparison.py` | Fourier-continuum convergence postprocessing | Reuses the warm-track atomistic profiles through their unsuffixed NPZ aliases, solves deterministic K=5/K=6/M=160 continuum references, and recomputes the paper's errors and slopes. | It requires the atomistic `profiles.npz`; it does not rerun the expensive atomistic optimization or analyze the seed-3 track. |
| `exploratory/notebooks/discrete_gsfe_lbfgsb_relaxation.ipynb`: `total_energy`, `total_energy_grad`, `energy_and_grad` | Historical optimizer prototype | It illustrates an `(energy, analytic gradient)` call to `fmin_l_bfgs_b` with periodic `np.roll`. | It is a one-field continuum discretization, uses an unseeded start, and has weaker convergence checks than the current scripts. |

The atomistic study and both LJ-Fourier continuum workflows import the reusable
potential module. The phase-fixed single case, cosine mesh-study core, and
atomistic study's continuum solver import the reusable odd reduction directly;
the two LJ-Fourier workflows use it through those imported solvers. The cosine
study remains available as a separate illustrative continuum calculation.

## Paper-to-code correspondence

### Atomistic grids and energy: Equations (12)--(13)

For atom count parameter `N`, construct

```text
N1 = N,       N2 = N + 1,
h1 = 2*pi/N1, h2 = 2*pi/N2,
h  = 2*h1*h2/(h1+h2) = 4*pi/(2*N+1),
Ii = {0, ..., Ni-1}.
```

Both displacement arrays are periodic. Use
`grad_i_plus(u)[n] = (u[n+1]-u[n])/h_i` and construct, for `j=3-i`,

```text
A^i_nm = (-1)^(3-i) * h*n
         + (h/h_j) * u^i_n
         - (h/h_i) * u^j_(n+m)
         - 2*pi*(h/h_i) * m,
```

where the opposite-layer index is taken modulo `N_j`. The implemented energy
must follow the paper literally:

```text
E_N = E^1 + E^2,
E^i = h_i * sum_n [0.5*(grad_i_plus(u^i)[n])**2
                   + sum_m V(A^i_nm)].
```

The two layer contributions are intentional, not accidental pair
double-counting. Numerically replace `m in Z` by the symmetric truncation
`m=-M,...,M` and verify convergence when `M` is increased.

### Symmetry and normalization: Equation (15)

The atomistic energy has the action

```text
(tau_(d,c) u)^1_n = u^1_(n+d) + h2*d + h2*c,
(tau_(d,c) u)^2_n = u^2_(n+d)        + h1*c,
(R u)^i_n         = -u^i_(-n).
```

Use the paper's representative with
`mean(u1)+mean(u2)=0` and `abs(mean(ui)) <= h/4`, where
`mean(ui)=sum(ui)/Ni`. The equality removes the continuous `c` direction; the
Appendix-B nearest-integer formula fixes `d`, while reflection is checked as a
symmetry rather than fitted by a continuous phase shift. Imposing oddness alone
would solve only a constrained problem unless an unrestricted calculation gives
the same minimum.

### Pair sum and continuum potential: Equation (37)

For the same pair potential used in `E_N`,

```text
Phi(s) = 2 * sum_(m in Z) V(s - 2*pi*m),
W(s)   = 2*Phi(s) = 4 * sum_(m in Z) V(s - 2*pi*m).
```

Thus, with `a2=2*pi`, a current `LatticeSummedPotential` base sum needs
`prefactor_W=4` to represent the paper's `W`; its default factor `2` represents
`Phi`. The atomistic pair terms themselves remain `V(A^i_nm)` with no added
factor four.

### Analytic Fourier realization of the LJ periodization

Let `c=a/(2*pi)` and include the fixed interlayer distance `L>0`, so that

```text
V(s) = 4*epsilon * [sigma^12/(L^2+c^2*s^2)^6
                    - sigma^6/(L^2+c^2*s^2)^3].
```

For the Fourier transform convention
`I_p(q)=integral_R exp(-i*q*y)/(y^2+L^2)^p dy`, direct evaluation gives

```text
I_3(q) = pi*exp(-z)/(8*L^5) * (z^2 + 3*z + 3),
I_6(q) = pi*exp(-z)/(3840*L^11)
         * (z^5 + 15*z^4 + 105*z^3 + 420*z^2 + 945*z + 945),
z = L*abs(q).
```

Poisson summation with `q_k=2*pi*k/a` therefore yields

```text
W(s) = C0 + sum_(k>=1) A_k*cos(k*s),
C0   = 16*epsilon/a * [sigma^12*I_6(0) - sigma^6*I_3(0)],
A_k  = 32*epsilon/a * [sigma^12*I_6(q_k) - sigma^6*I_3(q_k)],
W_K'(s) = -sum_(k=1)^K k*A_k*sin(k*s).
```

Equivalently, an elementary real-space closed form can be generated from

```text
T_1(s,lambda) = sinh(lambda)
                / [2*lambda*(cosh(lambda)-cos(s))],
T_(p+1) = -[1/(2*p*lambda)] * derivative_lambda T_p,
lambda = 2*pi*L/a,
```

and the required potential is the corresponding linear combination of `T_3`
and `T_6`. The Fourier evaluator is used in code because it is shorter and
makes the mode-cutoff error explicit. For `a=1`, `sigma=0.9`, `L=1`, and
`epsilon=0.5`, K=5 versus K=6 changes `W` by `1.7454e-11` and `W'` by
`1.0472e-10` on the checked phase grid.

### Phase-fixed continuum reference: Equation (6)

The continuum solver minimizes

```text
J[v] = integral_[−pi,pi] (0.5*abs(v')**2 + W(v)) dx,
v = x + q.
```

The variable named `u` in the current phase-fixed script is `q=v-x`. Sample a
fine periodic correction on each atomistic grid and compare with

```text
u_c^1 =  q/2,
u_c^2 = -q/2,
```

not with `q` or `v` directly.

### Convergence quantity: Theorem 5

For aligned errors `e^i = u_atom^i - u_c^i`, compute the paper's weighted
discrete norm

```text
||e||_(H_h^2)^2 = sum_i h_i * sum_n (
    abs(e^i_n)**2
    + abs(grad_i_plus(e^i)_n)**2
    + abs(Delta_i(e^i)_n)**2),
Delta_i(e)_n = (e_(n+1) - 2*e_n + e_(n-1))/h_i**2.
```

Theorem 5 predicts `||e||_(H_h^2) <= C*h`, with
`h=4*pi/(2*N+1)`. A log-log plot against `N` should therefore have slope `-1`;
a plot against `h` should have slope `+1`. The paper's corollary gives the same
order for the maximum norm.

The study reports two independently optimized tracks at
`N=20,40,80,160,320,640`: `sampled_continuum` is the warm-start benchmark and
`random_fourier_seed_3` is an intentionally selected phase-stress realization.
There is no lowest-energy selection between them. The seed-3 errors may have a
finite-grid sawtooth or plateau even while remaining inside an empirical O(h)
envelope; this example is not representative random-start statistics or proof
of a global minimum.

## Cosine/Lennard--Jones compatibility warning

The paper does not supply a decaying pair potential whose lattice sum is
exactly `Phi=cos`; the cosine is an illustrative continuum choice. Therefore an
LJ atomistic minimizer cannot be compared as a convergence test with the
current hard-coded `W=2*cos` minimizer. The continuum `W` and `W'` must be built
from the same LJ `V` used in `E_N`. In addition, the paper's sufficient
condition `Lip(W') < 1` is not met by `W=2*cos`, for which `Lip(W')=2`; a rate
observed for that unmatched or unstabilized choice would be empirical rather
than a verification of that sufficient-condition result.

## Implementation status

The atomistic producer now implements every component listed above. It runs the
two fixed tracks at `N=20,40,80,160,320,640`, checks `M=80` against `M=160`, and
compares deterministic 400- and 800-point continuum references. A normalized
atomistic endpoint is accepted only when it is finite, L-BFGS-B reports success,
the Appendix-B mean conditions hold, and
`el_residual_inf_norm / h <= 1e-2`.

The producer keeps four live outputs: `convergence_summary.csv`, `profiles.npz`,
`convergence_loglog.png`, and `check_report.txt`. The separate Fourier
postprocessor checks K=5 against K=6 and the M=160 real-image continuum without
overwriting the atomistic or cosine outputs.

The Fourier comparison deliberately loads the unsuffixed warm-track aliases
from `profiles.npz`; changing its continuum evaluator does not alter or rerun
the atomistic objective.

Its existing generated output directory is the preserved 2026-09-01 five-grid
run and was not overwritten during the two-track atomistic update. Rerunning
the postprocessor will refresh those artifacts to the six-grid `h2_error`
schema.
