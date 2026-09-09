# Appendix B normalization, relative means, and the seed-3 plateau

## 1. Scope and main conclusion

This handoff records the discussion and numerical checks concerning Appendix B
normalization in
`experiments/relaxation/atomistic_two_chain_lj_lbfgs_convergence_study.py`.
It distinguishes the optimizer's unnormalized relative mean from the
post-normalization mean used in the convergence error.

The main finding is:

> Applying Appendix B once to the seed-3 initial state before L-BFGS-B does not
> remove the normalized mean-mode plateau. It changes the discrete
> representative presented to the optimizer, but the usual post-solve
> normalization returns nearly the same centered phase remainder, mean-mode
> contribution, energy, and continuum error as the production workflow.

The observed plateau is therefore not explained merely by the absence of
pre-solve Appendix B normalization or by exact conservation of the original
layer mean. It is a finite-grid sawtooth in the centered fractional phase of a
local stationary symmetry orbit.

## 2. Verified production workflow

For layer 1 with `N` atoms and layer 2 with `N+1` atoms, the full atomistic
state has length `2N+1`. L-BFGS-B receives a reduced vector of length `2N`.
Every objective/gradient evaluation reconstructs the full state and enforces

\[
\overline u_1+\overline u_2=0.
\]

This is only the sum-of-means gauge. It leaves the physical relative-mean
direction

\[
(\overline u_1,\overline u_2)=(\mu,-\mu)
\]

free. The optimizer does not enforce or conserve `mu=0`.

For each `N` and each start, the production code calls
`fmin_l_bfgs_b` once. That call contains many internal iterations and line
search evaluations; `MAX_ITERATIONS=5000` is only a cap. Appendix B is not
called before or during this internal iteration. The order is

1. construct a sum-of-means-gauged initial state;
2. optimize the `2N` reduced coordinates;
3. reconstruct the raw `2N+1` endpoint;
4. apply `appendix_b_normalize()` once;
5. compare the normalized endpoint and its exact reflection with the fixed
   continuum reference;
6. retain the representative with smaller discrete `H_h^2` error.

## 3. Pre- and post-normalization quantities

Let `mu_1,N` and `mu_2,N` denote the endpoint means before Appendix B. Define

\[
a_N=\frac{h_2\mu_{2,N}-h_1\mu_{1,N}}{h_1h_2},
\qquad
d_N=R(a_N),
\qquad
q_N=d_N-a_N,
\]

where `R` is nearest-integer rounding. The code uses `numpy.rint`, so
half-integer ties go to the even integer, for example

\[
R(1/2)=0,\quad R(3/2)=2,\quad R(5/2)=2,\quad R(7/2)=4.
\]

This is compatible with the paper's nearest-integer requirement, although the
paper does not specify a half-integer tie rule.

Let

\[
m_N=\overline u_{1,N}^{\rm post}
=-\overline u_{2,N}^{\rm post}
\]

be the mean after Appendix B. Because an array roll preserves its mean, the
normalization formulas give exactly

\[
\boxed{m_N=\frac{h_N}{2}q_N},
\qquad
|q_N|\le\frac12,
\qquad
|m_N|\le\frac{h_N}{4}.
\]

Thus `mu_N` is genuinely a pre-normalization quantity. The remainder `q_N` is
computed from that pre-normalization mean, but it is simultaneously encoded in
the post-normalization mean through `q_N=2m_N/h_N`. It is the bridge between
the two stages, not a second unnormalized mean.

## 4. Why the normalized mean can dominate the `H_h^2` error

Appendix B makes the individual means `O(h)`; it does not generally make them
zero. The convergence plot uses the full discrete norm

\[
\|e\|_{H_h^2}^2
=\|e\|_{L_h^2}^2+\|D_he\|_{L_h^2}^2+\|D_h^2e\|_{L_h^2}^2.
\]

The phase-fixed continuum reference has zero layer means to numerical
precision. The constant atomistic error `(m_N,-m_N)` is invisible to the
first- and second-difference terms, but it remains in the `L_h^2` term. Since
each layer has weighted length `2*pi`, its exact contribution is

\[
M_N^2=4\pi |m_N|^2,
\qquad
\boxed{M_N=2\sqrt\pi|m_N|=\sqrt\pi h_N|q_N|}.
\]

Consequently,

\[
M_N\le\frac{\sqrt\pi}{2}h_N.
\]

This is a first-order envelope, the same order as the observed total error.
An `O(h)` quantity can therefore dominate another `O(h)` quantity when its
coefficient is larger. At `N=320`, for example,

\[
|m_{320}|=0.00414179,
\quad
M_{320}=0.0146823,
\quad
\|e_{320}\|_{H_h^2}=0.0151285.
\]

The mean mode is about `97.1%` of the norm (`94.2%` of its square), despite
satisfying the Appendix B bound.

Here `m_N` is the normalized first-layer mean, whereas `M_N` is its contribution
to the norm. They should not be used interchangeably.

## 5. Wrapped doubling mechanism

In the sum-of-means gauge, write the pre-normalization means as
`(mu_N,-mu_N)`. Then

\[
a_N=-\frac{2\mu_N}{h_N}
=-\frac{(2N+1)\mu_N}{2\pi}.
\]

If

\[
\varepsilon_N=a_{2N}-2a_N,
\]

the exact ties-to-even recurrence is

\[
\boxed{
q_{2N}=2q_N-\varepsilon_N-R(2q_N-\varepsilon_N).
}
\]

Equivalently,

\[
q_{2N}\equiv2q_N-\varepsilon_N\pmod{\mathbb Z},
\qquad |q_{2N}|\le\frac12.
\]

The congruence and bound determine the centered representative uniquely except
at a half-integer boundary; the ties-to-even convention resolves that final
ambiguity. The single-valued formula must not be asserted for an unspecified
nearest-integer tie rule.

For seed 3, the pre-normalization endpoint mean stays empirically near
`0.0433-0.0435`, making `epsilon_N` small enough that `q_N` approximately
follows wrapped doubling. The production remainders are

| `N` | `q_N` |
|---:|---:|
| 20 | +0.282933 |
| 40 | -0.439450 |
| 80 | +0.113199 |
| 160 | +0.216269 |
| 320 | +0.422538 |
| 640 | -0.165304 |

From `N=80` through `N=320`, `q_N` approximately doubles while `h_N`
approximately halves. Their product in
`M_N=sqrt(pi) h_N |q_N|` therefore stays nearly flat. At `N=640`, centered
rounding wraps `q_N`, and the mean mode drops. This is an observed finite-grid
sawtooth segment, not a proved universal period-four cycle or conservation
law.

## 6. Does L-BFGS-B preserve the relative mean?

A separate callback audit ran the continuum start and random Fourier seeds
`0-4` for every production `N`. Every callback reconstruction preserved
`mean(u1)+mean(u2)=0` to about `1e-17`, as required by the coordinates, but the
individual relative mean changed. At `N=640`, representative examples were

| start | first-layer mean entering optimization | raw endpoint mean |
|---|---:|---:|
| sampled continuum | approximately 0 | +1.14e-9 |
| seed 0 | +0.00674494 | +0.00823813 |
| seed 1 | +0.02213848 | +0.02343585 |
| seed 2 | +0.01882466 | +0.01485418 |
| seed 3 | +0.04411962 | +0.04333336 |
| seed 4 | -0.02126209 | -0.01595574 |

Seed 3 is unusually stable in relative percentage, but its mean is not an
optimizer invariant. The other seeds move by roughly `6-25%` at the finest
grid. All 36 audited solves had `warnflag=0` and passed the current scaled
Euler-Lagrange acceptance threshold, although several stopped through the
FACTR function-reduction criterion with a reduced gradient above `pgtol`.

This audit is documented in
`output/pdf/lbfgsb_relative_mean_preservation_deep_research_report.pdf`.

## 7. One-time pre-solve Appendix B A/B test

The requested read-only test changed only the initial state supplied to the
optimizer:

\[
u_{\rm seed3}
\longrightarrow
\operatorname{AppendixB}(u_{\rm seed3})
\longrightarrow
\text{existing reduced coordinates}
\longrightarrow
\text{existing L-BFGS-B solve}.
\]

The production objective, gradient, memory, tolerances, iteration cap,
line-search cap, endpoint Appendix B normalization, reflection comparison, and
continuum reference were unchanged. The external harness did not modify the
repository source.

| `N` | production `M_N` | pre-normalized `M_N` | production `H_h^2` | pre-normalized `H_h^2` |
|---:|---:|---:|---:|---:|
| 20 | 0.153704 | 0.153704 | 0.159089 | 0.159089 |
| 40 | 0.120840 | 0.121361 | 0.124476 | 0.125009 |
| 80 | 0.015660 | 0.015521 | 0.016874 | 0.016737 |
| 160 | 0.015006 | 0.014936 | 0.015619 | 0.015548 |
| 320 | 0.014682 | 0.014638 | 0.015129 | 0.015083 |
| 640 | 0.002874 | 0.002894 | 0.003020 | 0.003040 |

The `N=80-320` plateau and the `N=640` wrap/drop remain. The new and production
mean modes and continuum errors differ by less than `0.9%` at every grid.
All six new endpoints passed the current production acceptance policy.

The normalization integers explain the near-equivalence:

| `N` | pre-solve `d` | post-solve `d` | sum | production endpoint `d` |
|---:|---:|---:|---:|---:|
| 20 | 0 | 0 | 0 | 0 |
| 40 | -1 | 0 | -1 | -1 |
| 80 | -1 | 0 | -1 | -1 |
| 160 | -2 | 0 | -2 | -2 |
| 320 | -5 | +1 | -4 | -4 |
| 640 | -9 | 0 | -9 | -9 |

Thus pre-normalization mostly transfers the same total integer shift from the
end of the workflow to its beginning. It does not remove the final fractional
remainder. At `N=640`, it makes the initial mean almost zero
(`-2.45e-5`), but L-BFGS-B moves the raw endpoint to `-8.16e-4`, giving
`q_640=-0.16645` and essentially the same post-normalization mean mode as
production.

The largest initial energy defect under Appendix B was `8.88e-14`. Final
energies differed from production by at most `2.20e-12`. Small but measurable
endpoint differences remain because the reduced coordinate representative and
finite quasi-Newton stopping can alter the numerical path; they provide no
evidence that pre-normalization selected a different physical branch.

## 8. Scientific interpretation and recommendation

The calculations support the following distinctions:

- The sum-of-means constraint is enforced during every objective evaluation.
- The relative mean is free and is not generally preserved by L-BFGS-B.
- Appendix B chooses a representative in an exact discrete symmetry orbit; it
  does not impose zero individual means.
- The pre-normalization mean explains the unwrapped phase `a_N`.
- The post-normalization remainder `q_N`, equivalently `2m_N/h_N`, determines
  the plotted mean mode.
- A small `O(h)` normalized mean can dominate an `O(h)` full error.
- Pre-normalizing the same seed-3 symmetry orbit does not remove its fractional
  phase remainder or its finite-grid plateau.

The current production recommendation remains to enforce the smooth
sum-of-means gauge during optimization and apply Appendix B once after
convergence. Applying Appendix B during L-BFGS-B would introduce discontinuous
integer changes and array rolls while the optimizer retains limited-memory
curvature information; this would require a separately designed
quotient-aware method rather than an in-place callback modification.

Any future attempt to suppress the mean plateau must control the relative
phase modulo the exact discrete symmetry, select a different physical local
branch, or alter the comparison ansatz in a theorem-compatible way. It must
not simply set both layer means to zero, because that would remove a physical
relative-mean direction rather than only fix the common gauge.

The warm-start calculation is an important counterpoint: its normalized mean
is essentially zero, yet its full error is still first order. Better mean
control alone is therefore not sufficient for an `O(h^2)` result. The actual
atomistic consistency residual of the sampled continuum state remains a
separate diagnostic and has not been computed by the current production
workflow.

## 9. Relevant files and current state

- Main implementation:
  `experiments/relaxation/atomistic_two_chain_lj_lbfgs_convergence_study.py`
- Prior comprehensive handoff:
  `conversation_summaries/2026-09-07_atomistic_two_chain_lj_convergence_discussion.md`
- Production numerical table:
  `outputs/relaxation/output_atomistic_two_chain_lj_lbfgs_convergence_study/convergence_summary.csv`
- Production profiles:
  `outputs/relaxation/output_atomistic_two_chain_lj_lbfgs_convergence_study/profiles.npz`
- Mean-plateau report:
  `output/pdf/seed3_mean_plateau_deep_research_report.pdf`
- Relative-mean callback audit:
  `output/pdf/lbfgsb_relative_mean_preservation_deep_research_report.pdf`
- External pre-normalization A/B report:
  `C:/Users/lh201/.codex/visualizations/2026/09/08/01a07f71-b2a3-7dd1-8bae-9e0bad615420/pre_appendix_b_seed3_short_report.md`
- External read-only test harness:
  `C:/Users/lh201/.codex/visualizations/2026/09/08/01a07f71-b2a3-7dd1-8bae-9e0bad615420/pre_appendix_b_seed3_ab_test.py`

At the time this handoff was written, the existing modification to the main
Python file was preserved, the relative-mean PDF was untracked, and this
handoff was the only new repository file added for the present request. No
optimizer, model, parameter, output, or production source behavior was
changed.
