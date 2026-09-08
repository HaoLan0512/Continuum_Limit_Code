# Atomistic two-chain LJ convergence discussion — handoff summary

Date: 2026-09-07
Repository: `Continuum_Limit_Code`
Primary script: `experiments/relaxation/atomistic_two_chain_lj_lbfgs_convergence_study.py`

## How to use this document

This is a contextual handoff record for a new conversation. It summarizes the
decisions, implementation state, numerical evidence, mathematical
interpretation, and open questions from the preceding discussion. It is not a
new instruction set. In a later conversation, inspect the live repository and
the current manuscript before relying on details that may have changed.

## 1. Original question and literature conclusion

The original atomistic study used three starts:

- `sampled_continuum`, a warm start obtained from the continuum solution;
- `zero`, a standard zero start;
- `continuum_plus_nonsymmetric`, a nonsymmetric perturbation of the continuum
  start.

The third start was unsatisfactory because it was still deliberately built
around the continuum solution and therefore was not a genuinely independent
test of general-start behavior.

The cited numerical papers use smooth Fourier or plane-wave representations
and iterative minimization methods, including L-BFGS or self-consistent
iteration, but they do not prescribe a standard random-initialization recipe.
The analytical references establish existence, stability, and continuum-limit
results rather than numerical multistart protocols. The random-Fourier start
was therefore introduced honestly as a literature-motivated adaptation, not
as a published initialization recipe.

## 2. Evolution of the implementation

### Stage A: five-seed smooth random-Fourier ensemble

The old `continuum_plus_nonsymmetric` start was first replaced by five
deterministic random starts, `random_fourier_seed_0` through
`random_fourier_seed_4`. For each seed, two independent smooth fields were
constructed on the layers' native grids using five Fourier modes,

\[
r_1(x)=c+\sum_{k=1}^{5}\frac{a_{1k}\cos(kx)+b_{1k}\sin(kx)}{k^2},
\qquad
r_2(x)=-c+\sum_{k=1}^{5}\frac{a_{2k}\cos(kx)+b_{2k}\sin(kx)}{k^2}.
\]

The full state was projected to the sum-of-means gauge and scaled so that its
maximum magnitude was $0.1$. The $1/k^2$ decay made the initial fields
smooth and kept their elastic energy stable under grid refinement.

That version ran seven starts at each $N$: continuum, zero, and five random
seeds. Its results were preserved in
`outputs/archive/output_atomistic_two_chain_lj_lbfgs_convergence_study_seven_start_snapshot_20260907.zip`.
The still earlier three-start results remain in
`outputs/archive/output_atomistic_two_chain_lj_lbfgs_convergence_study_three_start_snapshot_20260906.zip`.

### Stage B: current two-track study

The code was then simplified to exactly two fixed, independent result tracks:

1. `sampled_continuum`: the best-case continuum warm start;
2. `random_fourier_seed_3`: a fixed phase-stress example selected because it
   visibly exposes the finite-grid mean/phase sawtooth.

The zero start, seeds 0/1/2/4, lowest-energy selection, energy-tolerance
tiebreaking, and all "best start" logic were removed. The two endpoints are
not candidates in a contest; neither replaces the other. They are plotted and
reported side by side.

This also resolved the earlier concern about tiny energy differences. In the
seven-start version, nearly symmetry-degenerate endpoints could differ in
energy at floating-point scale, so a raw minimum could select a random seed
arbitrarily. A tolerance plus stable ordering was temporarily used. The current
design is cleaner: it makes no cross-start selection at all.

The current atom counts are

\[
N=(20,40,80,160,320,640),
\]

where layer 1 has $N$ atoms, layer 2 has $N+1$, and the total full-state
dimension is $2N+1$. The cases are evaluated sequentially; "parallel tracks"
means equal side-by-side scientific cases, not multiprocessing.

## 3. Current continuum calculation

The continuum correction is denoted by

\[
q(x)=v_0(x)-x,
\qquad
u_1(x)=\frac{q(x)}2,
\qquad
u_2(x)=-\frac{q(x)}2.
\]

The continuum problem is phase-fixed by an odd-coordinate parametrization and
is solved deterministically from zero at 400 and 800 grid points. The
800-point solution is the reference used in every atomistic comparison. The
400-point solution is used only for a separate reference-resolution check at
$N=640$; it is not another curve in the atomistic convergence plot and is
not part of the definition of the atomistic-continuum error.

A continuum endpoint is accepted only if it is finite, L-BFGS-B reports no
warning, its reduced gradient is at most $2\times10^{-7}$, its continuum
Euler-Lagrange residual is at most $2\times10^{-5}$, and

\[
\min_n\left(1+D^+q_n\right)>0.
\]

If L-BFGS-B stops through its relative-energy-reduction condition before these
checks pass, the same endpoint may be continued at most twice, for at most
three solver calls in total.

## 4. Current atomistic optimization and what the gauge means

For a given $N$, the full displacement has $2N+1$ components, but L-BFGS-B
optimizes $2N$ reduced coordinates. Every objective/gradient callback
reconstructs the full state in

\[
\overline u_1+\overline u_2=0.
\]

This is only one common-translation gauge condition. It does **not** impose

\[
\overline u_1=\overline u_2=0.
\]

Consequently, the optimizer is free to move in the relative-mean direction
$(m,-m)$. It does not mathematically preserve each layer mean. For the fixed
seed-3 basin, however, the unnormalized endpoint mean stays empirically close
to the start's mean, approximately $0.0433$–$0.0435$. This reflects the
observed shallow phase landscape/local basin, not an optimizer conservation
law.

Each start is constructed independently at every resolution. There is no
continuation from the preceding $N$, and each case receives exactly one
atomistic L-BFGS-B call.

## 5. Exact atomistic symmetries and Appendix B normalization

The exact discrete translation used in the code is

\[
u_1' = \operatorname{roll}(u_1,-d)+h_2d+h_2c,
\qquad
u_2' = \operatorname{roll}(u_2,-d)+h_1c,
\]

where $d\in\mathbb Z$ and $c\in\mathbb R$. Reflection is

\[
u_i'(j)=-u_i(-j)
\]

with periodic layer indices. The code's low-cost symmetry test applies these
maps to a test state and checks that the atomistic energy changes by at most
$10^{-10}$. This validates the implementation of the energy symmetry; it
does not prove that a computed minimizer itself is reflection-symmetric, does
not prove global minimality, and does not by itself establish membership in
the theorem's local solution branch.

After optimization, the code applies only the paper's discrete Appendix B
normalization. If the endpoint means before this normalization are
$\mu_1,\mu_2$, it takes

\[
d=\operatorname{round}\!\left(
\frac{h_2\mu_2-h_1\mu_1}{h_1h_2}
\right),
\qquad
c=-\frac{\mu_1+\mu_2+h_2d}{h_1+h_2},
\]

and then applies the exact translation above. The result satisfies

\[
\overline u_1+\overline u_2=0,
\qquad
|\overline u_i|\le \frac h4.
\]

The direct normalized state and its exact reflection are both compared with
the continuum reference; the one with smaller $H_h^2$ error is retained.
This is a choice between exactly symmetry-equivalent representatives.

An earlier suggestion considered a continuously shifted formula such as

\[
\widetilde u_1(x)=u_1(x-\delta)-\frac\delta2,
\qquad
\widetilde u_2(x)=u_2(x-\delta)+\frac\delta2.
\]

That is not an admissible arbitrary atomistic symmetry: the atomistic lattice
supports the discrete integer shift $d$, not a free continuous phase shift
of a computed atomistic endpoint. The current code does not use this formula
and does not continuously move seed 3 to force agreement with the continuum
branch.

## 6. What stationarity and normalization do—and do not—guarantee

The endpoint Euler-Lagrange diagnostic is

\[
\|R_{\rm EL}\|_\infty
=
\max_i\left\|\frac{\partial E/\partial u_i}{h_i}\right\|_\infty,
\qquad
\frac{\|R_{\rm EL}\|_\infty}{h}\le10^{-2}.
\]

Equivalently, the allowed physical residual is at most $10^{-2}h$, so solver
error is required to be $O(h)$. Under the relevant local stability/inverse
estimate, this is intended to keep numerical stationarity error from
overwhelming an $O(h)$ atomistic-to-continuum comparison. The check is a
numerical accuracy policy, not itself a theorem that the numerical endpoint is
close to the exact atomistic minimizer.

Acceptance currently requires:

- finite optimizer state and diagnostics;
- `warnflag == 0`;
- the Appendix B mean sum and mean bounds;
- $\|R_{\rm EL}\|_\infty/h\le10^{-2}$.

The reduced-gradient norm is still recorded, but it is no longer a redundant
absolute acceptance gate.

The important logical gap discussed in the chat is:

- stationarity says the endpoint is a credible local stationary candidate;
- Appendix B normalization chooses a representative inside an exact discrete
  symmetry orbit;
- neither operation proves that the endpoint lies in the local neighborhood
  required by the paper's stability theorem;
- neither operation rules out another local stationary branch;
- the code does not verify the paper's stability-gap hypothesis or global
  minimality.

Thus the $N=320$ seed-3 result should not be described as a proved "other
branch." It is an observed phase-stressed stationary endpoint whose normalized
relative mean differs from the warm endpoint. The available calculations do
not supply a rigorous branch classifier.

## 7. Norms and empirical convergence-rate calculation

The layer spacings and paper spacing are

\[
h_1=\frac{2\pi}{N},
\qquad
h_2=\frac{2\pi}{N+1},
\qquad
h=\frac{2h_1h_2}{h_1+h_2}=\frac{4\pi}{2N+1}.
\]

For $e=u_{\rm atom}-u_{\rm continuum}$, the headline discrete norm is

\[
\|e\|_{H_h^2}^2
=
\sum_{i=1}^2 h_i\sum_n
\left(
|e_n^i|^2+|D_i^+e_n^i|^2+|\Delta_i e_n^i|^2
\right),
\]

and the maximum error is

\[
\|e\|_\infty=\max_{i,n}|e_n^i|.
\]

These are the paper-consistent headline norms. The CSV also reports

\[
\frac{\|e\|_{H_h^2}}h,
\qquad
\frac{\|e\|_\infty}h,
\]

as descriptive convergence diagnostics. The arbitrary constant
`SCALED_ERROR_MAX = 1` and its two pass/fail envelope checks were removed.
The ratios remain in the CSV and report; they are not forced to be below one
and are not assertions that the theorem's constants equal one.

For each track and norm, the displayed slope is one least-squares fit through
all six log-log data points:

\[
p=\underset{p,b}{\operatorname{argmin}}
\sum_{j=1}^{6}
\left(\log E_{N_j}-p\log N_j-b\right)^2.
\]

Thus $E_N\approx C N^p$. Since $h\sim2\pi/N$, an $O(h)$ error has
$p=-1$. The displayed seed-3 slope $-1.081$ is a global regression result,
not the local slope of the visibly flat $N=80,160,320$ segment.

## 8. Current numerical results

All 12 atomistic endpoints in the live CSV are accepted.

| $N$ | warm $H_h^2$ | warm max | seed-3 $H_h^2$ | seed-3 max | normalized seed-3 $\overline u_1$ |
|---:|---:|---:|---:|---:|---:|
| 20  | 2.0368e-2 | 4.7784e-3 | 1.5909e-1 | 5.3481e-2 | +4.3359e-2 |
| 40  | 1.0204e-2 | 2.3688e-3 | 1.2448e-1 | 4.1450e-2 | -3.4088e-2 |
| 80  | 5.1203e-3 | 1.1850e-3 | 1.6874e-2 | 5.7998e-3 | +4.4177e-3 |
| 160 | 2.5664e-3 | 5.9410e-4 | 1.5619e-2 | 5.2286e-3 | +4.2332e-3 |
| 320 | 1.2852e-3 | 2.9766e-4 | 1.5129e-2 | 5.0206e-3 | +4.1418e-3 |
| 640 | 6.4287e-4 | 1.4931e-4 | 3.0204e-3 | 1.0171e-3 | -8.1080e-4 |

All-grid fitted slopes are:

| track | $H_h^2$ slope | maximum-error slope |
|---|---:|---:|
| `sampled_continuum` | -0.996919 | -0.999255 |
| `random_fourier_seed_3` | -1.080796 | -1.081950 |

The seed-3 adjacent $H_h^2$ slopes are approximately

\[
-0.354,\;-2.883,\;-0.111,\;-0.046,\;-2.324.
\]

They expose the sawtooth that is hidden by the all-grid slope. In particular,
the errors are almost flat from $N=80$ through $N=320$, then the
$N=640$ value is about $0.200$ times the $N=320$ value.

The two tracks' energies agree to about $3\times10^{-12}$ or better in the
stored run. Such near-equality is useful evidence of a very flat or
near-degenerate energy landscape, but it is not evidence of global minimality
and it does not make two states symmetry-equivalent unless an exact discrete
symmetry map between them is exhibited.

## 9. The seed-3 mean mode

Because the normalized means satisfy

\[
\overline u_2=-\overline u_1=-m,
\]

the constant part of the two-layer error contributes nothing to the first- or
second-difference terms. Its exact $H_h^2$ contribution is its weighted
$L_h^2$ contribution:

\[
\begin{aligned}
M_N^2
&=h_1N|m|^2+h_2(N+1)|m|^2 \\
&=4\pi |m|^2,
\end{aligned}
\qquad
\boxed{M_N=2\sqrt\pi\,|\overline u_1|}.
\]

This is the red dotted curve in the $H_h^2$ panel. It is not another
optimizer error and is not the Euler-Lagrange residual. It isolates the
constant relative-mean component of the atomistic-continuum profile error.

For seed 3 it accounts for roughly 92.8%–97.1% of the total $H_h^2$ norm
(86.1%–94.2% of the squared norm). At $N=320$, for example,

\[
M_{320}=0.0146823,
\qquad
\|e_{320}\|_{H_h^2}=0.0151285.
\]

The mean therefore dominates the seed-3 error, but it is not the entire error.
Removing the constant component orthogonally gives

\[
E_{N,\mathrm{mean\mbox{-}free}}
=\sqrt{\|e_N\|_{H_h^2}^2-M_N^2}.
\]

The derived values are

| $N$ | seed-3 mean-free $H_h^2$ error |
|---:|---:|
| 20  | 4.1041e-2 |
| 40  | 2.9867e-2 |
| 80  | 6.2828e-3 |
| 160 | 4.3313e-3 |
| 320 | 3.6475e-3 |
| 640 | 9.2823e-4 |

Its all-grid slope is approximately $-1.0563$. This mean-free quantity is
still a **solution error**. It must not be called the paper's consistency
residual. The actual consistency residual would be obtained by inserting the
sampled continuum solution into the atomistic Euler-Lagrange operator.

The warm-start means are essentially zero, but the warm-start $H_h^2$ error
still has slope about $-0.997$, and $H_h^2/h\approx0.0655$ across the grids.
This is strong evidence that simply removing the relative mean from seed 3
does not automatically improve the present calculation to $O(h^2)$.

## 10. Why an $O(h)$ mean bound can produce a plateau

Let $m_N$ be the normalized seed-3 first-layer mean and define

\[
a_N=\frac{h_2\mu_{2,N}-h_1\mu_{1,N}}{h_1h_2},
\qquad
d_N=\operatorname{round}(a_N),
\qquad
q_N=d_N-a_N,
\]

where $\mu_{i,N}$ are the means before Appendix B normalization. Then

\[
m_N=\frac h2q_N,
\qquad |q_N|\le\frac12,
\]

so

\[
|m_N|\le\frac h4,
\qquad
M_N\le\frac{\sqrt\pi}{2}h.
\]

This is an $O(h)$ **envelope**, not a monotonicity or pointwise-halving
statement. The bounded phase remainder $q_N$ can move anywhere in
$[-1/2,1/2]$ as $N$ changes.

For the seed-3 endpoints, the stored data imply:

| $N$ | $d_N$ | $q_N=2m_N/h$ | reconstructed pre-normalization $\mu_{1,N}$ |
|---:|---:|---:|---:|
| 20  | 0  | +0.28293 | 0.0433591 |
| 40  | -1 | -0.43945 | 0.0434820 |
| 80  | -1 | +0.11320 | 0.0434437 |
| 160 | -2 | +0.21627 | 0.0433808 |
| 320 | -4 | +0.42254 | 0.0433504 |
| 640 | -9 | -0.16530 | 0.0433334 |

From $80\to160\to320$, $h$ approximately halves while $q_N$
approximately doubles. Their product $m_N=(h/2)q_N$ therefore changes very
little, producing the apparent plateau. At $N=640$, rounding changes the
integer representative and $q_N$ wraps to a smaller magnitude, so the mean
mode drops sharply.

The empirical relation $a_{2N}\approx2a_N$ is not a theorem. In the
sum-of-means gauge, $\mu_{2,N}=-\mu_{1,N}=-\mu_N$, and

\[
a_N=-\frac{2\mu_N}{h}
=-\frac{(2N+1)\mu_N}{2\pi}.
\]

It follows that $a_{2N}\approx2a_N$ only when the unnormalized relative mean
$\mu_N$ stays approximately constant, which happens empirically for this
fixed seed-3 basin. The rigorous part is only the remainder bound
$|q_N|\le1/2$ and the resulting $O(h)$ envelope.

A fixed positive plateau cannot persist to $N\to\infty$, because the
Appendix B bound forces the mean-mode envelope to zero with $h$. Smaller
sawtooths or short plateaus may recur as the nearest integer $d_N$ changes.

## 11. Relation to the paper's convergence theorem

At a high level, the paper's argument separates two ingredients:

1. consistency: the continuum solution inserted into the atomistic equation
   has a residual of the required small order;
2. stability/local invertibility: that residual, together with controlled mean
   modes and a local-branch assumption, controls the displacement error.

Schematically, the stability estimate has the structure

\[
\|w\|_{H_h^2}
\le C\left(\|r\|+|\overline w_1|+|\overline w_2|\right),
\]

and Theorem 5 concludes, in the paper's notation and under its hypotheses,

\[
\|u^{\rm atom}-u^{\rm cont}\|_{H_h^2}
+\|u^{\rm atom}-u^{\rm cont}\|_\infty
\le C_{\rm conv}h.
\]

The constant $C_{\rm conv}$ need not be less than one and can be large. This
is why the removed `SCALED_ERROR_MAX` gate was mathematically unjustified.
Bounded or nearly constant error-over-$h$ ratios are useful descriptive
evidence of first-order scaling; a universal threshold of one does not follow
from the theorem.

The numerical endpoint residual `el_residual_inf_norm` is also distinct from
the theorem's consistency residual:

- endpoint residual: insert the computed atomistic endpoint into the
  atomistic Euler-Lagrange equation; this measures incomplete optimization;
- consistency residual: insert the sampled continuum solution into the
  atomistic Euler-Lagrange equation; this measures model/discretization
  consistency.

The current production outputs contain the first quantity, not a direct saved
calculation of the second.

## 12. Implications for possible $O(h^2)$ future work

The seed-3 plot shows that relative-mean control is an important obstruction:
if the full error is to become $O(h^2)$, an $O(h)$ mean mode cannot remain
dominant. A sharper phase/mean estimate, a legitimate $N$-dependent
continuum representative, or another theorem-compatible phase treatment may
therefore be necessary.

However, the warm-start curve is the key counterpoint. Its mean is essentially
zero, yet its full error is still clearly $O(h)$, not $O(h^2)$. Therefore
better mean control alone is not sufficient. A second-order theorem would also
need the nonconstant error to improve, for example through a sharper
consistency expansion, additional regularity/cancellation, or a first-order
atomistic corrector to the continuum ansatz.

Useful future diagnostics, not yet implemented, are:

- retain the theorem's full $H_h^2$ and maximum errors as the headline
  quantities;
- additionally plot $d_N$, $q_N$, the exact mean contribution
  $2\sqrt\pi|\overline u_1|$, and the mean-free $H_h^2$ error;
- directly compute the paper consistency residual by applying the atomistic
  Euler-Lagrange operator to the sampled continuum state;
- plot consistency quantities divided by $h$ and $h^2$;
- investigate phase choices or correctors only if they are legitimate under
  the atomistic symmetry and the theorem's comparison framework.

## 13. Current validation and artifacts

The live `check_report.txt` records 21/21 checks passed. They comprise:

- centered directional-gradient checks for one atomistic and one continuum
  objective;
- a small-$N$ exact reflection/translation energy-invariance check;
- seed-3 reproducibility, reduced-vector length, initial gauge, amplitude, and
  finite initial energy/gradient checks;
- acceptance of all 12 atomistic endpoints;
- Appendix B mean-sum and individual-mean bounds for every endpoint;
- an $M=80$ versus $M=160$ endpoint residual comparison at $N=640$;
- the separate 400-versus-800 continuum-reference check at $N=640$;
- four all-grid fitted-slope checks in $[-1.2,-0.8]$;
- monotonic decrease of both warm-start errors;
- the seed-3 $N=640$ plateau-break check.

Selected reported diagnostic values are:

- atomistic directional-gradient relative error: $6.351\times10^{-10}$;
- continuum directional-gradient relative error: $5.024\times10^{-10}$;
- exact symmetry energy defect: $1.776\times10^{-13}$;
- 400/800 continuum-reference change at $N=640$:
  $9.816\times10^{-6}$ in $H_h^2$ and $1.381\times10^{-6}$ in maximum
  norm;
- largest $M=80/160$ Euler-Lagrange difference divided by $h$:
  $1.618\times10^{-11}$;
- largest endpoint `el_residual_over_h`: approximately $6.431\times10^{-3}$,
  below the $10^{-2}$ acceptance threshold.

The four live artifacts are:

- `outputs/relaxation/output_atomistic_two_chain_lj_lbfgs_convergence_study/convergence_summary.csv`;
- `outputs/relaxation/output_atomistic_two_chain_lj_lbfgs_convergence_study/profiles.npz`;
- `outputs/relaxation/output_atomistic_two_chain_lj_lbfgs_convergence_study/convergence_loglog.png`;
- `outputs/relaxation/output_atomistic_two_chain_lj_lbfgs_convergence_study/check_report.txt`.

Two related detailed reports also exist:

- `output/pdf/atomistic_branch_diagnostics_deep_research_report.pdf`;
- `output/pdf/seed3_mean_plateau_deep_research_report.pdf`.

## 14. Scientific wording to preserve

- Call the two computations "solution tracks" or "local stationary candidate
  tracks," not two proven global minimizers.
- Call seed 3 an intentionally selected illustrative phase-stress realization.
  It is not representative random-start statistics because it was selected
  after observing its behavior.
- Describe the mean behavior as a finite-grid sawtooth under an $O(h)$
  envelope. Do not claim smooth pointwise halving or that later smaller
  sawtooths are impossible.
- Do not claim that stationarity diagnostics and Appendix B normalization prove
  branch membership.
- Do not apply an arbitrary continuous shift to an atomistic endpoint.
- Do not identify the mean-free solution error with the consistency residual.
- The numerical results support first-order convergence behavior; they do not
  prove the theorem, global minimality, or the stability-gap hypothesis.

## 15. Recommended starting point for the next conversation

First inspect the live primary script, CSV, report, and this handoff. If the
next goal is to study the mechanism behind a possible $O(h^2)$ result, the
most focused next change would be to plan a diagnostic-only extension that
adds the mean-free decomposition and the actual atomistic consistency residual
of the sampled continuum solution, while preserving the full theorem norm as
the headline result and leaving the optimization unchanged.
