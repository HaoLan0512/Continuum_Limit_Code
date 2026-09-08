---
title: "Stationarity, Symmetry Normalization, and the Atomistic Branch Question"
subtitle: "A deep-research audit of the two-chain Lennard-Jones convergence study"
author: "Continuum Limit Relaxation research audit"
date: "September 6, 2026"
subject: "Logical, mathematical, numerical, and code audit"
keywords: [atomistic model, continuum limit, L-BFGS-B, symmetry, phase, convergence]
---

**Audience:** Continuum Limit Relaxation research project

**Scope:** The one-dimensional two-chain Lennard-Jones atomistic convergence study, its current L-BFGS-B multistart implementation, the July 2026 continuum-limit manuscript, and the cited numerical and analytical literature.

## Executive conclusion

The proposed implication is not mathematically valid:

`diagnostics pass -> atomistic minimizer -> Appendix B normalization -> correct branch -> observed 1/N rate`.

There are two general logical gaps and one important correction specific to the present data.

1. Passing `atomistic_diagnostics` establishes an approximate first-order critical point of the finite-cutoff numerical objective. It does not, by itself, establish a local minimum, a global minimum, or the global discrete minimizer required by the convergence theorem.

2. Appendix B normalization chooses a representative under the exact discrete symmetries. It enforces zero total mean and bounds each layer mean by `h/4`. It does not force both means to zero, impose the continuum phase condition, or merge distinct stationary branches.

3. Nevertheless, the present `random_fourier_seed_3` endpoint is not good evidence for a different physical branch. At `N=320`, its apparently large error is dominated by a residual relative mean, or phase, that Appendix B explicitly permits. A diagnostic alignment along the continuum translation orbit reduces its error to essentially the same level as the sampled-continuum endpoint. Its unaligned error is also already bounded by `0.772 h`, so it is compatible with an `O(h)` theorem.

I therefore correct the earlier interpretation: seed 3 should not presently be described as a different branch. The data support a nearly flat phase family representing the same continuum minimizer orbit.

The code still has a real selection problem. It groups all accepted endpoints within `2e-9` in energy and then selects the first one in start order, which is `sampled_continuum`. This is a deterministic warm-start phase convention, not the promised lowest-accepted-energy rule and not a theorem-backed global-minimum certificate.

## 1. The two invalid implications

### 1.1 Small residual does not imply minimization

For a differentiable finite-dimensional energy `E_N`, a small gradient means only that the numerical endpoint is close to satisfying the first-order equation

`grad E_N(u) = 0`.

The same equation is satisfied by local minima, local maxima, and saddle points. A positive or coercive Hessian supplies evidence for a strict local minimum. Even a positive Hessian is only local information; it does not compare the endpoint with all other configurations and therefore does not prove global minimality.

The current diagnostics check reduced and full gradient norms, a scaled Euler-Lagrange residual, and mean constraints [C1-C2]. They do not check the Hessian, an energy lower bound, exhaustive basin coverage, or proximity to a theorem-certified solution.

The analytical literature makes this distinction explicit. Cazeaux, Luskin, and Massatt formulate a local minimization problem and obtain a locally unique stable equilibrium in a quantitative inverse-function-theorem neighborhood under weak coupling; they separately warn that multiple stable branches may occur outside that regime [S2]. Hott, Watson, and Luskin likewise obtain local uniqueness near a reference solution under stability, smallness, and proximity assumptions [S5]. Ortner and Theil use coercivity of the second variation, consistency, and a quantitative inverse function theorem; first-order equilibrium alone is insufficient [S6].

### 1.2 Symmetry normalization does not identify a nonlinear branch

Appendix B uses exact symmetries of the discrete energy. In the notation of the manuscript, it chooses an integer shift `d` and a real common translation `c` so that the transformed pair satisfies

`mean(u1) + mean(u2) = 0`,

`abs(mean(ui)) <= h/4`.

This operation selects one representative from an exact discrete symmetry orbit. It does not prove that the orbit is globally minimizing. It also does not identify all configurations inside the normalization cell with each other. Distinct local minima could both satisfy these two mean conditions.

Crucially, the normalization leaves an `O(h)` residual phase. The proof of the convergence theorem does not eliminate that term; it explicitly estimates it as part of the final `O(h)` bound [S1, Appendix B and Theorem 5]. Thus an Appendix-B-normalized endpoint is not required to have the same zero-mean phase as the sampled continuum representative at each finite `N`.

This is also why imposing symmetry during optimization, as done in several cited numerical papers, is conceptually different from post-processing an unrestricted endpoint. Carr et al. optimize in a restricted space with `u2 = -u1` and mirror symmetry [S3]. Nam and Koshino use periodic Fourier representations and a fixed representative [S4]. These choices constrain the admissible set before solving; they do not show that an unrestricted stationary endpoint is globally unique after normalization.

## 2. What the July 2026 theorem actually gives

The relevant current source is the July 21, 2026 manuscript, not the December 2024 draft. The older draft explicitly states that it does not rigorously connect atomistic and continuum minimizers [S7].

The current manuscript establishes the following structure [S1].

- The continuum global minimizer is unique only modulo continuous translation and `2 pi` shifts under the stated structural condition. A particular representative is fixed by a phase condition such as `v0(0) = 0`.

- The discrete energy has integer translation, real common-gauge, and reflection symmetries. A discrete global minimizer exists. Discrete uniqueness is not proved.

- Appendix B maps a discrete configuration to a normalized representative whose total mean vanishes and whose individual means are no larger than `h/4`.

- The discrete stability estimate requires a uniform positive gap `1 - C_P L_F > 0`.

- The convergence theorem applies to an exact, symmetry-normalized global discrete minimizer, subject to the potential, regularity, decay, structural, and stability hypotheses. It proves

`||u_d - u_c||_(H_h^2) <= C_conv h`.

This is a uniform upper bound. It does not say that the error must be monotone or exactly halve whenever `N` doubles. Because the integer chosen in Appendix B changes discontinuously with the raw phase, normalized errors may show a sawtooth pattern while remaining inside a constant-times-`h` envelope.

The theorem also does not say that L-BFGS-B, started anywhere, must locate the global discrete minimizer. To apply the theorem directly to a computed endpoint, one needs a separate certification or argument linking that endpoint to the theorem's global minimizer. Multistart agreement, residual checks, Hessian information, and energy comparison can strengthen numerical evidence but are not an exact proof.

## 3. What is happening to seed 3 at N = 320

### 3.1 Appendix B leaves a visible residual phase

For `N=320`, the atomistic mismatch scale is

`h = 1.9604322331293562e-2`,

so Appendix B permits an individual mean as large as

`h/4 = 4.901080582823390e-3`.

The normalized seed-3 endpoint has

`(mean(u1), mean(u2)) = (4.141789390828e-3, -4.141789390828e-3)`.

It is therefore near, but still inside, the normalization cell. Its direct fixed-phase errors against the sampled continuum representative are

- square-root `H_h^2` error: `1.512854379771e-2`;
- maximum error: `5.020598876832e-3`.

The constant relative-mean contribution alone is `1.468226111082e-2` in the layer `L2` norm, so it explains nearly all of the conspicuous discrepancy.

### 3.2 A continuum-orbit diagnostic removes the discrepancy

The continuum translation action on a pair can be written diagnostically as

`u1_delta(x) = u1(x - delta) - delta/2`,

`u2_delta(x) = u2(x - delta) + delta/2`.

Using `delta = -8.2835605408203e-3` to align the sampled atomistic endpoint with seed 3 reduces their discrepancy to

- square-root `H_h^2`: `1.036243600088e-5`;
- maximum norm: `1.402961563236e-6`.

The finite-cutoff atomistic energy changes by only about `5e-12` under this diagnostic transformation. More directly, continuously phase-aligning the sampled continuum profile to seed 3 gives

- seed 3 versus aligned continuum: `H_h^2 = 1.284794769610e-3`, maximum `2.979266313617e-4`;
- sampled-start endpoint versus fixed continuum: `H_h^2 = 1.285167499100e-3`, maximum `2.976598173321e-4`.

Those two discretization errors are effectively the same at the displayed precision.

This arbitrary real `delta` is not an exact symmetry of the finite commensurate atomistic model; the exact discrete phase shift uses integer `d`. The alignment is therefore a branch-diagnosis tool, not a replacement for the theorem's normalization or an operation that should silently alter the reported theorem error.

### 3.3 The Hessian supports one soft phase family

An independently assembled analytic finite-cutoff Hessian was checked against a centered finite-difference Hessian-vector product, with relative errors about `2e-9`. The four smallest eigenvalues in the reduced mean-gauge coordinates were

- sampled endpoint: `8.59e-10, 1.96022475e-2, 1.96525137e-2, 1.97137853e-2`;
- seed-3 endpoint: `1.54e-10, 1.96022505e-2, 1.96525120e-2, 1.97138018e-2`.

No negative mode was detected. The softest eigenvector aligns with the continuum phase tangent to approximately `0.99999998`. This strongly supports local-minimum behavior with an almost-flat phase direction and explains why endpoints with visibly different means can differ in energy by only `1e-14` to `1e-12`.

It still does not constitute a global-minimum proof: only local curvature near the two endpoints was tested, and the objective uses the numerical pair cutoff.

### 3.4 Seed 3 is already compatible with O(h)

Without any continuous phase alignment,

`1.512854379771e-2 / 1.9604322331293562e-2 = 0.7716943`.

Thus the direct Appendix-B-normalized seed-3 error is already below `0.772 h` at `N=320`. Across `N = 20, 40, 80, 160, 320`, the corresponding seed-3 `H_h^2` errors are approximately

`0.1591, 0.1245, 0.01687, 0.01562, 0.01513`.

They do not halve at every refinement, but their all-mesh log-log slope is about `-0.978`, and the observed ratios `error/h` remain below `0.803`. The plateau over `N=80` through `320` is a phase-normalization sawtooth, not evidence against an `O(h)` envelope.

At `N=640`, the rerun endpoint failed the current strict Euler-Lagrange acceptance threshold, so it cannot be used as an accepted production point. Its diagnostic error nevertheless fell to about `3.02e-3`, illustrating the phase-cell jump. This rejected point is reported only as explanatory evidence.

## 4. The remaining theorem-to-code gaps

### 4.1 Global minimizer versus approximate stationary point

The theorem concerns a global discrete minimizer. The production code returns a finite-tolerance stationary candidate from a local quasi-Newton method. The current checks are necessary and useful, but they do not supply the missing global comparison.

### 4.2 Infinite interaction versus finite cutoff

The analysis uses the full decaying pair interaction, while the computation uses pair cutoff `M=80` and checks selected quantities with `M=160`. The observed cutoff discrepancy is very small, so this is unlikely to explain the seed-3 phenomenon. It remains a formal theorem-to-code approximation that should be stated rather than erased.

### 4.3 Uniform stability gap

The production report correctly states that the theorem's uniform stability-gap hypothesis was not verified. A targeted `N=320` path calculation gave `C_P L_F` near `0.483`, which supports a positive gap for the sampled and seed-3 paths. An approximate phase scan gave similar values on the production meshes. These calculations are encouraging, but they are not yet a rigorous state-independent, infinite-sum verification of the exact theorem constant.

### 4.4 Fixed tolerances add a numerical-error term

The theorem is formulated for an exact minimizer. A computed endpoint with residual `r_N` should be understood through an additional stability-controlled numerical error, schematically

`discretization error + solver error <= C h + C_stab ||r_N||`.

Fixed absolute tolerances can eventually dominate `h` under sufficiently fine refinement. For a clean asymptotic study, solver tolerances and residual reporting should be checked relative to the target `O(h)` scale.

### 4.5 The code does not track branches across N

Every `N` is optimized independently [C5]. The comparison code applies Appendix-B normalization and optional reflection only [C4]. It does not perform continuation from one mesh to the next, minimize distance over the continuum translation orbit, or cluster endpoints into solution orbits. Therefore the current output cannot, by itself, label a change of nonlinear branch.

## 5. The concrete selection bug and reporting ambiguity

The current selection logic first finds the exact lowest accepted floating-point energy. It then treats every accepted energy within `2e-9` of that value as equivalent and assigns

`best = energy_equivalent_runs[0]`.

Because initial guesses are inserted in a fixed order, this normally selects `sampled_continuum` [C3]. At `N=320`, seed 3 has the literal lowest recorded accepted energy, lower than the sampled endpoint by `2.309e-14`; both are placed in the large `2e-9` tie window, and the warm start is selected by order.

Calling `2.309e-14` a reliable physical energy ordering would be unjustified. But silently replacing the exact-lowest rule with first-in-order is also unjustified. It biases the convergence plot toward the warm-start phase and makes it look smoother than the full multistart behavior.

The solution is to separate two scientifically different outputs:

- Lowest recorded accepted energy candidate: the literal accepted minimum, accompanied by energy resolution, residuals, local curvature evidence, and the statement that it is not a global certificate.

- Phase-tracked comparison representative: the warm-start or continuation solution used to make a stable fixed-phase convergence plot, labeled explicitly as a representative-selection convention.

Neither should be called a proven global minimizer solely because it passed the diagnostics.

## 6. Recommended code and reporting changes

### Priority 1: make selection semantics honest

- Restore the literal lowest-accepted-energy candidate required by the approved plan.
- If a warm-start phase-tracked series is useful, report it as a second series rather than using start order as a hidden tie breaker.
- Retain full-precision energy differences and state the numerical energy-resolution caveat.

### Priority 2: report the symmetry coordinates that explain the norm

- Record the normalized layer means for every accepted endpoint, not only for the selected endpoint.
- Continue reporting the theorem-faithful Appendix-B-normalized, reflection-aligned error.
- Add a separate, explicitly diagnostic error minimized over reflection and the continuous continuum phase `delta`. Do not substitute this diagnostic for the theorem error.

### Priority 3: add local-minimum and orbit evidence

- For accepted endpoints, report a few smallest reduced-Hessian eigenvalues or a cheaper coercivity estimate.
- Identify the soft phase mode and cluster endpoints only after phase/reflection alignment.
- Label positive local curvature as local-minimum evidence, not a global proof.

### Priority 4: connect computation more closely to the theorem

- Numerically verify the theorem's stability gap with clearly documented finite-cutoff and sampling errors, or keep it explicitly unverified.
- Compare `M=80`, `M=160`, and, if practical, a tail bound for both energy and stability constants.
- Scale optimization accuracy so the solver error remains negligible relative to `h` on refined meshes.
- Use continuation across `N` if the scientific goal is to follow one representative branch. Keep multistart results as a separate robustness/basin study.

## 7. Answer to the user's exact question

The diagnostics and normalization do not, in general, guarantee a minimizer in the correct branch. Diagnostics give approximate first-order stationarity; Appendix B gives a symmetry representative in an `O(h)`-wide phase cell. The theorem then applies only if the candidate is the relevant global discrete minimizer and all hypotheses, including stability, hold.

For this dataset, however, nothing has gone wrong with the theorem. The apparent seed-3 anomaly came from interpreting that `O(h)` residual phase as a distinct branch and from expecting pointwise error halving instead of an `O(h)` upper envelope. Phase-aligned profiles, local Hessians, energy differences, and the actual ratio `error/h` all support the same continuum orbit.

The main code issue is the hidden start-order tie breaker. It should be replaced by explicit, separate definitions of the lowest recorded accepted candidate and the phase-tracked comparison representative.

## 8. Limitations

- No finite numerical experiment proves global minimality over the entire configuration space.
- The Hessian audit establishes only local curvature for selected finite-cutoff endpoints.
- The arbitrary-real continuum phase alignment is diagnostic and is not an exact finite-`N` atomistic symmetry.
- The stability-gap sampling is supportive rather than a rigorous infinite-sum uniform bound.
- The `N=640` seed-3 result failed the current acceptance test and is not treated as a valid production minimizer candidate.

\newpage

## Sources and evidence ledger

[S1] *Continuum Limit of a One-Dimensional Atomistic Model for Moire Relaxation*. July 21, 2026. [Local manuscript](file:///C:/Users/lh201/Desktop/Continuum_Limit_Relaxation/Continuum_Limit_Code/Hao_OralExam_Continuum_Limit.pdf). Relevant material: continuum uniqueness modulo translations, pp. 3-5; discrete symmetries and global-minimizer existence, pp. 8-9; stability gap, pp. 11-13; convergence theorem, pp. 16-17; Appendix B normalization, pp. 18-19.

[S2] Paul Cazeaux, Mitchell Luskin, and Daniel Massatt. *Energy Minimization of Two Dimensional Incommensurate Heterostructures*. Archive for Rational Mechanics and Analysis 235 (2020), 1289-1325. [DOI](https://doi.org/10.1007/s00205-019-01444-y); [local PDF](file:///C:/Users/lh201/Desktop/Continuum_Limit_Relaxation/references/Cazeaux2020.pdf). Relevant material: local minimization, p. 15; zero-mean gauge, pp. 17-18; local stable solution and conditional uniqueness, pp. 20-21; branch warning, p. 24; numerical symmetry restriction and L-BFGS, p. 30.

[S3] Stephen Carr, Daniel Massatt, Steven B. Torrisi, Paul Cazeaux, Mitchell Luskin, and Efthimios Kaxiras. *Relaxation and Domain Formation in Incommensurate Two-Dimensional Heterostructures*. Physical Review B 98, 224102 (2018). [DOI](https://doi.org/10.1103/PhysRevB.98.224102); [local PDF](file:///C:/Users/lh201/Desktop/Continuum_Limit_Relaxation/references/2018CarrMassattTorrisiCazeauxLuskinKaxiras.pdf). Relevant material: `u2 = -u1` and mirror constraint, p. 4; Fourier basis and standard optimizer, p. 5.

[S4] Nguyen N. T. Nam and Mikito Koshino. *Lattice Relaxation and Energy Band Modulation in Twisted Bilayer Graphene*. Physical Review B 96, 075311 (2017). [DOI](https://doi.org/10.1103/PhysRevB.96.075311); [local PDF](file:///C:/Users/lh201/Desktop/Continuum_Limit_Relaxation/references/Nam2017.pdf). Relevant material: Fourier self-consistent iteration and representative choice, pp. 4 and 6; stability identified as future work, p. 7.

[S5] Michael Hott, Alexander B. Watson, and Mitchell Luskin. *Mathematical Foundations of Phonons in Incommensurate Materials*. arXiv:2409.06151 (2024). [arXiv](https://arxiv.org/abs/2409.06151); [local PDF](file:///C:/Users/lh201/Desktop/Continuum_Limit_Relaxation/references/hott2024mathematicalfoundationsphononsincommensurate.pdf). Relevant material: local uniqueness and stability near a reference solution under quantitative assumptions, pp. 27-34.

[S6] Christoph Ortner and Florian Theil. *Justification of the Cauchy-Born Approximation of Elastodynamics*. Archive for Rational Mechanics and Analysis 207 (2013), 1025-1073. [DOI](https://doi.org/10.1007/s00205-012-0592-6); [local PDF](file:///C:/Users/lh201/Desktop/Continuum_Limit_Relaxation/references/Ortner2013.pdf). Relevant material: coercive second variation, consistency, local atomistic solution, and quantitative inverse-function argument, especially pp. 22-27.

[S7] December 12, 2024 project manuscript. [Local PDF](file:///C:/Users/lh201/Desktop/Continuum_Limit_Relaxation/references/version_sent_to_Jeff_2024-12-12.pdf). Relevant material: explicit limitation concerning atomistic/continuum minimizer convergence, p. 2; numerical L-BFGS appendix, p. 13.

[C1] `atomistic_diagnostics`, current source lines 418-435 in the [experiment script](file:///C:/Users/lh201/Desktop/Continuum_Limit_Relaxation/Continuum_Limit_Code/experiments/relaxation/atomistic_two_chain_lj_lbfgs_convergence_study.py).

[C2] Endpoint solve and acceptance criteria, current source lines 438-490 in the same script.

[C3] Energy-equivalence and start-order selection, current source lines 550-556 in the same script.

[C4] Normalization and reflection-only per-start comparison, current source lines 557-575 in the same script.

[C5] Independent loop over mesh sizes, current source lines 799-810 in the same script.

[D1] Production [multistart data](file:///C:/Users/lh201/Desktop/Continuum_Limit_Relaxation/Continuum_Limit_Code/outputs/relaxation/output_atomistic_two_chain_lj_lbfgs_convergence_study/multistart_summary.csv) and [check report](file:///C:/Users/lh201/Desktop/Continuum_Limit_Relaxation/Continuum_Limit_Code/outputs/relaxation/output_atomistic_two_chain_lj_lbfgs_convergence_study/check_report.txt).

[D2] Targeted read-only numerical audit: exact finite-cutoff Hessian assembly, centered Hessian-vector check, continuum-orbit alignment, stability sampling, and seed-3 refinement rerun. [Audit code](file:///C:/Users/lh201/Desktop/Continuum_Limit_Relaxation/Continuum_Limit_Code/tmp/pdfs/branch_stability_audit.py).
