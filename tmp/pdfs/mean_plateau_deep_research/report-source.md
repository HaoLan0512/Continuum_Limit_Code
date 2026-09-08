---
title: "Why an O(h) Mean Error Can Produce a Finite-Grid Plateau"
subtitle: "Seed-3 phase quantization in the two-chain atomistic convergence study"
author: "Continuum Limit Relaxation research audit"
date: "September 7, 2026"
subject: "Appendix B normalization, convergence envelopes, and the seed-3 sawtooth"
keywords: [atomistic model, continuum limit, symmetry normalization, phase, convergence]
---

**Audience:** Continuum Limit Relaxation research project

**Question:** Why is the seed-3 mean contribution almost constant from $N=80$ to
$N=320$ even though Appendix B proves that it is $O(h)$?

## Executive answer

There is no contradiction. The estimate $M_N=O(h_N)$ is an upper-envelope
statement, not a pointwise halving law. Appendix B leaves a dimensionless
nearest-integer remainder $q_N$ that may vary between $-1/2$ and $1/2$:

$$
\overline u_{1,N}=\frac{h_N}{2}q_N,
\qquad
M_N:=2\sqrt{\pi}|\overline u_{1,N}|
=\sqrt{\pi}h_N|q_N|.
$$

The bound $|q_N|\le 1/2$ proves $M_N\le(\sqrt{\pi}/2)h_N$, but it does not say
that $q_N$ is constant or convergent. In the saved seed-3 data, $h_N$ nearly
halves while $|q_N|$ nearly doubles over $N=80,160,320$. Those two effects
cancel, creating the plateau. At $N=640$, the nearest-integer remainder wraps
across the edge of its normalization cell, $q_N$ drops sharply, and the error
falls again.

## 1. Exact Appendix B formula

Let $a_N$ denote the unnormalized phase quantity

$$
a_N:=\frac{h_2\overline u_2-h_1\overline u_1}{h_1h_2}.
$$

Appendix B chooses $d_N\in\mathbb Z$ to be the nearest integer to $a_N$ and
then uses the continuous common translation $c_N$ to enforce
$\overline{\widetilde u}_1+\overline{\widetilde u}_2=0$. The normalized first
mean is exactly

$$
\overline{\widetilde u}_{1,N}
=\frac{h_1h_2}{h_1+h_2}(d_N-a_N)
=\frac{h_N}{2}q_N,
\qquad q_N:=d_N-a_N.
$$

Nearest-integer rounding gives $|q_N|\le 1/2$, hence

$$
|\overline{\widetilde u}_{1,N}|\le \frac{h_N}{4}.
$$

For the two-layer constant error $(m,-m)$, the derivative terms vanish and

$$
\|(m,-m)\|_{H_h^2}=2\sqrt{\pi}|m|.
$$

Therefore the plotted mean mode is

$$
M_N=\sqrt{\pi}h_N|q_N|
\le \frac{\sqrt{\pi}}{2}h_N.
$$

This proves an $O(h)$ envelope but imposes no monotonicity on $M_N$.

## 2. What happens in the saved seed-3 sequence

The current output gives the following exact reconstruction. The mean-free
column is computed from the orthogonal decomposition
$E_N^2=M_N^2+(E_N^{\rm mf})^2$.

| $N$ | $d_N$ | $h_N$ | $q_N=2\overline u_1/h_N$ | $M_N$ | full $H_h^2$ error | mean-free error |
|---:|---:|---:|---:|---:|---:|---:|
| 20  | 0  | 0.306497 |  0.282933 | 0.153704 | 0.159089 | 0.041041 |
| 40  | -1 | 0.155140 | -0.439450 | 0.120840 | 0.124476 | 0.029867 |
| 80  | -1 | 0.078052 |  0.113199 | 0.015660 | 0.016874 | 0.006283 |
| 160 | -2 | 0.039148 |  0.216269 | 0.015006 | 0.015619 | 0.004331 |
| 320 | -4 | 0.019604 |  0.422538 | 0.014682 | 0.015129 | 0.003647 |
| 640 | -9 | 0.009810 | -0.165304 | 0.002874 | 0.003020 | 0.000928 |

From $N=80$ to $160$,

$$
\frac{h_{160}}{h_{80}}=0.5016,
\qquad
\frac{|q_{160}|}{|q_{80}|}=1.9105,
\qquad
\frac{M_{160}}{M_{80}}=0.9582.
$$

From $N=160$ to $320$,

$$
\frac{h_{320}}{h_{160}}=0.5008,
\qquad
\frac{|q_{320}|}{|q_{160}|}=1.9538,
\qquad
\frac{M_{320}}{M_{160}}=0.9784.
$$

Thus the plateau is genuinely almost flat: its local mean-mode slopes are about
$-0.062$ and $-0.032$. It is not a plotting illusion.

## 3. Why the remainder nearly doubles

The reconstructed unrounded phase satisfies

$$
\frac{a_N}{N}
=-0.014147,-0.014014,-0.013915,-0.013852,-0.013820,-0.013804
$$

for $N=20,40,80,160,320,640$. Hence $a_N$ is approximately proportional to
$N$: the same physical phase displacement is being expressed in increasingly
fine lattice-index units.

On a dyadic refinement this gives $a_{2N}\approx2a_N$. Since
$q_N=\operatorname{round}(a_N)-a_N$, the remainder follows the sawtooth or
doubling map

$$
q_{2N}\approx2q_N-\operatorname{round}(2q_N).
$$

For $N=80,160,320,640$, the observed progression is

$$
0.1132\ \longrightarrow\ 0.2163\ \longrightarrow\ 0.4225
\ \longrightarrow\ -0.1653.
$$

Before the wrap, $|q_N|$ nearly doubles. Because $h_N$ nearly halves,

$$
M_{2N}\approx\sqrt{\pi}\frac{h_N}{2}(2|q_N|)\approx M_N.
$$

Once $2q_N$ crosses the nearest-integer cell boundary, the rounding subtracts
an integer and resets the remainder. This produces the sharp drop at $N=640$.
The observed next-grid remainders differ from this simple doubling prediction by
only about $0.005$ to $0.010$.

## 4. Why this remains O(h)

Big-O means that there are constants $C$ and $N_0$ such that

$$
M_N\le Ch_N\qquad (N\ge N_0).
$$

It does not require $M_{2N}=M_N/2$, a local log-log slope of $-1$, or monotone
decay. Here

$$
\frac{M_N}{h_N}=\sqrt{\pi}|q_N|\le\frac{\sqrt{\pi}}{2}
$$

uniformly. A nonzero absolute plateau cannot persist forever, because doing so
would force $M_N/h_N$ to exceed this bound as $h_N\to0$. It can persist for a
few refinements while $|q_N|$ grows, then it must drop. Similar, smaller
sawteeth may recur at later grids.

Arithmetic sensitivity of convergence rates is also natural in broader
incommensurate bilayer limits: Hott, Watson, and Luskin obtain a quantitative
rate under a Diophantine condition on the twist [S2]. This does not prove the
present rounding formula; the exact explanation here comes from Appendix B and
the saved normalization data.

## 5. Interpretation and next study

The mean explains most, but not all, of the gap between seed 3 and the warm
track. After removing the exact mean component, the seed-3 mean-free error is
still larger than the warm error at every saved grid. Its six-grid fitted slope
is about $-1.056$, so it is broadly first order but not perfectly straight.
This mean-free solution error must not be called the consistency residual:
$r_h=\Delta_hu_c-F_h(u_c)$ is a different quantity and is not currently saved.

A focused follow-up should retain the full Theorem 5 error and add:

1. $q_N=2\overline u_1/h_N$ and the selected integer $d_N$;
2. the exact mean mode $M_N$;
3. the mean-free $H_h^2$ error
   $E_N^{\rm mf}=\sqrt{E_N^2-M_N^2}$;
4. the actual consistency residual $\|\Delta_hu_c-F_h(u_c)\|$.

This separates phase quantization, profile-shape error, and consistency. It also
avoids an impermissible continuous shift of the atomistic endpoint.

## Sources and evidence ledger

**[S1]** *Continuum Limit of a One-Dimensional Atomistic Model for Moire
Relaxation*, July 21, 2026, local manuscript. Used for Theorem 3, Theorem 4,
Theorem 5, and Appendix B. Accessed from
`Hao_OralExam_Continuum_Limit.pdf` in the project repository.

**[D1]** Current atomistic convergence-study implementation and saved
`convergence_summary.csv`, September 7, 2026. Used for $d_N$, normalized
means, errors, and the verified sawtooth reconstruction.

**[S2]** Michael Hott, Alexander B. Watson, and Mitchell Luskin,
[*From Incommensurate Bilayer Heterostructures to Allen-Cahn: An Exact
Thermodynamic Limit*](https://arxiv.org/abs/2305.18186), arXiv:2305.18186,
May 29, 2023. Used only as broader context for arithmetic conditions in
quantitative incommensurate convergence rates.

**Limitations.** The six saved resolutions verify the mechanism for the current
seed-3 track, but do not prove that the same sawtooth timing persists for all
$N$. The optimizer endpoints are accepted stationary candidates, not certified
global minimizers, and the manuscript's uniform stability gap is not checked
numerically.
