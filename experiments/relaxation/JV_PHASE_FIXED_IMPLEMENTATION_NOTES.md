# Phase-fixed Equation (6) implementation

## Mathematical target

`jv_phase_fixed_finite_difference_lbfgs_single_case.py` now discretizes
Equation (6) from `Hao_OralExam_Continuum_Limit.pdf`:

\[
J[v]=\int_{-\pi}^{\pi}\left(\frac12|v'|^2+W(v)\right)\,dx,
\qquad v\in x+H^1_{\mathrm{per}}.
\]

The paper defines `W = 2 Phi` and uses `Phi(s) = cos(s)` as its canonical
example. The code therefore uses `alpha = 1`, `W(s) = 2 cos(s)`, and
`W'(s) = -2 sin(s)`.

Theorem 1 shows that the unique minimizer normalized by `v(0) = 0` is odd.
The computation uses this result by writing `v = x + u` and parametrizing the
odd periodic correction as `u = [0, a, 0, -a[::-1]]`. Oddness is therefore an
analytical input to this computation, not an independently observed property.

## Changes

- Replaced the epsilon-scaled model by the constants in Equation (6).
- Added deterministic zero, sinusoidal, negative-sinusoidal, and random starts.
- Tightened L-BFGS-B's function-reduction setting and required every run to
  attain `pgtol = 5e-7`; a failed run now raises an error.
- Replaced the construction-only constraint report by the reduced-gradient
  norm, full discrete Euler-Lagrange residual, and minimum forward derivative
  of `v`.
- Distinguished the transformed energy `E[u]` from the Equation (6) energy,
  using the exact discrete identity `J[v] = E[u] + pi`.
- Added the `x = pi` endpoint to the plots using degree-one quasi-periodicity.

## Verification at N = 400

- All four starts stopped with `NORM OF PROJECTED GRADIENT <= PGTOL`.
- Their transformed energies agreed within `2.0721e-12`.
- The selected reduced-gradient infinity norm was `4.1793e-7`.
- The selected Euler-Lagrange residual infinity norm was `1.3303e-5`.
- The minimum forward derivative of `v` was `1.3283e-1`, hence positive.
- A centered directional check of the reduced gradient had relative error
  `2.8211e-10`.
- Direct evaluation of `J[v]` agreed with `E[u] + pi` to `1.1102e-15`.

These checks support the computed discrete minimizer but do not prove global
minimality. Mesh refinement, an unrestricted nonsymmetric calculation with
phase alignment, and a boundary-value reference solve remain separate possible
validation studies.

## Run command

```powershell
C:\Users\lh201\anaconda3\envs\relax\python.exe -m experiments.relaxation.jv_phase_fixed_finite_difference_lbfgs_single_case
```
