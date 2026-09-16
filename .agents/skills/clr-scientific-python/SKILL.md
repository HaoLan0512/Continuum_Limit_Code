---
name: clr-scientific-python
description: Implement or review computational Python changes in CLR, with explicit mathematics and proportionate numerical validation.
---

# CLR scientific Python

Use this workflow for computational changes and their review. Prose-only edits,
ordinary conceptual explanations, and purely structural moves do not need it.
Follow [repository authorization and invariants](../../../AGENTS.md); selecting
this skill adds no approval requirement and grants no extra authority.

## Establish the calculation

Inspect the affected objective, derivatives, coordinate maps, callers, and I/O.
Read relevant paper/code notes when equations or conventions matter. Establish
the intended mathematical change or behavior to preserve before implementation.
For a review-only request, report findings without editing.

For numerical execution, verify `sys.executable` and required dependencies using
the documented project interpreter. Check entry points before importing; some
experiments perform calculations or write outputs at import time.

## Keep the mathematics visible

- Modify the narrowest necessary code. Avoid speculative modes, frameworks,
  fallback paths, generic validation, or refactors unrelated to the request.
- For a new or substantially revised single-case script, prefer the readable
  flow in [the phase-fixed example](../../../experiments/relaxation/jv_phase_fixed_finite_difference_lbfgs_single_case.py):
  constants, model functions, coordinate maps, objective/gradient, initial
  guesses and solve, diagnostics, then guarded output. This is a readability
  example, not a template for algorithms, constants, or tolerances.
- Introduce a helper for a genuine mathematical or workflow concept. Keep the
  flow shallow; simple arrays, tuples, and dictionaries usually suffice.
- For `F(a) = E(build_u(a))`, show reconstruction, full objective/gradient, and
  chain-rule reduction as separate named operations. Document array lengths,
  grid coordinates, fixed values, and the exact constraint.
- Group scientific constants and named tolerances near the top. Explain what
  each tolerance measures and verify its scale for the current stopping rule.
- Give new nontrivial functions concise docstrings covering role, inputs, and
  return values. Explain conventions and reasons in comments.
- Seed new random comparisons when practical and name cases clearly. Preserve
  existing seeds and starts unless their change is approved.
- Keep new imports side-effect free. Put calculation entry points, file writes,
  plotting, and printing under `if __name__ == "__main__":` or the approved
  explicit entry point.

## Validate the affected claim

| Change or claim | Relevant evidence |
|---|---|
| Objective, derivative, or coordinate map | Deterministic formula/construction checks; a suitable directional derivative or chain-rule check |
| Optimizer result | Solver status plus finite relevant full/reduced/projected gradients; scaled Euler-Lagrange and constraint residuals needed by the claim |
| Behavior-preserving computational edit | Before/after scalars and arrays under matched inputs, interpreter, dependencies, seeds, and conditions |
| Convergence or stability claim | Claim-specific mesh, cutoff, alignment, or stability evidence; state the experiment's assumptions and limitations |

Choose the smallest meaningful case. Syntax checks may target changed files
with `compileall` or compile source in memory; avoid a repository-wide sweep for
a local change. Do not add an unrelated test framework.

For a behavior-preserving change, capture a low-cost baseline when feasible and
justify absolute/relative tolerances before comparing results. Do not copy a
historical tolerance without checking scale. If behavior intentionally changes,
validate the new mathematical target rather than asserting old-output parity.
Successful process exits alone do not establish numerical equivalence.

Use existing parameter hooks, fixtures, or isolated temporary copies for small
tests. Do not reduce production parameters just to speed verification. Keep
baseline outputs intact, and do not run a full study merely by importing it.
Honor the repository's compute-approval boundary.

One-time formula checks and imposed symmetry checks do not certify each optimizer
endpoint. Distinguish function-reduction stagnation from stationarity and observed
agreement from proof. State separately whether an experiment tests the intended
hypothesis and whether its results support that hypothesis.

Investigate discrepancies and warnings using the repository verification rules.
After the relevant checks pass, review the diff and report measured diagnostics,
tolerances, remaining uncertainty, and any unrun expensive validation.
