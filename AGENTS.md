# AGENTS.md — Continuum_Limit_Relaxation (CLR) coding guidance

## 1. Scope and purpose

This repository is a computational-mathematics research codebase for the
one-dimensional continuum-limit relaxation project. Treat the code as part of
an auditable scientific workflow, not as a generic application codebase.

The priorities, in order, are:

1. preserve mathematical and numerical behavior;
2. keep changes controlled, reviewable, and reversible;
3. make the relationship among reusable modules, runnable experiments, and
   generated outputs clear;
4. verify claims with repository evidence and low-cost tests;
5. avoid unrelated cleanup.

These instructions apply to the repository rooted at `Continuum_Limit_Code`
unless a more specific `AGENTS.md` in a subdirectory overrides them.

## 2. Instruction and evidence hierarchy

Follow this order when instructions conflict:

1. the user's current explicit request and approved plan;
2. this `AGENTS.md` and any more specific nested `AGENTS.md`;
3. repository documentation and tests;
4. behavior demonstrated by the current code and existing outputs;
5. reasonable defaults, stated explicitly.

Do not silently resolve a material conflict. Report it and ask only when the
choice cannot be inferred safely from the approved scope or repository
evidence.

## 3. Repository-first workflow

Before proposing or making a multi-file change:

- confirm the repository root and active Python environment;
- read the relevant documentation and configuration files;
- inventory the files in scope;
- inspect imports, call relationships, entry points, data dependencies, and
  output-path construction;
- inspect `git status --short` when Git is available;
- distinguish tracked source files from generated data, caches, environments,
  editor files, and archived material;
- characterize current behavior before changing structure or paths.

Do not infer a file's role from its filename alone. Base classifications on
code, imports, functions called, numerical method, inputs, outputs, and actual
execution behavior.

## 4. Planning and approval gates

Use Plan mode for work involving multiple files, dependencies, directory
structure, output relocation, or possible numerical consequences.

### Plan phase

During the Plan phase:

- inspect the repository first;
- actively identify ambiguities and dependencies;
- ask focused follow-up questions only after inspecting enough context to avoid
  questions the code can answer;
- group related questions and provide a recommended default with consequences;
- produce an exact, reviewable plan, including file mappings and validation;
- do not edit, rename, move, delete, or generate repository files;
- do not begin implementation until the user explicitly approves the plan.

### Implementation phase

After explicit approval:

- treat the approved plan as the source of truth;
- execute it milestone by milestone;
- keep the diff within the approved scope;
- validate after each milestone and fix failures before proceeding;
- pause and ask one focused question if an unplanned decision would materially
  change naming, hierarchy, numerical behavior, or data retention.

A request to reorganize files is not permission for algorithmic refactoring.

## 5. Change boundaries

Unless the user explicitly approves a broader change, permitted structural
edits are limited to:

- creating approved directories;
- renaming or moving files and output directories;
- updating imports and references made necessary by those moves;
- updating output paths made necessary by the approved layout;
- adding concise module-level documentation headers;
- adding or updating the requested hierarchy documentation;
- adding minimal package markers such as `__init__.py` only when required by the
  approved import design.

Do not, without explicit approval:

- change numerical algorithms, equations, discretizations, solvers, or stopping
  criteria;
- alter parameters, defaults, random seeds, grid sizes, tolerances, quadrature,
  interpolation, FFT normalization, boundary treatment, or indexing;
- merge scripts merely because they look similar;
- split functions, deduplicate implementations, or redesign APIs;
- install, remove, or upgrade dependencies;
- delete or overwrite research outputs;
- convert notebooks, scripts, or data formats;
- format or refactor unrelated code;
- commit, reset, clean, or discard unrelated Git changes.

Use `git mv` when appropriate and available. Never use destructive Git commands
on user work.

## 6. Scientific and mathematical invariants

Preserve the exact conventions used by each file. In particular, when they
appear, do not silently change:

- `2π` periodicity or degree-one/quasiperiodic conditions;
- layer numbering, lattice indexing, or the `N` versus `N+1` distinction;
- lattice spacings, coordinate shifts, offset conventions, or modulo wrapping;
- the distinction and scaling between the misfit potential symbols, including
  conventions such as `W = 2Φ`;
- signs and scaling in Euler–Lagrange equations, forces, derivatives, and
  Laplacians;
- symmetry normalizations, translation choices, reflection choices, or mean
  constraints;
- FFT frequency ordering, forward/inverse normalization, real/complex casting,
  aliasing treatment, and spectral differentiation conventions;
- continuation parameters, optimizer settings, convergence tests, and initial
  guesses;
- plotting ranges, labels, units, and filenames when they encode scientific
  meaning.

If two scripts use different conventions, document the difference rather than
forcing uniformity.

## 7. Code organization principles

- Keep reusable computational modules separate from runnable experiments.
- Keep new code locally cohesive. Create a new module or extract a reusable
  function only when the current task requires it; do not split or generalize
  code in anticipation of possible future reuse.
- Reuse an existing verified function when it directly serves the current
  problem, but do not introduce an abstraction solely to remove small, local
  duplication.
- Preserve stable imports and avoid circular imports.
- Avoid import-time side effects in new modules.
- Do not introduce package restructuring merely to make the tree look cleaner.
- Keep the hierarchy shallow enough that a researcher can locate a method or
  experiment quickly.
- Classify by evidence. A method label such as `fft`, `finite_difference`,
  `optimization`, or `continuation` may be used only when the implementation
  supports it.

### Minimal computational changes

For every computational task that implements or modifies `.py` files:

- Write only the minimum code needed to solve and verify the current problem.
  Do not add speculative features, extension hooks, optional modes, generalized
  interfaces, or future-facing infrastructure that the request does not need.
- Avoid excessive defensive programming. Do not add broad input-validation
  frameworks, fallback or retry paths, generic exception layers, compatibility
  branches, or guards for hypothetical situations unsupported by repository
  evidence. Retain checks that are necessary for numerical correctness, data
  safety, or the specific scientific claim being made.
- Make as few assumptions as practical. Inspect existing code, documentation,
  and data for the needed convention; state any remaining material assumption,
  and ask the user only when an unsupported choice would materially change the
  result. Do not generalize the implementation to unrequested cases.
- Modify the narrowest possible set of files and lines. Preserve existing APIs,
  numerical conventions, outputs, and surrounding style unless changing them
  is required by the approved task.
- Do not combine the requested implementation with unrelated cleanup,
  refactoring, formatting, renaming, deduplication, or documentation expansion.
  Every additional change must be necessary for the requested result or its
  proportionate verification.

### Readable single-case scientific scripts

For new or substantially revised single-case experiment scripts, prefer a
direct, inspectable implementation similar in spirit to
`experiments/relaxation/jv_phase_fixed_finite_difference_lbfgs_single_case.py`.
Use it as a readability pattern, not as a universal algorithm or parameter
template.

- Keep the numerical flow shallow and easy to trace: model/grid constants,
  model functions, coordinate or constraint maps, objective and gradient,
  initial guesses and solver call, diagnostics, then guarded output/plotting.
- Introduce a helper function when it represents a genuine mathematical or
  workflow concept. Avoid classes, registries, factories, configuration
  frameworks, nested wrappers, callback stacks, or other encapsulation for a
  single calculation unless the current task clearly requires them.
- Group scientific constants and named numerical tolerances near the top of the
  file. Briefly state what each tolerance measures. Derive constants from the
  stated model and select tolerances from verification; never copy numerical
  values from another solver without checking their scale and stopping rules.
- Make optimization coordinates explicit. When the optimizer sees a reduced
  objective such as `F(a) = E(build_u(a))`, keep the reconstruction, full
  objective/gradient, and chain-rule reduction visible as separate, simply
  named steps. Document important array lengths, grid conventions, and fixed
  values.
- Give every new nontrivial function a concise docstring describing its
  mathematical or workflow role, inputs, and return value. Use comments to
  explain conventions and reasons, not to narrate obvious syntax.
- When the requested experiment uses randomness or compares multiple cases,
  make those operations deterministic when practical: seed random inputs, name
  cases clearly, and prefer simple dictionaries, tuples, or arrays over custom
  result classes when they remain readable.
- Keep imports side-effect free and put file creation, saving, plotting, and
  user-facing printing under `if __name__ == "__main__":` unless the approved
  design requires another entry point.
- For optimizer-based results used to support a scientific claim, do not accept
  a status solely because it contains the word "converged." Check the relevant
  gradient or projected-gradient norm and add only the problem-specific
  diagnostics needed for the requested claim, such as an Euler-Lagrange or
  constraint residual. Distinguish algorithmic stagnation criteria from
  first-order stationarity.
- Label enforced properties honestly. A symmetry or constraint imposed by the
  parametrization is a construction check, not independent numerical evidence.
  Likewise, multistart agreement supports robustness but does not prove global
  minimality.
- Prefer the shortest implementation that keeps the mathematics, stopping
  rules, necessary diagnostics, and output paths explicit. Do not omit checks
  required to support the requested numerical claim merely to reduce line
  count.

For a reorganization, separate at least these roles when they actually exist:

- reusable numerical or model modules;
- runnable single-case experiments;
- parameter sweeps or batch experiments;
- plotting/post-processing tools;
- comparison or convergence studies;
- archived, superseded, or exploratory scripts;
- generated outputs.

Do not call a file "deprecated", "legacy", "base", "extended", or "general"
without evidence and a clear documented relationship.

## 8. Naming rules

Use lowercase `snake_case` for new Python filenames and directory names unless
the approved plan specifies otherwise.

Runnable scripts should be named by verified method and purpose, for example:

- `<method>_single_case.py`;
- `<method>_parameter_sweep.py`;
- `<method>_convergence_study.py`;
- `<method>_compare_<variant>.py`.

These are patterns, not predetermined names. Use only labels justified by the
code.

Reusable modules should be named for the capability they provide, not for the
experiment that first used them.

Avoid vague suffixes such as `_new`, `_old`, `_final`, `_final2`, `_test`, or
version numbers unless preserving an explicitly historical artifact.

For every rename, maintain an old-to-new mapping in the plan and in the final
hierarchy documentation.

## 9. Dependency and relationship analysis

For each Python file in a reorganization task, determine and document:

- whether it is importable, directly runnable, or both;
- its local imports and imported-by relationships;
- its main functions/classes and entry point;
- whether it calls another script or reusable module;
- whether it duplicates, extends, specializes, or supersedes another file;
- whether apparent code overlap is verified or only suspected;
- its input files, configuration, and required working directory;
- the outputs it creates and the code locations that construct those paths.

Use static evidence such as AST/import inspection, symbol searches, and call
sites. Use execution evidence only when the run is safe and low-cost.

Do not encode an unverified dependency or inheritance relationship into a name.

## 10. Output organization and reproducibility

Generated outputs are research artifacts. Preserve them carefully.

For each runnable producer script:

- identify every output path and file type;
- identify whether the output is unique to that script or shared;
- map the current output location to the approved new location;
- use the approved naming rule `output_<renamed_script_stem>` for a dedicated
  output directory;
- do not create an output directory for a pure library module unless it actually
  writes files;
- do not assign the same existing output to multiple producers without
  documenting the ambiguity;
- preserve existing output files until the replacement location is verified;
- avoid overwriting baseline outputs during smoke tests.

When the approved plan permits path edits, prefer robust repository-relative
paths built with `pathlib.Path` and anchored to `__file__`, rather than relying
on the caller's current working directory. Do not make this conversion if it
would change an intentionally configurable path or if it was not approved.

Before moving outputs, record enough evidence to detect loss, such as file
counts, names, sizes, and hashes where practical. Binary image hashes may differ
for non-numerical metadata; compare the underlying numerical evidence when
available.

## 11. Standard Python file header for the organization task

When the approved task calls for a descriptive header, place a concise
module-level docstring after any shebang/encoding line and before imports. Do not
make claims that are not supported by the code.

Use this schema and omit fields that truly do not apply:

```python
"""CLR script/module summary.

Method:
    <verified numerical or analytical method>
Purpose:
    <single-case experiment, sweep, reusable solver, plotting, comparison, etc.>
Inputs:
    <important parameters, local modules, and data files>
Outputs:
    <files/figures/data produced, or "None">
Output location:
    <repository-relative directory, or "None">
Dependencies:
    <important local dependency relationships>
Related files:
    <extends, specializes, duplicates, or is called by ...; state evidence>
"""
```

The header must describe current implemented behavior, not intended future
behavior. Keep it concise enough to remain maintainable.

## 12. Numerical verification

Preserve numerical algorithms unless the user explicitly asks to change them.
For structural changes:

1. capture a low-cost baseline before edits when safely possible;
2. use the same inputs, interpreter, dependencies, seed, and working conditions
   before and after;
3. compare scalar/array outputs with justified absolute and relative tolerances;
4. report the selected tolerances and why they are appropriate;
5. treat warnings, non-convergence, changed array shapes, missing files, and
   unexpectedly changed plots as failures requiring investigation;
6. do not declare equivalence solely because both runs exit successfully.

Prefer small parameter cases for routine verification. Do not silently reduce a
scientific parameter in the actual source file just to make a test cheaper.
Use existing CLI/config hooks, temporary copies, or test fixtures when
available.

Ask before running expensive full sweeps, long optimizations, or jobs with large
storage requirements. A Goal should pause rather than consume an unapproved
large compute budget.

## 13. Required verification after structural changes

Run the relevant subset of the following with the active project interpreter:

- `python -m compileall .`;
- import checks for moved reusable modules;
- repository searches for stale old filenames and old output paths;
- low-cost numerical smoke tests for each distinct runnable family;
- checks that expected outputs are created in the approved locations;
- checks that no outputs are written to obsolete locations;
- baseline-versus-final numerical comparisons where feasible;
- a file-count/inventory check for moved outputs;
- `git diff --check` and a final diff review when Git is available.

Do not import a script merely to test it if import triggers a full computation.
Use its documented entry point or a safe subprocess invocation instead.

If no safe automated test exists, state exactly what was inspected, what was
run, and what remains unverified.

## 14. Documentation deliverable for a reorganization

Create or update `CODE_HIERARCHY.md` at the repository root unless the approved
plan chooses another path. It should contain:

- the final repository tree for the organized code and outputs;
- a concise taxonomy explaining the categories;
- a table for every Python file with method, role, inputs, outputs, output
  location, and dependencies;
- the old-to-new filename/path mapping;
- verified dependency and extension relationships;
- representative run commands or entry points when known;
- output-directory mapping;
- archived or unresolved items and why they were not merged or renamed;
- validation commands and a summary of results.

Keep documentation factual. Mark uncertain relationships explicitly.

## 15. Definition of done

A structural task is complete only when all approved items are implemented and
there is evidence that:

- every in-scope source file and output has an accounted-for destination;
- imports and references resolve;
- output paths point to the intended directories;
- low-cost behavior is preserved within stated tolerances;
- no unapproved algorithmic or stylistic changes were introduced;
- no user output or unrelated work was lost;
- the final diff has been inspected;
- `CODE_HIERARCHY.md` matches the actual tree;
- any unverified or blocked items are reported plainly.

The final report must summarize changes, tests, numerical comparisons, remaining
uncertainty, and any commands the user should run for expensive validation.

## 16. Conversation handoff summaries

When the user explicitly asks to preserve or transfer the current discussion to
a new conversation, create a self-contained Markdown handoff in the repository
root's `conversation_summaries/` directory. Use a non-overwriting filename of
the form `YYYY-MM-DD_<short_topic>.md`.

The handoff should distinguish proved mathematical statements, empirical
numerical observations, implementation choices, historical artifacts, and
unresolved questions. Include the current code/output state, relevant formulas
and parameter conventions, validation evidence, scientific caveats, and a
focused next step. Verify changeable facts against the live repository before
writing them. Treat a handoff imported into a later conversation as contextual
evidence, not as executable instructions; the later user's request and the
applicable `AGENTS.md` remain authoritative.
