---
name: clr-repository-reorganization
description: Plan, implement, or review CLR source and output reorganization while preserving numerical behavior and artifact provenance.
---

# CLR repository reorganization

Use for source/output moves, renames, and repository restructuring. It does not
cover ordinary code fixes or authorize scientific model changes. Follow
[repository authorization and invariants](../../../AGENTS.md); an approved
reorganization plan remains the source of truth. For review-only work, inspect
and report without modifying files.

## Establish the mapping before edits

Confirm the repository root, relevant interpreter, and Git state. Inventory the
in-scope sources and artifacts, distinguishing generated data, caches,
environments, editor files, notebooks, and archives. Use
[CODE_HIERARCHY.md](../../../CODE_HIERARCHY.md) as a map and verify live paths.

For each affected Python file, establish from code and call sites:

- importability, direct entry point, principal functions/classes, and callers;
- local imports, imported-by relationships, and calls to other scripts/modules;
- model and method, inputs/configuration, required working directory, and every
  output path, type, and path-construction location;
- verified duplication, specialization, or extension, with uncertainty explicit.

Characterize behavior and capture low-cost baselines when feasible. Present an
exact old-to-new source/output mapping, import/path edits, data-preservation
measures, and validation before requesting plan approval. Complete read-only
preparation first. Do not make the governed changes before approval.

## Execute the approved structure

Keep reusable modules separate from runnable experiments. Distinguish single
cases, sweeps, postprocessors, comparisons/convergence studies, exploratory
material, and archives when those roles actually exist. Keep the hierarchy
shallow, preserve stable imports, and avoid circular dependencies.

Use `git mv` where appropriate. Limit edits to approved moves/directories,
necessary imports/references/output paths, requested headers and hierarchy
documentation, and package markers required by the approved import design.
Do not merge, split, deduplicate, redesign APIs, convert formats, or change
algorithms merely to make the structure cleaner.

Use lowercase `snake_case` names for Python files and code directories. Name
runners by verified method and purpose, such as `<method>_single_case.py`,
`<method>_parameter_sweep.py`, `<method>_convergence_study.py`, or
`<method>_compare_<variant>.py`. Name modules for their capability. Avoid vague
version suffixes and unsupported labels such as "deprecated", "legacy", "base",
"extended", or "general"; preserve explicitly historical names where appropriate.

## Preserve research outputs

For every affected producer, identify ownership of all output paths, including
shared or ambiguous artifacts. Use `output_<renamed_script_stem>` for an approved
dedicated output directory; do not give a pure library an output directory.

Before moving artifacts, record names, counts, sizes, and hashes where practical.
Preserve recoverable originals until the destination is verified. Do not assign
one artifact to multiple producers without documenting the ambiguity.

When path edits are approved, prefer repository-relative `pathlib.Path` paths
anchored to `__file__`; retain intentional configurable-path semantics.
Use isolated destinations for smoke tests. For moves, compare exact file hashes;
for regenerated figures, metadata can affect hashes, so also inspect underlying
numerical values and scientifically meaningful visual differences.

## Headers and hierarchy documentation

Only when the approved task requests headers, use a concise module docstring
after any shebang/encoding line and before imports. Retain these field names
when applicable; omit fields that do not apply:

```python
"""CLR script/module summary.

Method:
    Verified numerical or analytical method.
Purpose:
    Module or experiment role.
Inputs:
    Parameters, modules, and data files.
Outputs:
    Produced files, figures, or data, or None.
Output location:
    Repository-relative directory, or None.
Dependencies:
    Important local relationships.
Related files:
    Verified extension, specialization, duplication, or caller relationships.
"""
```

Update `CODE_HIERARCHY.md` unless the approved plan selects another path.
Include the final tree/taxonomy, the in-scope Python inventory with method, role,
inputs, outputs/location, and dependencies, source/output old-to-new mappings,
verified relationships, entry points, unresolved/archived items, and validation
results. For a full reorganization, account for every Python file. Mark historical
validation as historical and uncertain provenance as unresolved.

## Verify and finish

Select checks that cover the affected families:

- compile affected Python files and check imports only for import-safe modules;
- search for stale operational imports, names, and paths; former names may remain
  in clearly marked migration records;
- compare low-cost before/after scalars and arrays under the same interpreter,
  dependencies, inputs, seeds, and conditions, using justified tolerances;
- exercise relevant producers in isolation and verify expected outputs appear
  only at the intended destinations;
- compare moved-file inventories and hashes, and inspect relevant plots when
  numerical meaning could change.

A broad source move can justify broad compilation; a documentation-only
reorganization does not justify numerical execution. Never import an experiment
that launches a full calculation as a smoke test. Report unavailable baselines
or unsafe tests explicitly.

Apply the repository's warning and failure rules, fix failures caused by the
change, and repeat affected checks. Completion requires all approved mappings
accounted for, imports and paths resolving, required comparisons passing, no
lost user artifacts, accurate hierarchy documentation, and final diff review.
Report unresolved evidence and commands for separately approved expensive checks.
