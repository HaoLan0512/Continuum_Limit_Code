# CLR agent guidance audit

Date: 2026-09-13.

The user approved a six-file instruction/documentation change, including a
narrower approval rule. This audit is a record of that change, not an additional
instruction layer. Scientific programs, notebooks, outputs, dependencies, and
model/runtime settings are outside the edit scope.

## Evidence and findings

The initial inventory found one repository `AGENTS.md` (437 lines, 2,763 words,
19,862 on-disk bytes), no `skills.md`, no local `SKILL.md` files, and no dedicated
CI workflow or test-runner configuration. Existing workflow information lives
in `CODE_HIERARCHY.md`, scientific implementation notes, and code entry points.
Historical handoffs and generated reports remain scientific context.

The main issues were repeated inspection/change/validation requirements,
reorganization-specific procedures loaded for all tasks, approval tied to file
count, and historical validation presented without an explicit current-state
qualification. The exact scientific invariants and minimal computational style
remain useful and have been retained.

The shell's bare Python resolved to
`C:\Users\lh201\anaconda3\python.exe` (3.13.9). The documented project interpreter,
`C:\Users\lh201\anaconda3\envs\relax\python.exe`, was checked during the audit:
Python 3.12.12, NumPy 2.4.2, SciPy 1.17.1, and Matplotlib 3.10.8.
These are observations on the audit date, not pinned dependency requirements.

## Official sources reviewed

Both requested pages were read, along with the current Codex discovery
documentation. Sources were accessed on 2026-09-13.

- [Rethinking skills and prompts for GPT-6 Astra](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra),
  published 2026-09-11: use narrow skill descriptions, load context by task, and
  reconsider procedural instructions and premature stopping rules.
- [GPT-6 Astra model guidance](https://developers.openai.com/api/docs/guides/latest-model):
  make authorization and completion clear, explain instruction-driven pauses,
  calibrate writing and verification, and fit delegation to the actual harness.
- [Custom instructions with AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md):
  repository instructions participate in an instruction chain; document discovery
  and fallback filenames are separate from skill discovery.
- [Build skills](https://learn.chatgpt.com/docs/build-skills):
  native repository skills live under `.agents/skills/<name>/SKILL.md`, with
  `name` and `description` metadata and focused, selectively loaded instructions.

The concrete workflow design below is a repository-specific application of that
guidance. The installed OpenAI Docs and Skill Creator skills also informed source
selection and skill authoring. No external prompt examples were copied wholesale.

## Approved file layout

| File | Role |
|---|---|
| [AGENTS.md](../AGENTS.md) | Enduring scientific requirements, authorization, contextual links, completion, and handoffs |
| [skills.md](../skills.md) | Human-readable inventory; not an automatic skill entry point |
| [Scientific Python SKILL.md](../.agents/skills/clr-scientific-python/SKILL.md) | Computational implementation/review workflow and numerical validation |
| [Reorganization SKILL.md](../.agents/skills/clr-repository-reorganization/SKILL.md) | Source/output mapping, preservation, naming, headers, and structural verification |
| [CODE_HIERARCHY.md](../CODE_HIERARCHY.md) | Instruction layout and explicitly historical numerical validation records |
| This audit | Sources, section mapping, intentional changes, and verification evidence |

Each skill is instruction-only and contains one workflow. No helper scripts,
additional routing files, plugin metadata, or new CI system are needed for this
scope. Skills retain default implicit selection; neither forces the other to
load. Native skill identifiers use hyphens; scientific code naming stays
`snake_case`.

## Mapping of the previous AGENTS.md

Section numbers refer to the pre-edit file. Repeated requirements have one
principal home; common authorization and scientific invariants remain in the
root file so they apply even when no skill is selected.

| Previous section | Destination and treatment |
|---|---|
| 1. Scope and purpose | Root purpose and change boundaries |
| 2. Instruction and evidence hierarchy | Root scope; distinguish authoritative instructions from contextual evidence |
| 3. Repository-first workflow | Root contextual reading; affected-code inspection in the scientific skill and full mapping in the reorganization skill |
| 4. Planning and approval gates | Root scope; replace file-count gate with the approved consequence-based boundary |
| 5. Change boundaries | Root boundaries; detailed permitted structural edits in the reorganization skill |
| 6. Scientific and mathematical invariants | Root scientific invariants, including plotting and output conventions |
| 7. Code organization principles | Root minimality/import rules; single-case style and explicit mathematics in the scientific skill; role taxonomy in the reorganization skill |
| 8. Naming rules | Root basic naming; method/purpose examples and historical-name treatment in the reorganization skill |
| 9. Dependency and relationship analysis | Reorganization mapping, with callers, entry points, inputs, outputs, and uncertainty |
| 10. Output organization and reproducibility | Root protection plus reorganization provenance, inventory, hashes, ownership, and path semantics |
| 11. Standard Python file header | Conditional header schema in the reorganization skill |
| 12. Numerical verification | Scientific claim-specific evidence and comparisons; structural parity in the reorganization skill; compute boundary in the root |
| 13. Required verification after structural changes | Reorganization checks selected by affected behavior; root warning and completion rules |
| 14. Documentation deliverable | Reorganization hierarchy requirements, including complete inventory for a full reorganization |
| 15. Definition of done | Root completion and reorganization acceptance evidence |
| 16. Conversation handoff summaries | Root communication/handoffs; unique dated filenames and explicit evidence categories |

## Intentional policy changes

- Documentation edits and other clearly scoped bounded work can proceed without
  a separate plan approval merely because multiple files are involved.
- Changes affecting numerical behavior, dependencies, repository structure, or
  research-output retention still need an inspected, reviewable, approved plan.
  Existing authorization covers implementation and relevant low-cost checks.
- Planning does not depend on the agent being able to switch application modes:
  when Plan mode is unavailable, it presents the plan without governed edits.
- Verification follows the affected claim. Whole-repository compilation and
  numerical sweeps are not default checks for prose. New failures need
  investigation; documented baseline warnings are assessed and reported rather
  than silently suppressed or mistaken for a warning-free result.
- Specialized methods are loaded through precise skill triggers. The root
  `skills.md` index is not configured as an alternative instruction filename.
- Historical results remain intact but are labeled as historical. No current
  numerical acceptance claim is inferred from those records.

No blanket delegation mandate or model-effort override was added. Existing
platform permissions and available tools govern execution. Markdown instructions
do not select GPT-6 Astra, change its reasoning effort, or enable API features.
This update targets instruction clarity and context use; an actual performance
claim would require comparable before/after task evaluations.

## Representative decision review

The following are manual instruction-consistency checks, not independent model
runs or empirical skill-selection benchmarks.

| Scenario | Required outcome from the revised instructions |
|---|---|
| Correct prose in two documents | Proceed; no computational skill or numerical sweep |
| Explain an existing formula | Inspect relevant evidence; no implementation approval or skill required |
| Change a solver tolerance | Scientific skill; inspect numerical consequences and obtain plan approval |
| Implement an already approved numerical fix | Complete scoped edits and relevant checks without requesting approval again |
| Move a producer and its artifacts | Reorganization skill; approved mapping, preserved artifacts, import/path and numerical checks |
| Make a structural move without changing computation | Reorganization workflow supplies parity checks; do not load the scientific skill automatically |
| Encounter a documented baseline warning | Assess impact and report it; do not claim warning-free acceptance |
| Run an unapproved expensive sweep | Ask for that computation before starting; continue independent authorized work |
| Preserve the discussion for another conversation | Create the requested unique dated handoff and verify live facts |

## Validation evidence

Observed checks for this six-file change:

- `AGENTS.md` now has 137 lines and 1,045 words, versus 437 lines and 2,763
  words before the edit: 62.2% fewer words in this always-loaded repository file.
  This measures text reduction, not model-token savings or task performance.
- Both skills passed the installed Skill Creator `quick_validate.py` under the
  `relax` interpreter with PyYAML 6.0.3. This checks skill metadata and basic
  structure, not behavioral effectiveness.
- All 23 relative Markdown links across the six files resolved. UTF-8 reads,
  final newlines, and trailing-whitespace checks passed.
- The mapping covers all 16 original sections. Manual review of the nine
  scenarios above found the specified approval, routing, evidence, and stopping
  boundaries consistent; no independent model evaluation was performed.
- SHA-256 comparison against a fresh implementation-start snapshot covered all
  136 tracked files. Only the two approved existing documentation files changed;
  all 134 other tracked files remained byte-identical, including the two
  pre-existing modified research files. Four new Markdown files complete the
  approved six-file scope.
- `git diff --check` passed. Git emitted Windows LF-to-CRLF conversion notices;
  these were not whitespace-check failures. The task diff was reviewed without
  staging or committing the user's work.

The protected research-file hashes at implementation start and verification were:

| File | SHA-256 |
|---|---|
| `experiments/relaxation/atomistic_two_chain_lj_lbfgs_convergence_study.py` | `73901b8d54ededdb1617f46936fd0dbf00eb49d8ef6fec1fcdcc4f14cfd64903` |
| `outputs/relaxation/output_atomistic_two_chain_lj_lbfgs_convergence_study/check_report.txt` | `f4ebe3768564943c0aacc44a19bebad3b8e392b14e0c282493090760f5b317c7` |

The source file changed between the initial audit and implementation start;
preservation was checked against the fresh implementation snapshot.

Reproduce the skill metadata checks from the repository root:

```powershell
& 'C:\Users\lh201\anaconda3\envs\relax\python.exe' -B -X utf8 'C:\Users\lh201\.codex\skills\.system\skill-creator\scripts\quick_validate.py' '.agents\skills\clr-scientific-python'
& 'C:\Users\lh201\anaconda3\envs\relax\python.exe' -B -X utf8 'C:\Users\lh201\.codex\skills\.system\skill-creator\scripts\quick_validate.py' '.agents\skills\clr-repository-reorganization'
git diff --check
```

No scientific calculations or full sweeps are needed for this documentation
change. Fresh-session skill discovery and actual Astra performance remain
unmeasured by static validation.
