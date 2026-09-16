# CLR skill index

Repository instructions and authorization live in [AGENTS.md](AGENTS.md).
This file is a human-readable index, not an automatically loaded skill.

Codex discovers the two instruction-only skills below through
`.agents/skills/<name>/SKILL.md`. Each contains its own focused workflow;
read the matching skill rather than loading both by default.

| Skill | Use when | Does not cover |
|---|---|---|
| [clr-scientific-python](.agents/skills/clr-scientific-python/SKILL.md) | Implementing or reviewing computational Python changes, including their numerical validation | Prose-only edits, ordinary conceptual explanations, or a purely structural move |
| [clr-repository-reorganization](.agents/skills/clr-repository-reorganization/SKILL.md) | Planning, implementing, or reviewing source/output moves and repository restructuring | Ordinary code fixes or changes to the scientific model |

Invoke a skill explicitly with `$clr-scientific-python` or
`$clr-repository-reorganization`, or let Codex match its description.
Selection does not authorize edits outside the user's request or approved plan.

Keep descriptions short and specific. Put enduring repository rules in
`AGENTS.md`, and task-specific procedures in the relevant skill. Add supporting
files only when an actual workflow needs them. For the rationale and validation
of this layout, see the [2026-09-13 audit](docs/agent_guidance_audit.md).
