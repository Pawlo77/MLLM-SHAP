---
title: "Onboarding"
type: handbook
domain: handbook
tags: [handbook, onboarding]
created: 2026-10-07
updated: 2026-10-08
---

# Onboarding

Part of [[nlp-shap]]. Next: [[How a run works]].

## Set up

1. Clone the repository; optionally clone `MLLM-Shap` as a sibling (see `AGENTS.md`).
2. `uv self update` (uv 0.12 or newer), then `make install`: Python 3.12 ([[Tooling decisions#2026-10-07 — Python upper bound lifted|Python upper bound lifted]]), all dependency groups, prek git hooks.
3. `make check`: filtered tests, every prek hook and `make vault-check`. It must pass before any hand-off.

## Open the vault

- Open the repository folder as a vault in Obsidian (`open -a Obsidian .` on macOS); accept the community plugins listed in `.obsidian/community-plugins.json` (Code Graph, Dataview, Templater, Mermaid Tools). Their code (`main.js`, `styles.css`) is committed next to each manifest, so the vault works on a fresh clone; after a version bump in a manifest run `make obsidian-plugins` to download that release. Obsidian drops an entry from `community-plugins.json` whose `main.js` is missing; restore the list after a failed download.
- Code Graph colours each package of `src/nlp_shap`; its settings are committed in `.obsidian/plugins/code-graph/data.json`.
- Without Obsidian, read the same notes on GitHub or in the documentation site's handbook section.
- Tool-specific hook behaviour found during the session-start smoke test: [[Fixed issues|T01]].

## Where things live

| Path | What |
|---|---|
| `src/nlp_shap/` | the package; one architecture note per subpackage ([[Modules]]) |
| `tests/` | pytest suite; markers `lms`, `gpu`, `bench`, `slow`, `practical` are excluded from `make check` |
| `tests/practical/` | real-model scenarios on a GPU, `make practical FEATURE=<glob>` ([[Process decisions#2026-10-08 — Practical tests on real small models\|Practical tests]]) |
| `docs/` | Sphinx site (`make docs`) |
| `examples/` | notebooks with stored outputs (`make notebooks`) |
| `knowledge/` | this vault ([[knowledge/README|rules]]) |
| `plans/` | implementation plans ([[Tooling decisions#2026-10-07 — Plans directory\|Plans directory]]) |
| `.agents/skills/` | agent skills shared by Cursor, Claude Code and Codex ([[Tooling decisions#2026-10-07 — Agent skills canonical directory\|Agent skills canonical directory]]) |
| `scripts/` | vault checker, knowledge builder, editor hooks |

## The vibe loop

1. Read the handoff (`.remember/remember.md`; [[Tooling decisions#2026-10-07 — Memory committed under remember|Memory committed under remember]]), the map and the architecture note of the package you touch.
2. Plan in `plans/` for anything larger than one task.
3. Test first, implement, run `make check`.
4. Update the architecture note, append decision sections and issue notes (skill `notes`) within the [[Process decisions#2026-10-08 — Brevity budgets|brevity budgets]], run `make knowledge vault-check`.
5. Review with skill `review` ([[Process decisions#2026-10-08 — Claude reviews before the pull request|Claude reviews before the pull request]]); commit only after the maintainer approves.
