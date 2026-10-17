---
title: "Onboarding"
type: handbook
domain: handbook
parent: "[[mllm-shap]]"
tags: [handbook, onboarding]
created: 2026-10-07
updated: 2026-10-08
---
# Onboarding

Next: [[How a run works]].

## Set up

1. Clone the repository; optionally clone `MLLM-Shap-legacy` as a sibling.
2. `uv self update` (uv 0.12 or newer), then `make install`: Python 3.12, all dependency groups, prek git hooks.
3. `make check`: filtered tests, every prek hook and `make vault-check`. It must pass before any hand-off.

## Open the vault

- Open the repository folder as a vault in Obsidian (`open -a Obsidian .` on macOS); copy the `.example.json` files as CONTRIBUTING.md (Obsidian vault setup) describes, then accept the community plugins listed in `.obsidian/community-plugins.json` (Code Graph, Dataview, Templater, Mermaid Tools). Their code (`main.js`, `styles.css`) is committed next to each manifest, so the vault works on a fresh clone; after a version bump in a manifest run `make obsidian-plugins` to download that release. Obsidian drops an entry from `community-plugins.json` whose `main.js` is missing; restore the list after a failed download.
- Code Graph colours each package of `src/mllm_shap`; its settings are committed in `.obsidian/plugins/code-graph/data.json`.
- Without Obsidian, read the same notes on GitHub or in the documentation site's handbook section.
- Tool-specific hook behaviour found during the session-start smoke test: [[Fixed issues|T01]].

## Where things live

| Path | What |
|---|---|
| `src/mllm_shap/` | the package; one architecture note per subpackage ([[Modules]]) |
| `tests/` | pytest suite; markers `lms`, `gpu`, `bench`, `slow`, `practical` are excluded from `make check` |
| `tests/practical/` | real-model scenarios on a GPU, `make practical FEATURE=<glob>` |
| `docs/` | Sphinx site (`make docs`) |
| `examples/` | notebooks with stored outputs (`make notebooks`) |
| `knowledge/` | this vault: map, architecture notes, decisions, issues |
| `scripts/` | vault checker, knowledge builder, editor hooks |

## Making a change

1. Read the map and the architecture note of the package you touch.
2. Test first, implement, run `make check`.
3. Update the architecture note and append decision sections and issue notes; run `make knowledge vault-check`.
4. Open a pull request; Cursor Bugbot and CodeQL review it.
