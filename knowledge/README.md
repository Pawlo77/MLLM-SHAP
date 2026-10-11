---
title: "Vault rules"
type: handbook
domain: index
tags: [handbook, vault]
created: 2026-10-07
updated: 2026-10-07
---

# Vault rules

The repository root is an Obsidian vault named `mllm-shap`. It is both the human handbook and the knowledge base agents read before changing code. Start at [[mllm-shap]].

## Where things live

| Path | Contents | Edited by |
|---|---|---|
| `knowledge/mllm-shap.md` | Map of the vault: packages, plugin groups, invariants | hand |
| `knowledge/Dashboard.md` | Dataview tables over note frontmatter, plus embeds of generated notes | hand |
| `knowledge/architecture/` | One module note per package of `src/mllm_shap` | hand, same change as the code |
| `knowledge/decisions/` | One log per domain, `<Domain> decisions.md`: a dated section per design choice, newest first | hand |
| `knowledge/issues/` | One issue note per bug found or fixed, `<id> - <title>.md` | hand |
| `knowledge/concepts/` | One concept note per term | hand |
| `knowledge/papers/` | `Papers.md`: one section per cited paper; DOI (digital object identifier) or arXiv links, never PDFs | hand |
| `knowledge/handbook/` | How-to notes for people: onboarding, run walkthrough, debugging, roadmap | hand |
| `knowledge/generated/` | Notes rebuilt from the code by `scripts/build_knowledge.py` | never by hand |
| `knowledge/_templates/` | Templater templates, one per note type; not notes | hand |
| `plans/` | Implementation plans (notes too) | hand |
| `.remember/*.md` | Committed agent memory and the session handoff (notes too) | the remember plugin |
| `.obsidian/` | Committed vault settings and plugin code; per-machine state is git-ignored | Obsidian, `make obsidian-plugins` |

Community plugins are vendored at the version pinned in each `manifest.json` (`scripts/install_obsidian_plugins.py` re-downloads them): Code Graph 0BSD, Dataview MIT, Mermaid Tools MIT, Templater AGPL-3.0 (its source is the upstream repository; it runs only inside Obsidian and is not part of the `mllm-shap` package or wheel).

## Frontmatter

Every note under `knowledge/` starts with YAML frontmatter:

- `title`: the display title.
- `type`: one of `map`, `dashboard`, `module`, `decision-log`, `issue`, `concept`, `paper-list`, `handbook`, `solution`, `plan`, `generated`.
- `domain`: the package (`estimation`, `runtime`, ...) or a grouping (`index`, `vault`, `theory`).
- `tags`: a YAML list without `#`.
- `created`: the creation date (`YYYY-MM-DD`); `updated` when the content changes.
- `status`: required on issue (`open`, `fixed`, `wontfix`, `deferred`) and plan (`draft`, `ready`, `in progress`, `done`, `superseded`) notes; a superseded decision says so in its heading.
- `related-code`: wikilinks to code by vault path with extension, for example `"[[src/mllm_shap/estimation/exact.py]]"`; these become Code Graph edges.
- `aliases`: other names, for example the abbreviation a concept is known by.

## Writing rules

- Link notes by file name without `.md` (`[[Run archive]]`); link code and other files by vault path with extension.
- One idea per bullet; content only, no narration of what the note is about.
- Spell out every abbreviation at its first use in a note; the list is `scripts/glossary.yaml`.
- Numbers come from executed output, never from memory.
- No secrets, tokens, private hostnames or personal data beyond the maintainer's public identity; a prek hook rejects them.
- Generated notes (`generated: true`) are rebuilt by `make knowledge` and never edited by hand.
- Note-writing procedure per type: skill `notes`; vault machinery: skill `obsidian`.

## The four rules

1. A change under `src/mllm_shap/<pkg>/` updates `knowledge/architecture/<pkg>.md` in the same change (bump `updated`).
2. A design choice appends a section to its `knowledge/decisions/` log.
3. A bug found or fixed adds or updates its note under `knowledge/issues/`.
4. `make vault-check` passes: frontmatter, sections, links and headings, isolated notes, abbreviations, secrets, generated notes and code traceability tags.
