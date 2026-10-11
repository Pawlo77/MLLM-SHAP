# Knowledge agents

The repository root is the Obsidian vault `mllm-shap` (map: `mllm-shap.md`); notes live in
`knowledge/`, plus `plans/` and `.remember/*.md`. Write notes only through skill `notes`
(procedures, templates); vault machinery: skill `obsidian`. `make vault-check` passes
before hand-off.

## Where things live

| Path | Contents |
|---|---|
| `mllm-shap.md`, `Dashboard.md` | map of the vault; Dataview tables and embeds of generated notes |
| `architecture/` | one module note per package of `src/mllm_shap`; a change under `src/mllm_shap/<pkg>/` updates it (bump `updated`) in the same change |
| `decisions/` | one log per domain, `<Domain> decisions.md`: a dated section per design choice, newest first |
| `issues/` | one note per bug found or fixed, `<id> - <title>.md` |
| `concepts/` | one note per term |
| `Papers.md` | one section per cited paper; DOI or arXiv links, never PDFs |
| `handbook/` | how-to notes for people; `Solutions/<date> - <title>.md` for solved problems |
| `generated/`, `_templates/` | rebuilt by `make knowledge`, never edited; Templater templates, not notes |

## Frontmatter

- `title`, `type` (`map`, `dashboard`, `module`, `decision-log`, `issue`, `concept`, `paper-list`, `handbook`, `solution`, `plan`, `generated`), `domain` (a package or `index`, `vault`, `theory`), `parent` (the wikilink of the note one level up, `"[[mllm-shap]]"`; none on the map and generated notes), `tags` (a YAML list without `#`), `created` (`YYYY-MM-DD`), `updated` when the content changes.
- `status`: required on issues (`open`, `fixed`, `wontfix`, `deferred`) and plans (`draft`, `ready`, `in progress`, `done`, `superseded`).
- `related-code`: wikilinks to code by vault path with extension, for example `"[[src/mllm_shap/estimation/exact.py]]"`; they become Code Graph edges. `aliases`: other names.

## Writing rules

- Link notes by file name without `.md` (`[[Run archive]]`), code and other files by vault path with extension; the parent is metadata, never a "Part of" sentence.
- One idea per bullet; content only, no narration of what the note is about.
- Spell out every abbreviation at its first use in a note (`scripts/glossary.yaml`).
- Numbers come from executed output, never from memory.
- No secrets, tokens, private hostnames or personal data beyond the maintainer's public identity; a prek hook rejects them.
- A design choice appends a section to its log; a bug found or fixed adds or updates its issue note.
- `make vault-check` covers frontmatter, sections, links, isolated notes, abbreviations, secrets, generated notes and code traceability tags.

## Why

- Decisions and papers are sections of one log or list, not one note each: 23 decision and 14 paper notes cost 6,447 words against about 4,000 as sections. A log is named `<Domain> decisions` because architecture notes own the package names.
- The vault is the repository root because design knowledge lived in chat logs and an audit report, and a repository-rooted vault lets Code Graph draw code, tests and notes as one graph. Rejected: a separate notes repository (drifts, invisible to agents).
- Community plugins in `.obsidian/` are vendored at the version pinned in each `manifest.json` (`scripts/install_obsidian_plugins.py` re-downloads them): Code Graph 0BSD, Dataview MIT, Mermaid Tools MIT, Templater AGPL-3.0 (runs only inside Obsidian, not part of the wheel).
