# Vault reference

## Where things live

| Path | Contents |
|---|---|
| `knowledge/mllm-shap.md`, `knowledge/Dashboard.md` | map of the vault; Dataview tables and embeds of generated notes |
| `knowledge/architecture/` | one module note per package of `src/mllm_shap`, in the same change as the code |
| `knowledge/decisions/` | one log per domain, `<Domain> decisions.md`: a dated section per design choice, newest first |
| `knowledge/issues/` | one note per bug found or fixed, `<id> - <title>.md` |
| `knowledge/concepts/` | one note per term |
| `knowledge/Papers.md` | one section per cited paper; DOI or arXiv links, never PDFs |
| `knowledge/handbook/` | how-to notes for people; `Solutions/<date> - <title>.md` for solved problems |
| `knowledge/generated/`, `knowledge/_templates/` | rebuilt by `make knowledge`; Templater templates |
| `plans/`, `.remember/*.md` | plans and committed memory (notes too) |

Community plugins in `.obsidian/` are vendored at the version pinned in each `manifest.json` (`scripts/install_obsidian_plugins.py` re-downloads them): Code Graph 0BSD, Dataview MIT, Mermaid Tools MIT, Templater AGPL-3.0 (runs only inside Obsidian, not part of the wheel).

## Frontmatter

- `title`, `type` (`map`, `dashboard`, `module`, `decision-log`, `issue`, `concept`, `paper-list`, `handbook`, `solution`, `plan`, `generated`), `domain` (a package or `index`, `vault`, `theory`), `tags` (a YAML list without `#`), `created` (`YYYY-MM-DD`), `updated` when the content changes.
- `status`: required on issues (`open`, `fixed`, `wontfix`, `deferred`) and plans (`draft`, `ready`, `in progress`, `done`, `superseded`).
- `related-code`: wikilinks to code by vault path with extension, for example `"[[src/mllm_shap/estimation/exact.py]]"`; they become Code Graph edges. `aliases`: other names.

## Writing rules

- Link notes by file name without `.md` (`[[Run archive]]`), code and other files by vault path with extension.
- One idea per bullet; content only, no narration of what the note is about.
- Spell out every abbreviation at its first use in a note (`scripts/glossary.yaml`).
- Numbers come from executed output, never from memory.
- No secrets, tokens, private hostnames or personal data beyond the maintainer's public identity.
- Generated notes (`generated: true`) are never edited by hand.
- `make vault-check` covers frontmatter, sections, links, isolated notes, abbreviations, secrets, generated notes and code traceability tags.
