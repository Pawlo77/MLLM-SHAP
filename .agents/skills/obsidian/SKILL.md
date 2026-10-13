---
name: obsidian
description: >-
  Use for Obsidian vault machinery: Dataview, Bases, Canvas, Mermaid, Code Graph,
<<<<<<< before updating
  Templater, obsidian:// URIs, rendering problems, or any edit under .obsidian/.
---

# mllm-shap Obsidian vault

The vault `mllm-shap` is the repository root. Notes live in `knowledge/` (written with
skill `notes`; the vault rules are in `knowledge/AGENTS.md`), plus
`plans/` and `.remember/*.md`. Read [references/syntax.md](references/syntax.md) before
writing Dataview, Bases, Canvas, Mermaid or Templater.

## Keeping this skill current

When `.obsidian/community-plugins.json` or `core-plugins.json` changes, rewrite this skill
before other work:

1. New plugin: read `.obsidian/plugins/<id>/manifest.json` (and `data.json`); add a
   line under Plugins, and a section if agents use it (files it writes, whether agents
   may touch them, Code Graph or `make vault-check` effects).
=======
  Templater, properties, tags, obsidian:// URIs, rendering problems, or any edit under
  .obsidian/.
---

# Obsidian vault

The vault is the repository root. `local.md` names the vault, the note folders, the
plugins and the files agents may touch; read it first. Read
[references/syntax.md](references/syntax.md) before writing Dataview, Bases, Canvas,
Mermaid or Templater.

## Keeping this skill current

When `.obsidian/community-plugins.json` or `core-plugins.json` changes, update the
Plugins list in `local.md` before other work:

1. New plugin: read `.obsidian/plugins/<id>/manifest.json` (and `data.json`); add a line,
   and a section if agents use it (files it writes, whether agents may touch them).
>>>>>>> after updating
2. Cache or index folder: add it to `.gitignore`, to `.cursorindexingignore`, and to
   `.cursorignore` when it can hold keys or personal data.
3. Drop plugins no longer enabled.

<<<<<<< before updating
## Plugins

- **Code Graph**: draws code, tests and notes as one graph from imports, `[[links]]`
  in comments and docstrings (`@see`, `@tested-by`), `@domain` tags and note
  `related-code`. Settings: the committed `.obsidian/plugins/code-graph/data.json`.
- **Dataview**: live tables in `knowledge/Dashboard.md`; DQL only, no DataviewJS.
- **Templater**: templates in `knowledge/_templates/` for people; agents write notes
  directly from skill `notes`.
  Excluded files (`app.json`) keep templates, `docs/`, `README.md`/`CONTRIBUTING.md` and
  every `AGENTS.md`/`CLAUDE.md` out of the graph, search and link suggestions (templates
  build links from split bracket strings, so none is clickable). The snippet
  `.obsidian/snippets/hide-non-notes.css` hides folders without notes (`src/`, `tests/`,
  ...) from the file explorer; list a new such folder there.
- **Mermaid Tools**: an editor toolbar; agents only write ```` ```mermaid ```` fences.
- **Bases, Canvas**: `.base` property views and `.canvas` maps under `knowledge/`.
- **Templates (core)**: folder `knowledge/_templates`.
- Other core plugins are user interface only; Graph view is superseded by Code Graph;
  Daily notes and Sync are off.

## Files agents may touch

- **Write freely**: `knowledge/**/*.md` except `knowledge/generated/` (rebuilt by
  `make knowledge`), `knowledge/**/*.base`, `knowledge/**/*.canvas`.
- **Change in Obsidian, then commit**: `.obsidian/plugins/code-graph/data.json`.
- **Never edit**: `.obsidian/workspace*.json`, `app.json`, `appearance.json`,
  `core-plugins.json`, `community-plugins.json`, other plugins' `data.json`. Tell the
  user which setting to change.
- **Never read or commit**: `.smart-env/`, `.obsidian/plugins/realclaudian/data.json`
  (optional local plugins that may hold keys; both git-ignored).
- Plugin code (`main.js`, `styles.css`) is committed next to each manifest at the pinned
  version (`make obsidian-plugins` re-downloads it); only `wasm/` caches are git-ignored.

## Links and Code Graph

- Link notes by file name without `.md` (`[[Run archive]]`); link code and other
  files by vault path with extension (`[[src/mllm_shap/estimation/exact.py]]`); link
  plans and memory by path without `.md` (`[[plans/<file>]]`,
  `[[.remember/remember]]`).
=======
## Never touch

- Never edit `.obsidian/workspace*.json`, `app.json`, `appearance.json`,
  `core-plugins.json`, `community-plugins.json` or other plugins' `data.json`: Obsidian
  rewrites them on every open. Tell the user which setting to change.
- Never read or commit `.smart-env/` or a plugin data file that may hold keys.

## Links

- Link notes by file name without `.md` (`[[Run archive]]`); aliases do not resolve.
- Link code and other files by vault path with extension.
>>>>>>> after updating
- Inside Markdown tables escape the alias pipe: `[[Note\|text]]`.
- Never use `#` in note file names (it starts a heading link).
- After renaming a note, fix every `[[Old name` link; `make vault-check` lists them.
- Every new note needs a link from an existing note (the checker flags isolated notes).
<<<<<<< before updating
- Code traceability tags: rule `python-docstrings`.
- Before changing a module or note: `make graph ARGS="--neighbors <path or note>"`;
  to find code: `make graph ARGS=--codemap`.

## Views

- Live tables go into `knowledge/Dashboard.md`; never create a second dashboard.
- Generated tables (`knowledge/generated/`) are embedded with `![[Open issues]]` so
  the dashboard also reads on GitHub.
- Agents cannot see renders: say so, and ask the user to check the note in Obsidian.

## Showing the user a note

- `open "obsidian://open?vault=mllm-shap&file=knowledge%2Fmllm-shap"` (URL-encoded
  path; `.md` optional) or `obsidian://search?vault=mllm-shap&query=<text>`.
- Never use URIs to write content.
=======
- Agents cannot see renders (callouts, Mermaid, math): say so and ask the user to check
  the note in Obsidian.

## Showing the user a note

`open "obsidian://open?vault=<vault>&file=<URL-encoded path>"` (`.md` optional) or
`obsidian://search?vault=<vault>&query=<text>`. Never use URIs to write content.
>>>>>>> after updating
