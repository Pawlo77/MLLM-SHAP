# Obsidian: mllm-shap

The vault `mllm-shap` is the repository root. Notes live in `knowledge/` (written with
skill `notes`; the vault rules are in `knowledge/AGENTS.md`), plus `plans/` and
`.remember/*.md`.

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
- **Never read or commit**: `.smart-env/`, `.obsidian/plugins/realclaudian/data.json`
  (optional local plugins that may hold keys; both git-ignored).
- Plugin code (`main.js`, `styles.css`) is committed next to each manifest at the pinned
  version (`make obsidian-plugins` re-downloads it); only `wasm/` caches are git-ignored.

## Links and Code Graph

- Code traceability tags: rule `python-docstrings`.
- Before changing a module or note: `make graph ARGS="--neighbors <path or note>"`;
  to find code: `make graph ARGS=--codemap`.

## Views

- Live tables go into `knowledge/Dashboard.md`; never create a second dashboard.
- Generated tables (`knowledge/generated/`) are embedded with `![[Open issues]]` so
  the dashboard also reads on GitHub.
- Agents cannot see renders: say so, and ask the user to check the note in Obsidian.

## Showing the user a note

Example: `open "obsidian://open?vault=mllm-shap&file=knowledge%2Fmllm-shap"`.
