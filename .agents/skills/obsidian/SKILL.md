---
name: obsidian
description: >-
  Use for Obsidian vault machinery: Dataview, Bases, Canvas, Mermaid, Code Graph,
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
2. Cache or index folder: add it to `.gitignore`, to `.cursorindexingignore`, and to
   `.cursorignore` when it can hold keys or personal data.
3. Drop plugins no longer enabled.

## Never touch

- Never edit `.obsidian/workspace*.json`, `app.json`, `appearance.json`,
  `core-plugins.json`, `community-plugins.json` or other plugins' `data.json`: Obsidian
  rewrites them on every open. Tell the user which setting to change.
- Never read or commit `.smart-env/` or a plugin data file that may hold keys.

## Links

- Link notes by file name without `.md` (`[[Run archive]]`); aliases do not resolve.
- Link code and other files by vault path with extension.
- Inside Markdown tables escape the alias pipe: `[[Note\|text]]`.
- Never use `#` in note file names (it starts a heading link).
- After renaming a note, fix every `[[Old name` link; `make vault-check` lists them.
- Every new note needs a link from an existing note (the checker flags isolated notes).
- Agents cannot see renders (callouts, Mermaid, math): say so and ask the user to check
  the note in Obsidian.

## Showing the user a note

`open "obsidian://open?vault=<vault>&file=<URL-encoded path>"` (`.md` optional) or
`obsidian://search?vault=<vault>&query=<text>`. Never use URIs to write content.
