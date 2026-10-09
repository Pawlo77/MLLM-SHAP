# Agent memory (`.remember/`)

Memory written by the [remember](https://github.com/Digital-Process-Tools/claude-remember)
plugin for this project, committed so every contributor and every agent tool starts
from the same context.

## Files

| File | What it is | Committed |
|---|---|---|
| `remember.md` | Handoff note: state, next steps, context. Written by `/remember` at the end of a session; read first at the start of the next. | yes |
| `identity.md` | Who the agent is in this repository and how it works here. Edited by hand. | yes |
| `config.json` | Per-project plugin settings: memory stays in `.remember/`, one shared handoff, stable prompt stamp, no plugin promotions. | yes |
| `today-*.md`, `recent.md`, `archive.md`, `recent-*.md`, `archive-*.md`, `core-memories.md` | Compressed daily, weekly and long-term memory layers produced by the plugin. | yes |
| `now.md` | Live session buffer; changes on every tool call. | no |
| `remember.<session>.md` | Per-session handoffs (only with `handoff_mode: per_session`). | no |
| `logs/`, `tmp/`, `run/`, locks, `.install-marker` | Machine-local plugin state. | no (`.gitignore`) |

## Privacy rule

Committed memory must not contain secrets, tokens, private hostnames or personal data
of anyone other than the maintainer's public identity. The prek hook
`memory-content-guard` (`scripts/hooks/check_memory_files.py`) rejects such lines in
`.remember/*.md` and `knowledge/**/*.md`; review memory diffs like any other change.

## Turning capture off

Disable or uninstall the plugin (in Claude Code through `/plugin`). The plugin
0.33.0 has no environment switch for this. The committed files stay readable either way.

## Other tools

- **Claude Code**: install `remember@claude-plugins-official`; the plugin reads and
  writes these files.
- **Codex**: `codex plugin marketplace add Digital-Process-Tools/claude-remember &&
  codex plugin install remember`; it shares the same store.
- **Cursor**: has no memory plugin; its session-start hook
  (`scripts/hooks/session_start.sh`) injects `remember.md`, `identity.md` and the tail
  of `recent.md` and the newest `today-*.md`.

Run `/remember` before ending a session so the handoff stays current.
