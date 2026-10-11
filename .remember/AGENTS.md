# Memory agents

- Committed memory of the `remember` plugin: `remember.md` is the handoff (read first, write with `/remember` before a session ends); `identity.md` says how to work here (hand-edited); `config.json` holds the plugin settings.
- The plugin writes the compressed layers (`today-*.md`, `recent*.md`, `archive*.md`, `core-memories.md`); `now.md`, per-session handoffs, `logs/`, `tmp/`, `run/` and locks stay local (`.gitignore`).
- No secrets, tokens, private hostnames or personal data; the prek hook `memory-content-guard` (`scripts/hooks/check_memory_files.py`) rejects them in `.remember/*.md` and `knowledge/**/*.md`.
- `config.json` pins `data_dir: .remember`, `handoff_mode: single`, `prompt_stamp: stable` and no plugin promotions. The plugin writes a `.remember/.gitignore` of `*` only when it is absent, so the committed one is honoured; before, memory never left one machine and Cursor and Codex sessions started without it. Rejected: `~/.remember` (per machine).
- Capture stops when the plugin is disabled (`/plugin`); plugin 0.33.0 has no environment switch.
- Other tools: Codex `codex plugin marketplace add Digital-Process-Tools/claude-remember && codex plugin install remember`; Cursor's session-start hook (`scripts/hooks/session_start.sh`) injects the handoff.
