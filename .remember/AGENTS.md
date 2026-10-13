# Memory agents

- Committed memory of the `remember` plugin: `remember.md` is the handoff (read first, write with `/remember` before a session ends); `identity.md` says how to work here (hand-edited); `config.json` holds the plugin settings.
- The plugin writes the compressed layers (`today-*.md`, `recent*.md`, `archive*.md`, `core-memories.md`); `now.md`, per-session handoffs, `logs/`, `tmp/`, `run/` and locks stay local (`.gitignore`).
- No secrets, tokens, private hostnames or personal data; the prek hook `memory-content-guard` (`scripts/hooks/check_memory_files.py`) rejects them in `.remember/*.md` and the vault notes.
- `config.json` pins `data_dir: .remember`, `handoff_mode: single`, `prompt_stamp: stable` and no plugin promotions. The plugin writes a `.remember/.gitignore` of `*` only when it is absent, so the committed one is honoured.
- Capture stops when the plugin is disabled (`/plugin`).
- Other tools: Codex `codex plugin marketplace add Digital-Process-Tools/claude-remember && codex plugin install remember`; Cursor's session-start hook (`scripts/hooks/session_start.sh`) injects the handoff.
- Memory is committed so Cursor and Codex sessions start with it; rejected: `~/.remember` (per machine).
