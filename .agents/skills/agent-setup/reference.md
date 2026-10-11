# Agent setup reference

## Install lines

- **Claude Code:** accept the plugin prompt, or `/plugin marketplace add` each marketplace in the settings and `/plugin install <name>@<marketplace>` each entry of `enabledPlugins`.
- **Codex:** `codex plugin marketplace add Digital-Process-Tools/claude-remember && codex plugin install remember`; `codex plugin marketplace add Digital-Process-Tools/codex-marketplace && codex plugin add jit-context@dpt-plugins`; `codex plugin install astral` for the uv, ruff and ty skills (or read them from the Claude Code plugin cache); Superpowers from the official Codex marketplace (`/plugins`).
- **Cursor:** `/add-plugin superpowers`; it also reads `.claude/agents/`, `.claude/skills/` and `.cursor/hooks.json`, and its `sessionStart` hook injects the `.remember/` handoff.
- `security-guidance`: the per-Stop model review is off (`ENABLE_STOP_REVIEW=0`); pattern warnings stay.

## Literature hosts

Claude Code fetches from Semantic Scholar, arXiv and Crossref (plus `doi.org` redirects) without a prompt, so `paper-scout` runs unattended. The accepted risk is exfiltration: an injected prompt could put file contents into a query string sent to those hosts. The Read deny list (secrets, keys, credential files, every `.cursorignore` entry) limits what could leak; the shell guard does not inspect network commands. Report a way around these limits as a vulnerability (`CONTRIBUTING.md`, section Security).

## Codex

Codex reads `AGENTS.md` and `.agents/skills/` natively. `.codex/config.toml` sets the sandbox and approval policy, `.codex/hooks.json` runs the shared `scripts/hooks/*.sh` (trust them once in Codex's hook review) and `.codex/agents/*.toml` mirror `.claude/agents/`. Codex has no `TaskCompleted`, `PostToolUseFailure` or `FileChanged` event, so those gates run in Claude Code only: run `make check` before hand-off and ask before any commit, push, tag or pull request.
