---
name: agent-setup
description: >-
  Use when setting up or changing agent tooling: Cursor, Claude Code or Codex entry
  points, plugins, hooks, memory, the sibling MLLM-Shap-legacy clone or its workspace file.
---

# mllm-shap agent setup

Delivery runs through skill `ship`: routine changes (`scripts/pr_policy.py`) merge after
Copilot's review, others stop before the merge. Copilot code review must be enabled on the
repository (Settings, Copilot, Code review).

## Sibling repository

`MLLM-Shap-legacy` (the legacy code skill `port-legacy` ports from) is cloned next to this
repository:

```bash
mkdir -p <parent> && cd <parent>
for r in MLLM-SHAP MLLM-Shap-legacy; do git clone https://github.com/Pawlo77/$r.git; done
```

Open `<parent>/MLLM-SHAP/mllm-shap.code-workspace`; its paths are relative to `MLLM-SHAP/`.

## Tool matrix

One source per concern; each tool reads it through its own entry point.

| Concern | Source | Cursor | Claude Code | Codex |
|---|---|---|---|---|
| Instructions | `AGENTS.md`, nested `AGENTS.md` | native | `CLAUDE.md` symlinks | native |
| Skills | `.agents/skills/` | `.cursor/skills` symlink | `.claude/skills` symlink | native |
| Commands | `.cursor/commands/` | native | `.claude/commands` symlink | none |
| Rules | `.cursor/rules/*.mdc` | native | `.claude/rules/*.md` (`scripts/sync_rules.py`) | skill `python-style` |
| Hooks | `scripts/hooks/*.sh` | `.cursor/hooks.json` | `.claude/settings.json` | `.codex/hooks.json` |
| Subagents | `.claude/agents/*.md` (read-only) | reads `.claude/agents/` | native | `.codex/agents/*.toml` (`scripts/sync_agents.py`) |
| Settings | per tool | `.cursor/hooks.json` | `.claude/settings.json` | `.codex/config.toml` |

Generated files are never edited by hand: change the source, rerun its generator.

## Plugins

`.claude/settings.json` declares the marketplaces and `enabledPlugins`; required are
`superpowers`, `remember` (memory in `.remember/`) and `astral` (uv, ruff and ty skills).
`claude-jit-context@dpt-plugins` injects `.claude/jit-context/`, written by
`make knowledge`; never run its `rebuild-tsv.sh` over `10-auto/`. Trail of Bits skills are
installed as plugins, never copied; vendored agents are listed in
`.claude/agents/VENDORED.md`. Install lines per tool, the literature hosts and Codex
limits: [reference.md](reference.md).

## Environment

`.env.example` lists the backend and logging variables; copy it to `.env` (git-ignored)
and never commit real values.
