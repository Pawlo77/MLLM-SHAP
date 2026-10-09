# Vendored subagents

Agents copied from upstream with attribution. Only MIT or Anthropic (Apache-2.0) sources are
vendored; Trail of Bits skills (CC-BY-SA-4.0) are installed as plugins and never copied here.
To update, re-fetch the file at a newer commit, re-apply the local edits listed below and change
the commit. `scripts/sync_agents.py` regenerates `.codex/agents/*.toml` from these files.

| Agent | Upstream path | Commit | Licence |
|-------|---------------|--------|---------|
| pr-writer | [agent37-platform/claude-code-subagents](https://github.com/agent37-platform/claude-code-subagents) `agents/<name>.md` | `3e2fbb6bb6fac2eb65d3350ef1f5398ee3213f0f` | MIT |
| docs-auditor (method, rules and patch section merged from `agents/doc-syncer.md`) | same repository and commit | `3e2fbb6bb6fac2eb65d3350ef1f5398ee3213f0f` | MIT |

## Removed

- `code-explorer` and `code-architect` (vendored from
  [anthropics/claude-plugins-official](https://github.com/anthropics/claude-plugins-official)
  `plugins/feature-dev/agents/<name>.md` at `aecd4c852f10b466245f18383fa6aad8c0b10d57`,
  Apache-2.0) were removed on 2026-10-08: Claude Code ships the built-in `Explore` and `Plan`
  agents, which cover the same ground without an extra description in every session.
- `doc-syncer` was merged into `docs-auditor` on 2026-10-08 (one documentation agent instead
  of two); its CHANGELOG wording was dropped because the changelog is `docs/release_notes.rst`.
- `flaky-test-hunter`, `dependency-auditor` and `git-archaeologist` (agent37, same commit,
  MIT) were removed on 2026-10-08: a single-maintainer repository does not need them, and
  the built-in `general-purpose` agent covers an occasional flake, dependency or history question.

## Licence notices

agent37 agents, MIT:

```text
MIT License

Copyright (c) 2026 Agent37

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Local edits

All vendored agents:

- `model: inherit` added to the agent37 agents (their frontmatter has none; the roster test
  requires it).
- One paragraph of nlp-shap context added after the opening paragraph (where the docs, vault,
  manifest or conventions live). The method, rules and report format are unchanged unless listed.

Per agent:

- `docs-auditor` (doc-syncer part): read-only (upstream `Edit` dropped): it proposes
  unified-diff patches in its report instead of editing; points at `README.md`, `docs/`,
  `examples/` and `knowledge/`; vault notes only through skill `notes`; release notes in
  `docs/release_notes.rst` replace the upstream CHANGELOG.
- `pr-writer`: title and body follow this repository's template and labels; it never runs
  `gh pr create` (the upstream "If asked to open the PR, run that command" is replaced by
  "Do not run them"), because opening a PR needs the user's approval in chat.

Every agent in `.claude/agents/` ends with a `## Report format` section (the agent37 "report
contract"); `tests/hooks/test_agents.py` enforces it.
