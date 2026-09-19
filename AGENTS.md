# AGENTS.md

## Exit criteria

```bash
make check
```

Same as `make agent-check`: filtered pytest + `prek-all`.

```bash
uv run pytest -m "not lms and not gpu and not bench" tests/
make prek-all
```

Optional markers: `make tests-all` · `make bench` (not part of `make check`).

## Agent-only

- Skill `caveman` at **ultra** for all chat (auto-invoked; code/commits/PRs still normal)
- No commit / push / PR unless explicitly asked — `git-commits.mdc`
- After `gh pr create`: always `--add-assignee Pawlo77` + label from title type (`enhancement` / `bug` / `documentation`); never draft/WIP PRs
- No new markdown or doc expansion unless asked
- No runtime deps without justification in `pyproject.toml`
- No force push; no `git commit --amend` unless user rules allow
- `pre-commit` CLI → use **prek**; `pip install` → `pyproject.toml` + `uv lock`
- Commit subjects: `type: summary` (feat|fix|docs|test|refactor|chore|ci|build|perf); prek `commit-msg` + Cursor hooks enforce
- No direct push to `main` (Cursor deny + GitHub branch protection — see `ci.mdc`)

Keep this file **general**. Detail lives in skills and `.cursor/rules/*.mdc`.

## Sibling repos (multi-root workspace)

Not vendored here. Clone as **siblings** of `nlp-shap/` under one parent:

```text
<parent>/
  nlp-shap/                 # https://github.com/Pawlo77/nlp-shap.git
  nlp-shap-research/        # https://github.com/Pawlo77/nlp-shap-research.git
  MLLM-Shap/                # https://github.com/Pawlo77/MLLM-Shap.git
```

```bash
mkdir -p <parent> && cd <parent>
git clone https://github.com/Pawlo77/nlp-shap.git
git clone https://github.com/Pawlo77/nlp-shap-research.git
git clone https://github.com/Pawlo77/MLLM-Shap.git
```

Open **`<parent>/nlp-shap/nlp-shap.code-workspace`** (File → Open Workspace from File).
Paths in that file are relative to `nlp-shap/` (`../nlp-shap-research`, `../MLLM-Shap`).

| Need | Path after clone |
|------|------------------|
| Legacy port | `../MLLM-Shap/mllm_shap/` |
| Rewrite plan | `../nlp-shap-research/docs/plans/infrastructure/` |

## Load order

1. This file (exit + agent-only)
2. Skill `caveman` (**ultra** — auto-invoked for all chat; off only via "stop caveman" / "normal mode")
3. Always-apply rules: `caveman`, `minimal-diff`, `git-commits`, `uv-workflow`, `package-boundaries`
4. Glob-matched rules for open files (types, docs, examples, backends, …)
5. Skill `nlp-shap-development` for implementation work
6. Skill `nlp-shap-rewrite` only for audio-track / named rewrite phases
7. Skill `nlp-shap-port-legacy` when porting from MLLM-Shap
