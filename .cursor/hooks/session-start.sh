#!/usr/bin/env bash
# Inject package version, rewrite track, and next audio phase at session start.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 0

version=unknown
if command -v uv >/dev/null 2>&1; then
  version=$(uv run python -c 'import tomllib; from pathlib import Path; print(tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]["version"])' 2>/dev/null || echo unknown)
elif [[ -f pyproject.toml ]]; then
  version=$(grep -E '^version\s*=' pyproject.toml | head -1 | sed -E 's/^version\s*=\s*"([^"]+)".*/\1/' || echo unknown)
fi

phase_hint=$(uv run python "$ROOT/.cursor/hooks/next_phase.py" "$version" 2>/dev/null || echo "Audio track: see skill nlp-shap-rewrite.")

python3 -c "
import json
print(json.dumps({
  'additional_context': (
    'nlp-shap v${version}. Text rewrite track (phases 0-14) done. '
    '${phase_hint} '
    'Legacy port -> ../MLLM-Shap/mllm_shap/. '
    'Exit: make agent-check. Hooks deny pip/conda/pre-commit/force-push. '
    'Sibling clones: see AGENTS.md / nlp-shap.code-workspace.'
  )
}))
"
exit 0
