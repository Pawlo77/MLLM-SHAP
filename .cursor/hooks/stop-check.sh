#!/usr/bin/env bash
# On stop: dirty src/tests → agent-check; else ahead of upstream → push/CI remind.
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)" || exit 0

dirty=$(git status --porcelain -- 'src/' 'tests/' 2>/dev/null | grep -E '\.py$' || true)
if [[ -n "$dirty" ]]; then
  python3 -c 'import json; print(json.dumps({
    "followup_message": "src/ or tests/ has uncommitted Python changes. Run make agent-check before hand-off (AGENTS.md)."
  }))'
  exit 0
fi

ahead=""
if git rev-parse --abbrev-ref '@{u}' >/dev/null 2>&1; then
  ahead=$(git rev-list --count '@{u}..HEAD' 2>/dev/null || echo 0)
elif git rev-parse --verify origin/main >/dev/null 2>&1; then
  ahead=$(git rev-list --count 'origin/main..HEAD' 2>/dev/null || echo 0)
fi

if [[ -n "$ahead" && "$ahead" != "0" ]]; then
  python3 -c "import json; print(json.dumps({
    'followup_message': (
      'Branch is ahead of upstream by ${ahead} commit(s). '
      'After user approval: push feature branch (not main), then '
      'gh run watch --exit-status. Propose a PR if ready (/pr).'
    )
  }))"
  exit 0
fi

echo '{}'
exit 0
