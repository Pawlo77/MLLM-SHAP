#!/usr/bin/env bash
# PreToolUse on Bash: `gh pr merge` runs only for a routine branch classified by skill
# ship (scripts/pr_policy.py marker for this session, made on HEAD). Every other command
# passes through to the tool's own permission settings.
source "$(dirname "$0")/lib.sh"
source "$(dirname "$0")/gate_lib.sh"

cmd="$(shell_command)"
case "$cmd" in
  *"gh pr merge"*) ;;
  *) is_cursor && jq -n '{permission: "allow"}'; exit 0 ;;
esac

verdict="$(cd "$ROOT" && uv run --quiet python scripts/pr_policy.py --merge-ok --session "$(session_key)" 2>&1)"
if [ "$verdict" = "ok" ]; then
  is_cursor && jq -n '{permission: "allow"}'
  exit 0
fi
emit_deny "gh pr merge blocked: ${verdict:-no routine marker for this session}. Routine branches merge through skill ship; others wait for the maintainer."
