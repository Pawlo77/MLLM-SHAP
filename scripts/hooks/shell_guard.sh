#!/usr/bin/env bash
# Before a shell command (Claude Code and Codex PreToolUse on Bash, Cursor
# beforeShellExecution): deny skipped hooks, force pushes, amends, malformed commit messages
# and draft or WIP PRs; ask before cluster jobs, submit/deploy/publish scripts and uploads.
# Classification: shell_guard.py (stdlib only). Allowed commands pass through silently to the
# tool's own permission settings. Fail-open when jq or python3 is missing.
source "$(dirname "$0")/lib.sh"
command -v jq >/dev/null 2>&1 || exit 0
command -v python3 >/dev/null 2>&1 || exit 0
source "$(dirname "$0")/gate_lib.sh"
cmd="$(shell_command)"
[ -n "$cmd" ] || quiet_exit
decision="$(python3 "$(dirname "$0")/shell_guard.py" "$cmd" 2>/dev/null)" || quiet_exit
permission="$(printf '%s' "$decision" | jq -r '.permission // "allow"')"
reason="$(printf '%s' "$decision" | jq -r '.reason // ""')"
case "$permission" in
  deny) emit_deny "$reason" ;;
  ask) emit_ask "$reason" ;;
  *) quiet_exit ;;
esac
exit 0
