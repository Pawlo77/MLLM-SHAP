#!/usr/bin/env bash
# After git push: remind agent to watch CI workflows.
set -u
input=$(cat)
cmd=$(printf '%s' "$input" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("command") or "")' 2>/dev/null || true)
case "$cmd" in
  *git\ push*) ;;
  *) echo '{}'; exit 0 ;;
esac

# Prefer additional_context (postToolUse/afterShell); empty object if ignored.
python3 -c 'import json; print(json.dumps({
  "additional_context": (
    "git push finished. Watch CI: "
    "gh run list --branch \"$(git branch --show-current)\" --limit 5 "
    "then gh run watch <run-id> --exit-status for Prek Checks / Pytest / Deploy Docs. "
    "Do not leave main red."
  )
}))'
exit 0
