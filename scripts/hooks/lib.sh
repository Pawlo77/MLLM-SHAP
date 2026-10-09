# Shared by the agent hooks. Claude Code and Cursor both run them; they differ in how
# context is returned. Cursor's input has cursor_version.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOOK_INPUT=""
[ -t 0 ] || HOOK_INPUT="$(cat)"
is_cursor() { printf '%s' "$HOOK_INPUT" | jq -e 'has("cursor_version")' >/dev/null 2>&1; }
# Edited paths, one per line: Claude's tool_input path, Cursor's file_path, else the
# "*** (Update|Add) File:" lines of a Codex apply_patch carried in tool_input.command.
edited_file() {
  local path
  path="$(printf '%s' "$HOOK_INPUT" | jq -r '.tool_input.file_path // .tool_input.notebook_path // .file_path // empty' 2>/dev/null)"
  if [ -n "$path" ]; then
    printf '%s\n' "$path"
    return 0
  fi
  shell_command | sed -n -E 's/^\*\*\* (Update|Add) File: (.*)$/\2/p'
}
shell_command() {
  printf '%s' "$HOOK_INPUT" | jq -r '.tool_input.command // .command // empty' 2>/dev/null
}
emit_context() { # <event> <text> [short message]
  if is_cursor; then jq -n --arg t "$2" '{additional_context: $t}'
  elif [ "$1" = SessionStart ]; then printf '%s\n' "$2"
  else jq -n --arg e "$1" --arg t "$2" --arg s "${3:-$2}" '{systemMessage: $s, hookSpecificOutput: {hookEventName: $e, additionalContext: $t}}'
  fi
}
emit_blocking() { # <text>
  if is_cursor; then jq -n --arg t "Fix these before continuing:
$1" '{additional_context: $t}'; exit 0
  else printf '%s\n' "$1" >&2; exit 2
  fi
}
emit_deny() { # <reason>
  if is_cursor; then jq -n --arg r "$1" '{permission: "deny", user_message: $r, agent_message: $r}'
  else jq -n --arg r "$1" '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: $r}}'
  fi
}
