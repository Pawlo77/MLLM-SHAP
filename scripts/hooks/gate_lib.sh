# Helpers for stop_test_gate.sh and merge_gate.sh. Sourced after lib.sh; bash 3.2
# compatible (macOS /bin/bash).

# Print nothing for Claude Code and Codex, "{}" for Cursor, and exit 0.
quiet_exit() {
  is_cursor && echo '{}'
  exit 0
}


# Terminate a process and its descendants (make -> uv -> pytest).
kill_tree() {
  local child
  for child in $(pgrep -P "$1" 2>/dev/null); do
    kill_tree "$child"
  done
  kill -TERM "$1" 2>/dev/null
}

# run_limited <seconds> <command...>: run with a time limit; exit status 124 on timeout.
# Uses timeout/gtimeout when installed, otherwise a background watchdog (macOS has neither).
run_limited() {
  local limit="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$limit" "$@" </dev/null
    return $?
  fi
  if command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$limit" "$@" </dev/null
    return $?
  fi
  local mark
  mark="$(mktemp "${TMPDIR:-/tmp}/gate-timeout.XXXXXX")" || return 1
  rm -f "$mark"
  "$@" </dev/null &
  local pid=$!
  (
    sleep "$limit"
    kill -0 "$pid" 2>/dev/null || exit 0
    touch "$mark"
    kill_tree "$pid"
  ) >/dev/null 2>&1 &
  local dog=$!
  wait "$pid" 2>/dev/null
  local rc=$?
  kill "$dog" 2>/dev/null
  wait "$dog" 2>/dev/null
  if [ -f "$mark" ]; then
    rm -f "$mark"
    return 124
  fi
  return "$rc"
}


# Session key for per-session gate state: session_id (Claude Code, Codex), else
# conversation_id (Cursor), else this process id (state then never outlives the hook).
session_key() {
  local id
  id="$(printf '%s' "$HOOK_INPUT" | jq -r '.session_id // .conversation_id // empty' 2>/dev/null \
    | tr -cd 'A-Za-z0-9_-' | cut -c1-64)"
  [ -z "$id" ] && id="pid$$"
  printf '%s' "$id"
}

# Content hash of the tree-dirs of [tool.agent-kit] (default src tests): every tracked or
# untracked (not ignored) file there,
# by path and blob hash. Catches edits made through the shell (sed, heredocs, scripts)
# that the edit hooks never see, and stays the same across staging and commits. Run from
# $ROOT. The hash of the last green stop-gate run (seeded at session start) is kept in
# .remember/tmp/tests-ok.<key>.
tree_hash() {
  local files
  files="$(git ls-files --cached --others --exclude-standard -- $(kit_conf tree-dirs 'src tests') 2>/dev/null \
    | while IFS= read -r f; do [ -f "$f" ] && printf '%s\n' "$f"; done)"
  {
    printf '%s\n' "$files"
    [ -n "$files" ] && printf '%s\n' "$files" | git hash-object --stdin-paths 2>/dev/null
  } | git hash-object --stdin 2>/dev/null
}
