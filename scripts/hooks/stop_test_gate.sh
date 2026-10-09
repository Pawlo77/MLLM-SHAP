#!/usr/bin/env bash
# Stop gate (Claude Code and Codex Stop, Cursor stop): the turn may not end with red tests.
# Per session (key: session_id, else Cursor's conversation_id, else the process id). The
# gate acts when tree_hash (gate_lib.sh: a content hash of the files under src/ and tests/)
# differs from .remember/tmp/tests-ok.<key>, the hash session_start.sh seeds and every
# green run updates; shell edits (sed, heredocs) change the hash too. Other sessions never
# gate this one, and a session without a tests-ok file (no session start seen) is not gated.
# It runs `make tests` (the pytest half of make agent-check; prek never runs from a hook)
# with a 180 s limit and a WARNING on stderr past 120 s. Red: Claude Code and Codex get
# decision "block" with at most 20 failing-test lines, Cursor gets a followup_message.
# Green: the run's tree hash goes to tests-ok. Each run's start time, exit code and tree
# hash go to .remember/tmp/stop-test-gate.<key>.last; within 90 s of a run a stop does not
# re-run: after green it passes quietly, after red it re-sends that run's result while the
# tree is unchanged, and re-runs otherwise. A run still in progress (no exit code yet)
# passes quietly. Older results always re-run; continuations (stop_hook_active) are gated
# too. A timeout is reported and never blocks.
# session_start.sh clears this session's state and state older than a day.
# Exits 0 silently when jq, make or .remember/ is missing; NLP_SHAP_TEST_GATE=off disables it.
source "$(dirname "$0")/lib.sh"
command -v jq >/dev/null 2>&1 || exit 0
source "$(dirname "$0")/gate_lib.sh"
[ "${NLP_SHAP_TEST_GATE:-on}" = off ] && quiet_exit
cd "$ROOT" || exit 0
[ -d .remember ] || quiet_exit
state=".remember/tmp"
key="$(session_key)"
ok_file="$state/tests-ok.$key"
hash="$(tree_hash)"
ok="$(cat "$ok_file" 2>/dev/null)"
[ -n "$ok" ] && [ "$hash" != "$ok" ] || quiet_exit
command -v make >/dev/null 2>&1 || quiet_exit
mkdir -p "$state" 2>/dev/null || quiet_exit

output="$state/agent-check.$key.out"
# report <exit code>: the message for a finished run, read from $output.
report() {
  if [ "$1" = 124 ]; then
    local msg="Stop test gate: make tests did not finish within 180 s; run make check before handing off."
    if is_cursor; then
      jq -n --arg t "$msg" '{followup_message: $t}'
    else
      jq -n --arg t "$msg" '{systemMessage: $t}'
    fi
    exit 0
  fi
  local failures reason
  failures="$(grep -E '^(FAILED|ERROR) ' "$output" 2>/dev/null | head -n 20)"
  [ -z "$failures" ] && failures="$(tail -n 20 "$output" 2>/dev/null)"
  reason="make tests is red (the pytest half of make agent-check). Fix these before handing off:
$failures
Then run make check."
  if is_cursor; then
    jq -n --arg t "$reason" '{followup_message: $t}'
  else
    jq -n --arg r "$reason" '{decision: "block", reason: $r}'
  fi
  exit 0
}

now="$(date +%s)"
last_file="$state/stop-test-gate.$key.last"
if [ -f "$last_file" ]; then
  # "<start epoch> <exit code> <tree hash>"; only the start while a run is in progress.
  read -r last last_rc last_hash _ <"$last_file"
  last="$(printf '%s' "$last" | tr -dc '0-9')"
  last_rc="$(printf '%s' "$last_rc" | tr -dc '0-9')"
  if [ -n "$last" ] && [ $((now - last)) -lt 90 ]; then
    [ -z "$last_rc" ] || [ "$last_rc" = 0 ] && quiet_exit
    # Red and no edit since that run started: the saved result still holds.
    [ "$hash" = "$last_hash" ] && report "$last_rc"
  fi
fi
printf '%s\n' "$now" >"$last_file"

run_limited 180 make tests >"$output" 2>&1
rc=$?
elapsed=$(($(date +%s) - now))
[ "$elapsed" -gt 120 ] && printf 'WARNING: stop test gate: make tests took %s s (limit 180 s)\n' "$elapsed" >&2
printf '%s %s %s\n' "$now" "$rc" "$hash" >"$last_file"
if [ "$rc" = 0 ]; then
  printf '%s\n' "$hash" >"$ok_file"
  quiet_exit
fi
report "$rc"
