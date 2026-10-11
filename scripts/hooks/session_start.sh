#!/usr/bin/env bash
# Session start (Claude Code SessionStart, Cursor sessionStart): project state, so a new agent
# session starts with context. Cursor has no memory plugin, so it also gets the committed
# .remember/ memory and the open-issue list inline; Claude Code and Codex load those through the
# remember plugin.
source "$(dirname "$0")/lib.sh"
source "$(dirname "$0")/gate_lib.sh"
cd "$ROOT" || exit 0
sid="$(printf '%s' "$HOOK_INPUT" | jq -r '.session_id // .conversation_id // empty' 2>/dev/null \
  | tr -cd 'A-Za-z0-9_-' | cut -c1-64)"
# Per-session gate state (stop_test_gate.sh): drop this session's own state and any older
# than a day, then seed its tests-ok tree hash so the Stop test gate acts only once src/
# or tests/ change after this point.
if [ -d .remember/tmp ]; then
  [ -n "$sid" ] && rm -f ".remember/tmp/stop-test-gate.$sid.last" ".remember/tmp/agent-check.$sid.out"
  find .remember/tmp -maxdepth 1 -type f \( -name 'stop-test-gate.*' -o -name 'tests-ok.*' \
    -o -name 'agent-check.*' \) -mmin +1440 -exec rm -f {} + 2>/dev/null
fi
if [ -n "$sid" ] && [ -d .remember ] && [ ! -f ".remember/tmp/tests-ok.$sid" ]; then
  mkdir -p .remember/tmp 2>/dev/null && tree_hash >".remember/tmp/tests-ok.$sid"
fi

# Total bytes of memory inlined for Cursor. The remember plugin's memory_inject_max_bytes
# defaults to 200000; Cursor pays for this text on every session start, so it stays small.
MEM_CAP=12288
mem_used=0

nbytes() { printf '%s' "$1" | wc -c | tr -d ' '; }

# Inline one memory item, or its path and size when it exceeds its own cap or the total cap.
mem_item() { # <title> <path> <item cap bytes> <text>
  local size fsize
  size="$(nbytes "$4")"
  [ "$size" -eq 0 ] && return 0
  if [ "$size" -gt "$3" ] || [ $((mem_used + size)) -gt "$MEM_CAP" ]; then
    fsize="$size"
    [ -f "$2" ] && fsize="$(wc -c <"$2" | tr -d ' ')"
    echo "- $1: $2 is $fsize bytes, over the inline cap; read it directly."
    return 0
  fi
  mem_used=$((mem_used + size))
  printf '\n### %s (%s)\n%s\n' "$1" "$2" "$4"
}

memory() {
  echo
  echo "## Memory and context"
  local f
  f=.remember/remember.md
  [ -f "$f" ] && mem_item "Handoff" "$f" 4096 "$(cat "$f")"
  f=.remember/identity.md
  [ -f "$f" ] && mem_item "Identity" "$f" 4096 "$(cat "$f")"
  f=.remember/recent.md
  [ -f "$f" ] && mem_item "Recent (last 60 lines)" "$f" 6144 "$(tail -n 60 "$f")"
  f="$(ls .remember/today-*.md 2>/dev/null | sort | tail -n 1)"
  [ -n "$f" ] && mem_item "Today (last 40 lines)" "$f" 4096 "$(tail -n 40 "$f")"
  if [ -f scripts/build_knowledge.py ] && command -v uv >/dev/null 2>&1; then
    local issues
    issues="$(uv run --quiet python scripts/build_knowledge.py --open-issues 2>/dev/null)"
    [ -n "$issues" ] && mem_item "Open issues" "knowledge/issues/" 4096 "$issues"
  fi
  return 0
}

state() {
  local version changes
  version="$(sed -n -E 's/^version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' pyproject.toml 2>/dev/null | head -n 1)"
  echo "## mllm-shap session state"
  echo "- package: mllm-shap ${version:-unknown}; branch: $(git branch --show-current 2>/dev/null)"
  changes="$(git status --short 2>/dev/null)"
  if [ -n "$changes" ]; then
    echo "- uncommitted changes ($(printf '%s\n' "$changes" | wc -l | tr -d ' ') files; first 5):"
    printf '%s\n' "$changes" | head -n 5 | sed 's/^/    /'
  else
    echo "- working tree clean"
  fi
  echo "- recent commits:"
  git log --oneline -5 2>/dev/null | sed 's/^/    /'
  echo "- exit criterion: make check (filtered pytest + prek-all + vault-check)"
  local sib missing=""
  for sib in MLLM-Shap-legacy; do
    [ -d "../$sib" ] || missing="$missing $sib"
  done
  if [ -n "$missing" ]; then
    echo "- siblings missing:$missing (clone next to MLLM-SHAP/; see skill agent-setup)"
  else
    echo "- sibling: ../MLLM-Shap-legacy (legacy port)"
  fi
  if [ -f scripts/check_vault.py ] && command -v uv >/dev/null 2>&1; then
    local vault rc
    vault="$(uv run --quiet python scripts/check_vault.py --quiet --strict-tags 2>&1)"
    rc=$?
    vault="$(printf '%s\n' "$vault" | tail -n 1)"
    echo "- vault check: ${vault:-ok}"
  fi
  if [ -f scripts/build_knowledge.py ] && command -v uv >/dev/null 2>&1; then
    echo "- open issues: $(uv run --quiet python scripts/build_knowledge.py --open-issues 2>/dev/null | grep -c .)"
  fi
  if grep -q '^graph:' Makefile 2>/dev/null; then
    echo "- find existing code: make graph ARGS=--codemap; before changing a module or note: make graph ARGS=\"--neighbors <path>\""
    echo "- session key (skill ship: uv run python scripts/pr_policy.py --write-marker --session <key>): $(session_key)"
  fi
  is_cursor && memory
}

emit_context SessionStart "$(state)"
exit 0
