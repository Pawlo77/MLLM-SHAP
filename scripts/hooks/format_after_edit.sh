#!/usr/bin/env bash
# After an agent edit (Claude Code PostToolUse, Codex PostToolUse on apply_patch, Cursor
# afterFileEdit/afterTabFileEdit): strip AI attribution lines (scripts/strip_watermarks.py)
# from each edited file, then Ruff-fix and format each edited .py file. --force-exclude
# keeps pyproject.toml's extend-exclude in force for explicitly passed paths. Unused imports
# and variables (F401, F841) are never auto-fixed: an import added one edit before the edit
# that uses it must survive. When something changed, one line per file names the fixed rule
# codes and whether the file was reformatted. Fail-open: formatting never blocks the agent.
source "$(dirname "$0")/lib.sh"
command -v uv >/dev/null 2>&1 || exit 0
cd "$ROOT" || exit 0
# fix_one <path>: fix and format one file; print its change line, if any.
fix_one() {
  local f="$1" fixed before what="" marks
  case "$f" in /*) ;; *) f="$ROOT/$f" ;; esac
  [ -f "$f" ] || return 0
  marks="$(uv run --quiet python "$ROOT/scripts/strip_watermarks.py" "$f" </dev/null 2>/dev/null | wc -l | tr -d ' ')"
  [ "${marks:-0}" -gt 0 ] && what="removed $marks AI attribution line(s)"
  case "$f" in
    *.py) ;;
    *)
      [ -n "$what" ] && printf 'format_after_edit: %s: %s\n' "${f#"$ROOT"/}" "$what"
      return 0 ;;
  esac
  fixed="$(uv run --quiet ruff check --fix --show-fixes --extend-unfixable F401,F841 \
    --force-exclude "$f" </dev/null 2>/dev/null \
    | grep -E '^[[:space:]]+[0-9]+ ' | grep -Eo '[A-Z]+[0-9]+ \(' | tr -d ' (' | sort -u \
    | paste -sd, -)"
  before="$(git hash-object "$f" 2>/dev/null)"
  uv run --quiet ruff format --force-exclude --quiet "$f" </dev/null >/dev/null 2>&1 || true
  [ -n "$fixed" ] && what="${what:+$what; }ruff fixed $fixed"
  if [ "$(git hash-object "$f" 2>/dev/null)" != "$before" ]; then
    what="${what:+$what; }reformatted"
  fi
  [ -n "$what" ] && printf 'format_after_edit: %s: %s\n' "${f#"$ROOT"/}" "$what"
  return 0
}

changes="$(edited_file | while IFS= read -r f; do fix_one "$f"; done)"
[ -n "$changes" ] && emit_context PostToolUse "$changes"
exit 0
