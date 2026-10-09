#!/usr/bin/env bash
# Claude Code status line: "nlp-shap <branch> <dirty file count>".
root="$(cd "$(dirname "$0")/../.." && pwd)"
[ -t 0 ] || cat >/dev/null
branch="$(git -C "$root" branch --show-current 2>/dev/null)"
dirty="$(git -C "$root" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
printf 'nlp-shap %s %s\n' "${branch:-detached}" "${dirty:-0}"
