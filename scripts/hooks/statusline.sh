#!/usr/bin/env bash
# Claude Code status line: "<name> <branch> <dirty file count>" (name: [tool.agent-kit]).
root="$(cd "$(dirname "$0")/../.." && pwd)"
source "$root/scripts/hooks/kit_conf.sh"
[ -t 0 ] || cat >/dev/null
branch="$(git -C "$root" branch --show-current 2>/dev/null)"
dirty="$(git -C "$root" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
printf '%s %s %s\n' "$(ROOT="$root" kit_conf name "$(basename "$root")")" "${branch:-detached}" "${dirty:-0}"
