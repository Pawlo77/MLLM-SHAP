#!/usr/bin/env bash
# Fail-open: format/lint edited Python files with ruff via uv.
set -u
input=$(cat)
file=$(printf '%s' "$input" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("file_path") or "")' 2>/dev/null || true)
case "$file" in
  *.py) ;;
  *) exit 0 ;;
esac
if [[ ! -f "$file" ]]; then
  exit 0
fi
if command -v uv >/dev/null 2>&1; then
  uv run ruff check --fix --quiet "$file" >/dev/null 2>&1 || true
  uv run ruff format --quiet "$file" >/dev/null 2>&1 || true
fi
exit 0
