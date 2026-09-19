#!/usr/bin/env bash
# Deny pip/conda/pre-commit/force-push/bad commits; allow valid git/PR without ask.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if command -v uv >/dev/null 2>&1; then
  uv run --project "$ROOT" python "$ROOT/.cursor/hooks/deny_shell.py"
else
  # Fallback: may fail on system Python <3.10 (PEP604 unions).
  python3 "$ROOT/.cursor/hooks/deny_shell.py"
fi
exit 0
