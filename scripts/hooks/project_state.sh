# Project extension of session_start.sh (not template-managed): extra banner lines and memory.
project_memory() {
  if [ -f scripts/build_knowledge.py ] && command -v uv >/dev/null 2>&1; then
    local issues
    issues="$(uv run --quiet python scripts/build_knowledge.py --open-issues 2>/dev/null)"
    [ -n "$issues" ] && mem_item "Open issues" "knowledge/issues/" 4096 "$issues"
  fi
}

project_state() {
  local version sib missing=""
  version="$(sed -n -E 's/^version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' pyproject.toml 2>/dev/null | head -n 1)"
  echo "- package: mllm-shap ${version:-unknown}"
  for sib in MLLM-Shap-legacy; do
    [ -d "../$sib" ] || missing="$missing $sib"
  done
  if [ -n "$missing" ]; then
    echo "- siblings missing:$missing (clone next to MLLM-SHAP/; see skill agent-setup)"
  else
    echo "- sibling: ../MLLM-Shap-legacy (legacy port)"
  fi
  if [ -f scripts/build_knowledge.py ] && command -v uv >/dev/null 2>&1; then
    echo "- open issues: $(uv run --quiet python scripts/build_knowledge.py --open-issues 2>/dev/null | grep -c .)"
  fi
  if grep -q '^graph:' Makefile 2>/dev/null; then
    echo "- find existing code: make graph ARGS=--codemap; before changing a module or note: make graph ARGS=\"--neighbors <path>\""
  fi
}
