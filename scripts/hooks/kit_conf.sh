# Reads string values from [tool.agent-kit] in pyproject.toml: kit_conf <key> [default].
# Plain awk, because the system python3 may predate tomllib.
kit_conf() {
  local root value
  root="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
  value="$(awk -v key="$1" '
    /^\[/ { in_table = ($0 == "[tool.agent-kit]"); next }
    in_table && $1 == key && $2 == "=" {
      sub(/^[^=]*=[ \t]*"/, ""); sub(/"[ \t]*(#.*)?$/, ""); print; exit
    }' "$root/pyproject.toml" 2>/dev/null)"
  printf '%s' "${value:-${2:-}}"
}
