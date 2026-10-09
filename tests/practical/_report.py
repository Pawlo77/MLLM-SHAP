"""Print ``results/latest.json`` as Markdown: scenarios, then the deep tables.

Usage: ``uv run python -m tests.practical._report [results file]``.
"""

import json
import sys
from pathlib import Path
from typing import Any

LATEST = Path(__file__).parent / "results" / "latest.json"
"""Results file of the last practical run."""


def render(payload: dict[str, Any]) -> str:
    """Return the scenario table, the error-versus-budget tables and the run line."""
    lines = ["| feature | result | seconds | VRAM GB |", "|---|---|---|---|"]
    tables: list[str] = []
    for entry in payload["scenarios"].values():
        vram = entry.get("driver_gb", entry.get("vram_gb", ""))
        lines.append(
            f"| {entry['feature']} | {entry['outcome']} | {entry['seconds']} | {vram} |"
        )
        errors = entry.get("tables", {}).get("error_vs_budget", {})
        for method, by_budget in errors.items():
            for budget, cell in by_budget.items():
                tables.append(
                    f"| {entry['feature']} | {method} | {budget} | {cell['median']} "
                    f"| {cell['per_seed']} |"
                )
    if tables:
        lines += ["", "| feature | method | budget | median L1 | per seed |"]
        lines += ["|---|---|---|---|---|", *tables]
    run = (
        f"tier {payload.get('tier')}, device {payload['device']}, wall "
        f"{payload['wall_seconds']} s, peak VRAM {payload.get('peak_vram_gb')} GB, "
        f"cache {payload['cache']}"
    )
    lines += ["", run]
    return "\n".join(lines)


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else LATEST
    print(render(json.loads(path.read_text("utf-8"))))
