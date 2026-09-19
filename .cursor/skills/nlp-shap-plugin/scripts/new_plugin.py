#!/usr/bin/env python3
"""Scaffold a new nlp_shap plugin: stub module, entry point, failing test."""

import argparse
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]

_GROUPS: dict[str, tuple[str, str]] = {
    "estimators": ("estimation", "Estimator"),
    "estimands": ("estimation/estimands", "Aggregator"),
    "partitions": ("masking", "Partitioner"),
    "absence_policies": ("masking", "Policy"),
    "value_fns": ("value", "Value"),
    "normalizers": ("estimation", "Normalizer"),
    "backends": ("backends", "Backend"),
    "renderers": ("viz", "Renderer"),
}


def _pascal(name: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[_\-]+", name) if part)


def _snake(name: str) -> str:
    return re.sub(r"[^\w]+", "_", name).strip("_").lower()


def main() -> int:
    """Create stub plugin files and print next steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "group",
        choices=sorted(_GROUPS),
        help="entry-point group under nlp_shap.*",
    )
    parser.add_argument("name", help="plugin id / module stem (snake_case)")
    parser.add_argument(
        "--class-name",
        default="",
        help="override PascalCase class name",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned paths only",
    )
    args = parser.parse_args()

    stem = _snake(args.name)
    class_name = args.class_name or f"{_pascal(stem)}{_GROUPS[args.group][1]}"
    pkg_rel, _ = _GROUPS[args.group]
    src_dir = _ROOT / "src" / "nlp_shap" / Path(pkg_rel)

    if args.group == "backends":
        src_dir = src_dir / stem
        module_file = src_dir / "backend.py"
        entry_target = f"nlp_shap.backends.{stem}:{class_name}"
    else:
        module_file = src_dir / f"{stem}.py"
        dotted = f"nlp_shap.{pkg_rel.replace('/', '.')}.{stem}"
        entry_target = f"{dotted}:{class_name}"

    test_dir = _ROOT / "tests" / Path(pkg_rel)
    test_file = test_dir / f"test_{stem}.py"
    pyproject = _ROOT / "pyproject.toml"

    print(f"group={args.group} id={stem} class={class_name}")
    print(f"module={module_file.relative_to(_ROOT)}")
    print(f"test={test_file.relative_to(_ROOT)}")
    print(f"entry={stem} = {entry_target}")

    if args.dry_run:
        return 0

    if module_file.exists():
        print(f"error: already exists: {module_file}", file=sys.stderr)
        return 1

    src_dir.mkdir(parents=True, exist_ok=True)
    if args.group == "backends":
        (src_dir / "__init__.py").write_text(
            f'"""{class_name} backend package."""\n\n'
            f"from .backend import {class_name}\n\n"
            f'__all__ = ["{class_name}"]\n',
            encoding="utf-8",
        )
        module_file.write_text(
            f'"""{class_name} backend (scaffold — replace with real impl)."""\n\n'
            f"class {class_name}:\n"
            f'    """Scaffold backend — implement Protocol methods."""\n\n'
            f"    def __init__(self) -> None:\n"
            f'        """Initialize the scaffold backend."""\n'
            f"        raise NotImplementedError("
            f'"{class_name} is a scaffold; implement before use")\n',
            encoding="utf-8",
        )
    else:
        module_file.write_text(
            f'"""{class_name} plugin (scaffold — replace with real impl)."""\n\n'
            f"class {class_name}:\n"
            f'    """Scaffold plugin — implement the matching Protocol."""\n\n'
            f"    def __init__(self) -> None:\n"
            f'        """Initialize the scaffold plugin."""\n'
            f"        raise NotImplementedError("
            f'"{class_name} is a scaffold; implement before use")\n',
            encoding="utf-8",
        )

    test_dir.mkdir(parents=True, exist_ok=True)
    if not test_file.exists():
        import_path = entry_target.split(":")[0]
        test_file.write_text(
            f'"""Tests for {class_name} (scaffold — make this fail then pass)."""\n\n'
            f"import pytest\n\n"
            f"from {import_path} import {class_name}\n\n\n"
            f"def test_{stem}_is_constructible() -> None:\n"
            f'    """{class_name} constructs; replace with behavior."""\n'
            f"    with pytest.raises(NotImplementedError):\n"
            f"        {class_name}()\n",
            encoding="utf-8",
        )

    ep_header = f'[project.entry-points."nlp_shap.{args.group}"]'
    text = pyproject.read_text(encoding="utf-8")
    if ep_header not in text:
        print(f"error: missing entry-point group {ep_header}", file=sys.stderr)
        return 1
    group_body = text.split(ep_header, 1)[1].split("\n[", 1)[0]
    if re.search(rf"^{re.escape(stem)}\s*=", group_body, re.MULTILINE):
        print(f"warning: entry point {stem} may already exist", file=sys.stderr)
    else:
        lines = text.splitlines(keepends=True)
        out: list[str] = []
        inserted = False
        for line in lines:
            out.append(line)
            if not inserted and line.strip() == ep_header:
                out.append(f'{stem} = "{entry_target}"\n')
                inserted = True
        pyproject.write_text("".join(out), encoding="utf-8")

    print("next: implement Protocol methods, fix test, make agent-check")
    print("docs/notebook if user-facing — see examples.mdc / docs.mdc")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
