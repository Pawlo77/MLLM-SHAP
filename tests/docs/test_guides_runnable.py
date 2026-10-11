"""Every guide snippet with a ``guide-result`` runs and has a stored output.

@see [[docs/scripts/render_guide_outputs.py]]
"""

import re
from collections import Counter
from pathlib import Path

import pytest
import render_guide_outputs as renderer
import yaml

from mllm_shap import ExplainConfig, explain_config_from_yaml

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "docs" / "_static" / "guide_outputs"
_YAML_BLOCK = re.compile(
    r"^\.\. code-block:: yaml\n\n(?P<body>(?:   .*\n|\n)+)", re.MULTILINE
)
_PYTHON_FENCE = re.compile(r"```python\n(?P<body>.*?)```", re.DOTALL)


def _scans() -> list[renderer.PageScan]:
    return renderer.scan_docs()


def test_every_guide_result_follows_a_snippet() -> None:
    """No ``guide-result`` directive is detached from a python code block."""
    orphans = [
        f"{scan.page.name}:{orphan.line} {orphan.snippet_id}"
        for scan in _scans()
        for orphan in scan.orphans
    ]
    assert not orphans, orphans


def test_guide_result_ids_are_unique() -> None:
    """Each output file belongs to exactly one directive."""
    ids = Counter(
        result.snippet_id
        for scan in _scans()
        for snippet in scan.snippets
        for result in snippet.results
    )
    assert [name for name, count in ids.items() if count > 1] == []


def test_snippets_run_and_match_stored_outputs(tmp_path: Path) -> None:
    """Rendering succeeds and produces exactly the committed output files."""
    written = renderer.render(output_dir=tmp_path)
    rendered = sorted(path.name for path in written)
    stored = sorted(path.name for path in OUTPUTS.iterdir() if path.is_file())
    assert rendered == stored, "run docs/scripts/render_guide_outputs.py"
    for path in written:
        if path.suffix == ".txt":
            assert path.read_text(encoding="utf-8") == (OUTPUTS / path.name).read_text(
                encoding="utf-8"
            ), f"{path.name} is stale; run docs/scripts/render_guide_outputs.py"


@pytest.mark.parametrize("page", ["guides/config.rst", "guides/backends.rst"])
def test_full_yaml_examples_validate(page: str) -> None:
    """YAML blocks that contain a ``backend`` section are valid configurations."""
    text = (ROOT / "docs" / page).read_text(encoding="utf-8")
    checked = 0
    for match in _YAML_BLOCK.finditer(text):
        body = "\n".join(line[3:] for line in match.group("body").splitlines())
        data = yaml.safe_load(body)
        if isinstance(data, dict) and "backend" in data:
            assert isinstance(explain_config_from_yaml(body), ExplainConfig)
            checked += 1
    assert checked


def test_readme_quickstart_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The README quickstart executes and prints the efficiency check."""
    monkeypatch.chdir(tmp_path)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    code = _PYTHON_FENCE.search(readme)
    assert code is not None
    exec(compile(code.group("body"), "README.md", "exec"), {"__name__": "__readme__"})
