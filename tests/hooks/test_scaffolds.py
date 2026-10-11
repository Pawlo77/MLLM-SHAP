"""Dry-run smoke for Cursor scaffold scripts.

@see [[knowledge/handbook/Onboarding]]
"""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from mllm_shap.pipeline.config import BackendConfig, ExplainConfig
from mllm_shap.plugins import PluginContext, PluginGroup, PluginRegistry

_ROOT = Path(__file__).resolve().parents[2]
_PLUGIN_SCRIPT = Path(".agents/skills/plugin/scripts/new_plugin.py")


def test_new_plugin_dry_run() -> None:
    """Plugin scaffold dry-run exits 0 and prints planned paths."""
    proc = subprocess.run(
        [
            sys.executable,
            str(_ROOT / _PLUGIN_SCRIPT),
            "value_fns",
            "scaffold_probe",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "scaffold_probe" in proc.stdout
    assert "module=" in proc.stdout


def _load_module(name: str, path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_new_plugin_generates_context_constructor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generated plugin takes a PluginContext and resolves through the registry."""
    script = tmp_path / _PLUGIN_SCRIPT
    script.parent.mkdir(parents=True)
    shutil.copy(_ROOT / _PLUGIN_SCRIPT, script)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "probe"\n\n[project.entry-points."mllm_shap.value_fns"]\n',
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(script), "value_fns", "scaffold_probe"],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr

    module_path = tmp_path / "src" / "mllm_shap" / "value" / "scaffold_probe.py"
    source = module_path.read_text(encoding="utf-8")
    assert (
        'def __init__(self, context: "PluginContext | None" = None) -> None:' in source
    )
    assert "    from ..plugins.context import PluginContext\n" in source
    assert "@domain value\n" in source
    assert "@tested-by [[tests/value/test_scaffold_probe.py]]" in source
    pyproject = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert 'scaffold_probe = "mllm_shap.value.scaffold_probe:ScaffoldProbeValue"' in (
        pyproject
    )

    module = _load_module("mllm_shap.value.scaffold_probe", module_path, monkeypatch)
    registry = PluginRegistry()
    registry.register(
        PluginGroup.VALUE_FNS, "scaffold_probe", module.ScaffoldProbeValue
    )
    config = ExplainConfig(backend=BackendConfig(kind="mock", model_id="stub"))
    plugin = registry.resolve(
        PluginGroup.VALUE_FNS,
        "scaffold_probe",
        PluginContext(config=config, registry=registry),
    )
    assert isinstance(plugin, module.ScaffoldProbeValue)

    generated_test = _load_module(
        "scaffold_probe_generated_test",
        tmp_path / "tests" / "value" / "test_scaffold_probe.py",
        monkeypatch,
    )
    generated_test.test_scaffold_probe_resolves_with_context()
    with pytest.raises(pytest.fail.Exception):
        generated_test.test_scaffold_probe_behaviour()


def test_new_plugin_backend_imports_context_from_package_root(tmp_path: Path) -> None:
    """A backend sits one package deeper: its PluginContext import has 3 dots."""
    script = tmp_path / _PLUGIN_SCRIPT
    script.parent.mkdir(parents=True)
    shutil.copy(_ROOT / _PLUGIN_SCRIPT, script)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "probe"\n\n[project.entry-points."mllm_shap.backends"]\n',
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(script), "backends", "scaffold_probe"],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    backend = (
        tmp_path / "src" / "mllm_shap" / "backends" / "scaffold_probe" / "backend.py"
    )
    assert "    from ...plugins.context import PluginContext\n" in backend.read_text(
        encoding="utf-8"
    )


def test_new_notebook_dry_run() -> None:
    """Notebook scaffold dry-run exits 0 and prints catalog reminder."""
    script = _ROOT / ".agents" / "skills" / "notebook" / "scripts" / "new_notebook.py"
    proc = subprocess.run(
        [sys.executable, str(script), "scaffold_probe", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        cwd=_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "scaffold_probe.ipynb" in proc.stdout
    assert "catalog:" in proc.stdout
