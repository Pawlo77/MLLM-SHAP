"""Every committed configuration in ``config/`` validates.

@see [[config/dev_api.yaml]]
"""

from pathlib import Path

import pytest

from mllm_shap.pipeline.config import explain_config_from_yaml

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@pytest.mark.parametrize(
    "path", sorted(_CONFIG_DIR.glob("*.yaml")), ids=lambda path: path.name
)
def test_dev_config_validates(path: Path) -> None:
    config = explain_config_from_yaml(path.read_text(encoding="utf-8"))
    assert config.backend.kind in path.stem
    assert config.explanation.archive.on_exists == "overwrite"


def test_config_dir_is_not_empty() -> None:
    assert len(list(_CONFIG_DIR.glob("*.yaml"))) == 3
