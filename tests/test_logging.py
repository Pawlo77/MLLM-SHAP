"""Tests for mllm_shap logging bootstrap.

@see [[src/mllm_shap/_logging.py]]
"""

import json
import logging
import textwrap
from pathlib import Path

import pytest

from mllm_shap._logging import (
    ENV_LOGGING_CONFIG,
    bootstrap_logging,
    configure_logging,
    find_logging_config,
)


@pytest.fixture
def reset_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset bootstrap state and logging handlers between tests."""
    monkeypatch.setattr("mllm_shap._logging._bootstrapped", False)
    for name in ("mllm_shap", "mllm_shap.tests"):
        package_logger = logging.getLogger(name)
        package_logger.handlers.clear()
        package_logger.setLevel(logging.NOTSET)
        package_logger.propagate = True
    monkeypatch.delenv(ENV_LOGGING_CONFIG, raising=False)


def test_find_logging_config_from_env(
    reset_bootstrap: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MLLM_SHAP_LOGGING_CONFIG selects an explicit TOML file."""
    config_path = tmp_path / "custom.toml"
    config_path.write_text("[tool.logging]\nversion = 1\n", encoding="utf-8")
    monkeypatch.setenv(ENV_LOGGING_CONFIG, str(config_path))

    assert find_logging_config() == config_path


def test_find_logging_config_from_pyproject(
    reset_bootstrap: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without env override, pyproject.toml in cwd is used."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    assert find_logging_config() == Path.cwd() / "pyproject.toml"


def test_bootstrap_applies_toml_config(
    reset_bootstrap: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bootstrap_logging loads handlers from [tool.logging]."""
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text(
        textwrap.dedent(
            """
            [tool.logging]
            version = 1
            disable_existing_loggers = false

            [tool.logging.formatters.standard]
            format = "%(levelname)s %(message)s"

            [tool.logging.handlers.console]
            class = "logging.StreamHandler"
            formatter = "standard"
            stream = "ext://sys.stderr"

            [tool.logging.loggers.mllm_shap]
            level = "INFO"
            handlers = ["console"]
            propagate = false
            """
        ).strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_LOGGING_CONFIG, str(config_path))
    bootstrap_logging()

    package_logger = logging.getLogger("mllm_shap")
    assert package_logger.level == logging.INFO
    assert any(
        isinstance(handler, logging.StreamHandler)
        for handler in package_logger.handlers
    )


def test_bootstrap_uses_null_handler_without_config(
    reset_bootstrap: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing config file falls back to a null handler."""
    monkeypatch.chdir(tmp_path)
    bootstrap_logging()

    package_logger = logging.getLogger("mllm_shap")
    assert len(package_logger.handlers) == 1
    assert isinstance(package_logger.handlers[0], logging.NullHandler)


def test_bootstrap_uses_null_handler_without_logging_section(
    reset_bootstrap: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TOML without [tool.logging] falls back to a null handler."""
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text('[project]\nname = "example"\n', encoding="utf-8")
    monkeypatch.setenv(ENV_LOGGING_CONFIG, str(config_path))
    bootstrap_logging()

    package_logger = logging.getLogger("mllm_shap")
    assert len(package_logger.handlers) == 1
    assert isinstance(package_logger.handlers[0], logging.NullHandler)


def test_bootstrap_is_idempotent(
    reset_bootstrap: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated bootstrap calls must not reconfigure logging."""
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text(
        textwrap.dedent(
            """
            [tool.logging]
            version = 1
            disable_existing_loggers = false

            [tool.logging.formatters.standard]
            format = "%(message)s"

            [tool.logging.handlers.console]
            class = "logging.StreamHandler"
            formatter = "standard"
            stream = "ext://sys.stderr"

            [tool.logging.loggers.mllm_shap]
            level = "WARNING"
            handlers = ["console"]
            propagate = false
            """
        ).strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_LOGGING_CONFIG, str(config_path))
    bootstrap_logging()
    handler_count = len(logging.getLogger("mllm_shap").handlers)

    bootstrap_logging()

    assert len(logging.getLogger("mllm_shap").handlers) == handler_count


def test_import_does_not_read_pyproject(
    reset_bootstrap: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the env variable, bootstrap leaves the host's logging alone."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    bootstrap_logging()

    package_logger = logging.getLogger("mllm_shap")
    assert [type(handler) for handler in package_logger.handlers] == [
        logging.NullHandler
    ]


def test_configure_logging_reads_pyproject(
    reset_bootstrap: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """configure_logging() applies [tool.logging] from the nearest pyproject."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    configure_logging()

    package_logger = logging.getLogger("mllm_shap")
    assert package_logger.level == logging.WARNING
    assert any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.NullHandler)
        for handler in package_logger.handlers
    )


def test_configure_logging_level_installs_stderr_handler(
    reset_bootstrap: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """configure_logging(level=...) shows INFO events without any TOML file."""
    assert configure_logging(level="INFO") is None
    logging.getLogger("mllm_shap.pipeline").info("stage %s", "done")

    assert "INFO [mllm_shap.pipeline] stage done" in capsys.readouterr().err
    assert len(logging.getLogger("mllm_shap").handlers) == 1


def test_configure_logging_json_lines(
    reset_bootstrap: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """json=True emits one parseable JSON object per record."""
    configure_logging(level=logging.DEBUG, json=True)
    logging.getLogger("mllm_shap.runtime").debug("cache %d", 3)

    record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert record["level"] == "DEBUG"
    assert record["logger"] == "mllm_shap.runtime"
    assert record["message"] == "cache 3"


def test_configure_logging_level_overrides_toml(
    reset_bootstrap: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A level passed with a TOML path overrides the file's level."""
    root = Path(__file__).resolve().parents[1]
    configure_logging(root / "pyproject.toml", level="DEBUG")

    assert logging.getLogger("mllm_shap").level == logging.DEBUG


def test_example_logging_config_applies(reset_bootstrap: None) -> None:
    """config/logging.example.toml is a valid [tool.logging] file at INFO."""
    path = Path(__file__).resolve().parents[1] / "config" / "logging.example.toml"

    assert configure_logging(path) == path
    assert logging.getLogger("mllm_shap").level == logging.INFO


def test_configure_logging_rejects_json_with_config_path(
    reset_bootstrap: None,
) -> None:
    """json=True only applies to the stderr handler, never to a TOML file."""
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="json"):
        configure_logging(root / "pyproject.toml", json=True)


def test_configure_logging_stderr_handler_does_not_propagate(
    reset_bootstrap: None,
) -> None:
    configure_logging(level="INFO")
    assert logging.getLogger("mllm_shap").propagate is False
