"""Central logging bootstrap for nlp_shap.

@domain core
@see [[knowledge/architecture/core]]
@tested-by [[tests/test_logging.py]]
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Final

import logging518.config

PACKAGE_NAME: Final[str] = "nlp_shap"
"""Root logger namespace for the package."""

ENV_LOGGING_CONFIG: Final[str] = "NLP_SHAP_LOGGING_CONFIG"
"""Environment variable pointing at a TOML file with a ``[tool.logging]`` section."""

PYPROJECT_FILENAME: Final[str] = "pyproject.toml"
"""Default TOML filename searched for logging configuration."""

STANDARD_FORMAT: Final[str] = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
"""Line format of the handler installed by ``configure_logging(level=...)``."""

_bootstrapped: bool = False
"""Whether :func:`bootstrap_logging` has run in this process."""


class JsonLogFormatter(logging.Formatter):
    """Format each record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Return ``time``, ``level``, ``logger``, ``message`` and ``exc_info``."""
        payload: dict[str, str] = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _ensure_null_handler() -> None:
    package_logger = logging.getLogger(PACKAGE_NAME)
    if not package_logger.handlers:
        package_logger.addHandler(logging.NullHandler())


def find_logging_config() -> Path | None:
    """Return the TOML file that defines ``[tool.logging]``, if any."""
    env_path = os.getenv(ENV_LOGGING_CONFIG)
    if env_path:
        path = Path(env_path)
        return path if path.is_file() else None

    current = Path.cwd()
    for directory in (current, *current.parents):
        candidate = directory / PYPROJECT_FILENAME
        if candidate.is_file():
            return candidate
    return None


def bootstrap_logging() -> None:
    """Keep the package quiet on import unless ``NLP_SHAP_LOGGING_CONFIG`` is set.

    Runs once per process. With the environment variable pointing at a TOML file,
    its ``[tool.logging]`` section is applied through ``logging518``; otherwise
    only a ``NullHandler`` is installed and the host application's logging is left
    untouched. Call :func:`configure_logging` to opt in explicitly.
    """
    global _bootstrapped
    if _bootstrapped:
        return
    _bootstrapped = True

    env_path = os.getenv(ENV_LOGGING_CONFIG)
    if env_path and Path(env_path).is_file():
        _apply_config(Path(env_path))
    else:
        _ensure_null_handler()


def configure_logging(
    config_path: Path | str | None = None,
    level: int | str | None = None,
    json: bool = False,
) -> Path | None:
    """Configure the ``nlp_shap`` loggers from TOML or with one stderr handler.

    Without ``level`` and ``json`` the ``[tool.logging]`` section of a TOML file
    is applied. With either of them and no ``config_path``, the package logger
    gets a single stderr handler instead (replacing earlier handlers) and stops
    propagating to the root logger (``propagate=False``), so records are not
    printed twice by a root handler; ``configure_logging(level="INFO")`` shows
    the pipeline's stage events.

    Args:
        config_path: TOML file to read. Defaults to ``NLP_SHAP_LOGGING_CONFIG`` or
            the nearest ``pyproject.toml`` above the working directory.
        level: Level of the ``nlp_shap`` logger (``"INFO"``, ``logging.DEBUG``...);
            overrides the TOML level when both are given. Defaults to ``INFO``
            for the stderr handler.
        json: Emit one JSON object per record (``time``, ``level``, ``logger``,
            ``message``) instead of plain lines. Only for the stderr handler.

    Returns:
        The TOML file that was read, or ``None`` when none was (a ``NullHandler``
        is installed when neither a file nor ``level``/``json`` was given).

    Raises:
        ValueError: If ``json`` is combined with ``config_path``; the TOML file
            defines its own formatters.
    """
    if json and config_path is not None:
        msg = "json=True cannot be combined with config_path; set formatters in TOML"
        raise ValueError(msg)
    package_logger = logging.getLogger(PACKAGE_NAME)
    if config_path is None and (level is not None or json):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            JsonLogFormatter() if json else logging.Formatter(STANDARD_FORMAT)
        )
        package_logger.handlers.clear()
        package_logger.addHandler(handler)
        package_logger.setLevel(level if level is not None else logging.INFO)
        package_logger.propagate = False
        return None
    path = Path(config_path) if config_path is not None else find_logging_config()
    if path is None:
        _ensure_null_handler()
        return None
    _apply_config(path)
    if level is not None:
        package_logger.setLevel(level)
    return path


def _apply_config(path: Path) -> None:
    try:
        logging518.config.fileConfig(path)
    except KeyError:
        _ensure_null_handler()
