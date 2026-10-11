"""Typed errors raised by mllm-shap pipeline components.

Every error derives from :class:`MllmShapError`, so ``except MllmShapError`` catches
anything the package raises on purpose. Each error also derives from the closest
built-in exception, so existing ``except ValueError`` or ``except LookupError``
handlers keep working.

@domain core
@see [[knowledge/architecture/core]]
"""


class MllmShapError(Exception):
    """Root of every error raised on purpose by mllm-shap."""


class PluginNotFoundError(LookupError, MllmShapError):
    """Raised when no plugin is registered under the requested group and name."""


class PluginConflictError(ValueError, MllmShapError):
    """Raised when a plugin name is registered twice without ``replace=True``."""


class PluginLoadError(ImportError, MllmShapError):
    """Raised when a packaging entry point cannot be imported."""

    def __init__(self, message: str, entry_point_name: str) -> None:
        """Store ``message`` and the failing entry point's name."""
        super().__init__(message)
        self.entry_point_name = entry_point_name
        """Name of the entry point whose ``load()`` failed."""


class ConfigError(ValueError, MllmShapError):
    """Raised when configuration values are invalid or inconsistent."""


class BackendUnavailableError(RuntimeError, MllmShapError):
    """Raised when a configured generative backend cannot be reached."""


class LogprobsUnavailableError(BackendUnavailableError):
    """Raised when a value function needs token logprobs a generation lacks."""


class ArchiveExistsError(FileExistsError, MllmShapError):
    """Raised when a run archive already exists and ``on_exists`` is ``"error"``."""


class ManifestVersionError(ValueError, MllmShapError):
    """Raised when a run manifest or archive uses an unsupported schema version."""


class ReanalyzeMismatchError(ValueError, MllmShapError):
    """Raised when reanalyze settings disagree with the archived run."""


class ArchiveIncompleteError(ValueError, MllmShapError):
    """Raised when reanalyze meets an archive of a run that did not finish."""


class ArchiveNotFoundError(FileNotFoundError, MllmShapError):
    """Raised when a run archive lacks its manifest or SQLite database."""


class ArchiveCorruptError(ValueError, MllmShapError):
    """Raised when a run archive or its manifest cannot be read or parsed."""
