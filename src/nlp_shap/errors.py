"""Typed errors raised by nlp-shap pipeline components.

Every error derives from :class:`NlpShapError`, so ``except NlpShapError`` catches
anything the package raises on purpose. Each error also derives from the closest
built-in exception, so existing ``except ValueError`` or ``except LookupError``
handlers keep working.

@domain core
@see [[knowledge/architecture/core]]
"""


class NlpShapError(Exception):
    """Root of every error raised on purpose by nlp-shap."""


class PluginNotFoundError(LookupError, NlpShapError):
    """Raised when no plugin is registered under the requested group and name."""


class PluginConflictError(ValueError, NlpShapError):
    """Raised when a plugin name is registered twice without ``replace=True``."""


class PluginLoadError(ImportError, NlpShapError):
    """Raised when a packaging entry point cannot be imported."""

    def __init__(self, message: str, entry_point_name: str) -> None:
        """Store ``message`` and the failing entry point's name."""
        super().__init__(message)
        self.entry_point_name = entry_point_name
        """Name of the entry point whose ``load()`` failed."""


class ConfigError(ValueError, NlpShapError):
    """Raised when configuration values are invalid or inconsistent."""


class BackendUnavailableError(RuntimeError, NlpShapError):
    """Raised when a configured generative backend cannot be reached."""


class LogprobsUnavailableError(BackendUnavailableError):
    """Raised when a value function needs token logprobs a generation lacks."""


class ArchiveExistsError(FileExistsError, NlpShapError):
    """Raised when a run archive already exists and ``on_exists`` is ``"error"``."""


class ManifestVersionError(ValueError, NlpShapError):
    """Raised when a run manifest or archive uses an unsupported schema version."""


class ReanalyzeMismatchError(ValueError, NlpShapError):
    """Raised when reanalyze settings disagree with the archived run."""


class ArchiveIncompleteError(ValueError, NlpShapError):
    """Raised when reanalyze meets an archive of a run that did not finish."""


class ArchiveNotFoundError(FileNotFoundError, NlpShapError):
    """Raised when a run archive lacks its manifest or SQLite database."""


class ArchiveCorruptError(ValueError, NlpShapError):
    """Raised when a run archive or its manifest cannot be read or parsed."""
