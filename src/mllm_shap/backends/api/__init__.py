"""OpenAI-compatible HTTP API backend plugin.

@domain backends
@see [[knowledge/architecture/backends]]
@tested-by [[tests/backends/test_api.py]]
"""

from .backend import ApiBackend

__all__ = ["ApiBackend"]
