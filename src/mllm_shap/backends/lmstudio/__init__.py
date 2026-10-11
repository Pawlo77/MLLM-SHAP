"""LM Studio backend plugin.

@domain backends
@see [[knowledge/architecture/backends]]
@tested-by [[tests/backends/test_lmstudio.py]]
"""

from .backend import LmStudioBackend

__all__ = ["LmStudioBackend"]
