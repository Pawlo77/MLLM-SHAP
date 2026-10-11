"""Backend plugin implementations.

@domain backends
@see [[knowledge/architecture/backends]]
"""

from .mock import MockBackend

__all__ = ["MockBackend"]
