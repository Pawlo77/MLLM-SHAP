"""Replay backend plugin.

@domain backends
@see [[knowledge/architecture/backends]]
@tested-by [[tests/backends/test_replay.py]]
"""

from .backend import ReplayBackend

__all__ = ["ReplayBackend"]
