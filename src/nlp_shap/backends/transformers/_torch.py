"""Lazy import of the optional ``torch`` dependency.

@domain backends
@see [[knowledge/architecture/backends]]
"""

from typing import Any


def import_torch() -> Any:
    """Return the ``torch`` module.

    Raises:
        ImportError: If the ``transformers`` extra is not installed.
    """
    try:
        import torch
    except ImportError as exc:
        msg = "torch is required for the transformers backend extra"
        raise ImportError(msg) from exc
    return torch
