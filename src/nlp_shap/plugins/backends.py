"""Backend plugin construction helpers.

@domain plugins
@see [[knowledge/architecture/plugins]]
@tested-by [[tests/plugins/test_backends.py]]
"""

from typing import cast

from ..protocols.backend import GenerativeBackend
from .context import PluginContext
from .groups import PluginGroup


def instantiate_backend(context: PluginContext) -> GenerativeBackend:
    """Build the backend named by ``context.config.backend.kind``."""
    return cast(
        GenerativeBackend,
        context.registry.resolve(
            PluginGroup.BACKENDS, context.config.backend.kind, context
        ),
    )
