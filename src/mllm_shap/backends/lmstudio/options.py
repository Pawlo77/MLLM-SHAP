"""Settings of the LM Studio backend read from ``BackendConfig.options``.

@domain backends
@see [[knowledge/architecture/backends]]
@tested-by [[tests/backends/test_lmstudio_options.py]]
"""

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from ...errors import ConfigError


class LmStudioOptions(BaseModel):
    """LM Studio settings accepted under ``backend.options``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hub_search: str | None = None
    """Repository search term used to download the model; defaults to ``model_id``."""

    quantization: str = "Q4_K_M"
    """GGUF quantization to download when the model is not already local."""

    auto_download: bool = True
    """Download missing models through the LM Studio repository API."""

    serialize_generate: bool = True
    """Serialize ``generate`` calls (needed for MLX models in LM Studio)."""

    audio: Literal["refuse", "transcript"] = "refuse"
    """What to do with audio parts, which the LM Studio SDK cannot send. ``refuse``
    raises ``ValueError``; ``transcript`` sends the text parts only."""


def parse_lmstudio_options(options: Mapping[str, object]) -> LmStudioOptions:
    """Validate ``backend.options`` for the LM Studio backend.

    Raises:
        ConfigError: If an option is unknown or has the wrong type.
    """
    try:
        return LmStudioOptions.model_validate(dict(options))
    except ValidationError as error:
        msg = f"invalid LM Studio backend options: {error}"
        raise ConfigError(msg) from error
