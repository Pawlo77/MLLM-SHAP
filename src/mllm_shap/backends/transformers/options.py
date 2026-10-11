"""Settings of the transformers backend read from ``BackendConfig.options``.

@domain backends
@see [[knowledge/architecture/backends]]
"""

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from ...errors import ConfigError

TorchDtype = Literal["auto", "float32", "float16", "bfloat16"]
"""Weight dtypes accepted by ``backend.options.dtype``."""


class TransformersOptions(BaseModel):
    """Transformers settings accepted under ``backend.options``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: str | None = None
    """Hub revision to load (branch, tag or commit hash); pin a commit hash for
    reproducible runs. ``None`` loads the default branch."""

    dtype: TorchDtype = "auto"
    """Weight dtype; ``auto`` picks bfloat16 on accelerators that support it,
    float16 on other accelerators and float32 on CPU."""


def parse_transformers_options(options: Mapping[str, object]) -> TransformersOptions:
    """Validate ``backend.options`` for the transformers backend.

    Raises:
        ConfigError: If an option is unknown or has the wrong value.
    """
    try:
        return TransformersOptions.model_validate(dict(options))
    except ValidationError as error:
        msg = f"invalid transformers backend options: {error}"
        raise ConfigError(msg) from error
