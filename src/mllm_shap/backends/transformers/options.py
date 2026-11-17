"""Settings of the transformers backend read from ``BackendConfig.options``.

@domain backends
@see [[knowledge/architecture/backends]]
"""

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

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

    model_class: str | None = None
    """``transformers`` class to load with the checkpoint's processor, for example
    ``Gemma3nForConditionalGeneration``. ``None`` uses ``AutoModelForCausalLM`` when
    it knows the checkpoint, else the checkpoint's architecture."""

    batch_size: int = Field(default=1, ge=1, strict=True)
    """Teacher-forced sequences scored in one padded forward pass. Concurrent
    ``score_continuations`` calls are collected up to this many sequences; ``1`` scores
    each prompt on its own. The scheduler runs at least this many coalitions at once."""

    batch_wait_ms: float = Field(default=5.0, ge=0.0)
    """Longest wait for a batch to fill before a smaller one runs."""

    attn_implementation: str | None = None
    """Attention kernel passed to ``from_pretrained`` (``eager``, ``sdpa``, ...);
    ``None`` keeps the model's default. Granite Speech on Apple MPS needs
    ``eager``: its projector fails under the default kernel."""


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
