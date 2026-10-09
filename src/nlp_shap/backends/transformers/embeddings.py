"""Embedding helpers for the transformers text backend.

@domain backends
@see [[knowledge/architecture/backends]]
"""

from typing import Any

from ._torch import import_torch


def static_embedding(
    model: Any,
    token_ids: list[int],
) -> tuple[float, ...]:
    """Mean-pool input embeddings for generated token ids."""
    torch = import_torch()
    if not token_ids:
        return ()
    embedding_layer = model.get_input_embeddings()
    ids = torch.tensor([token_ids], dtype=torch.long, device=_model_device(model))
    with torch.no_grad():
        vectors = embedding_layer(ids).squeeze(0).mean(dim=0)
    return tuple(float(value) for value in vectors.cpu().tolist())


def contextual_embedding(
    model: Any,
    token_ids: list[int],
) -> tuple[float, ...]:
    """Mean-pool the final hidden state for generated token ids."""
    torch = import_torch()
    if not token_ids:
        return ()
    ids = torch.tensor([token_ids], dtype=torch.long, device=_model_device(model))
    with torch.no_grad():
        outputs = model(ids, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[-1].squeeze(0).mean(dim=0)
    return tuple(float(value) for value in hidden.cpu().tolist())


def _model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        torch = import_torch()
        return torch.device("cpu")
