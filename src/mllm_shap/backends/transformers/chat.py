"""Conversation snapshot helpers for Hugging Face chat templates.

@domain backends
@see [[knowledge/architecture/backends]]
@tested-by [[tests/backends/test_transformers_chat.py]]
"""

from typing import Any


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Render chat messages to a model prompt string."""
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    chat_template = getattr(tokenizer, "chat_template", None)
    if apply_template is not None and chat_template:
        try:
            rendered = apply_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except ValueError:
            rendered = None
        if isinstance(rendered, str):
            return rendered
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)
