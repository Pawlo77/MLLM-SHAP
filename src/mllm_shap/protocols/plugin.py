"""Options contract for plugins configured through ``explanation.plugin_options``.

@domain protocols
@see [[knowledge/architecture/protocols]]
"""

from typing import ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class PluginOptions(BaseModel):
    """Options a plugin accepts; the base class accepts none.

    Subclasses declare one field per option. Unknown keys and values of the
    wrong type are rejected (``extra="forbid"``, strict types; integers are
    accepted for float fields).
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


@runtime_checkable
class DeclaresPluginOptions(Protocol):
    """Plugin class that declares the options it accepts.

    The registry validates ``explanation.plugin_options`` for the plugin against
    :attr:`options_model` before building it. Plugins without the attribute get
    their options unchecked.
    """

    options_model: ClassVar[type[PluginOptions]]
    """Pydantic model listing the accepted options."""
