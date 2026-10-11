Masking views
=============

This guide partitions a conversation into token or message players, renders the
prompt a coalition sees under each absence policy, and resolves both from
configuration.
The snippets share one snapshot.

Partition tokens
----------------

.. code-block:: python

   from nlp_shap import ConversationSnapshot, Message, Role, Turn
   from nlp_shap.masking import TokenPartitioner

   turn = Turn(messages=(Message(role=Role.USER, text="Who are you?"),))
   snapshot = ConversationSnapshot(turns=(turn,), snapshot_id="demo-snap")

   players = TokenPartitioner().partition(snapshot)
   print(players.player_ids)

.. guide-result:: masking_partition

Player ids are derived from the snapshot id and the token position, so they are
stable across runs and appear in the run manifest in mask-bit order.

Render a coalition
------------------

A :class:`~nlp_shap.CoalitionMask` marks which players are present. The
:class:`~nlp_shap.masking.MaskBuilder` pairs it with the snapshot (a cheap view
that shares the snapshot) and renders the prompt the backend receives:

.. code-block:: python

   from nlp_shap import CoalitionMask
   from nlp_shap.masking import DeletePolicy, MaskBuilder

   mask = CoalitionMask.from_sequence((True, False, True))
   builder = MaskBuilder(DeletePolicy())
   view = builder.view(snapshot, players, mask)
   rendered = builder.render(view)

   print(rendered.turns[0].messages[0].text)

.. guide-result:: masking_render_coalition

Compare absence policies
------------------------

For audio players (``sgpa_segments``, :doc:`alignment`) the policies also edit the
audio: ``delete`` cuts absent samples, ``pad`` and ``neutral`` silence them.

``delete`` drops absent tokens, ``pad`` replaces each with a placeholder and
``neutral`` with a width-matched filler. The same coalition under the other two:

.. code-block:: python

   from nlp_shap.masking import NeutralPolicy, PadPolicy

   for policy in (PadPolicy(), NeutralPolicy()):
       builder = MaskBuilder(policy)
       rendered = builder.render(builder.view(snapshot, players, mask))
       print(f"{type(policy).__name__}: {rendered.turns[0].messages[0].text}")

.. guide-result:: masking_absence_policies

:doc:`../theory/masking` discusses how the choice affects what the model sees.

Absence policies read the player spans the partitioner returned
(``PlayerSet.spans``) and never re-tokenise the prompt. Text outside every span is
kept as is, and ``delete`` removes a player together with the whitespace that
separated it from its neighbours (``"a b c"`` without ``b`` renders ``"a c"``).
Rendered messages keep their original whitespace.

Message players
---------------

``explanation.players: messages``
(:class:`~nlp_shap.masking.MessagePartitioner`) makes each non-empty message one
player whose span is the whole message text. Its label reads
``"<role>[/<speaker>] <n>: "`` plus the first 30 characters, with ``n`` the
message's 1-based position. ``delete`` empties an absent message, ``pad`` and
``neutral`` replace its text. The plugin option ``roles`` (default every role)
keeps the messages of other roles, such as the system prompt, always present;
``players: tokens`` takes the same option to explain only the words of the
listed roles:

.. code-block:: python

   from nlp_shap import ExplainConfig
   from nlp_shap.masking import MessagePartitioner
   from nlp_shap.plugins import PluginContext

   chat = ConversationSnapshot.from_turns((
       Turn(
           messages=(
               Message(role=Role.SYSTEM, text="Answer briefly."),
               Message(role=Role.USER, text="Is the sky blue?", speaker="ana"),
               Message(role=Role.USER, text="Why?", speaker="ben"),
           )
       ),
   ))
   config = ExplainConfig.model_validate({
       "backend": {"kind": "mock", "model_id": "stub"},
       "explanation": {
           "players": "messages",
           "plugin_options": {"messages": {"roles": ["user"]}},
       },
   })
   options = config.explanation.plugin_options["messages"]
   chat_players = MessagePartitioner(
       PluginContext(config=config, options=options)
   ).partition(chat)
   for span in chat_players.spans:
       print(span.display_label)
   view = MaskBuilder(DeletePolicy()).view(
       chat, chat_players, CoalitionMask.from_sequence((False, True))
   )
   print([m.text for m in MaskBuilder(DeletePolicy()).render(view).messages()])

.. guide-result:: masking_messages

Resolve plugins from config
---------------------------

Built-in plugins register the ``tokens`` and ``messages`` partitions and the ``delete`` / ``pad`` /
``neutral`` absence policies. :class:`~nlp_shap.ExplainRunner` resolves them from
``explanation.players`` and ``explanation.absence_policy``:

.. code-block:: python

   from nlp_shap import ExplainConfig, PluginGroup, PluginRegistry
   from nlp_shap.plugins import PluginContext, register_builtin_plugins

   registry = PluginRegistry()
   register_builtin_plugins(registry)
   config = ExplainConfig.model_validate({
       "backend": {"kind": "mock", "model_id": "stub"},
       "explanation": {"players": "tokens", "absence_policy": "pad"},
   })
   context = PluginContext(config=config, registry=registry)

   partitioner = registry.resolve(
       PluginGroup.PARTITIONS, config.explanation.players, context
   )
   policy = registry.resolve(
       PluginGroup.ABSENCE_POLICIES, config.explanation.absence_policy, context
   )
   print(type(partitioner).__name__)
   print(type(policy).__name__)

.. guide-result:: masking_plugins

See :doc:`config` for the full configuration schema and :doc:`extending` for
registering your own partitioner or policy.

Notebook
--------

For a runnable walkthrough with stored outputs, see
``examples/masking_views.ipynb`` (also embedded on :doc:`../examples`).
