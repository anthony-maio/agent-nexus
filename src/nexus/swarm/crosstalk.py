"""Organic collaboration system for the Agent Nexus swarm.

Instead of random probabilistic crosstalk, models take turns reacting to
what was *just said*.  Each model sees the full conversation (including
previous reactions in this round) and genuinely decides whether it has
something to add.  If not, it responds with ``[PASS]`` and the round
moves on.

This creates natural back-and-forth: sometimes one model replies,
sometimes all three pile on, sometimes nobody has anything to add.
"""

from __future__ import annotations

import logging
import random

log = logging.getLogger(__name__)

# Sentinel the model returns when it has nothing to add.
PASS_TOKEN = "[PASS]"

# Injected after the model's normal system prompt during the reaction round.
REACTION_PROMPT = (
    "\n\n## Reaction Round\n"
    "Another model just responded to the human. You can see what they said above.\n"
    "If you have something *genuinely new* to contribute — a different angle, "
    "a correction, additional context, or a follow-up question — reply naturally.\n"
    "If you'd just be restating what was already said, respond with exactly: [PASS]\n"
    "Do NOT repeat or paraphrase what another model already covered."
)


class CrosstalkManager:
    """Manages organic sequential collaboration between swarm models.

    After a primary model responds, the remaining models are offered the
    conversation in random order.  Each one either contributes or passes.
    The round ends when every remaining model has had a turn.
    """

    def __init__(self, probability: float = 0.3) -> None:
        # probability is kept for config compat but no longer drives random dice.
        self._enabled = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_reaction_order(
        self, primary_model: str, all_models: list[str]
    ) -> list[str]:
        """Return the remaining models in a shuffled order for the reaction round."""
        others = [m for m in all_models if m != primary_model]
        random.shuffle(others)
        return others

    @staticmethod
    def is_pass(response_text: str) -> bool:
        """Check whether the model elected to pass."""
        stripped = response_text.strip().lower()
        return stripped.startswith("[pass]")

    @staticmethod
    def get_reaction_suffix() -> str:
        """Return the prompt suffix that tells the model about the reaction round."""
        return REACTION_PROMPT

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Legacy shim — keep old call-sites working during transition
    # ------------------------------------------------------------------

    async def select_responders(
        self, trigger_model_id: str, available_models: list[str]
    ) -> list[str]:
        """Legacy method — returns empty list so old callers are a no-op."""
        return []
