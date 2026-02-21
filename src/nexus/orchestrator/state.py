"""State gathering for the Agent Nexus orchestrator.

The :class:`StateGatherer` is responsible for collecting a snapshot of the
current system state from all available sources before each orchestrator
cycle.  The gathered state dict is passed to the decision engine, which
determines what (if any) task-agent actions to dispatch.

Sources:

1. **Conversation history** -- Recent messages from ``#nexus`` via the
   :class:`~nexus.swarm.conversation.ConversationManager`.
2. **Semantic memory** -- Relevant memories retrieved from Qdrant via the
   :class:`~nexus.memory.store.MemoryStore`.
3. **Activity stream** -- Real-time user activity from PiecesOS (when the
   integration is enabled).
4. **C2 curiosity signals** -- Epistemic tensions, contradictions, and
   bridging questions from Continuity Core (when C2 is running).

Each source is gathered independently and wrapped in error handling so that
a failure in one source does not prevent the others from contributing.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nexus.memory.store import Memory
    from nexus.swarm.conversation import SwarmMessage

log = logging.getLogger(__name__)


class StateGatherer:
    """Gathers current state for orchestrator decision-making.

    Each call to :meth:`gather` produces a fresh state dict containing:

    - ``timestamp`` -- ISO-8601 UTC timestamp of the gather operation.
    - ``recent_messages`` -- List of recent conversation message dicts.
    - ``memories`` -- List of semantically relevant memory dicts.
    - ``activity`` -- Raw PiecesOS activity string, or ``None``.
    - ``curiosity`` -- C2 curiosity signals dict, or ``None``.
    - ``has_activity`` -- Boolean flag indicating whether *any* source
      produced data (used as a fast-path check by the orchestrator).

    Args:
        bot: The ``NexusBot`` instance that owns all subsystems.
    """

    # Number of recent conversation messages to include in state.
    _CONVERSATION_LIMIT: int = 10

    # Number of memory results to retrieve per semantic search.
    _MEMORY_LIMIT: int = 3

    # The query used for the general-purpose memory search.
    _MEMORY_QUERY: str = "current tasks and priorities"

    def __init__(self, bot: Any) -> None:
        self.bot = bot

    async def gather(self) -> dict[str, Any]:
        """Gather current state from all sources in parallel.

        Returns a state dict suitable for passing to the orchestrator's
        decision engine.  Individual source failures are logged as warnings
        and do not prevent the remaining sources from being collected.

        Returns:
            A dict with keys ``timestamp``, ``recent_messages``, ``memories``,
            ``activity``, ``curiosity``, and ``has_activity``.
        """
        state: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recent_messages": [],
            "memories": [],
            "activity": None,
            "curiosity": None,
            "has_activity": False,
        }

        # Gather all sources concurrently with individual timeouts.
        # return_exceptions=True so one failure doesn't cancel the others.
        results = await asyncio.gather(
            self._gather_conversation(),
            self._gather_memories(),
            self._gather_activity(),
            self._gather_curiosity(),
            return_exceptions=True,
        )

        # Unpack, treating exceptions as None
        messages = results[0] if not isinstance(results[0], BaseException) else None
        memories = results[1] if not isinstance(results[1], BaseException) else None
        activity = results[2] if not isinstance(results[2], BaseException) else None
        curiosity = results[3] if not isinstance(results[3], BaseException) else None

        # Log any exceptions
        for i, (name, r) in enumerate(zip(
            ["conversation", "memories", "activity", "curiosity"], results,
        )):
            if isinstance(r, BaseException):
                log.warning("State gather %s failed: %s", name, r)

        if messages is not None:
            state["recent_messages"] = messages
        if memories is not None:
            state["memories"] = memories
        if activity is not None:
            state["activity"] = activity
        if curiosity is not None:
            state["curiosity"] = curiosity

        state["has_activity"] = bool(
            state["recent_messages"]
            or state["memories"]
            or state["activity"]
            or state["curiosity"]
        )

        log.debug(
            "State gathered: %d message(s), %d memory(ies), activity=%s, curiosity=%s.",
            len(state["recent_messages"]),
            len(state["memories"]),
            "yes" if state["activity"] else "no",
            "yes" if state["curiosity"] else "no",
        )
        return state

    # ------------------------------------------------------------------
    # Individual source gatherers
    # ------------------------------------------------------------------

    async def _gather_conversation(self) -> list[dict[str, Any]] | None:
        """Retrieve recent conversation history from the swarm.

        Returns:
            A list of message dicts with keys ``author``, ``content``, and
            ``timestamp``, or ``None`` on failure.
        """
        try:
            history: list[SwarmMessage] = self.bot.conversation.get_history(
                limit=self._CONVERSATION_LIMIT,
            )
            return [
                {
                    "author": "human" if msg.is_human else msg.model_id,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                }
                for msg in history
            ]
        except Exception:
            log.warning(
                "Failed to gather conversation history.",
                exc_info=True,
            )
            return None

    async def _gather_memories(self) -> list[dict[str, Any]] | None:
        """Search Qdrant for semantically relevant memories.

        Uses the embedding provider to vectorise a general-purpose query and
        then performs a similarity search against the memory store.

        Returns:
            A list of memory dicts with keys ``content``, ``source``, and
            ``score``, or ``None`` on failure or when the store is not
            connected.
        """
        try:
            if not self.bot.memory_store.is_connected:
                log.debug("Memory store not connected -- skipping memory gather.")
                return None

            query_vector: list[float] = await asyncio.wait_for(
                self.bot.embeddings.embed_one(self._MEMORY_QUERY),
                timeout=15.0,
            )
            results: list[Memory] = await asyncio.wait_for(
                self.bot.memory_store.search(
                    query_vector, limit=self._MEMORY_LIMIT,
                ),
                timeout=10.0,
            )
            return [
                {
                    "content": m.content,
                    "source": m.source,
                    "score": m.score,
                }
                for m in results
            ]
        except Exception:
            log.warning(
                "Failed to gather memories from Qdrant.",
                exc_info=True,
            )
            return None

    async def _gather_activity(self) -> str | None:
        """Retrieve recent user activity from PiecesOS.

        PiecesOS integration is optional and may not be configured.  When
        the client is absent this method returns ``None`` silently.
        The client handles its own reconnection if the session expired.

        Returns:
            A raw activity summary string, or ``None`` if the integration
            is unavailable or returns nothing.
        """
        try:
            pieces = getattr(self.bot, "pieces", None)
            if pieces is None:
                return None

            # get_recent_activity handles auto-reconnect internally
            activity: str | None = await asyncio.wait_for(
                pieces.get_recent_activity(), timeout=20.0,
            )
            return activity if activity else None

        except asyncio.TimeoutError:
            log.warning("PiecesOS activity gather timed out after 20s.")
            return None
        except Exception:
            log.warning(
                "Failed to gather PiecesOS activity.",
                exc_info=True,
            )
            return None

    async def _gather_curiosity(self) -> dict[str, Any] | None:
        """Query C2 for epistemic tensions, contradictions, bridging questions.

        C2 integration is optional.  When the C2 client is absent or not
        running this method returns ``None`` silently.

        Returns:
            A dict with keys ``stress_level``, ``contradictions``,
            ``deep_tensions``, ``bridging_questions``, and
            ``suggested_action``, or ``None`` if C2 is unavailable.
        """
        try:
            c2 = getattr(self.bot, "c2", None)
            if c2 is None or not c2.is_running:
                return None

            result = await asyncio.wait_for(
                c2.curiosity(), timeout=30.0,
            )
            return result if result else None

        except asyncio.TimeoutError:
            log.warning("C2 curiosity gather timed out after 30s.")
            return None
        except Exception:
            log.warning(
                "Failed to gather C2 curiosity signals.",
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"StateGatherer(conversation_limit={self._CONVERSATION_LIMIT}, "
            f"memory_limit={self._MEMORY_LIMIT})"
        )
