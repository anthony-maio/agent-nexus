"""Multi-model conversation manager for the Agent Nexus swarm.

Models see each other's messages in the #nexus channel. The
``ConversationManager`` maintains a rolling conversation history and
builds OpenAI-compatible message lists so every model has full visibility
into the ongoing swarm discussion.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

log = logging.getLogger(__name__)


@dataclass
class SwarmMessage:
    """A message in the swarm conversation."""
    model_id: str
    content: str
    timestamp: datetime
    is_human: bool = False
    in_reply_to: str | None = None  # model_id this replies to


class ConversationManager:
    """Manages multi-model conversation flow in #nexus.

    When one model posts, the conversation history is available to all other
    models. Each model can see what others have said and respond accordingly.
    """

    def __init__(self) -> None:
        self._history: list[SwarmMessage] = []
        self._max_history = 50
        self._lock = asyncio.Lock()

    async def add_message(self, model_id: str, content: str, is_human: bool = False) -> SwarmMessage:
        """Record a message in the conversation."""
        msg = SwarmMessage(
            model_id=model_id,
            content=content,
            timestamp=datetime.now(timezone.utc),
            is_human=is_human,
        )
        async with self._lock:
            self._history.append(msg)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
        return msg

    def get_history(self, limit: int = 20) -> list[SwarmMessage]:
        """Get recent conversation history."""
        return self._history[-limit:]

    def build_messages_for_model(self, model_id: str, system_prompt: str, limit: int = 15) -> list[dict]:
        """Build OpenAI-format message list for a specific model.

        The model sees:
        1. Its system prompt
        2. Recent conversation history with other models labeled
        """
        messages = [{"role": "system", "content": system_prompt}]

        for msg in self._history[-limit:]:
            if msg.is_human:
                messages.append({
                    "role": "user",
                    "content": msg.content,
                })
            elif msg.model_id == model_id:
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                })
            else:
                # Other model's message appears as user message with attribution
                # This lets the model see what others said without confusing role assignments
                from nexus.personality.identities import format_name
                name = format_name(msg.model_id)
                messages.append({
                    "role": "user",
                    "content": f"[{name}]: {msg.content}",
                })

        return messages

    def clear(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    @property
    def message_count(self) -> int:
        return len(self._history)
