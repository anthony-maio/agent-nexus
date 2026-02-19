from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

log = logging.getLogger(__name__)


@dataclass
class ContextEntry:
    """A single context item for prompt building."""
    content: str
    source: str          # "memory", "message", "activity"
    relevance: float     # 0.0-1.0
    timestamp: datetime


@dataclass
class PromptContext:
    """Assembled context for a model prompt."""
    entries: list[ContextEntry] = field(default_factory=list)
    token_budget: int = 4000

    def add(self, entry: ContextEntry) -> None:
        self.entries.append(entry)

    def build(self) -> str:
        """Build context string sorted by relevance, trimmed to token budget."""
        sorted_entries = sorted(self.entries, key=lambda e: e.relevance, reverse=True)
        parts = []
        estimated_tokens = 0
        for entry in sorted_entries:
            # Rough estimate: 1 token ~ 4 chars
            entry_tokens = len(entry.content) // 4
            if estimated_tokens + entry_tokens > self.token_budget:
                break
            parts.append(f"[{entry.source}] {entry.content}")
            estimated_tokens += entry_tokens
        return "\n\n".join(parts)

    @property
    def is_empty(self) -> bool:
        return len(self.entries) == 0


class ContextBuilder:
    """Builds prompt context from memories, recent messages, and activity."""

    def __init__(self, memory_store, embedding_provider):
        self.memory = memory_store
        self.embeddings = embedding_provider
        self._recent_messages: list[dict] = []
        self._max_recent = 20

    def add_message(self, author: str, content: str, channel: str, timestamp: datetime) -> None:
        """Track a recent message for context building."""
        self._recent_messages.append({
            "author": author,
            "content": content,
            "channel": channel,
            "timestamp": timestamp,
        })
        # Keep only the most recent N messages
        if len(self._recent_messages) > self._max_recent:
            self._recent_messages = self._recent_messages[-self._max_recent:]

    async def build_context(
        self,
        query: str,
        token_budget: int = 4000,
        include_memories: bool = True,
        include_recent: bool = True,
    ) -> PromptContext:
        """Build context for a prompt."""
        ctx = PromptContext(token_budget=token_budget)

        # Add recent messages (high relevance since they're current)
        if include_recent and self._recent_messages:
            for msg in self._recent_messages[-10:]:
                ctx.add(ContextEntry(
                    content=f"{msg['author']}: {msg['content']}",
                    source="message",
                    relevance=0.7,
                    timestamp=msg["timestamp"],
                ))

        # Search memory for relevant context
        if include_memories and self.memory.is_connected:
            try:
                query_vector = await self.embeddings.embed_one(query)
                memories = await self.memory.search(query_vector, limit=5)
                for mem in memories:
                    ctx.add(ContextEntry(
                        content=mem.content,
                        source="memory",
                        relevance=mem.score,
                        timestamp=mem.timestamp,
                    ))
            except Exception as e:
                log.warning(f"Memory search failed: {e}")

        return ctx
