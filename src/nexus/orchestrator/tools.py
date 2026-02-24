"""LangGraph tool definitions for Agent Nexus task agents.

Tools wrap existing bot subsystems (Qdrant memory, C2 knowledge graph,
goal store) and expose them to LLM agents via LangChain's ``@tool``
decorator.  Every tool is async, catches exceptions internally, and
returns a human-readable string.

Usage::

    from nexus.orchestrator.tools import build_tools

    tools = build_tools(bot)
    agent_llm_with_tools = agent_llm.bind_tools(tools)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

log = logging.getLogger(__name__)


def build_tools(bot: Any) -> list:
    """Build LangGraph-compatible tools from bot subsystems.

    Each tool captures the ``bot`` reference via closure so it can
    access memory stores, C2, and the goal store at runtime.

    Args:
        bot: The ``NexusBot`` instance.

    Returns:
        List of LangChain tool objects ready for ``llm.bind_tools()``.
    """

    @tool
    async def query_memory(query: str, limit: int = 5) -> str:
        """Search Qdrant vector memory for relevant past conversations and knowledge.

        Use this tool when you need context from previous swarm discussions,
        user interactions, or stored knowledge.

        Args:
            query: Natural language search query describing what you're looking for.
            limit: Maximum number of results to return (1-10).
        """
        if not bot.memory_store.is_connected:
            return "Memory store is not connected."
        try:
            limit = max(1, min(limit, 10))
            vector = await bot.embeddings.embed_one(query)
            results = await bot.memory_store.search(vector, limit=limit)
            if not results:
                return "No relevant memories found."
            parts: list[str] = []
            for m in results:
                parts.append(
                    f"[{m.source}, score={m.score:.2f}]: "
                    f"{m.content[:300]}"
                )
            return "\n".join(parts)
        except Exception as exc:
            log.warning("query_memory tool failed: %s", exc)
            return f"Memory query failed: {exc}"

    @tool
    async def query_c2_context(query: str, token_budget: int = 1024) -> str:
        """Query Continuity Core for composed context relevant to a topic.

        Use this tool to retrieve structured knowledge from the C2
        knowledge graph, working memory, and event history.

        Args:
            query: The topic or question to get context for.
            token_budget: Maximum tokens of context to retrieve (256-4096).
        """
        c2 = getattr(bot, "c2", None)
        if c2 is None or not c2.is_running:
            return "C2 is not running."
        try:
            token_budget = max(256, min(token_budget, 4096))
            result = await c2.get_context(query, token_budget=token_budget)
            if result is None:
                return "No C2 context available."
            chosen = result.get("chosen", [])
            if not chosen:
                return "No relevant C2 context chunks found."
            parts: list[str] = []
            for c in chosen:
                parts.append(
                    f"[{c.get('store', '?')}] {c.get('text', '')[:200]}"
                )
            return "\n".join(parts)
        except Exception as exc:
            log.warning("query_c2_context tool failed: %s", exc)
            return f"C2 context query failed: {exc}"

    @tool
    async def query_c2_curiosity() -> str:
        """Get current epistemic tensions, contradictions, and bridging questions from C2.

        Use this tool to understand what knowledge conflicts or gaps
        have been detected in the swarm's collective understanding.
        """
        c2 = getattr(bot, "c2", None)
        if c2 is None or not c2.is_running:
            return "C2 is not running."
        try:
            result = await c2.curiosity()
            if result is None:
                return "No curiosity signals available."
            return json.dumps(result, indent=2, default=str)
        except Exception as exc:
            log.warning("query_c2_curiosity tool failed: %s", exc)
            return f"C2 curiosity query failed: {exc}"

    @tool
    async def write_c2_event(
        actor: str,
        intent: str,
        summary: str,
        tags: str = "",
    ) -> str:
        """Write an event to the C2 knowledge graph event log.

        Use this tool to record discoveries, analyses, or conclusions
        so they persist in the knowledge graph for future reference.

        Args:
            actor: Who performed the action (e.g. 'task_agent', 'analysis').
            intent: Event type (e.g. 'discovery', 'analysis', 'resolution').
            summary: Brief description of what was found or decided.
            tags: Comma-separated tags for categorization (e.g. 'security,code').
        """
        c2 = getattr(bot, "c2", None)
        if c2 is None or not c2.is_running:
            return "C2 is not running."
        try:
            tag_list = (
                [t.strip() for t in tags.split(",") if t.strip()]
                if tags
                else []
            )
            result = await c2.write_event(
                actor=actor,
                intent=intent,
                inp=summary,
                out="",
                tags=tag_list,
            )
            if result:
                return "Event logged to C2 knowledge graph."
            return "Failed to log event."
        except Exception as exc:
            log.warning("write_c2_event tool failed: %s", exc)
            return f"C2 event write failed: {exc}"

    @tool
    async def get_active_goals() -> str:
        """Get a summary of all active goals and their task progress.

        Use this tool to understand what the swarm is currently working
        on and what tasks are pending, dispatched, or completed.
        """
        goal_store = getattr(bot, "goal_store", None)
        if goal_store is None:
            return "Goal store is not available."
        try:
            summary = await goal_store.summarize_for_prompt()
            return summary
        except Exception as exc:
            log.warning("get_active_goals tool failed: %s", exc)
            return f"Goal query failed: {exc}"

    @tool
    async def get_recent_c2_events(limit: int = 10) -> str:
        """Read recent events from the C2 event log.

        Use this tool to see what has happened recently in the swarm --
        discoveries, analyses, dispatches, and maintenance events.

        Args:
            limit: Number of recent events to retrieve (1-50).
        """
        c2 = getattr(bot, "c2", None)
        if c2 is None or not c2.is_running:
            return "C2 is not running."
        try:
            limit = max(1, min(limit, 50))
            result = await c2.events(limit=limit)
            if result is None:
                return "No events available."
            return json.dumps(result, indent=2, default=str)
        except Exception as exc:
            log.warning("get_recent_c2_events tool failed: %s", exc)
            return f"C2 events query failed: {exc}"

    return [
        query_memory,
        query_c2_context,
        query_c2_curiosity,
        write_c2_event,
        get_active_goals,
        get_recent_c2_events,
    ]
