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
                    f"{m.content[:1000]}"
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
                    f"[{c.get('store', '?')}] {c.get('text', '')[:500]}"
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

    @tool
    async def remember_finding(content: str, importance: int = 7) -> str:
        """Store an important finding in vector memory for future reference.

        Use this when you discover something significant that future task
        agents should be able to find via semantic search. Be specific and
        descriptive — vague findings are not useful.

        Args:
            content: The finding to remember (be specific and descriptive).
            importance: How important this is (1-10, default 7).
        """
        if not bot.memory_store.is_connected:
            return "Memory store is not connected."
        try:
            importance = max(1, min(importance, 10))
            text = content[:500]
            vector = await bot.embeddings.embed_one(text)
            await bot.memory_store.store(
                content=text,
                vector=vector,
                source="task_agent",
                channel="nexus",
                metadata={
                    "type": "agent_discovery",
                    "importance": str(importance),
                },
            )
            return "Finding stored in vector memory."
        except Exception as exc:
            log.warning("remember_finding tool failed: %s", exc)
            return f"Failed to store finding: {exc}"

    @tool
    async def synthesize_code(requirement: str) -> str:
        """Build a Python function using TDD synthesis.

        Generates tests from the requirement, writes code, runs it in a
        sandbox, and iterates until tests pass. Returns the working code
        or an error description.

        Use this when a task requires producing working, tested code.

        Args:
            requirement: Plain English description of what the code should do.
        """
        tdd = getattr(bot, "tdd", None)
        if tdd is None:
            return "TDD engine is not available."
        try:
            result = await bot.tdd.synthesize(requirement)
            if result.is_success:
                return (
                    f"BUILD SUCCESS ({result.iterations} iterations, "
                    f"{result.tests_passed}/{result.total_tests} tests passed)\n\n"
                    f"```python\n{result.generated_code}\n```"
                )
            errors = "; ".join(result.errors) if result.errors else "Tests did not pass"
            return f"BUILD FAILED after {result.iterations} iterations: {errors}"
        except Exception as exc:
            log.warning("synthesize_code tool failed: %s", exc)
            return f"Synthesis failed: {exc}"

    @tool
    async def write_file(filename: str, content: str) -> str:
        """Write content to a file in the bot's workspace directory.

        Use this to save generated code, documentation, or other outputs.
        Files are written to the workspace/ directory relative to the bot.

        Args:
            filename: Name of the file (e.g. 'fibonacci.py'). No path traversal allowed.
            content: The content to write to the file.
        """
        import re as _re
        from pathlib import Path

        # Sanitize filename -- no path traversal
        if not _re.match(r"^[\w][\w.\-]*$", filename):
            return f"Invalid filename: {filename}. Use only letters, numbers, dots, hyphens."

        workspace = Path("workspace")
        workspace.mkdir(exist_ok=True)
        path = workspace / filename

        try:
            path.write_text(content, encoding="utf-8")
            return f"File written: {path}"
        except Exception as exc:
            return f"Failed to write file: {exc}"

    return [
        query_memory,
        query_c2_context,
        query_c2_curiosity,
        write_c2_event,
        get_active_goals,
        get_recent_c2_events,
        remember_finding,
        synthesize_code,
        write_file,
    ]
