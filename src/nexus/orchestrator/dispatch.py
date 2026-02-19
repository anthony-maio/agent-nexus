"""Task dispatch to LiquidAI Tier 2 agents for Agent Nexus.

The :class:`TaskDispatcher` receives action dicts from the orchestrator's
decision engine and routes each one to the appropriate LiquidAI task-agent
model.  Results are posted back to ``#nexus`` so the swarm can observe and
build on task-agent output.

Task agents are small, fast models (LiquidAI LFM 1.2B family) that execute
specific, bounded jobs:

- **Router/Classifier** -- Intent detection, routing, labelling.
- **Reasoning** -- Chain-of-thought analysis, research synthesis.
- **Tool Calling** -- Structured function invocation (Phase 3).
- **RAG** -- Context-grounded retrieval answers.
- **Data Extraction** -- Structured output from unstructured input.

Models are selected from :data:`nexus.models.registry.TASK_MODELS` based on
the action type.  OpenRouter-hosted models are preferred; Ollama-hosted models
are used as fallback for roles that require local inference.

Usage::

    dispatcher = TaskDispatcher(bot)
    result = await dispatcher.dispatch({
        "type": "research",
        "description": "Summarize recent activity patterns.",
        "priority": "medium",
    })
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TaskResult:
    """Result from a dispatched task agent.

    Attributes:
        action_type: The role of the task agent that executed the task
            (e.g. ``"reasoning"``, ``"extraction"``).
        description: The original task description (truncated to 200 chars).
        result: The task agent's response text, or an error message.
        model_used: The model identifier that handled the task.
        success: Whether the task completed without error.
        timestamp: UTC timestamp of when the result was produced.
        priority: The priority level from the original action.
    """

    action_type: str
    description: str
    result: str
    model_used: str
    success: bool
    timestamp: datetime
    priority: str = "medium"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class TaskDispatcher:
    """Dispatches tasks to LiquidAI Tier 2 task agents.

    The dispatcher maps action types to task-agent roles, selects the
    appropriate model from the registry, calls it via OpenRouter or Ollama,
    and posts the result back to ``#nexus``.

    Args:
        bot: The ``NexusBot`` instance that owns all subsystems.
        max_result_history: Maximum number of recent results to retain
            in memory for inspection.
    """

    # Map action types from the orchestrator to task-agent role keys in
    # TASK_MODELS.  Unknown types fall back to "reasoning".
    ROLE_MAP: dict[str, str] = {
        "research": "reasoning",
        "code": "reasoning",        # Tool-calling agent in Phase 3
        "analyze": "reasoning",
        "summarize": "extraction",
        "classify": "router",
        "extract": "extraction",
    }

    def __init__(self, bot: Any, max_result_history: int = 50) -> None:
        self.bot = bot
        self._max_result_history: int = max_result_history
        self._results: list[TaskResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def dispatch(self, action: dict[str, Any]) -> TaskResult | None:
        """Dispatch an action to the appropriate task agent.

        Args:
            action: An action dict from the orchestrator's decision engine.
                Expected keys: ``type``, ``description``, ``priority``.

        Returns:
            A :class:`TaskResult` on success or failure, or ``None`` if the
            action has no description and is therefore skipped.
        """
        action_type: str = action.get("type", "analyze")
        description: str = str(action.get("description", "")).strip()
        priority: str = action.get("priority", "medium")

        if not description:
            log.debug("Skipping action with empty description.")
            return None

        role = self.ROLE_MAP.get(action_type, "reasoning")

        log.info(
            "Dispatching %s task (priority=%s) to %s agent: %.100s",
            action_type,
            priority,
            role,
            description,
        )

        result = await self._call_task_agent(role, description, priority)

        if result is not None:
            self._store_result(result)
            await self._post_result_to_nexus(result, description)

        return result

    # ------------------------------------------------------------------
    # Agent invocation
    # ------------------------------------------------------------------

    async def _call_task_agent(
        self,
        role: str,
        prompt: str,
        priority: str = "medium",
    ) -> TaskResult | None:
        """Call the appropriate task-agent model.

        Resolves the model from the registry, routes to OpenRouter or Ollama,
        and wraps the response in a :class:`TaskResult`.
        """
        from nexus.models.registry import TASK_MODELS, ModelProvider, ModelSpec

        task_model: ModelSpec | None = TASK_MODELS.get(role)
        if task_model is None:
            log.warning(
                "No task model registered for role '%s'. "
                "Available roles: %s",
                role,
                sorted(TASK_MODELS.keys()),
            )
            return None

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a task agent in the Agent Nexus swarm. "
                    "Execute the given task concisely and return the result. "
                    "Be specific, actionable, and evidence-based. "
                    "Keep your response under 500 words."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self._route_to_provider(task_model, messages)

            return TaskResult(
                action_type=role,
                description=prompt[:200],
                result=response.content,
                model_used=task_model.id,
                success=True,
                timestamp=datetime.now(timezone.utc),
                priority=priority,
            )

        except Exception as exc:
            log.error(
                "Task agent '%s' (model=%s) failed: %s",
                role,
                task_model.id,
                exc,
                exc_info=True,
            )
            return TaskResult(
                action_type=role,
                description=prompt[:200],
                result=f"Task agent error: {exc}",
                model_used=task_model.id,
                success=False,
                timestamp=datetime.now(timezone.utc),
                priority=priority,
            )

    async def _route_to_provider(
        self,
        task_model: Any,
        messages: list[dict[str, str]],
    ) -> Any:
        """Route a chat request to the correct provider based on the model spec.

        Args:
            task_model: A :class:`~nexus.models.registry.ModelSpec` instance.
            messages: OpenAI-format message list.

        Returns:
            A ``ChatResponse`` from either the OpenRouter or Ollama client.

        Raises:
            RuntimeError: If the required provider client is not available.
        """
        from nexus.models.registry import ModelProvider

        if task_model.provider == ModelProvider.OPENROUTER:
            return await self.bot.openrouter.chat(
                model=task_model.id,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )

        if task_model.provider == ModelProvider.OLLAMA:
            ollama = getattr(self.bot, "ollama", None)
            if ollama is None:
                raise RuntimeError(
                    f"Ollama client is not available but model '{task_model.id}' "
                    f"requires local inference. Ensure Ollama is running."
                )
            return await ollama.chat(
                model=task_model.id,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )

        raise RuntimeError(
            f"Unsupported provider '{task_model.provider.value}' for "
            f"task model '{task_model.id}'."
        )

    # ------------------------------------------------------------------
    # Result posting
    # ------------------------------------------------------------------

    async def _post_result_to_nexus(
        self,
        result: TaskResult,
        full_description: str,
    ) -> None:
        """Post a task result to the ``#nexus`` channel for swarm visibility.

        Formats the result as a Discord embed using the task model's identity,
        including the original task description and the agent's response.
        """
        try:
            channel = self.bot.router.nexus
            from nexus.channels.formatter import MessageFormatter

            # Build a combined message showing the task and its result.
            status_tag = "completed" if result.success else "FAILED"
            body_parts: list[str] = [
                f"**Task [{status_tag}]:** {full_description[:300]}",
                "",
                result.result,
            ]
            body = "\n".join(body_parts)

            embed = MessageFormatter.format_response(
                result.model_used,
                body,
            )
            await channel.send(embed=embed)

        except Exception:
            log.error(
                "Failed to post task result to #nexus.",
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Result history
    # ------------------------------------------------------------------

    def _store_result(self, result: TaskResult) -> None:
        """Append a result to the in-memory history, pruning if needed."""
        self._results.append(result)
        if len(self._results) > self._max_result_history:
            self._results = self._results[-self._max_result_history :]

    @property
    def recent_results(self) -> list[TaskResult]:
        """The most recent task results (up to ``max_result_history``)."""
        return list(self._results)

    @property
    def success_count(self) -> int:
        """Number of successful task results in the history."""
        return sum(1 for r in self._results if r.success)

    @property
    def failure_count(self) -> int:
        """Number of failed task results in the history."""
        return sum(1 for r in self._results if not r.success)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TaskDispatcher(results={len(self._results)}, "
            f"success={self.success_count}, "
            f"failures={self.failure_count})"
        )
