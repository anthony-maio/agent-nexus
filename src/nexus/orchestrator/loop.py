"""Background orchestrator loop for Agent Nexus.

The :class:`OrchestratorLoop` is the autonomous heartbeat of the nexus.  It
runs on a configurable interval (default: 1 hour), gathering state from the
swarm conversation, Qdrant memory, and the PiecesOS activity stream, then
asking a swarm model to decide what actions to take, and finally dispatching
those actions to LiquidAI Tier 2 task agents.

Results are posted back to ``#nexus`` so the entire swarm can observe and
react to task-agent output.

Lifecycle::

    loop = OrchestratorLoop(bot, interval=3600)
    await loop.start()   # begins background task
    ...
    await loop.stop()    # cancels cleanly
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass  # forward references resolved at runtime

log = logging.getLogger(__name__)


class OrchestratorLoop:
    """Background orchestrator: gather state -> decide actions -> dispatch tasks.

    Runs on a configurable interval (default: 1 hour).  Informed by:

    - Recent swarm conversation history from ``#nexus``
    - Semantic memory search results from Qdrant
    - PiecesOS activity stream (when enabled)

    Dispatches task agents (LiquidAI Tier 2) for specific jobs and reports
    results back to the swarm in ``#nexus``.

    Args:
        bot: The ``NexusBot`` instance that owns all subsystems (OpenRouter
            client, conversation manager, memory store, channel router, etc.).
        interval: Seconds between orchestrator cycles.  Must be >= 10.
    """

    # Cap per-cycle actions to prevent runaway dispatches.
    _MAX_ACTIONS_PER_CYCLE: int = 5

    # Seconds to wait after bot startup before the first cycle fires.
    _STARTUP_DELAY: float = 30.0

    def __init__(self, bot: Any, interval: int = 3600) -> None:
        self.bot = bot
        self.interval: int = max(interval, 10)
        self._running: bool = False
        self._task: asyncio.Task[None] | None = None
        self._cycle_count: int = 0
        self._last_cycle: datetime | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the orchestrator background loop.

        Idempotent -- calling ``start()`` when already running is a no-op.
        """
        if self._running:
            log.warning("Orchestrator start() called but loop is already running.")
            return

        self._running = True
        self._task = asyncio.create_task(self._loop(), name="orchestrator-loop")
        log.info(
            "Orchestrator started (interval=%ds, startup_delay=%.0fs).",
            self.interval,
            self._STARTUP_DELAY,
        )

    async def stop(self) -> None:
        """Stop the orchestrator loop gracefully.

        Cancels the background task and waits for it to finish.  Safe to call
        even if the loop was never started.
        """
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info(
            "Orchestrator stopped after %d completed cycle(s).",
            self._cycle_count,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        """Main loop: gather -> decide -> dispatch, then sleep."""
        # Allow the rest of the bot to finish bootstrapping before the
        # first orchestrator cycle fires.
        await asyncio.sleep(self._STARTUP_DELAY)

        while self._running:
            try:
                self._cycle_count += 1
                cycle_start = datetime.now(timezone.utc)
                log.info("Orchestrator cycle #%d starting.", self._cycle_count)

                # 1. Gather state from all sources.
                state: dict[str, Any] = await self.bot.state_gatherer.gather()

                # 2. Ask a swarm model what to do about it.
                actions: list[dict[str, Any]] = await self._decide(state)

                # 3. Dispatch each action to the appropriate task agent.
                dispatched = 0
                for action in actions:
                    result = await self.bot.dispatcher.dispatch(action)
                    if result is not None:
                        dispatched += 1

                self._last_cycle = datetime.now(timezone.utc)
                elapsed = (self._last_cycle - cycle_start).total_seconds()
                log.info(
                    "Orchestrator cycle #%d complete: %d action(s) dispatched in %.1fs.",
                    self._cycle_count,
                    dispatched,
                    elapsed,
                )

            except Exception:
                log.error(
                    "Orchestrator cycle #%d raised an unhandled exception.",
                    self._cycle_count,
                    exc_info=True,
                )

            # Sleep until the next cycle.  If the loop was stopped during the
            # cycle we break out immediately.
            if self._running:
                await asyncio.sleep(self.interval)

    # ------------------------------------------------------------------
    # Decision engine
    # ------------------------------------------------------------------

    async def _decide(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Ask a swarm model what actions to take based on current state.

        The decision model receives a structured summary of the current state
        and returns a JSON array of action objects.  Each action has:

        - ``type`` -- one of ``research``, ``code``, ``analyze``, ``summarize``,
          ``classify``, ``extract``.
        - ``description`` -- human-readable description of the task.
        - ``priority`` -- ``high``, ``medium``, or ``low``.

        Returns an empty list when there is nothing actionable or the model
        response cannot be parsed.
        """
        if not state.get("has_activity"):
            log.debug("No activity detected -- skipping decision phase.")
            return []

        prompt = self._build_decision_prompt(state)

        # Select the first available swarm model for orchestration decisions.
        model_ids: list[str] = list(self.bot.swarm_models.keys())
        if not model_ids:
            log.warning("No swarm models configured -- cannot make orchestrator decisions.")
            return []

        decision_model = model_ids[0]

        try:
            response = await self.bot.openrouter.chat(
                model=decision_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are the orchestrator for an AI agent swarm called Agent Nexus. "
                            "Based on the current state, decide what tasks to dispatch to the "
                            "swarm's task agents. Respond with a JSON array of action objects. "
                            "Each action has:\n"
                            '  "type": one of "research", "code", "analyze", "summarize", '
                            '"classify", "extract"\n'
                            '  "description": a clear, specific task description\n'
                            '  "priority": "high", "medium", or "low"\n\n'
                            "Return an empty array [] if no actions are needed right now. "
                            "Only propose actions that are clearly useful based on the state. "
                            "Do not invent tasks with no basis in the provided context."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )

            return self._parse_actions(response.content, decision_model)

        except Exception:
            log.error(
                "Orchestrator decision failed (model=%s).",
                decision_model,
                exc_info=True,
            )
            return []

    def _parse_actions(
        self, raw_response: str, model_id: str
    ) -> list[dict[str, Any]]:
        """Parse the decision model's response into a list of action dicts.

        Handles common LLM quirks like wrapping JSON in markdown code fences.
        """
        text = raw_response.strip()

        # Strip markdown code fences if present.
        if text.startswith("```"):
            # Remove opening fence (with optional language tag) and closing fence.
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            log.warning(
                "Could not parse orchestrator decision as JSON (model=%s): %.200s",
                model_id,
                text,
            )
            return []

        if not isinstance(parsed, list):
            log.warning(
                "Orchestrator decision is not a JSON array (model=%s, type=%s).",
                model_id,
                type(parsed).__name__,
            )
            return []

        # Validate and cap actions.
        valid_types = {"research", "code", "analyze", "summarize", "classify", "extract"}
        valid_priorities = {"high", "medium", "low"}
        actions: list[dict[str, Any]] = []

        for item in parsed[: self._MAX_ACTIONS_PER_CYCLE]:
            if not isinstance(item, dict):
                continue
            action_type = item.get("type", "analyze")
            if action_type not in valid_types:
                action_type = "analyze"
            priority = item.get("priority", "medium")
            if priority not in valid_priorities:
                priority = "medium"
            description = str(item.get("description", "")).strip()
            if not description:
                continue
            actions.append({
                "type": action_type,
                "description": description,
                "priority": priority,
            })

        return actions

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_decision_prompt(self, state: dict[str, Any]) -> str:
        """Build the decision prompt from gathered state.

        The prompt is structured so the model can quickly scan the current
        state and determine whether any tasks should be dispatched.
        """
        parts: list[str] = [
            f"Timestamp: {state.get('timestamp', 'unknown')}",
            "",
            "=== Current Swarm State ===",
        ]

        # Recent conversation.
        recent_msgs: list[dict[str, Any]] = state.get("recent_messages", [])
        if recent_msgs:
            parts.append(f"\n--- Recent Conversation ({len(recent_msgs)} messages) ---")
            # Show only the last 5 to keep the prompt concise.
            for msg in recent_msgs[-5:]:
                author = msg.get("author", "unknown")
                content = msg.get("content", "")[:200]
                parts.append(f"  [{author}]: {content}")
        else:
            parts.append("\n--- Recent Conversation ---")
            parts.append("  (no recent messages)")

        # Relevant memories.
        memories: list[dict[str, Any]] = state.get("memories", [])
        if memories:
            parts.append(f"\n--- Relevant Memories ({len(memories)}) ---")
            for mem in memories[:3]:
                content = mem.get("content", "")[:200]
                source = mem.get("source", "unknown")
                score = mem.get("score", 0.0)
                parts.append(f"  [{source}, relevance={score:.2f}]: {content}")

        # PiecesOS activity.
        activity: str | None = state.get("activity")
        if activity:
            parts.append("\n--- Recent User Activity (PiecesOS) ---")
            parts.append(f"  {activity[:500]}")

        parts.append("\n=== End State ===")
        parts.append(
            "\nBased on this state, what tasks should be dispatched to the "
            "task agents? Return a JSON array of actions."
        )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Whether the orchestrator background loop is currently active."""
        return self._running

    @property
    def cycles_completed(self) -> int:
        """Total number of orchestrator cycles that have finished."""
        return self._cycle_count

    @property
    def last_cycle_time(self) -> datetime | None:
        """UTC timestamp of the most recent completed cycle, or ``None``."""
        return self._last_cycle

    def __repr__(self) -> str:
        status = "running" if self._running else "stopped"
        return (
            f"OrchestratorLoop(status={status!r}, "
            f"interval={self.interval}s, "
            f"cycles={self._cycle_count})"
        )
