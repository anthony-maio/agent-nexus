"""Background orchestrator loop for Agent Nexus.

The :class:`OrchestratorLoop` is the autonomous heartbeat of the nexus.  It
supports two modes:

- **Full cycle** (hourly): Gather state, decide actions, dispatch via autonomy
  gate, run night-cycle maintenance, log to C2.
- **Mini cycle** (triggered by activity change): Gather + decide + dispatch
  only — no maintenance.

External triggers (e.g. from :class:`~nexus.orchestrator.activity.ActivityMonitor`)
can wake the loop immediately via :meth:`trigger_cycle`.

Lifecycle::

    loop = OrchestratorLoop(bot, interval=3600)
    await loop.start()
    await loop.trigger_cycle()  # wake immediately for a mini-cycle
    await loop.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import discord

if TYPE_CHECKING:
    from nexus.orchestrator.autonomy import AutonomyGate

log = logging.getLogger(__name__)


class OrchestratorLoop:
    """Background orchestrator: gather state -> decide actions -> dispatch tasks.

    Runs on a configurable interval (default: 1 hour).  Informed by:

    - Recent swarm conversation history from ``#nexus``
    - Semantic memory search results from Qdrant
    - PiecesOS activity stream (when enabled)
    - C2 curiosity signals (when C2 is running)

    Dispatches task agents (LiquidAI Tier 2) for specific jobs, subject to
    the autonomy gate, and reports results back to the swarm in ``#nexus``.

    Args:
        bot: The ``NexusBot`` instance that owns all subsystems.
        interval: Seconds between full orchestrator cycles.  Must be >= 10.
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
        self._trigger_event: asyncio.Event = asyncio.Event()

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
        self._trigger_event = asyncio.Event()
        self._task = asyncio.create_task(self._loop(), name="orchestrator-loop")
        log.info(
            "Orchestrator started (interval=%ds, startup_delay=%.0fs).",
            self.interval,
            self._STARTUP_DELAY,
        )

    async def stop(self) -> None:
        """Stop the orchestrator loop gracefully."""
        self._running = False
        self._trigger_event.set()  # Wake the loop so it can exit
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

    async def trigger_cycle(self) -> None:
        """Wake the loop to run a mini-cycle immediately."""
        self._trigger_event.set()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        """Main loop: hybrid interval + trigger-based cycling."""
        await asyncio.sleep(self._STARTUP_DELAY)

        # Run an immediate first cycle after startup delay.
        first_cycle = True

        while self._running:
            is_triggered = False
            try:
                if first_cycle:
                    # Skip waiting — run the first cycle immediately.
                    first_cycle = False
                else:
                    # Wait for either the interval to elapse or a trigger event.
                    try:
                        await asyncio.wait_for(
                            self._trigger_event.wait(), timeout=self.interval,
                        )
                        # Trigger fired — this is a mini-cycle.
                        is_triggered = True
                        self._trigger_event.clear()
                    except asyncio.TimeoutError:
                        # Interval elapsed — this is a full cycle.
                        pass

                if not self._running:
                    break

                self._cycle_count += 1
                cycle_start = datetime.now(timezone.utc)
                cycle_type = "mini" if is_triggered else "full"
                log.info(
                    "Orchestrator %s cycle #%d starting.",
                    cycle_type,
                    self._cycle_count,
                )

                # 1. Gather state from all sources.
                try:
                    # Per-source timeouts handle individual failures.
                    # Outer timeout is a safety net only (max source = 90s).
                    state: dict[str, Any] = await asyncio.wait_for(
                        self.bot.state_gatherer.gather(), timeout=120.0,
                    )
                except asyncio.TimeoutError:
                    log.warning("State gather timed out after 120s -- skipping cycle.")
                    continue

                # 2. Ask a swarm model what to do about it.
                try:
                    actions: list[dict[str, Any]] = await asyncio.wait_for(
                        self._decide(state), timeout=60.0,
                    )
                except asyncio.TimeoutError:
                    log.warning("Decision phase timed out after 60s -- skipping dispatch.")
                    actions = []

                # 3. Dispatch each action through the autonomy gate.
                dispatched = 0
                for action in actions:
                    dispatched += await self._gate_and_dispatch(action)

                # 4. Full cycle only: run night-cycle maintenance via C2.
                if not is_triggered:
                    await self._run_night_cycle()

                self._last_cycle = datetime.now(timezone.utc)
                elapsed = (self._last_cycle - cycle_start).total_seconds()
                log.info(
                    "Orchestrator %s cycle #%d complete: %d action(s) dispatched in %.1fs.",
                    cycle_type,
                    self._cycle_count,
                    dispatched,
                    elapsed,
                )

                # Log cycle completion to C2
                await self._log_to_c2(
                    actor="orchestrator",
                    intent="cycle_complete",
                    inp=cycle_type,
                    out=f"dispatched={dispatched} elapsed={elapsed:.1f}s",
                    tags=["orchestrator", cycle_type],
                )

            except Exception:
                log.error(
                    "Orchestrator cycle #%d raised an unhandled exception.",
                    self._cycle_count,
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Autonomy gate
    # ------------------------------------------------------------------

    async def _gate_and_dispatch(self, action: dict[str, Any]) -> int:
        """Run an action through the autonomy gate and dispatch if allowed.

        Returns 1 if dispatched, 0 otherwise.
        """
        gate: AutonomyGate | None = getattr(self.bot, "autonomy_gate", None)

        if gate is not None:
            if gate.should_auto_execute(action):
                log.info("Auto-executing action: %s", action.get("description", ""))
            elif gate.should_escalate(action):
                log.info("Escalating action to #human: %s", action.get("description", ""))
                approved = await gate.propose_and_wait(self.bot, action)
                if not approved:
                    log.info("Action rejected or timed out: %s", action.get("description", ""))
                    return 0
            else:
                # Shouldn't happen, but fall through to dispatch.
                pass

        result = await self.bot.dispatcher.dispatch(action)
        if result is not None:
            # Log dispatch to C2
            await self._log_to_c2(
                actor="orchestrator",
                intent="dispatch",
                inp=action.get("description", ""),
                out=str(result)[:500],
                tags=["task", action.get("type", "unknown")],
            )
            return 1
        return 0

    # ------------------------------------------------------------------
    # Night cycle
    # ------------------------------------------------------------------

    async def _run_night_cycle(self) -> None:
        """Run C2 night-cycle maintenance if C2 is available."""
        c2 = getattr(self.bot, "c2", None)
        if c2 is None or not c2.is_running:
            return

        try:
            result = await c2.maintenance()
            if result is None:
                return

            log.info(
                "Night cycle complete: stress=%.3f, contradictions=%d, voids=%d.",
                result.get("stress_after", 0),
                result.get("contradictions_found", 0),
                result.get("voids_found", 0),
            )

            # Post curiosity findings to #nexus if there are contradictions.
            if result.get("contradictions_found", 0) > 0:
                await self._post_curiosity_findings(result)

            # Trigger swarm discussion if curiosity signals are actionable.
            should_discuss = (
                result.get("stress_after", 0) > 0.2
                or result.get("contradictions_found", 0) > 0
                or result.get("voids_found", 0) > 0
            )
            if should_discuss:
                curiosity_signals = await c2.curiosity()
                if curiosity_signals is not None:
                    await self._trigger_curiosity_discussion(curiosity_signals)

        except Exception:
            log.warning("Night cycle maintenance failed.", exc_info=True)

    async def _post_curiosity_findings(self, result: dict[str, Any]) -> None:
        """Post night-cycle findings to #nexus for swarm discussion."""
        router = getattr(self.bot, "router", None)
        if router is None or router.nexus is None:
            return

        embed = discord.Embed(
            title="Night Cycle - Curiosity Findings",
            color=0x9B59B6,
        )
        embed.add_field(
            name="Epistemic Stress",
            value=f"{result.get('stress_after', 0):.3f}",
            inline=True,
        )
        embed.add_field(
            name="Contradictions",
            value=str(result.get("contradictions_found", 0)),
            inline=True,
        )
        embed.add_field(
            name="Deep Tensions",
            value=str(result.get("deep_tensions_found", 0)),
            inline=True,
        )
        embed.add_field(
            name="Voids",
            value=str(result.get("voids_found", 0)),
            inline=True,
        )
        resolutions = result.get("resolutions", [])
        if resolutions:
            embed.add_field(
                name="Resolutions",
                value=str(len(resolutions)),
                inline=True,
            )
        embed.set_footer(text="Consider discussing these tensions in the swarm.")

        await router.nexus.send(embed=embed)

    async def _trigger_curiosity_discussion(
        self, curiosity_result: dict[str, Any],
    ) -> None:
        """Trigger a Tier 1 swarm discussion about curiosity findings.

        Picks a random model, sends the curiosity prompt, posts the response
        to #nexus, runs crosstalk reactions, and posts a summary to #memory.
        """
        import random

        from nexus.channels.formatter import MessageFormatter

        model_ids = list(self.bot.swarm_models.keys())
        if not model_ids:
            return

        # Build the curiosity discussion prompt
        prompt_parts: list[str] = [
            "The Continuity Core has detected epistemic tensions in our "
            "knowledge base that need investigation.",
            "",
        ]

        stress = curiosity_result.get("stress_level", 0)
        prompt_parts.append(f"Epistemic stress level: {stress:.3f}")

        contradictions = curiosity_result.get("contradictions", [])
        if contradictions:
            prompt_parts.append(f"\nContradictions ({len(contradictions)}):")
            for c in contradictions[:5]:
                prompt_parts.append(
                    f'  - "{c.get("s1", "")}" vs "{c.get("s2", "")}"'
                )

        tensions = curiosity_result.get("deep_tensions", [])
        if tensions:
            prompt_parts.append(f"\nDeep tensions ({len(tensions)}):")
            for t in tensions[:5]:
                prompt_parts.append(
                    f'  - "{t.get("s1", "")}" vs "{t.get("s2", "")}"'
                )

        questions = curiosity_result.get("bridging_questions", [])
        if questions:
            prompt_parts.append("\nBridging questions:")
            for q in questions[:5]:
                prompt_parts.append(f"  - {q}")

        suggested = curiosity_result.get("suggested_action", "")
        if suggested:
            prompt_parts.append(f"\nSuggested focus: {suggested}")

        prompt_parts.append(
            "\nDiscuss these findings. What do they mean? What should we "
            "investigate or resolve? Be specific and actionable."
        )

        curiosity_prompt = "\n".join(prompt_parts)

        # Pick a random primary responder
        primary_model = random.choice(model_ids)
        system_prompt = self.bot.get_system_prompt(primary_model)

        try:
            messages = self.bot.conversation.build_messages_for_model(
                primary_model, system_prompt, limit=10,
            )
            # Inject the curiosity prompt as the latest user message
            messages.append({"role": "user", "content": curiosity_prompt})

            response = await self.bot.openrouter.chat(
                model=primary_model,
                messages=messages,
            )

            # Record in conversation history
            await self.bot.conversation.add_message(
                primary_model, response.content,
            )

            # Post to #nexus
            embed = MessageFormatter.format_response(
                primary_model, response.content,
            )
            last_msg = await self.bot.router.nexus.send(embed=embed)

            # Store in memory
            if self.bot.memory_store.is_connected:
                self.bot._spawn(
                    self._store_discussion_memory(
                        response.content, primary_model,
                    )
                )

            # Log to C2
            await self._log_to_c2(
                actor=primary_model,
                intent="curiosity_discussion",
                out=response.content[:500],
                tags=["curiosity", "autonomous", "swarm"],
            )

            # Run crosstalk reactions if enabled
            if self.bot.crosstalk.is_enabled and last_msg is not None:
                await self._run_curiosity_reactions(
                    primary_model, model_ids, last_msg,
                )

            # Post summary to #memory
            summary_embed = MessageFormatter.format_memory_log(
                "curiosity_discussion",
                f"Triggered by stress={stress:.3f}, "
                f"{len(contradictions)} contradiction(s), "
                f"{len(tensions)} tension(s). "
                f"Primary: {primary_model}.",
            )
            await self.bot.router.memory.send(embed=summary_embed)

        except Exception:
            log.error("Curiosity discussion failed.", exc_info=True)

    async def _run_curiosity_reactions(
        self,
        primary_model: str,
        model_ids: list[str],
        last_msg: Any,
    ) -> None:
        """Run crosstalk reactions for the curiosity discussion."""
        from nexus.channels.formatter import MessageFormatter
        from nexus.swarm.crosstalk import CrosstalkManager

        reaction_order = self.bot.crosstalk.build_reaction_order(
            primary_model, model_ids,
        )
        reaction_suffix = CrosstalkManager.get_reaction_suffix()
        reactions_posted = 0

        for reactor_id in reaction_order:
            if reactions_posted >= 2:
                break
            try:
                reactor_prompt = (
                    self.bot.get_system_prompt(reactor_id) + reaction_suffix
                )
                reactor_messages = (
                    self.bot.conversation.build_messages_for_model(
                        reactor_id, reactor_prompt, limit=10,
                    )
                )
                reaction = await asyncio.wait_for(
                    self.bot.openrouter.chat(
                        model=reactor_id, messages=reactor_messages,
                    ),
                    timeout=30.0,
                )
                if CrosstalkManager.is_pass(reaction.content):
                    continue

                await self.bot.conversation.add_message(
                    reactor_id, reaction.content,
                )
                embed = MessageFormatter.format_response(
                    reactor_id, reaction.content,
                )
                last_msg = await last_msg.reply(
                    embed=embed, mention_author=False,
                )
                reactions_posted += 1

                if self.bot.memory_store.is_connected:
                    self.bot._spawn(
                        self._store_discussion_memory(
                            reaction.content, reactor_id,
                        )
                    )

                await self._log_to_c2(
                    actor=reactor_id,
                    intent="curiosity_discussion",
                    out=reaction.content[:500],
                    tags=["curiosity", "autonomous", "swarm"],
                )

            except asyncio.TimeoutError:
                log.warning(
                    "Curiosity reaction from %s timed out.", reactor_id,
                )
            except Exception:
                log.error(
                    "Curiosity reaction from %s failed.",
                    reactor_id,
                    exc_info=True,
                )

    async def _store_discussion_memory(
        self, content: str, source: str,
    ) -> None:
        """Store a curiosity discussion response in vector memory."""
        try:
            vector = await self.bot.embeddings.embed_one(content)
            await self.bot.memory_store.store(
                content=content,
                vector=vector,
                source=source,
                channel="nexus",
                metadata={"type": "curiosity_discussion"},
            )
        except Exception:
            log.warning(
                "Failed to store curiosity discussion in memory.",
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # C2 event logging helper
    # ------------------------------------------------------------------

    async def _log_to_c2(
        self,
        actor: str,
        intent: str,
        inp: str = "",
        out: str = "",
        tags: list[str] | None = None,
    ) -> None:
        """Log an event to C2 if available.  Failures are silently ignored."""
        c2 = getattr(self.bot, "c2", None)
        if c2 is None or not c2.is_running:
            return
        try:
            await c2.write_event(actor=actor, intent=intent, inp=inp, out=out, tags=tags)
        except Exception:
            pass  # Non-critical — don't let logging failures propagate.

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

        # Build autonomy context for the decision model.
        gate: AutonomyGate | None = getattr(self.bot, "autonomy_gate", None)
        autonomy_hint = ""
        if gate is not None:
            autonomy_hint = (
                f"\n\nCurrent autonomy mode: {gate.mode.value}. "
                "In 'observe' mode, all actions will be proposed to the human first. "
                "In 'escalate' mode, research/summarize/analyze auto-execute but "
                "code/classify/extract require approval. "
                "In 'autopilot' mode, all actions auto-execute."
            )

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
                            + autonomy_hint
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
        """Build the decision prompt from gathered state."""
        parts: list[str] = [
            f"Timestamp: {state.get('timestamp', 'unknown')}",
            "",
            "=== Current Swarm State ===",
        ]

        # Recent conversation.
        recent_msgs: list[dict[str, Any]] = state.get("recent_messages", [])
        if recent_msgs:
            parts.append(f"\n--- Recent Conversation ({len(recent_msgs)} messages) ---")
            for msg in recent_msgs[-5:]:
                author = msg.get("author", "unknown")
                content = msg.get("content", "")[:1500]
                parts.append(f"  [{author}]: {content}")
        else:
            parts.append("\n--- Recent Conversation ---")
            parts.append("  (no recent messages)")

        # Relevant memories.
        memories: list[dict[str, Any]] = state.get("memories", [])
        if memories:
            parts.append(f"\n--- Relevant Memories ({len(memories)}) ---")
            for mem in memories[:3]:
                content = mem.get("content", "")[:1500]
                source = mem.get("source", "unknown")
                score = mem.get("score", 0.0)
                parts.append(f"  [{source}, relevance={score:.2f}]: {content}")

        # PiecesOS activity.
        activity: str | None = state.get("activity")
        if activity:
            parts.append("\n--- Recent User Activity (PiecesOS) ---")
            parts.append(f"  {activity[:500]}")

        # C2 curiosity signals.
        curiosity: dict[str, Any] | None = state.get("curiosity")
        if curiosity:
            parts.append("\n--- Epistemic Signals (C2 Curiosity) ---")
            stress = curiosity.get("stress_level", 0)
            parts.append(f"  Stress level: {stress:.3f}")
            contradictions = curiosity.get("contradictions", [])
            if contradictions:
                parts.append(f"  Contradictions ({len(contradictions)}):")
                for c in contradictions[:3]:
                    parts.append(f"    - {c.get('s1', '')[:80]} vs {c.get('s2', '')[:80]}")
            tensions = curiosity.get("deep_tensions", [])
            if tensions:
                parts.append(f"  Deep tensions ({len(tensions)}):")
                for t in tensions[:3]:
                    parts.append(f"    - {t.get('s1', '')[:80]} vs {t.get('s2', '')[:80]}")
            questions = curiosity.get("bridging_questions", [])
            if questions:
                parts.append(f"  Bridging questions ({len(questions)}):")
                for q in questions[:3]:
                    parts.append(f"    - {q[:120]}")
            suggested = curiosity.get("suggested_action", "")
            if suggested:
                parts.append(f"  Suggested focus: {suggested}")

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
