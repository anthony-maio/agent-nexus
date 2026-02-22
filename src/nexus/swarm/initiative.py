"""Self-initiated swarm conversations for Agent Nexus.

The :class:`SwarmInitiative` system allows the swarm to proactively start
discussions without human prompting.  Triggers include:

- **Significant task results** -- A dispatched task returns high-impact findings.
- **Goal milestones** -- A goal completes or a majority of its tasks finish.
- **Curiosity findings** -- Contradictions or knowledge voids from C2.
- **Periodic reflection** -- The swarm hasn't talked in a while and should
  reflect on its state.

This generalises the existing ``_trigger_curiosity_discussion`` pattern from
:mod:`nexus.orchestrator.loop` into a reusable system that can be triggered
from multiple places.

Usage::

    initiative = SwarmInitiative(bot, cooldown_minutes=30)
    await initiative.maybe_initiate(
        reason="task_result",
        context="Deep analysis of user's codebase revealed 3 architectural issues.",
    )
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

log = logging.getLogger(__name__)


class SwarmInitiative:
    """Manages proactive, self-initiated swarm conversations.

    Uses a cooldown to prevent the swarm from being too chatty.  Each
    initiation picks a random model as the primary speaker, posts to
    ``#nexus``, and optionally runs a reaction round.

    Args:
        bot: The ``NexusBot`` instance.
        cooldown_minutes: Minimum minutes between self-initiated conversations.
        enabled: Whether the initiative system is active.
    """

    def __init__(
        self,
        bot: Any,
        cooldown_minutes: float = 30.0,
        enabled: bool = True,
    ) -> None:
        self.bot = bot
        self.cooldown_seconds: float = cooldown_minutes * 60
        self.enabled: bool = enabled
        self._last_initiative: float = 0.0
        self._initiative_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def maybe_initiate(
        self,
        reason: str,
        context: str,
        force: bool = False,
    ) -> bool:
        """Attempt to start a self-initiated swarm conversation.

        Args:
            reason: Why this conversation is being started (e.g.
                ``"task_result"``, ``"goal_milestone"``, ``"curiosity"``,
                ``"reflection"``).
            context: The content/findings to discuss.
            force: Bypass the cooldown check.

        Returns:
            ``True`` if a conversation was initiated, ``False`` if skipped
            due to cooldown, being disabled, or missing models.
        """
        if not self.enabled:
            return False

        if not force and not self._cooldown_elapsed():
            log.debug(
                "Initiative skipped (cooldown): reason=%s", reason,
            )
            return False

        model_ids = list(self.bot.swarm_models.keys())
        if not model_ids:
            return False

        self._last_initiative = time.monotonic()
        self._initiative_count += 1

        log.info(
            "Self-initiated conversation #%d (reason=%s)",
            self._initiative_count,
            reason,
        )

        await self._run_initiative(reason, context, model_ids)
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cooldown_elapsed(self) -> bool:
        if self._last_initiative == 0.0:
            return True
        return (time.monotonic() - self._last_initiative) >= self.cooldown_seconds

    async def _run_initiative(
        self,
        reason: str,
        context: str,
        model_ids: list[str],
    ) -> None:
        """Execute a self-initiated conversation."""
        from nexus.channels.formatter import MessageFormatter

        prompt = self._build_initiative_prompt(reason, context)

        # Pick a model suited to the reason if possible, else random
        primary_model = self._select_initiator(reason, model_ids)
        system_prompt = self.bot.get_system_prompt(primary_model)

        try:
            messages = self.bot.conversation.build_messages_for_model(
                primary_model, system_prompt, limit=10,
            )
            messages.append({"role": "user", "content": prompt})

            response = await asyncio.wait_for(
                self.bot.openrouter.chat(
                    model=primary_model, messages=messages,
                ),
                timeout=60.0,
            )

            await self.bot.conversation.add_message(
                primary_model, response.content,
            )

            embed = MessageFormatter.format_response(
                primary_model, response.content,
            )
            last_msg = await self.bot.router.nexus.send(embed=embed)

            # Store in memory
            if self.bot.memory_store.is_connected:
                self.bot._spawn(
                    self._store_memory(response.content, primary_model)
                )

            # Log to C2
            self.bot._spawn(self._log_to_c2(
                primary_model, reason, response.content,
            ))

            # Run reaction round
            if self.bot.crosstalk.is_enabled and last_msg is not None:
                await self._run_reactions(
                    primary_model, model_ids, last_msg,
                )

            # Post summary to #memory
            summary_embed = MessageFormatter.format_memory_log(
                f"initiative:{reason}",
                f"Self-initiated by {primary_model}. "
                f"Context: {context[:150]}",
            )
            await self.bot.router.memory.send(embed=summary_embed)

        except Exception:
            log.error(
                "Self-initiated conversation failed (reason=%s).",
                reason,
                exc_info=True,
            )

    def _select_initiator(self, reason: str, model_ids: list[str]) -> str:
        """Pick the most appropriate model to lead the conversation.

        Uses model strengths from the registry when a clear match exists,
        otherwise picks randomly.
        """
        # Map reasons to useful model strengths
        strength_hints: dict[str, list[str]] = {
            "task_result": ["reasoning", "analysis"],
            "goal_milestone": ["agentic-planning", "reasoning"],
            "curiosity": ["reasoning", "analysis"],
            "reflection": ["general-intelligence", "creativity"],
        }

        desired = strength_hints.get(reason, [])
        if desired:
            scored: list[tuple[str, int]] = []
            for mid in model_ids:
                spec = self.bot.swarm_models.get(mid)
                if spec is None:
                    scored.append((mid, 0))
                    continue
                score = sum(
                    1 for s in spec.strengths if s in desired
                )
                scored.append((mid, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            if scored[0][1] > 0:
                # Pick from top-scoring models with some randomness
                top_score = scored[0][1]
                top_models = [m for m, s in scored if s == top_score]
                return random.choice(top_models)

        return random.choice(model_ids)

    def _build_initiative_prompt(self, reason: str, context: str) -> str:
        """Build the prompt that kicks off the self-initiated discussion."""
        reason_intros: dict[str, str] = {
            "task_result": (
                "A task agent has returned significant findings that warrant "
                "swarm discussion."
            ),
            "goal_milestone": (
                "A goal has reached a milestone. Review progress and decide "
                "next steps."
            ),
            "curiosity": (
                "The Continuity Core has detected epistemic tensions in the "
                "knowledge base."
            ),
            "reflection": (
                "It's been a while since the swarm last reflected on its "
                "state and goals. Take stock of where things stand."
            ),
        }

        intro = reason_intros.get(
            reason,
            "New information has surfaced that the swarm should discuss.",
        )

        return (
            f"{intro}\n\n"
            f"Context:\n{context}\n\n"
            "Discuss these findings with the swarm. What do they mean? "
            "What actions should we take? Be specific and actionable. "
            "If you think a new goal or task should be created, say so explicitly."
        )

    async def _run_reactions(
        self,
        primary_model: str,
        model_ids: list[str],
        last_msg: Any,
    ) -> None:
        """Run crosstalk reactions for the initiative discussion."""
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
                        self._store_memory(reaction.content, reactor_id)
                    )

            except asyncio.TimeoutError:
                log.warning(
                    "Initiative reaction from %s timed out.", reactor_id,
                )
            except Exception:
                log.error(
                    "Initiative reaction from %s failed.",
                    reactor_id,
                    exc_info=True,
                )

    async def _store_memory(self, content: str, source: str) -> None:
        """Store initiative content in vector memory."""
        try:
            vector = await self.bot.embeddings.embed_one(content)
            await self.bot.memory_store.store(
                content=content,
                vector=vector,
                source=source,
                channel="nexus",
                metadata={"type": "initiative"},
            )
        except Exception:
            log.warning("Failed to store initiative in memory.", exc_info=True)

    async def _log_to_c2(
        self, actor: str, reason: str, content: str,
    ) -> None:
        """Log initiative to C2 if available."""
        if not self.bot.c2.is_running:
            return
        try:
            await self.bot.c2.write_event(
                actor=actor,
                intent=f"initiative:{reason}",
                out=content[:500],
                tags=["initiative", "autonomous", reason],
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def initiative_count(self) -> int:
        return self._initiative_count

    def __repr__(self) -> str:
        return (
            f"SwarmInitiative(enabled={self.enabled}, "
            f"cooldown={self.cooldown_seconds}s, "
            f"count={self._initiative_count})"
        )
