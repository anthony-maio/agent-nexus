"""Autonomy mode gate for the Agent Nexus orchestrator.

Three configurable modes control how the swarm handles task dispatch:

- **observe**: Proposes actions in #human and waits for user approval.
- **escalate**: Low-risk tasks auto-execute; high-risk tasks get escalated.
- **autopilot**: Everything auto-executes unless the model itself is uncertain.

Usage::

    from nexus.orchestrator.autonomy import AutonomyGate, AutonomyMode

    gate = AutonomyGate(AutonomyMode.ESCALATE)
    if gate.should_auto_execute(action):
        await dispatcher.dispatch(action)
    elif gate.should_escalate(action):
        approved = await gate.propose_and_wait(bot, action)
        if approved:
            await dispatcher.dispatch(action)
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Any

import discord

log = logging.getLogger(__name__)


class AutonomyMode(Enum):
    """Autonomy levels for orchestrator task dispatch."""

    OBSERVE = "observe"
    ESCALATE = "escalate"
    AUTOPILOT = "autopilot"


# Action types classified by risk level.
_LOW_RISK_TYPES = frozenset({"research", "summarize", "analyze"})
_HIGH_RISK_TYPES = frozenset({"code", "classify", "extract"})


class AutonomyGate:
    """Controls whether actions auto-execute or require human approval.

    Args:
        mode: The initial autonomy mode.
    """

    _PROPOSAL_TIMEOUT: float = 300.0  # 5 minutes

    def __init__(self, mode: AutonomyMode = AutonomyMode.ESCALATE) -> None:
        self.mode: AutonomyMode = mode

    def set_mode(self, mode: AutonomyMode | str) -> None:
        """Set the autonomy mode.  Accepts enum or string."""
        if isinstance(mode, str):
            mode = AutonomyMode(mode.lower())
        self.mode = mode
        log.info("Autonomy mode set to: %s", self.mode.value)

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def should_auto_execute(self, action: dict[str, Any]) -> bool:
        """Return ``True`` if the action should be dispatched without asking."""
        if self.mode == AutonomyMode.AUTOPILOT:
            return True

        if self.mode == AutonomyMode.ESCALATE:
            action_type = action.get("type", "")
            return action_type in _LOW_RISK_TYPES

        # OBSERVE mode: never auto-execute.
        return False

    def should_escalate(self, action: dict[str, Any]) -> bool:
        """Return ``True`` if the action should be proposed to the human."""
        if self.mode == AutonomyMode.OBSERVE:
            return True

        if self.mode == AutonomyMode.ESCALATE:
            action_type = action.get("type", "")
            return action_type in _HIGH_RISK_TYPES

        # AUTOPILOT: don't escalate (unless model expressed uncertainty,
        # but that's handled at the decision-engine level).
        return False

    # ------------------------------------------------------------------
    # Human proposal flow
    # ------------------------------------------------------------------

    async def propose_and_wait(
        self,
        bot: Any,
        action: dict[str, Any],
        timeout: float | None = None,
    ) -> bool:
        """Post a proposal to #human and wait for user reaction.

        Posts an embed describing the proposed action.  The user can react
        with a thumbs-up to approve or thumbs-down to reject.  If no
        reaction is received within the timeout (default 5 min), the
        action is skipped.

        Returns ``True`` if the user approved, ``False`` otherwise.
        """
        timeout = timeout or self._PROPOSAL_TIMEOUT
        router = getattr(bot, "router", None)
        if router is None or router.human is None:
            log.warning("Cannot propose action: #human channel not available.")
            return False

        embed = discord.Embed(
            title="Action Proposal",
            description=(
                f"**Type:** {action.get('type', 'unknown')}\n"
                f"**Priority:** {action.get('priority', 'medium')}\n"
                f"**Description:** {action.get('description', 'No description')}"
            ),
            color=0xF39C12,
        )
        embed.set_footer(text="React with \U0001f44d to approve or \U0001f44e to reject")

        msg = await router.human.send(embed=embed)
        await msg.add_reaction("\U0001f44d")
        await msg.add_reaction("\U0001f44e")

        def check(reaction: discord.Reaction, user: discord.User) -> bool:
            return (
                reaction.message.id == msg.id
                and user != bot.user
                and str(reaction.emoji) in ("\U0001f44d", "\U0001f44e")
            )

        try:
            reaction, _user = await bot.wait_for(
                "reaction_add", check=check, timeout=timeout,
            )
            approved = str(reaction.emoji) == "\U0001f44d"
            status = "approved" if approved else "rejected"
            log.info("Action %s by user: %s", status, action.get("description", ""))
            return approved

        except asyncio.TimeoutError:
            log.info(
                "Action proposal timed out (%.0fs): %s",
                timeout,
                action.get("description", ""),
            )
            await msg.edit(embed=embed.set_footer(text="Timed out - action skipped"))
            return False

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"AutonomyGate(mode={self.mode.value!r})"
