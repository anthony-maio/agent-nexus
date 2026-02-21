"""Three-channel Discord architecture for Agent Nexus.

The bot operates across exactly three channels that are auto-created on first
boot if they do not already exist in the target guild:

* ``#human``  -- User interaction.  Human talks to the swarm, receives flagged
  alerts and approval prompts.
* ``#nexus``  -- Main forum.  All models converse here.  They see each other's
  messages.
* ``#memory`` -- Audit trail.  Memory writes, activity logs, daily summaries.
  Read-only for models.

Usage::

    router = ChannelRouter()
    await router.ensure_channels(guild)
    await router.human.send("Hello from the swarm!")
"""

from __future__ import annotations

import logging
from typing import Final

import discord

log: Final = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Channel specifications
# ---------------------------------------------------------------------------

CHANNEL_SPECS: Final[dict[str, dict[str, str]]] = {
    "human": {
        "name": "human",
        "topic": "Talk to the swarm. Ask questions, give tasks, approve actions.",
    },
    "nexus": {
        "name": "nexus",
        "topic": "AI models collaborate here. Watch the swarm think.",
    },
    "memory": {
        "name": "memory",
        "topic": "Memory writes, activity logs, daily summaries. Audit trail.",
    },
    "logs": {
        "name": "logs",
        "topic": "Bot operational logs. Orchestrator cycles, C2 events, errors.",
    },
}


class ChannelRouter:
    """Manages the 3-channel architecture for Agent Nexus.

    The router is responsible for locating or creating the three required
    Discord text channels and exposing typed accessors so the rest of the
    codebase never needs to hard-code channel names or IDs.

    Attributes:
        channels: Mapping of logical channel key (``"human"``, ``"nexus"``,
            ``"memory"``) to the resolved :class:`discord.TextChannel`.
    """

    def __init__(self) -> None:
        self.channels: dict[str, discord.TextChannel] = {}
        self._ready: bool = False

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    async def ensure_channels(self, guild: discord.Guild) -> None:
        """Create channels if they do not exist.  Call on bot startup.

        For each entry in :data:`CHANNEL_SPECS` the method first checks
        whether the guild already contains a text channel with the expected
        name.  If it does, the existing channel is reused; otherwise a new
        one is created with the configured topic.

        Args:
            guild: The Discord guild to provision channels in.
        """
        existing: dict[str, discord.TextChannel] = {
            ch.name: ch for ch in guild.text_channels
        }

        for key, spec in CHANNEL_SPECS.items():
            name = spec["name"]
            if name in existing:
                self.channels[key] = existing[name]
                log.info("Found existing channel: #%s", name)
            else:
                self.channels[key] = await guild.create_text_channel(
                    name=name,
                    topic=spec["topic"],
                    reason="Agent Nexus auto-setup",
                )
                log.info("Created channel: #%s", name)

        self._ready = True

    # ------------------------------------------------------------------
    # Typed accessors
    # ------------------------------------------------------------------

    @property
    def human(self) -> discord.TextChannel:
        """The ``#human`` channel for user interaction."""
        if not self._ready:
            raise RuntimeError("ChannelRouter not initialized. Call ensure_channels() first.")
        return self.channels["human"]

    @property
    def nexus(self) -> discord.TextChannel:
        """The ``#nexus`` channel where models collaborate."""
        if not self._ready:
            raise RuntimeError("ChannelRouter not initialized. Call ensure_channels() first.")
        return self.channels["nexus"]

    @property
    def memory(self) -> discord.TextChannel:
        """The ``#memory`` channel for audit-trail logging."""
        if not self._ready:
            raise RuntimeError("ChannelRouter not initialized. Call ensure_channels() first.")
        return self.channels["memory"]

    @property
    def logs(self) -> discord.TextChannel:
        """The ``#logs`` channel for operational logging."""
        if not self._ready:
            raise RuntimeError("ChannelRouter not initialized. Call ensure_channels() first.")
        return self.channels["logs"]

    # ------------------------------------------------------------------
    # Channel identification helpers
    # ------------------------------------------------------------------

    def is_human_channel(self, channel_id: int) -> bool:
        """Return ``True`` if *channel_id* matches ``#human``."""
        return self.channels.get("human") is not None and self.channels["human"].id == channel_id

    def is_nexus_channel(self, channel_id: int) -> bool:
        """Return ``True`` if *channel_id* matches ``#nexus``."""
        return self.channels.get("nexus") is not None and self.channels["nexus"].id == channel_id

    def is_memory_channel(self, channel_id: int) -> bool:
        """Return ``True`` if *channel_id* matches ``#memory``."""
        return self.channels.get("memory") is not None and self.channels["memory"].id == channel_id

    def is_bot_channel(self, channel_id: int) -> bool:
        """Return ``True`` if *channel_id* belongs to any managed channel."""
        return any(ch.id == channel_id for ch in self.channels.values())
