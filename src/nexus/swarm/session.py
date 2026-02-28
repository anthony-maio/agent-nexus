"""Session lifecycle management for Agent Nexus.

Handles session start/stop summaries and context restoration so the
swarm maintains continuity across restarts.  Session events are persisted
to C2 and optionally announced in ``#nexus``.

Usage::

    from nexus.swarm.session import SessionManager

    session = SessionManager(bot)
    await session.on_startup()   # Recall last session context
    await session.on_shutdown()  # Summarise and persist current session
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)


class SessionManager:
    """Manages session lifecycle events for swarm continuity.

    At **startup**, loads the most recent session summary from C2 so the
    swarm has context about what happened last time.

    At **shutdown**, generates a summary of the current session (message
    count, models active, key topics) and writes it to C2 so the next
    session can pick up where this one left off.

    Args:
        bot: The ``NexusBot`` instance.
    """

    def __init__(self, bot: Any) -> None:
        self.bot = bot
        self._start_time: float = time.monotonic()
        self._last_session_summary: str | None = None

    # ------------------------------------------------------------------
    # Startup — restore previous session context
    # ------------------------------------------------------------------

    async def on_startup(self) -> str | None:
        """Load the last session summary from C2 and announce it.

        Returns the summary text, or ``None`` if unavailable.
        """
        self._start_time = time.monotonic()

        if not self.bot.c2.is_running:
            return None

        try:
            result = await self.bot.c2.events(limit=30)
            if result is None:
                return None

            events = result.get("events", [])
            # Find the most recent session_end event
            for evt in reversed(events):
                if evt.get("intent") == "session_end":
                    summary = evt.get("output", "")
                    if summary:
                        self._last_session_summary = summary
                        log.info("Restored previous session summary (%d chars)", len(summary))
                        return summary
        except Exception:
            log.debug("Could not restore previous session context.", exc_info=True)

        return None

    async def announce_restore(self) -> None:
        """Post the restored session context to #nexus if available."""
        if not self._last_session_summary:
            return

        try:
            from nexus.channels.formatter import MessageFormatter

            embed = MessageFormatter.format_memory_log(
                "session:restore",
                f"**Previous Session Context:**\n{self._last_session_summary[:2000]}",
            )
            await self.bot.router.nexus.send(embed=embed)
        except Exception:
            log.debug("Failed to announce session restore.", exc_info=True)

    @property
    def last_session_summary(self) -> str | None:
        """The summary from the previous session, if loaded."""
        return self._last_session_summary

    # ------------------------------------------------------------------
    # Shutdown — summarise and persist
    # ------------------------------------------------------------------

    async def on_shutdown(self) -> str | None:
        """Generate a session summary and persist to C2.

        Returns the summary text, or ``None`` if C2 is not available.
        """
        uptime_sec = time.monotonic() - self._start_time
        uptime_min = uptime_sec / 60

        conversation = self.bot.conversation
        history = conversation.get_history(limit=50)

        # Gather session stats
        model_ids = list(self.bot.swarm_models.keys())
        message_count = conversation.message_count
        human_msgs = sum(1 for m in history if m.is_human)
        model_msgs = message_count - human_msgs

        # Extract recent topics from human messages
        human_contents = [
            m.content[:200] for m in history if m.is_human
        ][-5:]
        topics = "; ".join(human_contents) if human_contents else "No human messages"

        # Build summary
        summary = (
            f"Session ran for {uptime_min:.0f} minutes. "
            f"{message_count} messages ({human_msgs} human, {model_msgs} model). "
            f"Models active: {len(model_ids)}. "
            f"Recent topics: {topics[:500]}"
        )

        # Cost tracking
        cost = getattr(self.bot.openrouter, "session_cost", 0)
        if cost > 0:
            summary += f" Session cost: ${cost:.4f}."

        # Goal status
        try:
            goals = await self.bot.goal_store.get_active_goals()
            if goals:
                summary += f" {len(goals)} active goal(s) at shutdown."
        except Exception:
            pass

        # Sentiment summary
        mood = self.bot.sentiment.current_mood
        avg_score = self.bot.sentiment.average_score
        summary += f" User mood trend: {mood.value} (avg={avg_score:.2f})."

        # Persist to C2
        if self.bot.c2.is_running:
            try:
                await self.bot.c2.write_event(
                    actor="system",
                    intent="session_end",
                    inp=f"uptime={uptime_min:.0f}m messages={message_count}",
                    out=summary,
                    tags=["session", "shutdown"],
                )
                log.info("Session summary persisted to C2 (%d chars)", len(summary))
            except Exception:
                log.debug("Failed to persist session summary.", exc_info=True)

        return summary
