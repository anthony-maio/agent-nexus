"""Expanded trigger system for the Agent Nexus orchestrator.

The :class:`TriggerManager` replaces the standalone
:class:`~nexus.orchestrator.activity.ActivityMonitor` with a pluggable
system that supports multiple trigger sources.  Each trigger checks a
condition periodically and can signal the orchestrator to run a
mini-cycle when the condition fires.

Built-in triggers:

- **ActivityTrigger** -- PiecesOS activity change detection (subsumes
  the original ``ActivityMonitor``).
- **MessageRateTrigger** -- Fires when N messages arrive in M minutes.
- **GoalStaleTrigger** -- Fires when an active goal has had no progress
  for X hours.
- **ScheduledTrigger** -- Fires at cron-like intervals (e.g. every 6h).

Usage::

    manager = TriggerManager(bot, check_interval=30)
    manager.add_trigger(MessageRateTrigger(threshold=10, window_minutes=5))
    manager.add_trigger(GoalStaleTrigger(stale_hours=6))
    await manager.start()
"""

from __future__ import annotations

import abc
import asyncio
import hashlib
import logging
import time
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base trigger
# ---------------------------------------------------------------------------


class BaseTrigger(abc.ABC):
    """Abstract base for orchestrator triggers."""

    name: str = "base"

    @abc.abstractmethod
    async def check(self, bot: Any) -> bool:
        """Return ``True`` if this trigger should fire."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ---------------------------------------------------------------------------
# Concrete triggers
# ---------------------------------------------------------------------------


class ActivityTrigger(BaseTrigger):
    """Fires when PiecesOS activity changes (hash comparison)."""

    name = "activity"

    def __init__(self) -> None:
        self._last_hash: str | None = None

    async def check(self, bot: Any) -> bool:
        pieces = getattr(bot, "pieces", None)
        if pieces is None or not getattr(pieces, "is_connected", False):
            return False

        try:
            activity: str | None = await asyncio.wait_for(
                pieces.get_recent_activity(), timeout=20.0,
            )
        except (asyncio.TimeoutError, Exception):
            return False

        if activity is None:
            return False

        h = hashlib.sha256(activity.encode()).hexdigest()[:16]
        if self._last_hash is None:
            self._last_hash = h
            return False

        if h != self._last_hash:
            self._last_hash = h
            return True
        return False


class MessageRateTrigger(BaseTrigger):
    """Fires when the conversation receives N messages within a time window."""

    name = "message_rate"

    def __init__(
        self, threshold: int = 10, window_minutes: float = 5.0,
    ) -> None:
        self.threshold = threshold
        self.window_seconds = window_minutes * 60
        self._last_check_count: int = 0

    async def check(self, bot: Any) -> bool:
        conversation = getattr(bot, "conversation", None)
        if conversation is None:
            return False

        from datetime import datetime, timedelta, timezone

        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=self.window_seconds
        )
        recent = [
            m
            for m in conversation.get_history(limit=50)
            if m.timestamp >= cutoff
        ]

        if len(recent) >= self.threshold and len(recent) > self._last_check_count:
            self._last_check_count = len(recent)
            return True

        self._last_check_count = len(recent)
        return False

    def __repr__(self) -> str:
        return (
            f"MessageRateTrigger(threshold={self.threshold}, "
            f"window={self.window_seconds}s)"
        )


class GoalStaleTrigger(BaseTrigger):
    """Fires when any active goal hasn't had progress for X hours.

    Tracks which goal IDs have already triggered so a single stale goal
    doesn't cause continuous mini-cycles on every poll.  The suppression
    resets when the goal's ``updated_at`` changes (i.e. it receives new
    progress) or when it is pruned/completed.
    """

    name = "goal_stale"

    def __init__(self, stale_hours: float = 6.0) -> None:
        self.stale_hours = stale_hours
        # Maps goal_id -> updated_at timestamp that was stale when we fired.
        # If the goal gets new activity (updated_at changes), we re-fire.
        self._fired_goals: dict[str, str] = {}

    async def check(self, bot: Any) -> bool:
        goal_store = getattr(bot, "goal_store", None)
        if goal_store is None:
            return False

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        active = await goal_store.get_active_goals()

        # Clean out fired entries for goals no longer active.
        active_ids = {g.id for g in active}
        self._fired_goals = {
            gid: ts for gid, ts in self._fired_goals.items()
            if gid in active_ids
        }

        for goal in active:
            try:
                updated = datetime.fromisoformat(goal.updated_at)
                age_hours = (now - updated).total_seconds() / 3600
                if age_hours > self.stale_hours:
                    # Only fire if we haven't already fired for this
                    # goal at this updated_at timestamp.
                    prev = self._fired_goals.get(goal.id)
                    if prev != goal.updated_at:
                        self._fired_goals[goal.id] = goal.updated_at
                        return True
            except (ValueError, TypeError):
                continue

        return False

    def __repr__(self) -> str:
        return f"GoalStaleTrigger(stale_hours={self.stale_hours})"


class ScheduledTrigger(BaseTrigger):
    """Fires at a fixed interval (e.g. every 6 hours)."""

    name = "scheduled"

    def __init__(self, interval_hours: float = 6.0) -> None:
        self.interval_seconds = interval_hours * 3600
        self._last_fire: float = time.monotonic()

    async def check(self, bot: Any) -> bool:
        elapsed = time.monotonic() - self._last_fire
        if elapsed >= self.interval_seconds:
            self._last_fire = time.monotonic()
            return True
        return False

    def __repr__(self) -> str:
        return f"ScheduledTrigger(interval={self.interval_seconds}s)"


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class TriggerManager:
    """Manages multiple trigger sources and signals the orchestrator.

    Runs a background loop that checks all registered triggers
    periodically.  When any trigger fires, the orchestrator is woken
    for a mini-cycle.

    Args:
        bot: The ``NexusBot`` instance.
        check_interval: Seconds between trigger checks.
    """

    def __init__(self, bot: Any, check_interval: int = 30) -> None:
        self.bot = bot
        self.check_interval: int = max(check_interval, 5)
        self._triggers: list[BaseTrigger] = []
        self._running: bool = False
        self._task: asyncio.Task[None] | None = None
        self._fire_counts: dict[str, int] = {}

    def add_trigger(self, trigger: BaseTrigger) -> None:
        """Register a trigger source (idempotent by name)."""
        # Prevent duplicate registrations on Discord reconnect.
        for existing in self._triggers:
            if existing.name == trigger.name:
                log.debug("Trigger %s already registered — skipping.", trigger.name)
                return
        self._triggers.append(trigger)
        self._fire_counts.setdefault(trigger.name, 0)
        log.info("Trigger registered: %s", trigger)

    async def start(self) -> None:
        """Start the background trigger check loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="trigger-manager")
        log.info(
            "TriggerManager started (%d triggers, interval=%ds).",
            len(self._triggers),
            self.check_interval,
        )

    async def stop(self) -> None:
        """Stop the trigger check loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("TriggerManager stopped. Fire counts: %s", self._fire_counts)

    async def _loop(self) -> None:
        """Main loop: check all triggers, fire orchestrator on any hit."""
        await asyncio.sleep(15.0)  # Let bot finish bootstrapping

        while self._running:
            try:
                for trigger in self._triggers:
                    try:
                        fired = await asyncio.wait_for(
                            trigger.check(self.bot), timeout=10.0,
                        )
                        if fired:
                            self._fire_counts[trigger.name] = (
                                self._fire_counts.get(trigger.name, 0) + 1
                            )
                            log.info("Trigger fired: %s", trigger.name)
                            orchestrator = getattr(
                                self.bot, "orchestrator", None,
                            )
                            if orchestrator is not None:
                                await orchestrator.trigger_cycle()
                            break  # One trigger per check cycle is enough
                    except asyncio.TimeoutError:
                        log.debug("Trigger %s timed out.", trigger.name)
                    except Exception:
                        log.warning(
                            "Trigger %s check failed.", trigger.name,
                            exc_info=True,
                        )
            except Exception:
                log.warning("TriggerManager loop error.", exc_info=True)

            if self._running:
                await asyncio.sleep(self.check_interval)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def fire_counts(self) -> dict[str, int]:
        return dict(self._fire_counts)

    def __repr__(self) -> str:
        status = "running" if self._running else "stopped"
        return (
            f"TriggerManager(status={status!r}, "
            f"triggers={len(self._triggers)}, "
            f"fires={self._fire_counts})"
        )
