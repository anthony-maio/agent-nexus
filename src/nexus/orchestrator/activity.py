"""Activity monitor for PiecesOS polling and change detection.

Polls PiecesOS at a configurable interval (default: 60s) and detects when
activity has meaningfully changed.  On change, signals the orchestrator
to run a mini-cycle (gather + decide, skip maintenance).

The monitor does NOT replace the existing Pieces query in StateGatherer --
it augments it with change detection so the orchestrator can react to new
user activity between hourly full cycles.

Usage::

    monitor = ActivityMonitor(bot, poll_interval=60)
    await monitor.start()
    ...
    if monitor.has_new_activity:
        activity = monitor.get_latest_activity()
    ...
    await monitor.stop()
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass  # forward references resolved at runtime

log = logging.getLogger(__name__)


class ActivityMonitor:
    """Polls PiecesOS for activity changes and triggers orchestrator mini-cycles.

    Hashes the Pieces response to detect meaningful changes.  When a change
    is detected, it sets a flag and signals the orchestrator loop to wake up.

    Args:
        bot: The ``NexusBot`` instance.
        poll_interval: Seconds between Pieces polls.  Defaults to 60.
    """

    def __init__(self, bot: Any, poll_interval: int = 60) -> None:
        self.bot = bot
        self.poll_interval: int = max(poll_interval, 10)
        self._last_activity_hash: str | None = None
        self._last_activity: str | None = None
        self._new_activity: bool = False
        self._running: bool = False
        self._task: asyncio.Task[None] | None = None
        self._poll_count: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background polling loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="activity-monitor")
        log.info(
            "Activity monitor started (poll_interval=%ds).", self.poll_interval,
        )

    async def stop(self) -> None:
        """Stop the background polling loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info(
            "Activity monitor stopped after %d poll(s).", self._poll_count,
        )

    # ------------------------------------------------------------------
    # Polling loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        """Main polling loop: query Pieces, detect changes, signal orchestrator."""
        # Small initial delay to let the bot finish bootstrapping.
        await asyncio.sleep(10.0)

        while self._running:
            try:
                await self._poll_cycle()
            except Exception:
                log.warning("Activity poll cycle failed.", exc_info=True)

            if self._running:
                await asyncio.sleep(self.poll_interval)

    async def _poll_cycle(self) -> None:
        """Single poll cycle: query Pieces and compare hash."""
        self._poll_count += 1
        pieces = getattr(self.bot, "pieces", None)
        if pieces is None or not getattr(pieces, "is_connected", False):
            return

        activity: str | None = await pieces.get_recent_activity()
        if activity is None:
            return

        activity_hash = hashlib.sha256(activity.encode()).hexdigest()[:16]

        if self._last_activity_hash is None:
            # First poll — store baseline, don't trigger.
            self._last_activity_hash = activity_hash
            self._last_activity = activity
            log.debug("Activity baseline set (hash=%s).", activity_hash)
            return

        if activity_hash != self._last_activity_hash:
            log.info(
                "Activity change detected (hash %s -> %s).",
                self._last_activity_hash,
                activity_hash,
            )
            self._last_activity_hash = activity_hash
            self._last_activity = activity
            self._new_activity = True

            # Signal the orchestrator to run a mini-cycle.
            orchestrator = getattr(self.bot, "orchestrator", None)
            if orchestrator is not None and hasattr(orchestrator, "trigger_cycle"):
                await orchestrator.trigger_cycle()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def has_new_activity(self) -> bool:
        """Whether new activity has been detected since last check."""
        return self._new_activity

    def get_latest_activity(self) -> str | None:
        """Return the latest activity text and clear the new-activity flag."""
        self._new_activity = False
        return self._last_activity

    @property
    def is_running(self) -> bool:
        """Whether the monitor is currently polling."""
        return self._running

    @property
    def poll_count(self) -> int:
        """Total number of poll cycles completed."""
        return self._poll_count

    def __repr__(self) -> str:
        status = "running" if self._running else "stopped"
        return (
            f"ActivityMonitor(status={status!r}, "
            f"interval={self.poll_interval}s, "
            f"polls={self._poll_count})"
        )
