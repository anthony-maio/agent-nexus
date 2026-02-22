"""Self-monitoring and health dashboard for Agent Nexus.

The :class:`HealthMonitor` runs periodic checks on all subsystems and
exposes a :class:`HealthSnapshot` for the admin command and #logs channel.
It detects stalls, disconnects, and cost overruns, and can trigger
recovery actions.

Usage::

    monitor = HealthMonitor(bot, check_interval=600)
    await monitor.start()
    snapshot = await monitor.check_now()
    print(snapshot)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import discord

log = logging.getLogger(__name__)


@dataclass
class HealthSnapshot:
    """Point-in-time health status of all Agent Nexus subsystems."""

    timestamp: str = ""
    # Orchestrator
    orchestrator_running: bool = False
    orchestrator_cycles: int = 0
    orchestrator_last_cycle_ago_seconds: float = -1.0
    # Memory
    memory_connected: bool = False
    # C2
    c2_running: bool = False
    # Models
    swarm_model_count: int = 0
    # Dispatcher
    dispatch_success: int = 0
    dispatch_failures: int = 0
    dispatch_success_rate: float = 1.0
    # Goals
    active_goals: int = 0
    # Triggers
    trigger_manager_running: bool = False
    trigger_fire_counts: dict[str, int] = field(default_factory=dict)
    # Alerts
    alerts: list[str] = field(default_factory=list)
    # Cost
    session_cost_usd: float = 0.0
    cost_limit_usd: float = 10.0

    @property
    def is_healthy(self) -> bool:
        return len(self.alerts) == 0


class HealthMonitor:
    """Periodic health checks and self-monitoring.

    Args:
        bot: The ``NexusBot`` instance.
        check_interval: Seconds between automatic health checks.
        stall_threshold: Seconds since last cycle before alerting.
    """

    def __init__(
        self,
        bot: Any,
        check_interval: int = 600,
        stall_threshold: int = 7200,
    ) -> None:
        self.bot = bot
        self.check_interval: int = max(check_interval, 60)
        self.stall_threshold: int = stall_threshold
        self._running: bool = False
        self._task: asyncio.Task[None] | None = None
        self._last_snapshot: HealthSnapshot | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="health-monitor")
        log.info("HealthMonitor started (interval=%ds).", self.check_interval)

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("HealthMonitor stopped.")

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------

    async def check_now(self) -> HealthSnapshot:
        """Run a health check immediately and return the snapshot."""
        snap = HealthSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        alerts: list[str] = []

        # Orchestrator
        orch = getattr(self.bot, "orchestrator", None)
        if orch is not None:
            snap.orchestrator_running = orch.is_running
            snap.orchestrator_cycles = orch.cycles_completed
            if orch.last_cycle_time is not None:
                ago = (
                    datetime.now(timezone.utc) - orch.last_cycle_time
                ).total_seconds()
                snap.orchestrator_last_cycle_ago_seconds = ago
                if ago > self.stall_threshold:
                    alerts.append(
                        f"Orchestrator stall: last cycle {ago:.0f}s ago "
                        f"(threshold: {self.stall_threshold}s)"
                    )
            if not orch.is_running:
                alerts.append("Orchestrator loop is not running")

        # Memory
        mem = getattr(self.bot, "memory_store", None)
        if mem is not None:
            snap.memory_connected = mem.is_connected
            if not mem.is_connected:
                alerts.append("Memory store (Qdrant) is disconnected")

        # C2
        c2 = getattr(self.bot, "c2", None)
        if c2 is not None:
            snap.c2_running = c2.is_running

        # Models
        snap.swarm_model_count = len(getattr(self.bot, "swarm_models", {}))
        if snap.swarm_model_count == 0:
            alerts.append("No swarm models configured")

        # Dispatcher
        disp = getattr(self.bot, "dispatcher", None)
        if disp is not None:
            snap.dispatch_success = disp.success_count
            snap.dispatch_failures = disp.failure_count
            total = snap.dispatch_success + snap.dispatch_failures
            if total > 0:
                snap.dispatch_success_rate = snap.dispatch_success / total
                if snap.dispatch_success_rate < 0.5 and total >= 5:
                    alerts.append(
                        f"High task failure rate: "
                        f"{snap.dispatch_success_rate:.0%} success "
                        f"({snap.dispatch_failures} failures)"
                    )

        # Goals
        goal_store = getattr(self.bot, "goal_store", None)
        if goal_store is not None:
            try:
                active = await goal_store.get_active_goals()
                snap.active_goals = len(active)
            except Exception:
                pass

        # Trigger manager
        tm = getattr(self.bot, "trigger_manager", None)
        if tm is not None:
            snap.trigger_manager_running = tm.is_running
            snap.trigger_fire_counts = tm.fire_counts

        # Cost
        openrouter = getattr(self.bot, "openrouter", None)
        if openrouter is not None:
            snap.session_cost_usd = getattr(openrouter, "session_cost", 0.0)
        settings = getattr(self.bot, "settings", None)
        if settings is not None:
            snap.cost_limit_usd = settings.SESSION_COST_LIMIT
            if snap.session_cost_usd > snap.cost_limit_usd * 0.9:
                alerts.append(
                    f"Cost approaching limit: ${snap.session_cost_usd:.2f} "
                    f"/ ${snap.cost_limit_usd:.2f}"
                )

        snap.alerts = alerts
        self._last_snapshot = snap
        return snap

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        await asyncio.sleep(60.0)  # Let bot finish bootstrapping

        while self._running:
            try:
                snap = await self.check_now()
                if snap.alerts:
                    log.warning(
                        "Health alerts: %s", "; ".join(snap.alerts),
                    )
                    await self._post_alerts(snap)
                else:
                    log.debug("Health check OK (cycle #%d).", snap.orchestrator_cycles)
            except Exception:
                log.warning("Health check failed.", exc_info=True)

            if self._running:
                await asyncio.sleep(self.check_interval)

    async def _post_alerts(self, snap: HealthSnapshot) -> None:
        """Post health alerts to the #logs channel."""
        router = getattr(self.bot, "router", None)
        if router is None or router.logs is None:
            return

        try:
            embed = discord.Embed(
                title="Health Alert",
                color=0xE74C3C,
                description="\n".join(f"- {a}" for a in snap.alerts),
            )
            embed.add_field(
                name="Orchestrator",
                value=(
                    f"Running: {snap.orchestrator_running}\n"
                    f"Cycles: {snap.orchestrator_cycles}"
                ),
                inline=True,
            )
            embed.add_field(
                name="Tasks",
                value=(
                    f"Success: {snap.dispatch_success}\n"
                    f"Failures: {snap.dispatch_failures}"
                ),
                inline=True,
            )
            embed.add_field(
                name="Goals",
                value=str(snap.active_goals),
                inline=True,
            )
            await router.logs.send(embed=embed)
        except Exception:
            log.warning("Failed to post health alerts.", exc_info=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def last_snapshot(self) -> HealthSnapshot | None:
        return self._last_snapshot

    @property
    def is_running(self) -> bool:
        return self._running

    def __repr__(self) -> str:
        status = "running" if self._running else "stopped"
        healthy = (
            self._last_snapshot.is_healthy if self._last_snapshot else "unknown"
        )
        return f"HealthMonitor(status={status!r}, healthy={healthy})"
