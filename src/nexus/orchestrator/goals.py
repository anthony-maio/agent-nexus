"""Persistent goal and task queue system for Agent Nexus.

The :class:`GoalStore` provides a Redis-backed persistence layer for
multi-step goals and their associated task queues.  Goals survive bot
restarts, accumulate progress notes from dispatched tasks, and are
injected into the orchestrator's decision prompt so the swarm can
reason about ongoing work.

Usage::

    store = GoalStore(redis_url="redis://nexus-redis:6379/0")
    await store.connect()

    goal_id = await store.add_goal(Goal(
        title="Analyze user codebase",
        description="Deep-dive into the repository structure.",
        priority="high",
        source="orchestrator",
    ))
    await store.enqueue_task(goal_id, TaskQueueItem(
        description="Map top-level directory layout",
        action_type="research",
    ))
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)

# Redis key prefixes
_GOAL_PREFIX = "nexus:goal:"
_TASK_PREFIX = "nexus:task:"
_GOAL_INDEX = "nexus:goals:active"


class GoalStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    STALE = "stale"
    CANCELLED = "cancelled"


class TaskStatus(Enum):
    PENDING = "pending"
    DISPATCHED = "dispatched"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskQueueItem:
    """A single task within a goal's execution plan."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str = ""
    action_type: str = "research"
    priority: str = "medium"
    status: str = TaskStatus.PENDING.value
    goal_id: str = ""
    depends_on: list[str] = field(default_factory=list)
    result_summary: str = ""
    attempts: int = 0
    max_attempts: int = 3
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str = ""

    def is_ready(self, completed_ids: set[str]) -> bool:
        """Whether all dependencies are satisfied."""
        return all(dep in completed_ids for dep in self.depends_on)


@dataclass
class Goal:
    """A persistent, multi-step objective tracked by the orchestrator."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    description: str = ""
    priority: str = "medium"
    status: str = GoalStatus.ACTIVE.value
    source: str = "orchestrator"
    progress_notes: list[str] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str = ""
    max_age_hours: float = 72.0


class GoalStore:
    """Redis-backed persistence for goals and task queues.

    Falls back to in-memory storage when Redis is unavailable, logging
    a warning on the first failed connection attempt.

    Args:
        redis_url: Redis connection string.
        max_active_goals: Cap on simultaneously active goals.
        default_max_age_hours: Default TTL for goals before they become stale.
    """

    def __init__(
        self,
        redis_url: str = "redis://nexus-redis:6379/0",
        max_active_goals: int = 10,
        default_max_age_hours: float = 72.0,
    ) -> None:
        self._redis_url = redis_url
        self._max_active_goals = max_active_goals
        self._default_max_age_hours = default_max_age_hours
        self._redis: Any = None  # redis.asyncio.Redis instance
        self._connected: bool = False

        # In-memory fallback
        self._mem_goals: dict[str, Goal] = {}
        self._mem_tasks: dict[str, TaskQueueItem] = {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Attempt to connect to Redis.  Returns ``True`` on success."""
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                self._redis_url, decode_responses=True
            )
            await self._redis.ping()
            self._connected = True
            log.info("GoalStore connected to Redis at %s", self._redis_url)
            return True
        except Exception as exc:
            log.warning(
                "GoalStore Redis connection failed (%s) — using in-memory fallback.",
                exc,
            )
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Goal CRUD
    # ------------------------------------------------------------------

    async def add_goal(self, goal: Goal) -> str:
        """Persist a new goal.  Returns the goal ID."""
        active = await self.get_active_goals()
        if len(active) >= self._max_active_goals:
            log.warning(
                "Active goal limit (%d) reached — pruning stale goals first.",
                self._max_active_goals,
            )
            await self.prune_stale_goals()
            active = await self.get_active_goals()
            if len(active) >= self._max_active_goals:
                log.warning("Still at goal limit after pruning.  Dropping oldest.")
                oldest = min(active, key=lambda g: g.created_at)
                await self.update_goal(oldest.id, status=GoalStatus.CANCELLED.value)

        if not goal.max_age_hours:
            goal.max_age_hours = self._default_max_age_hours

        if self._connected:
            await self._redis.set(
                f"{_GOAL_PREFIX}{goal.id}", json.dumps(asdict(goal))
            )
            await self._redis.sadd(_GOAL_INDEX, goal.id)
        else:
            self._mem_goals[goal.id] = goal

        log.info("Goal created: %s — %s", goal.id, goal.title)
        return goal.id

    async def get_goal(self, goal_id: str) -> Goal | None:
        """Retrieve a single goal by ID."""
        if self._connected:
            raw = await self._redis.get(f"{_GOAL_PREFIX}{goal_id}")
            if raw is None:
                return None
            return Goal(**json.loads(raw))
        return self._mem_goals.get(goal_id)

    async def get_active_goals(self) -> list[Goal]:
        """Return all goals with status ``active``."""
        if self._connected:
            goal_ids = await self._redis.smembers(_GOAL_INDEX)
            goals: list[Goal] = []
            for gid in goal_ids:
                g = await self.get_goal(gid)
                if g and g.status == GoalStatus.ACTIVE.value:
                    goals.append(g)
            return sorted(goals, key=lambda g: g.created_at)
        return sorted(
            [g for g in self._mem_goals.values() if g.status == GoalStatus.ACTIVE.value],
            key=lambda g: g.created_at,
        )

    async def update_goal(self, goal_id: str, **fields: Any) -> bool:
        """Update specific fields on a goal.  Returns ``True`` on success."""
        goal = await self.get_goal(goal_id)
        if goal is None:
            return False

        for key, value in fields.items():
            if hasattr(goal, key):
                setattr(goal, key, value)
        goal.updated_at = datetime.now(timezone.utc).isoformat()

        if self._connected:
            await self._redis.set(
                f"{_GOAL_PREFIX}{goal.id}", json.dumps(asdict(goal))
            )
            if goal.status != GoalStatus.ACTIVE.value:
                await self._redis.srem(_GOAL_INDEX, goal.id)
        else:
            self._mem_goals[goal.id] = goal

        return True

    async def add_progress_note(self, goal_id: str, note: str) -> bool:
        """Append a progress note to a goal."""
        goal = await self.get_goal(goal_id)
        if goal is None:
            return False

        timestamp = datetime.now(timezone.utc).strftime("%H:%M")
        goal.progress_notes.append(f"[{timestamp}] {note}")
        # Keep last 20 notes to prevent unbounded growth
        if len(goal.progress_notes) > 20:
            goal.progress_notes = goal.progress_notes[-20:]

        return await self.update_goal(goal_id, progress_notes=goal.progress_notes)

    async def complete_goal(self, goal_id: str) -> bool:
        """Mark a goal as completed."""
        return await self.update_goal(
            goal_id,
            status=GoalStatus.COMPLETED.value,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # Task queue
    # ------------------------------------------------------------------

    async def enqueue_task(self, goal_id: str, task: TaskQueueItem) -> str:
        """Add a task to a goal's queue.  Returns the task ID."""
        task.goal_id = goal_id

        goal = await self.get_goal(goal_id)
        if goal is not None and task.id not in goal.task_ids:
            goal.task_ids.append(task.id)
            await self.update_goal(goal_id, task_ids=goal.task_ids)

        if self._connected:
            await self._redis.set(
                f"{_TASK_PREFIX}{task.id}", json.dumps(asdict(task))
            )
        else:
            self._mem_tasks[task.id] = task

        return task.id

    async def enqueue_chain(
        self, goal_id: str, tasks: list[TaskQueueItem]
    ) -> list[str]:
        """Enqueue a sequence of tasks where each depends on the previous."""
        ids: list[str] = []
        prev_id: str | None = None
        for task in tasks:
            if prev_id is not None:
                task.depends_on = [prev_id]
            tid = await self.enqueue_task(goal_id, task)
            ids.append(tid)
            prev_id = task.id
        return ids

    async def get_task(self, task_id: str) -> TaskQueueItem | None:
        """Retrieve a single task by ID."""
        if self._connected:
            raw = await self._redis.get(f"{_TASK_PREFIX}{task_id}")
            if raw is None:
                return None
            return TaskQueueItem(**json.loads(raw))
        return self._mem_tasks.get(task_id)

    async def get_ready_tasks(self, goal_id: str | None = None) -> list[TaskQueueItem]:
        """Return pending tasks whose dependencies are all completed."""
        goals = [await self.get_goal(goal_id)] if goal_id else await self.get_active_goals()
        goals = [g for g in goals if g is not None]

        all_tasks: list[TaskQueueItem] = []
        completed_ids: set[str] = set()

        for goal in goals:
            for tid in goal.task_ids:
                task = await self.get_task(tid)
                if task is None:
                    continue
                all_tasks.append(task)
                if task.status == TaskStatus.COMPLETED.value:
                    completed_ids.add(task.id)

        return [
            t
            for t in all_tasks
            if t.status == TaskStatus.PENDING.value and t.is_ready(completed_ids)
        ]

    async def mark_task_dispatched(self, task_id: str) -> bool:
        """Mark a task as dispatched (in-flight)."""
        return await self._update_task(task_id, status=TaskStatus.DISPATCHED.value)

    async def mark_task_completed(
        self, task_id: str, result_summary: str = ""
    ) -> bool:
        """Mark a task as completed and update its parent goal."""
        success = await self._update_task(
            task_id,
            status=TaskStatus.COMPLETED.value,
            result_summary=result_summary,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        if success:
            task = await self.get_task(task_id)
            if task and task.goal_id:
                await self.add_progress_note(
                    task.goal_id,
                    f"Task completed: {task.description[:80]}",
                )
        return success

    async def mark_task_failed(self, task_id: str, error: str = "") -> bool:
        """Mark a task as failed.  Increments the attempt counter."""
        task = await self.get_task(task_id)
        if task is None:
            return False

        task.attempts += 1
        if task.attempts < task.max_attempts:
            # Reset to pending for retry
            return await self._update_task(
                task_id,
                status=TaskStatus.PENDING.value,
                attempts=task.attempts,
            )
        else:
            success = await self._update_task(
                task_id,
                status=TaskStatus.FAILED.value,
                result_summary=f"Failed after {task.attempts} attempts: {error}",
            )
            if success and task.goal_id:
                await self.add_progress_note(
                    task.goal_id,
                    f"Task failed: {task.description[:80]} ({error[:50]})",
                )
            return success

    async def _update_task(self, task_id: str, **fields: Any) -> bool:
        """Update specific fields on a task."""
        task = await self.get_task(task_id)
        if task is None:
            return False

        for key, value in fields.items():
            if hasattr(task, key):
                setattr(task, key, value)

        if self._connected:
            await self._redis.set(
                f"{_TASK_PREFIX}{task.id}", json.dumps(asdict(task))
            )
        else:
            self._mem_tasks[task.id] = task
        return True

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def prune_stale_goals(self) -> int:
        """Mark goals as stale if they exceed their max age.  Returns count."""
        now = datetime.now(timezone.utc)
        active = await self.get_active_goals()
        pruned = 0

        for goal in active:
            try:
                created = datetime.fromisoformat(goal.created_at)
                age_hours = (now - created).total_seconds() / 3600
                if age_hours > goal.max_age_hours:
                    await self.update_goal(goal.id, status=GoalStatus.STALE.value)
                    log.info(
                        "Goal %s marked stale (age=%.1fh, max=%.1fh): %s",
                        goal.id, age_hours, goal.max_age_hours, goal.title,
                    )
                    pruned += 1
            except (ValueError, TypeError):
                pass

        return pruned

    # ------------------------------------------------------------------
    # Serialization for decision prompt
    # ------------------------------------------------------------------

    async def summarize_for_prompt(self) -> str:
        """Build a text summary of active goals and their tasks for the decision engine."""
        active = await self.get_active_goals()
        if not active:
            return "(No active goals)"

        parts: list[str] = []
        for goal in active:
            parts.append(
                f"Goal [{goal.id}] ({goal.priority}): {goal.title}"
            )
            if goal.description:
                parts.append(f"  Description: {goal.description[:200]}")

            # Summarize tasks
            pending = 0
            completed = 0
            failed = 0
            for tid in goal.task_ids:
                task = await self.get_task(tid)
                if task is None:
                    continue
                if task.status == TaskStatus.COMPLETED.value:
                    completed += 1
                elif task.status == TaskStatus.FAILED.value:
                    failed += 1
                else:
                    pending += 1

            parts.append(
                f"  Tasks: {completed} done, {pending} pending, {failed} failed"
            )

            # Last 3 progress notes
            if goal.progress_notes:
                for note in goal.progress_notes[-3:]:
                    parts.append(f"  {note}")

            parts.append("")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        backend = "redis" if self._connected else "memory"
        return f"GoalStore(backend={backend!r})"
