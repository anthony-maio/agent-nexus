"""Adapter protocols for transport and execution boundaries."""

from __future__ import annotations

from typing import Protocol

from nexus_core.models import StepExecutionResult


class InteractionAdapter(Protocol):
    """Interface for external interaction channels (web, Discord, etc.)."""

    async def emit_message(self, channel: str, content: str) -> None:
        """Emit a user-visible message to a channel."""

    async def request_approval(
        self, run_id: str, step_id: str, summary: str, action_type: str
    ) -> None:
        """Notify that a step needs approval."""

    async def deliver_status(self, run_id: str, status: str, detail: str) -> None:
        """Emit a status update for a run."""


class ExecutionAdapter(Protocol):
    """Interface for sandbox execution backends."""

    async def execute_step(
        self, run_id: str, step_id: str, action_type: str, instruction: str
    ) -> StepExecutionResult:
        """Execute a step and return output/citations/artifacts."""


class NullInteractionAdapter:
    """No-op interaction adapter for local tests and headless runs."""

    async def emit_message(self, channel: str, content: str) -> None:
        return None

    async def request_approval(
        self, run_id: str, step_id: str, summary: str, action_type: str
    ) -> None:
        return None

    async def deliver_status(self, run_id: str, status: str, detail: str) -> None:
        return None
