"""API-side adapter implementations for nexus_core protocols."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from nexus_core.models import ArtifactRecord, CitationRecord, StepExecutionResult

log = logging.getLogger(__name__)


class WebInteractionAdapter:
    """Interaction adapter for app/web surfaces.

    For v1 this adapter emits to logs and defers real-time updates
    to the run event bus (published by core engine).
    """

    async def emit_message(self, channel: str, content: str) -> None:
        log.info("web:%s %s", channel, content)

    async def request_approval(
        self, run_id: str, step_id: str, summary: str, action_type: str
    ) -> None:
        log.info(
            "approval-needed run=%s step=%s action=%s summary=%s",
            run_id,
            step_id,
            action_type,
            summary,
        )

    async def deliver_status(self, run_id: str, status: str, detail: str) -> None:
        log.info("run-status run=%s status=%s detail=%s", run_id, status, detail)


class SandboxExecutionAdapter:
    """Execution adapter that delegates steps to sandbox-runner service."""

    def __init__(self, base_url: str, timeout_sec: float = 60.0, auth_token: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.auth_token = auth_token.strip()

    async def execute_step(
        self, run_id: str, step_id: str, action_type: str, instruction: str
    ) -> StepExecutionResult:
        payload: dict[str, Any] = {
            "run_id": run_id,
            "step_id": step_id,
            "action_type": action_type,
            "instruction": instruction,
        }
        headers = {"X-Sandbox-Token": self.auth_token} if self.auth_token else None
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            resp = await client.post(
                f"{self.base_url}/execute-step",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        citations = [
            CitationRecord(
                url=c.get("url", ""),
                title=c.get("title", ""),
                snippet=c.get("snippet", ""),
            )
            for c in data.get("citations", [])
        ]
        artifacts = [
            ArtifactRecord(
                kind=a.get("kind", "text"),
                name=a.get("name", "artifact.txt"),
                rel_path=a.get("rel_path", ""),
                sandbox_path=a.get("sandbox_path", ""),
                sha256=a.get("sha256", ""),
            )
            for a in data.get("artifacts", [])
        ]
        return StepExecutionResult(
            output_text=data.get("output_text", ""),
            citations=citations,
            artifacts=artifacts,
            metadata=data.get("metadata", {}),
        )
