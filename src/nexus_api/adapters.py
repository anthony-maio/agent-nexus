"""API-side adapter implementations for nexus_core protocols."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from nexus_api.external_tools import (
    ExternalToolInvoker,
    ExternalToolRegistry,
    parse_external_tool_instruction,
)
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


class ExternalToolDispatchExecutionAdapter:
    """Execution adapter that routes external_tool calls outside the sandbox plane."""

    def __init__(
        self,
        *,
        base_adapter: Any,
        tool_registry: ExternalToolRegistry,
        tool_invoker: ExternalToolInvoker | None = None,
    ) -> None:
        self.base_adapter = base_adapter
        self.tool_registry = tool_registry
        self.tool_invoker = tool_invoker

    async def execute_step(
        self, run_id: str, step_id: str, action_type: str, instruction: str
    ) -> StepExecutionResult:
        if action_type.strip().lower() != "external_tool":
            return await self.base_adapter.execute_step(run_id, step_id, action_type, instruction)

        if self.tool_invoker is None:
            raise RuntimeError("External tool invocation is not configured")

        tool_name, arguments = parse_external_tool_instruction(instruction)
        tool_spec = self.tool_registry.get_tool(tool_name)
        if tool_spec is None:
            raise ValueError(f"External tool `{tool_name}` is not registered")

        result = await self.tool_invoker.invoke_tool(
            run_id=run_id,
            step_id=step_id,
            tool_name=tool_name,
            arguments=arguments,
            instruction=instruction,
            tool_spec=tool_spec,
        )
        metadata = dict(result.metadata)
        metadata["external_tool"] = tool_spec.to_dict()
        return StepExecutionResult(
            output_text=result.output_text,
            citations=result.citations,
            artifacts=result.artifacts,
            metadata=metadata,
        )
