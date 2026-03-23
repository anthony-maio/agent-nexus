"""Generic external-tool registry and invocation helpers."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

from nexus_core.models import CitationRecord, StepExecutionResult


@dataclass(frozen=True, slots=True)
class ExternalToolSpec:
    """One external tool exposed to the runtime."""

    name: str
    description: str
    source: str = ""
    tags: tuple[str, ...] = ()
    risk_tier: str = "high"
    transport: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "risk_tier": self.risk_tier,
        }
        if self.source:
            payload["source"] = self.source
        if self.tags:
            payload["tags"] = list(self.tags)
        return payload


class ExternalToolRegistry:
    """Simple in-memory registry for external tool specs."""

    def __init__(self, tools: Iterable[ExternalToolSpec]) -> None:
        deduped: dict[str, ExternalToolSpec] = {}
        for tool in tools:
            normalized = tool.name.strip().lower()
            if not normalized:
                continue
            deduped[normalized] = ExternalToolSpec(
                name=tool.name.strip(),
                description=tool.description.strip(),
                source=tool.source.strip(),
                tags=_normalize_tags(tool.tags),
                risk_tier=(tool.risk_tier.strip().lower() or "high"),
                transport=_normalize_transport(tool.transport),
            )
        self._tools = deduped

    def list_tools(self) -> list[ExternalToolSpec]:
        return [self._tools[key] for key in sorted(self._tools)]

    def get_tool(self, name: str) -> ExternalToolSpec | None:
        return self._tools.get(name.strip().lower())


class ExternalToolInvoker(Protocol):
    async def invoke_tool(
        self,
        *,
        run_id: str,
        step_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        instruction: str,
        tool_spec: ExternalToolSpec | None,
    ) -> StepExecutionResult: ...


class StdioExternalToolInvoker:
    """Invoke a local MCP-compatible server over a one-shot stdio session."""

    def __init__(self, timeout_sec: float = 15.0) -> None:
        self.timeout_sec = max(float(timeout_sec), 1.0)

    async def invoke_tool(
        self,
        *,
        run_id: str,
        step_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        instruction: str,
        tool_spec: ExternalToolSpec | None,
    ) -> StepExecutionResult:
        _ = run_id, step_id, instruction
        if tool_spec is None:
            raise RuntimeError(f"External tool `{tool_name}` is not registered")
        transport = _normalize_transport(tool_spec.transport)
        if transport.get("kind") != "stdio":
            raise RuntimeError(f"External tool `{tool_name}` does not declare stdio transport")
        command = transport.get("command")
        if not isinstance(command, list) or not command:
            raise RuntimeError(f"External tool `{tool_name}` is missing a stdio command")

        result_payload = await asyncio.wait_for(
            self._call_stdio_tool(command, tool_name=tool_name, arguments=arguments),
            timeout=self.timeout_sec,
        )
        text = result_payload if isinstance(result_payload, str) else json.dumps(result_payload)
        source = tool_spec.source.strip() or f"tool://{tool_name}"
        metadata = {
            "tool_result": result_payload if isinstance(result_payload, dict) else {"raw": text},
            "tool_transport": "stdio",
        }
        return StepExecutionResult(
            output_text=text,
            citations=[
                CitationRecord(
                    url=source,
                    title=tool_name,
                    snippet=text[:240],
                )
            ],
            metadata=metadata,
        )

    async def _call_stdio_tool(
        self,
        command: list[str],
        *,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any] | str:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        try:
            for request in (
                {
                    "jsonrpc": "2.0",
                    "id": "init",
                    "method": "initialize",
                    "params": {},
                },
                {
                    "jsonrpc": "2.0",
                    "id": "call",
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments,
                    },
                },
            ):
                process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()

            response_payload: dict[str, Any] | None = None
            while True:
                raw_line = await process.stdout.readline()
                if not raw_line:
                    break
                decoded = raw_line.decode("utf-8").strip()
                if not decoded:
                    continue
                payload = json.loads(decoded)
                if payload.get("id") != "call":
                    continue
                response_payload = payload
                break

            if response_payload is None:
                stderr = await process.stderr.read() if process.stderr is not None else b""
                raise RuntimeError(
                    "External tool process returned no call response"
                    + (f": {stderr.decode('utf-8', errors='ignore').strip()}" if stderr else "")
                )
            if "error" in response_payload:
                message = str((response_payload.get("error") or {}).get("message", "")).strip()
                raise RuntimeError(message or f"External tool `{tool_name}` failed")

            result = response_payload.get("result") or {}
            content = result.get("content") if isinstance(result, dict) else None
            text_blocks = [
                str(item.get("text", "")).strip()
                for item in (content or [])
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            joined = "\n".join(block for block in text_blocks if block).strip()
            if not joined:
                return {}
            try:
                parsed = json.loads(joined)
            except json.JSONDecodeError:
                return joined
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        finally:
            if process.returncode is None:
                process.kill()
            await process.wait()


def parse_external_tool_config(raw: str) -> ExternalToolRegistry:
    """Parse JSON config into a registry of external tools."""

    if not raw.strip():
        return ExternalToolRegistry([])
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return ExternalToolRegistry([])

    if isinstance(parsed, dict):
        items = parsed.get("tools")
        if not isinstance(items, list):
            items = [parsed]
    elif isinstance(parsed, list):
        items = parsed
    else:
        items = []

    tools: list[ExternalToolSpec] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        description = str(item.get("description", "")).strip()
        if not name or not description:
            continue
        tools.append(
            ExternalToolSpec(
                name=name,
                description=description,
                source=str(item.get("source", "")).strip(),
                tags=_normalize_tags(item.get("tags")),
                risk_tier=str(item.get("risk_tier", "high")).strip().lower() or "high",
                transport=_normalize_transport(
                    item.get(
                        "transport",
                        {"kind": "stdio", "command": item["command"]} if "command" in item else {},
                    )
                ),
            )
        )
    return ExternalToolRegistry(tools)


def serialize_external_tool_context(
    tools: Iterable[ExternalToolSpec],
) -> list[dict[str, Any]]:
    return [tool.to_dict() for tool in tools]


def parse_external_tool_instruction(instruction: str) -> tuple[str, dict[str, Any]]:
    """Normalize an external_tool instruction payload into tool name plus arguments."""

    try:
        payload = json.loads(instruction)
    except json.JSONDecodeError as exc:
        raise ValueError("external_tool instructions must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("external_tool instructions must be a JSON object")

    tool_name = str(
        payload.get("tool_name", payload.get("name", payload.get("tool", "")))
    ).strip()
    if not tool_name:
        raise ValueError("external_tool instructions must include tool_name")

    raw_arguments = payload.get(
        "arguments",
        payload.get("payload", payload.get("params", payload.get("input", {}))),
    )
    if raw_arguments is None:
        arguments: dict[str, Any] = {}
    elif isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        arguments = {"value": raw_arguments}
    return tool_name, arguments


def _normalize_tags(raw_tags: Any) -> tuple[str, ...]:
    if isinstance(raw_tags, str):
        items = raw_tags.replace("|", ",").split(",")
    elif isinstance(raw_tags, (list, tuple, set)):
        items = list(raw_tags)
    else:
        items = []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        tag = str(item or "").strip().lower()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
    return tuple(normalized)


def _normalize_transport(raw_transport: Any) -> dict[str, Any]:
    if not isinstance(raw_transport, dict):
        return {}
    normalized: dict[str, Any] = {}
    kind = str(raw_transport.get("kind", "")).strip().lower()
    if kind:
        normalized["kind"] = kind
    raw_command = raw_transport.get("command")
    if isinstance(raw_command, (list, tuple)):
        command = [str(part).strip() for part in raw_command if str(part).strip()]
        if command:
            normalized["command"] = command
    return normalized
