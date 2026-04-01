from __future__ import annotations

import json
import sys
from typing import Any

import pytest

from nexus_api.adapters import ExternalToolDispatchExecutionAdapter, ToolAugmentedInteractionAdapter
from nexus_api.external_tools import (
    ExternalToolRegistry,
    ExternalToolSpec,
    StdioExternalToolInvoker,
    parse_external_tool_config,
)
from nexus_core.models import CitationRecord, StepExecutionResult


class _FakeBaseExecutionAdapter:
    async def execute_step(
        self, run_id: str, step_id: str, action_type: str, instruction: str
    ) -> StepExecutionResult:
        return StepExecutionResult(
            output_text=f"base:{action_type}",
            metadata={"run_id": run_id, "step_id": step_id, "instruction": instruction},
        )


class _FakeExternalToolInvoker:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "run_id": run_id,
                "step_id": step_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "instruction": instruction,
                "tool_spec": tool_spec,
            }
        )
        return StepExecutionResult(
            output_text="external-tool:ok",
            citations=[
                CitationRecord(
                    url="mcp://mnemos/memory/1",
                    title="Scoped memory hit",
                    snippet="Recovered prior payment retry context.",
                )
            ],
            metadata={"payload_echo": arguments},
        )


def _write_fake_stdio_mcp_server(root: Any) -> str:
    script_path = root / "fake_mcp_server.py"
    script_path.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "",
                "for line in sys.stdin:",
                "    raw = line.strip()",
                "    if not raw:",
                "        continue",
                "    req = json.loads(raw)",
                "    req_id = req.get('id')",
                "    method = req.get('method')",
                "    params = req.get('params') or {}",
                "    if method == 'initialize':",
                "        resp = {",
                "            'jsonrpc': '2.0',",
                "            'id': req_id,",
                "            'result': {",
                "                'protocolVersion': '2024-11-05',",
                "                'serverInfo': {'name': 'fake-mcp', 'version': '1.0'},",
                "                'capabilities': {'tools': {}},",
                "            },",
                "        }",
                "    elif method == 'tools/call':",
                "        resp = {",
                "            'jsonrpc': '2.0',",
                "            'id': req_id,",
                "            'result': {",
                "                'content': [",
                "                    {",
                "                        'type': 'text',",
                "                        'text': json.dumps({'neo4j': 'offline', 'echo': params.get('arguments') or {}}),",
                "                    }",
                "                ]",
                "            },",
                "        }",
                "    else:",
                "        resp = {'jsonrpc': '2.0', 'id': req_id, 'error': {'code': -32601, 'message': method}}",
                "    sys.stdout.write(json.dumps(resp) + '\\n')",
                "    sys.stdout.flush()",
            ]
        ),
        encoding="utf-8",
    )
    return str(script_path)


@pytest.mark.asyncio
async def test_external_tool_dispatch_adapter_invokes_registered_tools() -> None:
    registry = ExternalToolRegistry(
        [
            ExternalToolSpec(
                name="mnemos.retrieve",
                description="Retrieve scoped memory.",
                source="mcp://mnemos",
                tags=("memory", "retrieval"),
            )
        ]
    )
    invoker = _FakeExternalToolInvoker()
    adapter = ExternalToolDispatchExecutionAdapter(
        base_adapter=_FakeBaseExecutionAdapter(),
        tool_registry=registry,
        tool_invoker=invoker,
    )

    result = await adapter.execute_step(
        run_id="run-123",
        step_id="step-456",
        action_type="external_tool",
        instruction=json.dumps(
            {
                "tool_name": "mnemos.retrieve",
                "arguments": {"query": "payment retries"},
            }
        ),
    )

    assert result.output_text == "external-tool:ok"
    assert result.metadata["external_tool"]["name"] == "mnemos.retrieve"
    assert result.metadata["external_tool"]["source"] == "mcp://mnemos"
    assert result.metadata["payload_echo"]["query"] == "payment retries"
    assert invoker.calls[0]["tool_name"] == "mnemos.retrieve"
    assert invoker.calls[0]["tool_spec"].source == "mcp://mnemos"


@pytest.mark.asyncio
async def test_external_tool_dispatch_adapter_delegates_non_tool_actions() -> None:
    adapter = ExternalToolDispatchExecutionAdapter(
        base_adapter=_FakeBaseExecutionAdapter(),
        tool_registry=ExternalToolRegistry([]),
        tool_invoker=_FakeExternalToolInvoker(),
    )

    result = await adapter.execute_step(
        run_id="run-123",
        step_id="step-789",
        action_type="search_web",
        instruction="collect sources",
    )

    assert result.output_text == "base:search_web"
    assert result.metadata["step_id"] == "step-789"


@pytest.mark.asyncio
async def test_tool_augmented_interaction_adapter_notifies_halobot_for_approval() -> None:
    registry = ExternalToolRegistry(
        [
            ExternalToolSpec(
                name="halobot.notify",
                description="Send approval notifications to HaloBot.",
                source="mcp://halobot",
                tags=("notification", "approval"),
            )
        ]
    )
    invoker = _FakeExternalToolInvoker()
    class _BaseInteraction:
        def __init__(self) -> None:
            self.approvals: list[dict[str, str]] = []

        async def emit_message(self, channel: str, content: str) -> None:
            _ = channel, content

        async def request_approval(
            self, run_id: str, step_id: str, summary: str, action_type: str
        ) -> None:
            self.approvals.append(
                {
                    "run_id": run_id,
                    "step_id": step_id,
                    "summary": summary,
                    "action_type": action_type,
                }
            )

        async def deliver_status(self, run_id: str, status: str, detail: str) -> None:
            _ = run_id, status, detail

    base = _BaseInteraction()
    adapter = ToolAugmentedInteractionAdapter(
        base_adapter=base,
        tool_registry=registry,
        tool_invoker=invoker,
    )

    await adapter.request_approval(
        run_id="run-123",
        step_id="step-456",
        summary="submit payment workflow",
        action_type="submit",
    )

    assert base.approvals[0]["step_id"] == "step-456"
    assert invoker.calls[0]["tool_name"] == "halobot.notify"
    assert invoker.calls[0]["arguments"]["event"] == "approval_needed"
    assert invoker.calls[0]["arguments"]["action_type"] == "submit"


@pytest.mark.asyncio
async def test_tool_augmented_interaction_adapter_notifies_halobot_for_failed_status() -> None:
    registry = ExternalToolRegistry(
        [
            ExternalToolSpec(
                name="halobot.notify",
                description="Send runtime notifications to HaloBot.",
                source="mcp://halobot",
                tags=("notification",),
            )
        ]
    )
    invoker = _FakeExternalToolInvoker()

    class _BaseInteraction:
        def __init__(self) -> None:
            self.statuses: list[dict[str, str]] = []

        async def emit_message(self, channel: str, content: str) -> None:
            _ = channel, content

        async def request_approval(self, run_id: str, step_id: str, summary: str, action_type: str) -> None:
            _ = run_id, step_id, summary, action_type

        async def deliver_status(self, run_id: str, status: str, detail: str) -> None:
            self.statuses.append({"run_id": run_id, "status": status, "detail": detail})

    base = _BaseInteraction()
    adapter = ToolAugmentedInteractionAdapter(
        base_adapter=base,
        tool_registry=registry,
        tool_invoker=invoker,
    )

    await adapter.deliver_status("run-123", "failed", "verification blocked")

    assert base.statuses[0]["status"] == "failed"
    assert invoker.calls[0]["tool_name"] == "halobot.notify"
    assert invoker.calls[0]["arguments"]["event"] == "run_status"
    assert invoker.calls[0]["arguments"]["status"] == "failed"


def test_parse_external_tool_config_preserves_stdio_transport_details(tmp_path: Any) -> None:
    script_path = _write_fake_stdio_mcp_server(tmp_path)
    registry = parse_external_tool_config(
        json.dumps(
            [
                {
                    "name": "c2.status",
                    "description": "Return continuity-core status.",
                    "source": "mcp://continuity-core",
                    "transport": {
                        "kind": "stdio",
                        "command": [sys.executable, script_path],
                    },
                }
            ]
        )
    )

    tool = registry.get_tool("c2.status")

    assert tool is not None
    assert tool.transport["kind"] == "stdio"
    assert tool.transport["command"][1] == script_path


@pytest.mark.asyncio
async def test_stdio_external_tool_invoker_calls_local_mcp_server(tmp_path: Any) -> None:
    script_path = _write_fake_stdio_mcp_server(tmp_path)
    invoker = StdioExternalToolInvoker(timeout_sec=10.0)
    tool = ExternalToolSpec(
        name="c2.status",
        description="Return continuity-core status.",
        source="mcp://continuity-core",
        transport={
            "kind": "stdio",
            "command": [sys.executable, script_path],
        },
    )

    result = await invoker.invoke_tool(
        run_id="run-123",
        step_id="step-456",
        tool_name="c2.status",
        arguments={},
        instruction=json.dumps({"tool_name": "c2.status", "arguments": {}}),
        tool_spec=tool,
    )

    assert result.output_text
    assert result.metadata["tool_result"]["neo4j"] in {"offline", "connected", "error"}
    assert result.citations[0].url == "mcp://continuity-core"
