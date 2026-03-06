"""Contract tests for transport and execution adapter boundaries."""

from __future__ import annotations

import pytest

from nexus_api.adapters import WebInteractionAdapter
from nexus_core.adapters import NullInteractionAdapter
from nexus_core.models import ArtifactRecord, CitationRecord, StepExecutionResult
from nexus_core.policy import is_high_risk_action, risk_tier_for_action
from nexus_discord_bridge.adapter import DiscordBridgeInteractionAdapter


@pytest.mark.asyncio
async def test_interaction_adapters_share_contract_methods() -> None:
    adapters = [
        NullInteractionAdapter(),
        WebInteractionAdapter(),
        DiscordBridgeInteractionAdapter(),
    ]
    for adapter in adapters:
        await adapter.emit_message("status", "hello")
        await adapter.request_approval("run1", "step1", "summary", "export")
        await adapter.deliver_status("run1", "running", "detail")


def test_discord_bridge_adapter_captures_events() -> None:
    adapter = DiscordBridgeInteractionAdapter()
    # Smoke check public buffers used by bridge tests/diagnostics.
    assert adapter.messages == []
    assert adapter.approvals == []
    assert adapter.statuses == []


def test_risk_policy_tiers() -> None:
    assert is_high_risk_action("submit")
    assert risk_tier_for_action("submit").value == "high"
    assert not is_high_risk_action("navigate")
    assert risk_tier_for_action("navigate").value == "low"


def test_step_execution_result_shape() -> None:
    result = StepExecutionResult(
        output_text="ok",
        citations=[CitationRecord(url="https://example.com", title="x", snippet="y")],
        artifacts=[
            ArtifactRecord(
                kind="text",
                name="a.txt",
                rel_path="run/a.txt",
                sandbox_path="D:/tmp/a.txt",
                sha256="abc",
            )
        ],
    )
    assert result.output_text == "ok"
    assert result.citations[0].url == "https://example.com"
    assert result.artifacts[0].name == "a.txt"
