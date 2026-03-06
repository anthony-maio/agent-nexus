"""Tests for Discord bridge command behavior."""

from __future__ import annotations

import pytest

from nexus_discord_bridge.service import BridgeCommands, NexusDiscordBridge


class FakeApi:
    def __init__(self) -> None:
        self.decisions: list[tuple[str, str, str, str]] = []

    async def pending_approvals(self):
        return [
            {
                "run_id": "run1",
                "step_id": "step1",
                "action_type": "export",
                "instruction": "export report",
            }
        ]

    async def run_status(self, run_id: str):
        return {"id": run_id, "status": "pending_approval", "steps": [1, 2, 3]}

    async def decide(self, run_id: str, step_id: str, decision: str, reason: str = ""):
        self.decisions.append((run_id, step_id, decision, reason))
        return {"id": run_id, "status": "completed"}


class FakeContext:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send(self, content: str) -> None:
        self.messages.append(content)


@pytest.mark.asyncio
async def test_bridge_pending_and_approve_commands():
    bot = NexusDiscordBridge(api=FakeApi(), channel_name="human")
    commands = BridgeCommands(bot)
    ctx = FakeContext()

    await commands.pending.callback(commands, ctx)
    assert "Pending approvals" in ctx.messages[-1]

    await commands.approve.callback(commands, ctx, "run1", "step1")
    assert "Approved step" in ctx.messages[-1]
    assert bot.api.decisions[0][2] == "approve"
