"""Discord interaction adapter implementation."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class DiscordBridgeInteractionAdapter:
    """InteractionAdapter-compatible class for Discord bridge notifications."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []
        self.approvals: list[dict[str, str]] = []
        self.statuses: list[dict[str, str]] = []

    async def emit_message(self, channel: str, content: str) -> None:
        self.messages.append((channel, content))
        log.info("discord-adapter message channel=%s content=%s", channel, content)

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
        log.info("discord-adapter approval requested run=%s step=%s", run_id, step_id)

    async def deliver_status(self, run_id: str, status: str, detail: str) -> None:
        self.statuses.append({"run_id": run_id, "status": status, "detail": detail})
        log.info("discord-adapter status run=%s status=%s", run_id, status)
