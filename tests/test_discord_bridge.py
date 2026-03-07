"""Tests for Discord bridge command behavior."""

from __future__ import annotations

import httpx
import pytest

import nexus_discord_bridge.service as bridge_service
from nexus_discord_bridge.service import BridgeApiClient, BridgeCommands, NexusDiscordBridge


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


class _MockResponse:
    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, object]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://test.local")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError(
                f"status {self.status_code}",
                request=request,
                response=response,
            )


@pytest.mark.asyncio
async def test_bridge_api_client_uses_sessions_and_refreshes_on_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str]] = []
    session_tokens = ["token-1", "token-2"]

    class MockAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            _ = timeout

        async def __aenter__(self) -> "MockAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = exc_type, exc, tb

        async def request(
            self,
            method: str,
            url: str,
            *,
            json: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
        ) -> _MockResponse:
            auth = (headers or {}).get("Authorization", "")
            calls.append((method, url, auth))
            if url.endswith("/sessions"):
                return _MockResponse(200, {"token": session_tokens.pop(0)})
            if url.endswith("/approvals/pending"):
                if auth == "Bearer token-1":
                    return _MockResponse(401, {"detail": "expired"})
                if auth == "Bearer token-2":
                    return _MockResponse(
                        200,
                        {
                            "items": [
                                {
                                    "run_id": "run-1",
                                    "step_id": "step-1",
                                    "action_type": "export",
                                }
                            ]
                        },
                    )
            return _MockResponse(404, {"detail": "not found"})

    monkeypatch.setattr(bridge_service.httpx, "AsyncClient", MockAsyncClient)
    api = BridgeApiClient(
        base_url="http://api.local",
        username="admin",
        password="secret",
    )
    items = await api.pending_approvals()
    assert len(items) == 1
    session_call_count = sum(1 for _, url, _ in calls if url.endswith("/sessions"))
    assert session_call_count == 2
    approval_auth_headers = [auth for _, url, auth in calls if url.endswith("/approvals/pending")]
    assert approval_auth_headers == ["Bearer token-1", "Bearer token-2"]


def test_run_bridge_accepts_admin_credentials_without_api_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "discord-token")
    monkeypatch.setenv("APP_API_URL", "http://api.local")
    monkeypatch.setenv("APP_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("APP_ADMIN_PASSWORD", "secret")
    monkeypatch.delenv("APP_API_TOKEN", raising=False)

    captured: dict[str, object] = {}

    class FakeBridge:
        def __init__(self, api: BridgeApiClient, channel_name: str) -> None:
            captured["api"] = api
            captured["channel_name"] = channel_name

        def run(self, token: str, log_handler=None) -> None:
            captured["discord_token"] = token
            captured["log_handler"] = log_handler

    monkeypatch.setattr(bridge_service, "NexusDiscordBridge", FakeBridge)
    bridge_service.run_bridge()

    assert captured["discord_token"] == "discord-token"
    api = captured["api"]
    assert isinstance(api, BridgeApiClient)
    assert api.token == ""
    assert api.username == "admin"


def test_run_bridge_requires_api_auth_when_token_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "discord-token")
    monkeypatch.setenv("APP_API_URL", "http://api.local")
    monkeypatch.delenv("APP_API_TOKEN", raising=False)
    monkeypatch.delenv("APP_ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("APP_ADMIN_PASSWORD", raising=False)

    with pytest.raises(RuntimeError, match="Set APP_API_TOKEN"):
        bridge_service.run_bridge()
