"""Discord bridge service for remote approvals/status summaries."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import discord
import httpx
from discord.ext import commands

log = logging.getLogger(__name__)


class BridgeApiClient:
    """Thin async client for Nexus API endpoints used by bridge."""

    def __init__(
        self,
        base_url: str,
        token: str = "",
        username: str = "",
        password: str = "",
        timeout_sec: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token.strip()
        self.username = username.strip()
        self.password = password
        self.timeout_sec = timeout_sec
        self._session_token = ""

    def _can_create_session(self) -> bool:
        return bool(self.username and self.password)

    async def _create_session_token(
        self,
        client: httpx.AsyncClient | None = None,
    ) -> str:
        if not self._can_create_session():
            raise RuntimeError(
                "Bridge API auth requires APP_API_TOKEN, or "
                "APP_ADMIN_USERNAME and APP_ADMIN_PASSWORD."
            )
        body = {
            "username": self.username,
            "password": self.password,
        }
        if client is not None:
            resp = await client.request(
                method="POST",
                url=f"{self.base_url}/sessions",
                json=body,
            )
        else:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as standalone_client:
                resp = await standalone_client.request(
                    method="POST",
                    url=f"{self.base_url}/sessions",
                    json=body,
                )
        resp.raise_for_status()
        data = resp.json()
        session_token = str(data.get("token", "")).strip()
        if not session_token:
            raise RuntimeError("Nexus API /sessions response did not include a token.")
        return session_token

    async def _resolve_bearer_token(
        self,
        *,
        force_refresh: bool = False,
        client: httpx.AsyncClient | None = None,
    ) -> str:
        if self.token:
            return self.token
        if force_refresh:
            self._session_token = ""
        if not self._session_token:
            self._session_token = await self._create_session_token(client=client)
        return self._session_token

    async def _request(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            bearer = await self._resolve_bearer_token(client=client)
            headers = {"Authorization": f"Bearer {bearer}"}
            resp = await client.request(
                method=method,
                url=f"{self.base_url}{path}",
                json=json_body,
                headers=headers,
            )
            if resp.status_code == 401 and not self.token and self._can_create_session():
                bearer = await self._resolve_bearer_token(
                    force_refresh=True,
                    client=client,
                )
                headers = {"Authorization": f"Bearer {bearer}"}
                resp = await client.request(
                    method=method,
                    url=f"{self.base_url}{path}",
                    json=json_body,
                    headers=headers,
                )
            resp.raise_for_status()
            return resp.json()

    async def pending_approvals(self) -> list[dict[str, Any]]:
        data = await self._request("GET", "/approvals/pending")
        return data.get("items", [])

    async def run_status(self, run_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/runs/{run_id}")

    async def decide(
        self,
        run_id: str,
        step_id: str,
        decision: str,
        reason: str = "",
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/runs/{run_id}/approvals/{step_id}",
            {"decision": decision, "reason": reason},
        )


class NexusDiscordBridge(commands.Bot):
    """Minimal Discord bridge bot for app-first runtime."""

    def __init__(self, api: BridgeApiClient, channel_name: str) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)
        self.api = api
        self.channel_name = channel_name
        self._last_seen_step_ids: set[str] = set()
        self._poll_task: asyncio.Task[None] | None = None

    async def setup_hook(self) -> None:
        await self.add_cog(BridgeCommands(self))

    async def on_ready(self) -> None:
        log.info("Discord bridge online as %s", self.user)
        if self._poll_task is None:
            self._poll_task = asyncio.create_task(
                self._poll_approvals(),
                name="bridge-poll-approvals",
            )

    async def _poll_approvals(self) -> None:
        await self.wait_until_ready()
        while not self.is_closed():
            try:
                items = await self.api.pending_approvals()
                new_items = [i for i in items if i.get("step_id") not in self._last_seen_step_ids]
                if new_items:
                    channel = self._resolve_channel()
                    if channel is not None:
                        for item in new_items:
                            self._last_seen_step_ids.add(item.get("step_id", ""))
                            await channel.send(
                                (
                                    f"Approval needed for run `{item.get('run_id')}` "
                                    f"step `{item.get('step_id')}` (`{item.get('action_type')}`)\n"
                                    f"Instruction: {item.get('instruction', '')[:180]}"
                                )
                            )
            except Exception:
                log.warning("Approval poll failed.", exc_info=True)
            await asyncio.sleep(15)

    def _resolve_channel(self) -> discord.TextChannel | None:
        for guild in self.guilds:
            for channel in guild.text_channels:
                if channel.name == self.channel_name:
                    return channel
        return None


class BridgeCommands(commands.Cog):
    """Discord bridge commands."""

    def __init__(self, bot: NexusDiscordBridge) -> None:
        self.bot = bot

    @commands.command(name="pending")
    async def pending(self, ctx: commands.Context) -> None:
        items = await self.bot.api.pending_approvals()
        if not items:
            await ctx.send("No pending approvals.")
            return
        lines = [
            f"- run `{i['run_id']}` step `{i['step_id']}` `{i['action_type']}`"
            for i in items[:10]
        ]
        await ctx.send("Pending approvals:\n" + "\n".join(lines))

    @commands.command(name="runstatus")
    async def runstatus(self, ctx: commands.Context, run_id: str) -> None:
        run = await self.bot.api.run_status(run_id)
        await ctx.send(
            f"Run `{run['id']}` status: **{run['status']}** "
            f"({len(run.get('steps', []))} step(s))"
        )

    @commands.command(name="approve")
    async def approve(self, ctx: commands.Context, run_id: str, step_id: str) -> None:
        run = await self.bot.api.decide(
            run_id,
            step_id,
            decision="approve",
            reason="approved via Discord bridge",
        )
        await ctx.send(f"Approved step `{step_id}`. Run status: **{run.get('status', 'unknown')}**")

    @commands.command(name="reject")
    async def reject(
        self,
        ctx: commands.Context,
        run_id: str,
        step_id: str,
        *,
        reason: str = "",
    ) -> None:
        run = await self.bot.api.decide(
            run_id,
            step_id,
            decision="reject",
            reason=reason or "rejected via Discord bridge",
        )
        await ctx.send(f"Rejected step `{step_id}`. Run status: **{run.get('status', 'unknown')}**")


def run_bridge() -> None:
    """Entrypoint used by `python -m nexus_discord_bridge`."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    token = os.environ.get("DISCORD_TOKEN", "").strip()
    if not token:
        raise RuntimeError("DISCORD_TOKEN is required for nexus-discord-bridge")
    api_base = os.environ.get("APP_API_URL", "http://localhost:8000")
    api_token = os.environ.get("APP_API_TOKEN", "").strip()
    api_username = os.environ.get("APP_ADMIN_USERNAME", "").strip()
    api_password = os.environ.get("APP_ADMIN_PASSWORD", "")
    if not api_token and not (api_username and api_password):
        raise RuntimeError(
            "Set APP_API_TOKEN, or set APP_ADMIN_USERNAME and APP_ADMIN_PASSWORD for "
            "nexus-discord-bridge."
        )
    if not api_token:
        log.info("Discord bridge auth mode: /sessions (admin credentials).")
    channel_name = os.environ.get("DISCORD_BRIDGE_CHANNEL", "human")
    bridge = NexusDiscordBridge(
        api=BridgeApiClient(
            base_url=api_base,
            token=api_token,
            username=api_username,
            password=api_password,
        ),
        channel_name=channel_name,
    )
    bridge.run(token, log_handler=None)
