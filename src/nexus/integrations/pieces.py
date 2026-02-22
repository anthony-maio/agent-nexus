"""PiecesOS MCP client for activity-stream awareness.

PiecesOS is a retail productivity tool that tracks user activity (screen
content, clipboard, keystrokes) and exposes a Long-Term Memory (LTM) engine
via an MCP server.  This module provides an async client that queries that
server so the Agent Nexus swarm can passively observe what the user is
working on -- without the user needing to send explicit prompts.

Uses the MCP Streamable HTTP transport (2025-03-26 spec):

1. POST ``initialize`` to ``/mcp`` — get ``Mcp-Session-Id`` header back.
2. POST ``tools/call`` to ``/mcp`` with that session header for each query.

Usage::

    from nexus.integrations.pieces import PiecesMCPClient

    client = PiecesMCPClient()
    if await client.connect():
        activity = await client.get_recent_activity()
        print(activity)
    await client.close()
"""

from __future__ import annotations

import json
import logging
from types import TracebackType

import aiohttp

log = logging.getLogger(__name__)

# MCP endpoint path (Streamable HTTP transport)
_MCP_PATH = "/model_context_protocol/2025-03-26/mcp"


class PiecesMCPClient:
    """Async client for the PiecesOS MCP server.

    Establishes a session via the ``initialize`` handshake, then uses
    the ``Mcp-Session-Id`` header for all subsequent tool calls.

    Args:
        base_url: HTTP base URL for the PiecesOS MCP server.
    """

    def __init__(
        self,
        base_url: str = "http://host.docker.internal:39300",
    ) -> None:
        self.base_url: str = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None
        self._connected: bool = False
        self._session_id: str | None = None
        self._request_id: int = 0

    # -- Async context manager ------------------------------------------------

    async def __aenter__(self) -> PiecesMCPClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    # -- Helpers --------------------------------------------------------------

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Return the shared HTTP session, creating it lazily if needed."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _next_id(self) -> int:
        """Return a monotonically increasing request ID."""
        self._request_id += 1
        return self._request_id

    @property
    def _mcp_url(self) -> str:
        return f"{self.base_url}{_MCP_PATH}"

    def _headers(self) -> dict[str, str]:
        """Headers for MCP requests, including session ID if available."""
        h: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            h["Mcp-Session-Id"] = self._session_id
        return h

    # -- Connection management ------------------------------------------------

    async def connect(self) -> bool:
        """Initialize an MCP session with PiecesOS.

        Sends the ``initialize`` JSON-RPC request.  The server returns
        the ``Mcp-Session-Id`` header which is used for all subsequent
        requests.

        Returns:
            ``True`` if the session was established, ``False`` otherwise.
        """
        try:
            log.info("Connecting to PiecesOS at %s", self._mcp_url)
            session = await self._ensure_session()
            payload = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"tools": {}},
                    "clientInfo": {
                        "name": "agent-nexus",
                        "version": "0.1.0",
                    },
                },
            }

            async with session.post(
                self._mcp_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    log.warning(
                        "PiecesOS initialize returned HTTP %d", resp.status,
                    )
                    return False

                self._session_id = resp.headers.get("Mcp-Session-Id")
                if not self._session_id:
                    log.warning(
                        "PiecesOS initialize missing Mcp-Session-Id header",
                    )
                    return False

                # Parse server info for logging
                raw = await resp.text()
                try:
                    data = json.loads(raw)
                    server_info = data.get("result", {}).get(
                        "serverInfo", {},
                    )
                    log.info(
                        "PiecesOS MCP session established "
                        "(server=%s v%s, session=%s)",
                        server_info.get("name", "unknown"),
                        server_info.get("version", "?"),
                        self._session_id[:16] + "...",
                    )
                except (json.JSONDecodeError, TypeError):
                    log.info(
                        "PiecesOS MCP session established (session=%s)",
                        self._session_id[:16] + "...",
                    )

            self._connected = True
            return True

        except Exception as exc:
            log.warning(
                "PiecesOS not available at %s (%s): %s",
                self._mcp_url,
                type(exc).__name__,
                exc or "(no details)",
            )
            self._connected = False
            self._session_id = None
            return False

    # -- Public API -----------------------------------------------------------

    async def get_recent_activity(
        self,
        query: str = "recent activity and context",
        _is_retry: bool = False,
    ) -> str | None:
        """Query PiecesOS LTM for recent user activity.

        Posts a ``tools/call`` request for ``ask_pieces_ltm`` with the
        established ``Mcp-Session-Id``.  Auto-reconnects if the session
        has expired.

        Args:
            query: Free-text query forwarded to the LTM engine.

        Returns:
            A text summary of recent activity, or ``None`` if unavailable.
        """
        # Auto-reconnect if needed
        if not self._connected or self._session_id is None:
            if not await self.connect():
                return None
        if self._session is None:
            return None

        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": "ask_pieces_ltm",
                "arguments": {"query": query},
            },
        }

        try:
            async with self._session.post(
                self._mcp_url,
                json=payload,
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    log.warning(
                        "Pieces LTM query returned HTTP %d", resp.status,
                    )
                    # Session expired — reconnect and retry once
                    if resp.status in (400, 404) and not _is_retry:
                        self._connected = False
                        self._session_id = None
                        log.info("Pieces session expired, reconnecting...")
                        return await self.get_recent_activity(
                            query=query, _is_retry=True,
                        )
                    return None

                # Response may be text/plain — force parse
                raw = await resp.text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    log.warning("Pieces non-JSON response: %.200s", raw)
                    return None

                # Check for JSON-RPC error
                if "error" in data:
                    log.warning(
                        "Pieces LTM error: %s",
                        data["error"].get("message", data["error"]),
                    )
                    return None

                result = data.get("result", {})
                content = result.get("content", [])
                if content and isinstance(content, list):
                    text = content[0].get("text", "")
                    log.info("Pieces LTM returned %d chars", len(text))
                    return text

                log.warning(
                    "Pieces LTM empty content: result=%s",
                    str(result)[:200],
                )
                return None

        except Exception as exc:
            log.warning(
                "Pieces LTM query failed (%s): %s",
                type(exc).__name__,
                exc or "(no details)",
            )
            return None

    # -- Lifecycle ------------------------------------------------------------

    async def close(self) -> None:
        """Close all connections. Safe to call multiple times."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
        self._connected = False
        self._session_id = None

    @property
    def is_connected(self) -> bool:
        """Whether the client has an active MCP session with PiecesOS."""
        return self._connected
