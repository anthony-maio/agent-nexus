"""PiecesOS MCP client for activity-stream awareness.

PiecesOS is a retail productivity tool that tracks user activity (screen
content, clipboard, keystrokes) and exposes a Long-Term Memory (LTM) engine
via an MCP server.  This module provides an async client that queries that
server so the Agent Nexus swarm can passively observe what the user is
working on -- without the user needing to send explicit prompts.

The MCP server uses SSE transport and listens on ``localhost:39300`` by
default.  When running inside Docker the client connects via
``host.docker.internal:39300`` so it can reach the host machine.

Usage::

    from nexus.integrations.pieces import PiecesMCPClient

    async with PiecesMCPClient() as client:
        if await client.connect():
            activity = await client.get_recent_activity()
            print(activity)
"""

from __future__ import annotations

import logging
from types import TracebackType

import aiohttp

log = logging.getLogger(__name__)


class PiecesMCPClient:
    """Async client for the PiecesOS MCP server.

    PiecesOS provides a Long-Term Memory (LTM) engine that tracks user
    activity.  The MCP server exposes tools via SSE transport at
    ``localhost:39300``.

    Primary tool: ``ask_pieces_ltm`` -- queries activity context.

    Args:
        base_url: HTTP base URL for the PiecesOS MCP server.  Defaults to
            ``http://host.docker.internal:39300`` which works when Agent
            Nexus runs inside a Docker container on the same host.
    """

    def __init__(
        self,
        base_url: str = "http://host.docker.internal:39300",
    ) -> None:
        self.base_url: str = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None
        self._connected: bool = False

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

    # -- Connection management ------------------------------------------------

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Return the shared HTTP session, creating it lazily if needed."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def connect(self) -> bool:
        """Test connectivity to the PiecesOS MCP server.

        Sends a lightweight health-check request.  On success the client
        is marked as connected and subsequent calls to
        :meth:`get_recent_activity` will attempt real queries.

        Returns:
            ``True`` if the server responded successfully, ``False``
            otherwise (the failure is logged as a warning, not raised).
        """
        try:
            session = await self._ensure_session()
            async with session.get(
                f"{self.base_url}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                self._connected = resp.status == 200
                if self._connected:
                    log.info("PiecesOS MCP connected at %s", self.base_url)
                else:
                    log.warning(
                        "PiecesOS health-check returned HTTP %d",
                        resp.status,
                    )
                return self._connected
        except Exception as exc:
            log.warning("PiecesOS not available: %s", exc)
            self._connected = False
            return False

    # -- Public API -----------------------------------------------------------

    async def get_recent_activity(
        self,
        query: str = "recent activity and context",
    ) -> str | None:
        """Query PiecesOS LTM for recent user activity.

        Uses the ``ask_pieces_ltm`` tool exposed by the MCP server to
        retrieve a natural-language summary of what the user has been
        doing recently.

        Args:
            query: Free-text query forwarded to the LTM engine.  Defaults
                to a broad "recent activity and context" request.

        Returns:
            A text summary of recent activity, or ``None`` if the server
            is unreachable or the query fails.
        """
        if not self._connected or self._session is None:
            return None

        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "ask_pieces_ltm",
                    "arguments": {"query": query},
                },
            }
            mcp_url = (
                f"{self.base_url}/model_context_protocol/2025-03-26/mcp"
            )
            async with self._session.post(
                mcp_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    log.warning(
                        "Pieces LTM query returned HTTP %d", resp.status,
                    )
                    return None

                data = await resp.json()
                result = data.get("result", {})
                content = result.get("content", [])
                if content and isinstance(content, list):
                    return content[0].get("text", "")
                return None
        except Exception as exc:
            log.warning("Pieces LTM query failed: %s", exc)
            return None

    # -- Lifecycle ------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP session.

        Safe to call multiple times.  After closing, :attr:`is_connected`
        returns ``False`` and a new session will be created on the next
        :meth:`connect` call.
        """
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Whether the client has an active connection to PiecesOS."""
        return self._connected
