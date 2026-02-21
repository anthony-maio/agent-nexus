"""PiecesOS MCP client for activity-stream awareness.

PiecesOS is a retail productivity tool that tracks user activity (screen
content, clipboard, keystrokes) and exposes a Long-Term Memory (LTM) engine
via an MCP server.  This module provides an async client that queries that
server so the Agent Nexus swarm can passively observe what the user is
working on -- without the user needing to send explicit prompts.

The MCP server uses SSE transport (2024-11-05 spec) at ``localhost:39300``.
On connect, we read the SSE stream to discover the messages endpoint URL
(which includes a sessionId and token), then POST JSON-RPC tool calls there.

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

# SSE endpoint path for the 2024-11-05 MCP transport
_SSE_PATH = "/model_context_protocol/2024-11-05/sse"


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
        self._messages_url: str | None = None
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

    # -- Connection management ------------------------------------------------

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Return the shared HTTP session, creating it lazily if needed."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _next_id(self) -> int:
        """Return a monotonically increasing request ID."""
        self._request_id += 1
        return self._request_id

    async def connect(self) -> bool:
        """Establish a session with the PiecesOS MCP server via SSE.

        Connects to the SSE endpoint and reads the first ``endpoint``
        event to discover the messages URL (which includes sessionId and
        token parameters).

        Returns:
            ``True`` if the session was established, ``False`` otherwise.
        """
        try:
            session = await self._ensure_session()
            sse_url = f"{self.base_url}{_SSE_PATH}"

            async with session.get(
                sse_url,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    log.warning(
                        "PiecesOS SSE endpoint returned HTTP %d",
                        resp.status,
                    )
                    return False

                # Read the SSE stream until we get the endpoint event.
                # Format: "event: endpoint\ndata: /path?sessionId=...&token=...\n\n"
                messages_path: str | None = None
                event_type: str | None = None

                async for raw_line in resp.content:
                    line = raw_line.decode().rstrip("\r\n")

                    if line.startswith("event:"):
                        event_type = line[len("event:"):].strip()
                    elif line.startswith("data:") and event_type == "endpoint":
                        messages_path = line[len("data:"):].strip()
                        break
                    elif not line:
                        # Empty line = end of event; reset if we didn't
                        # match the one we wanted.
                        event_type = None

                if not messages_path:
                    log.warning(
                        "PiecesOS SSE did not return an endpoint event",
                    )
                    return False

                # Build full messages URL from the relative path
                self._messages_url = f"{self.base_url}{messages_path}"
                self._connected = True
                log.info(
                    "PiecesOS MCP connected at %s (messages=%s)",
                    self.base_url,
                    self._messages_url[:80],
                )
                return True

        except Exception as exc:
            log.warning("PiecesOS not available: %s", exc)
            self._connected = False
            self._messages_url = None
            await self.close()
            return False

    # -- Public API -----------------------------------------------------------

    async def get_recent_activity(
        self,
        query: str = "recent activity and context",
    ) -> str | None:
        """Query PiecesOS LTM for recent user activity.

        Uses the ``ask_pieces_ltm`` tool exposed by the MCP server to
        retrieve a natural-language summary of what the user has been
        doing recently.  Automatically reconnects if the session expired.

        Args:
            query: Free-text query forwarded to the LTM engine.  Defaults
                to a broad "recent activity and context" request.

        Returns:
            A text summary of recent activity, or ``None`` if the server
            is unreachable or the query fails.
        """
        # Auto-reconnect if session expired
        if not self._connected or self._messages_url is None:
            if not await self.connect():
                return None
        if self._session is None:
            return None

        try:
            payload = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {
                    "name": "ask_pieces_ltm",
                    "arguments": {"query": query},
                },
            }
            async with self._session.post(
                self._messages_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    log.warning(
                        "Pieces LTM query returned HTTP %d", resp.status,
                    )
                    # Session may have expired — mark for reconnect.
                    if resp.status in (400, 404):
                        self._connected = False
                        self._messages_url = None
                    return None

                data = await resp.json()

                # Check for JSON-RPC error
                if "error" in data:
                    log.warning(
                        "Pieces LTM query error: %s",
                        data["error"].get("message", data["error"]),
                    )
                    return None

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
        self._messages_url = None

    @property
    def is_connected(self) -> bool:
        """Whether the client has an active MCP session with PiecesOS."""
        return self._connected
