"""PiecesOS MCP client for activity-stream awareness.

PiecesOS is a productivity tool that tracks user activity and exposes a
Long-Term Memory (LTM) engine via an MCP server.  This module provides an
async client that queries that server so the Agent Nexus swarm can passively
observe what the user is working on -- without the user needing to send
explicit prompts.

Uses the MCP SSE transport (2024-11-05 spec):

1. Open an SSE stream on ``/sse`` to discover the ``messages`` URL.
2. POST ``initialize`` to the messages URL.
3. POST ``tools/call`` to the messages URL for each query.
4. Read responses from the SSE stream.

Each query creates a **fresh SSE connection** to avoid socket state issues.
Includes response caching to reduce redundant LTM queries.

Usage::

    from nexus.integrations.pieces import PiecesMCPClient

    client = PiecesMCPClient()
    if await client.connect():
        activity = await client.get_recent_activity()
        print(activity)
    await client.close()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import socket as sync_socket
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from types import TracebackType
from urllib.parse import urlparse

log = logging.getLogger(__name__)

# MCP protocol version supported by PiecesOS
_MCP_VERSION = "2024-11-05"


class PiecesMCPClient:
    """Async client for the PiecesOS MCP server (SSE transport).

    Each LTM query opens a fresh SSE connection, sends the request via
    the discovered messages URL, and reads the response from the SSE
    stream.  This avoids socket state issues with long-lived connections.

    Includes a response cache to reduce redundant queries.

    Args:
        base_url: HTTP base URL for PiecesOS
            (e.g. ``http://192.168.86.34:39300``).
        cache_ttl_minutes: How long to cache LTM responses.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:39300",
        cache_ttl_minutes: int = 15,
    ) -> None:
        self.base_url: str = base_url.rstrip("/")
        self._connected: bool = False

        # Response cache: hash -> (response_text, timestamp)
        self._cache: dict[str, tuple[str, datetime]] = {}
        self._cache_ttl = timedelta(minutes=cache_ttl_minutes)

        # Parse host/port once for socket connections
        parsed = urlparse(self.base_url)
        self._host: str = parsed.hostname or "localhost"
        self._port: int = parsed.port or 39300

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

    # -- Cache ----------------------------------------------------------------

    def _cache_key(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def _get_cached(self, query: str) -> str | None:
        key = self._cache_key(query)
        if key in self._cache:
            response, ts = self._cache[key]
            if datetime.now(timezone.utc) - ts < self._cache_ttl:
                log.debug("Pieces cache hit: %s", query[:40])
                return response
            del self._cache[key]
        return None

    def _set_cache(self, query: str, response: str) -> None:
        key = self._cache_key(query)
        self._cache[key] = (response, datetime.now(timezone.utc))
        # Prune expired entries if cache grows too large
        if len(self._cache) > 100:
            now = datetime.now(timezone.utc)
            self._cache = {
                k: v for k, v in self._cache.items()
                if now - v[1] < self._cache_ttl
            }

    # -- Low-level sync transport (runs in thread) ----------------------------

    def _sync_discover_messages_url(self, timeout: float = 10.0) -> str | None:
        """Open SSE stream and discover the messages URL.

        Returns the full messages URL, or ``None`` on failure.
        """
        sock = sync_socket.socket(sync_socket.AF_INET, sync_socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((self._host, self._port))
            sock.send(
                f"GET /model_context_protocol/{_MCP_VERSION}/sse HTTP/1.1\r\n"
                f"Host: {self._host}:{self._port}\r\n"
                f"Accept: text/event-stream\r\n"
                f"\r\n".encode()
            )

            data = b""
            while True:
                try:
                    chunk = sock.recv(1024)
                    if not chunk:
                        break
                    data += chunk
                    if b"messages?" in data:
                        break
                except sync_socket.timeout:
                    break

            for line in data.decode(errors="ignore").split("\n"):
                if line.startswith("data:"):
                    path = line[5:].strip()
                    if "messages" in path:
                        return f"{self.base_url}{path}"
            return None
        finally:
            sock.close()

    def _sync_query(self, question: str, timeout: float = 30.0) -> str | None:
        """Run a full SSE query cycle synchronously (called via to_thread).

        Opens a fresh SSE connection, sends ``initialize`` +
        ``tools/call`` for ``ask_pieces_ltm``, and reads the response
        from the SSE stream.
        """
        sock = None
        try:
            # 1. Open SSE stream
            sock = sync_socket.socket(
                sync_socket.AF_INET, sync_socket.SOCK_STREAM,
            )
            sock.settimeout(timeout)
            sock.connect((self._host, self._port))
            sock.send(
                f"GET /model_context_protocol/{_MCP_VERSION}/sse HTTP/1.1\r\n"
                f"Host: {self._host}:{self._port}\r\n"
                f"Accept: text/event-stream\r\n"
                f"\r\n".encode()
            )

            # 2. Read until we find the messages URL
            data = b""
            while True:
                chunk = sock.recv(1024)
                if not chunk:
                    return None
                data += chunk
                if b"messages?" in data:
                    break

            messages_url = None
            for line in data.decode(errors="ignore").split("\n"):
                if line.startswith("data:"):
                    path = line[5:].strip()
                    if "messages" in path:
                        messages_url = f"{self.base_url}{path}"
                        break

            if not messages_url:
                log.warning("Pieces SSE: no messages URL discovered")
                return None

            headers = {"Content-Type": "application/json"}

            # 3. Send initialize
            init_payload = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": _MCP_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "agent-nexus", "version": "1.0"},
                },
            }).encode()

            init_req = urllib.request.Request(
                messages_url,
                data=init_payload,
                headers=headers,
                method="POST",
            )
            urllib.request.urlopen(init_req, timeout=10)
            time.sleep(0.3)
            sock.recv(8192)  # Clear init response from SSE buffer

            # 4. Send LTM query
            query_payload = json.dumps({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "ask_pieces_ltm",
                    "arguments": {"question": question},
                },
            }).encode()

            query_req = urllib.request.Request(
                messages_url,
                data=query_payload,
                headers=headers,
                method="POST",
            )
            urllib.request.urlopen(query_req, timeout=30)

            # 5. Read response from SSE stream (look for id: 2)
            buffer = b""
            start = time.time()
            while time.time() - start < timeout:
                try:
                    chunk = sock.recv(8192)
                    if chunk:
                        buffer += chunk

                    if b'"id":2' in buffer or b'"id": 2' in buffer:
                        decoded = buffer.decode(errors="ignore")
                        for sse_line in decoded.split("\n"):
                            if "data:" not in sse_line:
                                continue
                            if '"id":2' not in sse_line and '"id": 2' not in sse_line:
                                continue
                            json_start = sse_line.find("{")
                            if json_start < 0:
                                continue
                            try:
                                resp = json.loads(sse_line[json_start:])
                            except json.JSONDecodeError:
                                continue

                            if "error" in resp:
                                log.warning(
                                    "Pieces LTM error: %s",
                                    resp["error"].get(
                                        "message", resp["error"],
                                    ),
                                )
                                return None

                            if "result" in resp:
                                content = resp["result"].get("content", [])
                                if content:
                                    item = content[0]
                                    if isinstance(item, dict) and "text" in item:
                                        return item["text"]
                                    if isinstance(item, str):
                                        return item
                except sync_socket.timeout:
                    break

            log.warning("Pieces LTM query timed out after %.0fs", timeout)
            return None

        except Exception as exc:
            log.warning(
                "Pieces sync query failed (%s): %s",
                type(exc).__name__,
                exc or "(no details)",
            )
            return None
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

    # -- Connection management ------------------------------------------------

    async def connect(self) -> bool:
        """Test connection to PiecesOS MCP server.

        Attempts to open the SSE endpoint and discover the messages URL
        to verify the server is reachable.

        Returns:
            ``True`` if the server is reachable, ``False`` otherwise.
        """
        sse_url = (
            f"{self.base_url}/model_context_protocol/{_MCP_VERSION}/sse"
        )
        log.info("Connecting to PiecesOS at %s (SSE transport)", sse_url)
        try:
            url = await asyncio.to_thread(
                self._sync_discover_messages_url, 10.0,
            )
            if url:
                log.info("PiecesOS MCP connected (messages URL discovered)")
                self._connected = True
                return True

            log.warning(
                "PiecesOS not available at %s: "
                "SSE stream did not return messages URL",
                self.base_url,
            )
            self._connected = False
            return False

        except Exception as exc:
            log.warning(
                "PiecesOS not available at %s (%s): %s",
                self.base_url,
                type(exc).__name__,
                exc or "(no details)",
            )
            self._connected = False
            return False

    # -- Public API -----------------------------------------------------------

    async def get_recent_activity(
        self,
        query: str = "What has the user been working on recently?",
    ) -> str | None:
        """Query PiecesOS LTM for recent user activity.

        Each call opens a fresh SSE connection to avoid stale socket
        issues.  Results are cached to reduce redundant queries.

        Args:
            query: Question forwarded to the ``ask_pieces_ltm`` tool.

        Returns:
            A text summary of recent activity, or ``None`` if unavailable.
        """
        if not self._connected:
            if not await self.connect():
                return None

        # Check cache first
        cached = self._get_cached(query)
        if cached is not None:
            return cached

        try:
            result = await asyncio.to_thread(self._sync_query, query, 30.0)
            if result:
                log.info("Pieces LTM returned %d chars", len(result))
                self._set_cache(query, result)
                return result
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
        """Clean up. Safe to call multiple times."""
        self._connected = False
        self._cache.clear()

    @property
    def is_connected(self) -> bool:
        """Whether the client has successfully connected to PiecesOS."""
        return self._connected
