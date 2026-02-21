"""Continuity Core (C2) subprocess MCP client.

Spawns the C2 MCP server as a subprocess and communicates via stdin/stdout
JSON-RPC.  Provides async methods for each C2 tool: write_event, context,
introspect, curiosity, and maintenance.

The subprocess runs ``python -m continuity_core.mcp.server`` which reads
newline-delimited JSON-RPC from stdin and writes responses to stdout.

Usage::

    c2 = C2Client()
    if await c2.start():
        result = await c2.curiosity()
        print(result)
    await c2.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

log = logging.getLogger(__name__)


class C2Client:
    """Async client for the Continuity Core MCP subprocess.

    Manages the lifecycle of the C2 MCP server subprocess and provides
    typed async wrappers for each tool.
    """

    _REQUEST_TIMEOUT: float = 90.0
    _WARMUP_TIMEOUT: float = 120.0
    _RESTART_DELAY: float = 2.0

    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._request_id: int = 0
        self._lock = asyncio.Lock()
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> bool:
        """Spawn the C2 MCP subprocess and initialize the protocol.

        Returns ``True`` if the subprocess started and the MCP handshake
        succeeded, ``False`` otherwise.
        """
        try:
            self._process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "continuity_core.mcp.server",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # MCP initialization handshake
            resp = await self._send("initialize", {})
            if resp and "result" in resp:
                server_info = resp["result"].get("serverInfo", {})
                log.info(
                    "C2 subprocess started (pid=%d, server=%s v%s)",
                    self._process.pid,
                    server_info.get("name", "unknown"),
                    server_info.get("version", "?"),
                )
                self._initialized = True

                # Warmup: first tool call triggers lazy backend init which
                # can take 30-60s.  Use extended timeout so we don't desync.
                log.info("C2 warmup: initializing backends...")
                warmup = await self._send_with_timeout(
                    "tools/call",
                    {"name": "c2.status", "arguments": {}},
                    timeout=self._WARMUP_TIMEOUT,
                )
                if warmup and "result" in warmup:
                    log.info("C2 warmup complete — backends ready")
                else:
                    log.warning("C2 warmup failed — backends may be unavailable")

                return True

            log.warning("C2 initialization handshake failed: %s", resp)
            await self.stop()
            return False

        except Exception as exc:
            log.warning("Failed to start C2 subprocess: %s", exc)
            self._process = None
            return False

    async def stop(self) -> None:
        """Terminate the C2 subprocess gracefully."""
        self._initialized = False
        if self._process is not None:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    self._process.kill()
                except ProcessLookupError:
                    pass
            self._process = None
            log.info("C2 subprocess stopped")

    @property
    def is_running(self) -> bool:
        """Whether the C2 subprocess is alive and initialized."""
        return (
            self._initialized
            and self._process is not None
            and self._process.returncode is None
        )

    # ------------------------------------------------------------------
    # JSON-RPC transport
    # ------------------------------------------------------------------

    async def _send(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Send a JSON-RPC request and read the matching response."""
        return await self._send_with_timeout(
            method, params, timeout=self._REQUEST_TIMEOUT,
        )

    async def _send_with_timeout(
        self, method: str, params: dict[str, Any], *, timeout: float,
    ) -> dict[str, Any] | None:
        """Send a JSON-RPC request and read the response with ID matching.

        Skips stale responses from previously timed-out calls to keep the
        stdin/stdout pipe synchronized.
        """
        if self._process is None or self._process.stdin is None:
            return None

        async with self._lock:
            self._request_id += 1
            req_id = self._request_id
            request = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params,
            }

            try:
                line = json.dumps(request) + "\n"
                self._process.stdin.write(line.encode())
                await self._process.stdin.drain()

                if self._process.stdout is None:
                    return None

                # Read lines until we get the response matching our request ID.
                # This skips stale responses from previous timed-out calls.
                deadline = asyncio.get_event_loop().time() + timeout
                while True:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError()

                    raw = await asyncio.wait_for(
                        self._process.stdout.readline(),
                        timeout=remaining,
                    )
                    if not raw:
                        log.warning("C2 subprocess returned empty response")
                        return None

                    resp = json.loads(raw.decode())
                    if resp.get("id") == req_id:
                        return resp

                    log.debug(
                        "C2 skipping stale response (got id=%s, want id=%s)",
                        resp.get("id"), req_id,
                    )

            except asyncio.TimeoutError:
                log.warning("C2 request timed out: method=%s", method)
                return None
            except Exception as exc:
                log.warning("C2 request failed: %s", exc)
                return None

    async def _call_tool(
        self, name: str, arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Call a C2 MCP tool and return the parsed result.

        Handles auto-restart if the subprocess has died.
        """
        if not self.is_running:
            log.debug("C2 not running, attempting restart for %s", name)
            await asyncio.sleep(self._RESTART_DELAY)
            if not await self.start():
                return None

        resp = await self._send("tools/call", {
            "name": name,
            "arguments": arguments or {},
        })

        if resp is None:
            return None

        if "error" in resp:
            log.warning("C2 tool %s error: %s", name, resp["error"])
            return None

        # Parse the text content from MCP response envelope
        result = resp.get("result", {})
        content = result.get("content", [])
        if content and isinstance(content, list):
            text = content[0].get("text", "{}")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text}

        return result

    # ------------------------------------------------------------------
    # Tool wrappers
    # ------------------------------------------------------------------

    async def write_event(
        self,
        actor: str,
        intent: str,
        inp: str = "",
        out: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Append an event to the C2 Event Log."""
        return await self._call_tool("c2.write_event", {
            "actor": actor,
            "intent": intent,
            "input": inp,
            "output": out,
            "tags": tags or [],
            "metadata": metadata or {},
        })

    async def get_context(
        self, query: str, token_budget: int = 2048,
    ) -> dict[str, Any] | None:
        """Compose a context pack from C2 memory."""
        return await self._call_tool("c2.context", {
            "query": query,
            "token_budget": token_budget,
        })

    async def introspect(
        self,
        statements: list[str],
        concept_contexts: dict[str, Any] | None = None,
        graph: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Run MRA stress and void detection."""
        return await self._call_tool("c2.introspect", {
            "statements": statements,
            "concept_contexts": concept_contexts or {},
            "graph": graph or {},
        })

    async def curiosity(self) -> dict[str, Any] | None:
        """Return epistemic tensions, contradictions, and bridging questions."""
        return await self._call_tool("c2.curiosity")

    async def maintenance(
        self, graph: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Run a Night Cycle maintenance pass."""
        return await self._call_tool("c2.maintenance", {
            "graph": graph or {},
        })

    async def status(self) -> dict[str, Any] | None:
        """Query C2 backend health and system metrics."""
        return await self._call_tool("c2.status")

    async def events(self, limit: int = 10) -> dict[str, Any] | None:
        """Read recent events from the C2 event log."""
        return await self._call_tool("c2.events", {"limit": limit})
