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
import re
import socket as sync_socket
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import TracebackType
from urllib.parse import urlparse

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activity digest: parsed PiecesOS LTM response
# ---------------------------------------------------------------------------


@dataclass
class ActivityDigest:
    """Structured summary parsed from a PiecesOS LTM response."""

    projects: list[str] = field(default_factory=list)
    summary: str = ""
    recent_focus: str = ""
    raw_summaries: list[str] = field(default_factory=list)
    active_apps: list[str] = field(default_factory=list)
    timestamp: str = ""
    most_recent_at: str = ""  # ISO timestamp of newest summary

    @property
    def is_empty(self) -> bool:
        return not self.summary and not self.projects

    @property
    def age_hours(self) -> float | None:
        """Hours since the most recent summary, or None if unknown."""
        if not self.most_recent_at:
            return None
        try:
            created = datetime.fromisoformat(
                self.most_recent_at.replace("Z", "+00:00"),
            )
            delta = datetime.now(timezone.utc) - created
            return delta.total_seconds() / 3600
        except (ValueError, TypeError):
            return None

    @property
    def age_description(self) -> str:
        """Human-readable age like '2h ago' or '3d ago'."""
        hours = self.age_hours
        if hours is None:
            return ""
        if hours < 1:
            return f"{int(hours * 60)}m ago"
        if hours < 24:
            return f"{int(hours)}h ago"
        days = hours / 24
        return f"{int(days)}d ago"

    @property
    def is_stale(self) -> bool:
        """True if the most recent summary is older than 6 hours."""
        hours = self.age_hours
        return hours is not None and hours > 6


# Regex for "### Core Tasks & Projects" section items and TL;DR project names.
_PROJECT_LINE_RE = re.compile(
    r"[-*]\s+(?:\*\*)?([A-Z][A-Za-z0-9 _\-/:.]+?)(?:\*\*)?(?:\s*[:;]|\s*$)",
)
# Broad pattern for project-like names in TL;DR (capitalized multi-word).
_TLDR_PROJECT_RE = re.compile(
    r"\b([A-Z][a-z]+(?:[-_ ][A-Z][a-z]+)+(?:[-_ ][A-Z0-9]+)*)\b",
)
# Garbage app titles that PiecesOS sometimes returns.
_GARBAGE_APP_TITLES = frozenset(
    {
        "[COULD NOT RETRIEVE APP TITLE]",
        "unknown",
        "",
    }
)


def _strip_summary_metadata(text: str) -> str:
    """Strip the PiecesOS metadata header from a combined_string.

    PiecesOS prepends lines like:
        Automated Summary:
        Created: 2 days, 20 hrs ago (2026-02-20 ...)
        Summarized time-range: ...

    Strip everything before the first markdown heading (## or ###).
    """
    # Find first markdown heading.
    heading_match = re.search(r"^#{2,3}\s", text, re.MULTILINE)
    if heading_match:
        return text[heading_match.start() :].strip()
    # No headings — strip known metadata prefixes line by line.
    lines = text.split("\n")
    content_lines: list[str] = []
    past_metadata = False
    for line in lines:
        stripped = line.strip()
        if not past_metadata:
            if stripped.lower().startswith(
                ("automated summary", "created:", "summarized time-range")
            ):
                continue
            if not stripped:
                continue
            past_metadata = True
        content_lines.append(line)
    return "\n".join(content_lines).strip() if content_lines else text.strip()


def _extract_projects(text: str) -> list[str]:
    """Extract project names from a PiecesOS summary string."""
    projects: list[str] = []
    seen: set[str] = set()

    # 1. Look for "### Core Tasks & Projects" section lines.
    in_core_section = False
    for line in text.split("\n"):
        stripped = line.strip()
        if "core tasks" in stripped.lower() and "project" in stripped.lower():
            in_core_section = True
            continue
        if in_core_section:
            if stripped.startswith("#") and "core" not in stripped.lower():
                in_core_section = False
                continue
            m = _PROJECT_LINE_RE.match(stripped)
            if m:
                name = m.group(1).strip()
                if len(name) >= 3 and name.lower() not in seen:
                    seen.add(name.lower())
                    projects.append(name)

    # 2. Look for project-like names in TL;DR section.
    tldr_start = text.lower().find("tl;dr")
    if tldr_start >= 0:
        tldr_block = text[tldr_start : tldr_start + 500]
        for m in _TLDR_PROJECT_RE.finditer(tldr_block):
            name = m.group(1).strip()
            if len(name) >= 5 and name.lower() not in seen:
                # Skip common false positives.
                if name.lower() not in {
                    "the user",
                    "this week",
                    "last week",
                    "today",
                    "yesterday",
                }:
                    seen.add(name.lower())
                    projects.append(name)

    return projects[:10]


def _extract_tldr(text: str) -> str:
    """Extract the TL;DR section from a PiecesOS summary."""
    tldr_start = text.lower().find("tl;dr")
    if tldr_start < 0:
        return ""
    # Skip the "## TL;DR\n" header.
    content_start = text.find("\n", tldr_start)
    if content_start < 0:
        return ""
    content_start += 1
    # Read until the next section header or end.
    next_header = text.find("\n#", content_start)
    if next_header >= 0:
        return text[content_start:next_header].strip()
    return text[content_start : content_start + 500].strip()


def parse_activity_response(raw: str) -> ActivityDigest:
    """Parse a PiecesOS LTM response into a structured digest.

    Handles two formats:
    - JSON with ``summaries`` and ``events`` arrays (structured response)
    - Plain text narrative (direct LTM summary)

    Args:
        raw: The raw text from the ``ask_pieces_ltm`` MCP tool.

    Returns:
        An :class:`ActivityDigest` with extracted fields.
    """
    if not raw or not raw.strip():
        return ActivityDigest(timestamp=datetime.now(timezone.utc).isoformat())

    now_iso = datetime.now(timezone.utc).isoformat()

    # Try JSON parse first.
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        data = None

    if isinstance(data, dict) and "summaries" in data:
        return _parse_structured(data, now_iso)

    # Plain text fallback.
    projects = _extract_projects(raw)
    tldr = _extract_tldr(raw)
    summary = tldr if tldr else raw[:500]
    return ActivityDigest(
        projects=projects,
        summary=summary,
        recent_focus=summary[:300],
        raw_summaries=[raw[:2000]],
        active_apps=[],
        timestamp=now_iso,
    )


def _parse_structured(data: dict, now_iso: str) -> ActivityDigest:
    """Parse a JSON response with ``summaries`` and ``events`` arrays."""
    summaries_raw = data.get("summaries", [])
    events_raw = data.get("events", [])

    # Sort summaries by created timestamp (most recent first).
    valid_summaries: list[dict] = []
    for s in summaries_raw:
        if not isinstance(s, dict):
            continue
        combined = s.get("combined_string", "")
        if not combined:
            continue
        valid_summaries.append(s)

    valid_summaries.sort(
        key=lambda s: s.get("created", ""),
        reverse=True,
    )

    # Track the most recent summary's timestamp.
    most_recent_at = ""
    if valid_summaries:
        most_recent_at = valid_summaries[0].get("created", "")

    # Process summaries: strip metadata, extract projects and content.
    raw_summaries: list[str] = []
    all_projects: list[str] = []
    best_summary = ""
    best_focus = ""

    for i, s in enumerate(valid_summaries):
        combined = s.get("combined_string", "")
        cleaned = _strip_summary_metadata(combined)
        raw_summaries.append(cleaned[:2000])

        # Extract projects from this summary.
        for p in _extract_projects(cleaned):
            if p not in all_projects:
                all_projects.append(p)

        # Most recent summary (i=0) provides the focus and summary.
        if i == 0:
            tldr = _extract_tldr(cleaned)
            best_focus = tldr if tldr else cleaned[:300]
            best_summary = tldr if tldr else cleaned[:500]

    # Extract active apps from events, filtering garbage titles.
    active_apps: list[str] = []
    seen_apps: set[str] = set()
    for ev in events_raw:
        if not isinstance(ev, dict):
            continue
        app = ev.get("app_title", "")
        if app and app not in _GARBAGE_APP_TITLES and app not in seen_apps:
            # Clean up ".exe" suffix.
            clean = app.removesuffix(".exe").strip()
            if clean and clean not in _GARBAGE_APP_TITLES:
                seen_apps.add(app)
                active_apps.append(clean)
        window = ev.get("window_title", "")
        if window and not app:
            if window not in _GARBAGE_APP_TITLES:
                active_apps.append(window[:60])

    return ActivityDigest(
        projects=all_projects[:10],
        summary=best_summary[:1000],
        recent_focus=best_focus[:500],
        raw_summaries=raw_summaries[:5],
        active_apps=active_apps[:10],
        timestamp=now_iso,
        most_recent_at=most_recent_at,
    )


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
            self._cache = {k: v for k, v in self._cache.items() if now - v[1] < self._cache_ttl}

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
                sync_socket.AF_INET,
                sync_socket.SOCK_STREAM,
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
            init_payload = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": _MCP_VERSION,
                        "capabilities": {},
                        "clientInfo": {"name": "agent-nexus", "version": "1.0"},
                    },
                }
            ).encode()

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
            query_payload = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "ask_pieces_ltm",
                        "arguments": {"question": question},
                    },
                }
            ).encode()

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
                                        "message",
                                        resp["error"],
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
        sse_url = f"{self.base_url}/model_context_protocol/{_MCP_VERSION}/sse"
        log.info("Connecting to PiecesOS at %s (SSE transport)", sse_url)
        try:
            url = await asyncio.to_thread(
                self._sync_discover_messages_url,
                10.0,
            )
            if url:
                log.info("PiecesOS MCP connected (messages URL discovered)")
                self._connected = True
                return True

            log.warning(
                "PiecesOS not available at %s: SSE stream did not return messages URL",
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

    async def get_activity_digest(
        self,
        query: str = "What has the user been working on recently?",
    ) -> ActivityDigest | None:
        """Query PiecesOS LTM and return a parsed activity digest.

        Convenience wrapper around :meth:`get_recent_activity` that
        parses the raw response into a structured :class:`ActivityDigest`.

        Args:
            query: Question forwarded to the ``ask_pieces_ltm`` tool.

        Returns:
            A parsed :class:`ActivityDigest`, or ``None`` if unavailable.
        """
        raw = await self.get_recent_activity(query=query)
        if raw is None:
            return None
        return parse_activity_response(raw)

    # -- Lifecycle ------------------------------------------------------------

    async def close(self) -> None:
        """Clean up. Safe to call multiple times."""
        self._connected = False
        self._cache.clear()

    @property
    def is_connected(self) -> bool:
        """Whether the client has successfully connected to PiecesOS."""
        return self._connected
