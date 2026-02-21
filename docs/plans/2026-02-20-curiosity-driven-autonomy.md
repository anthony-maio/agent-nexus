# Curiosity-Driven Autonomous Action Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the Agent Nexus swarm autonomously investigate epistemic tensions from C2, fix silent infrastructure failures, and add visibility into C2 internals.

**Architecture:** Add Neo4j + PostgreSQL to Docker with optional profiles, add OpenRouter cloud embeddings to C2, make the night cycle trigger Tier 1 group discussions on curiosity findings, add `!c2status`, `!c2events`, and `!discuss` commands.

**Tech Stack:** Docker Compose profiles, Neo4j 5 Community, PostgreSQL 16, OpenRouter embeddings API, discord.py embeds, C2 MCP JSON-RPC tools.

---

### Task 1: Docker Infrastructure — Add Neo4j + PostgreSQL with Profiles

**Files:**
- Modify: `docker/docker-compose.yml`

**Context:** The Docker stack currently has `nexus-bot`, `nexus-qdrant`, and `nexus-redis`. C2 needs Neo4j (knowledge graph) and PostgreSQL (event log) but both are missing. All infrastructure services should use Docker Compose profiles so users who already have these services can skip them and provide their own connection URLs.

**Step 1: Add Neo4j service with profile**

Add to `docker/docker-compose.yml` after the `nexus-redis` service:

```yaml
  nexus-neo4j:
    image: neo4j:5-community
    container_name: nexus-neo4j
    restart: unless-stopped
    profiles:
      - neo4j
    environment:
      NEO4J_AUTH: neo4j/nexus-c2-graph
    ports:
      - "127.0.0.1:${NEO4J_BOLT_PORT:-7687}:7687"
      - "127.0.0.1:${NEO4J_HTTP_PORT:-7474}:7474"
    volumes:
      - neo4j_data:/data
    healthcheck:
      test: ["CMD-SHELL", "wget -qO- http://localhost:7474 || exit 1"]
      interval: 15s
      timeout: 10s
      retries: 5
    networks:
      - nexus-net
    deploy:
      resources:
        limits:
          memory: 512M
```

**Step 2: Add PostgreSQL service with profile**

Add after `nexus-neo4j`:

```yaml
  nexus-postgres:
    image: postgres:16-alpine
    container_name: nexus-postgres
    restart: unless-stopped
    profiles:
      - postgres
    environment:
      POSTGRES_USER: c2
      POSTGRES_PASSWORD: c2
      POSTGRES_DB: continuity_core
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U c2 -d continuity_core"]
      interval: 10s
      timeout: 5s
      retries: 3
    networks:
      - nexus-net
```

**Step 3: Add profiles to existing Qdrant and Redis services**

Add `profiles: [qdrant]` to `nexus-qdrant` and `profiles: [redis]` to `nexus-redis`. This lets users skip them if they provide external URLs.

**Step 4: Add C2 environment variables to nexus-bot**

Add these to the `nexus-bot` service `environment` section:

```yaml
      # C2 backend connections (override with external URLs if not using Docker services)
      C2_NEO4J_URI: "${C2_NEO4J_URI:-bolt://nexus-neo4j:7687}"
      C2_NEO4J_USER: "${C2_NEO4J_USER:-neo4j}"
      C2_NEO4J_PASSWORD: "${C2_NEO4J_PASSWORD:-nexus-c2-graph}"
      C2_POSTGRES_URL: "${C2_POSTGRES_URL:-postgresql://c2:c2@nexus-postgres:5432/continuity_core}"
      C2_QDRANT_URL: "${C2_QDRANT_URL:-http://nexus-qdrant:6333}"
      C2_REDIS_URL: "${C2_REDIS_URL:-redis://nexus-redis:6379/0}"
      C2_EMBEDDING_BACKEND: "${C2_EMBEDDING_BACKEND:-openrouter}"
      C2_OPENROUTER_API_KEY: "${OPENROUTER_API_KEY}"
      C2_OPENROUTER_EMBED_MODEL: "${C2_OPENROUTER_EMBED_MODEL:-qwen/qwen3-embedding-8b}"
```

**Step 5: Update volumes**

Add `neo4j_data:` and `postgres_data:` to the `volumes:` section.

**Step 6: Update nexus-bot depends_on**

Make `depends_on` conditional — the bot should not hard-depend on services the user might not run. Remove the `depends_on` block entirely (the bot already handles missing backends gracefully via C2's fallback chain).

**Step 7: Verify**

Run: `docker compose -f docker/docker-compose.yml config --profiles neo4j --profiles postgres --profiles qdrant --profiles redis`
Expected: Valid YAML with all 5 services shown.

Run: `docker compose -f docker/docker-compose.yml config`
Expected: Valid YAML with only `nexus-bot` shown (no profiled services active).

**Step 8: Commit**

```bash
git add docker/docker-compose.yml
git commit -m "infra: add Neo4j + Postgres with Docker Compose profiles

All infrastructure services (Qdrant, Redis, Neo4j, Postgres) now use
profiles so users can skip any they already run externally. C2 env
vars are wired with sensible defaults pointing to Docker services."
```

---

### Task 2: C2 OpenRouter Embeddings

**Files:**
- Modify: `src/continuity_core/memory/embeddings.py`
- Modify: `src/continuity_core/config.py`
- Test: `tests/test_c2_embeddings.py`

**Context:** C2's embedder defaults to `hash` (SHA256), which gives near-random similarity scores. We add an `OpenRouterEmbedder` that calls OpenRouter's embeddings API via `requests.post()`. C2 code is synchronous (not async), so we use the `requests` library. The existing `OllamaEmbedder` shows the pattern.

**Step 1: Write the failing test**

Create `tests/test_c2_embeddings.py`:

```python
"""Tests for C2 embedding backends."""

from unittest.mock import MagicMock, patch

import pytest


def test_openrouter_embedder_returns_vector():
    """OpenRouterEmbedder calls the API and returns a float list."""
    from continuity_core.memory.embeddings import OpenRouterEmbedder

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [{"embedding": [0.1, 0.2, 0.3]}],
    }
    mock_response.raise_for_status = MagicMock()

    with patch("continuity_core.memory.embeddings.requests.post", return_value=mock_response) as mock_post:
        embedder = OpenRouterEmbedder(api_key="test-key", model="test/model")
        result = embedder.embed("hello world")

    assert result == [0.1, 0.2, 0.3]
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "Authorization" in call_kwargs[1]["headers"]
    assert call_kwargs[1]["headers"]["Authorization"] == "Bearer test-key"


def test_build_embedder_openrouter():
    """build_embedder returns OpenRouterEmbedder for 'openrouter' backend."""
    from continuity_core.config import C2Config
    from continuity_core.memory.embeddings import OpenRouterEmbedder, build_embedder

    config = C2Config(
        embedding_backend="openrouter",
        openrouter_api_key="test-key",
        openrouter_embed_model="test/model",
    )
    embedder = build_embedder(config)
    assert isinstance(embedder, OpenRouterEmbedder)


def test_build_embedder_hash_fallback():
    """build_embedder returns HashEmbedder for unknown backend."""
    from continuity_core.config import C2Config
    from continuity_core.memory.embeddings import HashEmbedder, build_embedder

    config = C2Config(embedding_backend="unknown")
    embedder = build_embedder(config)
    assert isinstance(embedder, HashEmbedder)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_c2_embeddings.py -v`
Expected: FAIL — `OpenRouterEmbedder` doesn't exist yet, `C2Config` missing new fields.

**Step 3: Add config fields**

In `src/continuity_core/config.py`, add two new fields to `C2Config`:

```python
    openrouter_api_key: str = os.getenv("C2_OPENROUTER_API_KEY", "")
    openrouter_embed_model: str = os.getenv("C2_OPENROUTER_EMBED_MODEL", "qwen/qwen3-embedding-8b")
```

**Step 4: Add OpenRouterEmbedder class**

In `src/continuity_core/memory/embeddings.py`, add after `SentenceTransformerEmbedder`:

```python
class OpenRouterEmbedder:
    """Embedding via OpenRouter's /api/v1/embeddings endpoint."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._url = "https://openrouter.ai/api/v1/embeddings"

    def embed(self, text: str) -> List[float]:
        resp = requests.post(
            self._url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"input": text, "model": self._model},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]
```

**Step 5: Update build_embedder**

In `build_embedder()`, add the `openrouter` case before the `ollama` case:

```python
def build_embedder(config: C2Config) -> Embedder:
    backend = config.embedding_backend.lower()
    if backend == "openrouter":
        return OpenRouterEmbedder(config.openrouter_api_key, config.openrouter_embed_model)
    if backend == "ollama":
        return OllamaEmbedder(config.ollama_base_url, config.ollama_embed_model)
    if backend in {"sbert", "sentence-transformers"}:
        return SentenceTransformerEmbedder(config.embedding_model)
    return HashEmbedder()
```

**Step 6: Run tests**

Run: `pytest tests/test_c2_embeddings.py -v`
Expected: All 3 PASS.

**Step 7: Commit**

```bash
git add src/continuity_core/memory/embeddings.py src/continuity_core/config.py tests/test_c2_embeddings.py
git commit -m "feat: add OpenRouter cloud embeddings to C2

Adds OpenRouterEmbedder that calls /api/v1/embeddings via requests.
Replaces hash embeddings as the default C2 backend for production.
Includes config fields for API key and model selection."
```

---

### Task 3: C2 MCP Status Tool

**Files:**
- Create: `src/continuity_core/mcp/tools/status.py`
- Modify: `src/continuity_core/mcp/tools/__init__.py`
- Modify: `src/continuity_core/mcp/server.py`
- Test: `tests/test_c2_status.py`

**Context:** There's no way to see which C2 backends are connected. We add a `c2.status` MCP tool that queries the TieredMemorySystem for backend health. All existing tools follow the same pattern: a function in `mcp/tools/` that calls `get_memory_system()` and returns a dict.

**Step 1: Write the failing test**

Create `tests/test_c2_status.py`:

```python
"""Tests for the C2 status tool."""

from unittest.mock import MagicMock, patch


def test_status_returns_backend_health():
    """c2.status returns a dict with backend connection status."""
    mock_mem = MagicMock()
    mock_mem.neo4j = None
    mock_mem.qdrant = MagicMock()
    mock_mem.redis = None
    mock_mem._fallback = None
    mock_mem.event_log = MagicMock()
    mock_mem.event_log._store = MagicMock()
    mock_mem.event_log._store._events = [1, 2, 3]
    mock_mem.embedder = MagicMock()
    mock_mem.embedder.__class__.__name__ = "HashEmbedder"
    mock_mem.get_mra_signals.return_value = None

    with patch("continuity_core.mcp.tools.status.get_memory_system", return_value=mock_mem):
        from continuity_core.mcp.tools.status import status
        result = status({})

    assert result["neo4j"] == "offline"
    assert result["qdrant"] == "connected"
    assert result["redis"] == "offline"
    assert result["embedding_backend"] == "HashEmbedder"
    assert "stress_level" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_c2_status.py -v`
Expected: FAIL — module `status` does not exist.

**Step 3: Create the status tool**

Create `src/continuity_core/mcp/tools/status.py`:

```python
"""c2.status — Return backend health and system metrics."""

from __future__ import annotations

from typing import Any, Dict

from continuity_core.services.runtime import get_memory_system


def status(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Return C2 backend health, item counts, and MRA stress level."""
    mem = get_memory_system()

    # Backend connection status
    neo4j_status = "offline"
    neo4j_nodes = 0
    if mem.neo4j is not None:
        neo4j_status = "connected"
        try:
            with mem.neo4j._driver.session() as session:
                result = session.run("MATCH (n:PKMNode) RETURN count(n) AS cnt")
                neo4j_nodes = result.single()["cnt"]
        except Exception:
            neo4j_status = "error"

    qdrant_status = "connected" if mem.qdrant is not None else "offline"
    redis_status = "connected" if mem.redis is not None else "offline"

    # Event log info
    event_count = 0
    event_backend = "in-memory"
    try:
        from continuity_core.storage.postgres import PostgresEventStore
        if isinstance(mem.event_log._store, PostgresEventStore):
            event_backend = "postgres"
        events = mem.event_log.tail(n=1000)
        event_count = len(events)
    except Exception:
        pass

    # Fallback memory count
    fallback_count = 0
    if mem._fallback is not None:
        fallback_count = len(mem._fallback._items)

    # MRA stress
    mra = mem.get_mra_signals()
    stress_level = 0.0
    if mra is not None and mra.last_stress is not None:
        stress_level = mra.last_stress.s_omega

    return {
        "neo4j": neo4j_status,
        "neo4j_nodes": neo4j_nodes,
        "qdrant": qdrant_status,
        "redis": redis_status,
        "event_backend": event_backend,
        "event_count": event_count,
        "fallback_memory_count": fallback_count,
        "embedding_backend": type(mem.embedder).__name__,
        "stress_level": stress_level,
    }
```

**Step 4: Register the tool**

In `src/continuity_core/mcp/tools/__init__.py`, add:

```python
from .status import status
```

And update `__all__` to include `"status"`.

In `src/continuity_core/mcp/server.py`, import `status` from the tools module and register it in `_build_registry()`:

```python
from continuity_core.mcp.tools import build_context, curiosity, introspect, write_event, status
from continuity_core.mcp.tools.maintenance import maintenance
```

Add to the registry:

```python
    registry.register(
        Tool(
            name="c2.status",
            description="Return C2 backend health, item counts, and MRA stress level.",
            input_schema={"type": "object", "properties": {}},
            handler=status,
        )
    )
```

**Step 5: Run tests**

Run: `pytest tests/test_c2_status.py -v`
Expected: PASS.

**Step 6: Commit**

```bash
git add src/continuity_core/mcp/tools/status.py src/continuity_core/mcp/tools/__init__.py src/continuity_core/mcp/server.py tests/test_c2_status.py
git commit -m "feat: add c2.status MCP tool for backend health inspection"
```

---

### Task 4: C2 MCP Events Tool

**Files:**
- Create: `src/continuity_core/mcp/tools/events_read.py`
- Modify: `src/continuity_core/mcp/tools/__init__.py`
- Modify: `src/continuity_core/mcp/server.py`
- Test: `tests/test_c2_events_read.py`

**Context:** There's a `c2.write_event` tool but no way to read events back. We add `c2.events` that returns the last N events from the event log.

**Step 1: Write the failing test**

Create `tests/test_c2_events_read.py`:

```python
"""Tests for the C2 events read tool."""

from unittest.mock import MagicMock, patch

from continuity_core.event_log import Event


def test_events_returns_recent_events():
    """c2.events returns recent events as a list of dicts."""
    mock_events = [
        Event(timestamp=1000.0, actor="human", intent="message", input="hello", output="", tags=["human"]),
        Event(timestamp=1001.0, actor="model", intent="response", input="", output="hi there", tags=["swarm"]),
    ]
    mock_mem = MagicMock()
    mock_mem.event_log.tail.return_value = mock_events

    with patch("continuity_core.mcp.tools.events_read.get_memory_system", return_value=mock_mem):
        from continuity_core.mcp.tools.events_read import read_events
        result = read_events({"limit": 10})

    assert len(result["events"]) == 2
    assert result["events"][0]["actor"] == "human"
    assert result["events"][1]["output"] == "hi there"
    mock_mem.event_log.tail.assert_called_once_with(n=10)


def test_events_default_limit():
    """c2.events defaults to 10 events when no limit specified."""
    mock_mem = MagicMock()
    mock_mem.event_log.tail.return_value = []

    with patch("continuity_core.mcp.tools.events_read.get_memory_system", return_value=mock_mem):
        from continuity_core.mcp.tools.events_read import read_events
        read_events({})

    mock_mem.event_log.tail.assert_called_once_with(n=10)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_c2_events_read.py -v`
Expected: FAIL — module doesn't exist.

**Step 3: Create the events_read tool**

Create `src/continuity_core/mcp/tools/events_read.py`:

```python
"""c2.events — Read recent events from the event log."""

from __future__ import annotations

from typing import Any, Dict

from continuity_core.services.runtime import get_memory_system


def read_events(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Return the most recent events from the C2 event log."""
    limit = int(arguments.get("limit", 10))
    limit = max(1, min(limit, 50))

    mem = get_memory_system()
    events = mem.event_log.tail(n=limit)

    return {
        "count": len(events),
        "events": [
            {
                "timestamp": e.timestamp,
                "actor": e.actor,
                "intent": e.intent,
                "input": e.input,
                "output": e.output,
                "tags": e.tags,
            }
            for e in events
        ],
    }
```

**Step 4: Register the tool**

In `src/continuity_core/mcp/tools/__init__.py`, add:

```python
from .events_read import read_events
```

Update `__all__` to include `"read_events"`.

In `src/continuity_core/mcp/server.py`, import `read_events` and register:

```python
from continuity_core.mcp.tools import build_context, curiosity, introspect, write_event, status, read_events
```

```python
    registry.register(
        Tool(
            name="c2.events",
            description="Return the most recent events from the C2 event log.",
            input_schema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Number of events (1-50, default 10)"},
                },
            },
            handler=read_events,
        )
    )
```

**Step 5: Run tests**

Run: `pytest tests/test_c2_events_read.py -v`
Expected: All PASS.

**Step 6: Commit**

```bash
git add src/continuity_core/mcp/tools/events_read.py src/continuity_core/mcp/tools/__init__.py src/continuity_core/mcp/server.py tests/test_c2_events_read.py
git commit -m "feat: add c2.events MCP tool for reading recent events"
```

---

### Task 5: C2Client Wrappers — status() and events()

**Files:**
- Modify: `src/nexus/integrations/c2_client.py`
- Test: `tests/test_c2_client.py`

**Context:** The `C2Client` in the bot wraps each MCP tool as an async method. We need `status()` and `events()` wrappers for the new tools.

**Step 1: Write the failing test**

Create `tests/test_c2_client.py`:

```python
"""Tests for C2Client tool wrappers."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from nexus.integrations.c2_client import C2Client


@pytest.mark.asyncio
async def test_status_calls_correct_tool():
    """C2Client.status() calls c2.status tool."""
    client = C2Client()
    client._initialized = True
    client._process = MagicMock()
    client._process.returncode = None

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {"neo4j": "connected"}
        result = await client.status()

    mock_call.assert_called_once_with("c2.status")
    assert result == {"neo4j": "connected"}


@pytest.mark.asyncio
async def test_events_calls_correct_tool():
    """C2Client.events() calls c2.events tool with limit."""
    client = C2Client()
    client._initialized = True
    client._process = MagicMock()
    client._process.returncode = None

    with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {"events": [], "count": 0}
        result = await client.events(limit=5)

    mock_call.assert_called_once_with("c2.events", {"limit": 5})
    assert result == {"events": [], "count": 0}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_c2_client.py -v`
Expected: FAIL — `status()` and `events()` methods don't exist.

**Step 3: Add the wrappers**

In `src/nexus/integrations/c2_client.py`, add after the `maintenance()` method:

```python
    async def status(self) -> dict[str, Any] | None:
        """Query C2 backend health and system metrics."""
        return await self._call_tool("c2.status")

    async def events(self, limit: int = 10) -> dict[str, Any] | None:
        """Read recent events from the C2 event log."""
        return await self._call_tool("c2.events", {"limit": limit})
```

**Step 4: Run tests**

Run: `pytest tests/test_c2_client.py -v`
Expected: All PASS.

**Step 5: Commit**

```bash
git add src/nexus/integrations/c2_client.py tests/test_c2_client.py
git commit -m "feat: add status() and events() wrappers to C2Client"
```

---

### Task 6: Discord Commands — !c2status, !c2events

**Files:**
- Modify: `src/nexus/commands/admin.py`

**Context:** Add two new Discord commands to the `AdminCommands` cog. They call the C2Client wrappers and format results as Discord embeds. Follow the pattern of the existing `!curiosity` command.

**Step 1: Add !c2status command**

In `src/nexus/commands/admin.py`, add after the `curiosity_scan` command:

```python
    @commands.command(name="c2status")
    async def c2_status(self, ctx: commands.Context) -> None:
        """Show Continuity Core backend health and system metrics."""
        if not self.bot.c2.is_running:
            await ctx.send("C2 is not running.")
            return

        result = await self.bot.c2.status()
        if result is None:
            await ctx.send("C2 status unavailable.")
            return

        embed = discord.Embed(title="Continuity Core Status", color=0x3498DB)
        embed.add_field(name="Neo4j", value=f"{result.get('neo4j', 'unknown')} ({result.get('neo4j_nodes', 0)} nodes)", inline=True)
        embed.add_field(name="Qdrant (C2)", value=result.get("qdrant", "unknown"), inline=True)
        embed.add_field(name="Redis", value=result.get("redis", "unknown"), inline=True)
        embed.add_field(name="Event Log", value=f"{result.get('event_backend', 'unknown')} ({result.get('event_count', 0)} events)", inline=True)
        embed.add_field(name="Embeddings", value=result.get("embedding_backend", "unknown"), inline=True)
        embed.add_field(name="MRA Stress", value=f"{result.get('stress_level', 0):.3f}", inline=True)
        if result.get("fallback_memory_count", 0) > 0:
            embed.add_field(name="Fallback Memory", value=f"{result['fallback_memory_count']} items (in-memory)", inline=True)

        await ctx.send(embed=embed)
```

**Step 2: Add !c2events command**

Add after `!c2status`:

```python
    @commands.command(name="c2events")
    async def c2_events(self, ctx: commands.Context, limit: int = 10) -> None:
        """Show recent C2 events. Usage: !c2events [count]"""
        if not self.bot.c2.is_running:
            await ctx.send("C2 is not running.")
            return

        limit = max(1, min(limit, 20))
        result = await self.bot.c2.events(limit=limit)
        if result is None or not result.get("events"):
            await ctx.send("No C2 events found.")
            return

        lines = []
        for evt in result["events"]:
            from datetime import datetime, timezone
            ts = datetime.fromtimestamp(evt["timestamp"], tz=timezone.utc).strftime("%H:%M:%S")
            actor = evt.get("actor", "?")
            intent = evt.get("intent", "?")
            output = evt.get("output", "")[:80]
            lines.append(f"`{ts}` **{actor}** [{intent}] {output}")

        embed = discord.Embed(
            title=f"C2 Events (last {result.get('count', 0)})",
            description="\n".join(lines),
            color=0x555555,
        )
        await ctx.send(embed=embed)
```

**Step 3: Verify manually**

After deployment, run `!c2status` and `!c2events` in Discord.
Expected: Embeds showing backend health and event history.

**Step 4: Commit**

```bash
git add src/nexus/commands/admin.py
git commit -m "feat: add !c2status and !c2events Discord commands"
```

---

### Task 7: Curiosity-Driven Swarm Discussion

**Files:**
- Modify: `src/nexus/orchestrator/loop.py`
- Test: `tests/test_curiosity_discussion.py`

**Context:** This is the core feature. When the night cycle finds actionable curiosity signals, trigger a Tier 1 group discussion. The flow mirrors how human messages work: pick a primary model, get its response, run crosstalk reactions. The key new method is `_trigger_curiosity_discussion()` on `OrchestratorLoop`.

**Step 1: Write the test**

Create `tests/test_curiosity_discussion.py`:

```python
"""Tests for curiosity-driven swarm discussion."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_bot_mock():
    """Build a mock bot with the minimum attributes needed."""
    bot = MagicMock()
    bot.swarm_models = {"model/a": MagicMock(), "model/b": MagicMock()}
    bot.openrouter = MagicMock()
    bot.conversation = MagicMock()
    bot.conversation.add_message = AsyncMock()
    bot.conversation.build_messages_for_model = MagicMock(return_value=[{"role": "user", "content": "test"}])
    bot.memory_store = MagicMock()
    bot.memory_store.is_connected = True
    bot.embeddings = MagicMock()
    bot.embeddings.embed_one = AsyncMock(return_value=[0.1] * 10)
    bot.memory_store.store = AsyncMock(return_value="mem-id")
    bot.crosstalk = MagicMock()
    bot.crosstalk.is_enabled = False
    bot.router = MagicMock()
    bot.router.nexus = MagicMock()
    bot.router.nexus.send = AsyncMock(return_value=MagicMock())
    bot.router.memory = MagicMock()
    bot.router.memory.send = AsyncMock()
    bot.c2 = MagicMock()
    bot.c2.is_running = True
    bot._system_prompts = {"model/a": "You are A.", "model/b": "You are B."}
    bot.get_system_prompt = MagicMock(return_value="You are a model.")
    bot._spawn = MagicMock(side_effect=lambda coro: MagicMock())
    return bot


@pytest.mark.asyncio
async def test_trigger_curiosity_discussion_posts_to_nexus():
    """_trigger_curiosity_discussion posts model response to #nexus."""
    from nexus.orchestrator.loop import OrchestratorLoop

    bot = _make_bot_mock()
    response_mock = MagicMock()
    response_mock.content = "Interesting tension between X and Y."
    bot.openrouter.chat = AsyncMock(return_value=response_mock)

    loop = OrchestratorLoop(bot, interval=3600)

    curiosity_result = {
        "stress_level": 0.45,
        "contradictions": [{"s1": "A is true", "s2": "A is false", "score": 0.8}],
        "deep_tensions": [],
        "bridging_questions": ["What connects A and B?"],
        "suggested_action": "resolve_contradiction",
    }

    with patch("nexus.orchestrator.loop.MessageFormatter") as mock_fmt:
        mock_fmt.format_response.return_value = MagicMock()
        mock_fmt.format_memory_log.return_value = MagicMock()
        await loop._trigger_curiosity_discussion(curiosity_result)

    # Verify a model was called
    bot.openrouter.chat.assert_called_once()
    # Verify response posted to #nexus
    bot.router.nexus.send.assert_called()
    # Verify summary posted to #memory
    bot.router.memory.send.assert_called()


@pytest.mark.asyncio
async def test_night_cycle_triggers_discussion_on_high_stress():
    """_run_night_cycle triggers discussion when contradictions found."""
    from nexus.orchestrator.loop import OrchestratorLoop

    bot = _make_bot_mock()
    bot.c2.maintenance = AsyncMock(return_value={
        "stress_after": 0.35,
        "contradictions_found": 2,
        "deep_tensions_found": 0,
        "voids_found": 0,
    })
    bot.c2.curiosity = AsyncMock(return_value={
        "stress_level": 0.35,
        "contradictions": [{"s1": "X", "s2": "Y", "score": 0.7}],
        "deep_tensions": [],
        "bridging_questions": [],
        "suggested_action": "resolve_contradiction",
    })

    loop = OrchestratorLoop(bot, interval=3600)

    with patch.object(loop, "_trigger_curiosity_discussion", new_callable=AsyncMock) as mock_discuss:
        with patch.object(loop, "_post_curiosity_findings", new_callable=AsyncMock):
            with patch.object(loop, "_log_to_c2", new_callable=AsyncMock):
                await loop._run_night_cycle()

    mock_discuss.assert_called_once()


@pytest.mark.asyncio
async def test_night_cycle_skips_discussion_when_no_signals():
    """_run_night_cycle does NOT trigger discussion when nothing found."""
    from nexus.orchestrator.loop import OrchestratorLoop

    bot = _make_bot_mock()
    bot.c2.maintenance = AsyncMock(return_value={
        "stress_after": 0.05,
        "contradictions_found": 0,
        "deep_tensions_found": 0,
        "voids_found": 0,
    })

    loop = OrchestratorLoop(bot, interval=3600)

    with patch.object(loop, "_trigger_curiosity_discussion", new_callable=AsyncMock) as mock_discuss:
        with patch.object(loop, "_post_curiosity_findings", new_callable=AsyncMock):
            with patch.object(loop, "_log_to_c2", new_callable=AsyncMock):
                await loop._run_night_cycle()

    mock_discuss.assert_not_called()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_curiosity_discussion.py -v`
Expected: FAIL — `_trigger_curiosity_discussion` doesn't exist yet.

**Step 3: Add `_trigger_curiosity_discussion()` method**

In `src/nexus/orchestrator/loop.py`, add this new method after `_post_curiosity_findings()`:

```python
    async def _trigger_curiosity_discussion(
        self, curiosity_result: dict[str, Any],
    ) -> None:
        """Trigger a Tier 1 swarm discussion about curiosity findings.

        Picks a random model, sends the curiosity prompt, posts the response
        to #nexus, runs crosstalk reactions, and posts a summary to #memory.
        """
        import random
        from nexus.channels.formatter import MessageFormatter

        model_ids = list(self.bot.swarm_models.keys())
        if not model_ids:
            return

        # Build the curiosity discussion prompt
        prompt_parts = [
            "The Continuity Core has detected epistemic tensions in our knowledge base "
            "that need investigation.",
            "",
        ]

        stress = curiosity_result.get("stress_level", 0)
        prompt_parts.append(f"Epistemic stress level: {stress:.3f}")

        contradictions = curiosity_result.get("contradictions", [])
        if contradictions:
            prompt_parts.append(f"\nContradictions ({len(contradictions)}):")
            for c in contradictions[:5]:
                prompt_parts.append(f"  - \"{c.get('s1', '')}\" vs \"{c.get('s2', '')}\"")

        tensions = curiosity_result.get("deep_tensions", [])
        if tensions:
            prompt_parts.append(f"\nDeep tensions ({len(tensions)}):")
            for t in tensions[:5]:
                prompt_parts.append(f"  - \"{t.get('s1', '')}\" vs \"{t.get('s2', '')}\"")

        questions = curiosity_result.get("bridging_questions", [])
        if questions:
            prompt_parts.append(f"\nBridging questions:")
            for q in questions[:5]:
                prompt_parts.append(f"  - {q}")

        suggested = curiosity_result.get("suggested_action", "")
        if suggested:
            prompt_parts.append(f"\nSuggested focus: {suggested}")

        prompt_parts.append(
            "\nDiscuss these findings. What do they mean? What should we "
            "investigate or resolve? Be specific and actionable."
        )

        curiosity_prompt = "\n".join(prompt_parts)

        # Pick a random primary responder
        primary_model = random.choice(model_ids)
        system_prompt = self.bot.get_system_prompt(primary_model)

        try:
            messages = self.bot.conversation.build_messages_for_model(
                primary_model, system_prompt, limit=10,
            )
            # Inject the curiosity prompt as the latest user message
            messages.append({"role": "user", "content": curiosity_prompt})

            response = await self.bot.openrouter.chat(
                model=primary_model,
                messages=messages,
            )

            # Record in conversation history
            await self.bot.conversation.add_message(
                primary_model, response.content,
            )

            # Post to #nexus
            embed = MessageFormatter.format_response(primary_model, response.content)
            last_msg = await self.bot.router.nexus.send(embed=embed)

            # Store in memory
            if self.bot.memory_store.is_connected:
                self.bot._spawn(self._store_discussion_memory(
                    response.content, primary_model,
                ))

            # Log to C2
            await self._log_to_c2(
                actor=primary_model,
                intent="curiosity_discussion",
                out=response.content[:500],
                tags=["curiosity", "autonomous", "swarm"],
            )

            # Run crosstalk reactions if enabled
            if self.bot.crosstalk.is_enabled and last_msg is not None:
                await self._run_curiosity_reactions(
                    primary_model, model_ids, last_msg,
                )

            # Post summary to #memory
            summary_embed = MessageFormatter.format_memory_log(
                "curiosity_discussion",
                f"Triggered by stress={stress:.3f}, "
                f"{len(contradictions)} contradiction(s), "
                f"{len(tensions)} tension(s). "
                f"Primary: {primary_model}.",
            )
            await self.bot.router.memory.send(embed=summary_embed)

        except Exception:
            log.error("Curiosity discussion failed.", exc_info=True)
```

**Step 4: Add helper methods**

Add `_run_curiosity_reactions` and `_store_discussion_memory` after the method above:

```python
    async def _run_curiosity_reactions(
        self,
        primary_model: str,
        model_ids: list[str],
        last_msg: Any,
    ) -> None:
        """Run crosstalk reactions for the curiosity discussion."""
        import asyncio
        from nexus.channels.formatter import MessageFormatter
        from nexus.swarm.crosstalk import CrosstalkManager

        reaction_order = self.bot.crosstalk.build_reaction_order(
            primary_model, model_ids,
        )
        reaction_suffix = CrosstalkManager.get_reaction_suffix()
        reactions_posted = 0

        for reactor_id in reaction_order:
            if reactions_posted >= 2:
                break
            try:
                reactor_prompt = self.bot.get_system_prompt(reactor_id) + reaction_suffix
                reactor_messages = self.bot.conversation.build_messages_for_model(
                    reactor_id, reactor_prompt, limit=10,
                )
                reaction = await asyncio.wait_for(
                    self.bot.openrouter.chat(
                        model=reactor_id, messages=reactor_messages,
                    ),
                    timeout=30.0,
                )
                if CrosstalkManager.is_pass(reaction.content):
                    continue

                await self.bot.conversation.add_message(reactor_id, reaction.content)
                embed = MessageFormatter.format_response(reactor_id, reaction.content)
                last_msg = await last_msg.reply(embed=embed, mention_author=False)
                reactions_posted += 1

                if self.bot.memory_store.is_connected:
                    self.bot._spawn(self._store_discussion_memory(
                        reaction.content, reactor_id,
                    ))

                await self._log_to_c2(
                    actor=reactor_id,
                    intent="curiosity_discussion",
                    out=reaction.content[:500],
                    tags=["curiosity", "autonomous", "swarm"],
                )

            except asyncio.TimeoutError:
                log.warning("Curiosity reaction from %s timed out.", reactor_id)
            except Exception:
                log.error("Curiosity reaction from %s failed.", reactor_id, exc_info=True)

    async def _store_discussion_memory(self, content: str, source: str) -> None:
        """Store a curiosity discussion response in vector memory."""
        try:
            vector = await self.bot.embeddings.embed_one(content)
            await self.bot.memory_store.store(
                content=content,
                vector=vector,
                source=source,
                channel="nexus",
                metadata={"type": "curiosity_discussion"},
            )
        except Exception:
            log.warning("Failed to store curiosity discussion in memory.", exc_info=True)
```

**Step 5: Update `_run_night_cycle()` to trigger discussions**

In the existing `_run_night_cycle()` method, add the discussion trigger after posting the curiosity embed. Replace the current method body with:

```python
    async def _run_night_cycle(self) -> None:
        """Run C2 night-cycle maintenance if C2 is available."""
        c2 = getattr(self.bot, "c2", None)
        if c2 is None or not c2.is_running:
            return

        try:
            result = await c2.maintenance()
            if result is None:
                return

            log.info(
                "Night cycle complete: stress=%.3f, contradictions=%d, voids=%d.",
                result.get("stress_after", 0),
                result.get("contradictions_found", 0),
                result.get("voids_found", 0),
            )

            # Post curiosity findings to #nexus if there are contradictions.
            if result.get("contradictions_found", 0) > 0:
                await self._post_curiosity_findings(result)

            # Trigger swarm discussion if curiosity signals are actionable.
            should_discuss = (
                result.get("stress_after", 0) > 0.2
                or result.get("contradictions_found", 0) > 0
                or result.get("voids_found", 0) > 0
            )
            if should_discuss:
                curiosity_signals = await c2.curiosity()
                if curiosity_signals is not None:
                    await self._trigger_curiosity_discussion(curiosity_signals)

        except Exception:
            log.warning("Night cycle maintenance failed.", exc_info=True)
```

**Step 6: Run tests**

Run: `pytest tests/test_curiosity_discussion.py -v`
Expected: All 3 PASS.

Run: `pytest tests/ -v`
Expected: All existing + new tests pass.

**Step 7: Commit**

```bash
git add src/nexus/orchestrator/loop.py tests/test_curiosity_discussion.py
git commit -m "feat: curiosity-driven swarm discussion

When the night cycle finds epistemic tensions (stress > 0.2,
contradictions, or voids), the orchestrator now triggers a Tier 1
group discussion. A random model analyzes the findings, others
react via crosstalk, and a summary is posted to #memory."
```

---

### Task 8: !discuss Command

**Files:**
- Modify: `src/nexus/commands/admin.py`

**Context:** Add `!discuss` command that lets the user manually trigger a curiosity discussion on demand. Calls `c2.curiosity()` and, if there are actionable signals, runs `_trigger_curiosity_discussion()`.

**Step 1: Add the command**

In `src/nexus/commands/admin.py`, add after `c2_events`:

```python
    @commands.command(name="discuss")
    async def discuss_curiosity(self, ctx: commands.Context) -> None:
        """Trigger the swarm to discuss C2 curiosity findings."""
        if not self.bot.c2.is_running:
            await ctx.send("C2 is not running. Cannot trigger discussion.")
            return

        await ctx.send("Querying C2 for epistemic tensions...")

        result = await self.bot.c2.curiosity()
        if result is None:
            await ctx.send("C2 returned no curiosity data.")
            return

        has_signals = (
            result.get("stress_level", 0) > 0.1
            or result.get("contradictions")
            or result.get("deep_tensions")
            or result.get("bridging_questions")
        )

        if not has_signals:
            await ctx.send("No epistemic tensions detected — nothing to discuss.")
            return

        await ctx.send(
            f"Found signals (stress={result.get('stress_level', 0):.3f}, "
            f"{len(result.get('contradictions', []))} contradiction(s), "
            f"{len(result.get('deep_tensions', []))} tension(s)). "
            f"Triggering swarm discussion..."
        )

        await self.bot.orchestrator._trigger_curiosity_discussion(result)
```

**Step 2: Commit**

```bash
git add src/nexus/commands/admin.py
git commit -m "feat: add !discuss command for on-demand curiosity discussions"
```

---

### Task 9: Update CLAUDE.md — Monorepo Decision

**Files:**
- Modify: `CLAUDE.md`

**Context:** The user confirmed continuity_core is part of the agent-nexus monorepo, not a separate upstream dependency. Update the guidance.

**Step 1: Update CLAUDE.md**

Replace the line about continuity_core in the Key Patterns section:

Old:
```
- `continuity_core/` is an upstream dependency copied as-is. Edit it in its own repo and re-copy, don't modify in-place here.
```

New:
```
- `continuity_core/` is part of this monorepo. It was originally a separate project but is now maintained here. Edit freely.
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md — continuity_core is part of monorepo"
```

---

### Task 10: Run Full Test Suite + Verify

**Files:**
- None (verification only)

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass (existing 31 + new tests from this plan).

**Step 2: Lint check**

Run: `ruff check src/`
Expected: No errors. Fix any that appear.

**Step 3: Verify Docker compose config**

Run: `docker compose -f docker/docker-compose.yml config`
Expected: Valid YAML, all C2 env vars present on nexus-bot.

Run: `docker compose -f docker/docker-compose.yml --profile neo4j --profile postgres --profile qdrant --profile redis config`
Expected: All 5 services shown.

**Step 4: Commit any fixes**

If any lint or test fixes were needed, commit them.
