# Monorepo Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the C2 subprocess with a direct Python facade, unify config/infrastructure, and port the synthesis TDD engine into agent-nexus.

**Architecture:** C2Engine wraps `TieredMemorySystem` with a bridged `C2Config` built from `NexusSettings`. All sync C2 calls run in `asyncio.to_thread()`. Synthesis TDD engine is adapted with a `NexusLLMAdapter` wrapping `OpenRouterClient`. Postgres added to Docker stack.

**Tech Stack:** Python 3.12, asyncio, pydantic-settings, asyncpg, qdrant-client, redis, discord.py

**Design doc:** `docs/plans/2026-02-22-monorepo-integration-design.md`

---

## Task 1: Add Postgres to Docker and NexusSettings

**Files:**
- Modify: `docker/docker-compose.yml:104-121`
- Modify: `src/nexus/config.py:153-165`
- Test: `tests/test_config.py`

**Context:** Postgres already exists in docker-compose.yml at line 104 with profile `postgres`, user `c2`, db `continuity_core`. We need to change it to use the unified `nexus` user/db and remove the profile gate so it starts by default. We also need to add `POSTGRES_URL` to NexusSettings.

**Step 1: Write a failing test for POSTGRES_URL**

```python
# tests/test_config.py — append this test

def test_postgres_url_default(monkeypatch):
    """NexusSettings should include POSTGRES_URL with a sensible default."""
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from nexus.config import NexusSettings
    settings = NexusSettings()
    assert hasattr(settings, "POSTGRES_URL")
    assert "postgresql" in settings.POSTGRES_URL
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_postgres_url_default -v`
Expected: FAIL with `AttributeError: 'NexusSettings' has no attribute 'POSTGRES_URL'`

**Step 3: Add POSTGRES_URL to NexusSettings**

Add after the `REDIS_URL` field (around line 165 in `src/nexus/config.py`):

```python
    POSTGRES_URL: str = Field(
        default="postgresql://nexus:nexus@nexus-postgres:5432/nexus",
        description="PostgreSQL connection URL for C2 event log and synthesis.",
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::test_postgres_url_default -v`
Expected: PASS

**Step 5: Update docker-compose.yml Postgres service**

Change `docker/docker-compose.yml` lines 104-121:

```yaml
  nexus-postgres:
    image: postgres:16-alpine
    container_name: nexus-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: nexus
      POSTGRES_PASSWORD: nexus
      POSTGRES_DB: nexus
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U nexus -d nexus"]
      interval: 10s
      timeout: 5s
      retries: 3
    networks:
      - nexus-net
```

Key changes: removed `profiles: [postgres]` (always starts), changed user/password to `nexus`, changed db to `nexus`.

**Step 6: Remove the C2_OPENROUTER_API_KEY passthrough from nexus-bot environment**

In `docker/docker-compose.yml`, delete line 27:
```yaml
      C2_OPENROUTER_API_KEY: "${OPENROUTER_API_KEY}"
```

The C2Engine will read from NexusSettings directly — no separate env var needed.

**Step 7: Commit**

```bash
git add src/nexus/config.py docker/docker-compose.yml tests/test_config.py
git commit -m "feat: add Postgres to default Docker stack and NexusSettings"
```

---

## Task 2: Add C2 algorithm tuning fields to NexusSettings

**Files:**
- Modify: `src/nexus/config.py:165-200`
- Test: `tests/test_config.py`

**Context:** `src/continuity_core/config.py` has algorithm tuning params (`C2_TOKEN_BUDGET`, `C2_EPSILON`, `C2_LAMBDA`, decay params). We migrate these to NexusSettings so the C2Engine can build a `C2Config` from them.

**Step 1: Write failing tests for C2 fields**

```python
# tests/test_config.py — append

def test_c2_tuning_fields_exist(monkeypatch):
    """NexusSettings should include C2 algorithm tuning fields."""
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from nexus.config import NexusSettings
    settings = NexusSettings()
    assert settings.C2_TOKEN_BUDGET == 2048
    assert settings.C2_EPSILON == 0.05
    assert settings.C2_LAMBDA == 0.001
    assert settings.C2_EDGE_HALF_LIFE_DAYS == 7.0
    assert settings.C2_DECAY_RATE == 0.95
    assert settings.C2_RECENCY_HALF_LIFE_DAYS == 14.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_c2_tuning_fields_exist -v`
Expected: FAIL

**Step 3: Add C2 fields to NexusSettings**

Add a new section after the `POSTGRES_URL` field in `src/nexus/config.py`:

```python
    # ------------------------------------------------------------------
    # Continuity Core (C2) algorithm tuning
    # ------------------------------------------------------------------
    C2_TOKEN_BUDGET: int = Field(
        default=2048,
        ge=256,
        description="Token budget for C2 context composition.",
    )
    C2_EPSILON: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Exploration fraction in C2 context selection.",
    )
    C2_LAMBDA: float = Field(
        default=0.001,
        ge=0.0,
        description="Per-token cost penalty in C2 context selection.",
    )
    C2_EDGE_HALF_LIFE_DAYS: float = Field(
        default=7.0,
        ge=0.1,
        description="Half-life in days for C2 graph edge decay.",
    )
    C2_DECAY_RATE: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Salience decay rate for C2 memory items.",
    )
    C2_RECENCY_HALF_LIFE_DAYS: float = Field(
        default=14.0,
        ge=0.1,
        description="Half-life in days for C2 recency scoring.",
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::test_c2_tuning_fields_exist -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nexus/config.py tests/test_config.py
git commit -m "feat: add C2 algorithm tuning fields to NexusSettings"
```

---

## Task 3: Create C2Engine facade — core lifecycle

**Files:**
- Create: `src/nexus/integrations/c2_engine.py`
- Create: `tests/test_c2_engine.py`

**Context:** The `C2Engine` replaces `C2Client` (subprocess). It creates a `TieredMemorySystem` from C2 internals using a bridged config. All sync calls run in `asyncio.to_thread()` to avoid blocking the event loop. Neo4j is disabled (empty URI causes graceful skip in `_init_neo4j`).

**Step 1: Write failing test for C2Engine instantiation**

```python
# tests/test_c2_engine.py

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_settings(monkeypatch):
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from nexus.config import NexusSettings
    return NexusSettings()


def test_c2_engine_import():
    """C2Engine class should be importable."""
    from nexus.integrations.c2_engine import C2Engine
    assert C2Engine is not None


def test_c2_engine_creates_config(mock_settings):
    """C2Engine should build a C2Config from NexusSettings."""
    from nexus.integrations.c2_engine import C2Engine
    engine = C2Engine.__new__(C2Engine)
    config = engine._build_c2_config(mock_settings)
    assert config.redis_url == mock_settings.REDIS_URL
    assert config.qdrant_url == mock_settings.QDRANT_URL
    assert config.postgres_url == mock_settings.POSTGRES_URL
    assert config.neo4j_uri == ""  # disabled
    assert config.embedding_backend == "openrouter"
    assert config.openrouter_api_key == mock_settings.OPENROUTER_API_KEY
    assert config.token_budget == mock_settings.C2_TOKEN_BUDGET


@pytest.mark.asyncio
async def test_c2_engine_start_stop(mock_settings):
    """C2Engine start/stop should work without real backends."""
    from nexus.integrations.c2_engine import C2Engine
    engine = C2Engine(mock_settings)
    assert not engine.is_running
    started = await engine.start()
    # start() succeeds even if backends are unavailable (graceful degradation)
    assert isinstance(started, bool)
    await engine.stop()
    assert not engine.is_running
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_c2_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nexus.integrations.c2_engine'`

**Step 3: Create C2Engine with core lifecycle**

Create `src/nexus/integrations/c2_engine.py`:

```python
"""Continuity Core (C2) direct integration engine.

Replaces the subprocess-based ``C2Client`` with a direct Python facade
over ``TieredMemorySystem``.  All sync C2 internals run in
``asyncio.to_thread()`` to avoid blocking the event loop.

Usage::

    engine = C2Engine(settings)
    await engine.start()
    result = await engine.curiosity()
    await engine.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from nexus.config import NexusSettings

log = logging.getLogger(__name__)


class C2Engine:
    """Async facade over continuity_core internals.

    Builds a ``C2Config`` from ``NexusSettings`` so C2 shares the same
    infrastructure (Redis, Qdrant, Postgres) as the rest of Nexus.
    Neo4j is disabled.
    """

    def __init__(self, settings: NexusSettings) -> None:
        self._settings = settings
        self._system: Any = None  # TieredMemorySystem
        self._running = False
        self._embed_cache: dict[str, list[float]] = {}

    def _build_c2_config(self, settings: NexusSettings) -> Any:
        """Bridge NexusSettings into a C2Config."""
        from continuity_core.config import C2Config

        return C2Config(
            redis_url=settings.REDIS_URL,
            qdrant_url=settings.QDRANT_URL,
            postgres_url=settings.POSTGRES_URL,
            neo4j_uri="",  # disabled — graph in Qdrant metadata
            neo4j_user="",
            neo4j_password="",
            embedding_backend="openrouter",
            embedding_model="",
            ollama_base_url="",
            ollama_embed_model="",
            openrouter_api_key=settings.OPENROUTER_API_KEY,
            openrouter_embed_model=settings.EMBEDDING_MODEL,
            token_budget=settings.C2_TOKEN_BUDGET,
            epsilon=settings.C2_EPSILON,
            lambda_penalty=settings.C2_LAMBDA,
            edge_half_life_days=settings.C2_EDGE_HALF_LIFE_DAYS,
            decay_rate=settings.C2_DECAY_RATE,
            decay_time_unit_sec=86400,
            recency_half_life_days=settings.C2_RECENCY_HALF_LIFE_DAYS,
        )

    async def start(self) -> bool:
        """Initialize the C2 memory system.  Returns True on success."""
        try:
            config = self._build_c2_config(self._settings)
            self._system = await asyncio.to_thread(
                self._create_system, config,
            )
            self._running = True
            log.info("C2Engine started (direct integration)")
            return True
        except Exception as exc:
            log.warning("C2Engine start failed: %s", exc)
            self._running = False
            return False

    @staticmethod
    def _create_system(config: Any) -> Any:
        """Create TieredMemorySystem (sync, runs in thread)."""
        from continuity_core.memory.system import TieredMemorySystem
        return TieredMemorySystem(config)

    async def stop(self) -> None:
        """Shut down the C2 memory system."""
        if self._system is not None:
            neo4j = getattr(self._system, "_neo4j", None)
            if neo4j is not None:
                try:
                    await asyncio.to_thread(neo4j.close)
                except Exception:
                    pass
        self._system = None
        self._running = False
        log.info("C2Engine stopped")

    @property
    def is_running(self) -> bool:
        return self._running and self._system is not None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_c2_engine.py -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add src/nexus/integrations/c2_engine.py tests/test_c2_engine.py
git commit -m "feat: create C2Engine facade with core lifecycle"
```

---

## Task 4: Add C2Engine tool methods

**Files:**
- Modify: `src/nexus/integrations/c2_engine.py`
- Modify: `tests/test_c2_engine.py`

**Context:** C2Engine needs the same 7 methods as C2Client: `write_event`, `get_context`, `introspect`, `curiosity`, `maintenance`, `status`, `events`. Each wraps sync C2 internal calls in `asyncio.to_thread()`. The MCP tool handler code in `continuity_core/mcp/tools/*.py` shows the exact logic — we replicate it directly.

**Step 1: Write failing test for write_event**

```python
# tests/test_c2_engine.py — append

@pytest.mark.asyncio
async def test_write_event(mock_settings):
    """write_event should log to the C2 event store."""
    from nexus.integrations.c2_engine import C2Engine
    engine = C2Engine(mock_settings)
    await engine.start()
    result = await engine.write_event(
        actor="test", intent="test_intent", inp="hello", out="world",
    )
    assert result is not None
    assert result.get("actor") == "test"
    await engine.stop()


@pytest.mark.asyncio
async def test_events(mock_settings):
    """events() should return recently logged events."""
    from nexus.integrations.c2_engine import C2Engine
    engine = C2Engine(mock_settings)
    await engine.start()
    await engine.write_event(actor="test", intent="ping")
    result = await engine.events(limit=5)
    assert result is not None
    assert result.get("count", 0) >= 1
    await engine.stop()


@pytest.mark.asyncio
async def test_status(mock_settings):
    """status() should return backend health info."""
    from nexus.integrations.c2_engine import C2Engine
    engine = C2Engine(mock_settings)
    await engine.start()
    result = await engine.status()
    assert result is not None
    assert "backends" in result or "event_log" in result
    await engine.stop()


@pytest.mark.asyncio
async def test_curiosity_returns_dict_or_none(mock_settings):
    """curiosity() should return a dict or None."""
    from nexus.integrations.c2_engine import C2Engine
    engine = C2Engine(mock_settings)
    await engine.start()
    result = await engine.curiosity()
    assert result is None or isinstance(result, dict)
    await engine.stop()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_c2_engine.py -v`
Expected: FAIL with `AttributeError: 'C2Engine' object has no attribute 'write_event'`

**Step 3: Implement all 7 tool methods**

Append to `src/nexus/integrations/c2_engine.py` inside the `C2Engine` class:

```python
    # ------------------------------------------------------------------
    # Tool methods (same interface as C2Client)
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
        if not self.is_running:
            return None
        try:
            def _write() -> dict[str, Any]:
                event = self._system.event_log.log(
                    actor, intent, inp, out,
                    tags=tags or [], metadata=metadata or {},
                )
                return {
                    "actor": event.actor,
                    "intent": event.intent,
                    "timestamp": event.timestamp,
                }
            return await asyncio.to_thread(_write)
        except Exception as exc:
            log.warning("C2Engine write_event failed: %s", exc)
            return None

    async def get_context(
        self, query: str, token_budget: int = 2048,
    ) -> dict[str, Any] | None:
        """Compose a context pack from C2 memory."""
        if not self.is_running:
            return None
        try:
            def _context() -> dict[str, Any]:
                from continuity_core.context import ContextPipeline, ContextResult
                pipeline = ContextPipeline(
                    memory_system=self._system,
                    config=self._system.config,
                )
                result: ContextResult = pipeline.run(query, thread_id=None)
                chosen = [
                    {"id": c.id, "text": c.text, "store": c.store,
                     "token_cost": c.token_cost}
                    for c in result.chosen
                ]
                return {
                    "token_budget": token_budget,
                    "chosen": chosen,
                    "working_context": result.working_context,
                }
            return await asyncio.to_thread(_context)
        except Exception as exc:
            log.warning("C2Engine get_context failed: %s", exc)
            return None

    async def introspect(
        self,
        statements: list[str],
        concept_contexts: dict[str, Any] | None = None,
        graph: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Run MRA stress and void detection."""
        if not self.is_running:
            return None
        try:
            def _introspect() -> dict[str, Any]:
                from continuity_core.mra.stress import EpistemicStressMonitor
                from continuity_core.mra.voids import VoidDetector

                # Per-call embedding cache — fixes O(n²) redundant calls
                embed_cache: dict[str, list[float]] = {}
                raw_embed = self._system.embedder.embed

                def cached_embed(text: str) -> list[float]:
                    if text not in embed_cache:
                        embed_cache[text] = raw_embed(text)
                    return embed_cache[text]

                monitor = EpistemicStressMonitor(
                    nli_fn=None,
                    embed_fn=cached_embed,
                )

                # Compute graph sparsity
                sparsity = 0.0
                graph_dict: dict[str, set[str]] = {}
                if graph:
                    for node, neighbors in graph.items():
                        if isinstance(neighbors, list):
                            graph_dict[node] = set(neighbors)
                        elif isinstance(neighbors, set):
                            graph_dict[node] = neighbors
                    n = len(graph_dict)
                    if n > 1:
                        edges = sum(len(v) for v in graph_dict.values()) / 2
                        max_edges = n * (n - 1) / 2
                        sparsity = 1.0 - (edges / max_edges) if max_edges > 0 else 0.0

                stress = monitor.compute(
                    statements,
                    concept_contexts=concept_contexts,
                    graph_sparsity=sparsity,
                )

                voids = VoidDetector().detect_voids(graph_dict) if graph_dict else None

                # Update MRA cache
                self._system.update_mra_cache(stress, voids)

                result: dict[str, Any] = {
                    "stress": {
                        "s_omega": stress.s_omega,
                        "d_log": stress.d_log,
                        "d_sem": stress.d_sem,
                        "v_top": stress.v_top,
                        "should_trigger": stress.should_trigger,
                        "contradictions": [
                            {"s1": s1, "s2": s2, "score": sc}
                            for s1, s2, sc in stress.contradictions
                        ],
                        "deep_tensions": [
                            {"s1": s1, "s2": s2, "score": sc, "similarity": sim}
                            for s1, s2, sc, sim in stress.deep_tensions
                        ],
                    },
                }
                if voids is not None:
                    result["voids"] = {
                        "pairs": len(voids.void_pairs),
                        "questions": voids.questions,
                    }
                return result
            return await asyncio.to_thread(_introspect)
        except Exception as exc:
            log.warning("C2Engine introspect failed: %s", exc)
            return None

    async def curiosity(self) -> dict[str, Any] | None:
        """Return epistemic tensions, contradictions, and bridging questions."""
        if not self.is_running:
            return None
        try:
            def _curiosity() -> dict[str, Any] | None:
                cache = self._system.get_mra_signals()
                if cache is None:
                    # Auto-run introspect from recent events
                    events = self._system.event_log.tail(20)
                    if not events:
                        return None
                    statements = [
                        f"{e.intent}: {e.input[:100]}" for e in events
                        if e.input
                    ]
                    if len(statements) < 2:
                        return None
                    # This will populate the cache
                    return None  # Signal to run introspect async

                stress = cache.last_stress
                if stress is None:
                    return None

                result: dict[str, Any] = {
                    "stress_level": stress.s_omega,
                    "contradictions": [
                        {"s1": s1, "s2": s2, "score": sc}
                        for s1, s2, sc in stress.contradictions[:5]
                    ],
                    "deep_tensions": [
                        {"s1": s1, "s2": s2, "score": sc, "similarity": sim}
                        for s1, s2, sc, sim in stress.deep_tensions[:3]
                    ],
                }

                if cache.last_voids and cache.last_voids.questions:
                    result["bridging_questions"] = cache.last_voids.questions[:3]

                if stress.should_trigger:
                    result["suggested_action"] = "investigate_tensions"
                elif stress.s_omega > 0.1:
                    result["suggested_action"] = "monitor"
                else:
                    result["suggested_action"] = "none"

                return result

            sync_result = await asyncio.to_thread(_curiosity)

            # If cache was stale, run introspect and try again
            if sync_result is None and self.is_running:
                events = await asyncio.to_thread(
                    lambda: self._system.event_log.tail(20)
                )
                statements = [
                    f"{e.intent}: {e.input[:100]}" for e in events if e.input
                ]
                if len(statements) >= 2:
                    await self.introspect(statements)
                    sync_result = await asyncio.to_thread(_curiosity)

            return sync_result
        except Exception as exc:
            log.warning("C2Engine curiosity failed: %s", exc)
            return None

    async def maintenance(
        self, graph: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Run a Night Cycle maintenance pass."""
        if not self.is_running:
            return None
        try:
            def _maintenance() -> dict[str, Any]:
                from continuity_core.services.night_cycle import NightCycle

                graph_dict: dict[str, set[str]] | None = None
                if graph:
                    graph_dict = {
                        k: set(v) if isinstance(v, list) else v
                        for k, v in graph.items()
                    }

                cycle = NightCycle(self._system)
                result = cycle.run(graph_dict)
                return {
                    "decay_applied": result.decay_applied,
                    "items_pruned": result.items_pruned,
                    "stress_before": result.stress_before,
                    "stress_after": result.stress_after,
                    "stress_delta": result.stress_delta,
                    "duration_sec": result.duration_sec,
                }
            return await asyncio.to_thread(_maintenance)
        except Exception as exc:
            log.warning("C2Engine maintenance failed: %s", exc)
            return None

    async def status(self) -> dict[str, Any] | None:
        """Query C2 backend health and system metrics."""
        if not self.is_running:
            return None
        try:
            def _status() -> dict[str, Any]:
                info: dict[str, Any] = {
                    "engine": "direct",
                    "event_log": "ok",
                    "event_count": len(self._system.event_log.tail(1)),
                }
                info["qdrant"] = "connected" if self._system.qdrant is not None else "unavailable"
                info["redis"] = "connected" if self._system.redis is not None else "unavailable"
                info["neo4j"] = "disabled"
                info["embedding"] = self._system.config.openrouter_embed_model

                cache = self._system.get_mra_signals()
                if cache and cache.last_stress:
                    info["stress_level"] = cache.last_stress.s_omega
                return info
            return await asyncio.to_thread(_status)
        except Exception as exc:
            log.warning("C2Engine status failed: %s", exc)
            return None

    async def events(self, limit: int = 10) -> dict[str, Any] | None:
        """Read recent events from the C2 event log."""
        if not self.is_running:
            return None
        try:
            def _events() -> dict[str, Any]:
                items = self._system.event_log.tail(min(limit, 50))
                return {
                    "count": len(items),
                    "events": [
                        {
                            "timestamp": e.timestamp,
                            "actor": e.actor,
                            "intent": e.intent,
                            "input": e.input[:200] if e.input else "",
                            "output": e.output[:200] if e.output else "",
                            "tags": e.tags,
                        }
                        for e in items
                    ],
                }
            return await asyncio.to_thread(_events)
        except Exception as exc:
            log.warning("C2Engine events failed: %s", exc)
            return None
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_c2_engine.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/nexus/integrations/c2_engine.py tests/test_c2_engine.py
git commit -m "feat: add C2Engine tool methods (write_event, context, introspect, curiosity, maintenance, status, events)"
```

---

## Task 5: Wire C2Engine into bot.py

**Files:**
- Modify: `src/nexus/bot.py:13,131-132,218-223,263,275`
- Test: Manual verification (Discord bot startup)

**Context:** Replace `C2Client` import and instantiation with `C2Engine`. The rest of the codebase already calls `self.c2.write_event()`, `self.c2.curiosity()`, etc. — same interface, no other changes needed.

**Step 1: Update the import**

In `src/nexus/bot.py`, change line 13:

```python
# Before:
from nexus.integrations.c2_client import C2Client

# After:
from nexus.integrations.c2_engine import C2Engine
```

**Step 2: Update the constructor**

In `src/nexus/bot.py`, change line 132:

```python
# Before:
self.c2 = C2Client()

# After:
self.c2 = C2Engine(settings)
```

**Step 3: Update the on_ready startup message**

In `src/nexus/bot.py`, change lines 218-223:

```python
# Before:
        # Start Continuity Core subprocess
        c2_started = await self.c2.start()
        if c2_started:
            log.info("Continuity Core (C2) subprocess started")
        else:
            log.info("C2 not available - cognitive memory features disabled")

# After:
        # Start Continuity Core (direct integration)
        c2_started = await self.c2.start()
        if c2_started:
            log.info("Continuity Core (C2) engine started (direct)")
        else:
            log.info("C2 not available - cognitive memory features disabled")
```

**Step 4: Update the close method**

Find the `close()` or `async def close()` method in bot.py. Change `await self.c2.stop()` — no change needed, same method name.

**Step 5: Run existing tests to verify nothing breaks**

Run: `pytest tests/ -v`
Expected: All existing tests PASS (c2_client tests may fail since they test the old client — that's expected and we'll delete those in Task 7)

**Step 6: Commit**

```bash
git add src/nexus/bot.py
git commit -m "feat: wire C2Engine into NexusBot (replaces C2Client subprocess)"
```

---

## Task 6: Update state gatherer for C2Engine

**Files:**
- Modify: `src/nexus/orchestrator/state.py`

**Context:** The `_gather_curiosity` method in `state.py` calls `bot.c2.curiosity()` with a 90s timeout. This still works because C2Engine has the same `curiosity()` signature. But the 90s timeout was needed for subprocess overhead — we can reduce it to 30s now.

**Step 1: Reduce the curiosity timeout**

In `src/nexus/orchestrator/state.py`, find the `_gather_curiosity` method. Change the timeout:

```python
# Before:
result = await asyncio.wait_for(
    c2.curiosity(), timeout=90.0,
)

# After:
result = await asyncio.wait_for(
    c2.curiosity(), timeout=30.0,
)
```

**Step 2: Update the outer gather timeout if needed**

The outer `gather()` method has a 120s timeout for all sources. With C2 now much faster, we can reduce to 60s:

```python
# Before:
results = await asyncio.wait_for(
    asyncio.gather(*tasks, return_exceptions=True),
    timeout=120.0,
)

# After:
results = await asyncio.wait_for(
    asyncio.gather(*tasks, return_exceptions=True),
    timeout=60.0,
)
```

**Step 3: Run tests**

Run: `pytest tests/ -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/nexus/orchestrator/state.py
git commit -m "perf: reduce C2 gather timeouts (subprocess overhead eliminated)"
```

---

## Task 7: Delete old C2 subprocess code

**Files:**
- Delete: `src/nexus/integrations/c2_client.py`
- Delete: `src/continuity_core/config.py` — **NO, keep this.** TieredMemorySystem imports it.
- Delete: `src/continuity_core/services/runtime.py` — Only if nothing imports `get_memory_system`
- Delete: `tests/test_c2_client.py`
- Modify: `tests/test_c2_embeddings.py` (update if it imports from deleted files)
- Modify: `tests/test_c2_events_read.py` (update if needed)
- Modify: `tests/test_c2_status.py` (update if needed)

**Context:** Now that C2Engine replaces C2Client, we remove the subprocess client. We keep `continuity_core/config.py` because `TieredMemorySystem` imports `C2Config` and `load_config` from it. We keep `continuity_core/memory/embeddings.py` because `TieredMemorySystem` calls `build_embedder(config)`. These are C2 internals we don't touch.

**Step 1: Check what imports c2_client**

Run: `grep -r "c2_client" src/ tests/` and note all references.

**Step 2: Delete c2_client.py**

```bash
rm src/nexus/integrations/c2_client.py
```

**Step 3: Delete test_c2_client.py**

```bash
rm tests/test_c2_client.py
```

**Step 4: Check runtime.py usage**

Run: `grep -r "get_memory_system\|from continuity_core.services.runtime" src/` — if only used by MCP server handlers, it's safe to leave (MCP server still exists for potential standalone use).

**Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: PASS (tests that imported c2_client are deleted)

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor: remove C2 subprocess client (replaced by C2Engine)"
```

---

## Task 8: Port synthesis data models

**Files:**
- Create: `src/nexus/synthesis/__init__.py`
- Create: `src/nexus/synthesis/models.py`
- Create: `tests/test_synthesis_models.py`

**Context:** Port the essential Pydantic models from `D:\Development\synthesis\synthesis\core\models.py`. We only need: `CapabilityCategory`, `RiskLevel`, `SynthesisStatus`, `ExecutionStatus`, `TestCase`, `TestResult`, `TestSuite`, `ValidationIssue`, `ValidationResult`, `SynthesisAttempt`. Skip trust/composition models (YAGNI).

**Step 1: Write failing test for model imports**

```python
# tests/test_synthesis_models.py

def test_synthesis_models_import():
    """Core synthesis models should be importable."""
    from nexus.synthesis.models import (
        CapabilityCategory,
        RiskLevel,
        SynthesisStatus,
        TestCase,
        TestResult,
        TestSuite,
        ValidationIssue,
        ValidationResult,
        SynthesisAttempt,
    )
    assert SynthesisStatus.COMPLETE.value == "complete"


def test_test_case_creation():
    from nexus.synthesis.models import TestCase
    tc = TestCase(
        name="test_add",
        inputs={"a": 1, "b": 2},
        expected_output=3,
    )
    assert tc.name == "test_add"
    assert tc.expected_output == 3


def test_synthesis_attempt_pass_rate():
    from nexus.synthesis.models import SynthesisAttempt, SynthesisStatus
    attempt = SynthesisAttempt(
        id="test-1",
        requirement="add two numbers",
        status=SynthesisStatus.COMPLETE,
        tests_generated=5,
        tests_passed=4,
        total_tests=5,
    )
    assert attempt.test_pass_rate == 0.8
    assert not attempt.is_success  # needs all tests passing
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthesis_models.py -v`
Expected: FAIL

**Step 3: Create the models**

Create `src/nexus/synthesis/__init__.py`:

```python
"""Synthesis TDD engine for Agent Nexus."""
```

Create `src/nexus/synthesis/models.py` — copy and trim the models from `D:\Development\synthesis\synthesis\core\models.py`. Include only the classes listed in the test. Keep `pydantic.BaseModel` for all.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthesis_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nexus/synthesis/ tests/test_synthesis_models.py
git commit -m "feat: port synthesis TDD data models"
```

---

## Task 9: Port CodeValidator

**Files:**
- Create: `src/nexus/synthesis/validator.py`
- Create: `tests/test_synthesis_validator.py`

**Context:** Port `CodeValidator` from `D:\Development\synthesis\synthesis\core\validator.py`. It validates generated code for safety (no os/subprocess/eval, etc.) using AST analysis. Minimal adaptation needed — just update the import paths to use `nexus.synthesis.models`.

**Step 1: Write failing tests**

```python
# tests/test_synthesis_validator.py

import pytest


@pytest.mark.asyncio
async def test_valid_code_passes():
    from nexus.synthesis.validator import CodeValidator
    v = CodeValidator()
    result = await v.validate("def add(a, b):\n    return a + b\n")
    assert result.is_valid


@pytest.mark.asyncio
async def test_os_import_blocked():
    from nexus.synthesis.validator import CodeValidator
    v = CodeValidator()
    result = await v.validate("import os\ndef run():\n    os.system('ls')\n")
    assert not result.is_valid
    assert any("os" in i.message for i in result.issues)


@pytest.mark.asyncio
async def test_eval_blocked():
    from nexus.synthesis.validator import CodeValidator
    v = CodeValidator()
    result = await v.validate("def run(x):\n    return eval(x)\n")
    assert not result.is_valid


@pytest.mark.asyncio
async def test_syntax_error_caught():
    from nexus.synthesis.validator import CodeValidator
    v = CodeValidator()
    result = await v.validate("def broken(\n")
    assert not result.is_valid
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_synthesis_validator.py -v`
Expected: FAIL

**Step 3: Create validator.py**

Create `src/nexus/synthesis/validator.py` — port from `D:\Development\synthesis\synthesis\core\validator.py`. Change imports from `synthesis.core.models` to `nexus.synthesis.models`.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_synthesis_validator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nexus/synthesis/validator.py tests/test_synthesis_validator.py
git commit -m "feat: port CodeValidator from synthesis"
```

---

## Task 10: Port SandboxRuntime

**Files:**
- Create: `src/nexus/synthesis/sandbox.py`
- Create: `tests/test_synthesis_sandbox.py`

**Context:** Port `SandboxRuntime` from `D:\Development\synthesis\synthesis\sandbox\runtime.py`. It executes generated code in an isolated subprocess with timeouts. Change imports to use `nexus.synthesis.models`.

**Step 1: Write failing tests**

```python
# tests/test_synthesis_sandbox.py

import pytest


@pytest.mark.asyncio
async def test_sandbox_executes_simple_function():
    from nexus.synthesis.sandbox import SandboxRuntime
    sandbox = SandboxRuntime()
    result = await sandbox.execute(
        code="def add(a, b):\n    return a + b\n",
        function_name="add",
        arguments={"a": 2, "b": 3},
    )
    assert result.status.value == "success"
    assert result.output == 5


@pytest.mark.asyncio
async def test_sandbox_handles_timeout():
    from nexus.synthesis.sandbox import SandboxRuntime, SandboxConfig
    sandbox = SandboxRuntime(config=SandboxConfig(timeout_seconds=1.0))
    result = await sandbox.execute(
        code="import time\ndef slow():\n    time.sleep(10)\n    return True\n",
        function_name="slow",
        arguments={},
    )
    assert result.status.value == "timeout"


@pytest.mark.asyncio
async def test_sandbox_handles_exception():
    from nexus.synthesis.sandbox import SandboxRuntime
    sandbox = SandboxRuntime()
    result = await sandbox.execute(
        code="def fail():\n    raise ValueError('boom')\n",
        function_name="fail",
        arguments={},
    )
    assert result.status.value in ("failed", "error")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_synthesis_sandbox.py -v`
Expected: FAIL

**Step 3: Create sandbox.py**

Create `src/nexus/synthesis/sandbox.py` — port from `D:\Development\synthesis\synthesis\sandbox\runtime.py`. Change imports to `nexus.synthesis.models`.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_synthesis_sandbox.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nexus/synthesis/sandbox.py tests/test_synthesis_sandbox.py
git commit -m "feat: port SandboxRuntime from synthesis"
```

---

## Task 11: Create NexusLLMAdapter and TDDEngine

**Files:**
- Create: `src/nexus/synthesis/tdd_engine.py`
- Create: `tests/test_tdd_engine.py`

**Context:** The synthesis `TDDSynthesizer` needs an `LLMProvider` with an `async complete(prompt, system, temperature, max_tokens)` method. We create `NexusLLMAdapter` wrapping Nexus's `OpenRouterClient.chat()`, then port `TDDSynthesizer` as `TDDEngine`.

**Step 1: Write failing tests**

```python
# tests/test_tdd_engine.py

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_llm():
    """Mock LLM that returns simple Python code."""
    llm = AsyncMock()

    async def mock_complete(prompt, **kwargs):
        from nexus.synthesis.tdd_engine import LLMResponse
        if "test" in prompt.lower() and "generate" in prompt.lower():
            return LLMResponse(
                content='```json\n{"name": "test_add", "tests": [{"name": "test_basic", "inputs": {"a": 1, "b": 2}, "expected_output": 3}]}\n```',
                finish_reason="stop",
            )
        return LLMResponse(
            content='```python\ndef add(a, b):\n    return a + b\n```',
            finish_reason="stop",
        )

    llm.complete = mock_complete
    return llm


def test_nexus_llm_adapter_import():
    from nexus.synthesis.tdd_engine import NexusLLMAdapter
    assert NexusLLMAdapter is not None


def test_tdd_engine_import():
    from nexus.synthesis.tdd_engine import TDDEngine
    assert TDDEngine is not None


@pytest.mark.asyncio
async def test_nexus_llm_adapter_wraps_openrouter():
    """NexusLLMAdapter should wrap OpenRouterClient.chat()."""
    from nexus.synthesis.tdd_engine import NexusLLMAdapter

    mock_or = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "Hello world"
    mock_response.finish_reason = "stop"
    mock_response.input_tokens = 10
    mock_response.output_tokens = 5
    mock_or.chat = AsyncMock(return_value=mock_response)

    adapter = NexusLLMAdapter(mock_or, model="test/model")
    result = await adapter.complete("Say hello")
    assert result.content == "Hello world"
    mock_or.chat.assert_called_once()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tdd_engine.py -v`
Expected: FAIL

**Step 3: Create tdd_engine.py**

Create `src/nexus/synthesis/tdd_engine.py`:

```python
"""TDD Synthesis Engine for Agent Nexus.

Generates code through iterative test-driven development:
generate tests -> generate implementation -> run in sandbox -> refine.

Adapted from d:/Development/synthesis/synthesis/core/synthesis.py
"""

from __future__ import annotations

import ast
import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel

from nexus.synthesis.models import (
    CapabilityCategory,
    RiskLevel,
    SynthesisAttempt,
    SynthesisStatus,
    TestCase,
    TestResult,
    TestSuite,
    ValidationResult,
)
from nexus.synthesis.sandbox import SandboxRuntime
from nexus.synthesis.validator import CodeValidator

log = logging.getLogger(__name__)


# -- LLM abstraction -------------------------------------------------------


class LLMResponse(BaseModel):
    content: str
    finish_reason: str
    tokens_used: int | None = None


class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse: ...


class NexusLLMAdapter(LLMProvider):
    """Wraps Nexus's OpenRouterClient for the TDD synthesizer."""

    def __init__(self, openrouter: Any, model: str | None = None) -> None:
        self._client = openrouter
        self._model = model

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        model = self._model or getattr(self._client, "default_model", None)
        if model is None:
            raise ValueError("No model specified for NexusLLMAdapter")

        resp = await self._client.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return LLMResponse(
            content=resp.content,
            finish_reason=resp.finish_reason or "stop",
            tokens_used=(resp.input_tokens or 0) + (resp.output_tokens or 0),
        )


# -- TDD Engine -------------------------------------------------------------


class TDDEngine:
    """Generates code through iterative test-driven development.

    Usage::

        engine = TDDEngine(llm=NexusLLMAdapter(openrouter, model="..."))
        result = await engine.synthesize("A function that adds two numbers")
    """

    def __init__(
        self,
        llm: LLMProvider,
        max_iterations: int = 5,
        sandbox_timeout: float = 30.0,
    ) -> None:
        self.llm = llm
        self.max_iterations = max_iterations
        self.validator = CodeValidator()
        self.sandbox = SandboxRuntime()

    async def synthesize(
        self,
        requirement: str,
        category: CapabilityCategory | None = None,
    ) -> SynthesisAttempt:
        """Run full TDD synthesis loop for a requirement."""
        attempt = SynthesisAttempt(
            id=uuid.uuid4().hex[:12],
            requirement=requirement,
            category=category,
            status=SynthesisStatus.GENERATING_TESTS,
        )

        try:
            # Phase 1: Generate tests
            test_suite = await self._generate_tests(requirement)
            attempt.test_suite = test_suite
            attempt.tests_generated = len(test_suite.tests)
            attempt.total_tests = len(test_suite.tests)

            # Phase 2: Generate initial implementation
            attempt.status = SynthesisStatus.GENERATING_CODE
            code = await self._generate_implementation(requirement, test_suite)
            attempt.generated_code = code

            # Phase 3: Iterate until tests pass
            for i in range(self.max_iterations):
                attempt.iterations = i + 1
                attempt.status = SynthesisStatus.RUNNING_TESTS

                test_results, validation = await self._run_tests(code, test_suite)
                attempt.test_results = test_results
                attempt.tests_passed = sum(1 for t in test_results if t.passed)

                all_pass = all(t.passed for t in test_results) and validation.is_valid

                if all_pass:
                    attempt.status = SynthesisStatus.COMPLETE
                    attempt.generated_code = code
                    attempt.completed_at = datetime.now(timezone.utc)
                    break

                if i < self.max_iterations - 1:
                    attempt.status = SynthesisStatus.REFINING
                    code = await self._refine_implementation(
                        requirement, code, test_suite, test_results, validation, i,
                    )
                    attempt.generated_code = code
                else:
                    attempt.status = SynthesisStatus.FAILED
                    attempt.errors.append(
                        f"Failed after {self.max_iterations} iterations"
                    )

        except Exception as exc:
            attempt.status = SynthesisStatus.FAILED
            attempt.errors.append(str(exc))
            log.warning("TDD synthesis failed: %s", exc)

        return attempt

    async def _generate_tests(self, requirement: str) -> TestSuite:
        prompt = f"""Generate a JSON test suite for this requirement:

{requirement}

Return ONLY valid JSON in this format:
```json
{{
  "name": "test_suite_name",
  "tests": [
    {{
      "name": "test_case_name",
      "description": "what this tests",
      "inputs": {{"param1": "value1"}},
      "expected_output": "expected_value"
    }}
  ]
}}
```

Generate 3-5 test cases covering normal cases, edge cases, and error cases."""

        resp = await self.llm.complete(prompt, temperature=0.3)
        content = resp.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        data = json.loads(content.strip())
        tests = [TestCase(**t) for t in data.get("tests", [])]
        return TestSuite(
            name=data.get("name", "generated_tests"),
            tests=tests,
        )

    async def _generate_implementation(
        self, requirement: str, test_suite: TestSuite,
    ) -> str:
        tests_desc = "\n".join(
            f"- {t.name}: inputs={t.inputs}, expected={t.expected_output}"
            for t in test_suite.tests
        )
        prompt = f"""Write a Python function for this requirement:

{requirement}

It must pass these tests:
{tests_desc}

Return ONLY the function code in a ```python block. No imports of os, subprocess, sys, etc."""

        resp = await self.llm.complete(prompt, temperature=0.3)
        return self._extract_code(resp.content)

    async def _run_tests(
        self, code: str, test_suite: TestSuite,
    ) -> tuple[list[TestResult], ValidationResult]:
        validation = await self.validator.validate(code)
        results: list[TestResult] = []

        func_name = self._find_function_name(code)
        if func_name is None:
            return results, validation

        for tc in test_suite.tests:
            try:
                exec_result = await self.sandbox.execute(
                    code=code,
                    function_name=func_name,
                    arguments=tc.inputs,
                )
                passed = (
                    exec_result.status.value == "success"
                    and exec_result.output == tc.expected_output
                )
                results.append(TestResult(
                    test_case=tc,
                    passed=passed,
                    actual_output=exec_result.output,
                    error_message=exec_result.error,
                    execution_time_ms=exec_result.execution_time_ms,
                ))
            except Exception as exc:
                results.append(TestResult(
                    test_case=tc,
                    passed=False,
                    error_message=str(exc),
                    execution_time_ms=0.0,
                ))

        return results, validation

    async def _refine_implementation(
        self,
        requirement: str,
        current_code: str,
        test_suite: TestSuite,
        test_results: list[TestResult],
        validation: ValidationResult,
        iteration: int,
    ) -> str:
        failures = [
            f"- {r.test_case.name}: expected={r.test_case.expected_output}, "
            f"got={r.actual_output}, error={r.error_message}"
            for r in test_results if not r.passed
        ]
        issues = [f"- {i.message}" for i in validation.issues] if not validation.is_valid else []

        prompt = f"""Fix this Python function (iteration {iteration + 1}):

Requirement: {requirement}

Current code:
```python
{current_code}
```

Test failures:
{chr(10).join(failures) if failures else "None"}

Validation issues:
{chr(10).join(issues) if issues else "None"}

Return ONLY the fixed function in a ```python block."""

        resp = await self.llm.complete(prompt, temperature=0.2)
        return self._extract_code(resp.content)

    @staticmethod
    def _extract_code(content: str) -> str:
        if "```python" in content:
            return content.split("```python")[1].split("```")[0].strip()
        if "```" in content:
            return content.split("```")[1].split("```")[0].strip()
        return content.strip()

    @staticmethod
    def _find_function_name(code: str) -> str | None:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except SyntaxError:
            pass
        return None
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tdd_engine.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nexus/synthesis/tdd_engine.py tests/test_tdd_engine.py
git commit -m "feat: create TDDEngine with NexusLLMAdapter"
```

---

## Task 12: Wire TDDEngine into bot

**Files:**
- Modify: `src/nexus/bot.py`

**Context:** Add TDDEngine as an optional component on the bot, initialized after OpenRouter. The orchestrator can dispatch synthesis tasks through it later.

**Step 1: Add import and initialization**

In `src/nexus/bot.py`, add the import:

```python
from nexus.synthesis.tdd_engine import NexusLLMAdapter, TDDEngine
```

In `__init__`, after the C2 initialization (around line 132), add:

```python
        # --- Synthesis TDD Engine ---
        self.tdd = TDDEngine(
            llm=NexusLLMAdapter(self.openrouter),
        )
```

**Step 2: Update the #nexus announcement**

In the feature list (around line 259), add:

```python
        features.append("tdd=on")
```

**Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add src/nexus/bot.py
git commit -m "feat: wire TDDEngine into NexusBot"
```

---

## Task 13: Final cleanup

**Files:**
- Modify: `docker/docker-compose.yml` (remove neo4j volume if desired)
- Verify: all tests pass

**Step 1: Remove neo4j from default startup (optional)**

The neo4j service already has `profiles: [neo4j]` so it won't start by default. No change needed. But we can remove `neo4j_data` from the volumes section if we want to clean up:

```yaml
volumes:
  qdrant_data:
  redis_data:
  postgres_data:
```

**Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 3: Run linter**

Run: `ruff check src/ tests/`
Expected: No errors (fix any that appear)

Run: `ruff format src/ tests/`

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup for monorepo integration"
```

---

## Summary

| Task | Description | Files Created | Files Modified |
|------|------------|---------------|---------------|
| 1 | Postgres + NexusSettings | — | config.py, docker-compose.yml, test_config.py |
| 2 | C2 tuning fields | — | config.py, test_config.py |
| 3 | C2Engine lifecycle | c2_engine.py, test_c2_engine.py | — |
| 4 | C2Engine tool methods | — | c2_engine.py, test_c2_engine.py |
| 5 | Wire into bot.py | — | bot.py |
| 6 | Update state gatherer | — | state.py |
| 7 | Delete old C2 client | — | delete c2_client.py, test_c2_client.py |
| 8 | Synthesis models | models.py, test_synthesis_models.py | — |
| 9 | CodeValidator | validator.py, test_synthesis_validator.py | — |
| 10 | SandboxRuntime | sandbox.py, test_synthesis_sandbox.py | — |
| 11 | TDDEngine + adapter | tdd_engine.py, test_tdd_engine.py | — |
| 12 | Wire TDD into bot | — | bot.py |
| 13 | Final cleanup | — | docker-compose.yml |
