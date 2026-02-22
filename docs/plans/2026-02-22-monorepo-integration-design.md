# Monorepo Integration Design: C2 + Synthesis into Agent Nexus

**Date:** 2026-02-22
**Approach:** B — Adapter Layer (keep C2 internals, wrap in async facade)
**Status:** Approved

---

## Goal

Integrate `continuity_core` (C2 cognitive memory engine) and the TDD synthesizer from `synthesis` into the agent-nexus monorepo as first-class libraries. Eliminate the subprocess MCP boundary, unify infrastructure (embeddings, Qdrant, Redis, Postgres), and make the entire system deployable from a single `docker-compose.yml`.

## Architecture Overview

```
NexusBot
├── models/          (OpenRouter + Ollama — shared by everyone)
├── memory/          (Qdrant vectors — shared collection)
├── orchestrator/    (decision loop, dispatch, triggers, goals)
├── swarm/           (multi-model conversation)
├── integrations/
│   ├── c2_engine.py      ← NEW: async facade replacing c2_client.py
│   └── pieces.py         (PiecesOS MCP — unchanged)
├── synthesis/
│   └── tdd_engine.py     ← NEW: TDD synthesizer adapted from d:/synthesis
└── continuity_core/      (C2 internals — kept as-is, dependency-injected)
```

**Key principle:** C2's internal modules (`event_log`, `context`, `mra`, `graph`, `memory`) stay intact. A thin `C2Engine` facade wraps them with async methods and injected dependencies. No rewrite of C2 internals.

## Section 1: C2Engine Facade

### What changes

| Component | Before | After |
|-----------|--------|-------|
| `c2_client.py` | Subprocess + JSON-RPC | **Deleted** |
| `C2Engine` | Does not exist | New async facade at `src/nexus/integrations/c2_engine.py` |
| C2 internals | Called via MCP server | Called directly as Python imports |
| Embedding calls | Sync `requests.post` per pair | Async + per-statement cache |

### C2Engine interface

```python
class C2Engine:
    """Async facade over continuity_core internals.

    Accepts injected dependencies so C2 shares Nexus infrastructure
    instead of managing its own connections.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], Awaitable[list[float]]],
        qdrant_client: AsyncQdrantClient,
        redis: Redis,
        pg_pool: asyncpg.Pool,
    ) -> None: ...

    async def start(self) -> bool: ...
    async def stop(self) -> None: ...

    # Tool equivalents (same signatures as C2Client)
    async def write_event(self, actor, intent, inp, out, tags, metadata) -> dict | None: ...
    async def get_context(self, query, token_budget=2048) -> dict | None: ...
    async def introspect(self, statements, concept_contexts, graph) -> dict | None: ...
    async def curiosity(self) -> dict | None: ...
    async def maintenance(self, graph) -> dict | None: ...
    async def status(self) -> dict | None: ...
    async def events(self, limit=10) -> dict | None: ...
```

### O(n²) embedding fix

The `EpistemicStressMonitor.compute()` in `mra/stress.py` calls `self.embed_fn(s1)` and `self.embed_fn(s2)` for every pair — 20 statements = 380 API calls with no caching. The C2Engine facade fixes this:

```python
async def _cached_embed(self, text: str) -> list[float]:
    """Per-request embedding cache — eliminates O(n²) redundant calls."""
    if text not in self._embed_cache:
        self._embed_cache[text] = await self._embed_fn(text)
    return self._embed_cache[text]

async def introspect(self, statements, ...):
    self._embed_cache.clear()  # fresh cache per introspect call
    # Pass self._cached_embed as embed_fn to stress monitor
```

This reduces 380 API calls to 20 (one per unique statement).

### Integration point

In `bot.py`, replace:

```python
# Before
self.c2 = C2Client()
await self.c2.start()

# After
self.c2 = C2Engine(
    embed_fn=self.embedding_provider.embed,
    qdrant_client=self.qdrant_client,
    redis=self.redis,
    pg_pool=self.pg_pool,
)
await self.c2.start()
```

The rest of the codebase (orchestrator, state gathering, etc.) continues calling `self.c2.curiosity()`, `self.c2.write_event()`, etc. — same interface, no downstream changes.

## Section 2: Unified Config and Backends

### Config consolidation

C2 currently has its own `C2Config` dataclass with 12 `C2_*` env vars. These get merged into `NexusSettings`:

| C2 Setting | Nexus Equivalent | Action |
|------------|-----------------|--------|
| `C2_REDIS_URL` | `REDIS_URL` | **Drop** — share existing |
| `C2_QDRANT_URL` | `QDRANT_URL` | **Drop** — share existing |
| `C2_POSTGRES_URL` | `POSTGRES_URL` | **New** — add to NexusSettings |
| `C2_NEO4J_*` (3 vars) | — | **Drop** — not used in adapter approach |
| `C2_EMBEDDING_BACKEND` | `EMBEDDING_MODEL` | **Drop** — Nexus owns embeddings |
| `C2_EMBEDDING_MODEL` | `EMBEDDING_MODEL` | **Drop** — unified |
| `C2_OLLAMA_*` (2 vars) | `OLLAMA_BASE_URL` | **Drop** — Nexus owns embeddings |
| `C2_OPENROUTER_*` (2 vars) | `OPENROUTER_API_KEY` | **Drop** — unified |
| `C2_TOKEN_BUDGET` | `C2_TOKEN_BUDGET` | **Keep** — C2-specific tuning |
| `C2_EPSILON` | `C2_EPSILON` | **Keep** — C2-specific tuning |
| `C2_LAMBDA` | `C2_LAMBDA` | **Keep** — C2-specific tuning |
| `C2_*_DAYS/SEC` (4 vars) | `C2_*` | **Keep** — decay/consolidation tuning |

Net result: 12 env vars become 6 (only C2-specific algorithm tuning params survive).

### New NexusSettings fields

```python
# Infrastructure — new
POSTGRES_URL: str = Field(
    default="postgresql://nexus:nexus@nexus-postgres:5432/nexus",
    description="PostgreSQL connection URL for event log and synthesis.",
)

# C2 algorithm tuning — migrated
C2_TOKEN_BUDGET: int = Field(default=2048)
C2_EPSILON: float = Field(default=0.05)
C2_LAMBDA: float = Field(default=0.001)
C2_EDGE_HALF_LIFE_DAYS: float = Field(default=7.0)
C2_DECAY_RATE: float = Field(default=0.95)
C2_RECENCY_HALF_LIFE_DAYS: float = Field(default=14.0)
```

### Backend sharing

| Backend | Current | After |
|---------|---------|-------|
| **Qdrant** | Nexus: `nexus_memory` collection / C2: own collection | Single client, separate collections (C2 keeps its own namespace) |
| **Redis** | Nexus: goals/cache / C2: own connection | Single connection pool, C2 keys prefixed `c2:` |
| **Postgres** | C2: event log (separate DB) | **New container** `nexus-postgres`, shared by C2 event log + synthesis |
| **Neo4j** | C2: concept graph | **Dropped** — concept graph stored in Qdrant metadata + Postgres |

### Docker additions

```yaml
# docker/docker-compose.yml — new service
nexus-postgres:
  image: postgres:16-alpine
  environment:
    POSTGRES_USER: nexus
    POSTGRES_PASSWORD: nexus
    POSTGRES_DB: nexus
  ports:
    - "5432:5432"
  volumes:
    - postgres_data:/var/lib/postgresql/data
```

### Files deleted

- `src/continuity_core/config.py` — replaced by NexusSettings
- `src/continuity_core/services/runtime.py` — replaced by dependency injection
- `src/continuity_core/memory/embeddings.py` — replaced by Nexus EmbeddingProvider
- `src/nexus/integrations/c2_client.py` — replaced by C2Engine

## Section 3: Synthesis TDD Engine

### Scope

From `d:/Development/synthesis/synthesis`, we take **only** the TDD synthesizer:

- `core/synthesis.py` — `TDDSynthesizer` class (iterative test-driven code generation)
- `core/validator.py` — `CodeValidator` (syntax/import/type checking)
- `core/models.py` — data models (`Capability`, `TestResult`, `SynthesisResult`, etc.)
- `sandbox/runtime.py` — `SandboxRuntime` (isolated code execution)

We do **not** take: exchange server, trust marketplace, MCP server, observatory, composition planner.

### Architecture

```
src/nexus/synthesis/
├── __init__.py
├── tdd_engine.py     ← Adapted TDDSynthesizer
├── validator.py      ← CodeValidator (mostly unchanged)
├── models.py         ← Data models (trimmed)
└── sandbox.py        ← SandboxRuntime (mostly unchanged)
```

### LLM adapter

The synthesis code expects an `LLMProvider` interface. We adapt it to use Nexus's `OpenRouterClient`:

```python
class NexusLLMAdapter:
    """Wraps Nexus's OpenRouterClient for the TDD synthesizer."""

    def __init__(self, openrouter: OpenRouterClient, model: str | None = None):
        self._client = openrouter
        self._model = model  # If None, uses swarm model selection

    async def generate(self, prompt: str, **kwargs) -> str:
        response = await self._client.chat_completion(
            model=self._model or self._client.default_model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.content
```

### Integration point

```python
# In bot.py or orchestrator
from nexus.synthesis.tdd_engine import TDDEngine

self.tdd = TDDEngine(
    llm=NexusLLMAdapter(self.openrouter),
    pg_pool=self.pg_pool,      # For storing synthesis results
    sandbox_timeout=30,
)
```

### How the swarm uses it

The orchestrator can dispatch a TDD synthesis task when the swarm identifies a need for a new tool:

1. Swarm discussion identifies need: "We need a function that parses X"
2. Orchestrator creates a goal with TDD synthesis task
3. Task agent calls `tdd.synthesize(spec="Parse X from Y format", tests=[...])`
4. TDD engine iterates: generate tests → generate impl → run in sandbox → refine
5. Result stored in Postgres, available to swarm

## Section 4: Migration Path

### Phase order

1. **Add Postgres** — Docker service + NexusSettings field + asyncpg pool in bot.py
2. **C2Engine facade** — New file, swap in bot.py, verify all C2 calls work
3. **Kill subprocess** — Delete c2_client.py, continuity_core/config.py, runtime.py, embeddings.py
4. **Synthesis TDD** — Port files, add NexusLLMAdapter, wire to orchestrator
5. **Cleanup** — Remove dead C2 env vars from .env.example, update setup wizard

### Risk mitigation

- C2Engine exposes the **same interface** as C2Client — downstream code doesn't change
- Each phase is independently testable and committable
- Subprocess can be kept as fallback during Phase 2 (feature flag) if needed
- Synthesis is additive — doesn't touch existing code paths

## Appendix: Known Issues (from PR #2 review, fixed)

These were identified during code review and fixed before this design:

- **GoalStaleTrigger repeated firing** — Fixed: dedup by goal_id + updated_at
- **prune_stale_goals wrong timestamp** — Fixed: uses updated_at not created_at
- **Trigger registration not idempotent** — Fixed: dedup by name in add_trigger()
- **Dispatch None stuck tasks** — Fixed: mark_task_failed on None dispatch result

## Appendix: Known Issues (deferred to implementation)

- N+1 Redis queries in GoalStore (batch with pipeline)
- initiative.py duplicates reaction loop code from bot.py
- Fragile keyword-based model selection in orchestrator
- Consensus NEEDS_HUMAN treated as approved in autopilot mode
