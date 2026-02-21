# Curiosity-Driven Autonomous Action — Design Document

## Goal

Make the Agent Nexus swarm autonomously investigate and discuss epistemic tensions discovered by Continuity Core (C2), producing tangible output in Discord. Fix silent infrastructure failures so the MRA runs at full power.

## Problem Statement

1. **C2's knowledge graph can't build** — Neo4j and PostgreSQL are not in the Docker stack. C2 falls back to in-memory stores for everything, losing state on restart and disabling graph-based void detection entirely.
2. **C2 uses hash embeddings** — SHA256-based vectors give near-random similarity scores, making contradiction and tension detection unreliable.
3. **Curiosity findings are decorative** — The night cycle posts a passive embed to #nexus saying "Consider discussing these tensions." Nobody acts on it.
4. **No visibility into C2** — No way to see which backends are connected, what the knowledge graph looks like, or what events C2 has logged.

## Architecture

### 1. Docker Infrastructure

Add two new services to `docker/docker-compose.yml`:

- **`nexus-neo4j`** — Neo4j 5 Community Edition. Knowledge graph for C2's PKM nodes/edges and void detection. Browser UI at `127.0.0.1:7474` for manual inspection.
- **`nexus-postgres`** — PostgreSQL 16. Persistent event log for C2 (replaces in-memory fallback).

Wire C2 environment variables on the `nexus-bot` service so C2 connects to all four backends (Neo4j, Postgres, Qdrant, Redis) over the Docker network.

### 2. C2 Embedding Backend — OpenRouter Cloud

Add a new `OpenRouterEmbedder` class to `continuity_core/memory/embeddings.py` that calls OpenRouter's `/api/v1/embeddings` endpoint via `requests.post()`. This uses the same cloud embedding model as the main nexus memory store, ensuring consistent quality.

- New config fields: `C2_OPENROUTER_API_KEY`, `C2_OPENROUTER_EMBED_MODEL`
- Default model: `qwen/qwen3-embedding-8b` (4096 dimensions)
- `build_embedder()` updated with `openrouter` backend option
- Set `C2_EMBEDDING_BACKEND=openrouter` in Docker environment

### 3. C2 MCP Tools (New)

Two new read-only introspection tools added to the C2 MCP server:

- **`c2.status`** — Returns backend health: which stores are connected, item/node/edge counts, current stress level, embedding backend, last maintenance timestamp.
- **`c2.events`** — Returns the last N events from the event log (default 10).

Corresponding async wrappers added to `nexus/integrations/c2_client.py`.

### 4. Curiosity-Driven Swarm Discussion

When the night cycle finds actionable curiosity signals, the orchestrator triggers a **Tier 1 group discussion** instead of just posting a passive embed.

**Trigger criteria** (any of):
- `stress_after > 0.2`
- `contradictions_found > 0`
- `voids_found > 0`

**Discussion flow:**
1. Build a curiosity prompt summarizing the findings (contradictions, tensions, bridging questions, suggested action)
2. Pick a random Tier 1 model as primary responder
3. Send the curiosity prompt with the model's normal system prompt
4. Post the response to #nexus as a regular model embed
5. Run the crosstalk reaction round (up to 2 other models react)
6. Store all responses in Qdrant memory (feeds back into future cycles)
7. Post a structured summary to #memory

This reuses the existing human-message flow pattern (`_handle_human_message` → primary response → crosstalk reactions), adapted for system-initiated prompts.

**Capped at 1 curiosity discussion per full cycle** to prevent spam.

### 5. Discord Commands

- **`!c2status`** — Dashboard embed showing C2 backend health, counts, stress level, embedding backend.
- **`!c2events [N]`** — Show last N events from C2's event log (default 10).
- **`!discuss`** — Manually trigger a curiosity discussion. Calls `c2.curiosity()`, checks for actionable signals, runs the discussion flow.

### 6. Housekeeping

- Update `CLAUDE.md` to reflect monorepo decision (continuity_core is part of agent-nexus, editable here).
- Add `sentence-transformers` as an optional dependency (fallback embedding option).

## Data Flow

```
Night Cycle → c2.maintenance() → NightCycleResult
    ↓
Gate: stress > 0.2 OR contradictions > 0 OR voids > 0?
    ↓ yes
Build curiosity prompt from findings
    ↓
Pick random Tier 1 model → openrouter.chat() → primary response
    ↓
Post to #nexus as model embed
    ↓
Crosstalk reaction round (up to 2 other models respond)
    ↓
Store all responses in Qdrant memory
    ↓
Post summary to #memory
```

## Files Changed

### New Files
- `continuity_core/mcp/tools/status.py` — c2.status tool
- `continuity_core/mcp/tools/events_read.py` — c2.events tool

### Modified Files
- `docker/docker-compose.yml` — Add Neo4j + Postgres services, C2 env vars
- `continuity_core/memory/embeddings.py` — Add OpenRouterEmbedder + update build_embedder()
- `continuity_core/config.py` — Add openrouter_api_key, openrouter_embed_model fields
- `continuity_core/mcp/server.py` — Register new tools
- `continuity_core/mcp/tools/__init__.py` — Export new tools
- `src/nexus/integrations/c2_client.py` — Add status() and events() wrappers
- `src/nexus/orchestrator/loop.py` — Add _trigger_curiosity_discussion(), update _run_night_cycle()
- `src/nexus/commands/admin.py` — Add !c2status, !c2events, !discuss commands
- `CLAUDE.md` — Update continuity_core guidance (monorepo)
