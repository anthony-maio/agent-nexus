# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Agent Nexus is an app-first multi-model AI swarm platform. The primary surface is the web app/API stack (`nexus-api` + frontend), with supervised browser-first automation running through an isolated sandbox runner. Discord remains optional as a secondary remote inbox for status and approvals.

## Architecture

Primary app stack:
- **`nexus_core`**: transport-agnostic orchestration runtime (`RunEngine`, risk policy, adapter protocols, event stream).
- **`nexus_api`**: FastAPI control plane with single-admin auth, run lifecycle APIs, approvals, citations, artifact promotion.
- **`nexus_sandbox_runner`**: isolated execution service for browser-first step execution and run-local artifacts.
- **Frontend (`frontend/`)**: React app with single-assistant UI, optional trace view, approvals queue, citations panel.

Secondary optional adapters:
- **`nexus_discord_bridge`**: status + approval bridge for remote/away-from-app usage.
- **Legacy `nexus` Discord bot runtime**: still available, but no longer the primary product direction.

## Package Layout

```
src/
  nexus_core/      # App-first transport-agnostic runtime
  nexus_api/       # FastAPI control plane
  nexus_sandbox_runner/ # Isolated step execution service
  nexus_discord_bridge/ # Optional status/approval bridge
  nexus/           # Legacy Discord bot package (secondary)
    config.py      # All env/config loading
    bot.py         # NexusBot class (discord.py)
    models/        # OpenRouter + Ollama clients, model registry
    channels/      # 3-channel routing + auto-creation
    memory/        # Qdrant vector store + context packing
    orchestrator/  # Background loop: gather -> decide -> dispatch
    swarm/         # Multi-model conversation, crosstalk, consensus
    commands/      # Discord commands (!ask, !think, !memory, etc.)
    integrations/  # PiecesOS MCP client
    personality/   # System prompts, model identities
  continuity_core/ # Memory engine (part of monorepo, edit freely)
  synthesis/       # Capability ecosystem (Phase 3)
```

## Build and Run

```bash
# App-first stack
python -m nexus_sandbox_runner
python -m nexus_api
alembic upgrade head

# Optional legacy/bridge services
python -m nexus_discord_bridge
python -m nexus

# Infra
docker compose -f docker/docker-compose.yml up -d

# Development (without Docker)
pip install -e ".[dev]"

# Tests
pytest tests/

# Lint
ruff check src/
ruff format src/
```

## Key Patterns

- All LLM calls go through `models/openrouter.py` (primary) or `models/ollama.py` (local fallback). Never call APIs directly.
- Embeddings are locked to a single provider chosen at setup time. Changing breaks all vectors. See `models/embeddings.py`.
- App-first run execution goes through `nexus_core/engine.py` with adapter boundaries:
  - `InteractionAdapter` for channel/status/approval notifications.
  - `ExecutionAdapter` for sandbox step execution.
- Risk-tier policy gates high-impact actions (`submit`, `write`, `export`, `promote`, etc.) behind supervised approvals.
- Canonical outputs are promoted from sandbox artifact paths into app workspace only via explicit promote actions.
- App control-plane schema is managed via Alembic in `alembic/`; avoid re-introducing `metadata.create_all` startup behavior.
- Sandbox runner supports optional shared-token auth via `SANDBOX_RUNNER_TOKEN`; keep API and runner tokens aligned.
- Sandbox execution backend is selectable via `SANDBOX_EXECUTION_BACKEND`:
  - `local`: in-process per-step ephemeral workspace.
  - `docker`: throwaway container per step using `SANDBOX_DOCKER_*` settings.
- `continuity_core/` is part of this monorepo. It was originally a separate project but is now maintained here. Edit freely.

## Environment Variables

App-first required:
- `APP_DATABASE_URL`
- `APP_ADMIN_USERNAME`
- `APP_ADMIN_PASSWORD`

Optional/secondary:
- `DISCORD_TOKEN` (for bridge or legacy bot runtime)
- `OPENROUTER_API_KEY` (for model-backed execution paths)

See `config/.env.example` for full list.

## Docker Services

Infrastructure services can be mixed with external equivalents via environment overrides.

- `nexus-api` - App control plane (FastAPI)
- `nexus-sandbox-runner` - Isolated execution service
- `nexus-discord-bridge` - Optional remote approval/status bridge
- `nexus-bot` - Legacy Discord bot runtime (optional)
- `nexus-qdrant` - Vector memory (port 6333, profile: qdrant)
- `nexus-redis` - Working memory cache (port 6379, profile: redis)
- `nexus-neo4j` - C2 knowledge graph (port 7687, profile: neo4j)
- `nexus-postgres` - App/C2 persistence (port 5432)
