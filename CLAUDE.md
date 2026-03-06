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

# Docker sandbox backend (ephemeral containers per step)
SANDBOX_EXECUTION_BACKEND=docker docker compose -f docker/docker-compose.yml --profile sandbox-docker up -d

# Docker sandbox backend via host socket (trusted environments only)
SANDBOX_EXECUTION_BACKEND=docker docker compose -f docker/docker-compose.yml -f docker/docker-compose.host-socket.yml up -d

# Production app stack with reverse proxy + frontend
docker compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up -d --build

# One-click helpers
scripts/dev-up.ps1 -SandboxBackend local
scripts/dev-up.ps1 -SandboxBackend docker
scripts/dev-up.ps1 -SandboxBackend docker-host
./scripts/dev-up.sh local
./scripts/dev-up.sh docker
./scripts/dev-up.sh docker-host
scripts/prod-up.ps1 -SandboxBackend docker
scripts/prod-up.ps1 -SandboxBackend docker-host
./scripts/prod-up.sh docker
./scripts/prod-up.sh docker-host

# First-run bootstrap
# If config/.env is missing, the web app now owns setup before login.

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
  - `docker`: throwaway container per step using `SANDBOX_DOCKER_*` settings (`network=none`, caps dropped, read-only rootfs, pids/memory/cpu limits).
  - Docker backend enforces pinned digest images and allowlist validation (`SANDBOX_DOCKER_IMAGE`, `SANDBOX_DOCKER_ALLOWED_IMAGES`).
  - Helper scripts auto-build `agent-nexus-sandbox-step:local` and opt it in with `SANDBOX_DOCKER_ALLOW_UNPINNED_LOCAL=1` for browser-capable local runs.
  - Browser mode is controlled by `SANDBOX_BROWSER_MODE` (`simulated|auto|real`).
  - Compose `sandbox-docker` profile uses a dedicated TLS-enabled `nexus-sandbox-dind` daemon on an internal network.
  - Host-socket mode is supported via `docker/docker-compose.host-socket.yml` and should only be used in trusted local/dev environments.
  - Production reverse proxy + frontend bundle is defined in `docker/docker-compose.prod.yml`.
- `continuity_core/` is part of this monorepo. It was originally a separate project but is now maintained here. Edit freely.

## Environment Variables

App-first required:
- `APP_DATABASE_URL`
- `APP_ADMIN_USERNAME`
- `APP_ADMIN_PASSWORD`
- `APP_SANDBOX_ARTIFACT_ROOT` (promotion source-root enforcement)

Optional/secondary:
- `DISCORD_TOKEN` (for bridge or legacy bot runtime)
- `OPENROUTER_API_KEY` (for model-backed execution paths)

See `config/.env.example` for full list.

## Docker Services

Infrastructure services can be mixed with external equivalents via environment overrides.

- `nexus-api` - App control plane (FastAPI)
- `nexus-sandbox-runner` - Isolated execution service
- `nexus-sandbox-dind` - Optional Docker daemon for sandbox backend (profile: sandbox-docker)
- `nexus-frontend` - Primary web app UI bundle (production compose override)
- `nexus-proxy` - Caddy reverse proxy with optional automatic TLS
- `nexus-discord-bridge` - Optional remote approval/status bridge
- `nexus-bot` - Legacy Discord bot runtime (optional)
- `nexus-qdrant` - Vector memory (port 6333, profile: qdrant)
- `nexus-redis` - Working memory cache (port 6379, profile: redis)
- `nexus-neo4j` - C2 knowledge graph (port 7687, profile: neo4j)
- `nexus-postgres` - App/C2 persistence (port 5432)
