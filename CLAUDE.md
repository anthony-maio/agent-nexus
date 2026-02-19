# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Agent Nexus is a multi-model AI swarm orchestrated through Discord. Multiple LLMs (via OpenRouter) converse in a shared channel, coordinate on tasks, and dispatch lightweight task agents. Users interact through a `#human` channel; models collaborate in `#nexus`.

## Architecture

Two-tier model system:
- **Tier 1 (Main Swarm):** General intelligence models (MiniMax M2.5, GLM-5, Kimi K2.5, Qwen3 Coder Next) that converse and make decisions in `#nexus`. They see each other's messages.
- **Tier 2 (Task Agents):** LiquidAI 1.2B models dispatched by the orchestrator for specific tasks (routing, extraction, tool calling). They execute and return results, don't converse.

Three Discord channels: `#human` (user interaction), `#nexus` (model collaboration), `#memory` (audit trail).

## Package Layout

```
src/
  nexus/           # Main bot package
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
  continuity_core/ # Memory engine (copied from upstream, do not rewrite)
  synthesis/       # Capability ecosystem (Phase 3)
```

## Build and Run

```bash
# Setup
python setup/setup.py          # Interactive onboarding wizard
docker compose -f docker/docker-compose.yml up -d

# Development (without Docker)
pip install -e ".[dev]"
python -m nexus                # Run bot directly

# Tests
pytest tests/
pytest tests/test_openrouter.py -k "test_chat_completion"

# Lint
ruff check src/
ruff format src/
```

## Key Patterns

- All LLM calls go through `models/openrouter.py` (primary) or `models/ollama.py` (local fallback). Never call APIs directly.
- Embeddings are locked to a single provider chosen at setup time. Changing breaks all vectors. See `models/embeddings.py`.
- Channel routing goes through `channels/router.py`. The three channel references (`human`, `nexus`, `memory`) are resolved at startup.
- The orchestrator background loop (`orchestrator/loop.py`) runs on a configurable interval and dispatches LiquidAI task agents via `orchestrator/dispatch.py`.
- Consensus decisions (`swarm/consensus.py`) post to `#nexus`, collect model responses, and require configurable agreement threshold.
- `continuity_core/` is an upstream dependency copied as-is. Edit it in its own repo and re-copy, don't modify in-place here.

## Environment Variables

Required: `DISCORD_TOKEN`, `OPENROUTER_API_KEY`
Everything else has defaults. See `config/.env.example` for full list.

## Docker Services

- `nexus-bot` — The Discord bot (Python 3.12)
- `nexus-qdrant` — Vector memory (port 6333)
- `nexus-redis` — Working memory cache (port 6379)
