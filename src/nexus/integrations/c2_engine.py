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
            neo4j_uri="",  # disabled -- graph in Qdrant metadata
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
                except Exception as exc:
                    log.debug("Neo4j close failed: %s", exc)
        self._system = None
        self._running = False
        log.info("C2Engine stopped")

    @property
    def is_running(self) -> bool:
        return self._running and self._system is not None
