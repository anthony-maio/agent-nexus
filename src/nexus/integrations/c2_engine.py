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

                # Per-call embedding cache — fixes O(n^2) redundant calls
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
                    "stress_delta": result.stress_after - result.stress_before,
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
