from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from continuity_core.config import C2Config, load_config
from continuity_core.event_log import EventLog
from continuity_core.memory.consolidation import RecallGatedConsolidator
from continuity_core.memory.embeddings import build_embedder
from continuity_core.memory.stores import InMemoryStore
from continuity_core.mra.stress import StressResult
from continuity_core.mra.voids import VoidReport
from continuity_core.storage import Neo4jGraphStore, PostgresEventStore, QdrantMemoryStore, QdrantResult, RedisWorkingContext


@dataclass
class ScoredMemory:
    id: str
    score: float
    content: str
    memory_type: str
    payload: Dict[str, Any]


@dataclass
class MRACache:
    """Holds the most recent MRA stress and void results."""
    last_stress: Optional[StressResult] = None
    last_voids: Optional[VoidReport] = None
    updated_at: float = 0.0
    staleness_sec: float = 300.0

    def is_stale(self, now: Optional[float] = None) -> bool:
        if self.last_stress is None:
            return True
        now = now if now is not None else time.time()
        return (now - self.updated_at) > self.staleness_sec


class TieredMemorySystem:
    def __init__(self, config: Optional[C2Config] = None) -> None:
        self.config = config or load_config()
        self.embedder = build_embedder(self.config)
        self._event_log = self._init_event_log()
        self._redis = self._init_redis()
        self._qdrant, self._fallback = self._init_qdrant()
        self._neo4j = self._init_neo4j()
        self._mra_cache = MRACache()
        self._consolidator = RecallGatedConsolidator()
        self._recall_count = 0
        self._decay_every_n = 10

    @property
    def event_log(self) -> EventLog:
        return self._event_log

    @property
    def redis(self) -> Optional[RedisWorkingContext]:
        return self._redis

    @property
    def qdrant(self) -> Optional[QdrantMemoryStore]:
        return self._qdrant

    @property
    def neo4j(self) -> Optional[Neo4jGraphStore]:
        return self._neo4j

    def write_event(self, actor: str, intent: str, inp: str, out: str,
                    tags: Optional[List[str]] = None, metadata: Optional[Dict[str, str]] = None) -> None:
        self._event_log.log(actor, intent, inp, out, tags=tags, metadata=metadata)

    def append_working_context(self, thread_id: str, message: Dict[str, Any]) -> None:
        if self._redis is None:
            return
        self._redis.append(thread_id, message)

    def get_working_context(self, thread_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        if self._redis is None:
            return []
        return self._redis.get_recent(thread_id, limit=limit)

    def remember(self, content: str, memory_type: str, importance: int = 5,
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        if self._qdrant is not None:
            return self._qdrant.remember(content, memory_type, importance, metadata)
        # Fallback store for offline tests
        if self._fallback is None:
            raise RuntimeError("No memory store available")
        item = self._fallback.add(content, salience=float(importance) / 10.0, metadata=metadata)
        return item.id

    # -- MRA cache ---------------------------------------------------------

    def update_mra_cache(self, stress: StressResult, voids: Optional[VoidReport] = None) -> None:
        self._mra_cache.last_stress = stress
        self._mra_cache.last_voids = voids
        self._mra_cache.updated_at = time.time()

    def get_mra_signals(self) -> Optional[MRACache]:
        if self._mra_cache.is_stale():
            return None
        return self._mra_cache

    # -- Credit assignment + Harmonic Integration ---------------------------

    def credit(self, memory_ids: List[str], signal: float) -> None:
        """Boost salience of memories that were helpful."""
        signal = max(0.0, min(1.0, signal))
        if self._qdrant is not None:
            for mid in memory_ids:
                try:
                    pts = self._qdrant._client.retrieve(
                        collection_name=self._qdrant._collection,
                        ids=[mid],
                        with_payload=True,
                    )
                    if pts:
                        cur = float(pts[0].payload.get("importance", 5))
                        new_importance = min(10, cur + signal * 2.0)
                        self._qdrant._client.set_payload(
                            collection_name=self._qdrant._collection,
                            payload={"importance": new_importance},
                            points=[mid],
                        )
                except Exception:
                    pass  # best-effort
        elif self._fallback is not None:
            for item in self._fallback._items:
                if item.id in memory_ids:
                    item.salience = min(1.0, item.salience + signal * 0.2)
                    item.touch()

    def harmonic_integration(
        self,
        stress_delta: float,
        resolutions: List[Dict[str, Any]],
        compression_weight: float = 0.6,
        resonance_weight: float = 0.4,
    ) -> float:
        """Compute and distribute the Harmonic Integration reward.

        H = alpha_c * delta_C + alpha_r * delta_R

        Where delta_C (compression) is measured by stress reduction and
        delta_R (resonance) is measured by resolved contradictions
        propagated through the memory store as credit.

        Returns the reward magnitude H.
        """
        delta_c = max(0.0, -stress_delta)
        delta_r = min(1.0, len(resolutions) * 0.2) if resolutions else 0.0
        h = compression_weight * delta_c + resonance_weight * delta_r

        if h > 0.01 and self._fallback is not None:
            now = time.time()
            recent_ids = [
                item.id for item in self._fallback._items
                if (now - item.last_access) < 3600
            ]
            if recent_ids:
                self.credit(recent_ids, signal=min(1.0, h))

        return h

    # -- Recall with decay + consolidation gating -------------------------

    def recall(self, query: str, top_k: int = 5, type_filter: Optional[str] = None) -> List[ScoredMemory]:
        self._recall_count += 1

        # Periodic decay sweep
        if self._recall_count % self._decay_every_n == 0:
            self._run_decay()

        if self._qdrant is not None:
            results = self._qdrant.recall(query, top_k=top_k, type_filter=type_filter)
            # Update access timestamps so recency scoring stays fresh.
            accessed_ids = [r.id for r in results]
            if accessed_ids:
                self._qdrant.update_access(accessed_ids)
            return self._score_results(results)
        if self._fallback is None:
            return []
        scored = self._fallback.query(query, top_k=top_k)
        out: List[ScoredMemory] = []
        occupancy = len(self._fallback._items) / max(1, self._fallback.capacity)
        for item, sim in scored:
            # Consolidation gating: check if this recall should reinforce the item
            if self._consolidator.should_consolidate(sim, occupancy):
                item.touch()
            out.append(ScoredMemory(
                id=item.id,
                score=sim,
                content=str(item.content),
                memory_type=item.metadata.get("type", "unknown"),
                payload=item.metadata,
            ))
        return out

    def _run_decay(self) -> None:
        if self._fallback is not None:
            self._fallback.apply_decay(
                self.config.decay_rate,
                self.config.decay_time_unit_sec,
            )

    def _score_results(self, results: List[QdrantResult]) -> List[ScoredMemory]:
        out: List[ScoredMemory] = []
        for r in results:
            payload = r.payload
            recency = self._recency_score(payload.get("last_accessed"))
            importance = float(payload.get("importance", 5)) / 10.0
            score = (0.5 * r.score) + (0.3 * recency) + (0.2 * importance)
            out.append(ScoredMemory(
                id=r.id,
                score=score,
                content=payload.get("content", ""),
                memory_type=payload.get("type", "unknown"),
                payload=payload,
            ))
        out.sort(key=lambda x: x.score, reverse=True)
        return out

    def _recency_score(self, last_accessed: Optional[float]) -> float:
        if last_accessed is None:
            return 0.0
        age_days = (time.time() - float(last_accessed)) / 86400.0
        return math.exp(-age_days * math.log(2.0) / max(0.1, self.config.recency_half_life_days))

    def _init_event_log(self) -> EventLog:
        try:
            store = PostgresEventStore(self.config.postgres_url)
            return EventLog(store)
        except Exception as exc:
            logging.getLogger(__name__).warning("Postgres event log unavailable, falling back to in-memory: %s", exc)
            return EventLog()

    def _init_redis(self) -> Optional[RedisWorkingContext]:
        try:
            return RedisWorkingContext(self.config.redis_url)
        except Exception as exc:
            logging.getLogger(__name__).warning("Redis working context unavailable: %s", exc)
            return None

    def _init_qdrant(self) -> tuple[Optional[QdrantMemoryStore], Optional[InMemoryStore]]:
        try:
            vector = self.embedder.embed("seed")
            store = QdrantMemoryStore(
                url=self.config.qdrant_url,
                collection="c2_memories",
                embed_fn=self.embedder.embed,
                vector_size=len(vector),
            )
            return store, None
        except Exception as exc:
            logging.getLogger(__name__).warning("Qdrant memory store unavailable, using in-memory fallback: %s", exc)
            fallback = InMemoryStore(capacity=5000, embed_fn=self.embedder.embed)
            return None, fallback

    def _init_neo4j(self) -> Optional[Neo4jGraphStore]:
        try:
            return Neo4jGraphStore(self.config.neo4j_uri, self.config.neo4j_user, self.config.neo4j_password)
        except Exception as exc:
            logging.getLogger(__name__).warning("Neo4j graph store unavailable: %s", exc)
            return None
