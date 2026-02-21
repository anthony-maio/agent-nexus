"""c2.status -- Return backend health and system metrics."""

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
