"""c2.events — Read recent events from the event log."""

from __future__ import annotations

from typing import Any, Dict

from continuity_core.services.runtime import get_memory_system


def read_events(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Return the most recent events from the C2 event log."""
    limit = int(arguments.get("limit", 10))
    limit = max(1, min(limit, 50))

    mem = get_memory_system()
    events = mem.event_log.tail(n=limit)

    return {
        "count": len(events),
        "events": [
            {
                "timestamp": e.timestamp,
                "actor": e.actor,
                "intent": e.intent,
                "input": e.input,
                "output": e.output,
                "tags": e.tags,
            }
            for e in events
        ],
    }
