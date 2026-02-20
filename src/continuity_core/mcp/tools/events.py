from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from continuity_core.services.runtime import get_memory_system


def write_event(arguments: Dict[str, Any]) -> Dict[str, Any]:
    actor = arguments.get("actor", "agent")
    intent = arguments.get("intent", "unspecified")
    inp = arguments.get("input", "")
    out = arguments.get("output", "")
    tags = arguments.get("tags") or []
    metadata = arguments.get("metadata") or {}
    event = get_memory_system().event_log.log(
        actor=actor,
        intent=intent,
        inp=inp,
        out=out,
        tags=tags,
        metadata=metadata,
    )
    return {
        "timestamp": event.timestamp,
        "actor": event.actor,
        "intent": event.intent,
        "input": event.input,
        "output": event.output,
        "tags": event.tags,
        "metadata": event.metadata,
    }
