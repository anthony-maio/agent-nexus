"""Per-run event bus used by API timeline streams and adapters."""

from __future__ import annotations

import asyncio
from collections import defaultdict

from nexus_core.models import RunEvent


class RunEventBus:
    """In-memory pub/sub for run timeline events."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[RunEvent]]] = defaultdict(list)

    def subscribe(self, run_id: str) -> asyncio.Queue[RunEvent]:
        """Subscribe to a run's event stream."""
        q: asyncio.Queue[RunEvent] = asyncio.Queue(maxsize=200)
        self._subscribers[run_id].append(q)
        return q

    def unsubscribe(self, run_id: str, queue: asyncio.Queue[RunEvent]) -> None:
        """Remove a previously registered subscriber queue."""
        subs = self._subscribers.get(run_id, [])
        if queue in subs:
            subs.remove(queue)
        if not subs and run_id in self._subscribers:
            del self._subscribers[run_id]

    async def publish(self, event: RunEvent) -> None:
        """Publish event to all subscribers for the run."""
        for queue in list(self._subscribers.get(event.run_id, [])):
            if queue.full():
                # Drop oldest item to keep stream moving.
                _ = queue.get_nowait()
            await queue.put(event)
